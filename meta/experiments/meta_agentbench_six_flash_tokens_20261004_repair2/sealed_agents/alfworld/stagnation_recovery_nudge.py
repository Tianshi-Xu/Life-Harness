"""Stagnation-aware anti-repetition guidance runtime.

Mechanism (one focused idea): from the public request messages we reconstruct
the ordered list of actions the agent has already submitted via tool calls. If
that history looks stalled -- the same action twice in a row, or a short recent
window dominated by few distinct actions -- we append a single short, general
guidance block to the text of the *last* request message (the freshest tool
observation, or the initial task message). The block reminds the agent how many
actions it has spent, which actions it has already repeated, and which of the
currently advertised AVAILABLE ACTIONS it has not yet tried, and suggests (as a
conditional, not a command) trying an untried action that best advances the
goal. When no stagnation trigger fires the request is returned completely
untouched, so easy/progressing episodes are unaffected. The response side is a
pure pass-through; model, sampling and budgets are untouched.
"""

import copy
import json

# Marker used both for the emitted block and as an idempotence guard.
GUIDANCE_MARKER = "[Guidance]"

# Trigger thresholds: (window length, max distinct actions in window).
_WINDOW_RULES = ((8, 4), (10, 5), (12, 6))

# How many repeated actions / untried available actions to name at most.
_MAX_REPEATS = 4
_MAX_UNTRIED = 10

AVAILABLE_TAG = "AVAILABLE ACTIONS"


def register():
    return None


class AgentHarness:
    def __init__(self, public_context=None):
        self.public_context = public_context

    # ------------------------------------------------------------------ #
    # Public surface
    # ------------------------------------------------------------------ #
    def transform_request(self, request):
        try:
            return self._transform_request(request)
        except Exception:
            return request

    def transform_response(self, response, public_history):
        return response

    # ------------------------------------------------------------------ #
    # Core logic
    # ------------------------------------------------------------------ #
    def _transform_request(self, request):
        if not isinstance(request, dict):
            return request
        messages = request.get("messages")
        if not isinstance(messages, list) or not messages:
            return request

        actions = self._extract_actions(messages)
        if len(actions) < 2:
            return request
        if not self._is_stagnating(actions):
            return request

        last = messages[-1]
        if not isinstance(last, dict):
            return request

        text = self._message_text(last)
        if text is not None and GUIDANCE_MARKER in text:
            return request

        block = self._build_block(actions, text)
        if not block:
            return request

        updated = copy.deepcopy(request)
        self._append_text(updated["messages"][-1], block)
        return updated

    # ------------------------------------------------------------------ #
    # Parsing helpers
    # ------------------------------------------------------------------ #
    def _extract_actions(self, messages):
        """Ordered list of non-empty top-level string 'action' values."""
        actions = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            if msg.get("role") != "assistant":
                continue
            tool_calls = msg.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                fn = call.get("function")
                if not isinstance(fn, dict):
                    continue
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        continue
                if not isinstance(args, dict):
                    continue
                action = args.get("action")
                if isinstance(action, str) and action.strip():
                    actions.append(action.strip())
        return actions

    @staticmethod
    def _is_stagnating(actions):
        n = len(actions)
        if n >= 2 and actions[-1] == actions[-2]:
            return True
        for window, max_distinct in _WINDOW_RULES:
            if n >= window and len(set(actions[-window:])) <= max_distinct:
                return True
        return False

    @staticmethod
    def _message_text(msg):
        """Return the textual content of a message, or None if there is none."""
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [
                p.get("text")
                for p in content
                if isinstance(p, dict) and isinstance(p.get("text"), str)
            ]
            if parts:
                return "\n".join(parts)
        return None

    def _available_actions(self, text):
        """Parse newline-separated items after the first ':' of AVAILABLE ACTIONS."""
        if not text:
            return []
        idx = text.find(AVAILABLE_TAG)
        if idx < 0:
            return []
        colon = text.find(":", idx)
        if colon < 0:
            return []
        tail = text[colon + 1:]
        items = []
        for line in tail.split("\n"):
            item = line.strip()
            if item:
                items.append(item)
        return items

    # ------------------------------------------------------------------ #
    # Block construction / insertion
    # ------------------------------------------------------------------ #
    def _build_block(self, actions, text):
        lines = [
            "%s You have already taken %d action(s) in this session."
            % (GUIDANCE_MARKER, len(actions))
        ]

        if len(actions) >= 2 and actions[-1] == actions[-2]:
            lines.append(
                'You just took "%s" twice in a row; taking it again is unlikely '
                "to change the environment." % actions[-1]
            )

        repeats = self._repeat_summary(actions)
        if repeats:
            lines.append("Actions you have repeated: " + ", ".join(repeats) + ".")

        lines.append(
            "Repeating an action you have already taken does not make progress "
            "and consumes your limited turns."
        )

        avail_line = self._untried_line(actions, text)
        if avail_line:
            lines.append(avail_line)

        lines.append(
            "If you are stuck, choose an available action you have not tried yet "
            "that best advances the task goal."
        )
        return "\n".join(lines)

    @staticmethod
    def _repeat_summary(actions):
        counts = {}
        for a in actions:
            counts[a] = counts.get(a, 0) + 1
        repeated = [(a, c) for a, c in counts.items() if c >= 2]
        # Most frequent first; stable tie-break by first occurrence order.
        order = {}
        for i, a in enumerate(actions):
            order.setdefault(a, i)
        repeated.sort(key=lambda ac: (-ac[1], order.get(ac[0], 0)))
        return ["%s (x%d)" % (a, c) for a, c in repeated[:_MAX_REPEATS]]

    def _untried_line(self, actions, text):
        avail = self._available_actions(text)
        if not avail:
            return None
        tried = set(actions)
        # De-duplicate available items while preserving order.
        seen = set()
        avail_unique = []
        for a in avail:
            if a not in seen:
                seen.add(a)
                avail_unique.append(a)
        untried = [a for a in avail_unique if a not in tried]
        if not untried:
            return (
                "Every action currently available already appears in the actions "
                "you have tried; consider whether a different kind of progress is "
                "needed."
            )
        total = len(untried)
        if total > _MAX_UNTRIED:
            m = total
            shown = [untried[int(i * m / _MAX_UNTRIED)] for i in range(_MAX_UNTRIED)]
            remaining = total - len(shown)
            body = "; ".join(shown)
            if remaining > 0:
                body += " (+%d more)" % remaining
            return (
                "Actions that are available and that you have not yet tried: "
                + body + "."
            )
        return (
            "Actions that are available and that you have not yet tried: "
            + "; ".join(untried) + "."
        )

    @staticmethod
    def _append_text(msg, block):
        """Append block to a message's text without clobbering existing content."""
        content = msg.get("content")
        if content is None:
            msg["content"] = block
            return
        if isinstance(content, str):
            sep = "" if content.endswith("\n") or content == "" else "\n\n"
            msg["content"] = content + sep + block
            return
        if isinstance(content, list):
            for part in reversed(content):
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    part["text"] = part["text"] + "\n\n" + block
                    return
            content.append({"type": "text", "text": block})
            return
        # Unknown content type: do not risk mangling it.
        return
