"""Goal-precondition guard runtime.

One new focused mechanism layered on the retained anti-repetition block.

The runtime reads the goal sentence the environment states (first user message)
and, using only the goal's own words -- never a hardcoded object/condition table
-- derives (i) the object the goal names, (ii) the condition the goal requires
that object to satisfy, and (iii) the placement target.  It then tracks, from
the agent's own submitted tool-call actions, whether that condition has actually
been applied to that object and whether the object has been placed.  When the
objective is in danger (a premature placement, a stalled action history, or a
long run with no condition action ever applied) it appends ONE short
"[Goal check]" block to the text of the *last* request message, naming the
prerequisite that is still unmet and the correct order of operations.

The gate is deliberately narrow: a goal whose own words name no condition
(e.g. a plain collect-and-place goal, or a countable/plural goal) always leaves
the request untouched.  The retained round-1 anti-repetition "[Guidance]" block
is kept behaviourally identical and, when both fire on the same request, is
appended in the SAME single deep-copy edit (goal block first).  When neither
block applies the *identical* request object is returned, so untouched episodes
are byte-for-byte unchanged.  The whole body is exception-guarded and the
response side is a pure pass-through: model, sampling and budgets are untouched.
"""

import copy
import json
import re

# --- markers (also used as idempotence guards) -------------------------------
GOAL_MARKER = "[Goal check]"
GUIDANCE_MARKER = "[Guidance]"

# --- generic English scaffolding (goal-derived, never a household table) -----
# Determiners / quantifiers / pronouns: these are NOT condition words.
_DET = {
    "a", "an", "the", "some", "any", "both", "all", "its", "this", "that",
    "these", "those", "it", "them", "their", "his", "her", "your", "of",
}
# Explicit counts: a goal that names a count is a collect-and-place goal,
# not a single-object condition goal.
_NUM = {
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "both", "several", "multiple",
}
# Verbs that only navigate, inspect or move the agent/object: they do NOT
# change an object's condition.  Any other leading verb counts as state.
_NON_STATE_VERBS = {
    "go", "look", "inventory", "examine", "take", "pick", "open", "close",
    "find", "reach", "put", "place", "drop",
}

_GOAL_RE = re.compile(r"task\s+is\s+to\s*:?\s*(.*?)\s*AVAILABLE\s+ACTIONS",
                      re.S | re.I)
# Form A: "<verb> <obj...> and put/place it|them in/on <dest>"
_PLACE_RE = re.compile(
    r"^(.*?)\band\b\s+(?:put|place)\s+(it|them)\s+(?:in|on)\s+(.+)$", re.I)
# Form B: "put/place/find <obj...> in/on <dest>"
_PUT_RE = re.compile(r"^(?:put|place|find)\s+(.+?)\s+(?:in|on)\s+(.+)$", re.I)
# Form D: "examine/look at/look/use <obj...> with/under/using <target>"
_USE_RE = re.compile(
    r"^(?:examine|look\s+at|look|use)\s+(.+?)\s+(?:with|under|using)\s+(.+)$",
    re.I)

# --- retained anti-repetition thresholds (unchanged from round 1) ------------
_WINDOW_RULES = ((8, 4), (10, 5), (12, 6))
_MAX_REPEATS = 4
_MAX_UNTRIED = 10
AVAILABLE_TAG = "AVAILABLE ACTIONS"


def register():
    return None


def _tokens(text):
    return [w for w in re.split(r"\s+", text.strip()) if w]


def _clean(word):
    return word.strip(".,;:!?\"'()")


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
    # Core
    # ------------------------------------------------------------------ #
    def _transform_request(self, request):
        if not isinstance(request, dict):
            return request
        messages = request.get("messages")
        if not isinstance(messages, list) or not messages:
            return request
        last = messages[-1]
        if not isinstance(last, dict):
            return request

        actions = self._extract_actions(messages)
        last_text = self._message_text(last) or ""

        parts = []
        goal_block = self._goal_block(messages, last_text, actions)
        if goal_block:
            parts.append(goal_block)

        # Retained round-1 anti-repetition mechanism, unchanged.
        if len(actions) >= 2 and self._is_stagnating(actions):
            if GUIDANCE_MARKER not in last_text:
                guide = self._guidance_block(actions, last_text)
                if guide:
                    parts.append(guide)

        if not parts:
            return request

        updated = copy.deepcopy(request)
        self._append_text(updated["messages"][-1], "\n\n".join(parts))
        return updated

    # ------------------------------------------------------------------ #
    # Goal derivation (the new mechanism)
    # ------------------------------------------------------------------ #
    def _goal_block(self, messages, last_text, actions):
        if GOAL_MARKER in last_text:
            return None
        goal = self._goal_text(messages)
        if not goal:
            return None
        info = self._parse_goal(goal)
        head = info.get("head")
        req = info.get("req")
        # Guard: nothing to say unless the goal's own words name a condition
        # on a concrete object.
        if not req or head is None or head in _NUM or head in _DET:
            return None
        status = self._progress(actions, head, info.get("dest"))
        if not self._should_emit(status, actions):
            return None
        return self._render_goal_block(goal, info, status, actions)

    @staticmethod
    def _goal_text(messages):
        """Extract the stated goal sentence from the first user message."""
        for msg in messages:
            if not isinstance(msg, dict) or msg.get("role") != "user":
                continue
            text = AgentHarness._message_text(msg) or ""
            m = _GOAL_RE.search(text)
            if m:
                return m.group(1).strip().rstrip(".")
            idx = text.upper().find(AVAILABLE_TAG)
            seg = text[:idx] if idx >= 0 else text
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", seg)
                         if s.strip()]
            if sentences:
                return sentences[-1].rstrip(".")
            return ""
        return ""

    @staticmethod
    def _parse_goal(goal):
        """Return {'head','req','dest','form'} from the goal's own words only."""
        g = goal.strip().rstrip(".")
        gl = g.lower()
        empty = {"head": None, "req": [], "dest": None, "form": "?"}

        # Form A: "<verb ...> and put/place it|them in/on <dest>"
        m = _PLACE_RE.match(gl)
        if m:
            pre = _tokens(m.group(1))
            if not pre:
                return dict(empty)
            verb = _clean(pre[0])
            obj = [_clean(w) for w in pre[1:] if _clean(w) not in _DET]
            plural = (m.group(2).lower() == "them") or any(
                _clean(w) in _NUM for w in pre)
            head = obj[-1] if obj else None
            # A single-object pre-placement clause states an ordered condition
            # requirement for that object; a plural/counted clause does not.
            req = [verb] if (obj and not plural) else []
            return {"head": head, "req": req, "dest": m.group(3).strip(),
                    "form": "A"}

        # Form B: "put/place/find <obj...> in/on <dest>"
        m = _PUT_RE.match(gl)
        if m:
            core = [_clean(w) for w in _tokens(m.group(1))
                    if _clean(w) not in _DET]
            if not core:
                return dict(empty)
            head = core[-1]
            req = [w for w in core[:-1] if w not in _NUM]
            return {"head": head, "req": req, "dest": m.group(2).strip(),
                    "form": "B"}

        # Form D: "examine/look at/look/use <obj...> with/under/using <target>"
        # Kept for completeness; never emits on its own because req stays [].
        m = _USE_RE.match(gl)
        if m:
            core = [_clean(w) for w in _tokens(m.group(1))
                    if _clean(w) not in _DET]
            if not core:
                return dict(empty)
            head = core[-1]
            req = [w for w in core[:-1] if w not in _NUM]
            return {"head": head, "req": req, "dest": m.group(2).strip(),
                    "form": "D"}

        return dict(empty)

    @staticmethod
    def _progress(actions, head, dest):
        """Track the goal object's condition/placement from submitted actions."""
        related = [a for a in actions if head in _tokens(a)]
        obtained = any(a.split()[0] in ("take", "pick") for a in related)
        condition_actions = [a for a in related
                             if a.split()[0] not in _NON_STATE_VERBS]
        placements = [i for i, a in enumerate(actions)
                      if a.split()[0] in ("put", "place") and head in _tokens(a)]
        n_placements = len(placements)

        dest_words = []
        if dest:
            dest_words = [_clean(w) for w in _tokens(dest)
                          if _clean(w) not in _DET and _clean(w) not in _NUM]
        placed_at_target = any(
            w in _tokens(actions[i]) for i in placements for w in dest_words
        )

        premature = False
        if placements:
            first = placements[0]
            cond_before = any(
                head in _tokens(actions[j])
                and actions[j].split()[0] not in _NON_STATE_VERBS
                for j in range(first)
            )
            premature = not cond_before

        return {
            "obtained": obtained,
            "condition_actions": condition_actions,
            "n_placements": n_placements,
            "placed_at_target": placed_at_target,
            "premature": premature,
        }

    @staticmethod
    def _should_emit(status, actions):
        if status["premature"]:
            return True
        if AgentHarness._is_stagnating(actions):
            return True
        if len(actions) >= 10 and not status["condition_actions"]:
            return True
        return False

    @staticmethod
    def _render_goal_block(goal, info, status, actions):
        head = info["head"]
        dest = info.get("dest") or ""
        req = ", ".join('"%s"' % r for r in info["req"])
        conds = status["condition_actions"]
        named = "; ".join(conds) if conds else "none"
        lines = [
            '%s The task states: "%s"' % (GOAL_MARKER, goal),
            '- Object the task names: "%s". Target the task names: "%s".'
            % (head, dest),
            '- Condition the task requires on "%s": %s. Correct order: obtain '
            '"%s" (take it), apply the action that gives it that condition, and '
            'only then place it at "%s". Placing it before that will not '
            "complete the task." % (head, req, head, dest),
            '- Your progress on "%s" after %d action(s): obtained=%s; condition '
            "actions applied to it=%s; times placed=%d."
            % (head, len(actions), "yes" if status["obtained"] else "no",
               named, status["n_placements"]),
        ]
        if status["premature"]:
            lines.append(
                'You have already placed "%s" without applying the condition '
                'first, so the task is still incomplete: apply the condition '
                'action to "%s" and then place it again.' % (head, head)
            )
        elif status["condition_actions"] and status["n_placements"] == 0:
            lines.append(
                'The condition on "%s" is already satisfied; the remaining '
                'step is to place it at "%s".' % (head, dest)
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Shared parsing helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _extract_actions(messages):
        """Ordered list of non-empty top-level string 'action' values."""
        actions = []
        for msg in messages:
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
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

    @staticmethod
    def _available_actions(text):
        """Parse newline-separated items after the first ':' of AVAILABLE ACTIONS."""
        if not text:
            return []
        idx = text.find(AVAILABLE_TAG)
        if idx < 0:
            return []
        colon = text.find(":", idx)
        if colon < 0:
            return []
        return [ln.strip() for ln in text[colon + 1:].split("\n") if ln.strip()]

    # ------------------------------------------------------------------ #
    # Retained anti-repetition block (round 1, unchanged)
    # ------------------------------------------------------------------ #
    def _guidance_block(self, actions, text):
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
            lines.append("Actions you have repeated: " + ", ".join(repeats)
                         + ".")
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
        order = {}
        for i, a in enumerate(actions):
            counts[a] = counts.get(a, 0) + 1
            order.setdefault(a, i)
        repeated = [(a, c) for a, c in counts.items() if c >= 2]
        repeated.sort(key=lambda ac: (-ac[1], order.get(ac[0], 0)))
        return ["%s (x%d)" % (a, c) for a, c in repeated[:_MAX_REPEATS]]

    def _untried_line(self, actions, text):
        avail = self._available_actions(text)
        if not avail:
            return None
        seen = set()
        avail_unique = []
        for a in avail:
            if a not in seen:
                seen.add(a)
                avail_unique.append(a)
        tried = set(actions)
        untried = [a for a in avail_unique if a not in tried]
        if not untried:
            return ("Every action currently available already appears in the "
                    "actions you have tried; consider whether a different kind "
                    "of progress is needed.")
        total = len(untried)
        if total > _MAX_UNTRIED:
            shown = [untried[int(i * total / _MAX_UNTRIED)]
                     for i in range(_MAX_UNTRIED)]
            body = "; ".join(shown)
            remaining = total - len(shown)
            if remaining > 0:
                body += " (+%d more)" % remaining
            return ("Actions that are available and that you have not yet tried: "
                    + body + ".")
        return ("Actions that are available and that you have not yet tried: "
                + "; ".join(untried) + ".")

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
