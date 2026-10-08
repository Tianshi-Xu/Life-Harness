"""Stuck-state recovery guidance middleware for the retail LLM agent.

ONE mechanism: detect when the agent conversation has become "stuck" -- the
model is repeating an identical tool call, or the same error/result has already
recurred several times -- and inject a single, advisory system message telling
it to break the loop: fix the specific cause, never invent identifiers, change
approach, or (last resort) hand off to a human. Otherwise the middleware is a
pure pass-through.

The middleware never calls a model, never touches the filesystem/network, keeps
no cross-episode mutable state (only the single overwritten ``_last_history``),
and degrades to identity behaviour on any unexpected structure.
"""

import json

_MARKER = "RECOVERY NOTE (advisory)"

RECOVERY_NOTE = (
    "RECOVERY NOTE (advisory): You appear stuck - you are repeating the same "
    "tool call, or the same error/result has already recurred, so repeating it "
    "again will not change the outcome. Before your next action: (1) re-read "
    "the latest error/result and fix its specific cause; (2) never guess or "
    "invent identifiers (ids, codes, emails) - obtain real values from the "
    "user or from records you can already access, because a lookup of one "
    "record exposes the related valid identifiers; (3) make sure your "
    "arguments satisfy the tool's stated requirements; (4) if the same "
    "approach has already failed, change it - use a different tool or "
    "different arguments, ask the user for the missing information, or, only "
    "if it truly cannot be resolved with the available tools, tell the user "
    "and transfer to a human agent. Do not send an identical call again. If "
    "your next planned action is already different from the repeated call, "
    "ignore this note and continue."
)


# --------------------------------------------------------------------------- #
# Pure helpers (no side effects)
# --------------------------------------------------------------------------- #
def _canonical_args(args):
    """Normalize tool-call arguments so repeats are comparable."""
    if isinstance(args, dict):
        try:
            return json.dumps(args, sort_keys=True)
        except Exception:
            return str(args).strip()
    if args is None:
        return ""
    return str(args).strip()


def _text_of(content):
    """Render a message/tool content value as text (str-or-JSON)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, sort_keys=True)
    except Exception:
        return str(content)


def _extract_calls(messages):
    """Ordered list of (name, canonical_args) for every assistant tool call."""
    calls = []
    for m in messages:
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        tcs = m.get("tool_calls")
        if isinstance(tcs, list):
            for tc in tcs:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function")
                if isinstance(fn, dict):
                    name, args = fn.get("name"), fn.get("arguments")
                else:
                    name, args = tc.get("name"), tc.get("arguments")
                if name is not None:
                    calls.append((str(name), _canonical_args(args)))
        fc = m.get("function_call")
        if isinstance(fc, dict) and fc.get("name") is not None:
            calls.append((str(fc.get("name")), _canonical_args(fc.get("arguments"))))
    return calls


def _tool_results(messages):
    """Ordered textual results of every tool-role message."""
    out = []
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "tool":
            out.append(_text_of(m.get("content")))
    return out


def _is_error(text):
    low = text.lower()
    return low.startswith("error") or "error:" in low


def _last_message_is_tool(messages):
    if not messages:
        return False
    m = messages[-1]
    return isinstance(m, dict) and m.get("role") == "tool"


def _is_stuck(messages):
    """True if any of the three stuck conditions (a), (b), (c) holds."""
    calls = _extract_calls(messages)
    results = _tool_results(messages)

    # (a) direct repeat loop: last two calls identical (name AND args).
    if len(calls) >= 2 and calls[-1] == calls[-2]:
        return True

    last_result = results[-1] if results else None
    prev_result = results[-2] if len(results) >= 2 else None

    # (b) last call already tried >= 3 times AND it is failing / redundant.
    if calls and calls.count(calls[-1]) >= 3:
        if last_result is not None and _is_error(last_result):
            return True
        if last_result is not None and prev_result is not None and last_result == prev_result:
            return True

    # (c) the same result/error string recurred >= 3 times, latest msg is a tool.
    if last_result is not None and _last_message_is_tool(messages):
        if results.count(last_result) >= 3:
            return True

    return False


def _has_marker(messages):
    for m in messages:
        if isinstance(m, dict) and _MARKER in _text_of(m.get("content")):
            return True
    return False


def _is_message_list(value):
    return (
        isinstance(value, list)
        and len(value) > 0
        and all(isinstance(x, dict) for x in value)
    )


# --------------------------------------------------------------------------- #
# Middleware
# --------------------------------------------------------------------------- #
class AgentHarness:
    def __init__(self, public_context=None):
        self.public_context = public_context
        self._last_history = None

    # -- extraction -------------------------------------------------------- #
    def _extract_messages(self, request):
        """Return (messages, attach) where ``attach`` re-binds msgs to request.

        ``attach`` is a tuple ``("dict", key)`` or ``("attr", name)`` when the
        list lives on the request, or ``None`` when it came from the recorded
        history fallback (in which case a copy is used and nothing is bound).
        """
        if isinstance(request, dict):
            for key in ("messages", "input", "history", "chat"):
                if _is_message_list(request.get(key)):
                    return request[key], ("dict", key)
        else:
            msgs = getattr(request, "messages", None)
            if _is_message_list(msgs):
                return msgs, ("attr", "messages")

        if _is_message_list(self._last_history):
            # fallback source: work on a copy so no cross-call accumulation.
            return list(self._last_history), None

        return None, None

    # -- public API -------------------------------------------------------- #
    def transform_request(self, request):
        try:
            msgs, attach = self._extract_messages(request)
            if msgs is None or not msgs:
                return request
            if _has_marker(msgs):
                return request
            if _is_stuck(msgs):
                msgs.append({"role": "system", "content": RECOVERY_NOTE})
                if attach is not None:
                    kind, key = attach
                    try:
                        if kind == "dict":
                            request[key] = msgs
                        else:
                            setattr(request, key, msgs)
                    except Exception:
                        pass
            return request
        except Exception:
            return request

    def transform_response(self, response, public_history):
        try:
            if isinstance(public_history, list):
                self._last_history = public_history
        except Exception:
            pass
        return response


def register():
    return None
