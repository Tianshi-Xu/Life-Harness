"""Post-authentication id-grounding middleware for the retail LLM agent.

ONE mechanism: when the IDENTITY-RESOLUTION tool (``find_user_id_by_email`` /
``find_user_id_by_name_zip``) returns a real user id, append a single concrete
next-step line to that tool observation -- echo the resolved user id and tell
the solver to call ``get_user_details(user_id=...)`` and then use exactly the
order ids / payment method ids that result returns, instead of guessing,
counting, or reusing an example id.  Nothing else is touched.

The middleware never calls a model, never touches the filesystem/network, keeps
no cross-episode mutable state (only the single overwritten ``_last_history``),
and degrades to identity behaviour on any unexpected structure.
"""

import json
import re

MARKER = "[NEXT-STEP]"

# Ordered candidate keys for the conversation list on a request object/dict.
_MESSAGE_KEYS = ("messages", "input", "history", "chat")

# A find_user_id_* success value is a bare ``firstname_lastname_digits`` string
# (e.g. 'yusuf_rossi_9620', 'sara_doe_496').  No other tool returns this shape,
# so requiring the WHOLE content to match keeps the detector false-positive free.
_BARE_USER_ID = re.compile(r"^[A-Za-z][A-Za-z0-9]*_[A-Za-z][A-Za-z0-9]*_\d{2,}$")


# --------------------------------------------------------------------------- #
# Pure helpers (no side effects)
# --------------------------------------------------------------------------- #
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


def _is_message_list(value):
    """True for a non-empty list whose elements are all dicts."""
    return (
        isinstance(value, list)
        and len(value) > 0
        and all(isinstance(x, dict) for x in value)
    )


def _grounding_line(user_id):
    """The single fixed grounding line, parameterised only by the real user id."""
    return (
        "\n" + MARKER + " You are authenticated as user '" + user_id + "'. You do "
        "not yet know this user's order ids, and guessing, counting, or reusing an "
        "example id will fail. Call "
        "get_user_details(user_id='" + user_id + "') now, then use exactly the "
        "order ids and payment method ids it returns for this user."
    )


def _name_by_call_id(messages):
    """Map assistant tool-call id -> tool name (from assistant tool_calls)."""
    names = {}
    for m in messages:
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        tcs = m.get("tool_calls")
        if not isinstance(tcs, list):
            continue
        for tc in tcs:
            if not isinstance(tc, dict):
                continue
            cid = tc.get("id")
            fn = tc.get("function")
            if isinstance(fn, dict) and cid is not None:
                names[cid] = fn.get("name")
    return names


def _append(message, line):
    """Append ``line`` to a message's content in place. Returns True on change."""
    content = message.get("content")
    if isinstance(content, str):
        message["content"] = content + line
        return True
    if isinstance(content, list):
        content.append({"type": "text", "text": line})
        return True
    return False


# --------------------------------------------------------------------------- #
# Middleware
# --------------------------------------------------------------------------- #
class AgentHarness:
    def __init__(self, public_context=None):
        self.public_context = public_context
        self._last_history = None

    # -- extraction -------------------------------------------------------- #
    def _extract_messages(self, request):
        """Return (messages, attach); ``attach`` re-binds msgs to the request."""
        if isinstance(request, dict):
            for key in _MESSAGE_KEYS:
                if _is_message_list(request.get(key)):
                    return request[key], ("dict", key)
        else:
            msgs = getattr(request, "messages", None)
            if _is_message_list(msgs):
                return msgs, ("attr", "messages")

        if _is_message_list(self._last_history):
            # fallback source: work on a copy so nothing accumulates across calls.
            return list(self._last_history), None

        return None, None

    # -- public API -------------------------------------------------------- #
    def transform_request(self, request):
        try:
            msgs, _attach = self._extract_messages(request)
            if msgs is None or not msgs:
                return request

            names = _name_by_call_id(msgs)
            for m in msgs:
                if not isinstance(m, dict) or m.get("role") != "tool":
                    continue
                text = _text_of(m.get("content"))
                if MARKER in text:
                    continue
                user_id = text.strip()
                if not _BARE_USER_ID.match(user_id):
                    continue
                name = names.get(m.get("tool_call_id"))
                # Only the identity-resolution tool qualifies; when the call id
                # is unknown, the bare-id shape is already unique to it.
                if name is not None and not str(name).startswith("find_user_id"):
                    continue
                _append(m, _grounding_line(user_id))

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
