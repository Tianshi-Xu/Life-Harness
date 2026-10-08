"""General operating-checklist middleware for the retail LLM agent.

ONE mechanism: idempotently inject a single, fixed, general "operating
checklist" into the agent's SYSTEM-INSTRUCTION channel -- either the dedicated
system/prompt field of the request, or the first ``role == "system"`` message of
the message list, or (when a message list exists but has no system message) a
prepended system message.  The checklist is generic developer guidance about the
three recurring failure modes of a small solver: inventing/guessing ids (or
reusing the placeholder ids that appear in tool descriptions), failing to use
the available lookup tools, and blindly re-issuing a call after a tool error.

The middleware is a pure pass-through when the marker is already present or when
no system-instruction channel and no message list exist.  It never calls a
model, never touches the filesystem/network/subprocess, keeps NO mutable state
(not even a recorded history), never edits user/assistant/tool messages or tool
calls, and degrades to identity behaviour on any unexpected structure.
"""

MARKER = "[OPERATING CHECKLIST]"

# Fixed, general guidance.  Contains no task names and no episode-specific data.
BLOCK = (
    "[OPERATING CHECKLIST]\n"
    "1) IDS ARE DATA, NEVER GUESSWORK. Never invent, guess, or count through "
    "identifiers, and never reuse the example/placeholder values that appear "
    "inside tool descriptions (such as #W0000000, 1008292230, 6086499569, "
    "credit_card_0000000, gift_card_0000000, sara_doe_496) - they are "
    "illustrations, not real records. Copy every id exactly from a user message "
    "or from a tool result.\n"
    "2) USE THE LOOKUP TOOLS YOU HAVE. If you are missing an order id: after "
    "authenticating the user, call get_user_details(user_id) - it lists that "
    "user's orders and payment methods. If you are missing a product/item id: "
    "call list_all_product_types, then get_product_details(product_id), and "
    "choose a variant whose available is true. Never tell the user you cannot "
    "look something up when one of these tools can, and do not transfer to a "
    "human before you have tried them.\n"
    "3) AFTER A TOOL ERROR, DO NOT SEND THE SAME CALL AGAIN. Change an argument "
    "or use a different tool. If a lookup says not found, go obtain the real id "
    "(rules 1-2); never retry the same value."
)

# Candidate names for a dedicated system-instruction channel.
_SYSTEM_NAMES = ("system", "system_prompt", "system_message", "instructions", "prompt")
# Candidate names for a message list.
_MESSAGE_KEYS = ("messages", "input", "history", "chat")


# --------------------------------------------------------------------------- #
# Pure helpers (no side effects)
# --------------------------------------------------------------------------- #
def _is_message_list(value):
    """True for a non-empty list whose elements are all dicts."""
    return (
        isinstance(value, list)
        and len(value) > 0
        and all(isinstance(x, dict) for x in value)
    )


def _system_field(dict_target, obj_target):
    """Locate a dedicated system-instruction channel.

    Returns ``("dict", name, text)`` or ``("attr", name, text)`` for the first
    candidate name whose value is a non-empty ``str``, else ``None``.
    """
    if isinstance(dict_target, dict):
        for name in _SYSTEM_NAMES:
            value = dict_target.get(name)
            if isinstance(value, str) and value != "":
                return ("dict", name, value)
    if obj_target is not None:
        for name in _SYSTEM_NAMES:
            value = getattr(obj_target, name, None)
            if isinstance(value, str) and value != "":
                return ("attr", name, value)
    return None


def _write_system_field(request, kind, name, text):
    """Write ``text`` back to the located channel (dict key or attribute)."""
    if kind == "dict":
        request[name] = text
    else:
        setattr(request, name, text)


def _find_message_list(request):
    """Return ``(messages, attach)`` or ``(None, None)``.

    ``attach`` is ``("dict", key)`` or ``("attr", "messages")`` so the list can
    be re-bound to the request.
    """
    if isinstance(request, dict):
        for key in _MESSAGE_KEYS:
            if _is_message_list(request.get(key)):
                return request[key], ("dict", key)
        return None, None
    value = getattr(request, "messages", None)
    if _is_message_list(value):
        return value, ("attr", "messages")
    return None, None


def _find_system_message(messages):
    """Return the index of the first system message, or -1."""
    for i, msg in enumerate(messages):
        if isinstance(msg, dict) and msg.get("role") == "system":
            if isinstance(msg.get("content"), str):
                return i
    return -1


def _marker_in_messages(messages):
    """True when the marker appears in any message's str content."""
    for msg in messages:
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, str) and MARKER in content:
                return True
    return False


# --------------------------------------------------------------------------- #
# Middleware
# --------------------------------------------------------------------------- #
class AgentHarness:
    def __init__(self, public_context=None):
        self.public_context = public_context

    def transform_request(self, request):
        """Idempotently inject the operating checklist into the system channel."""
        try:
            return self._inject(request)
        except Exception:
            return request

    def _inject(self, request):
        # (a)/(b) Dedicated system-instruction channel on a dict or object.
        if request is None:
            return request
        target = _system_field(
            request if isinstance(request, dict) else None,
            request,
        )
        if target is not None:
            kind, name, text = target
            if MARKER in text:
                return request
            _write_system_field(request, kind, name, text + "\n\n" + BLOCK)
            return request

        # (c) Message list with an existing system message.
        messages, attach = _find_message_list(request)
        if messages is None:
            return request

        idx = _find_system_message(messages)
        if idx >= 0:
            content = messages[idx].get("content")
            if MARKER in content:
                return request
            messages[idx]["content"] = content + "\n\n" + BLOCK
            return request

        # Fallback: no system message anywhere, no marker anywhere -> prepend.
        if _marker_in_messages(messages):
            return request
        new_msg = {"role": "system", "content": BLOCK}
        messages.insert(0, new_msg)
        if attach is not None:
            kind, name = attach
            _write_system_field(request, kind, name, messages)
        return request

    def transform_response(self, response, public_history):
        """Pure pass-through; stores no state."""
        return response


def register():
    return None
