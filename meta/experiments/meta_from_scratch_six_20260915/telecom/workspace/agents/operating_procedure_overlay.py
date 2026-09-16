"""Turn-invariant operating-procedure overlay appended to the system message."""

OVERLAY_MARKER = "[OPERATING PROCEDURE"

OVERLAY = """[OPERATING PROCEDURE - FOLLOW ON EVERY TURN]
1. Your callable tools are ONLY the functions listed in your own tool schema. The device functions in the policy (the customer's phone checks and settings actions) are NOT callable by you - they run on the customer's device. To use one, ask the customer in plain language to perform exactly one action, then wait for their reply. Never treat such a step as complete until the customer confirms it.
2. Before any account-level change, resolve the exact account and line identifiers from information the customer gave you (look them up instead of guessing), and use only those identifiers. Never apply a change to a different line than the customer's.
3. Do not say an issue is resolved, and do not transfer or end the conversation, while any suspected cause of the customer's symptom is still unaddressed. If the symptom persists, revisit both the device-side settings and the account-side state (usage, plan, billing and suspension status) before repeating a fix that already failed."""


def register():
    return None


def _field(message, name):
    """Read a field from a dict message or an attribute-style message object."""
    if isinstance(message, dict):
        return message.get(name)
    return getattr(message, name, None)


def _content_text(content):
    """Best-effort plain text of a message content (str, parts list, or None)."""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        chunks = []
        for part in content:
            text = _field(part, "text")
            if isinstance(text, str):
                chunks.append(text)
        return "\n".join(chunks)
    if content is None:
        return ""
    text = getattr(content, "text", None)
    return text if isinstance(text, str) else ""


def _append_text(content, text):
    """Append text to message content, mutating in place when possible.

    Returns a (new_content, ok) pair; ok is False when the shape is unknown.
    """
    if content is None:
        return text, True
    if isinstance(content, str):
        return content + "\n\n" + text, True
    if isinstance(content, (list, tuple)):
        for part in reversed(content):
            if isinstance(part, dict):
                value = part.get("text")
                if isinstance(value, str):
                    part["text"] = value + "\n\n" + text
                    return content, True
            else:
                value = getattr(part, "text", None)
                if isinstance(value, str):
                    try:
                        setattr(part, "text", value + "\n\n" + text)
                    except Exception:
                        continue
                    return content, True
        new_part = {"type": "text", "text": text}
        if isinstance(content, list):
            content.append(new_part)
            return content, True
        return list(content) + [new_part], True
    return content, False


def _set_content(message, content):
    """Write content back onto a dict message or an attribute-style message."""
    if isinstance(message, dict):
        message["content"] = content
    else:
        setattr(message, "content", content)


def _locate_messages(request):
    """Return the message list of a request, or None when no list is found."""
    if isinstance(request, list):
        return request
    if isinstance(request, dict):
        messages = request.get("messages")
        return messages if isinstance(messages, list) else None
    messages = getattr(request, "messages", None)
    return messages if isinstance(messages, list) else None


class AgentHarness:
    """Appends a fixed operating procedure to the first system message."""

    def __init__(self, public_context=None):
        self.public_context = public_context

    def transform_request(self, request):
        try:
            messages = _locate_messages(request)
            if messages is None:
                return request

            system_message = None
            for message in messages:
                role = _field(message, "role")
                if isinstance(role, str) and role.lower() == "system":
                    system_message = message
                    break

            if system_message is None:
                messages.insert(0, {"role": "system", "content": OVERLAY})
                return request

            content = _field(system_message, "content")
            if OVERLAY_MARKER in _content_text(content):
                return request

            new_content, ok = _append_text(content, OVERLAY)
            if not ok:
                return request
            if new_content is not content:
                _set_content(system_message, new_content)
            return request
        except Exception:
            return request

    def transform_response(self, response, public_history):
        return response
