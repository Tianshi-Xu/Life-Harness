"""Relay out-of-schema device intent to the customer instead of letting it fail.

Why: the solver's callable schema exposes only account-side functions, while the
phone-level device actions are executed by the simulated customer. A call to one
of those device functions is executed by nobody: it returns a "Tool not found"
error, burns a turn, and is frequently followed by an early hand-off. Rewriting
exactly those turns into one plain-language request converts a dead turn into a
customer-visible action. Shape-robust and fail-safe: any unexpected shape,
failed write, or failed read-back leaves the response untouched.
"""
import json

try:  # the customer's phone-side toolkit defines which names are device actions
    from tau2.domains.telecom.user_tools import TelecomUserTools as _UserTools
except Exception:  # tau2 absent -> no name gating, see _should_relay
    _UserTools = None
try:  # account-side toolkit, the schema-unseen fallback for allowed names
    from tau2.domains.telecom.tools import TelecomTools as _AccountTools
except Exception:
    _AccountTools = None


def _public_names(cls, skip_assert=False):
    """Public method names of a toolkit class; empty when the class is absent."""
    if cls is None:
        return frozenset()
    return frozenset(n for n in dir(cls) if not n.startswith("_")
                     and not (skip_assert and n.startswith("assert_"))
                     and callable(getattr(cls, n, None)))


# Evaluator-only predicates are never customer actions and are not relayed.
_DEVICE_NAMES = _public_names(_UserTools, skip_assert=True)
_ACCOUNT_NAMES = _public_names(_AccountTools, skip_assert=True)
_RELAY_PREFIX = ("I can't reach your device directly - please do this on your "
                 "phone and tell me what you see: ")
_RELAY_AGAIN_PREFIX = ("If you have already done that, just tell me what your "
                       "phone shows now: ")
# Purely lexical rendering of a function name into a general-English action.
_VERB_MAP = {
    "check": "check", "toggle": "switch", "set": "set", "reset": "reset",
    "restart": "restart", "reboot": "restart", "enable": "turn on",
    "disable": "turn off", "grant": "grant permission for",
    "revoke": "revoke permission for", "lock": "lock", "unseat": "remove",
    "run": "try", "send": "try",
}


def _field(payload, name):
    """Read a field from a dict payload or an attribute-style payload."""
    if isinstance(payload, dict):
        return payload.get(name)
    return getattr(payload, name, None)


def _set_field(payload, name, value):
    """Write a field; False when the payload shape is not writable."""
    try:
        if isinstance(payload, dict):
            payload[name] = value
        else:
            setattr(payload, name, value)
        return True
    except Exception:
        return False


def _locate_message(response, depth=0):
    """Assistant message payload inside the response, or None if unrecognized."""
    def is_message(payload):
        if isinstance(payload, dict):
            return "tool_calls" in payload or "role" in payload
        return payload is not None and hasattr(payload, "tool_calls")

    if response is None or depth > 3:
        return None
    if isinstance(response, (list, tuple)):
        fallback = None  # first message-like item; one that has calls wins
        for item in response:
            message = _locate_message(item, depth + 1)
            if message is not None and _field(message, "tool_calls"):
                return message
            fallback = fallback or message
        return fallback
    choices = _field(response, "choices")
    if isinstance(choices, (list, tuple)) and choices:
        message = _field(choices[0], "message")
        if is_message(message):
            return message
        nested = _locate_message(choices[0], depth + 1)
        if nested is not None:
            return nested
    return response if is_message(response) else None


def _call_name(call):
    """Function name of a tool call in nested, flattened, or attribute form."""
    name = _field(_field(call, "function"), "name")
    if not isinstance(name, str) or not name:
        name = _field(call, "name")
    return name if isinstance(name, str) and name else None


def _render_action(call):
    """Lexically render a call as a plain-English action for the customer."""
    words = [w for w in str(_call_name(call) or "").split("_") if w]
    if not words:
        return None
    head = _VERB_MAP.get(words[0].lower(), words[0])
    if any(w in words[1:] for w in head.split()[1:]):
        head = head.split()[0]  # avoids "grant permission for app permission"
    action = " ".join([head] + words[1:])
    args = _field(_field(call, "function"), "arguments")
    if args is None:
        args = _field(call, "arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = None  # arguments are decoration, never block the relay
    if isinstance(args, dict) and args:
        action += " with " + ", ".join(f"{k} = {v}" for k, v in args.items())
    return action


def _append_text(content, text):
    """Return content with text appended; never mutates the input in place."""
    if content is None:
        return text, True
    if isinstance(content, str):
        return ((content + "\n\n" + text) if content.strip() else text), True
    if isinstance(content, (list, tuple)):
        parts = list(content)  # copy: the original stays restorable
        last = parts[-1] if parts else None
        if isinstance(last, dict) and isinstance(last.get("text"), str):
            parts[-1] = dict(last)
            parts[-1]["text"] += "\n\n" + text
        else:
            parts.append({"type": "text", "text": text})
        return parts, True
    return content, False


def _content_text(content):
    """Plain text of a message content, used for read-back verification."""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        return "\n".join(str(_field(part, "text")) for part in content)
    return ""


def _should_relay(name, allowed, device):
    """True only for readable names outside the schema but on the device side."""
    if not isinstance(name, str) or not name or name in allowed:
        return False  # unreadable or schema-visible names are never rewritten
    return not device or name in device


def _schema_names(request):
    """Tool/function names advertised in a model request, if discoverable."""
    container = None
    for key in ("tools", "functions"):
        value = _field(request, key)
        if value is not None:
            container = value
            break
    if isinstance(container, dict):
        names = {k for k in container if isinstance(k, str)}  # dict-keyed schema
        entries = list(container.values())
    elif isinstance(container, (list, tuple)):
        names, entries = set(), list(container)
    else:
        return set()
    for entry in entries:
        name = _field(entry, "name") or _field(_field(entry, "function"), "name")
        if isinstance(name, str) and name:
            names.add(name)
    return names


class AgentHarness:
    """Rewrites out-of-schema device calls into a request the customer can act on."""

    def __init__(self, public_context=None):
        self.public_context = public_context
        self._allowed_names = set()  # tool names captured from the request schema
        self._relay_counts = {}  # per episode: name -> relays already requested

    def transform_request(self, request):
        """Identity; the only side effect is recording the callable tool names."""
        try:
            names = _schema_names(request)
            if names:
                self._allowed_names = names
        except Exception:
            pass
        return request

    def transform_response(self, response, public_history):
        try:
            self._relay_device_turns(response)
        except Exception:
            pass  # fail-safe: never let rewriting break a turn
        return response

    def _relay_device_turns(self, response):
        message = _locate_message(response)
        if message is None:
            return
        calls = _field(message, "tool_calls")
        if not isinstance(calls, (list, tuple)) or not calls:
            return  # no calls to rewrite: leave the response alone
        allowed = self._allowed_names or _ACCOUNT_NAMES  # schema-unseen fallback
        found = [(i, c) for i, c in enumerate(calls)
                 if _should_relay(_call_name(c), allowed, _DEVICE_NAMES)]
        if not found:
            return  # every call is valid: touch nothing
        kept = [c for i, c in enumerate(calls) if i not in {i for i, _ in found}]
        name = _call_name(found[0][1])
        request_text = new_content = None
        if not kept:
            # all calls are device actions: ask the customer once, drop the rest
            action = _render_action(found[0][1])
            if action is None:
                return
            prefix = _RELAY_AGAIN_PREFIX if self._relay_counts.get(name) else _RELAY_PREFIX
            request_text = prefix + action + "."
            new_content, ok = _append_text(_field(message, "content"), request_text)
            if not ok:
                return
        # snapshot before mutating so a failed write or read-back can be undone
        old_calls, old_content = _field(message, "tool_calls"), _field(message, "content")
        try:
            if kept:
                _set_field(message, "tool_calls", list(kept))
                readback = _field(message, "tool_calls")
                ok = (isinstance(readback, list) and len(readback) == len(kept)
                      and all(a is b for a, b in zip(readback, kept)))
            else:
                _set_field(message, "tool_calls", None)
                _set_field(message, "content", new_content)
                readback = _field(message, "content")
                ok = (_field(message, "tool_calls") is None
                      and request_text in _content_text(readback))
        except Exception:
            ok = False
        if not ok:
            _set_field(message, "tool_calls", old_calls)
            _set_field(message, "content", old_content)
            return
        if not kept:
            self._relay_counts[name] = self._relay_counts.get(name, 0) + 1


def register():
    return None
