"""Harness variant: keep every agent action inside its own 13-tool schema.

A telecom support conversation mixes two kinds of work. Account work -- looking
up a customer, billing, payment requests, suspending or roaming a line, data
refueling -- is done by the AGENT with the functions it holds. Device work --
reading the status bar or the network/SIM/APN/Wi-Fi/VPN state, running a
connectivity or messaging test, and changing airplane mode, mobile data, data
saver, roaming, Wi-Fi, Wi-Fi Calling, network mode, APN, SIM, app permissions
or rebooting -- is done by the CUSTOMER on their own phone. The support manual
merely *describes* those device steps, so a small model can mistake them for its
own tools and emit a call for one of them. The runtime rejects every such call
and the episode ends once a fixed error budget is spent, so a couple of
misplaced calls can kill an otherwise solvable episode.

This harness keeps the model's intent but fixes its channel. Everything below is
defensive: each hook is independently wrapped, never raises, and returns the
original object whenever nothing needs to change.

Channel 1 (transform_response, primary): walk the model's own reply, in whatever
shape it arrives, to the message-like containers that carry ``tool_calls``
(the critical case is a LiteLLM ``ModelResponse``: ``choices[i].message.tool_calls``).
Split those calls into native (name in the 13) and off-schema. If native calls
remain, keep only the native ones. If none remain, turn the turn into ordinary
speech: set ``content`` to a plain request asking the customer to perform the
step on their phone and set ``tool_calls`` to ``[]`` so the turn is delivered as
a message instead of a rejected call.

Channel 2 (transform_request, fallback): append, at most once, a short rule to
the text of the request message carrying the native policy anchor
("Telecom Agent Policy" / "What the user can do on their device"), preferring a
system/developer message. The rule states that the agent may act only through its
own tool list and that all on-device steps are the customer's to perform and must
be asked for in plain words.

Channel 3 (transform_request, history repair): only matters when channel 1 did
not intercept an off-schema call, i.e. when a tool-role message actually reports
an unknown/unavailable tool. That one error is met with a single marker-guarded
reminder that the step is the customer's to do and must be asked for in words.
Other errors (e.g. "Customer with ID ... not found") are left untouched.

Only the current request and the model's own output are inspected. No task
identity, no hidden state, no per-task hints, no cross-episode mutable state.
"""

from __future__ import annotations

import json
import re

# The 13 tools the agent actually holds (static, public native schema).
_AGENT_TOOLS = frozenset(
    {
        "get_customer_by_phone",
        "get_customer_by_id",
        "get_customer_by_name",
        "get_details_by_id",
        "suspend_line",
        "resume_line",
        "get_bills_for_customer",
        "send_payment_request",
        "get_data_usage",
        "enable_roaming",
        "disable_roaming",
        "transfer_to_human_agents",
        "refuel_data",
    }
)

_POLICY_ANCHORS = ("Telecom Agent Policy", "What the user can do on their device")

_RULE_MARK = "<off_schema_action_routing>"
_RULE = (
    "\n\n"
    + _RULE_MARK
    + "\n"
    + "You may act ONLY through the functions in your own tool list. Everything "
    + "done on the phone itself is performed by the CUSTOMER, not by you: reading "
    + "device state (status bar, network/SIM/APN/Wi-Fi/VPN/app settings, "
    + "connectivity and messaging tests) and changing it (airplane mode, mobile "
    + "data, data saver, roaming, Wi-Fi, Wi-Fi Calling, network mode, APN, SIM "
    + "reseat, app permissions, reboot). Never emit a tool call for any of those "
    + "steps -- ask the customer to do that one step in plain words, wait for "
    + "their reply, and continue from what they report.\n"
    + _RULE_MARK.replace("<", "</")
)

_REMINDER_MARK = "[off-schema-step]"
_REMINDER = (
    "\n"
    + _REMINDER_MARK
    + " That step is the customer's to perform on their own phone -- ask them in "
    + "plain words to do it and report back instead of calling a tool for it."
)

# Unknown / unavailable tool error signatures (never a "customer not found").
_UNKNOWN_TOOL_RES = (
    re.compile(r"(?i)\btool\b[^.\n]{0,80}\bnot\s+found\b"),
    re.compile(r"(?i)\bnot\s+a\s+valid\s+tool\b"),
    re.compile(r"(?i)\bunknown\s+tool\b"),
    re.compile(r"(?i)\bno\s+such\s+tool\b"),
    re.compile(r"(?i)\btool\b[^.\n]{0,80}\b(?:unavailable|does\s+not\s+exist|not\s+available)\b"),
)

# Keep only short, non-identifier hints in a spoken request; never echo ids.
_ID_KEYS_RE = re.compile(r"(?i)(id|phone|number|sim|imei)")
_LONG_DIGITS_RE = re.compile(r"\d{6,}")

_DELEGATE_OPEN = (
    "That step has to be done on your phone and I can't run it from my side. "
    "Please do the following on your device and tell me what you see: "
)
_MAX_STEPS = 4


# --------------------------------------------------------------------------- #
# Guarded get / set helpers (dict and plain-object safe)
# --------------------------------------------------------------------------- #
def _get(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        try:
            return obj.get(key, default)
        except Exception:
            return default
    try:
        return getattr(obj, key, default)
    except Exception:
        return default


def _has(obj, key):
    if obj is None:
        return False
    if isinstance(obj, dict):
        return key in obj
    try:
        return hasattr(obj, key)
    except Exception:
        return False


def _set(obj, key, value):
    """Set a field on a dict or plain object; return True only on success."""
    if obj is None:
        return False
    if isinstance(obj, dict):
        try:
            obj[key] = value
            return True
        except Exception:
            return False
    try:
        setattr(obj, key, value)
        return True
    except Exception:
        pass
    try:
        object.__setattr__(obj, key, value)
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Tool-call inspection
# --------------------------------------------------------------------------- #
def _tool_name(call):
    fn = _get(call, "function", call)
    name = _get(fn, "name", None)
    if isinstance(name, str) and name.strip():
        return name.strip()
    return None


def _tool_args(call):
    fn = _get(call, "function", call)
    args = _get(fn, "arguments", None)
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
        except Exception:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def _humanise(name):
    words = [w for w in re.split(r"[_\s]+", str(name).strip()) if w]
    return " ".join(words) if words else "that step"


def _hints(args):
    """Short, non-identifier key=value hints only; never echo ids/long digits."""
    out = []
    for key in sorted(args):
        name = str(key)
        if _ID_KEYS_RE.search(name):
            continue
        value = args[key]
        if isinstance(value, (dict, list, tuple)):
            continue
        if value is None or value == "" or isinstance(value, bool):
            continue
        text = str(value)
        if _LONG_DIGITS_RE.search(text) or len(text) > 32:
            continue
        out.append("%s=%s" % (name, text))
        if len(out) >= 3:
            break
    return out


def _spoken_request(calls):
    steps = []
    for call in calls[:_MAX_STEPS]:
        phrase = _humanise(_tool_name(call) or "that step")
        hints = _hints(_tool_args(call))
        steps.append(phrase + (" (" + ", ".join(hints) + ")" if hints else ""))
    if not steps:
        steps = ["that step"]
    return _DELEGATE_OPEN + "; ".join(steps) + "."


# --------------------------------------------------------------------------- #
# Channel 1 — route off-schema calls in the model's own reply
# --------------------------------------------------------------------------- #
def _route_container(container):
    calls = _get(container, "tool_calls", None)
    if not isinstance(calls, list) or not calls:
        return
    native = []
    off_schema = []
    for call in calls:
        if _tool_name(call) in _AGENT_TOOLS:
            native.append(call)
        else:
            off_schema.append(call)
    if not off_schema:
        return
    if native:
        # Keep the agent's own calls; drop only the off-schema ones.
        _set(container, "tool_calls", native)
        return
    # No native call remains -> deliver the turn as ordinary speech.
    if not _set(container, "tool_calls", []):
        return
    text = _spoken_request(off_schema)
    content = _get(container, "content", None)
    if isinstance(content, str) and content.strip():
        _set(container, "content", content.rstrip() + "\n\n" + text)
    else:
        _set(container, "content", text)


def _walk_reply(node, seen):
    """Shape-agnostic descent to message-like containers holding tool_calls."""
    if node is None:
        return
    oid = id(node)
    if oid in seen:
        return
    seen.add(oid)

    if isinstance(node, dict):
        if isinstance(node.get("tool_calls"), list):
            _route_container(node)
        for key in ("choices", "message"):
            if key in node:
                _walk_reply(node[key], seen)
        return

    if isinstance(node, (list, tuple)):
        for item in node:
            _walk_reply(item, seen)
        return

    # Plain objects: LiteLLM ModelResponse -> .choices -> .message -> .tool_calls
    if isinstance(_get(node, "tool_calls", None), list):
        _route_container(node)
    for key in ("choices", "message"):
        child = _get(node, key, None)
        if child is not None:
            _walk_reply(child, seen)


# --------------------------------------------------------------------------- #
# Channels 2 & 3 — request-side rule injection and history repair
# --------------------------------------------------------------------------- #
def _content_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            else:
                text = _get(part, "text", None)
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return ""


def _role(obj):
    role = _get(obj, "role", None)
    return role if isinstance(role, str) else None


def _message_list(request):
    if request is None:
        return []
    if isinstance(request, dict):
        messages = request.get("messages")
        if isinstance(messages, (list, tuple)):
            return list(messages)
        if "messages" in request:
            return []
        if "role" in request or "content" in request:
            return [request]
        return []
    if isinstance(request, (list, tuple)):
        return list(request)
    if _has(request, "messages"):
        messages = _get(request, "messages", None)
        if isinstance(messages, (list, tuple)):
            return list(messages)
        return []
    if _has(request, "role") or _has(request, "content"):
        return [request]
    return []


def _has_anchor(message):
    text = _content_text(_get(message, "content", None))
    return any(anchor in text for anchor in _POLICY_ANCHORS)


def _append_rule(message):
    content = _get(message, "content", None)
    if isinstance(content, str):
        if _RULE_MARK in content:
            return
        _set(message, "content", content.rstrip() + _RULE)
    elif isinstance(content, list):
        if _RULE_MARK in _content_text(content):
            return
        content.append({"type": "text", "text": _RULE.strip()})
    elif isinstance(content, tuple):
        if _RULE_MARK in _content_text(content):
            return
        _set(message, "content", list(content) + [{"type": "text", "text": _RULE.strip()}])


def _inject_rule(request):
    if request is None:
        return None
    if isinstance(request, str):
        if _RULE_MARK in request or not any(a in request for a in _POLICY_ANCHORS):
            return request
        return request.rstrip() + _RULE
    messages = _message_list(request)
    if not messages:
        return request
    target = None
    for message in messages:
        if _role(message) in ("system", "developer") and _has_anchor(message):
            target = message
            break
    if target is None:
        for message in messages:
            if _has_anchor(message):
                target = message
                break
    if target is not None:
        _append_rule(target)
    return request


def _is_unknown_tool_error(text):
    return any(pattern.search(text) for pattern in _UNKNOWN_TOOL_RES)


def _repair_history(request):
    """Only reached when channel 1 did not intercept an off-schema call.

    The presence of an unknown-tool error tool-message is itself the evidence
    that the misplaced call was executed, so we add one reminder there. Other
    errors are left exactly as they are.
    """
    for message in _message_list(request):
        if _role(message) != "tool":
            continue
        content = _get(message, "content", None)
        if isinstance(content, str):
            if _REMINDER_MARK in content or not _is_unknown_tool_error(content):
                continue
            _set(message, "content", content.rstrip() + _REMINDER)
        elif isinstance(content, list):
            if _REMINDER_MARK in _content_text(content):
                continue
            for i, part in enumerate(content):
                text = part if isinstance(part, str) else _get(part, "text", None)
                if not isinstance(text, str) or not _is_unknown_tool_error(text):
                    continue
                if isinstance(part, str):
                    content[i] = text.rstrip() + _REMINDER
                else:
                    _set(part, "text", text.rstrip() + _REMINDER)
                break


# --------------------------------------------------------------------------- #
# Plugin surface
# --------------------------------------------------------------------------- #
class AgentHarness:
    """Routes off-schema (customer-device) actions to the customer in plain words."""

    def __init__(self, public_context=None):
        self.public_context = public_context

    def transform_request(self, request):
        try:
            request = _inject_rule(request)
        except Exception:
            pass
        try:
            _repair_history(request)
        except Exception:
            pass
        return request

    def transform_response(self, response, public_history=None):
        try:
            _walk_reply(response, set())
        except Exception:
            pass
        return response


def register():
    return None
