"""Harness variant: keep the agent inside its own toolset.

A customer-support conversation mixes two kinds of work. Some of it -- looking
up accounts, handling billing, suspending or roaming a line -- is done by the
agent with the tools it holds. Much of it, though, happens on the customer's own
device: reading the status bar or the network state, and flipping settings such
as airplane mode, mobile data, data saver, roaming, network mode, APN, SIM, app
permissions and VPN. The support manual lists those device steps, so a model can
mistake them for its own tools and emit a call for one of them. The runtime
rejects every such call, and because an episode ends after a fixed number of
errors, a handful of misplaced calls can end the conversation before the real
problem is solved.

This harness keeps the agent's intent but fixes its channel. On the way to the
model it states, once, that device-side steps belong to the customer and have to
be requested in plain language. If the model still emits a call for a step it
does not hold, the same rule is applied on the way back: the call is turned into
an ordinary spoken request so the customer performs it.

Only the current request and the model's own output are inspected. No task
identity, hidden state, or cross-episode state is used.
"""

from __future__ import annotations

import json
import re

# The tools the agent actually holds (public native schema).
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

_ROLES = frozenset({"system", "user", "assistant", "tool"})

_MARK = "<tool_use_rule>"
_POLICY_ANCHORS = ("What the user can do on their device", "Telecom Agent Policy")

_NOTE = (
    "\n\n"
    + _MARK
    + "\n"
    + "You may act ONLY through the tools listed in your tool schema. Those tools "
    + "cover account lookup, billing and payment, line suspension, roaming on the "
    + "line, and data refueling.\n"
    + "Everything that happens on the customer's own device is performed by the "
    + "CUSTOMER, not by you. That includes reading the device (status bar, network "
    + "status, network-mode preference, SIM status, data-restriction status, APN "
    + "settings, Wi-Fi and Wi-Fi-Calling status, VPN status, installed apps, app "
    + "permissions and status, speed tests, and whether messaging can send) and "
    + "changing the device (airplane mode, mobile data, data saver, data roaming, "
    + "Wi-Fi, Wi-Fi Calling, network mode, APN set or reset, SIM reseat, app "
    + "permissions, VPN connect or disconnect, reboot).\n"
    + "Never emit a tool call for any of those device-side steps; the runtime will "
    + "reject it. Instead, ask the customer to do the step in plain language, one "
    + "step at a time, and wait for their reply. For example: \"Please turn "
    + "Airplane Mode off on your phone and tell me what the status bar shows.\"\n"
    + "Before you call a tool, confirm its name is one of the tools you were given; "
    + "if it is not, ask the customer instead of calling it.\n"
    + "</tool_use_rule>"
)

_DELEGATE_PREFIX = (
    "I can't run that step from my side -- it has to be done on your phone. "
)
_MAX_STEPS = 4


def _get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _has(obj, key):
    if isinstance(obj, dict):
        return key in obj
    return hasattr(obj, key)


def _role(obj):
    value = _get(obj, "role", None)
    return value if isinstance(value, str) else None


def _tool_name(call):
    fn = _get(call, "function", call)
    name = _get(fn, "name", None)
    return name if isinstance(name, str) and name else None


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


def _phrase(name):
    words = [w for w in re.split(r"[_\s]+", str(name).strip()) if w]
    return " ".join(words) if words else "that step"


def _apply_updates(obj, updates):
    """Set several fields on a plain/pydantic object; return the (possibly new) object."""
    if isinstance(obj, dict):
        obj.update(updates)
        return obj
    ok = True
    for key, value in updates.items():
        try:
            setattr(obj, key, value)
        except Exception:
            ok = False
            break
    if ok:
        return obj
    try:
        for key, value in updates.items():
            object.__setattr__(obj, key, value)
        return obj
    except Exception:
        pass
    copy_fn = getattr(obj, "model_copy", None)
    if callable(copy_fn):
        try:
            return copy_fn(update=updates)
        except Exception:
            pass
    return None


def _replace_in(node, transform, seen):
    """Depth-first rewrite. ``transform(value)`` returns ``(value, changed)``.

    Dicts and lists are edited in place; tuples and frozen objects are rebuilt
    and re-attached to their parent, so the change survives whatever container
    shape the runtime hands us.
    """
    if node is None or isinstance(node, (str, int, float, bool)):
        new, changed = transform(node)
        return new if changed else node
    if id(node) in seen:
        new, changed = transform(node)
        return new if changed else node
    new, changed = transform(node)
    if changed:
        node = new
    seen.add(id(node))
    if isinstance(node, dict):
        for key in list(node.keys()):
            updated = _replace_in(node[key], transform, seen)
            if updated is not node[key]:
                node[key] = updated
    elif isinstance(node, list):
        for i in range(len(node)):
            updated = _replace_in(node[i], transform, seen)
            if updated is not node[i]:
                node[i] = updated
    elif isinstance(node, tuple):
        node = tuple(_replace_in(item, transform, seen) for item in node)
    else:
        data = getattr(node, "__dict__", None)
        if isinstance(data, dict):
            updates = {}
            for key in list(data.keys()):
                updated = _replace_in(data[key], transform, seen)
                if updated is not data[key]:
                    updates[key] = updated
            if updates:
                replaced = _apply_updates(node, updates)
                if replaced is not None:
                    node = replaced
    return node


def _inject(value):
    if (
        isinstance(value, str)
        and _MARK not in value
        and any(anchor in value for anchor in _POLICY_ANCHORS)
    ):
        return value.rstrip() + _NOTE, True
    return value, False


def _handoff(calls):
    steps = []
    for call in calls:
        phrase = _phrase(_tool_name(call) or "that step")
        args = _tool_args(call)
        details = [
            "%s=%s" % (key, args[key])
            for key in sorted(args)
            if args[key] not in (None, "", [], {})
        ]
        steps.append("%s (%s)" % (phrase, ", ".join(details)) if details else phrase)
    steps = steps[:_MAX_STEPS] or ["that step"]
    return (
        _DELEGATE_PREFIX
        + "Please do the following on your phone and tell me what you see: "
        + "; ".join(steps)
        + "."
    )


def _rewrite(value):
    calls = _get(value, "tool_calls", None)
    if not isinstance(calls, list) or not calls:
        return value, False
    if not (_has(value, "content") or _role(value) in _ROLES):
        return value, False
    names = [_tool_name(call) for call in calls]
    invalid = [c for c, n in zip(calls, names) if n not in _AGENT_TOOLS]
    if not invalid:
        return value, False
    valid = [c for c, n in zip(calls, names) if n in _AGENT_TOOLS]
    updates = {"tool_calls": valid}
    if not valid:
        text = _handoff(invalid)
        content = _get(value, "content", None)
        if isinstance(content, str) and content.strip():
            text = content.rstrip() + "\n\n" + text
        updates["content"] = text
    if isinstance(value, dict):
        value.update(updates)
        return value, True
    replaced = _apply_updates(value, updates)
    if replaced is None:
        return value, False
    return replaced, True


class AgentHarness:
    """Keeps the agent's actions inside its own toolset: device-side steps are the customer's job."""

    def __init__(self, public_context=None):
        self.public_context = public_context

    def transform_request(self, request):
        try:
            return _replace_in(request, _inject, set())
        except Exception:
            return request

    def transform_response(self, response, public_history=None):
        try:
            return _replace_in(response, _rewrite, set())
        except Exception:
            return response


def register():
    return None
