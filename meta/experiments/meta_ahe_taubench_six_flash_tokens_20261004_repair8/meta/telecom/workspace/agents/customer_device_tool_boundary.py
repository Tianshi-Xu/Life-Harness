"""Harness variant: keep the agent's actions inside its own toolset.

A customer-support conversation mixes two kinds of work. Some of it -- account
lookup, billing and payment, line management, roaming and data refueling -- is
done by the agent with the tools it holds. Much of it happens on the customer's
own device: reading phone state (status bar, network status, network-mode
preference, SIM status, data-restriction status, APN settings, Wi-Fi/VPN status,
app permissions, or the result of a connectivity or messaging test) and changing
it (airplane mode, mobile data, data saver, roaming, Wi-Fi, Wi-Fi Calling,
network mode, APN, SIM reseat, app permissions, VPN, reboot).

The support manual lists those device steps, so the model can mistake them for
its own tools and emit a call for one of them. The runtime rejects every such
call, and because an episode ends once it has spent its fixed error budget, a
handful of misplaced calls can end the conversation before the real problem is
solved.

This harness keeps the agent's intent but fixes its channel: on the way to the
model it appends, once, a short rule stating that the agent acts only through
the functions in its own tool list, that every device-side step belongs to the
customer, and that such a step must be requested in plain words and answered by
the customer before the agent continues.

The rule rides on the agent's own public prompt: it is appended (idempotently,
marker-guarded) to any message the harness hands over whose text carries the
native policy, and, as a fallback, to the agent's own policy/system text through
a guarded patch of the native LLMAgent. No task identity, hidden state, new
model calls, or cross-episode state is used.
"""

from __future__ import annotations

# Stable anchors for the agent's own policy text (public native policy).
_ANCHORS = (
    "What the user can do on their device",
    "Telecom Agent Policy",
)

_MARK = "<customer_device_tool_boundary>"

_NOTE = (
    "\n\n"
    + _MARK
    + "\n"
    + "You can act ONLY through the functions defined in your tool list; those "
    + "tools cover customer/account lookup, bills and payments, suspending or "
    + "resuming a line, enabling or disabling line roaming, checking data usage, "
    + "refueling data, and transferring to a human agent.\n"
    + "Everything listed under \"What the user can do on their device\" is "
    + "performed by the CUSTOMER on their phone, not by you. Those steps are not "
    + "among your tools and a tool call for one of them will be rejected by the "
    + "runtime. They include reading device state (status bar, network status, "
    + "network-mode preference, SIM status, data-restriction status, APN settings, "
    + "Wi-Fi status, VPN status, installed apps, app permissions, or the result of "
    + "a connectivity or messaging test) and changing device state (airplane mode, "
    + "mobile data, data saver, roaming, Wi-Fi, Wi-Fi Calling, network mode, APN, "
    + "SIM reseat, app permissions, VPN, or reboot).\n"
    + "When a fix needs such a step, do NOT call a tool for it. Ask the customer "
    + "to do that one step in plain words, wait for their reply, and continue from "
    + "their answer.\n"
    + "</customer_device_tool_boundary>"
)


def _inject_text(text):
    """Append the rule to a policy string once; return the (possibly new) string."""
    if not isinstance(text, str) or not text:
        return text
    if _MARK in text:
        return text
    for anchor in _ANCHORS:
        if anchor in text:
            return text.rstrip() + _NOTE
    return text


def _inject_content(content):
    """Inject into a message content that is a string or a list of text parts."""
    if isinstance(content, str):
        new = _inject_text(content)
        return new, new != content
    if isinstance(content, list):
        changed = False
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                new = _inject_text(part["text"])
                if new != part["text"]:
                    part["text"] = new
                    changed = True
        return content, changed
    return content, False


def _inject_message(message):
    """Inject the rule into one message dict; return True if it changed."""
    if not isinstance(message, dict):
        return False
    new, changed = _inject_content(message.get("content"))
    if changed:
        message["content"] = new
    return changed


def _inject_messages(messages):
    """Inject once, preferring the system message, else the first policy message."""
    if not isinstance(messages, list):
        return False
    for role in ("system", "developer"):
        for message in messages:
            if isinstance(message, dict) and message.get("role") == role:
                if _inject_message(message):
                    return True
    for message in messages:
        if _inject_message(message):
            return True
    return False


def _transform_request(request):
    """Append the rule to the agent's own policy text inside an outgoing request."""
    if isinstance(request, dict):
        if _inject_messages(request.get("messages")):
            return request
        _inject_message(request)
        return request
    if isinstance(request, list):
        _inject_messages(request)
        return request
    if isinstance(request, str):
        return _inject_text(request)
    return request


def _patch_llm_agent():
    """Best-effort: ensure the native agent's own prompt carries the rule."""
    try:
        import tau2.agent.llm_agent as _module
    except Exception:
        return
    cls = getattr(_module, "LLMAgent", None)
    if cls is None or getattr(cls, "_customer_device_tool_boundary", False):
        return
    try:
        _orig_init = cls.__init__

        def _init(self, *args, **kwargs):
            _orig_init(self, *args, **kwargs)
            try:
                for name, value in list(vars(self).items()):
                    new = _inject_text(value)
                    if new != value:
                        try:
                            setattr(self, name, new)
                        except Exception:
                            pass
            except Exception:
                pass

        cls.__init__ = _init

        _sp = getattr(cls, "system_prompt", None)
        if isinstance(_sp, property) and _sp.fget is not None:
            _orig_fget = _sp.fget

            def _fget(self):
                return _inject_text(_orig_fget(self))

            cls.system_prompt = property(_fget, _sp.fset, _sp.fdel, _sp.__doc__)

        cls._customer_device_tool_boundary = True
    except Exception:
        return


class AgentHarness:
    """Surfaces a customer-device tool-boundary rule in the agent's own prompt."""

    def __init__(self, public_context=None):
        self.public_context = public_context

    def transform_request(self, request):
        try:
            return _transform_request(request)
        except Exception:
            return request

    def transform_response(self, response, public_history=None):
        return response


def register():
    try:
        _patch_llm_agent()
    except Exception:
        pass
    return None
