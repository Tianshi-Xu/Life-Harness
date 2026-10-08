"""Harness variant: route device-side actions back to the user.

The agent is offered only a small set of account/service tools. Plenty of the
work in a support conversation belongs to the *customer*: looking at a phone
screen, flipping a settings switch, re-inserting a SIM, rebooting a handset,
running a connectivity test, granting an app a permission, and so on. Those
steps are carried out by the user, who reports the outcome back in plain
language; the agent is expected to ask for them.

The model nevertheless sometimes emits one of those user-side actions as if it
were its own tool. The runtime answers every such call with a tool error, and
because an episode stops once it has spent its fixed error budget, a single
misplaced call can end the conversation before the real problem is solved.

This harness keeps the model's *intent* but changes its *channel*: any tool call
whose name is not among the tools the agent actually holds is rewritten into an
ordinary spoken request addressed to the user, so the user performs the step and
the episode keeps its error budget for genuine mistakes.

Only the model's own (public) output message is inspected. No task identity,
hidden state, or cross-episode state is used.
"""

from __future__ import annotations

import json
import re

# The tools the agent is actually offered (public native schema).
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

_PREFIX = (
    "I'm not able to carry that out from my side -- it has to be done on your "
    "phone."
)
_MAX_STEPS = 4


def _get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _set(obj, key, value):
    if isinstance(obj, dict):
        obj[key] = value
        return
    try:
        setattr(obj, key, value)
    except Exception:
        pass


def _tool_name(call):
    fn = _get(call, "function", call)
    name = _get(fn, "name", None)
    if isinstance(name, str) and name:
        return name
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
            return {"value": args}
        if isinstance(parsed, dict):
            return parsed
    return {}


def _phrase(name):
    words = [w for w in re.split(r"[_\s]+", str(name).strip()) if w]
    return " ".join(words) if words else "that step"


def _step_text(name, args):
    phrase = _phrase(name)
    details = []
    for key in sorted(args):
        value = args[key]
        if value in (None, "", [], {}):
            continue
        details.append("%s=%s" % (key, value))
    if details:
        return "%s (%s)" % (phrase, ", ".join(details))
    return phrase


def _handoff_text(calls):
    steps = [_step_text(_tool_name(c) or "that step", _tool_args(c)) for c in calls]
    if not steps:
        steps = ["that step"]
    steps = steps[:_MAX_STEPS]
    joined = "; ".join(steps)
    return (
        "%s Please do the following on your device and tell me what you see: %s."
        % (_PREFIX, joined)
    )


def _rewrite_container(container):
    """Rewrite one message-like object that carries ``tool_calls``.

    Returns True if anything changed.
    """
    calls = _get(container, "tool_calls", None)
    if not isinstance(calls, list) or not calls:
        return False

    valid = [c for c in calls if _tool_name(c) in _AGENT_TOOLS]
    invalid = [c for c in calls if _tool_name(c) not in _AGENT_TOOLS]
    if not invalid:
        return False

    if valid:
        # Keep the genuine calls; simply drop the unusable ones.
        _set(container, "tool_calls", valid)
        return True

    text = _handoff_text(invalid)
    content = _get(container, "content", None)
    if isinstance(content, str) and content.strip():
        text = content.rstrip() + "\n\n" + text
    _set(container, "content", text)
    _set(container, "tool_calls", [])
    return True


def _walk(obj, seen):
    if id(obj) in seen:
        return
    seen.add(id(obj))
    if isinstance(obj, dict):
        if isinstance(obj.get("tool_calls"), list):
            _rewrite_container(obj)
        for value in list(obj.values()):
            _walk(value, seen)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            _walk(value, seen)


class AgentHarness:
    """Model-output post-processor that redirects off-schema actions to the user."""

    def __init__(self, public_context=None):
        self.public_context = public_context

    def transform_request(self, request):
        # Nothing to change on the way out: the mechanism is response-side only.
        return request

    def transform_response(self, response, public_history):
        try:
            if isinstance(response, (dict, list, tuple)):
                _walk(response, set())
            elif response is not None and hasattr(response, "tool_calls"):
                _rewrite_container(response)
        except Exception:
            # Never let the guard itself break an episode.
            return response
        return response


def register():
    return None
