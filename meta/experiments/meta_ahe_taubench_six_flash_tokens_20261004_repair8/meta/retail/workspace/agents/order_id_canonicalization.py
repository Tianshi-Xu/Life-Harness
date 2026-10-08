"""Order-id canonicalization middleware for the tau2 retail LLM agent.

ONE mechanism: a pure, non-instructional argument repair that rewrites a tool
call's ``order_id`` argument into the domain's stored spelling ``"#W<digits>"``
before the call executes.  The 4B solver routinely drops the leading ``#`` (and
sometimes the ``W``) when it retranscribes an id it already saw, which makes the
native lookup fail with ``Error: Order not found``.  In the public trajectories
every non-canonical-but-normalizable order_id argument failed (115/115) while
every canonical one succeeded (1155/1155), so the repair is monotone: it can
only turn a guaranteed miss into the lookup the model intended, never a success
into a failure.

No instruction text is injected; no message is added, removed or edited; no
model / network / filesystem / subprocess access, no hidden state, no task-id
tables, no cross-episode mutable state; unexpected structures pass unchanged.
"""
import json
import re

_ORDER_ID_RE = re.compile(r"#?[Ww]?(\d{4,12})\Z")   # canonical stored order id
_TARGET_TOOL = "get_order_details"                  # only lookup taking order_id
_MESSAGE_KEYS = ("messages", "input", "history", "chat")


def canonical_order_id(value):
    """Canonical ``#W<digits>``, or None for non-string / junk / canonical input."""
    if not isinstance(value, str):
        return None
    s = value.strip()
    m = _ORDER_ID_RE.match(s)
    if not m:
        return None
    fixed = "#W" + m.group(1)
    return fixed if fixed != s else None


def _get(obj, key):
    if obj is None:
        return None
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


def _set(obj, key, value):
    if isinstance(obj, dict):
        obj[key] = value
    else:
        setattr(obj, key, value)
    return True


def _fix_args_dict(args):
    if not isinstance(args, dict):
        return False
    fixed = canonical_order_id(args.get("order_id"))
    if fixed is None:
        return False
    args["order_id"] = fixed
    return True


def _fix_args(holder):
    """Fix the ``order_id`` in a holder's arguments (dict or JSON string)."""
    args = _get(holder, "arguments")
    if isinstance(args, dict):
        return _fix_args_dict(args)
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
        except Exception:
            return False
        if not _fix_args_dict(parsed):
            return False
        try:
            return _set(holder, "arguments", json.dumps(parsed))
        except Exception:
            return False
    return False


def _rewrite_tool_call(tc):
    """Rewrite one tool call's order_id argument in place; True if it changed."""
    try:
        fn = _get(tc, "function")
        holder = fn if fn is not None else tc
        if _get(holder, "name") != _TARGET_TOOL:
            return False
        return _fix_args(holder)
    except Exception:
        return False


def _iter_tool_calls(response):
    """Yield tool-call containers reachable on any common response shape."""
    try:
        containers = []
        if isinstance(response, dict):
            containers.append(response.get("tool_calls"))
            containers.append(_get(response.get("message"), "tool_calls"))
            choices = response.get("choices")
        elif isinstance(response, list):
            choices = None
            for item in response:         # bare list of tool calls (or choices)
                inner = _get(item, "tool_calls")
                if isinstance(inner, list):
                    containers.extend(inner)
                else:
                    yield item
        else:
            containers.append(getattr(response, "tool_calls", None))
            choices = getattr(response, "choices", None)
        if isinstance(choices, list):
            for choice in choices:
                containers.append(_get(_get(choice, "message"), "tool_calls"))
        for calls in containers:
            if isinstance(calls, list):
                for c in calls:
                    yield c
    except Exception:
        return


def _message_list(request):
    """Return the request's non-empty list-of-dicts message list, or None."""
    def is_msgs(v):
        return isinstance(v, list) and len(v) > 0 and all(
            isinstance(m, dict) for m in v)
    try:
        if isinstance(request, dict):
            for key in _MESSAGE_KEYS:
                if is_msgs(request.get(key)):
                    return request[key]
            return None
        value = getattr(request, "messages", None)
        return value if is_msgs(value) else None
    except Exception:
        return None


class AgentHarness:
    def __init__(self, public_context=None):
        self.public_context = public_context

    def transform_request(self, request):
        """Defensive: repair pending (unexecuted) tool calls on the tail only."""
        try:
            messages = _message_list(request)
            if not messages:
                return request
            last = messages[-1]
            if not isinstance(last, dict) or last.get("role") != "assistant":
                return request
            calls = last.get("tool_calls")
            if not isinstance(calls, list):
                return request
            for call in calls:
                try:
                    _rewrite_tool_call(call)
                except Exception:
                    pass
        except Exception:
            pass
        return request

    def transform_response(self, response, public_history):
        """Repair order_id arguments on the freshly produced assistant turn."""
        try:
            for call in _iter_tool_calls(response):
                try:
                    _rewrite_tool_call(call)
                except Exception:
                    pass
        except Exception:
            pass
        return response


def register():
    return None
