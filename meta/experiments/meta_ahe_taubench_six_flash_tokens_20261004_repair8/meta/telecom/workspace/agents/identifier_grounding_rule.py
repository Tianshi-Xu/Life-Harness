"""Harness variant: ground every identifier argument in real data.

A support agent acts through a small set of tools, many of which take an
identifier: a customer id, a line id, a bill id, a plan id, a device id, or a
phone number. Those values are only meaningful when they come from the actual
conversation -- something the customer said, or something an earlier tool call
returned. A value that is invented, guessed, or copied from an example in the
manual simply does not exist, so the call fails and (because an episode ends
once it has spent its fixed error budget) a few such calls can end the
conversation before the customer's problem is solved.

This harness keeps the agent's intent but grounds its inputs: on the way to the
model it states, once, that an identifier argument must be an exact value from
the conversation, and that when the needed identifier is not yet known the
agent must obtain it first -- by asking the customer in plain words, or by
calling the lookup tool that returns it -- and then copy that exact value into
later calls. The same reminder rides alongside the descriptions of the tools
that take identifiers.

The rule is delivered through the agent's own public surface: it is appended
(idempotently, marker-guarded) to any message text carrying the native policy,
to the tools that take identifiers, to the agent's own policy/system text, and,
as a fallback, to outgoing requests whose text carries the native policy. No
task identity, hidden state, new model calls, filesystem/network access, or
cross-episode state is used.
"""

from __future__ import annotations

# Stable anchors for the agent's own policy text (public native policy).
_ANCHORS = (
    "Telecom Agent Policy",
    "What the user can do on their device",
)

# Marker used to make every injection idempotent.
_MARK = "<identifier_grounding>"

# The one rule, appended once to policy text.
_RULE = (
    "<identifier_grounding>\n"
    "When you call a tool that takes an identifier (a customer id, line id, bill "
    "id, plan id, device id, or phone number), the value you pass must come from "
    "this conversation: either something the customer told you, or a value "
    "returned by an earlier tool call. Never invent, guess, or copy an example / "
    "placeholder value - a made-up identifier will not resolve and the call will "
    "fail.\n"
    "If you do not yet have the identifier a call needs, obtain it first: ask the "
    "customer for it in plain words, or call the lookup tool that returns it, and "
    "then use the exact value you receive. When a lookup returns identifiers, read "
    "them out of the result and copy them exactly into the later calls.\n"
    "</identifier_grounding>"
)

# One-line reminder appended to the descriptions of identifier-taking tools.
# Carries its own marker so it is appended at most once.
_DESC_MARK = "[identifier-grounding]"
_REMINDER = (
    " " + _DESC_MARK + " The identifier argument must be an exact value that the "
    "customer gave you or that an earlier tool call returned - never a guessed, "
    "invented, or example value."
)

# Native tools whose call takes an identifier (public native schema).
_IDENTIFIER_TOOLS = frozenset(
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
        "refuel_data",
    }
)

_MAX_DEPTH = 4
_MAX_ITEMS = 200


def _has_anchor(text):
    if not isinstance(text, str):
        return False
    for anchor in _ANCHORS:
        if anchor in text:
            return True
    return False


def _inject_str(text):
    """Append the rule once to a policy string; return the (possibly new) string."""
    if not isinstance(text, str) or not text:
        return text
    if _MARK in text:
        return text
    if _has_anchor(text):
        return text.rstrip() + "\n\n" + _RULE
    return text


def _set_item(container, key, value):
    """Write a value back into a dict/list slot, guarded; never raise."""
    try:
        container[key] = value
        return True
    except Exception:
        return False


def _patch_schema_dict(node, seen):
    """Append the reminder to an identifier-taking tool's description."""
    fn = node.get("function")
    if not isinstance(fn, dict) or id(fn) in seen:
        return
    name = fn.get("name")
    if not isinstance(name, str) or name not in _IDENTIFIER_TOOLS:
        return
    desc = fn.get("description")
    if isinstance(desc, str) and _DESC_MARK not in desc:
        try:
            fn["description"] = desc + _REMINDER
        except Exception:
            pass


def _walk(node, seen, depth):
    """Bounded, cycle-safe, never-raising walk that grounds policy strings."""
    if depth > _MAX_DEPTH:
        return
    if isinstance(node, str) or node is None:
        return
    nid = id(node)
    if nid in seen:
        return
    seen.add(nid)
    if isinstance(node, dict):
        try:
            _patch_schema_dict(node, seen)
        except Exception:
            pass
        try:
            items = list(node.items())[:_MAX_ITEMS]
        except Exception:
            return
        for key, value in items:
            if isinstance(value, str):
                new = _inject_str(value)
                if new != value:
                    _set_item(node, key, new)
            else:
                _walk(value, seen, depth + 1)
    elif isinstance(node, (list, tuple)):
        is_list = isinstance(node, list)
        try:
            items = list(node)[:_MAX_ITEMS]
        except Exception:
            return
        for index, value in enumerate(items):
            if isinstance(value, str):
                new = _inject_str(value)
                if new != value and is_list:
                    _set_item(node, index, new)
            else:
                _walk(value, seen, depth + 1)
    else:
        data = getattr(node, "__dict__", None)
        if isinstance(data, dict):
            try:
                items = list(data.items())[:_MAX_ITEMS]
            except Exception:
                return
            for key, value in items:
                if isinstance(value, str):
                    new = _inject_str(value)
                    if new != value:
                        try:
                            setattr(node, key, new)
                        except Exception:
                            pass
                else:
                    _walk(value, seen, depth + 1)


def _wrap_domain(instance):
    """Ground the rule on a domain object's own public policy callables."""
    try:
        domain = getattr(instance, "domain", None)
    except Exception:
        return
    if domain is None:
        return
    try:
        names = dir(domain)
    except Exception:
        return
    for name in names:
        try:
            if name.startswith("__") or not name.islower() or "polic" not in name:
                continue
        except Exception:
            continue
        try:
            attr = getattr(domain, name)
        except Exception:
            continue
        if not callable(attr):
            continue
        if getattr(attr, "_identifier_grounding_wrapped", False):
            continue
        try:
            owned = name in getattr(domain, "__dict__", {}) or name in getattr(
                type(domain), "__dict__", {}
            )
        except Exception:
            owned = False
        if not owned:
            continue

        def _make_wrapper(original):
            def _wrapper(*args, **kwargs):
                result = original(*args, **kwargs)
                if isinstance(result, str):
                    return _inject_str(result)
                return result

            _wrapper._identifier_grounding_wrapped = True
            return _wrapper

        try:
            setattr(domain, name, _make_wrapper(attr))
        except Exception:
            continue


def _post_process_instance(instance):
    """Best-effort grounding walk over a freshly built LLMAgent instance."""
    try:
        data = getattr(instance, "__dict__", None)
    except Exception:
        data = None
    if isinstance(data, dict):
        try:
            _walk(data, set(), 0)
        except Exception:
            pass
    try:
        _wrap_domain(instance)
    except Exception:
        pass


def _patch_llm_agent():
    """Best-effort: ride the rule on the native agent's own prompt and tools."""
    try:
        import tau2.agent.llm_agent as _module
    except Exception:
        return
    cls = getattr(_module, "LLMAgent", None)
    if cls is None or getattr(cls, "_identifier_grounding_rule", False):
        return
    try:
        _orig_init = cls.__init__

        def _init(self, *args, **kwargs):
            _orig_init(self, *args, **kwargs)
            try:
                _post_process_instance(self)
            except Exception:
                pass

        cls.__init__ = _init

        _sp = getattr(cls, "system_prompt", None)
        if isinstance(_sp, property) and _sp.fget is not None:
            _orig_fget = _sp.fget

            def _fget(self):
                return _inject_str(_orig_fget(self))

            cls.system_prompt = property(_fget, _sp.fset, _sp.fdel, _sp.__doc__)

        cls._identifier_grounding_rule = True
    except Exception:
        return


# --------------------------------------------------------------------------
# Request-side fallback (used when register() cannot reach the prompt).
# --------------------------------------------------------------------------

def _inject_content(content):
    """Ground a message content that is a string or a list of text parts."""
    if isinstance(content, str):
        new = _inject_str(content)
        return new, new != content
    if isinstance(content, list):
        changed = False
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                new = _inject_str(part["text"])
                if new != part["text"]:
                    try:
                        part["text"] = new
                        changed = True
                    except Exception:
                        pass
        return content, changed
    return content, False


def _inject_message(message):
    """Ground one message dict; return True if it changed."""
    if not isinstance(message, dict):
        return False
    new, changed = _inject_content(message.get("content"))
    if changed:
        try:
            message["content"] = new
        except Exception:
            return False
    return changed


def _inject_messages(messages):
    """Ground once, preferring a system/developer message, else the first match."""
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


def _ground_container(request):
    """Also ground tool schemas that ride along inside a request, guarded."""
    try:
        _walk(request, set(), 0)
    except Exception:
        pass


def _transform_request(request):
    """Ground the agent's own policy text inside an outgoing request."""
    if isinstance(request, str):
        return _inject_str(request)
    if isinstance(request, dict):
        messages = request.get("messages")
        if isinstance(messages, list):
            _inject_messages(messages)
        else:
            _inject_message(request)
        _ground_container(request)
        return request
    if isinstance(request, list):
        _inject_messages(request)
        _ground_container(request)
        return request
    messages = getattr(request, "messages", None)
    if isinstance(messages, list):
        _inject_messages(messages)
        _ground_container(request)
        return request
    content = getattr(request, "content", None)
    if isinstance(content, (str, list)):
        new, changed = _inject_content(content)
        if changed:
            try:
                setattr(request, "content", new)
            except Exception:
                pass
    _ground_container(request)
    return request


class AgentHarness:
    """Surfaces an identifier-grounding rule on the agent's own public prompt."""

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
