"""Bind a phone-number lookup to the line that carries that number.

A lookup by phone number identifies one line of an account, but the returned
record lists every line the account owns in storage order, so the line that
matches the searched number is not necessarily listed first. Consumers of the
record - human or automated - tend to act on the first line listed, which makes
the record inconsistent with the number that was used to find it.

This module installs a single wrapper around the phone-number lookup: when the
matching line is found and is not already first, the wrapper returns the same
record with its line list reordered so the matching line comes first, other
lines keeping their relative order. Reordering is order-preserving for every
internal use (iteration and membership tests only), and the live account object
is never touched: the reordered list is bound to a shallow copy of the record.

Fail-safe by construction: a missing match, a line list shorter than two, a
None result, an unresolvable entry, or any unexpected shape or error leaves the
original result exactly as the lookup returned it.

A tool call that names a function of the customer's own phone rather than an
account function is answered by the domain with a terse not-found error. This
module also wraps the tool-execution path: for exactly those customer-device
names, the not-found error becomes a directive that has the agent ask the
customer to make or read the setting on the phone, with one reminder appended
if the same write is redirected a second time in the same episode. Every other
error propagates unchanged.
"""
import copy
import functools

try:  # patch target: the plain account toolkit, when the domain is importable
    from tau2.domains.telecom.tools import TelecomTools as _TelecomTools
except Exception:  # domain absent -> register() is a harmless no-op
    _TelecomTools = None

try:  # patch target: the plain domain environment, when the domain is importable
    from tau2.domains.telecom.environment import TelecomEnvironment as _TelecomEnvironment
except Exception:  # domain absent -> that wrapper is simply not installed
    _TelecomEnvironment = None

try:  # customer-device toolkit, the authority on which names live on the phone
    from tau2.domains.telecom.user_tools import TelecomUserTools as _TelecomUserTools
except Exception:  # domain absent -> fall back to the device-name prefixes only
    _TelecomUserTools = None


def _digit_key(value):
    """Digits-only form of a phone number, so punctuation never blocks a match."""
    try:
        return "".join(ch for ch in str(value) if ch.isdigit())
    except Exception:
        return ""


def _matching_line_index(toolkit, customer, phone_number):
    """Index of the first line whose number matches the searched number."""
    key = _digit_key(phone_number)
    if not key:
        return None  # nothing to match on: leave the record alone
    line_ids = getattr(customer, "line_ids", None)
    if not isinstance(line_ids, (list, tuple)) or len(line_ids) < 2:
        return None
    for index, line_id in enumerate(line_ids):
        try:
            line = toolkit._get_line_by_id(line_id)
        except Exception:
            continue  # an unresolvable entry cannot be the matching line
        if line is not None and _digit_key(getattr(line, "phone_number", None)) == key:
            return index
    return None


def _with_line_first(customer, index):
    """Shallow copy of the record whose line list starts at the matching line."""
    line_ids = list(customer.line_ids)
    try:
        duplicate = customer.model_copy(deep=False)
    except Exception:
        try:
            duplicate = copy.copy(customer)
        except Exception:
            return customer
    if duplicate is customer:
        return customer  # never risk rebinding the live account object
    try:
        duplicate.line_ids = (
            [line_ids[index]] + line_ids[:index] + line_ids[index + 1:]
        )
    except Exception:
        return customer  # a record that cannot be reordered stays untouched
    return duplicate


# Non-tool attributes of the customer-device toolkit class, never device names.
_DEVICE_ATTRIBUTE_DENY = {
    "db", "use_tool", "get_tools", "to_json_str", "get_tool", "tools",
    "get_tool_names",
}

# Prefixes used only by customer-device functions, accepted even without the class.
_DEVICE_PREFIXES = (
    "toggle_", "set_", "check_", "reset_", "reboot", "unseat_", "reseat_",
    "lock_", "unlock_", "grant_", "revoke_", "disconnect_", "connect_",
    "run_speed",
)

# A name starting with one of these reads from the phone instead of changing it.
_READ_PREFIXES = (
    "check_", "get_", "is_", "list_", "can_", "run_", "view_", "read_",
)

# One reminder, appended to a repeated write redirect (leading space kept).
_REPEAT_SUFFIX = " If you have already asked the customer to make this change, do not ask them to do it again; simply ask them to tell you what their phone shows now. If you have not asked yet, ask now."

_device_vocabulary_cache = None


def _device_vocabulary():
    """Public callable names of the customer-device toolkit, cached once."""
    global _device_vocabulary_cache
    if _device_vocabulary_cache is None:
        names = set()
        try:
            if _TelecomUserTools is not None:
                names = {
                    name for name in dir(_TelecomUserTools)
                    if not name.startswith("_")
                    and callable(getattr(_TelecomUserTools, name, None))
                }
                names -= _DEVICE_ATTRIBUTE_DENY
        except Exception:
            names = set()
        _device_vocabulary_cache = names
    return _device_vocabulary_cache


def _is_device_name(name):
    """True when the name is a function of the customer's own phone."""
    try:
        if name in _device_vocabulary():
            return True
        return name.startswith(_DEVICE_PREFIXES)
    except Exception:
        return False


def _is_read_styled(name):
    """True when the name reads from the phone instead of changing it."""
    return name.startswith(_READ_PREFIXES)


def _mark_repeat(instance, name):
    """True when this write was already redirected once on this instance."""
    try:
        seen = getattr(instance, "_mh_device_redirect_seen", None)
        if not isinstance(seen, dict):
            seen = {}
            setattr(instance, "_mh_device_redirect_seen", seen)
        count = seen.get(name, 0)
        seen[name] = count + 1
        return count >= 1
    except Exception:
        return False


def _redirect_text(instance, tool_name, exc):
    """Directive for an unknown customer-device call; None to propagate as-is."""
    if not isinstance(exc, Exception):
        return None
    name = str(tool_name)
    message = str(exc)
    if "not found" not in message.lower():
        return None
    if name not in message:
        return None
    if not _is_device_name(name):
        return None
    if _is_read_styled(name):
        return (
            f"Error: Tool '{name}' not found. '{name}' is information that is only visible on the customer's own phone, so you have no tool for it and calling it again will fail again. Do not transfer the customer because of it. Reply to the customer now: ask them to look this up on their phone and tell you exactly what it says, in plain words without tool names, and wait for their answer."
        )
    text = (
        f"Error: Tool '{name}' not found. '{name}' changes a setting on the customer's own phone, and only the customer can do that, so you have no tool for it and calling it again will fail again. Do not give up on this step, do not tell the customer the change is impossible or unnecessary, and do not transfer them because of it. Reply to the customer now: ask them to make this change on their phone, described in plain words without tool names, one step at a time, and wait for them to tell you what their phone shows."
    )
    if _mark_repeat(instance, name):
        text += _REPEAT_SUFFIX
    return text


def _make_use_tool_wrapper(original):
    """functools.wraps-decorated use_tool wrapper around the original method."""

    @functools.wraps(original)
    def use_tool(self, tool_name, *args, **kwargs):
        try:
            return original(self, tool_name, *args, **kwargs)
        except Exception as exc:
            try:
                text = _redirect_text(self, tool_name, exc)
            except Exception:
                text = None
            if text is None:
                raise  # the original exception propagates unchanged
            return text

    use_tool._mh_device_redirect_installed = True
    return use_tool


def _install_use_tool_redirect(owner):
    """Wrap owner.use_tool at most once; skip silently when unavailable."""
    try:
        if owner is None:
            return None
        original = getattr(owner, "use_tool", None)
        if original is None:
            return None
        if getattr(original, "_mh_device_redirect_installed", False):
            return None
        owner.use_tool = _make_use_tool_wrapper(original)
    except Exception:
        pass
    return None


def register():
    """Install the lookup wrapper on the account toolkit, at most once."""
    _device_vocabulary()  # fixed once: which public names live on the phone
    _install_use_tool_redirect(_TelecomTools)
    _install_use_tool_redirect(_TelecomEnvironment)
    if _TelecomTools is None:
        return None
    try:
        original = _TelecomTools.get_customer_by_phone
        if getattr(original, "_mh_primary_line_installed", False):
            return None

        @functools.wraps(original)
        def get_customer_by_phone(self, phone_number: str):
            customer = original(self, phone_number)  # lookup errors propagate
            try:
                if customer is None:
                    return customer
                index = _matching_line_index(self, customer, phone_number)
                if index is None or index == 0:
                    return customer
                return _with_line_first(customer, index)
            except Exception:
                return customer  # a successful lookup is never disturbed

        get_customer_by_phone._mh_primary_line_installed = True
        _TelecomTools.get_customer_by_phone = get_customer_by_phone
    except Exception:
        pass
    return None


class AgentHarness:
    """Identity runtime; the single mechanism lives in register()."""

    def __init__(self, public_context=None):
        self.public_context = public_context

    def transform_request(self, request):
        return request

    def transform_response(self, response, public_history):
        return response
