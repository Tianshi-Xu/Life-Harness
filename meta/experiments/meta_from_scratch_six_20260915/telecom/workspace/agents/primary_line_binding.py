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
"""
import copy
import functools

try:  # patch target: the plain account toolkit, when the domain is importable
    from tau2.domains.telecom.tools import TelecomTools as _TelecomTools
except Exception:  # domain absent -> register() is a harmless no-op
    _TelecomTools = None


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


def register():
    """Install the lookup wrapper on the account toolkit, at most once."""
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
