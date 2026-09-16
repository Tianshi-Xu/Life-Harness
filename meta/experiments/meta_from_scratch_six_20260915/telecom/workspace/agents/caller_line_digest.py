"""Fold the facts a line read omits into the line read itself.

Two facts about an account are present in the domain data but are never stated
by a single read:

* how much data the plan of a line allows, so that a reported usage can be
  judged as over or under its allowance, and
* which of the account's lines the caller actually owns, which is only known to
  the phone-number lookup that resolved that account.

A lookup by phone number resolves one line of the account, but the returned
record lists every line in storage order, so the line that answers the searched
number is not necessarily listed first, and no read afterwards says which line
that was. A plain line read reports usage without the plan allowance, so the
over-limit fact has to be assembled from several reads before it can be seen.

This module installs three wrappers around the plain account toolkit. The
lookup wrapper keeps returning the same record with its line list reordered so
the matching line comes first, other lines keeping their relative order, on a
shallow copy of the record - the live account object is never touched - and it
remembers, once per toolkit instance, the id of the line that answered the
lookup, falling back to the record's own contact number when the search itself
resolves no line. The two read wrappers leave every read result untouched
except for line-shaped ones: those come back as a plain dictionary that also
carries the plan's data allowance, whether the line has reached that allowance,
and, when the caller's line is known, whether the read describes that line.

Fail-safe by construction: an unexpected shape, a missing attribute, a plan
that cannot be resolved, a non-numeric amount, or any raised error returns the
original result exactly as the tool produced it. The remembered line id is an
attribute of the per-episode toolkit instance, written at most once.
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


def _line_id_at(toolkit, customer, index):
    """Line id held at an index of the record's list, resolved when not a plain id."""
    try:
        line_id = customer.line_ids[index]
    except Exception:
        return None
    if isinstance(line_id, str) and line_id:
        return line_id
    try:
        line = toolkit._get_line_by_id(line_id)
    except Exception:
        return None
    resolved = getattr(line, "line_id", None)
    return resolved if isinstance(resolved, str) and resolved else None


def _lone_contact_line_id(toolkit, customer):
    """The only line whose number equals the record's contact number, if unique."""
    key = _digit_key(getattr(customer, "phone_number", None))
    if not key:
        return None
    line_ids = getattr(customer, "line_ids", None)
    if not isinstance(line_ids, (list, tuple)):
        return None
    found = None
    for line_id in line_ids:
        try:
            line = toolkit._get_line_by_id(line_id)
        except Exception:
            continue  # an unresolvable entry cannot be judged either way
        if line is None or _digit_key(getattr(line, "phone_number", None)) != key:
            continue
        if found is not None:
            return None  # several candidates: never remember a guess
        found = getattr(line, "line_id", None) or line_id
    return found if isinstance(found, str) and found else None


def _remember_caller_line(toolkit, customer, index):
    """Remember the caller's line once per instance; a set memo is never replaced."""
    try:
        memo = getattr(toolkit, "_mh_caller_line_id", None)
        if isinstance(memo, str) and memo:
            return  # the first successful resolution wins
        line_id = None
        if index is not None:
            line_id = _line_id_at(toolkit, customer, index)
        if line_id is None:
            line_id = _lone_contact_line_id(toolkit, customer)
        if isinstance(line_id, str) and line_id:
            toolkit._mh_caller_line_id = line_id  # instance attribute only
    except Exception:
        return  # a memo that cannot be written leaves the lookup result untouched


def _caller_line_keys(toolkit, line_id):
    """Association keys for a line read, only when the caller's line is known."""
    memo = getattr(toolkit, "_mh_caller_line_id", None)
    if not isinstance(memo, str) or not memo:
        return {}
    return {"is_customer_line": line_id == memo, "customer_line_id": memo}


def _as_number(value):
    """Float form of a numeric amount, or None when it is not a usable number."""
    try:
        if isinstance(value, bool) or value is None:
            return None
        number = float(value)
    except Exception:
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None  # a non-finite amount cannot be compared with an allowance
    return number


def _allowance(used_value, limit_value, refuel_value):
    """Allowance and over-limit status of a line, or None when unknown."""
    used = _as_number(used_value)
    limit = _as_number(limit_value)
    refuel = 0.0 if refuel_value is None else _as_number(refuel_value)
    if used is None or limit is None or refuel is None:
        return None  # never emit a partial or guessed comparison
    return limit, bool(used >= limit + refuel)


def _payload_dict(result):
    """Plain-dict form of a model result, or None when none can be produced."""
    for method_name in ("model_dump", "dict"):
        method = getattr(result, method_name, None)
        if not callable(method):
            continue
        for kwargs in (({"mode": "json"}, {}) if method_name == "model_dump" else ({},)):
            try:
                payload = method(**kwargs)
            except Exception:
                continue
            if isinstance(payload, dict):
                return payload
    return None


def register():
    """Install the lookup and read wrappers on the account toolkit, at most once."""
    if _TelecomTools is None:
        return None

    # Phone-number lookup: matching line first, and the caller's line remembered.
    try:
        original_lookup = _TelecomTools.get_customer_by_phone
        if not getattr(original_lookup, "_mh_primary_line_installed", False):

            @functools.wraps(original_lookup)
            def get_customer_by_phone(self, phone_number: str):
                customer = original_lookup(self, phone_number)  # lookup errors propagate
                try:
                    if customer is None:
                        return customer
                    index = _matching_line_index(self, customer, phone_number)
                    try:
                        _remember_caller_line(self, customer, index)
                    except Exception:
                        pass  # an unusable memo never disturbs the record
                    if index is None or index == 0:
                        return customer
                    return _with_line_first(customer, index)
                except Exception:
                    return customer  # a successful lookup is never disturbed

            get_customer_by_phone._mh_primary_line_installed = True
            _TelecomTools.get_customer_by_phone = get_customer_by_phone
    except Exception:
        pass

    # Detail reads: a line-shaped result carries its plan allowance and status.
    try:
        original_details = _TelecomTools.get_details_by_id
        if not getattr(original_details, "_mh_digest_installed", False):

            @functools.wraps(original_details)
            def get_details_by_id(self, id: str):
                result = original_details(self, id)  # lookup errors propagate
                try:
                    line_id = getattr(result, "line_id", None)
                    if not isinstance(line_id, str) or not line_id:
                        return result  # records without a line identity pass through
                    if not hasattr(result, "plan_id") or not hasattr(
                        result, "data_used_gb"
                    ):
                        return result  # not a line-shaped read
                    try:
                        plan = self._get_plan_by_id(result.plan_id)
                    except Exception:
                        return result  # an unresolvable plan states nothing
                    allowance = _allowance(
                        getattr(result, "data_used_gb", None),
                        getattr(plan, "data_limit_gb", None),
                        getattr(result, "data_refueling_gb", None),
                    )
                    if allowance is None:
                        return result  # never emit a partial or guessed status
                    payload = _payload_dict(result)
                    if payload is None:
                        return result  # no plain form: keep the original object
                    payload["plan_data_limit_gb"] = allowance[0]
                    payload["data_limit_exceeded"] = allowance[1]
                    payload.update(_caller_line_keys(self, line_id))
                    return payload
                except Exception:
                    return result  # a successful read is never disturbed

            get_details_by_id._mh_digest_installed = True
            _TelecomTools.get_details_by_id = get_details_by_id
    except Exception:
        pass

    # Usage reads: the same derivation, applied to the usage dictionary.
    try:
        original_usage = _TelecomTools.get_data_usage
        if not getattr(original_usage, "_mh_digest_installed", False):

            @functools.wraps(original_usage)
            def get_data_usage(self, customer_id: str, line_id: str):
                result = original_usage(self, customer_id, line_id)  # errors propagate
                try:
                    if not isinstance(result, dict):
                        return result
                    allowance = _allowance(
                        result.get("data_used_gb"),
                        result.get("data_limit_gb"),
                        result.get("data_refueling_gb"),
                    )
                    if allowance is None:
                        return result  # never emit a partial or guessed status
                    payload = dict(result)  # a copy: the tool's dict is never mutated
                    payload["data_limit_exceeded"] = allowance[1]
                    payload.update(_caller_line_keys(self, result.get("line_id")))
                    return payload
                except Exception:
                    return result  # a successful read is never disturbed

            get_data_usage._mh_digest_installed = True
            _TelecomTools.get_data_usage = get_data_usage
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
