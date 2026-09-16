"""Pre-execution enforcement of documented airline policy preconditions on DB-mutating tool calls the plain domain API leaves unchecked."""

from tau2.domains.airline.tools import AirlineTools

_DAY = 24 * 3600


def _epoch(ts):
    # Julian-day arithmetic; the ``datetime`` module is not an allowed import here.
    y, mo, d = int(ts[0:4]), int(ts[5:7]), int(ts[8:10])
    hh, mm, ss = int(ts[11:13]), int(ts[14:16]), int(ts[17:19])
    a = (14 - mo) // 12
    y, m = y + 4800 - a, mo + 12 * a - 3
    jdn = d + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045
    return (jdn - 2440588) * _DAY + hh * 3600 + mm * 60 + ss


def _recent(now, created):
    try:
        return _epoch(now) - _epoch(created) <= _DAY
    except (TypeError, ValueError):
        return False


def _status(db, flight_number, date):
    flight = db.flights.get(flight_number)
    instance = getattr(flight, "dates", {}).get(date) if flight is not None else None
    return getattr(instance, "status", None)


def _statuses(db, res):
    return [_status(db, f.flight_number, f.date) for f in res.flights]


def _cancel_gate(tools, res):
    now = tools._get_datetime()
    today = now[:10]
    facts, statuses, flown = [], [], []
    for f in res.flights:
        status = _status(tools.db, f.flight_number, f.date)
        facts.append(f"{f.flight_number}@{f.date}:{status}")
        statuses.append(status)
        if status in ("flying", "landed") or (isinstance(f.date, str) and f.date < today):
            flown.append(f"{f.flight_number}@{f.date}:{status}")
    observed = ", ".join(facts) or "none"
    if res.status == "cancelled":
        raise ValueError(
            f"POLICY BLOCK (cancel_reservation {res.reservation_id}): this reservation is already cancelled. "
            f"Observed: cabin={res.cabin}, insurance={res.insurance}, created_at={res.created_at}, now={now}, segments={observed}. "
            f"Do not cancel it again - tell the user the reservation is already cancelled and offer only policy-compliant options."
        )
    if flown:
        raise ValueError(
            f"POLICY BLOCK (cancel_reservation {res.reservation_id}): a reservation cannot be cancelled once any of its flights has departed or been flown; "
            f"the request must be transferred to a human agent. "
            f"Observed: cabin={res.cabin}, insurance={res.insurance}, created_at={res.created_at}, now={now}, segments={observed}. "
            f"Departed/flown segments: {', '.join(flown)}. "
            f"Do not cancel - explain that a portion of the itinerary has already departed or been flown and transfer the user to a human agent."
        )
    if not (res.cabin == "business" or res.insurance == "yes" or _recent(now, res.created_at) or "cancelled" in statuses):
        raise ValueError(
            f"POLICY BLOCK (cancel_reservation {res.reservation_id}): a reservation can only be cancelled if it was booked within the last 24 hours, "
            f"the airline cancelled the flight, it is a business flight, or the user has travel insurance. "
            f"Observed: cabin={res.cabin}, insurance={res.insurance}, created_at={res.created_at}, now={now}, segments={observed}. "
            f"Unmet: booked more than 24 hours ago; no segment cancelled by the airline; cabin is not business; no travel insurance. "
            f"Do not cancel - deny the request per policy, explain which condition is missing, offer only policy-compliant options, "
            f"and transfer the user to a human agent if it cannot be handled."
        )


def _update_gate(tools, res, flights):
    resolved = []
    for entry in flights:
        fn = entry.get("flight_number") if isinstance(entry, dict) else getattr(entry, "flight_number", None)
        flight = tools.db.flights.get(fn) if isinstance(fn, str) else None
        if flight is None:
            return
        resolved.append(flight)
    first_origin = resolved[0].origin
    last_destination = resolved[-1].destination
    expected_last = res.origin if res.flight_type == "round_trip" else res.destination
    if first_origin != res.origin or last_destination != expected_last:
        raise ValueError(
            f"POLICY BLOCK (update_reservation_flights {res.reservation_id}): reservations can be modified without changing the origin, destination, and trip type. "
            f"Observed: reservation origin={res.origin}, destination={res.destination}, flight_type={res.flight_type}; "
            f"proposed first origin={first_origin}, last destination={last_destination}. "
            f"Do not change the route - keep the original origin, destination, and trip type, or deny the request and explain that origin/destination cannot change."
        )


def _certificate_gate(tools, user, amount):
    matched = []
    for rid in user.reservations:
        res = tools.db.reservations.get(rid)
        if res is None:
            continue
        statuses = _statuses(tools.db, res)
        passengers = len(res.passengers)
        if ("cancelled" in statuses and amount == 100 * passengers) or ("delayed" in statuses and amount == 50 * passengers):
            matched.append(res)
    if not matched:
        raise ValueError(
            f"POLICY BLOCK (send_certificate {user.user_id}): a compensation certificate may only be sent for a flight cancelled by the airline "
            f"($100 per passenger) or a delayed flight ($50 per passenger). "
            f"Observed: user_id={user.user_id}, amount={amount}; no reservation of this user has a cancelled or delayed flight supporting that amount. "
            f"Do not send the certificate - deny the compensation request and explain the policy."
        )
    if user.membership not in ("silver", "gold") and not any(r.insurance == "yes" or r.cabin == "business" for r in matched):
        details = ", ".join(f"{r.reservation_id}(insurance={r.insurance},cabin={r.cabin})" for r in matched)
        raise ValueError(
            f"POLICY BLOCK (send_certificate {user.user_id}): do not compensate a regular member who has no travel insurance and flies basic economy. "
            f"Observed: user_id={user.user_id}, membership={user.membership}, amount={amount}, matching reservations={details}. "
            f"Do not send the certificate - deny the compensation request per policy and explain the eligibility conditions."
        )


def _gate(tools, tool_name, kwargs):
    if tool_name == "cancel_reservation":
        rid = kwargs.get("reservation_id")
        res = tools.db.reservations.get(rid) if isinstance(rid, str) else None
        if res is not None:
            _cancel_gate(tools, res)
    elif tool_name == "update_reservation_flights":
        rid = kwargs.get("reservation_id")
        res = tools.db.reservations.get(rid) if isinstance(rid, str) else None
        flights = kwargs.get("flights")
        if res is not None and flights:
            _update_gate(tools, res, flights)
    elif tool_name == "send_certificate":
        uid = kwargs.get("user_id")
        user = tools.db.users.get(uid) if isinstance(uid, str) else None
        amount = kwargs.get("amount")
        if user is not None and isinstance(amount, int):
            _certificate_gate(tools, user, amount)


def register():
    if getattr(AirlineTools, "_write_policy_gate", False):
        return
    original_use_tool = AirlineTools.use_tool

    def use_tool(self, tool_name, **kwargs):
        _gate(self, tool_name, kwargs)
        return original_use_tool(self, tool_name, **kwargs)

    AirlineTools.use_tool = use_tool
    AirlineTools._write_policy_gate = True


class AgentHarness:
    def __init__(self, public_context=None):
        self.public_context = public_context
    def transform_request(self, request):
        return request
    def transform_response(self, response, public_history):
        return response
