"""Pre-execution enforcement of documented airline policy preconditions on DB-mutating tool calls the plain domain API leaves unchecked,
plus repair of the derived non-free baggage count on booking and baggage-update calls,
plus rejection of written itineraries that are not a connected chain or whose declared endpoints contradict the flights,
plus a post-execution derived-facts digest appended to the two raw READ tools (user and reservation lookup)."""

import json

from tau2.domains.airline.tools import AirlineTools

_DAY = 24 * 3600
_FREE = {
    "regular": {"basic_economy": 0, "economy": 1, "business": 2},
    "silver": {"basic_economy": 1, "economy": 2, "business": 3},
    "gold": {"basic_economy": 2, "economy": 3, "business": 4},
}


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


def _count(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _allowance(membership, cabin):
    table = _FREE.get(membership) if isinstance(membership, str) else None
    return table.get(cabin) if table is not None and isinstance(cabin, str) else None


def _normalize_baggages(tools, tool_name, kwargs):
    # Repair the derived non-free count from the checked-bag allowance; only when the state is provably on the invariant line.
    if tool_name not in ("update_reservation_baggages", "book_reservation"):
        return
    db = getattr(tools, "db", None)
    total, requested = kwargs.get("total_baggages"), kwargs.get("nonfree_baggages")
    if db is None or not (_count(total) and _count(requested)):
        return
    if tool_name == "update_reservation_baggages":
        rid = kwargs.get("reservation_id")
        res = db.reservations.get(rid) if isinstance(rid, str) else None
        if res is None:
            return
        uid = getattr(res, "user_id", None)
        user = db.users.get(uid) if isinstance(uid, str) else None
        per = _allowance(getattr(user, "membership", None), getattr(res, "cabin", None))
        passengers = getattr(res, "passengers", None)
        if per is None or not isinstance(passengers, list) or not passengers:
            return
        n = len(passengers)
        current_total, current_nonfree = getattr(res, "total_baggages", None), getattr(res, "nonfree_baggages", None)
        if not (_count(current_total) and _count(current_nonfree)):
            return
        if current_nonfree != max(0, current_total - per * n):
            return
        normalized = max(0, total - per * n)
    else:
        uid = kwargs.get("user_id")
        user = db.users.get(uid) if isinstance(uid, str) else None
        per = _allowance(getattr(user, "membership", None), kwargs.get("cabin"))
        passengers = kwargs.get("passengers")
        if per is None or not isinstance(passengers, list) or not passengers:
            return
        normalized = max(0, total - per * len(passengers))
    if normalized != requested:
        kwargs["nonfree_baggages"] = normalized


def _chain(tools, flights):
    # Resolve an itinerary to ordered (flight_number, origin, destination); None as soon as any segment cannot be read safely.
    resolved = []
    for entry in flights:
        fn = entry.get("flight_number") if isinstance(entry, dict) else getattr(entry, "flight_number", None)
        flight = tools.db.flights.get(fn) if isinstance(fn, str) else None
        origin = getattr(flight, "origin", None) if flight is not None else None
        destination = getattr(flight, "destination", None) if flight is not None else None
        if not (isinstance(origin, str) and isinstance(destination, str)):
            return None
        resolved.append((fn, origin, destination))
    return resolved


def _continuity_gate(tools, tool_name, kwargs):
    # One invariant: a proposed itinerary must be a connected chain consistent with the booking endpoints.
    if tool_name not in ("update_reservation_flights", "book_reservation"):
        return
    db = getattr(tools, "db", None)
    flights = kwargs.get("flights")
    if db is None or not isinstance(flights, list) or not flights:
        return
    try:
        chain = _chain(tools, flights)
    except Exception:
        return
    if chain is None or not chain:
        return
    rid = kwargs.get("reservation_id")
    label = f"{tool_name} {rid}" if isinstance(rid, str) else tool_name
    observed = ", ".join(f"{fn} {o}->{d}" for fn, o, d in chain)
    for prev, nxt in zip(chain, chain[1:]):
        if prev[2] != nxt[1]:
            raise ValueError(
                f"POLICY BLOCK ({label}): the proposed itinerary is not a connected chain - flight {prev[0]} arrives at {prev[2]} "
                f"but the next flight {nxt[0]} departs from {nxt[1]}. "
                f"Observed: {observed}. "
                f"Do not write an unconnected itinerary - use only flights that connect end to end; "
                f"re-issue the full itinerary as one connected chain, or tell the user the requested itinerary cannot be arranged."
            )
    if tool_name == "book_reservation":
        origin, destination, flight_type = kwargs.get("origin"), kwargs.get("destination"), kwargs.get("flight_type")
        if not (isinstance(origin, str) and isinstance(destination, str)):
            return
        if flight_type == "one_way":
            if chain[0][1] != origin or chain[-1][2] != destination:
                raise ValueError(
                    f"POLICY BLOCK ({label}): the booked flight does not serve the requested route - "
                    f"the booking is {origin}->{destination} but its first flight {chain[0][0]} departs {chain[0][1]} and its last flight {chain[-1][0]} arrives {chain[-1][2]}. "
                    f"Observed: {observed}. "
                    f"Do not book a flight that does not depart from {origin} and arrive at {destination}; re-check the search results and rebook with flights that actually serve that route."
                )
        elif flight_type == "round_trip":
            if chain[0][1] != origin or chain[-1][2] != origin:
                raise ValueError(
                    f"POLICY BLOCK ({label}): the round trip does not start and end at the booking origin - "
                    f"the booking origin is {origin} but its first flight {chain[0][0]} departs {chain[0][1]} and its last flight {chain[-1][0]} arrives {chain[-1][2]}. "
                    f"Observed: {observed}. "
                    f"Do not book a round trip that does not return to {origin}; re-check the search results and rebook a connected round trip."
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
    _normalize_baggages(tools, tool_name, kwargs)


def _context_digest(tools, tool_name, kwargs, result):
    # Post-execution annotation of the two raw READ tools: append compact facts re-derived from the live DB.
    # Read-only and fail-open: it never blocks a call, never changes a write result, and on any problem returns
    # the original result object untouched.
    if tool_name not in ("get_user_details", "get_reservation_details"):
        return result
    if isinstance(result, str) or result is None:
        return result

    def serialized(value):
        # Mirror the plain environment's serialisation of successful tool returns.
        if isinstance(value, str):
            return value
        if callable(getattr(value, "model_dump", None)):
            return json.dumps(value.model_dump(), default=str)
        if isinstance(value, list):
            return json.dumps(
                [
                    item.model_dump()
                    if callable(getattr(item, "model_dump", None))
                    else item
                    for item in value
                ],
                default=str,
            )
        return json.dumps(value, default=str)

    def number(value):
        return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    def flag(value):
        return "yes" if value is True else ("no" if value is False else "?")

    def paid_net(res):
        total, complete = 0, True
        for payment in getattr(res, "payment_history", None) or []:
            amount = number(getattr(payment, "amount", None))
            if amount is None:
                complete = False
            else:
                total += amount
        return total if complete else None

    def person_name(person):
        parts = [
            part
            for part in (getattr(person, "first_name", None), getattr(person, "last_name", None))
            if isinstance(part, str) and part
        ]
        return " ".join(parts) if parts else "unknown"

    def grounds(now, res):
        # The four published cancellation grounds, evaluated against the current DB.
        try:
            within = _recent(now, getattr(res, "created_at", None)) if isinstance(now, str) else None
        except Exception:
            within = None
        try:
            cancelled = "cancelled" in _statuses(tools.db, res)
        except Exception:
            cancelled = None
        cabin, insurance = getattr(res, "cabin", None), getattr(res, "insurance", None)
        return (
            "grounds[within_24h={} airline_cancelled_segment={} business_cabin={} insurance={}]".format(
                flag(within),
                flag(cancelled),
                flag(cabin == "business") if isinstance(cabin, str) else "?",
                flag(insurance == "yes") if isinstance(insurance, str) else "?",
            )
        )

    def baggage_state(membership, cabin, res):
        passengers = getattr(res, "passengers", None)
        count = len(passengers) if isinstance(passengers, list) else None
        parts = []
        total, nonfree = number(getattr(res, "total_baggages", None)), number(
            getattr(res, "nonfree_baggages", None)
        )
        if total is not None:
            parts.append(f"booked_total_bags={total}")
        if nonfree is not None:
            parts.append(f"booked_nonfree_bags={nonfree} ($50 each)")
        per = _allowance(membership, cabin)
        if per is not None and count:
            parts.append(f"free_bag_entitlement={per * count} ({per} per passenger x {count} passengers)")
        return " ".join(parts)

    def segment_line(segment):
        flight_number = getattr(segment, "flight_number", None)
        date = getattr(segment, "date", None)
        text = "{}@{} {}->{}".format(
            flight_number,
            date,
            getattr(segment, "origin", None),
            getattr(segment, "destination", None),
        )
        status, prices = None, None
        try:
            flight = tools.db.flights.get(flight_number) if isinstance(flight_number, str) else None
            dates = getattr(flight, "dates", None) if flight is not None else None
            instance = dates.get(date) if isinstance(dates, dict) and isinstance(date, str) else None
            if instance is not None:
                status = getattr(instance, "status", None)
                prices = getattr(instance, "prices", None)
        except Exception:
            status, prices = None, None
        if isinstance(status, str) and status:
            text += f" status={status}"
        if isinstance(prices, dict):
            rendered = [
                "{}=${}".format(cabin, prices[cabin])
                for cabin in ("basic_economy", "economy", "business")
                if cabin in prices and prices[cabin] is not None
            ]
            if rendered:
                text += " prices: " + " ".join(rendered)
        return text

    try:
        db = getattr(tools, "db", None)
        if db is None:
            return result
        now = tools._get_datetime()
        lines = []
        if tool_name == "get_user_details":
            user_id = kwargs.get("user_id")
            user = db.users.get(user_id) if isinstance(user_id, str) else None
            if user is None:
                return result
            membership = getattr(user, "membership", None)
            table = _FREE.get(membership) if isinstance(membership, str) else None
            head = f"membership={membership}"
            if table is not None:
                head += (
                    " free checked bags per passenger per cabin: basic_economy={} economy={} business={}".format(
                        table["basic_economy"], table["economy"], table["business"]
                    )
                )
            reservation_ids = getattr(user, "reservations", None)
            reservation_ids = reservation_ids if isinstance(reservation_ids, list) else []
            if len(reservation_ids) > 12:
                head += f"; listing the first 12 of {len(reservation_ids)} reservations"
            lines.append(head)
            for reservation_id in reservation_ids[:12]:
                try:
                    res = db.reservations.get(reservation_id) if isinstance(reservation_id, str) else None
                    if res is None:
                        continue
                    cabin = getattr(res, "cabin", None)
                    passengers = getattr(res, "passengers", None)
                    names = (
                        ", ".join(person_name(p) for p in passengers)
                        if isinstance(passengers, list)
                        else "unknown"
                    )
                    segments = getattr(res, "flights", None)
                    first_date = getattr(segments[0], "date", None) if isinstance(segments, list) and segments else None
                    status = getattr(res, "status", None)
                    net = paid_net(res)
                    lines.append(
                        "res {} {}->{} {} cabin={} first_segment_date={} insurance={} status={} "
                        "passengers={} [{}] paid_net={} {} {}".format(
                            reservation_id,
                            getattr(res, "origin", "?"),
                            getattr(res, "destination", "?"),
                            getattr(res, "flight_type", "?"),
                            cabin,
                            first_date if isinstance(first_date, str) and first_date else "unknown",
                            getattr(res, "insurance", "?"),
                            status if isinstance(status, str) and status else "none",
                            len(passengers) if isinstance(passengers, list) else "unknown",
                            names or "unknown",
                            f"${net}" if net is not None else "unknown",
                            baggage_state(membership, cabin, res),
                            grounds(now, res),
                        )
                    )
                except Exception:
                    continue
        else:
            reservation_id = kwargs.get("reservation_id")
            res = db.reservations.get(reservation_id) if isinstance(reservation_id, str) else None
            if res is None:
                return result
            user_id = getattr(res, "user_id", None)
            user = db.users.get(user_id) if isinstance(user_id, str) else None
            membership = getattr(user, "membership", None) if user is not None else None
            cabin = getattr(res, "cabin", None)
            passengers = getattr(res, "passengers", None)
            if isinstance(passengers, list):
                lines.append(
                    "passengers={} [{}]".format(
                        len(passengers),
                        ", ".join(
                            "{} dob={}".format(person_name(p), getattr(p, "dob", "unknown"))
                            for p in passengers
                        ),
                    )
                )
            net = paid_net(res)
            lines.append(f"paid_net={f'${net}' if net is not None else 'unknown'}")
            lines.append("membership={} {}".format(membership, baggage_state(membership, cabin, res)))
            lines.append(grounds(now, res))
            segments = getattr(res, "flights", None)
            segments = segments if isinstance(segments, list) else []
            shown = segments[:8]
            for index, segment in enumerate(shown):
                try:
                    line = segment_line(segment)
                    if index == len(shown) - 1 and len(segments) > len(shown):
                        line += f" (+{len(segments) - len(shown)} more segments)"
                    lines.append(line)
                except Exception:
                    continue
        lines = [line for line in lines if isinstance(line, str) and line]
        if not lines:
            return result
        payload = serialized(result)
        if not isinstance(payload, str):
            return result
        return (
            payload
            + "\n\nDerived facts (current DB):\n"
            + "\n".join(lines)
            + "\nCancellation requires booked within 24h, an airline-cancelled segment, a business cabin, "
            "or travel insurance; insurance alone still requires a covered reason under the published policy."
        )
    except Exception:
        return result


def register():
    if getattr(AirlineTools, "_bag_fee_normalizer", False) and getattr(AirlineTools, "_itinerary_continuity_gate", False) and getattr(AirlineTools, "_reservation_context_digest", False):
        return
    original_use_tool = AirlineTools.use_tool

    def use_tool(self, tool_name, **kwargs):
        _gate(self, tool_name, kwargs)
        _continuity_gate(self, tool_name, kwargs)
        result = original_use_tool(self, tool_name, **kwargs)
        return _context_digest(self, tool_name, kwargs, result)

    AirlineTools.use_tool = use_tool
    AirlineTools._write_policy_gate = True
    AirlineTools._bag_fee_normalizer = True
    AirlineTools._itinerary_continuity_gate = True
    AirlineTools._reservation_context_digest = True


class AgentHarness:
    def __init__(self, public_context=None):
        self.public_context = public_context
    def transform_request(self, request):
        return request
    def transform_response(self, response, public_history):
        return response
