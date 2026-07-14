"""Airline domain harness rules and annotators.

All rules and annotators inspect only DB state (no conversation history),
ensuring replay safety during set_state evaluation.

Policy source: data/tau2/domains/airline/policy.md

H2 Rule index
-------------
book_reservation
  BookReservationPaymentRule      – enforce payment-method limits (≤1 cert, ≤1 CC, ≤3 GC)
  MaxPassengersRule               – at most 5 passengers per reservation
  FlightStatusBookableRule        – all flights must have status "available" on the given date
  BaggageAllowanceRule            – nonfree_baggages must equal total − free-allowance
  BookReservationPaymentTotalRule – sum(payment amounts) must equal sum(flight prices × pax)
                                    gives agent the correct total before the DB write

cancel_reservation
  CancelFlightRule                – already-flown guard + 4-condition eligibility check

update_reservation_flights
  BasicEconomyFlightChangeRule    – block same-cabin changes on basic_economy;
                                    also block upgrade+different-flights (must cancel+rebook)
  NoOriginDestChangeRule          – block changing origin/destination/trip-type;
                                    round_trip: detect missing return legs
  CabinChangeOnFlyingReservationRule – block cabin change when any leg has flown
  FlightUpdateNoCertificateRule   – payment for flight update must be gift_card or credit_card

update_reservation_baggages
  NoBaggageDecreaseRule           – block reducing bag count
  BaggageAllowanceUpdateRule      – nonfree_baggages must equal total − free-allowance
                                    (mirrors BaggageAllowanceRule for the update path)

update_reservation_passengers
  NoPassengerCountChangeRule      – passenger count cannot change (policy + tool enforce this)

send_certificate
  CertificateEligibilityRule      – block for ineligible (regular/no-ins/no-biz)
  CertificateFlightStatusRule     – block when no flight in user's reservations is actually
                                    delayed or cancelled (prevents phantom compensation)
  DelayedCompensationRequiresChangeRule – delayed-flight certificates require a prior
                                    successful change/cancel in this episode
  CertificateAmountCapRule        – amount ≤ $100×max_passengers; must be multiple of $50

H4 Annotator index
------------------
get_user_details
  UserReservationSummaryAnnotator – when user has multiple reservations, list each with
                                    origin→destination, cabin, date, and total paid; helps
                                    agent identify the correct reservation without extra
                                    round-trips.

get_reservation_details
  ReservationPaymentAnnotator     – surface total charged (sum of payment_history positives)
                                    broken down by payment method; reminds agent to communicate
                                    exact amounts. Silent when no payment history.
  ReservationFlightSummaryAnnotator – remind agent of round-trip completeness: flag if the
                                    reservation is round_trip but fewer legs than expected.

cancel_reservation
  CancelRefundAnnotator           – after a successful cancel, compute and highlight the exact
                                    refund amount per payment instrument so agent communicates
                                    it verbatim.

update_reservation_flights
  FlightUpdatePaymentAnnotator    – after a successful flight update, surface the net payment
                                    or refund amount so the agent communicates it.

book_reservation
  BookingPaymentAnnotator         – after a successful booking, surface total charged per
                                    instrument so agent can confirm the exact amount with user.
"""

from datetime import timedelta
from typing import Any

from tau2.domains.airline.data_model import FlightDB
from tau2.domains.airline.tools import AirlineTools
from tau2.domains.airline.utils import SIMULATION_TIME
from tau2.harness.base import HarnessedToolKitMixin
from tau2.harness.h3_tools import H3AirlineToolDescriptionMixin


class BookReservationPaymentRule:
    """Enforce per-reservation payment-method limits for book_reservation.

    Policy: "Each reservation can use at most one travel certificate, at most
    one credit card, and at most three gift cards."

    The tool itself does not check these counts; the harness enforces them
    before any DB write occurs.
    """

    tool_name = "book_reservation"

    def check(
        self,
        db: FlightDB,
        user_id: str,
        payment_methods: list,
        **_: Any,
    ) -> None:
        user = db.users.get(user_id)
        if user is None:
            return

        certs = cards = gifts = 0
        for pm in payment_methods:
            pm_id = (
                pm.get("payment_id") if isinstance(pm, dict) else getattr(pm, "payment_id", None)
            )
            if pm_id is None:
                continue
            pm_data = user.payment_methods.get(pm_id)
            if pm_data is None:
                continue
            src = getattr(pm_data, "source", None)
            if src == "certificate":
                certs += 1
            elif src == "credit_card":
                cards += 1
            elif src == "gift_card":
                gifts += 1

        seen_payment_ids = [
            pm.get("payment_id") if isinstance(pm, dict) else getattr(pm, "payment_id", None)
            for pm in payment_methods
        ]
        duplicates = sorted(
            {
                pm_id
                for pm_id in seen_payment_ids
                if pm_id and seen_payment_ids.count(pm_id) > 1
            }
        )
        if duplicates:
            raise ValueError(
                "Cannot use the same payment method more than once in one "
                f"reservation: {duplicates}. Combine each payment_id into a "
                "single payment entry with the desired amount."
            )

        if certs > 1:
            raise ValueError(
                f"Cannot use more than 1 travel certificate per reservation "
                f"({certs} provided). Per policy, each reservation allows at most "
                "one travel certificate. "
                "To use multiple certificates, make separate reservations each "
                "paying with 1 certificate."
            )
        if cards > 1:
            raise ValueError(
                f"Cannot use more than 1 credit card per reservation "
                f"({cards} provided). Per policy, each reservation allows at most "
                "one credit card."
            )
        if gifts > 3:
            raise ValueError(
                f"Cannot use more than 3 gift cards per reservation "
                f"({gifts} provided). Per policy, each reservation allows at most "
                "three gift cards."
            )


class BookReservationItineraryRule:
    """Pre-execution itinerary-shape check for book_reservation.

    The tool validates flight existence and availability, but without this guard
    a syntactically valid list can still be a disconnected route (for example
    JFK->SEA followed by SFO->SEA). This is a deterministic interface error:
    every adjacent leg must connect, and the first/last endpoints must match
    the requested trip type.
    """

    tool_name = "book_reservation"

    def check(
        self,
        db: FlightDB,
        origin: str = "",
        destination: str = "",
        flight_type: str = "",
        flights: list | None = None,
        **_: Any,
    ) -> None:
        flights = flights or []
        if not origin or not destination or not flights:
            return

        route = []
        for fi in flights:
            fn = (
                fi.get("flight_number")
                if isinstance(fi, dict)
                else getattr(fi, "flight_number", None)
            )
            if not fn:
                return
            flight = db.flights.get(fn)
            if flight is None:
                return
            route.append((fn, flight.origin, flight.destination))

        first_fn, first_origin, _ = route[0]
        if first_origin != origin:
            raise ValueError(
                f"Invalid itinerary: first flight {first_fn} departs from "
                f"{first_origin}, but booking origin is {origin}."
            )

        for (prev_fn, _, prev_dest), (next_fn, next_origin, _) in zip(route, route[1:]):
            if prev_dest != next_origin:
                raise ValueError(
                    f"Invalid itinerary: flight {prev_fn} arrives at {prev_dest}, "
                    f"but the next flight {next_fn} departs from {next_origin}. "
                    "Adjacent flight legs must connect in order."
                )

        final_dest = route[-1][2]
        if flight_type == "one_way" and final_dest != destination:
            raise ValueError(
                f"Invalid one_way itinerary: final destination is {final_dest}, "
                f"but booking destination is {destination}."
            )

        if flight_type == "round_trip":
            visits_destination = any(dest == destination for _, _, dest in route)
            if not visits_destination:
                raise ValueError(
                    f"Invalid round_trip itinerary: no leg reaches booking "
                    f"destination {destination}."
                )
            if final_dest != origin:
                raise ValueError(
                    f"Invalid round_trip itinerary: final destination is {final_dest}, "
                    f"but a round trip from {origin} to {destination} must end at {origin}."
                )


class CancelFlightRule:
    """Pre-execution eligibility check for cancel_reservation.

    Two sequential checks (both from policy.md "Cancel flight" section):

    1. Already-flown guard: if any flight leg has status "flying" or "landed"
       the agent must transfer to a human agent rather than cancel directly.

    2. Eligibility guard: at least one of the four policy conditions must hold:
       a) booking created within the last 24 hours
       b) airline cancelled a flight in the reservation
       c) business cabin
       d) user purchased travel insurance
    """

    tool_name = "cancel_reservation"

    def check(self, db: FlightDB, reservation_id: str, **_: Any) -> None:
        r = db.reservations.get(reservation_id)
        if r is None:
            return  # let the tool raise its own "not found" error

        # --- Check 1: already-flown guard ---
        for rf in r.flights:
            flight = db.flights.get(rf.flight_number)
            if flight is None:
                continue
            date_status = flight.dates.get(rf.date)
            if date_status is None:
                continue
            if date_status.status in {"flying", "landed"}:
                raise ValueError(
                    f"Flight {rf.flight_number} on {rf.date} has already "
                    f"departed or landed (status: '{date_status.status}'). "
                    "Cannot cancel directly — please call transfer_to_human_agents."
                )

        # --- Check 2: cancellation eligibility ---
        eligible_reasons: list[str] = []

        # (a) booking within last 24 hours
        try:
            from datetime import datetime

            created_at = datetime.fromisoformat(r.created_at)
            if (SIMULATION_TIME - created_at) <= timedelta(hours=24):
                eligible_reasons.append("booking was made within the last 24 hours")
        except (ValueError, TypeError):
            pass

        # (b) airline cancelled a flight leg
        for rf in r.flights:
            flight = db.flights.get(rf.flight_number)
            if flight is None:
                continue
            date_status = flight.dates.get(rf.date)
            if date_status is not None and date_status.status == "cancelled":
                eligible_reasons.append("the airline cancelled a flight in this reservation")
                break

        # (c) business cabin
        if r.cabin == "business":
            eligible_reasons.append("business cabin")

        # (d) travel insurance purchased
        if r.insurance == "yes":
            eligible_reasons.append("travel insurance was purchased")

        if not eligible_reasons:
            # Build per-condition diagnostic to help agent reason faster
            try:
                from datetime import datetime

                created_at = datetime.fromisoformat(r.created_at)
                hrs_ago = (SIMULATION_TIME - created_at).total_seconds() / 3600
                cond1_status = f"NOT MET (booked {hrs_ago:.0f}h ago)"
            except (ValueError, TypeError):
                cond1_status = "NOT MET"

            cond2_status = "NOT MET"
            for rf in r.flights:
                flight = db.flights.get(rf.flight_number)
                if flight and flight.dates.get(rf.date, None) and flight.dates[rf.date].status == "cancelled":
                    cond2_status = "MET"
                    break

            cond3_status = f"NOT MET (cabin is '{r.cabin}')"
            cabin_hint = ""
            if r.cabin == "basic_economy":
                cabin_hint = (
                    "\nHint: This reservation is basic_economy. "
                    "You may first upgrade it to business (update_reservation_flights "
                    "with cabin='business'), which would make it eligible for "
                    "cancellation under the business-cabin condition."
                )

            cond4_status = f"NOT MET (insurance='{r.insurance}')"

            raise ValueError(
                f"Cancellation not allowed for reservation {reservation_id}: "
                "none of the required conditions are met.\n"
                "Per policy, cancellation requires at least one of the following:\n"
                f"1) Booked within last 24h — {cond1_status}\n"
                f"2) Airline cancelled a flight — {cond2_status}\n"
                f"3) Business cabin — {cond3_status}\n"
                f"4) Travel insurance purchased — {cond4_status}\n"
                "Please check other reservations or inform the customer why cancellation "
                f"is not available for this reservation.{cabin_hint}"
            )


class BasicEconomyFlightChangeRule:
    """Block update_reservation_flights when changing flights within basic economy.

    Policy: "basic economy is its own class … [it] does not allow flight
    changes."  Two cases are blocked:

    1. Same-cabin change: reservation is basic_economy AND requested cabin is
       also basic_economy (or unspecified) — flight changes not allowed.

    2. Upgrade-with-new-flights: reservation is basic_economy AND cabin is
       being upgraded (economy/business) AND the submitted flight numbers
       differ from the current ones.  In this case the correct flow is
       cancel → rebook at the new cabin; upgrading cabin while simultaneously
       switching to different flights is not a valid operation.

    The one allowed path for basic_economy: upgrade cabin while keeping the
    EXACT SAME flight numbers (e.g. cabin basic→business, same flight numbers).
    """

    tool_name = "update_reservation_flights"

    def check(
        self,
        db: FlightDB,
        reservation_id: str,
        flights: list | None = None,
        cabin: str = "",
        **_: Any,
    ) -> None:
        r = db.reservations.get(reservation_id)
        if r is None:
            return
        if r.cabin != "basic_economy":
            return

        is_upgrade = cabin and cabin != "basic_economy"

        if not is_upgrade:
            # Case 1: same-cabin change — always blocked
            raise ValueError(
                "Basic economy reservations do not allow flight changes within the same cabin. "
                "To change flights, the passenger must first upgrade to economy or business. "
                "Please inform the customer of this basic economy restriction."
            )

        # Case 2: upgrade requested — only allowed when keeping the SAME flight numbers
        if flights:
            current_fns = {rf.flight_number for rf in r.flights}
            new_fns: set[str] = set()
            for fi in flights:
                fn = (
                    fi.get("flight_number")
                    if isinstance(fi, dict)
                    else getattr(fi, "flight_number", None)
                )
                if fn:
                    new_fns.add(fn)
            if new_fns and not new_fns.issubset(current_fns):
                raise ValueError(
                    f"Cannot upgrade reservation {reservation_id} from basic_economy "
                    f"to '{cabin}' while simultaneously switching to different flights "
                    f"({sorted(new_fns - current_fns)}).\n"
                    "Per policy, for a basic_economy reservation you may either:\n"
                    "  a) Upgrade cabin keeping the SAME flights "
                    "(update_reservation_flights with same flight numbers), or\n"
                    "  b) Cancel this reservation and book a new one at the desired cabin.\n"
                    "Please cancel reservation first (cancel_reservation), then book a new "
                    "reservation with the requested flights."
                )


class CertificateEligibilityRule:
    """Block send_certificate when user is clearly ineligible for compensation.

    Policy: "Only compensate if the user is a silver/gold member or has travel
    insurance or flies business."

    Because send_certificate only receives user_id (not reservation_id), this
    rule takes a conservative approach: it checks whether the user has at least
    one reservation that could justify compensation (business cabin or
    insurance).  If the user is regular-tier AND no reservation meets the
    criteria, the certificate is blocked.
    """

    tool_name = "send_certificate"

    def check(self, db: FlightDB, user_id: str, **_: Any) -> None:
        user = db.users.get(user_id)
        if user is None:
            return

        # Silver/gold membership always qualifies
        if user.membership in {"silver", "gold"}:
            return

        # Regular member: check if any reservation has business cabin or insurance
        for res_id in user.reservations:
            r = db.reservations.get(res_id)
            if r is None:
                continue
            if r.cabin == "business" or r.insurance == "yes":
                return  # at least one reservation could justify compensation

        raise ValueError(
            "User does not qualify for compensation; certificate denied.\n"
            "Per policy, compensation requires at least one of the following:\n"
            "- Silver or gold membership\n"
            "- Travel insurance on a reservation\n"
            "- Business cabin on a reservation\n"
            "Please explain to the customer why they do not qualify."
        )


class NoOriginDestChangeRule:
    """Block update_reservation_flights when the new itinerary changes the
    reservation's origin or destination.

    Policy: "Other reservations can be modified without changing the origin,
    destination, and trip type."

    The check compares the first new flight's origin against
    reservation.origin, and the last new flight's destination against the
    expected endpoint:
      - one_way:   last destination == reservation.destination
      - round_trip: last destination == reservation.origin  (trip ends where it began)
    """

    tool_name = "update_reservation_flights"

    def check(
        self,
        db: FlightDB,
        reservation_id: str,
        flights: list,
        **_: Any,
    ) -> None:
        r = db.reservations.get(reservation_id)
        if r is None or not flights:
            return

        def get_fn(fi: Any) -> str | None:
            return (
                fi.get("flight_number")
                if isinstance(fi, dict)
                else getattr(fi, "flight_number", None)
            )

        first_fn = get_fn(flights[0])
        last_fn = get_fn(flights[-1])

        first_flight = db.flights.get(first_fn) if first_fn else None
        last_flight = db.flights.get(last_fn) if last_fn else None

        if first_flight is None or last_flight is None:
            return

        # First leg must depart from reservation origin
        if first_flight.origin != r.origin:
            raise ValueError(
                f"The new itinerary departs from {first_flight.origin}, "
                f"but the reservation origin is {r.origin}. "
                "Per policy, the origin, destination, and trip type cannot be changed."
            )

        route = []
        for fi in flights:
            fn = get_fn(fi)
            flight = db.flights.get(fn) if fn else None
            if flight is None:
                return
            route.append((fn, flight.origin, flight.destination))

        for (prev_fn, _, prev_dest), (next_fn, next_origin, _) in zip(route, route[1:]):
            if prev_dest != next_origin:
                raise ValueError(
                    f"Invalid itinerary for reservation {reservation_id}: flight "
                    f"{prev_fn} arrives at {prev_dest}, but the next flight {next_fn} "
                    f"departs from {next_origin}. Adjacent flight legs must connect "
                    "in order; do not include stale legs from the previous itinerary."
                )

        if r.flight_type == "round_trip" and not any(
            dest == r.destination for _, _, dest in route
        ):
            raise ValueError(
                f"Invalid round_trip itinerary for reservation {reservation_id}: "
                f"no submitted leg reaches destination {r.destination}. Include the "
                "outbound leg(s) to the destination and the return leg(s) back to "
                f"{r.origin}."
            )

        # Last leg must arrive at the expected endpoint
        # one_way → reservation.destination; round_trip → reservation.origin
        expected_last_dest = (
            r.destination if r.flight_type == "one_way" else r.origin
        )
        if last_flight.destination != expected_last_dest:
            # Special case: round_trip where submitted flights end at r.destination
            # (not r.origin). This means the agent only submitted the outbound legs
            # and forgot the return segments — give a targeted hint.
            if (
                r.flight_type == "round_trip"
                and last_flight.destination == r.destination
            ):
                raise ValueError(
                    f"Reservation {reservation_id} is a round_trip "
                    f"({r.origin} → {r.destination} → {r.origin}), but the submitted "
                    f"flights only cover the outbound leg (ending at {r.destination}). "
                    "You must include BOTH the outbound and return flight segments in "
                    "the same request. If the user only asked to change the outbound "
                    "leg, keep the existing return flight segment(s) from the current "
                    "reservation unchanged and append them to the flights list. "
                    "Only search for new return flights when the user explicitly wants "
                    f"to change the return leg ({r.destination} → {r.origin})."
                )
            raise ValueError(
                f"The new itinerary ends at {last_flight.destination}, "
                f"but the expected endpoint is {expected_last_dest}. "
                "Per policy, the origin, destination, and trip type cannot be changed."
            )


class CabinChangeOnFlyingReservationRule:
    """Block update_reservation_flights when changing cabin on a reservation
    where at least one flight has already departed.

    Policy: "Cabin cannot be changed if any flight in the reservation has
    already been flown."
    """

    tool_name = "update_reservation_flights"

    def check(
        self,
        db: FlightDB,
        reservation_id: str,
        cabin: str = "",
        **_: Any,
    ) -> None:
        r = db.reservations.get(reservation_id)
        if r is None:
            return

        # Only applies when the cabin is actually being changed
        if not cabin or cabin == r.cabin:
            return

        for rf in r.flights:
            flight = db.flights.get(rf.flight_number)
            if flight is None:
                continue
            date_status = flight.dates.get(rf.date)
            if date_status is None:
                continue
            if date_status.status in {"flying", "landed"}:
                raise ValueError(
                    f"Flight {rf.flight_number} on {rf.date} in reservation "
                    f"{reservation_id} has status '{date_status.status}'. "
                    "Per policy, cabin class cannot be changed once any flight "
                    "in the reservation has already been flown."
                )


class NoBaggageDecreaseRule:
    """Block update_reservation_baggages when the new total_baggages is less
    than the current total.

    Policy: "The user can add but not remove checked bags."
    """

    tool_name = "update_reservation_baggages"

    def check(
        self,
        db: FlightDB,
        reservation_id: str,
        total_baggages: int,
        **_: Any,
    ) -> None:
        r = db.reservations.get(reservation_id)
        if r is None:
            return
        if total_baggages < r.total_baggages:
            raise ValueError(
                f"Cannot reduce checked bags from {r.total_baggages} to {total_baggages}. "
                "Per policy, checked bags can only be added, not removed. "
                "Please inform the customer of this restriction."
            )


class MaxPassengersRule:
    """Enforce the 5-passenger limit per reservation for book_reservation.

    Policy: "Each reservation can have at most five passengers."

    The underlying tool does not validate this count; the harness prevents
    the DB write before seat-availability checks reduce available seats.
    """

    tool_name = "book_reservation"

    def check(self, db: FlightDB, passengers: list, **_: Any) -> None:
        if len(passengers) > 5:
            raise ValueError(
                f"Cannot book a reservation for {len(passengers)} passengers. "
                "Per policy, each reservation allows at most 5 passengers. "
                "Please split the booking into separate reservations if needed."
            )


class FlightStatusBookableRule:
    """Block book_reservation when any requested flight is not 'available'.

    Policy: "If the status is delayed or on time, the flight has not taken off,
    cannot be booked.  If the status is flying, cannot be booked."

    Only flights with status 'available' may be booked.  This check runs before
    the tool writes to the DB, giving the agent an early, actionable error.
    """

    tool_name = "book_reservation"

    def check(self, db: FlightDB, flights: list, **_: Any) -> None:
        for fi in flights:
            fn = fi.get("flight_number") if isinstance(fi, dict) else getattr(fi, "flight_number", None)
            date = fi.get("date") if isinstance(fi, dict) else getattr(fi, "date", None)
            if not fn or not date:
                continue
            flight = db.flights.get(fn)
            if flight is None:
                continue
            date_status = flight.dates.get(date)
            if date_status is None:
                continue
            status = getattr(date_status, "status", None)
            if status and status != "available":
                raise ValueError(
                    f"Flight {fn} on {date} has status '{status}' and cannot be booked. "
                    "Per policy, only flights with status 'available' can be booked. "
                    f"Flights with status 'delayed', 'on time', 'flying', or 'cancelled' "
                    "are not bookable. Please search for an alternative flight."
                )


class BaggageAllowanceRule:
    """Ensure nonfree_baggages equals total_baggages minus the free allowance.

    Policy (free checked bag allowance by membership × cabin):
      regular: basic_economy=0, economy=1, business=2 per passenger
      silver:  basic_economy=1, economy=2, business=3 per passenger
      gold:    basic_economy=2, economy=3, business=4 per passenger
    Extra bags are $50 each.

    The agent must compute nonfree_baggages correctly; this rule catches
    miscalculations before the DB is written.
    """

    tool_name = "book_reservation"

    _FREE_BAGS: dict[str, dict[str, int]] = {
        "regular": {"basic_economy": 0, "economy": 1, "business": 2},
        "silver":  {"basic_economy": 1, "economy": 2, "business": 3},
        "gold":    {"basic_economy": 2, "economy": 3, "business": 4},
    }

    def check(
        self,
        db: FlightDB,
        user_id: str,
        cabin: str,
        passengers: list,
        total_baggages: int,
        nonfree_baggages: int,
        **_: Any,
    ) -> None:
        user = db.users.get(user_id)
        if user is None:
            return
        membership = getattr(user, "membership", "regular")
        free_per_pax = self._FREE_BAGS.get(membership, {}).get(cabin, 0)
        num_pax = len(passengers)
        free_total = free_per_pax * num_pax
        expected_nonfree = max(0, total_baggages - free_total)
        if nonfree_baggages != expected_nonfree:
            raise ValueError(
                f"nonfree_baggages is {nonfree_baggages}, but should be {expected_nonfree}. "
                f"User '{user_id}' is a {membership} member booking {cabin} for {num_pax} "
                f"passenger(s), which entitles them to {free_total} free checked bag(s). "
                f"With total_baggages={total_baggages}, the correct nonfree_baggages is "
                f"{expected_nonfree} (each extra bag costs $50). "
                "Please recalculate and correct nonfree_baggages."
            )


class BookReservationPaymentTotalRule:
    """Pre-validate that the sum of payment amounts equals the actual flight total.

    Addresses the common failure where the agent submits payment_methods whose
    amounts do not add up to the real ticket cost (e.g. uses a certificate's
    face value instead of the actual charge needed, or forgets to scale by
    number of passengers).

    The per-flight price is read from db.flights[fn].dates[date].prices[cabin].
    Only flights with status 'available' carry a prices map; for any other
    status FlightStatusBookableRule will already have fired.  Flights absent
    from the DB are skipped (the tool itself will raise a suitable error).

    Formula: expected_total = sum(prices[cabin] for each flight) × num_passengers
    + insurance ($30/passenger if selected) + baggage fees ($50 × nonfree_baggages)

    When a mismatch is detected the error message also computes the optimal
    per-instrument breakdown (certificates and gift cards first, remainder to
    credit card) so the agent can correct immediately without arithmetic.
    """

    tool_name = "book_reservation"

    def check(
        self,
        db: FlightDB,
        user_id: str = "",
        flights: list | None = None,
        cabin: str = "",
        passengers: list | None = None,
        payment_methods: list | None = None,
        insurance: str = "",
        nonfree_baggages: int = 0,
        **_: Any,
    ) -> None:
        flights = flights or []
        passengers = passengers or []
        payment_methods = payment_methods or []

        num_pax = len(passengers)
        expected_total = 0
        missing_price = False

        for fi in flights:
            fn = fi.get("flight_number") if isinstance(fi, dict) else getattr(fi, "flight_number", None)
            date = fi.get("date") if isinstance(fi, dict) else getattr(fi, "date", None)
            if not fn or not date:
                missing_price = True
                continue
            flight = db.flights.get(fn)
            if flight is None:
                missing_price = True
                continue
            date_status = flight.dates.get(date)
            if date_status is None:
                missing_price = True
                continue
            prices = getattr(date_status, "prices", None)
            if prices is None:
                missing_price = True
                continue
            price = prices.get(cabin)
            if price is None:
                missing_price = True
                continue
            expected_total += price

        if missing_price or expected_total == 0:
            return

        expected_total *= num_pax
        if insurance == "yes":
            expected_total += 30 * num_pax
        expected_total += 50 * nonfree_baggages
        paid_total = sum(
            (pm.get("amount") if isinstance(pm, dict) else getattr(pm, "amount", 0))
            for pm in payment_methods
        )

        if paid_total == expected_total:
            return

        # --- Build actionable breakdown ---
        user = db.users.get(user_id) if user_id else None

        # Separate fixed-balance instruments (cert/GC) from credit card
        fixed: list[tuple[str, str, int, int]] = []   # (id, source, submitted, balance)
        cc_list: list[tuple[str, int]] = []            # (id, submitted)

        for pm in payment_methods:
            pm_id = pm.get("payment_id") if isinstance(pm, dict) else getattr(pm, "payment_id", "")
            amount = int(pm.get("amount") if isinstance(pm, dict) else getattr(pm, "amount", 0))
            user_pm = user.payment_methods.get(pm_id) if user and pm_id else None
            src = getattr(user_pm, "source", None) if user_pm else None
            if src is None:
                if pm_id.startswith("certificate"):
                    src = "certificate"
                elif pm_id.startswith("gift_card"):
                    src = "gift_card"
                elif pm_id.startswith("credit_card"):
                    src = "credit_card"
            balance = int(getattr(user_pm, "amount", amount) if user_pm else amount)
            if src in ("certificate", "gift_card"):
                fixed.append((pm_id, src, amount, balance))
            else:
                cc_list.append((pm_id, amount))

        # Greedily assign fixed instruments up to expected_total
        remaining = expected_total
        fixed_lines: list[str] = []
        for pm_id, src, submitted, balance in fixed:
            correct = min(balance, remaining)
            remaining -= correct
            flag = " ✓" if correct == submitted else f" ← change to ${correct}"
            fixed_lines.append(f"    {src} {pm_id}: ${submitted}{flag}")

        # Remainder goes to credit card
        cc_correct = remaining  # may be 0 or negative (over-charged via fixed)
        cc_lines: list[str] = []
        for pm_id, submitted in cc_list:
            flag = " ✓" if submitted == cc_correct else f" ← change to ${cc_correct}"
            cc_lines.append(f"    credit_card {pm_id}: ${submitted}{flag}")
            cc_correct = 0  # only first CC gets the remainder; subsequent should be 0

        if remaining > 0 and not cc_list:
            user_cards = []
            if user:
                user_cards = [
                    pm_id
                    for pm_id, pm_data in user.payment_methods.items()
                    if getattr(pm_data, "source", None) == "credit_card"
                ]
            card_hint = (
                f" Add a credit_card payment entry for ${remaining}"
                + (f" (for example {user_cards[0]})." if user_cards else ".")
            )
            cc_lines.append(f"    credit_card: missing ${remaining} ←{card_hint}")

        breakdown = "\n".join(fixed_lines + cc_lines) or "    (no payment methods provided)"

        selected_fare = expected_total // num_pax if num_pax else expected_total
        raise ValueError(
            f"Payment total mismatch: submitted ${paid_total} ≠ actual total cost "
            f"${expected_total} "
            f"(selected {cabin} itinerary costs ${selected_fare}/passenger × "
            f"{num_pax} passenger(s)).\n"
            "Correct allocation — apply each certificate/gift card first, credit card gets the rest:\n"
            f"{breakdown}\n"
            "Rule: sum of ALL payment amounts must equal the ticket total exactly. "
            "Use only the amount needed from each instrument, not its full balance."
        )


class FlightUpdateNoCertificateRule:
    """Block update_reservation_flights when a travel certificate is used as payment.

    Policy: "If the flights are changed, the user needs to provide a single
    gift card or credit card for payment or refund method."

    Certificates cannot be used to pay for flight changes.  The tool already
    raises an error, but this H2 rule gives an earlier, more actionable message.
    """

    tool_name = "update_reservation_flights"

    def check(self, db: FlightDB, payment_id: str = "", **_: Any) -> None:
        if payment_id and payment_id.startswith("certificate"):
            raise ValueError(
                f"Payment method '{payment_id}' is a travel certificate, which cannot "
                "be used to pay for flight changes. "
                "Per policy, flight change payments must use a gift card or credit card. "
                "Please provide a valid gift card or credit card payment ID."
            )


class FlightUpdatePaymentCapacityRule:
    """Pre-validate payment capacity for update_reservation_flights.

    The base tool reports only "Gift card balance is not enough".  That is too
    little guidance for tasks where the user asks for the smallest usable gift
    card: the agent often retries the smallest balance even when it cannot cover
    the net charge.  This rule computes the exact net payment before execution
    and gives the deterministic next payment choice.
    """

    tool_name = "update_reservation_flights"

    def check(
        self,
        db: FlightDB,
        reservation_id: str,
        flights: list | None = None,
        cabin: str = "",
        payment_id: str = "",
        **_: Any,
    ) -> None:
        r = db.reservations.get(reservation_id)
        if r is None or not flights or not cabin or not payment_id:
            return
        user = db.users.get(r.user_id)
        if user is None:
            return

        payment_method = user.payment_methods.get(payment_id)
        if payment_method is None:
            return

        num_pax = len(r.passengers)
        new_total = 0
        missing_price = False

        for fi in flights:
            fn = (
                fi.get("flight_number")
                if isinstance(fi, dict)
                else getattr(fi, "flight_number", None)
            )
            date = fi.get("date") if isinstance(fi, dict) else getattr(fi, "date", None)
            if not fn or not date:
                missing_price = True
                continue

            # The base tool keeps an existing segment's historical price only
            # when both flight/date and cabin are unchanged.
            matching_current = next(
                (
                    rf
                    for rf in r.flights
                    if rf.flight_number == fn
                    and rf.date == date
                    and cabin == r.cabin
                ),
                None,
            )
            if matching_current is not None:
                new_total += matching_current.price * num_pax
                continue

            flight = db.flights.get(fn)
            if flight is None:
                missing_price = True
                continue
            date_status = flight.dates.get(date)
            if date_status is None:
                missing_price = True
                continue
            prices = getattr(date_status, "prices", None)
            if prices is None or cabin not in prices:
                missing_price = True
                continue
            new_total += prices[cabin] * num_pax

        if missing_price:
            return

        old_total = sum(rf.price for rf in r.flights) * num_pax
        net_charge = new_total - old_total
        if net_charge <= 0:
            return

        if getattr(payment_method, "source", None) != "gift_card":
            return

        balance = int(getattr(payment_method, "amount", 0))
        if balance >= net_charge:
            return

        gift_cards = sorted(
            (
                (pm_id, int(getattr(pm_data, "amount", 0)))
                for pm_id, pm_data in user.payment_methods.items()
                if getattr(pm_data, "source", None) == "gift_card"
            ),
            key=lambda item: item[1],
        )
        sufficient_gift_cards = [
            (pm_id, amount) for pm_id, amount in gift_cards if amount >= net_charge
        ]
        credit_cards = [
            pm_id
            for pm_id, pm_data in user.payment_methods.items()
            if getattr(pm_data, "source", None) == "credit_card"
        ]

        if sufficient_gift_cards:
            suggested = (
                f"Use the smallest gift card that can cover the net charge: "
                f"{sufficient_gift_cards[0][0]} (${sufficient_gift_cards[0][1]} balance)."
            )
        elif credit_cards:
            suggested = (
                "No gift card can cover this net charge; use a credit card instead, "
                f"for example {credit_cards[0]}."
            )
        else:
            suggested = "No available gift card or credit card can cover this net charge."

        balances = ", ".join(f"{pm_id}=${amount}" for pm_id, amount in gift_cards)
        raise ValueError(
            f"Gift card {payment_id} has balance ${balance}, but this flight update "
            f"requires a net payment of ${net_charge} (new fare ${new_total} - "
            f"current fare ${old_total}). {suggested} Gift card balances: {balances}. "
            "Do not choose the smallest balance overall unless it can cover the full "
            "positive net charge."
        )


class BaggageAllowanceUpdateRule:
    """Ensure nonfree_baggages equals total_baggages minus the free allowance
    for update_reservation_baggages.

    Mirrors BaggageAllowanceRule (which covers book_reservation) for the
    update path.  The membership and cabin are read from the existing
    reservation; the number of passengers is len(reservation.passengers).

    Free allowance (same table as BaggageAllowanceRule):
      regular: basic_economy=0, economy=1, business=2 per passenger
      silver:  basic_economy=1, economy=2, business=3 per passenger
      gold:    basic_economy=2, economy=3, business=4 per passenger
    """

    tool_name = "update_reservation_baggages"

    _FREE_BAGS: dict[str, dict[str, int]] = BaggageAllowanceRule._FREE_BAGS

    def check(
        self,
        db: FlightDB,
        reservation_id: str,
        total_baggages: int,
        nonfree_baggages: int,
        **_: Any,
    ) -> None:
        r = db.reservations.get(reservation_id)
        if r is None:
            return
        user = db.users.get(r.user_id)
        if user is None:
            return
        membership = getattr(user, "membership", "regular")
        free_per_pax = self._FREE_BAGS.get(membership, {}).get(r.cabin, 0)
        num_pax = len(r.passengers)
        free_total = free_per_pax * num_pax
        expected_nonfree = max(0, total_baggages - free_total)
        if nonfree_baggages != expected_nonfree:
            raise ValueError(
                f"nonfree_baggages is {nonfree_baggages}, but should be {expected_nonfree}. "
                f"Reservation {reservation_id}: user '{r.user_id}' is a {membership} member, "
                f"{r.cabin} cabin, {num_pax} passenger(s) → {free_total} free checked bag(s) total. "
                f"With total_baggages={total_baggages}, "
                f"nonfree_baggages = max(0, {total_baggages} − {free_total}) = {expected_nonfree}. "
                "Please correct nonfree_baggages (each extra bag costs $50)."
            )


class NoPassengerCountChangeRule:
    """Block update_reservation_passengers when the submitted count differs.

    Policy: "The user can modify passengers but cannot modify the number of
    passengers.  Even a human agent cannot modify the number of passengers."

    The tool raises a generic ValueError; this rule raises a clearer message
    before the call reaches the tool.
    """

    tool_name = "update_reservation_passengers"

    def check(
        self,
        db: FlightDB,
        reservation_id: str,
        passengers: list,
        **_: Any,
    ) -> None:
        r = db.reservations.get(reservation_id)
        if r is None:
            return
        if len(passengers) != len(r.passengers):
            raise ValueError(
                f"Cannot change the number of passengers in reservation {reservation_id}. "
                f"Current passenger count: {len(r.passengers)}; "
                f"submitted: {len(passengers)}. "
                "Per policy, passenger count is fixed at booking time and cannot be "
                "modified — not even by a human agent. "
                "Please submit exactly the same number of passengers with updated details."
            )


class CertificateFlightStatusRule:
    """Block send_certificate when no flight in the user's active reservations
    is actually delayed or cancelled by the airline.

    Policy: compensation certificates are issued only when the airline has
    delayed or cancelled a flight in the customer's reservation.  If every
    flight across all active reservations has status 'available', 'on time',
    'flying', or 'landed', there is nothing to compensate and the certificate
    should not be issued.

    Only active (non-cancelled) reservations are checked; a cancelled
    reservation itself cannot justify a new certificate issuance since the
    refund path handles that.

    This rule fires *after* CertificateEligibilityRule has already confirmed
    the user is eligible in principle, adding a second guard: "is there an
    actual event to compensate?"
    """

    tool_name = "send_certificate"

    _COMPENSABLE_STATUSES = {"delayed", "cancelled"}

    def check(self, db: FlightDB, user_id: str, **_: Any) -> None:
        user = db.users.get(user_id)
        if user is None:
            return  # unknown user — let the tool raise its own error

        for res_id in user.reservations:
            r = db.reservations.get(res_id)
            if r is None:
                continue
            # Skip reservations that were already cancelled by the user
            if getattr(r, "status", None) == "cancelled":
                continue
            for rf in r.flights:
                flight = db.flights.get(rf.flight_number)
                if flight is None:
                    continue
                date_status = flight.dates.get(rf.date)
                if date_status is None:
                    continue
                if getattr(date_status, "status", "") in self._COMPENSABLE_STATUSES:
                    return  # found at least one delayed/cancelled flight → allow

        raise ValueError(
            "No delayed or cancelled flights found in any of this user's active reservations. "
            "Per policy, compensation certificates are only issued when the airline has "
            "actually delayed or cancelled a flight in the customer's reservation. "
            "Please verify the flight status with get_flight_status before issuing compensation. "
            "If all flights are operating normally, the customer is not eligible for a certificate."
        )


class CertificateAmountCapRule:
    """Cap the compensation certificate amount at policy-defined limits.

    Policy:
      - Cancelled flight: $100 × number of passengers in the affected reservation
      - Delayed flight:   $50  × number of passengers in the affected reservation
      - Maximum passengers per reservation: 5
      - Therefore hard cap: $500 per certificate send

    Additional constraint: the amount must be a positive multiple of $50,
    since all valid amounts are n × $50 or n × $100 where n ∈ {1..5}.

    Because send_certificate does not receive a reservation_id, this rule
    uses the user's reservations to determine the maximum eligible amount.
    """

    tool_name = "send_certificate"

    def check(self, db: FlightDB, user_id: str, amount: int, **_: Any) -> None:
        # Must be a positive multiple of 50
        if amount <= 0 or amount % 50 != 0:
            raise ValueError(
                f"Certificate amount ${amount} is invalid. "
                "Per policy, compensation amounts are $50 × passengers (for delays) "
                "or $100 × passengers (for cancellations), so the amount must be "
                "a positive multiple of $50. Please correct the amount."
            )

        # Hard cap: $100 × max passengers across user's reservations (≤ 5)
        user = db.users.get(user_id)
        if user is None:
            return
        max_pax = max(
            (len(db.reservations[rid].passengers)
             for rid in user.reservations
             if rid in db.reservations),
            default=1,
        )
        max_allowed = 100 * max_pax  # worst-case: all passengers, cancelled flight
        if amount > max_allowed:
            raise ValueError(
                f"Certificate amount ${amount} exceeds the maximum allowed "
                f"${max_allowed} (= $100 × {max_pax} passengers). "
                "Per policy, compensation is at most $100 per passenger for cancelled "
                "flights or $50 per passenger for delayed flights. "
                f"Please use at most ${max_allowed}."
            )


class DelayedCompensationRequiresChangeRule:
    """Block certificates for delay-only complaints before a change/cancel.

    Policy: for delayed flights, compensation is only available when the user
    wants to change or cancel the reservation, and only after that write has
    completed. Cancelled-flight complaints may receive a gesture certificate
    after fact confirmation without requiring an additional write.
    """

    tool_name = "send_certificate"

    def check(self, db: FlightDB, user_id: str, toolkit: Any = None, **_: Any) -> None:
        user = db.users.get(user_id)
        if user is None:
            return

        has_cancelled_flight = False
        has_delayed_flight = False
        for res_id in user.reservations:
            r = db.reservations.get(res_id)
            if r is None:
                continue
            for rf in r.flights:
                flight = db.flights.get(rf.flight_number)
                if flight is None:
                    continue
                date_status = flight.dates.get(rf.date)
                if date_status is None:
                    continue
                status = getattr(date_status, "status", "")
                has_cancelled_flight = has_cancelled_flight or status == "cancelled"
                has_delayed_flight = has_delayed_flight or status == "delayed"

        if has_cancelled_flight or not has_delayed_flight:
            return

        successful_calls = getattr(toolkit, "_harness_successful_calls", []) or []
        has_prior_change_or_cancel = any(
            call.get("tool_name")
            in {"cancel_reservation", "update_reservation_flights"}
            for call in successful_calls
        )
        if has_prior_change_or_cancel:
            return

        raise ValueError(
            "Cannot issue a compensation certificate for a delayed flight before "
            "a reservation change or cancellation has been completed. Per policy, "
            "delay compensation is only available when the user wants to change or "
            "cancel the reservation, after confirming the delay and completing that "
            "change/cancel. If the user only wants to complain or keep the flight "
            "unchanged, confirm the delay and explain that no certificate should be "
            "issued."
        )


class HarnessedAirlineTools(HarnessedToolKitMixin, AirlineTools):
    """AirlineTools with pre-execution harness validation enabled."""

    harness_rules: dict[str, list] = {
        "book_reservation": [
            BookReservationPaymentRule(),
            BookReservationItineraryRule(),
            MaxPassengersRule(),
            FlightStatusBookableRule(),
            BaggageAllowanceRule(),
            BookReservationPaymentTotalRule(),
        ],
        "cancel_reservation": [CancelFlightRule()],
        "update_reservation_flights": [
            BasicEconomyFlightChangeRule(),
            NoOriginDestChangeRule(),
            CabinChangeOnFlyingReservationRule(),
            FlightUpdateNoCertificateRule(),
            FlightUpdatePaymentCapacityRule(),
        ],
        "update_reservation_baggages": [
            NoBaggageDecreaseRule(),
            BaggageAllowanceUpdateRule(),        # A16: nonfree_baggages == total − free
        ],
        "update_reservation_passengers": [NoPassengerCountChangeRule()],
        "send_certificate": [
            CertificateEligibilityRule(),
            CertificateFlightStatusRule(),       # A17: must have actual delay/cancellation
            DelayedCompensationRequiresChangeRule(),
            CertificateAmountCapRule(),
        ],
    }


# ---------------------------------------------------------------------------
# H3 variants — tool-description policy embedding
# ---------------------------------------------------------------------------


class H3AirlineTools(H3AirlineToolDescriptionMixin, AirlineTools):
    """AirlineTools with H3 tool-description policy hints only (no H2 runtime validation)."""


class H3HarnessedAirlineTools(H3AirlineToolDescriptionMixin, HarnessedAirlineTools):
    """AirlineTools with both H3 description hints and H2 runtime validation."""


# ---------------------------------------------------------------------------
# H4 Annotators
# ---------------------------------------------------------------------------


class UserReservationSummaryAnnotator:
    """H4: When get_user_details returns a user with multiple reservations, list
    each reservation with a one-line summary.

    Agents commonly pick the wrong reservation when a user has several (e.g.
    most-recent vs. the one they're calling about).  This annotation surfaces
    all reservations with enough context for the agent to identify the right one
    without extra tool calls.
    """

    tool_name = "get_user_details"

    def annotate(self, db: FlightDB, result: Any, **_: Any) -> str | None:
        res_ids = getattr(result, "reservations", None)
        if not res_ids or len(res_ids) < 2:
            return None

        lines: list[str] = [
            f"[H4] User has {len(res_ids)} reservations — identify the correct one before acting:"
        ]
        for rid in res_ids:
            r = db.reservations.get(rid)
            if r is None:
                lines.append(f"  • {rid}: (not found)")
                continue
            status_tag = " [CANCELLED]" if getattr(r, "status", None) == "cancelled" else ""
            # Compute total charged (sum of positive payments)
            total = sum(
                p.amount for p in r.payment_history if p.amount > 0
            )
            dates = ", ".join(rf.date for rf in r.flights) if r.flights else "?"
            lines.append(
                f"  • {rid}: {r.origin}→{r.destination} ({r.flight_type}), "
                f"{r.cabin}, {dates}, ${total} paid, "
                f"{len(r.passengers)} pax{status_tag}"
            )
        lines.append(
            "Ask the user for the reservation ID or use the flight/date details above "
            "to confirm which reservation they mean before taking any action. "
            "If the request is about all upcoming flights, duplicate/same-day bookings, "
            "or a schedule mixup, inspect every reservation_id before deciding what to "
            "cancel, keep, or transfer."
        )
        return "\n".join(lines)


class ReservationPaymentAnnotator:
    """H4: After get_reservation_details, surface the total charged and per-method
    breakdown.

    Addresses COMMUNICATE failures where the agent performs correct DB writes but
    omits the exact dollar amount when replying to the user.
    """

    tool_name = "get_reservation_details"

    def annotate(self, db: FlightDB, result: Any, **_: Any) -> str | None:
        payment_history = getattr(result, "payment_history", None)
        if not payment_history:
            return None

        total_charged = sum(p.amount for p in payment_history if p.amount > 0)
        if total_charged == 0:
            return None

        # Per-method breakdown
        by_method: dict[str, int] = {}
        user_id = getattr(result, "user_id", None)
        user = db.users.get(user_id) if user_id else None
        for p in payment_history:
            if p.amount <= 0:
                continue
            pm_data = user.payment_methods.get(p.payment_id) if user else None
            src = getattr(pm_data, "source", p.payment_id)
            by_method[f"{src} ({p.payment_id})"] = by_method.get(
                f"{src} ({p.payment_id})", 0
            ) + p.amount

        detail = "; ".join(f"${v} via {k}" for k, v in by_method.items())
        return (
            f"[H4] Total charged for this reservation: ${total_charged} ({detail}). "
            "If you cancel or modify this reservation, communicate the exact refund or "
            "charge amount — do not just say 'you will receive a refund'."
        )


class ReservationPolicyContextAnnotator:
    """H4: Surface compact reservation-specific facts that agents often infer wrong."""

    tool_name = "get_reservation_details"

    _FREE_BAGS = {
        "regular": {"basic_economy": 0, "economy": 1, "business": 2},
        "silver": {"basic_economy": 1, "economy": 2, "business": 3},
        "gold": {"basic_economy": 2, "economy": 3, "business": 4},
    }

    def annotate(self, db: FlightDB, result: Any, **_: Any) -> str | None:
        user_id = getattr(result, "user_id", None)
        user = db.users.get(user_id) if user_id else None
        if user is None:
            return None

        cabin = getattr(result, "cabin", "")
        num_pax = len(getattr(result, "passengers", None) or [])
        membership = getattr(user, "membership", "regular")
        free_per_pax = self._FREE_BAGS.get(membership, {}).get(cabin, 0)
        free_total = free_per_pax * num_pax
        insurance = getattr(result, "insurance", "no")

        parts = [
            f"[H4] User membership is {membership}; this {cabin} reservation has "
            f"{num_pax} passenger(s), so checked-bag allowance is {free_total} total "
            f"({free_per_pax}/passenger). Insurance={insurance}; insurance only supports "
            "cancellation for covered health/weather reasons, not personal conflicts."
        ]

        if cabin == "basic_economy":
            parts.append(
                " This is a basic_economy reservation — cabin upgrades (same flight "
                "numbers, new cabin class) are available via update_reservation_flights."
            )

        return "".join(parts)


class ReservationFlightSummaryAnnotator:
    """H4: After get_reservation_details, remind agent of round-trip completeness.

    For round-trip reservations, surface the exact current legs and the
    reservation-wide cabin constraint before the agent attempts a flight update.
    """

    tool_name = "get_reservation_details"

    def annotate(self, db: FlightDB, result: Any, **_: Any) -> str | None:
        flight_type = getattr(result, "flight_type", None)
        flights = getattr(result, "flights", None)
        if flight_type != "round_trip" or not flights:
            return None
        leg_list = ", ".join(
            f"{f.flight_number} {f.origin}->{f.destination} ({f.date})"
            for f in flights
        )
        return (
            f"[H4] This is a round_trip reservation with {len(flights)} leg(s): {leg_list}. "
            "When changing flights, include ALL legs in the same update_reservation_flights "
            "call. If only one direction changes, copy the other direction exactly from this "
            "result. Cabin is reservation-wide, so changing cabin affects every leg; do not "
            "represent it as a one-direction-only cabin change."
        )


class CancelRefundAnnotator:
    """H4: After a successful cancel_reservation, compute and surface the exact
    refund amount per payment instrument.

    The cancel tool reverses all payments (appends negative entries).  This
    annotation sums the NEGATIVE entries added by the cancellation to give the
    agent the precise refund amount to communicate.
    """

    tool_name = "cancel_reservation"

    def annotate(self, db: FlightDB, result: Any, **_: Any) -> str | None:
        payment_history = getattr(result, "payment_history", None)
        if not payment_history:
            return None

        # Negative entries = refunds added by this cancellation
        refunds: dict[str, int] = {}
        user_id = getattr(result, "user_id", None)
        user = db.users.get(user_id) if user_id else None
        for p in payment_history:
            if p.amount >= 0:
                continue
            pm_data = user.payment_methods.get(p.payment_id) if user else None
            src = getattr(pm_data, "source", p.payment_id)
            key = f"{src} ({p.payment_id})"
            refunds[key] = refunds.get(key, 0) + abs(p.amount)

        if not refunds:
            return None

        total = sum(refunds.values())
        detail = "; ".join(f"${v} to {k}" for k, v in refunds.items())
        return (
            f"[H4] Cancellation processed. Refund: ${total} total ({detail}). "
            "Communicate the EXACT refund amount to the user — e.g. "
            f"'${total} has been refunded to your {list(refunds.keys())[0]}'. "
            "Do not assume this refund is immediately reusable for another booking "
            "in the same conversation; use only currently listed payment-method "
            "balances unless a later get_user_details result shows the balance changed."
        )


class FlightUpdatePaymentAnnotator:
    """H4: After a successful update_reservation_flights, surface the net payment
    or refund so the agent communicates it.

    The update tool appends a payment entry (positive = charged, negative = refunded).
    This annotation finds the most recent entry to extract the amount.
    """

    tool_name = "update_reservation_flights"

    def annotate(self, db: FlightDB, result: Any, **_: Any) -> str | None:
        payment_history = getattr(result, "payment_history", None)
        if not payment_history:
            return None

        # The last payment entry is the one added by this update
        last = payment_history[-1]
        amount = last.amount
        if amount == 0:
            return None

        user_id = getattr(result, "user_id", None)
        user = db.users.get(user_id) if user_id else None
        pm_data = user.payment_methods.get(last.payment_id) if user else None
        src = getattr(pm_data, "source", last.payment_id)

        if amount > 0:
            return (
                f"[H4] Flight update: ${amount} charged to {src} ({last.payment_id}). "
                "Tell the user the EXACT amount charged — do not just say 'an additional "
                "charge has been applied'."
            )
        else:
            return (
                f"[H4] Flight update: ${abs(amount)} refunded to {src} ({last.payment_id}). "
                "Tell the user the EXACT refund amount."
            )


class BookingPaymentAnnotator:
    """H4: After a successful book_reservation, surface the total charged per
    instrument so the agent can confirm the exact amount with the user.
    """

    tool_name = "book_reservation"

    def annotate(self, db: FlightDB, result: Any, **_: Any) -> str | None:
        payment_history = getattr(result, "payment_history", None)
        if not payment_history:
            return None

        total = sum(p.amount for p in payment_history if p.amount > 0)
        if total == 0:
            return None

        user_id = getattr(result, "user_id", None)
        user = db.users.get(user_id) if user_id else None
        by_method: dict[str, int] = {}
        for p in payment_history:
            if p.amount <= 0:
                continue
            pm_data = user.payment_methods.get(p.payment_id) if user else None
            src = getattr(pm_data, "source", p.payment_id)
            key = f"{src} ({p.payment_id})"
            by_method[key] = by_method.get(key, 0) + p.amount

        detail = "; ".join(f"${v} via {k}" for k, v in by_method.items())
        return (
            f"[H4] Booking confirmed. Total charged: ${total} ({detail}). "
            "Confirm the EXACT amount with the user."
        )


# ---------------------------------------------------------------------------
# H4 Annotation Mixin + combined classes
# ---------------------------------------------------------------------------


class H4AirlineAnnotationMixin:
    """Mixin that adds H4 post-execution tool-response annotations to airline tools.

    Must be placed before HarnessedToolKitMixin in the MRO so that its
    ``harness_annotators`` dict takes precedence.
    """

    harness_annotators: dict[str, list] = {
        "get_user_details": [
            UserReservationSummaryAnnotator(),
        ],
        "get_reservation_details": [
            ReservationPaymentAnnotator(),
            ReservationPolicyContextAnnotator(),
            ReservationFlightSummaryAnnotator(),
        ],
        "cancel_reservation": [
            CancelRefundAnnotator(),
        ],
        "update_reservation_flights": [
            FlightUpdatePaymentAnnotator(),
        ],
        "book_reservation": [
            BookingPaymentAnnotator(),
        ],
    }


class H4AirlineTools(H4AirlineAnnotationMixin, HarnessedToolKitMixin, AirlineTools):
    """AirlineTools with H4 post-execution annotations only (no H2 rules)."""

    harness_rules: dict[str, list] = {}


class H4HarnessedAirlineTools(H4AirlineAnnotationMixin, HarnessedAirlineTools):
    """AirlineTools with H2 harness rules and H4 post-execution annotations."""


class H3H4HarnessedAirlineTools(
    H3AirlineToolDescriptionMixin, H4AirlineAnnotationMixin, HarnessedAirlineTools
):
    """AirlineTools with H2 rules, H3 description hints, and H4 annotations."""
