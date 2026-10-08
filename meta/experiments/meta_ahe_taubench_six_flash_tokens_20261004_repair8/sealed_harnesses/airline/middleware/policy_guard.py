"""Policy guard: surface the cancellation-eligibility verdict exactly where it is decided.

Evidence (iteration-3 traces, the 10 tasks whose raw trajectories were preserved
this round; all of them are stable-fail tasks):

* 8 of the 20 preserved trajectories perform a cancellation that the native
  policy *forbids*.  In every one of them the agent had already read the
  reservation, so the decisive facts were sitting in the conversation:

  ============ ===== ========================= ============= ============
  task         res.  cabin                     created_at    insurance
  ============ ===== ========================= ============= ============
  19581        IFOYYZ basic_economy             2024-05-12    no
  3d914        UDMOP1 basic_economy             2024-05-09    no
  44cb73       D1EW9B basic_economy             2024-05-04    no
  5feceb       EHGLP3 basic_economy             2024-05-04    no
  31489        H8Q05L basic_economy             2024-05-03    yes (but the
                                                              reason was a
                                                              birthday, not
                                                              health/weather)
  ============ ===== ========================= ============= ============

  The policy allows a cancellation only when the booking is < 24 h old, the
  flight was cancelled by the airline, the cabin is business, or travel
  insurance covers the reason (health/weather).  None of the four mechanical
  grounds hold for the rows above, so in each case the agent cancelled when it
  should have refused.  "The API does not check that cancellation rules are
  met, so the agent must make sure the rules apply before calling the API!"

* Iteration 1 tried to fix this class with *generic instruction text* (a 1.4k
  token <operating_rules> block plus a per-turn reminder message).  It did not
  work: the patterns persisted, and the bundle cost 26.7 points of pass@1.
  Iteration 2's narrower lesson was the useful one - condition a tool payload
  only when the situation is already unambiguous and leave healthy
  trajectories byte-identical.

* Iteration 5 (the traces analysed this round) adds three facts the first
  version of the verdict got wrong or left out:

  ============ ====== ======================================================
  task         what   the guard said / should have said
  ============ ====== ======================================================
  0b918943     4XGCCM business, but its flights departed on 2024-05-03/04
                      while the policy's current time is 2024-05-15.  The
                      policy's *first* bar - "if any portion of the flight has
                      already been flown, the agent cannot help" - was never
                      tested: the business branch returned "eligible" first.
                      Both rollouts cancelled it.
  7902699b     XEHM4B basic_economy, no insurance, older than 24 h.  Both
                      rollouts proposed to change the cabin to business *in
                      order to* make it cancellable, and rollout 1 executed
                      the upgrade.
  7902699b     59XX6W economy + insurance, and the user said they were sick.
                      Rollout 1 reported "'sick' (not health or weather
                      related)" and refused a cancellation the insurance does
                      cover.
  73475cb4     HTR26G economy, no insurance, older than 24 h.  The note said
                      the cancellation was NOT permitted and the agent
                      cancelled it anyway four messages later.
  ============ ====== ======================================================

  So the verdict now also (i) tests the already-flown bar *before* every other
  criterion, (ii) spells out that insurance covers illness/injury and weather
  and nothing else, and (iii) closes the loophole explicitly: the grounds are
  judged on the reservation as it stands, so changing the cabin / re-booking
  does not create one.

This middleware keeps that discipline.  It never plans, never executes
anything, never blocks a call, never edits arguments, and never touches the
user's request.  It only *states a fact that is already derivable* from the
policy and the reservation the agent just read:

1. ``before_model`` records two episode-local facts (in agent-state context,
   so nothing is written outside the episode): the current time parsed from the
   native policy, and whether the user has asked for a cancellation at all.
2. ``after_tool`` on ``get_reservation_details`` prefixes a short, accurate
   eligibility verdict to the unchanged payload when - and only when - the
   reservation cannot satisfy the mechanical grounds (a flight has already
   departed, or the cabin is not business and the booking is older than 24 h)
   *and* a cancellation is actually on the table.  Every other read, every
   other tool and every other task is returned as ``HookResult.no_changes()``.

It is a strict superset-free no-op for a task whose cancellation is legitimate:
business cabins, fresh bookings and non-cancellation conversations never see a
note.  All exceptions are swallowed so the middleware can only degrade to the
unmodified runtime.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from typing import Any

from nexau.archs.main_sub.execution.hooks import HookResult, Middleware

_STATE_TIME = "airline_policy_guard_now"
_STATE_CANCEL_INTENT = "airline_policy_guard_cancel_intent"

# "The current time is 2024-05-15 15:00:00 EST." - stated once in the policy.
_TIME_RE = re.compile(
    r"current time is\s+(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2}:\d{2})",
    re.IGNORECASE,
)
_CANCEL_RE = re.compile(r"cancel", re.IGNORECASE)

_RESERVATION_TOOL = "get_reservation_details"

# Keys under which tool payloads carry their human-readable text (outermost
# first), mirroring the shapes returned by the native RPC bridge.
_TEXT_KEYS = ("result", "output", "content", "text", "message", "data")

_BUSINESS = "business"
_INSURANCE_YES = frozenset({"yes", "true", "1"})

# Appended to every negative verdict: the grounds are a property of the
# reservation as the user presented it, so a workaround that changes the
# reservation cannot manufacture a ground.
_NO_WORKAROUND = (
    "The grounds are judged on the reservation as it stands: changing the cabin, re-booking, or "
    "cancelling and re-booking does not make an ineligible reservation cancellable, so do not "
    "modify a reservation in order to cancel it."
)

_NO_INSURANCE_NOTE = (
    "CANCELLATION ELIGIBILITY CHECK (policy, computed on the reservation above): the cabin is "
    "`{cabin}` (not business), the booking was made {created}, which is more than 24 hours before "
    "the current time {now}, and the reservation has no travel insurance. None of the "
    "cancellation grounds apply unless the airline cancelled one of these flights. If the user is "
    "asking to cancel, this is NOT permitted: say that the reservation cannot be cancelled and "
    "why, and do not call cancel_reservation. " + _NO_WORKAROUND
)

_INSURANCE_NOTE = (
    "CANCELLATION ELIGIBILITY CHECK (policy, computed on the reservation above): the cabin is "
    "`{cabin}` (not business) and the booking was made {created}, which is more than 24 hours "
    "before the current time {now}, so only two grounds remain. The reservation can be cancelled "
    "only if the airline cancelled one of these flights, or if the travel insurance covers the "
    "user's reason. Travel insurance covers health (illness or injury) and weather reasons - a "
    "user who says they are sick, unwell, ill or injured has given a health reason - and nothing "
    "else: change of plan, personal events and other reasons are not covered. If neither ground "
    "holds, the cancellation is NOT permitted: say so and why, and do not call cancel_reservation. "
    + _NO_WORKAROUND
)

# The policy's first bar: "If any portion of the flight has already been
# flown, the agent cannot help".  It overrides every other ground, including a
# business cabin, so it is tested before them.
_FLOWN_NOTE = (
    "CANCELLATION ELIGIBILITY CHECK (policy, computed on the reservation above): this reservation "
    "cannot be cancelled at all - the flight(s) on {dates} have already departed, and the policy "
    "forbids cancelling any reservation a portion of which has already been flown. Do not call "
    "cancel_reservation on this reservation: tell the user it has already flown and cannot be "
    "cancelled. Judge any other reservation the user named separately, on its own merits. "
    + _NO_WORKAROUND
)


def _block_text(block: Any) -> str:
    if isinstance(block, str):
        return block
    return getattr(block, "text", "") or ""


def _message_text(message: Any) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        return " ".join(_block_text(block) for block in content)
    return ""


def _role_of(message: Any) -> str:
    role = getattr(message, "role", "")
    return getattr(role, "value", role) or ""


def _parse_dt(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip().replace("Z", "")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _state_of(hook_input: Any) -> dict[str, Any] | None:
    """Episode-scoped scratchpad, stored on the agent state."""
    agent_state = getattr(hook_input, "agent_state", None)
    getter = getattr(agent_state, "get_context_value", None)
    setter = getattr(agent_state, "set_context_value", None)
    if getter is None or setter is None:
        return None
    state = getter(_STATE_KEY, None)
    if not isinstance(state, dict):
        state = {}
        setter(_STATE_KEY, state)
    return state


_STATE_KEY = "airline_policy_guard_state"


def _find_reservation(value: Any, depth: int = 0) -> dict[str, Any] | None:
    """Locate the reservation JSON embedded in a (possibly nested) payload."""
    if depth > 6:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text.startswith("{"):
            return None
        try:
            parsed = json.loads(text)
        except Exception:
            return None
        if isinstance(parsed, dict) and "reservation_id" in parsed:
            return parsed
        return None
    if isinstance(value, dict):
        for key in list(_TEXT_KEYS) + [k for k in value if k not in _TEXT_KEYS]:
            if key in value:
                found = _find_reservation(value[key], depth + 1)
                if found is not None:
                    return found
        return None
    if isinstance(value, (list, tuple)):
        for item in value[:10]:
            found = _find_reservation(item, depth + 1)
            if found is not None:
                return found
        return None
    return None


def _prefix_first_string(value: Any, note: str, depth: int = 0) -> Any:
    """Copy of ``value`` with ``note`` prefixed to its first text payload.

    Everything else about the payload - including sibling keys such as
    ``status`` and the surrounding dict/list structure - is preserved.
    """
    if depth > 6:
        return None
    if isinstance(value, str):
        return note + "\n" + value
    if isinstance(value, dict):
        for key in list(_TEXT_KEYS) + [k for k in value if k not in _TEXT_KEYS]:
            if key in value:
                updated = _prefix_first_string(value[key], note, depth + 1)
                if updated is not None:
                    copy = dict(value)
                    copy[key] = updated
                    return copy
        return None
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value[:50]):
            updated = _prefix_first_string(item, note, depth + 1)
            if updated is not None:
                copy = list(value)
                copy[index] = updated
                return copy
        return None
    return None


def _departed_dates(reservation: dict[str, Any], now: datetime) -> list[str]:
    """Dates of flights that have already departed relative to ``now``.

    The tool payload gives each flight a plain ``YYYY-MM-DD`` date and no time
    of day, so a flight is only treated as flown once its date is strictly
    before the current date.
    """
    flights = reservation.get("flights")
    if not isinstance(flights, (list, tuple)):
        return []
    departed: list[str] = []
    for flight in flights[:20]:
        if not isinstance(flight, dict):
            continue
        date = flight.get("date")
        if not isinstance(date, str):
            continue
        day = _parse_dt(date[:10])
        if day is not None and day.date() < now.date():
            departed.append(str(date))
    return departed


def _verdict(reservation: dict[str, Any], now: datetime) -> str | None:
    """The eligibility note, or ``None`` when the reservation is fine."""
    departed = _departed_dates(reservation, now)
    if departed:
        return _FLOWN_NOTE.format(dates=", ".join(departed[:4]))
    cabin = str(reservation.get("cabin") or "").strip()
    created = _parse_dt(reservation.get("created_at"))
    if not cabin or created is None:
        return None
    if _BUSINESS in cabin.lower():
        # A business flight may always be cancelled.
        return None
    if created + timedelta(hours=24) > now:
        # Booked within the last 24 hours - also permitted.
        return None
    insurance = str(reservation.get("insurance") or "").strip().lower()
    template = _INSURANCE_NOTE if insurance in _INSURANCE_YES else _NO_INSURANCE_NOTE
    return template.format(
        cabin=cabin,
        created=str(reservation.get("created_at")),
        now=now.strftime("%Y-%m-%d %H:%M:%S"),
    )


class PolicyGuardMiddleware(Middleware):
    """State the cancellation-eligibility verdict on the reservation read."""

    def before_model(self, hook_input: Any) -> HookResult:
        try:
            self._refresh(hook_input)
        except Exception:  # never let bookkeeping disturb a run
            pass
        return HookResult.no_changes()

    def after_tool(self, hook_input: Any) -> HookResult:
        try:
            return self._guard(hook_input)
        except Exception:  # never let the guard disturb a run
            return HookResult.no_changes()

    @staticmethod
    def _refresh(hook_input: Any) -> None:
        state = _state_of(hook_input)
        if state is None:
            return
        now = state.get(_STATE_TIME)
        intent = bool(state.get(_STATE_CANCEL_INTENT))
        for message in getattr(hook_input, "messages", None) or []:
            role = _role_of(message)
            text = _message_text(message)
            if not text:
                continue
            if now is None and role == "system":
                match = _TIME_RE.search(text)
                if match:
                    now = _parse_dt(match.group(1) + "T" + match.group(2))
            elif not intent and role in ("user", "FRAMEWORK"):
                if _CANCEL_RE.search(text):
                    intent = True
        state[_STATE_TIME] = now
        state[_STATE_CANCEL_INTENT] = intent

    @staticmethod
    def _guard(hook_input: Any) -> HookResult:
        tool_name = getattr(hook_input, "tool_name", "") or ""
        if tool_name != _RESERVATION_TOOL:
            return HookResult.no_changes()

        state = _state_of(hook_input)
        if state is None or not state.get(_STATE_CANCEL_INTENT):
            return HookResult.no_changes()
        now = state.get(_STATE_TIME)
        if not isinstance(now, datetime):
            return HookResult.no_changes()

        output = getattr(hook_input, "tool_output", None)
        reservation = _find_reservation(output)
        if reservation is None:
            return HookResult.no_changes()

        note = _verdict(reservation, now)
        if note is None:
            return HookResult.no_changes()

        updated = _prefix_first_string(output, note)
        if updated is None:
            return HookResult.no_changes()
        return HookResult.with_modifications(tool_output=updated)
