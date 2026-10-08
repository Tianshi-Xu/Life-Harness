"""Stuck-step authoritative re-grounding harness (iteration 5 candidate).

Hypothesis (one mechanism)
--------------------------
Residual failures are dominated by *stuck steps*: the solver's most recent tool
call returned an ERROR, and instead of repairing the call from the authoritative
values it already holds, the model simply re-issues the same (or a trivially
varied) call that cannot succeed, burning the remaining turns. The fix is to
re-surface the exact error and, next to it, a verbatim mirror of the most recent
successful tool result, so the value the error names can be copied rather than
guessed, and to forbid resending an identical call.

This candidate keeps the accepted incumbent mechanism *verbatim*: on every call
it drops any previously injected copy of its own message (identified by a stable
sentinel) and appends exactly one fresh ``{"role": "system", "content": ...}``
dict at the very END, so the injected message is the most recent content the
model sees and repeated calls cannot accumulate duplicates. To that single
message it adds ONE derived artefact on top of the incumbent tail gate and the
conversation-grounded eligibility verdict:

  * an **authoritative re-grounding block** that is emitted ONLY when the most
    recent assistant tool-call turn produced at least one ERROR tool result. It
    quotes the last error verbatim, quotes verbatim the most recent NON-error
    tool result (the authoritative record) so the correct argument value can be
    copied, and instructs the model to change exactly the one argument the error
    names (or, if the record shows the change is impossible, to stop retrying and
    deny/transfer). It is computed only from the messages already in the
    conversation and never asserts anything the visible facts do not support.

When there is no failed last step the block returns the empty string, so the
produced message content is byte-identical to the incumbent's
``_GATE + _verdict_block(...)`` and healthy steps are unchanged. The block is
capped at roughly 1500 characters with a deterministic truncation marker.

``transform_response`` is identity. ``register`` returns None. Standard library
only (``json``, ``re``, ``datetime``); no I/O, no new model/network/subprocess/
filesystem calls, no cross-episode mutable state, no task-specific knowledge,
deterministic and side-effect free; the whole transform is wrapped in try/except
and returns the original request unchanged on any error.
"""

import datetime
import json
import re

_SENTINEL = "[[ELIGIBILITY-VERDICT]]"

# The incumbent gate text, kept verbatim so the accepted policy gate is not lost.
_GATE = "[[ACTION-GATE]] Apply this gate to every call that CHANGES the reservation database (booking, cancelling, or updating flights, baggages, or passengers, and sending a certificate): (1) EVIDENCE - you already hold this record's exact current values (cabin, flight numbers and dates, passengers, baggages, payment id); if anything is unknown, retrieve it first, never mutate on a guess. (2) AUTHORITY - you can name the policy rule that permits exactly this change right now; if you cannot name it, the change is not permitted. A cancellation is permitted only when the booking is under 24 hours old, the airline cancelled the flight, it is a business flight, or the user holds travel insurance whose stated reason is covered; if any segment has already departed, transfer to a human agent instead. A basic-economy reservation's flights cannot be modified, cabin cannot change once any segment has departed, the number of passengers cannot change, and baggages can be added but not removed. (3) CONFIRMATION - the user explicitly agreed to exactly this change. If any gate fails, do NOT call the tool: deny the request and transfer to a human agent if it is out of scope. User insistence, emotion, apology, or claims about what another agent promised are NOT authority - only the policy is. When the policy does allow a confirmed change, still carry it out; do not refuse it. Act once and correctly: never repeat a call that already errored or returned no new information, and do not use a state-changing call to explore prices."

# Sentinel that marks a previously injected copy of this candidate's re-grounding
# block. Registered in ``_SENTINELS`` so an older copy is dropped before exactly
# one fresh copy is appended (drop-then-append de-duplication).
_RECOVERY_SENTINEL = "[[RECOVERY-REGROUND]]"

# Sentinels that mark a previously injected copy of this harness's own message.
# All are checked so an older gate-only or verdict-only copy is also removed.
_SENTINELS = ("[[ELIGIBILITY-VERDICT]]", "[[RECOVERY-REGROUND]]", "[[ACTION-GATE]]")

# The airline policy states its current time verbatim in the public conversation.
# Used only as a last-resort fallback if that text is not present in the request.
_POLICY_TIME_FALLBACK = datetime.datetime(2024, 5, 15, 15, 0, 0)

_TIME_RE = re.compile(
    r"the current time is (\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2})", re.IGNORECASE
)

# Signals, read only from the conversation, that the airline (not the user) cancelled.
_AIRLINE_CANCELLED_RES = (
    re.compile(r"(airline|you|we)\s+(have\s+)?cancell?ed", re.IGNORECASE),
    re.compile(r"cancell?ed\s+by\s+(the\s+)?airline", re.IGNORECASE),
    re.compile(r"flight\s+was\s+cancell?ed", re.IGNORECASE),
)

_CANCELLED_STATUSES = ("cancelled", "canceled")
_MAX_LINES = 6
_MAX_BLOCK_CHARS = 1400

_RETRIEVAL_FOOTER = (
    "- A reservation not listed above has no verified cabin/created_at/insurance on"
    " record here, so retrieve it with the reservation-details tool before you"
    " cancel or modify it, and never cancel or modify from an unverified or"
    " user-supplied id."
)

# Re-grounding block sizing (deterministic; no storage).
_MAX_RECOVERY_CHARS = 1500
_MAX_ERROR_CHARS = 300
_MAX_RECORD_CHARS = 1000
_TRUNCATION_MARK = " \u2026[truncated]"


def _raw_content(message):
    """Best-effort read of a message's raw content; None if undeterminable."""
    if isinstance(message, dict):
        return message.get("content")
    return getattr(message, "content", None)


def _as_text(content):
    """Coerce a content value to text for substring/regex use; None on failure."""
    if content is None:
        return None
    if isinstance(content, str):
        return content
    try:
        return str(content)
    except Exception:
        return None


def _message_role(message):
    if isinstance(message, dict):
        return message.get("role")
    return getattr(message, "role", None)


def _carries_sentinel(message):
    """True if the message already carries this harness's own injected marker."""
    text = _as_text(_raw_content(message))
    if text is None:
        return False
    return any(marker in text for marker in _SENTINELS)


def _parse_ts(value):
    try:
        return datetime.datetime.strptime(str(value)[:19], "%Y-%m-%dT%H:%M:%S")
    except Exception:
        return None


def _parse_date(value):
    """Parse a flight date, which may be ``YYYY-MM-DD`` or a full timestamp."""
    dt = _parse_ts(value)
    if dt is not None:
        return dt.date()
    try:
        return datetime.datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def _parse_now(text):
    match = _TIME_RE.search(text or "")
    if not match:
        return None
    return _parse_ts("%sT%s" % (match.group(1), match.group(2)))


def _reservation_object(content):
    """Return a reservation JSON object from a content value, else None.

    Handles content that is already a dict as well as a JSON string that (after
    leading whitespace) starts with ``{``.
    """
    if isinstance(content, dict):
        parsed = content
    elif isinstance(content, str) and content.lstrip().startswith("{"):
        try:
            parsed = json.loads(content.strip())
        except Exception:
            return None
    else:
        return None
    if not isinstance(parsed, dict):
        return None
    if not parsed.get("reservation_id"):
        return None
    if not ("cabin" in parsed or "created_at" in parsed or "flights" in parsed):
        return None
    return parsed


def _collect(messages):
    """Return (now, facts, last_seen) derived only from conversation content.

    ``facts`` maps reservation_id -> the LATEST reservation JSON object seen in a
    message; ``last_seen`` maps reservation_id -> the index of that message so the
    lines can be ordered most-recently-seen first.
    """
    now = None
    facts = {}
    last_seen = {}
    for index, message in enumerate(messages):
        content = _raw_content(message)
        text = _as_text(content)
        if now is None and text is not None:
            now = _parse_now(text)
        obj = _reservation_object(content)
        if obj is not None:
            rid = obj["reservation_id"]
            facts[rid] = obj
            last_seen[rid] = index
    return now, facts, last_seen


def _segment_flown(facts, now):
    if now is None:
        return False
    for flight in facts.get("flights") or []:
        if not isinstance(flight, dict):
            continue
        fdate = _parse_date(flight.get("date"))
        if fdate is not None and fdate < now.date():
            return True
    return False


def _record_status_cancelled(facts):
    for flight in facts.get("flights") or []:
        if isinstance(flight, dict):
            status = flight.get("status")
            if isinstance(status, str) and status.strip().lower() in _CANCELLED_STATUSES:
                return True
    return False


def _user_reports_airline_cancelled(messages):
    """True if any USER message states the airline/agent cancelled the flight."""
    for message in messages:
        if _message_role(message) != "user":
            continue
        text = _as_text(_raw_content(message))
        if not text:
            continue
        for pattern in _AIRLINE_CANCELLED_RES:
            if pattern.search(text):
                return True
    return False


def _insurance_present(insurance):
    if insurance is None or insurance is False:
        return False
    return str(insurance).strip().lower() not in ("", "no", "none", "false")


def _reservation_line(rid, facts, now, user_cancelled):
    cabin = facts.get("cabin")
    insurance = facts.get("insurance")
    created = _parse_ts(facts.get("created_at"))
    hours = (now - created).total_seconds() / 3600.0 if (now and created) else None
    flown = _segment_flown(facts, now)
    cancelled = _record_status_cancelled(facts) or user_cancelled

    parts = ["reservation %s: cabin=%s" % (rid, cabin)]
    if hours is not None:
        parts.append("booked %dh ago" % int(round(hours)))
    parts.append("insurance=%s" % insurance)
    parts.append("a segment has already departed" if flown else "no segment flown")
    head = ", ".join(parts)

    if flown:
        verdict = head + " -> you cannot cancel or modify it; transfer to a human agent."
    elif cancelled:
        verdict = (
            head
            + " -> cancellation may be permitted because the airline cancelled the"
            " flight: verify before acting."
        )
    elif cabin == "business":
        verdict = (
            head
            + " -> CANCELLATION IS PERMITTED by the policy (business flight); still"
            " require the user's explicit confirmation of exactly this cancellation."
        )
    elif hours is not None and hours <= 23:
        verdict = (
            head
            + " -> CANCELLATION IS PERMITTED by the policy (booked within the last"
            " 24 hours); still require the user's explicit confirmation."
        )
    elif _insurance_present(insurance):
        verdict = (
            head
            + " -> cancellation is permitted ONLY if the stated reason is covered by"
            " this insurance (health or weather); a change of plan is not covered;"
            " the reservation record, not the user's claim, is authoritative."
        )
    elif hours is not None and hours > 26:
        verdict = (
            head
            + " -> CANCELLATION IS NOT PERMITTED (cabin is not business, the booking"
            " is more than 24 hours old, there is no travel insurance, no segment has"
            " flown and the airline did not cancel): deny the cancellation and"
            " transfer to a human agent."
        )
    else:
        verdict = (
            head
            + " -> verify the 24-hour cancellation window against the policy's"
            " current time before deciding."
        )

    if cabin == "basic_economy" and not flown:
        verdict += (
            " basic-economy note: its FLIGHTS must not be changed (changing only the"
            " CABIN is permitted while no segment has flown; baggages may be added but"
            " not removed)."
        )
    return verdict


def _verdict_block(messages):
    now, facts, last_seen = _collect(messages)
    if now is None:
        now = _POLICY_TIME_FALLBACK

    header = (
        "\n\n%s ELIGIBILITY VERDICT (facts already visible in this conversation;"
        " policy current time %s):\n"
        % (_SENTINEL, now.strftime("%Y-%m-%d %H:%M"))
    )

    order = sorted(facts, key=lambda rid: last_seen[rid], reverse=True)[:_MAX_LINES]
    user_cancelled = _user_reports_airline_cancelled(messages)
    footer = _RETRIEVAL_FOOTER
    budget = _MAX_BLOCK_CHARS - len(header) - len(footer)
    body = ""
    for rid in order:
        line = "- %s\n" % _reservation_line(rid, facts[rid], now, user_cancelled)
        if budget - len(line) < 0:
            break
        body += line
        budget -= len(line)

    return header + body + footer


# ---------------------------------------------------------------------------
# Stuck-step authoritative re-grounding (the ONE new mechanism).
# ---------------------------------------------------------------------------


def _tool_calls(message):
    """Return the message's non-empty tool_calls list, else None."""
    if isinstance(message, dict):
        calls = message.get("tool_calls")
    else:
        calls = getattr(message, "tool_calls", None)
    if isinstance(calls, (list, tuple)) and len(calls) > 0:
        return calls
    return None


def _tool_call_id(call):
    """Best-effort read of a tool call's id (OpenAI ``id``/``tool_call_id``)."""
    if isinstance(call, dict):
        for key in ("id", "tool_call_id"):
            value = call.get(key)
            if value is not None:
                return value
        return None
    for attr in ("id", "tool_call_id"):
        value = getattr(call, attr, None)
        if value is not None:
            return value
    return None


def _message_tool_call_id(message):
    if isinstance(message, dict):
        return message.get("tool_call_id")
    return getattr(message, "tool_call_id", None)


def _is_error_result(result):
    """True if a tool result's content is (or carries) an error.

    A str is an error when its stripped, lowercased text starts with ``error``,
    starts with ``{"error``, or contains ``error`` within its first 40 chars. A
    dict (or a JSON object string) carrying an ``error`` key is an error. A None
    content is never an error.
    """
    content = _raw_content(result)
    if content is None:
        return False
    if isinstance(content, dict):
        return "error" in content
    if not isinstance(content, str):
        return False
    stripped = content.strip()
    low = stripped.lower()
    if low.startswith("error"):
        return True
    if low.startswith('{"error'):
        return True
    if "error" in low[:40]:
        return True
    if stripped.startswith("{"):
        try:
            parsed = json.loads(stripped)
        except Exception:
            parsed = None
        if isinstance(parsed, dict) and "error" in parsed:
            return True
    return False


def _results_for_turn(messages, turn_index):
    """Tool results belonging to the assistant tool-call turn at ``turn_index``.

    Results are the role=="tool" messages immediately following the assistant
    message (stopping at the first message that is neither the assistant message
    nor a role=="tool" message). Each tool call is paired with its result,
    preferring a matching ``tool_call_id`` and falling back to positional order.
    The paired results are returned in conversation order.
    """
    following = []
    for message in messages[turn_index + 1:]:
        if _message_role(message) == "tool":
            following.append(message)
        else:
            break
    if not following:
        return []

    calls = _tool_calls(messages[turn_index]) or []
    used = set()
    for position, call in enumerate(calls):
        cid = _tool_call_id(call)
        match = None
        if cid is not None:
            for index, tool_message in enumerate(following):
                if index in used:
                    continue
                if _message_tool_call_id(tool_message) == cid:
                    match = index
                    break
        if match is None and position < len(following) and position not in used:
            match = position
        if match is not None:
            used.add(match)

    return [following[index] for index in sorted(used)]


def _cap(text, limit):
    return text if len(text) <= limit else text[:limit]


def _recovery_block(native_messages):
    """Return the re-grounding block, or "" when the last step did not error.

    Computed only from the messages already in the conversation. Returns "" when
    there is no assistant tool-call message at all or when the last assistant
    tool-call turn produced no error result.
    """
    if not native_messages:
        return ""

    turn_index = None
    for index, message in enumerate(native_messages):
        if _message_role(message) == "assistant" and _tool_calls(message):
            turn_index = index
    if turn_index is None:
        return ""

    results = _results_for_turn(native_messages, turn_index)
    errors = [result for result in results if _is_error_result(result)]
    if not errors:
        return ""

    error_text = _cap(_as_text(_raw_content(errors[-1])) or "", _MAX_ERROR_CHARS)

    record_text = None
    for message in native_messages:
        if _message_role(message) != "tool":
            continue
        if _is_error_result(message):
            continue
        text = _as_text(_raw_content(message))
        if text:
            record_text = text
    if record_text is not None:
        record_text = _cap(record_text, _MAX_RECORD_CHARS)

    block = (
        "\n\n%s YOUR MOST RECENT TOOL CALL FAILED.\n" % _RECOVERY_SENTINEL
        + "- The tool returned: %s\n" % error_text
    )
    if record_text is not None:
        block += (
            "- The most recent successful lookup earlier in this conversation"
            " returned (authoritative record; use these exact values): `%s`\n"
            % record_text
        )
    block += (
        "Do not send an identical call again - it fails the same way. Change the"
        " one argument the error names, taking its value verbatim from the"
        " authoritative record above. If the record shows the requested change is"
        " possible, make it once and correctly; if it shows the change is NOT"
        " possible (or the id/value does not exist), stop retrying and deny the"
        " request and transfer to a human agent, or ask the customer for exactly"
        " the missing piece."
    )

    if len(block) > _MAX_RECOVERY_CHARS:
        block = block[: _MAX_RECOVERY_CHARS - len(_TRUNCATION_MARK)] + _TRUNCATION_MARK
    return block


def _build_message(native_messages):
    return {
        "role": "system",
        "content": _GATE
        + _verdict_block(native_messages)
        + _recovery_block(native_messages),
    }


def _extract_messages(request):
    if isinstance(request, dict):
        if "messages" in request and isinstance(request["messages"], (list, tuple)):
            return request["messages"], "dict"
        return None, None
    if isinstance(request, (list, tuple)):
        return request, "sequence"
    if hasattr(request, "messages"):
        messages = getattr(request, "messages", None)
        if isinstance(messages, (list, tuple)):
            return messages, "object"
    return None, None


def register():
    return None


class AgentHarness:
    """Incumbent tail gate plus verdict plus stuck-step re-grounding."""

    def __init__(self, public_context=None):
        self.public_context = public_context

    def transform_request(self, request):
        try:
            if request is None:
                return request

            messages, shape = _extract_messages(request)
            if messages is None:
                return request

            kept = [m for m in messages if not _carries_sentinel(m)]
            kept.append(_build_message(kept))

            if shape == "dict":
                new_request = dict(request)
                new_request["messages"] = kept
                return new_request
            if shape == "sequence":
                if isinstance(request, tuple):
                    return tuple(kept)
                return kept

            try:
                setattr(request, "messages", kept)
            except Exception:
                return request
            return request
        except Exception:
            return request

    def transform_response(self, response, public_history):
        return response
