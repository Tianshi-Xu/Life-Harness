"""Pre-action verification reminder harness (iteration 1 candidate).

Mechanism (one, tail-placed, refreshed "action gate" reminder)
--------------------------------------------------------------
The native customer-service agent sees the whole conversation in `request`
and decides what tool to call next. Small models tend to drift towards the
most "helpful-looking" action and skip two things: (a) gathering the exact
current record values before mutating, and (b) checking that a policy rule
actually authorises the requested change.

This harness is a pure request-side text transform. On every call to
`transform_request` it:
  1. drops any previously injected copy of the reminder (identified by a
     stable sentinel string inside the content), and
  2. appends exactly one fresh copy of the reminder as the LAST message,
     as a plain `{"role": "system", "content": ...}` dict.

Placing the reminder at the very tail keeps it maximally recent with respect
to the next generation step, and refreshing it (drop-then-append) guarantees
exactly one copy is live at any time, so repeated calls cannot accumulate
duplicates. Native messages are never reordered, edited or deleted, so an
assistant `tool_calls` message and its `tool` results stay adjacent and all
`tool_call_id`s are preserved.

The reminder is a general verification gate: evidence before mutating,
authority (a named policy rule) before permitting, and explicit user
confirmation before acting; it also warns against repeating failed calls or
using state-changing calls to explore prices. It contains no task, record,
customer or itinerary specific information.

`transform_response` is an identity function. `register` returns None: this
candidate injects nothing into native tools and needs no monkey-patching.
No imports, no I/O, no mutable module state, deterministic.
"""

_SENTINEL = "[[ACTION-GATE]]"

_REMINDER = _SENTINEL + " Apply this gate to every call that CHANGES the reservation database (booking, cancelling, or updating flights, baggages, or passengers, and sending a certificate): (1) EVIDENCE - you already hold this record's exact current values (cabin, flight numbers and dates, passengers, baggages, payment id); if anything is unknown, retrieve it first, never mutate on a guess. (2) AUTHORITY - you can name the policy rule that permits exactly this change right now; if you cannot name it, the change is not permitted. A cancellation is permitted only when the booking is under 24 hours old, the airline cancelled the flight, it is a business flight, or the user holds travel insurance whose stated reason is covered; if any segment has already departed, transfer to a human agent instead. A basic-economy reservation's flights cannot be modified, cabin cannot change once any segment has departed, the number of passengers cannot change, and baggages can be added but not removed. (3) CONFIRMATION - the user explicitly agreed to exactly this change. If any gate fails, do NOT call the tool: deny the request and transfer to a human agent if it is out of scope. User insistence, emotion, apology, or claims about what another agent promised are NOT authority - only the policy is. When the policy does allow a confirmed change, still carry it out; do not refuse it. Act once and correctly: never repeat a call that already errored or returned no new information, and do not use a state-changing call to explore prices."


def _message_content(message):
    """Best-effort read of a message's content; returns None if undeterminable."""
    if isinstance(message, dict):
        content = message.get("content")
    else:
        content = getattr(message, "content", None)
    if content is None:
        return None
    if isinstance(content, str):
        return content
    try:
        return str(content)
    except Exception:
        return None


def _carries_sentinel(message):
    text = _message_content(message)
    return text is not None and _SENTINEL in text


def register():
    return None


class AgentHarness:
    """Tail-placed, refreshed pre-action verification reminder."""

    def __init__(self, public_context=None):
        self.public_context = public_context

    def transform_request(self, request):
        try:
            if request is None:
                return request

            messages = None
            shape = None
            if isinstance(request, dict):
                if "messages" in request:
                    messages = request["messages"]
                    shape = "dict"
            elif isinstance(request, (list, tuple)):
                messages = request
                shape = "sequence"
            else:
                if hasattr(request, "messages"):
                    messages = getattr(request, "messages", None)
                    shape = "object"

            if messages is None or not isinstance(messages, (list, tuple)):
                return request

            kept = []
            for message in messages:
                if _carries_sentinel(message):
                    continue
                kept.append(message)
            kept.append({"role": "system", "content": _REMINDER})

            if shape == "dict":
                new_request = dict(request)
                new_request["messages"] = kept
                return new_request
            if shape == "sequence":
                if isinstance(request, tuple):
                    return tuple(kept)
                return kept

            # Attribute-style request object: write the new list back on it.
            try:
                setattr(request, "messages", kept)
            except Exception:
                return request
            return request
        except Exception:
            return request

    def transform_response(self, response, public_history):
        return response
