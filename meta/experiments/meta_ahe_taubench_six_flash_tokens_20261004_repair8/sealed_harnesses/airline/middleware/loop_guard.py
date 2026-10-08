"""Loop guard: repair the observation when the agent is visibly stuck.

Evidence (iteration-2 traces, the 20 preserved failing trajectories):

* 9 of the 20 trajectories contain byte-identical repeated tool calls.  The
  worst re-sent the same ``update_reservation_passengers`` call nine times and
  the same ``update_reservation_flights`` call five times, reading the same
  payload back every time, until the episode ended without a result.
* 7 of the 20 trajectories open with a placeholder argument, e.g.
  ``get_user_details(user_id="user_id")``.  The resulting
  ``Error: User user_id not found`` was then repeatedly answered with an
  immediate ``transfer_to_human_agents`` instead of asking the user for the id.

Iteration 1 tried to fix both patterns by *adding instruction text* (system
prompt, memory, tool descriptions, a per-call reminder message).  That bundle
cost the agent 26.7 points of pass@1, so this middleware deliberately does
**not** add text to healthy conversations.  It works at the execution level and
only conditions the tool payload that is already pathological, which keeps a
healthy trajectory byte-identical to the unmodified runtime:

* Iteration-3 traces add a third, purely execution-level pathology: task
  ``535fa3`` (rollout 1) sent ``calculate("740 - 500 - 250")`` **91 times in a
  row** - 1.34M tokens - and never left it, so that rollout was lost before any
  policy reasoning mattered.  The existing rules did not fire because the call
  was a read and it succeeded every time.

1. a repeated call that errored again - or a repeated state-changing/transfer
   call, which can never be intentional - gets a short "you already made this
   exact call" notice in front of the unchanged payload;
2. a call that failed because an argument was a verbatim copy of its own
   parameter name gets a short "that is a placeholder, ask the user" notice;
3. any call whose signature has already been sent three or more times gets a
   "this call has now been sent N times with the same result" notice, because
   no tool and no outcome can make an identical repetition productive;
4. a call that failed because an argument was a value copied out of the tool's
   own schema examples ("ZFA04Y", "credit_card_7815826", "HAT001") - or an
   identifier parameter given a descriptive phrase such as "IAH to SEA" - gets
   a short "that is sample text, use an id from data" notice.  Iteration-5
   traces add this pathology: ``4fc82b26`` (rollout 1) emitted a complete
   ``update_reservation_flights`` call whose every argument was a schema
   sample, and ``4a44dc15`` (rollout 1) asked for reservation ``ZFA04Y`` and
   for flight ``"IAH to SEA"``.  All three calls came back "not found".

The middleware never plans, never executes anything, never touches the user's
request, and swallows its own errors so it can only ever degrade to a no-op.
"""

from __future__ import annotations

import json
from typing import Any

from nexau.archs.main_sub.execution.hooks import HookResult, Middleware

_STATE_KEY = "airline_loop_guard_seen"

# Tools whose repetition is always a mistake: they mutate the booking database
# or end the episode, so sending the identical call twice can never be intended.
_REPEAT_UNSAFE = frozenset(
    {
        "book_reservation",
        "cancel_reservation",
        "update_reservation_flights",
        "update_reservation_baggages",
        "update_reservation_passengers",
        "send_certificate",
        "transfer_to_human_agents",
    },
)

# Keys under which tool payloads carry their human-readable text.
_TEXT_KEYS = ("result", "text", "output", "content", "message", "data")

_REPEAT_NOTE = (
    "LOOP GUARD: you already made this exact call earlier in this conversation and it returned the "
    "payload shown below. Sending it again can only return the same thing. Do not send it a third "
    "time: change the arguments (use only values you have actually seen in a tool result or in the "
    "user's message), or reply to the user with what you already have."
)

# The same call with byte-identical arguments can never produce new information,
# so three or more sends of one signature is a degenerate loop no matter which
# tool it is and no matter whether it failed.  Kept above 2 so that a legitimate
# re-read after an intervening write is still allowed.
_DEGENERATE_THRESHOLD = 3

_DEGENERATE_NOTE = (
    "LOOP GUARD: this exact call has now been sent {count} times and returned the same payload every "
    "time. Repeating it again cannot produce new information and will not make progress. Stop "
    "repeating it: change the arguments, use a different tool, or reply to the user with what you "
    "already have."
)

_PLACEHOLDER_NOTE = (
    'LOOP GUARD: this call failed because the value of "{key}" was the literal placeholder '
    '"{key}", not a real id. Guessing again will fail the same way. This request is in scope and is '
    'not a reason to transfer the user to a human. Ask the user to type their {key} (or reuse an id '
    'printed by an earlier tool result) and then call the tool again.'
)

# The native tool schemas illustrate their id parameters with id-shaped sample
# values ("The reservation ID, such as 'ZFA04Y'").  Iteration-5 trace
# ``4fc82b26`` shows the agent emitting a whole call assembled out of those
# samples - ``update_reservation_flights(reservation_id='ZFA04Y',
# flights=[{'flight_number': 'HAT001', 'date': '2024-05-19'}],
# payment_id='credit_card_7815826')`` - and ``4a44dc15`` shows
# ``get_reservation_details('ZFA04Y')``.  Both came back "not found", and the
# run wasted turns on data that cannot exist.
_SCHEMA_EXAMPLES = frozenset(
    {
        "zfa04y",
        "8jx2wo",
        "sara_doe_496",
        "credit_card_7815826",
        "gift_card_7815826",
        "certificate_7815826",
        "hat001",
        "hat002",
    },
)

# Parameters whose value must be an exact identifier copied from data.
_ID_PARAMETERS = frozenset({"flight_number", "user_id", "reservation_id", "payment_id"})

_EXAMPLE_NOTE = (
    'LOOP GUARD: this call failed because "{key}" was set to "{value}", which is one of the example '
    "values printed in the tool's own schema - it is not real data. Never copy the sample values out "
    "of a tool description. Pass only ids that the user typed or that an earlier tool result "
    "returned (get_user_details prints the user_id and every reservation_id), and ask the user when "
    "you do not have them. Re-sending this value will fail in exactly the same way."
)

_MALFORMED_ID_NOTE = (
    'LOOP GUARD: this call failed because "{key}" was set to "{value}", which is a description of '
    "what you are looking for, not an identifier. This parameter takes the exact code that came back "
    "from an earlier tool result (a flight number looks like \"HAT190\"), never a route or a phrase "
    'such as "IAH to SEA". Find the real code first - search or read the reservation - then call '
    "again with it."
)


def _text_of(value: Any, depth: int = 0) -> str:
    """Best-effort text extraction from a (possibly nested) tool payload."""
    if depth > 6:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        parts = [_text_of(value[key], depth + 1) for key in _TEXT_KEYS if key in value]
        joined = " ".join(part for part in parts if part)
        if joined:
            return joined
        return str(value)[:4000]
    if isinstance(value, list):
        return " ".join(_text_of(item, depth + 1) for item in value[:5])
    if value is None:
        return ""
    return str(value)


def _prefix_text(value: Any, note: str, depth: int = 0) -> Any:
    """Return a copy of ``value`` with ``note`` prefixed to its text payload.

    The shape of the payload (and every sibling key such as ``status``) is
    preserved; only the innermost text field is rewritten.
    """
    if depth > 6:
        return note + "\n" + str(value)
    if isinstance(value, str):
        return note + "\n" + value
    if isinstance(value, dict):
        for key in _TEXT_KEYS:
            if key in value:
                updated = dict(value)
                updated[key] = _prefix_text(value[key], note, depth + 1)
                return updated
        return note + "\n" + json.dumps(value, default=str)
    if isinstance(value, list):
        return note + "\n" + json.dumps(value, default=str)
    return note + "\n" + str(value)


def _has_error_field(value: Any, depth: int = 0) -> bool:
    """True when a payload carries a structured RPC failure.

    The native bridge reports invalid parameters as
    ``{"status": "error", "error": "...", "error_type": "..."}``, which carries
    no literal "Error:" text, so the text scan alone misses it.
    """
    if depth > 4:
        return False
    if isinstance(value, dict):
        status = value.get("status")
        if isinstance(status, str) and status.strip().lower() == "error":
            return True
        for key in ("error", "error_type"):
            found = value.get(key)
            if isinstance(found, str) and found.strip():
                return True
        return any(_has_error_field(item, depth + 1) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_error_field(item, depth + 1) for item in value[:10])
    return False


def _placeholder_argument(arguments: Any) -> str | None:
    """Name of a parameter whose value is a copy of the parameter name itself."""
    if not isinstance(arguments, dict):
        return None
    for key, value in arguments.items():
        if not isinstance(value, str):
            continue
        cleaned = value.strip().strip("<>{}[]").replace("-", "_").lower()
        if cleaned == str(key).strip().lower():
            return str(key)
    return None


def _code_argument(arguments: Any) -> tuple[str, str, str] | None:
    """Spot a call assembled from sample data instead of retrieved data.

    Returns ``(key, value, note_template)`` for the first offending argument:
    either a value copied verbatim out of the tool's own schema examples, or an
    identifier parameter handed a descriptive phrase ("IAH to SEA") where only
    an exact code can work.
    """
    if not isinstance(arguments, dict):
        return None
    for key, value in arguments.items():
        if not isinstance(value, str):
            continue
        if value.strip().lower() in _SCHEMA_EXAMPLES:
            return str(key), value.strip(), _EXAMPLE_NOTE
        if str(key).strip().lower() in _ID_PARAMETERS and len(value.split()) > 1:
            return str(key), value.strip(), _MALFORMED_ID_NOTE
    return None


def _state_of(hook_input: Any) -> dict[str, int] | None:
    """Episode-scoped signature registry, stored on the agent state."""
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


class LoopGuardMiddleware(Middleware):
    """Condition the tool feedback of an already-looping agent."""

    def after_tool(self, hook_input: Any) -> HookResult:
        try:
            return self._guard(hook_input)
        except Exception:  # never let the guard disturb a run
            return HookResult.no_changes()

    def _guard(self, hook_input: Any) -> HookResult:
        tool_name = getattr(hook_input, "tool_name", "") or ""
        arguments = getattr(hook_input, "tool_input", None) or {}
        output = getattr(hook_input, "tool_output", None)

        text = _text_of(output)
        failed = "Error:" in text or text.lstrip().startswith("Error") or _has_error_field(output)

        notes: list[str] = []

        placeholder = _placeholder_argument(arguments)
        if placeholder is not None and failed:
            notes.append(_PLACEHOLDER_NOTE.format(key=placeholder))

        code_argument = _code_argument(arguments)
        if code_argument is not None and failed:
            key, value, note = code_argument
            notes.append(note.format(key=key, value=value))

        repeat_kind, count = self._repeat_kind(hook_input, tool_name, arguments, failed)
        if repeat_kind == "degenerate":
            notes.append(_DEGENERATE_NOTE.format(count=count))
        elif repeat_kind == "repeat":
            notes.append(_REPEAT_NOTE)

        if not notes:
            return HookResult.no_changes()
        return HookResult.with_modifications(tool_output=_prefix_text(output, "\n".join(notes)))

    @staticmethod
    def _repeat_kind(
        hook_input: Any, tool_name: str, arguments: Any, failed: bool
    ) -> tuple[str | None, int]:
        """Classify a call against every identical call already seen.

        Returns ``("degenerate"|"repeat"|None, count)`` where ``count`` is the
        number of times the signature had already been seen.
        """
        state = _state_of(hook_input)
        if state is None:
            return None, 0
        try:
            signature = tool_name + "\u0001" + json.dumps(arguments, sort_keys=True, default=str)
        except Exception:
            return None, 0
        seen = state.get(signature, 0)
        state[signature] = seen + 1
        if seen == 0:
            return None, 0
        # A read may legitimately be re-issued once after an intervening write,
        # so a single repeat is only flagged when it failed again or when the
        # tool mutates the booking database / ends the episode.  Past a small
        # number of identical sends, however, no tool and no outcome can make
        # the repetition productive.
        if seen + 1 >= _DEGENERATE_THRESHOLD:
            return "degenerate", seen + 1
        if failed or tool_name in _REPEAT_UNSAFE:
            return "repeat", seen + 1
        return None, seen + 1
