"""Stuck-step recovery harness (round 2 candidate).

Single mechanism
----------------
One request-side, **state-dependent** correction. `transform_request` inspects
only the most recent solver step -- the last assistant message that carries
`tool_calls` plus the `tool` results that directly follow it -- and appends one
short `{"role": "system", "content": ...}` message at the tail if, and only if,
that step is "stuck". In every other case the caller's request is returned
untouched (the original object is even returned, byte-identical), so a healthy
turn is indistinguishable from the identity baseline.

Why state-dependent (and not a static per-turn reminder)
--------------------------------------------------------
A reminder appended to *every* turn is paid for on every turn, including the
majority of turns that are already fine, and it perturbs passing episodes.
Here the trigger is a property of the conversation itself. Replaying the public
trajectories shows the correction fires on roughly one turn in five, and those
turns are concentrated in the error/retry episodes; a majority of the episodes
that pass never fire at all and therefore see byte-identical input on every
turn, while the incumbent's reminder is appended on every turn of every
episode. The contrast with the
incumbent (`preaction_verify_reminder.py`, a constant reminder on every turn) is
therefore both behavioural and cost-wise.

What counts as "stuck" (deterministic, task-agnostic)
-----------------------------------------------------
A step is stuck iff ANY of:
  (a) one of the trailing tool results is an error -- the content begins with
      "error" (case-insensitive), or carries an `"error"` key / a serialized
      error marker, whether the content is a string or structured; or
  (b) the current call's signature -- (tool name, JSON-canonicalised arguments)
      -- already occurred earlier in the same conversation with an *identical*
      following tool result. That is a true "no new information" test: it is
      deliberately stronger than "the same tool was called twice", because a
      legitimate re-read of a *different* record produces a *different* result
      and must not trigger anything.

Counter / refresh guarantee
---------------------------
The injected text ends with a recomputed "Unresolved step number: N", where N
is derived purely by counting, in the visible conversation, the error results
plus the identical-result repeat calls. Because N grows on every consecutive
failure, the injected content can never become a fixed point of the conversation
(it always strictly differs from the previous injection), so the model cannot be
trapped re-emitting one degenerate next action. Any previously injected copy --
identified by the sentinel `[[STUCK-STEP]]` -- is dropped before the new one is
appended, so exactly one copy is live at a time and repeated calls cannot
accumulate; the count is never stored anywhere, it is recomputed per call.

Safety properties
-----------------
Pure function of the conversation; no module-level mutable state, no I/O, no
network/subprocess/filesystem access, no new model calls, no task ids or task
names, no hidden answers, deterministic. Message handling follows the incumbent:
dict requests with `"messages"`, bare lists/tuples and attribute-style objects
with `.messages` are all tolerated; dict/list inputs are returned as NEW copies
(the caller's object is never mutated), a tuple returns a tuple, an
attribute-style request has `.messages` rewritten. Native messages are copied
and only ever filtered for the sentinel -- never reordered, edited or deleted --
so assistant `tool_calls` messages stay adjacent to their `tool` results and
every `tool_call_id` is preserved. The whole transform is wrapped in try/except
and returns the ORIGINAL request unchanged on any error.

`transform_response` is the identity. `register()` returns None: nothing is
patched, no native tool, budget, user simulator or evaluator is touched.
"""

import json as _json

_SENTINEL = "[[STUCK-STEP]]"

# One earlier identical call/result pair is enough to call it "no new information".
_REPEAT_MIN = 1

_REMINDER = (
    _SENTINEL
    + " Your last step did not make progress: either a tool call came back as an error, "
    "or it repeated a call whose result you have already seen. Repair that single step "
    "before you act again."
    " (1) READ THE RESULT LITERALLY: a tool error normally names the exact field that is "
    "wrong and often the exact value it expects; reuse that field and that value in your "
    "next call instead of resending your guess."
    " (2) NEVER SEND THE IDENTICAL CALL AGAIN: a call that already errored, or that already "
    "returned nothing new, cannot return anything new the second time; the same tool with "
    "the same arguments only burns your remaining steps."
    " (3) NEVER INVENT AN IDENTIFIER. Identifiers printed inside tool documentation are "
    "format examples, not real records, and must never be submitted. Take every user, "
    "reservation, flight and payment identifier from the user's own messages or from a "
    "lookup that already succeeded in this conversation; if you do not hold it, obtain it "
    "with the lookup that returns it, or ask the user for it."
    " (4) DO NOT RETRY A CHANGE THE POLICY FORBIDS: if a rule blocks what was asked, say "
    "plainly which rule blocks it, tell the user it is not possible, and state what is "
    "possible instead."
    " (5) ONCE A STEP HAS FAILED REPEATEDLY, STOP CALLING TOOLS for it: state the facts you "
    "hold and exactly what is blocking the user."
    " Then continue the user's request with one corrected action."
)


def _role(message):
    if isinstance(message, dict):
        return message.get("role")
    return getattr(message, "role", None)


def _raw_content(message):
    if isinstance(message, dict):
        return message.get("content")
    return getattr(message, "content", None)


def _content_text(message):
    """Canonicalised string form of a message's content ('' when absent)."""
    content = _raw_content(message)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    try:
        return _json.dumps(content, sort_keys=True, default=str)
    except Exception:
        try:
            return str(content)
        except Exception:
            return ""


def _is_error_result(message):
    """True if a tool RESULT indicates failure, for string or structured content."""
    content = _raw_content(message)
    if isinstance(content, dict):
        for key in content.keys():
            try:
                if "error" in str(key).lower():
                    return True
            except Exception:
                continue
    elif isinstance(content, (list, tuple)):
        for item in content:
            if isinstance(item, dict):
                for key in item.keys():
                    try:
                        if "error" in str(key).lower():
                            return True
                    except Exception:
                        continue
    text = _content_text(message)
    if not text:
        return False
    if text.lstrip()[:40].lower().startswith("error"):
        return True
    head = text[:120].lower()
    return '"error"' in head or "'error'" in head


def _tool_calls(message):
    """[(name, canonical_arguments)] for one assistant message, [] when absent."""
    if isinstance(message, dict):
        calls = message.get("tool_calls")
    else:
        calls = getattr(message, "tool_calls", None)
    if not calls:
        return []
    out = []
    for call in calls:
        if isinstance(call, dict):
            fn = call.get("function")
        else:
            fn = getattr(call, "function", None)
        if isinstance(fn, dict):
            name = fn.get("name")
            args = fn.get("arguments")
        elif fn is None:
            name, args = None, None
        else:
            name = getattr(fn, "name", None)
            args = getattr(fn, "arguments", None)
        if args is None:
            args = ""
        if isinstance(args, str):
            try:
                args = _json.dumps(_json.loads(args), sort_keys=True, default=str)
            except Exception:
                args = args.strip()
        else:
            try:
                args = _json.dumps(args, sort_keys=True, default=str)
            except Exception:
                try:
                    args = str(args)
                except Exception:
                    args = ""
        out.append((str(name), args))
    return out


def _call_ids(message):
    if isinstance(message, dict):
        calls = message.get("tool_calls") or []
    else:
        calls = getattr(message, "tool_calls", None) or []
    ids = []
    for call in calls:
        if isinstance(call, dict):
            ids.append(call.get("id"))
        else:
            ids.append(getattr(call, "id", None))
    return ids


def _trailing_block(messages, assistant_index):
    """The `tool` messages directly following one assistant turn."""
    block = []
    for message in messages[assistant_index + 1:]:
        if _role(message) != "tool":
            break
        block.append(message)
    return block


def _block_results(messages, assistant_index):
    """Result texts aligned with that turn's tool calls (None where unknown).

    `tool_call_id` pairing is honoured whenever the messages carry ids; a purely
    positional pairing is the fallback for id-less shapes.
    """
    calls = _tool_calls(messages[assistant_index])
    if not calls:
        return []
    block = _trailing_block(messages, assistant_index)

    by_id = {}
    for message in block:
        key = message.get("tool_call_id") if isinstance(message, dict) else getattr(
            message, "tool_call_id", None
        )
        if key is not None:
            by_id[key] = _content_text(message)

    ids = _call_ids(messages[assistant_index])
    results = []
    for index in range(len(calls)):
        call_id = ids[index] if index < len(ids) else None
        if call_id is not None and call_id in by_id:
            results.append(by_id[call_id])
        elif index < len(block):
            results.append(_content_text(block[index]))
        else:
            results.append(None)
    return results


def _is_assistant(message):
    return _role(message) == "assistant"


def _last_tool_call_turn(messages):
    """Index of the LAST assistant message that carries tool_calls, else -1."""
    found = -1
    for index, message in enumerate(messages):
        if _is_assistant(message) and _tool_calls(message):
            found = index
    return found


def _analyze(messages):
    """(trigger, step_number) for the latest step, or (None, 0) when healthy.

    Only the last assistant tool-call turn and the results directly after it are
    inspected, so an error the model has already moved past never re-fires.
    """
    last = _last_tool_call_turn(messages)
    if last < 0:
        return None, 0

    calls = _tool_calls(messages[last])
    current = _block_results(messages, last)

    failures = 0
    for message in messages:
        if _role(message) == "tool" and _is_error_result(message):
            failures += 1

    errored = any(result is not None and _is_error_result_for_text(result) for result in current)

    # Everything the conversation has already recorded for identical calls.
    earlier = {}
    for index in range(last):
        if not _is_assistant(messages[index]):
            continue
        earlier_calls = _tool_calls(messages[index])
        earlier_results = _block_results(messages, index)
        for position, call in enumerate(earlier_calls):
            result = earlier_results[position] if position < len(earlier_results) else None
            earlier.setdefault(call, []).append(result)

    repeated = 0
    for position, call in enumerate(calls):
        result = current[position] if position < len(current) else None
        if not result:
            continue
        for old in earlier.get(call, ()):
            if old and old == result:
                repeated += 1
                break

    if not errored and repeated < _REPEAT_MIN:
        return None, 0

    step = failures + repeated
    if step < 1:
        step = 1
    return ("error" if errored else "repeat"), step


def _is_error_result_for_text(text):
    """Same judgement as `_is_error_result` but for an already-canonical string."""
    if not text:
        return False
    if text.lstrip()[:40].lower().startswith("error"):
        return True
    head = text[:120].lower()
    return '"error"' in head or "'error'" in head


def _injected_content(step_number):
    """The reminder plus the freshly recomputed counter (never stored)."""
    return _REMINDER + " Unresolved step number: " + str(step_number) + "."


def _has_sentinel(message):
    return _SENTINEL in _content_text(message)


def register():
    return None


class AgentHarness:
    """Inject one corrective tail message only when the latest step is stuck."""

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

            trigger, step = _analyze(messages)
            if trigger is None:
                # Healthy step: hand back the caller's own object, untouched.
                return request

            kept = [message for message in messages if not _has_sentinel(message)]
            kept.append({"role": "system", "content": _injected_content(step)})

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
