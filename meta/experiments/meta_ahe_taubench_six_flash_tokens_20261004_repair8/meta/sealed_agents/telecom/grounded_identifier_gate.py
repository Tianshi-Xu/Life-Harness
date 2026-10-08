"""Harness variant: gate identifier arguments on the model's own reply.

A telecom support agent acts through a small set of functions, and almost every
one of them takes an identifier: a customer id, a line id, a bill id, a plan id,
a device id, a phone number, or a date of birth for a name lookup. Those values
are only meaningful when they come from the actual conversation -- something the
customer said, or something an earlier successful call returned.

A replay of the public training trajectories shows the small solver very often
invents the value instead: it passes literal placeholders such as ``1234567890``
/ ``9876543210`` (or ``line_1234567890``), the strings ``"user's customer id"`` /
``"user's line id"``, an example-format id such as ``C1001`` / ``L1001``, or a
guessed date of birth such as ``1970-01-01``. Such a call does not resolve: the
runtime answers ``Customer with ID ... not found`` (or ``[]`` for a name search)
and the turn is a dead end -- measured, the solvers' next move after that error is
to stop calling tools 175 times out of 207, so the turn is simply lost.

This harness keeps the model's intent but refuses to let it act on an invented
identifier. When the model's own reply carries a call whose gated identifier
argument is NOT grounded in the public conversation so far, that call is dropped
from the reply and the turn is delivered as ordinary speech: a plain-language
request asking the customer for the missing account detail (or for the phone
number / full name, which are the account-lookup keys, when the missing field is
not something a customer would know). The next turn therefore asks for real data
instead of spending a turn -- and an error-budget slot -- on an unresolvable call.

Only the model's own (public) reply and the public conversation are inspected.
No task identity, no hidden state, no per-task hints, no cross-episode state, and
no new model/network/subprocess/filesystem calls.
"""

from __future__ import annotations

import json
import re

# The 13 functions the agent actually holds (static, public native schema).
_NATIVE_TOOLS = frozenset(
    {
        "get_customer_by_phone",
        "get_customer_by_id",
        "get_customer_by_name",
        "get_details_by_id",
        "suspend_line",
        "resume_line",
        "get_bills_for_customer",
        "send_payment_request",
        "get_data_usage",
        "enable_roaming",
        "disable_roaming",
        "transfer_to_human_agents",
        "refuel_data",
    }
)

# Arguments that carry an identifier / account key. Any key ending in "_id" also
# counts. `full_name`/`name` are included because a fabricated name lookup is the
# same failure class: it silently returns an empty match list.
_GATED_KEYS = frozenset(
    {
        "id",
        "customer_id",
        "line_id",
        "bill_id",
        "plan_id",
        "device_id",
        "phone_number",
        "dob",
        "full_name",
        "name",
    }
)

_MIN_DIGITS = 4

_OPEN = (
    "I can't complete that yet -- I don't have the account detail it needs. "
)
_ASK_TAIL = (
    " Could you please tell me the {fields} for your account? If you don't have "
    "that to hand, just give me the phone number on your account (or your full "
    "name) and I'll look it up first, then continue from there."
)
_MAX_FIELDS = 3


def _join_fields(fields):
    if len(fields) <= 1:
        return fields[0] if fields else "account details"
    if len(fields) == 2:
        return "%s and %s" % (fields[0], fields[1])
    return ", ".join(fields[:-1]) + ", and " + fields[-1]


# --------------------------------------------------------------------------- #
# Guarded get / set helpers (dict and plain-object safe, never raise)
# --------------------------------------------------------------------------- #
def _get(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        try:
            return obj.get(key, default)
        except Exception:
            return default
    try:
        return getattr(obj, key, default)
    except Exception:
        return default


def _set(obj, key, value):
    if obj is None:
        return False
    if isinstance(obj, dict):
        try:
            obj[key] = value
            return True
        except Exception:
            return False
    try:
        setattr(obj, key, value)
        return True
    except Exception:
        pass
    try:
        object.__setattr__(obj, key, value)
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Grounding
# --------------------------------------------------------------------------- #
def _norm(text):
    """Lowercase and strip every non-alphanumeric character."""
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def _digits(text):
    return "".join(re.findall(r"\d+", str(text)))


def _is_gated(key):
    k = str(key).strip().lower()
    return k in _GATED_KEYS or k.endswith("_id")


def _grounded(value, gtext, gsig):
    """True when the value can be found in the public conversation.

    ``gtext`` is the normalized user/machine text; ``gsig`` is the set of digit
    signatures (normalized digit runs) that occur in it.
    """
    if value is None:
        return True
    text = str(value).strip()
    if not text:
        return True
    norm = _norm(text)
    if len(norm) < 2:
        # Too short to judge safely -> treat as grounded (never over-drop).
        return True
    if norm in gtext:
        return True
    sig = _digits(text)
    if len(sig) >= _MIN_DIGITS and sig in gtext:
        return True
    if len(sig) >= _MIN_DIGITS and sig in gsig:
        return True
    # Format-agnostic fallback: the same multiset of digits, whatever the
    # arrangement/formatting -- so a date the customer gave as "01/01/1990"
    # still grounds the model's reformatted "1990-01-01". Purely additive
    # (it can only ever accept more values), i.e. it strictly reduces drops.
    ssig = "".join(sorted(sig))
    if len(ssig) >= _MIN_DIGITS and ssig in gsig:
        return True
    return False


def _iter_messages(history):
    """Best-effort message iterator over whatever the public history is."""
    if history is None:
        return
    if isinstance(history, dict):
        msgs = history.get("messages")
        history = msgs if isinstance(msgs, (list, tuple)) else [history]
    elif _get(history, "messages", None) is not None and not isinstance(
        history, (list, tuple, str)
    ):
        history = _get(history, "messages")
    if isinstance(history, (list, tuple)):
        for item in history:
            yield item


def _public_grounding(history):
    """Normalized grounding text from user turns + non-error tool results."""
    parts = []
    seen_any = False
    for msg in _iter_messages(history) or ():
        role = _get(msg, "role", None)
        content = _get(msg, "content", None)
        if not isinstance(content, str):
            continue
        if role == "user":
            seen_any = True
            parts.append(content)
        elif role == "tool" and not content.lstrip().lower().startswith("error"):
            seen_any = True
            parts.append(content)
    if not seen_any:
        return None, set()
    raw = " ".join(parts)
    gsig = set(re.findall(r"\d{%d,}" % _MIN_DIGITS, raw))
    # Also record SORTED-DIGIT signatures (order-insensitive multisets of the
    # digits) so a value the customer gave in one formatting still grounds the
    # model's reformatted value: once per digit run in the source, and once per
    # message's full run of digits. This is purely additive to the grounding
    # set, so it can only ever accept more values, never drop more.
    for run in re.findall(r"\d+", raw):
        if len(run) >= _MIN_DIGITS:
            gsig.add("".join(sorted(run)))
    for part in parts:
        all_digits = "".join(re.findall(r"\d+", part))
        if len(all_digits) >= _MIN_DIGITS:
            gsig.add("".join(sorted(all_digits)))
    return _norm(raw), gsig


# --------------------------------------------------------------------------- #
# Tool-call inspection
# --------------------------------------------------------------------------- #
def _tool_name(call):
    fn = _get(call, "function", call)
    name = _get(fn, "name", None)
    if isinstance(name, str) and name.strip():
        return name.strip()
    return None


def _tool_args(call):
    fn = _get(call, "function", call)
    args = _get(fn, "arguments", None)
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
        except Exception:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def _humanise(name):
    words = [w for w in re.split(r"[_\s]+", str(name).strip()) if w]
    return " ".join(words) if words else "that detail"


def _ungrounded_fields(call, gtext, gsig):
    """Return the gated argument names of ``call`` that are not grounded."""
    missing = []
    for key, value in _tool_args(call).items():
        if not _is_gated(key):
            continue
        if not _grounded(value, gtext, gsig):
            name = _humanise(key)
            if name and name not in missing:
                missing.append(name)
    return missing


def _spoken_request(calls_and_missing):
    fields = []
    for _call, missing in calls_and_missing:
        for name in missing:
            if name not in fields:
                fields.append(name)
    fields = fields[:_MAX_FIELDS]
    return _OPEN + _ASK_TAIL.format(fields=_join_fields(fields))


# --------------------------------------------------------------------------- #
# Response-side walk (shape-agnostic, cycle-safe)
# --------------------------------------------------------------------------- #
def _gate_container(container, gtext, gsig):
    calls = _get(container, "tool_calls", None)
    if not isinstance(calls, list) or not calls:
        return

    keep = []
    dropped = []
    for call in calls:
        name = _tool_name(call)
        if name not in _NATIVE_TOOLS:
            # Off-schema (device-side) calls are deliberately left untouched by
            # this mechanism: a different error class, a different candidate.
            keep.append(call)
            continue
        missing = _ungrounded_fields(call, gtext, gsig)
        if missing:
            dropped.append((call, missing))
        else:
            keep.append(call)

    if not dropped:
        return

    if keep:
        # Still deliver the grounded calls; drop only the unresolvable ones.
        _set(container, "tool_calls", keep)
        return

    if not _set(container, "tool_calls", []):
        return
    text = _spoken_request(dropped)
    content = _get(container, "content", None)
    if isinstance(content, str) and content.strip():
        _set(container, "content", content.rstrip() + "\n\n" + text)
    else:
        _set(container, "content", text)


def _walk(node, seen, gtext, gsig):
    if node is None:
        return
    oid = id(node)
    if oid in seen:
        return
    seen.add(oid)

    if isinstance(node, dict):
        if isinstance(node.get("tool_calls"), list):
            _gate_container(node, gtext, gsig)
        for key in ("choices", "message"):
            if key in node:
                _walk(node[key], seen, gtext, gsig)
        return

    if isinstance(node, (list, tuple)):
        for item in node:
            _walk(item, seen, gtext, gsig)
        return

    if isinstance(_get(node, "tool_calls", None), list):
        _gate_container(node, gtext, gsig)
    for key in ("choices", "message"):
        child = _get(node, key, None)
        if child is not None:
            _walk(child, seen, gtext, gsig)


class AgentHarness:
    """Drops the model's own calls that rest on an invented identifier."""

    def __init__(self, public_context=None):
        self.public_context = public_context

    def transform_request(self, request):
        # Identity: the whole mechanism lives at the response boundary, so the
        # agent's own prompt and tool schema are left exactly as the runtime set
        # them (no reliance on the model obeying an injected rule).
        return request

    def transform_response(self, response, public_history=None):
        try:
            gtext, gsig = _public_grounding(public_history)
            if gtext is None:
                # No usable public conversation -> cannot ground safely.
                return response
            _walk(response, set(), gtext, gsig)
        except Exception:
            # The guard must never break an episode.
            pass
        return response


def register():
    return None
