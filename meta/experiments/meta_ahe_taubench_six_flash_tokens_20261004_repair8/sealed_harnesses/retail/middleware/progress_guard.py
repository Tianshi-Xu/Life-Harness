"""Episode-local progress guard for the native retail agent.

Every behaviour here was derived from a concrete observed failure. The middleware
never executes an action itself - it only removes tool calls that are provably
redundant, hands the model the data it already owns and continues the turn with an
internal note. All state is re-derived from the episode's own message history, so
the middleware is stateless and never writes outside the running episode.

1. Duplicate read suppression - re-issuing a read call whose result is already in
   the conversation (and which no later successful write can have invalidated)
   burns a turn; one traced episode re-authenticated seven times and never
   performed the write the task needed.
2. Failed call suppression - this environment is deterministic: an identical
   repeat of a call that already errored errors again, eats the error budget and
   can end the episode.
3. Turn continuation - when the model finishes a turn either claiming a write that
   no tool call actually performed, or - after a *short* affirmation that answers a
   confirmation question this agent actually asked - asking for that confirmation
   yet again, the turn is continued with a note so the agent acts instead of
   talking about acting. The affirmation test is deliberately narrow: three
   traced episodes showed the earlier, looser version firing on a customer
   message that merely started with "okay"/"yes", pushing the model straight into
   an irreversible write with the wrong variant or the wrong payment method.
4. Asking for what the profile already holds - once the customer is identified,
   asking them for an order id wastes the turn that the customer's own patience
   budget is paying for; the turn is continued with the user id and the call
   (get_user_details) that lists the orders and payment methods.
5. One-write-per-order - the native environment accepts one write per order
   lifetime (one return or exchange for a delivered order, one item modification
   for a pending order, and a cancelled order takes nothing). Recorded episodes
   show the agent, told by the customer to "also add" a second item, re-issuing a
   return on an order it had already written and then churning until the episode
   ended. A write aimed at an order that already has one of those writes is
   removed and replaced by a note that states what was already done and that the
   customer has to be told it cannot be changed, undone or extended.
"""

from __future__ import annotations

import json
import re
from typing import Any

from nexau.archs.main_sub.execution.hooks import (
    AfterModelHookInput,
    AfterToolHookInput,
    HookResult,
    Middleware,
)
from nexau.core.messages import Message, Role, TextBlock, ToolResultBlock, ToolUseBlock

_NOTE_MARK = "[PROGRESS NOTE]"
_MAX_NOTES = 14

_READ_TOOLS = frozenset(
    {
        "find_user_id_by_email",
        "find_user_id_by_name_zip",
        "get_user_details",
        "get_order_details",
        "get_product_details",
        "get_item_details",
        "list_all_product_types",
    },
)

_WRITE_TOOLS = frozenset(
    {
        "cancel_pending_order",
        "modify_pending_order_address",
        "modify_pending_order_items",
        "modify_pending_order_payment",
        "modify_user_address",
        "return_delivered_order_items",
        "exchange_delivered_order_items",
    },
)

# Writes after which the order can never be written again: a cancel leaves the
# order "cancelled" and a return/exchange leaves it in a requested state, so no
# further cancel, modification, return or exchange can succeed.
_ORDER_CONSUMING = frozenset(
    {
        "cancel_pending_order",
        "return_delivered_order_items",
        "exchange_delivered_order_items",
    },
)

# Writes the environment allows only once for the order they touch. The native
# tool documentation says a delivered order can be returned or exchanged only
# once and a pending order's items can be modified only once.
_ONCE_WRITE_TOOLS = _ORDER_CONSUMING | {"modify_pending_order_items"}

# "#W1234567", "W1234567", "# w 1234567" -> "#W1234567"
_ORDER_ID = re.compile(r"#?\s*[Ww]\s*(\d{5,})")

# The order object returned by a successful write carries its new status. Tool
# results arrive JSON-escaped (\"status\": \"return requested\"), so the pattern
# tolerates backslashes, quotes and spaces around the token and the colon.
_STATUS_TEXT = re.compile(r"status[\\\"'\s]*:[\s\\\"']*([A-Za-z][A-Za-z ]{2,24})", re.IGNORECASE)

_WRITE_VERB = {
    "cancel_pending_order": "cancelled",
    "return_delivered_order_items": "returned",
    "exchange_delivered_order_items": "exchanged",
    "modify_pending_order_items": "modified",
}

_ERROR_TEXT = re.compile(r"(?:^|\n)\s*Error\b|Error: ")

_AUTH_TOOLS = frozenset({"find_user_id_by_email", "find_user_id_by_name_zip"})

_USER_ID = re.compile(r"[A-Za-z][A-Za-z_]*_\d{3,}")

# "Could you please provide the order ID ..." - asking the customer for an
# identifier that get_user_details already returns for this customer. The
# request form matters: merely mentioning an order id (for example while listing
# variants) is not asking the customer for one.
_ASK_ID = re.compile(
    r"(?:could|can|would|will)\s+you\s+(?:please\s+)?"
    r"(?:provide|share|give|send|tell|confirm|specify|let me know)"
    r"[^.?!\n]{0,70}(?:order|item|product)\s*(?:id|ids|number|#)"
    r"|please\s+(?:provide|share|give|send|tell|specify|confirm)"
    r"[^.?!\n]{0,70}(?:order|item|product)\s*(?:id|ids|number|#)"
    r"|\b(?:what|which)\s+(?:is\s+|are\s+)?(?:the\s+|your\s+|those\s+|these\s+)?"
    r"(?:order|item|product)\s*(?:id|ids|number)"
    r"|\bI\s+(?:need|require|will need)\b[^.?!\n]{0,40}(?:order|item|product)\s*(?:id|ids|number)"
    r"|\bdo you (?:have|know|remember)\b[^.?!\n]{0,50}(?:order|item|product)\s*(?:id|ids|number)"
    r"|\b(?:which|what)\s+orders?\b[^.?!\n]{0,40}\b(?:refer|cancel|return|exchange|modify)"
    r"|\bcan(?:not|'t) (?:proceed|help|assist)[^.?!\n]{0,60}(?:order|item)\s*(?:id|ids|number)",
    re.IGNORECASE,
)

_CONFIRM_ASK = re.compile(
    r"would you like me to (?:proceed|cancel|modify|return|exchange|update|change|go ahead)"
    r"|shall i (?:proceed|cancel|modify|return|exchange|update|go ahead)"
    r"|do you want me to (?:proceed|cancel|modify|return|exchange|update|go ahead)"
    r"|please confirm (?:if|whether|that) you"
    r"|proceed with the (?:cancellation|modification|return|exchange|address|change|update)"
    r"|confirm the (?:cancellation|modification|return|exchange|address|change|update|refund)"
    r"|\(yes/no\)|yes/no\b",
    re.IGNORECASE,
)

_AFFIRM = re.compile(
    r"^\s*(?:yes|yeah|yep|yup|sure|ok|okay|correct|confirmed|confirm|that'?s right"
    r"|please (?:do|proceed|go ahead)|go ahead|proceed|do it|perfect|great)\b",
    re.IGNORECASE,
)

# action kind -> (what the user must have asked for, completion phrasing,
#                 tools that would really have performed it)
_CLAIMS = (
    (
        "address",
        re.compile(r"address", re.IGNORECASE),
        re.compile(
            r"(?:has|have|had)\s+been\s+(?:updated|modified|changed|set)"
            r"|i(?:'ve| have)\s+(?:successfully\s+)?(?:updated|modified|changed|set)"
            r"|successfully\s+(?:updated|modified|changed|set)"
            r"|(?:updated|modified|changed)\s+successfully",
            re.IGNORECASE,
        ),
        ("modify_pending_order_address", "modify_user_address"),
    ),
    (
        "cancel",
        re.compile(r"cancel", re.IGNORECASE),
        re.compile(
            r"(?:has|have|had)\s+been\s+cancel"
            r"|i(?:'ve| have)\s+(?:successfully\s+)?cancel"
            r"|successfully\s+cancel"
            r"|cancel(?:led|ation)\s+(?:has been|was|is)\s+(?:process|complete|done|request)"
            r"|cancel(?:led|ation)\s+successfully",
            re.IGNORECASE,
        ),
        ("cancel_pending_order",),
    ),
    (
        "change",
        re.compile(r"address|updat|modif|chang", re.IGNORECASE),
        re.compile(
            r"(?:has|have|had)\s+been\s+(?:updated|modified|changed)"
            r"|i(?:'ve| have)\s+(?:successfully\s+)?(?:updated|modified|changed)"
            r"|successfully\s+(?:updated|modified|changed)"
            r"|(?:updated|modified|changed)\s+successfully",
            re.IGNORECASE,
        ),
        (
            "modify_pending_order_address",
            "modify_pending_order_items",
            "modify_pending_order_payment",
            "modify_user_address",
        ),
    ),
    (
        "exchange",
        re.compile(r"exchang", re.IGNORECASE),
        re.compile(
            r"(?:has|have|had)\s+been\s+exchanged"
            r"|i(?:'ve| have)\s+(?:successfully\s+)?exchanged"
            r"|successfully\s+exchanged"
            r"|exchange\s+(?:has been|was|is)\s+(?:process|complete|done|request)",
            re.IGNORECASE,
        ),
        ("exchange_delivered_order_items",),
    ),
    (
        "return",
        re.compile(r"return|refund", re.IGNORECASE),
        re.compile(
            r"(?:has|have|had)\s+been\s+returned"
            r"|i(?:'ve| have)\s+(?:successfully\s+)?returned"
            r"|successfully\s+returned"
            r"|(?:has|have|had)\s+been\s+refunded"
            r"|i(?:'ve| have)\s+(?:successfully\s+)?refunded"
            r"|successfully\s+refunded",
            re.IGNORECASE,
        ),
        ("return_delivered_order_items", "exchange_delivered_order_items", "cancel_pending_order"),
    ),
)

_NEGATIVE_LEAD = re.compile(
    r"(?:will|would|can|could|should|may|might|to be|if|when|once|not|never|"
    r"cannot|can't|unable to|unless|before|after|already|must be|needs to be|has to be|"
    r"is going to be|are going to be)\s*$",
    re.IGNORECASE,
)

# A sentence that says a change happened "already" / "previously" is normally
# describing the state the order arrived in, not something this agent just did.
_STATE_QUALIFIER = re.compile(r"\b(?:already|previously|prior to|before this|earlier)\b", re.IGNORECASE)
_SENTENCE_SPLIT = re.compile(r"[.!?\n]")


def _bare_name(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    name = value.strip()
    if name.startswith("Tool: "):
        name = name[len("Tool: ") :].strip()
    return name


def _signature(name: str, arguments: Any) -> str:
    try:
        body = json.dumps(arguments, sort_keys=True, ensure_ascii=False)
    except Exception:  # pragma: no cover - defensive
        body = repr(arguments)
    return name + "|" + body[:400]


def _order_key(arguments: Any) -> str:
    """Canonical ``#W<digits>`` form of an ``order_id`` argument, else ``""``."""
    if not isinstance(arguments, dict):
        return ""
    raw = arguments.get("order_id")
    if not isinstance(raw, str):
        return ""
    found = _ORDER_ID.search(raw)
    if not found:
        return ""
    return "#W" + found.group(1)


def _status_of(text: str) -> str:
    found = _STATUS_TEXT.search(text or "")
    return found.group(1).strip().lower() if found else ""


def _text_of(block: Any) -> str:
    if isinstance(block, TextBlock):
        return block.text or ""
    if isinstance(block, str):
        return block
    return ""


def _result_text(block: ToolResultBlock) -> str:
    content = block.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(_text_of(inner) for inner in content)
    return ""


def _is_error(text: str) -> bool:
    return bool(_ERROR_TEXT.search(text or ""))


class _Scan:
    """A read-only view of the episode built from the message history."""

    def __init__(self) -> None:
        self.notes: dict[str, int] = {}
        self.read_result: dict[str, tuple[int, str]] = {}
        self.failed_result: dict[str, tuple[int, str, int]] = {}
        self.write_index: list[int] = []
        self.successful_writes: set[str] = set()
        self.order_write: dict[str, tuple[str, str]] = {}
        self.last_user_text: str = ""
        self.user_text: str = ""
        self.confirm_asks: int = 0
        self.auth_user_id: str = ""
        self.profile_read: bool = False


def _scan(messages: Any) -> _Scan:
    scan = _Scan()
    pending: dict[str, tuple[int, str, Any]] = {}
    index = 0
    for message in messages or []:
        role = getattr(message, "role", None)
        blocks = getattr(message, "content", None) or []
        texts: list[str] = []
        for block in blocks:
            if isinstance(block, ToolUseBlock):
                index += 1
                name = _bare_name(block.name)
                arguments = block.input if isinstance(block.input, dict) else {}
                pending[str(block.id)] = (index, name, arguments)
            elif isinstance(block, ToolResultBlock):
                info = pending.pop(str(block.tool_use_id), None)
                text = _result_text(block)
                if info is None:
                    continue
                position, name, arguments = info
                key = _signature(name, arguments)
                if not _is_error(text):
                    if name in _AUTH_TOOLS:
                        found = _USER_ID.search(text)
                        if found:
                            scan.auth_user_id = found.group(0)
                    elif name == "get_user_details":
                        scan.profile_read = True
                    if name in _ONCE_WRITE_TOOLS:
                        order = _order_key(arguments)
                        if order:
                            scan.order_write[order] = (name, _status_of(text))
                if name in _READ_TOOLS and not _is_error(text):
                    scan.read_result[key] = (position, text)
                if _is_error(text):
                    previous = scan.failed_result.get(key)
                    count = (previous[2] if previous else 0) + 1
                    scan.failed_result[key] = (position, text, count)
                elif name in _WRITE_TOOLS:
                    scan.write_index.append(position)
                    scan.successful_writes.add(name)
            else:
                text = _text_of(block)
                if text:
                    texts.append(text)

        joined = "\n".join(texts)
        if joined:
            for chunk in joined.split(_NOTE_MARK)[1:]:
                token = chunk.split("\n", 1)[0].strip()
                if token:
                    scan.notes[token] = scan.notes.get(token, 0) + 1
            if role is Role.ASSISTANT and _CONFIRM_ASK.search(joined):
                scan.confirm_asks += 1
        if role is Role.USER and joined.strip():
            scan.last_user_text = joined
            scan.user_text = (scan.user_text + "\n" + joined)[-6000:]
    return scan


def _write_after(scan: _Scan, position: int) -> bool:
    for index in scan.write_index:
        if index > position:
            return True
    return False


def _strip_calls(messages: Any, dropped: set) -> list:
    updated = []
    for message in messages or []:
        if getattr(message, "role", None) is not Role.ASSISTANT:
            updated.append(message)
            continue
        kept = []
        removed = False
        for block in message.content or []:
            if isinstance(block, ToolUseBlock) and str(block.id) in dropped:
                removed = True
                continue
            kept.append(block)
        if not removed:
            updated.append(message)
            continue
        if not kept:
            kept = [TextBlock(text="[redundant tool call removed by progress guard]")]
        updated.append(message.model_copy(update={"content": kept}))
    return updated


def _framework_note(text: str) -> Message:
    return Message(role=Role.FRAMEWORK, content=[TextBlock(text=text)])


def _sentence_around(text: str, position: int) -> str:
    start = 0
    end = len(text)
    for match in _SENTENCE_SPLIT.finditer(text):
        if match.end() <= position:
            start = match.end()
        elif match.start() >= position:
            end = match.start()
            break
    return text[start:end]


def _claim_without_action(
    text: str, scan: _Scan, proposed: frozenset[str] = frozenset()
) -> tuple[str, str] | None:
    for key, subject, phrasing, tools in _CLAIMS:
        if not subject.search(scan.user_text):
            # the customer never asked for this kind of action: the sentence is
            # almost certainly describing the state the order arrived in
            continue
        if not subject.search(text):
            continue
        match = phrasing.search(text)
        if not match:
            continue
        if _NEGATIVE_LEAD.search(text[max(0, match.start() - 24) : match.start()]):
            continue
        if _STATE_QUALIFIER.search(_sentence_around(text, match.start())):
            continue
        if any(tool in scan.successful_writes or tool in proposed for tool in tools):
            continue
        return key, tools[0]
    return None


class ProgressGuardMiddleware(Middleware):
    """Removes provably redundant calls and keeps the agent moving forward."""

    def after_model(self, hook_input: AfterModelHookInput) -> HookResult:
        try:
            return self._after_model(hook_input)
        except Exception:  # pragma: no cover - never break the loop
            return HookResult.no_changes()

    def _after_model(self, hook_input: AfterModelHookInput) -> HookResult:
        scan = _scan(hook_input.messages)
        parsed = hook_input.parsed_response
        calls = list(getattr(parsed, "tool_calls", None) or [])
        if calls:
            result = self._filter_calls(hook_input, parsed, calls, scan)
            if result.has_modifications():
                return result
            # The turn can also end with prose next to the calls. A sentence that
            # claims an action none of those calls performs is a false report that
            # no later tool result can repair.
            proposed = frozenset(_bare_name(getattr(call, "tool_name", "")) for call in calls)
            return self._claim_turn(hook_input, scan, proposed)
        return self._continue_turn(hook_input, scan)

    def _claim_turn(
        self, hook_input: AfterModelHookInput, scan: _Scan, proposed: frozenset
    ) -> HookResult:
        total_notes = sum(scan.notes.values())
        if total_notes >= _MAX_NOTES:
            return HookResult.no_changes()
        text = hook_input.original_response or ""
        if not text.strip():
            return HookResult.no_changes()
        note = self._claim_note(text, scan, proposed)
        if note is None:
            return HookResult.no_changes()
        messages = list(hook_input.messages or [])
        messages.append(_framework_note(note))
        return HookResult.with_modifications(messages=messages, force_continue=True)

    def _claim_note(self, text: str, scan: _Scan, proposed: frozenset) -> str | None:
        claim = _claim_without_action(text, scan, proposed)
        if claim is None:
            return None
        key = "claim:" + claim[0]
        if scan.notes.get(key, 0) >= 1:
            return None
        return (
            _NOTE_MARK
            + " "
            + key
            + "\nYour message states that an action was completed, but no successful "
            + claim[1]
            + " call exists in this conversation. Only a tool result is evidence that "
            "something happened. If the user has already confirmed this action and you never "
            "executed it, make the tool call now (in this turn or the next one). If it was never "
            "agreed or is not possible with these tools, rewrite the message so it does not claim "
            "the action was done."
        )

    # ------------------------------------------------------------------ calls

    def _filter_calls(self, hook_input, parsed, calls, scan: _Scan) -> HookResult:
        total_notes = sum(scan.notes.values())
        if total_notes >= _MAX_NOTES:
            return HookResult.no_changes()

        kept = []
        dropped: set[str] = set()
        notes: list[str] = []
        for call in calls:
            name = _bare_name(getattr(call, "tool_name", ""))
            arguments = getattr(call, "parameters", None) or {}
            key = _signature(name, arguments)
            note_key = "dup:" + key
            failure = scan.failed_result.get(key)
            stuck = failure is not None and not _write_after(scan, failure[0])
            if scan.notes.get(note_key, 0) >= 1 and not stuck:
                kept.append(call)
                continue

            note = None
            if name in _READ_TOOLS:
                previous = scan.read_result.get(key)
                if previous is not None and not _write_after(scan, previous[0]):
                    note = (
                        _NOTE_MARK
                        + " "
                        + note_key
                        + "\nThis exact call was already made in this conversation and its result was:\n"
                        + previous[1][:1200]
                        + "\nThe data is unchanged, so the call was not executed again. Read your own "
                        "history, use this result and take the next concrete step of the request."
                    )
            if note is None and stuck:
                note = (
                    _NOTE_MARK
                    + " "
                    + note_key
                    + "\nThis exact call already failed in this conversation:\n"
                    + failure[1][:600]
                    + "\nSending it again with identical arguments cannot succeed, and it will be "
                    "refused every time you send it. Change the arguments using the latest tool "
                    "results, use a different tool, or tell the user clearly what cannot be done."
                )
            if note is None and name in _ONCE_WRITE_TOOLS:
                order = _order_key(arguments)
                previous = scan.order_write.get(order) if order else None
                if previous is not None and (
                    previous[0] in _ORDER_CONSUMING or name == previous[0]
                ):
                    blocked_key = "once:" + order
                    if scan.notes.get(blocked_key, 0) < 1:
                        verb = _WRITE_VERB.get(previous[0], "changed")
                        status = (
                            " and the environment now reports its status as '" + previous[1] + "'"
                            if previous[1]
                            else ""
                        )
                        note = (
                            _NOTE_MARK
                            + " "
                            + blocked_key
                            + "\nOrder "
                            + order
                            + " was already "
                            + verb
                            + " in this conversation"
                            + status
                            + ". An order accepts ONE write in its lifetime - one return or exchange "
                            "for a delivered order, one item modification for a pending order, and "
                            "nothing at all once it is cancelled - so this call cannot succeed and was "
                            "NOT sent. Do not retry it: tell the customer plainly that what was already "
                            "done cannot be changed, undone or extended, and use the rest of your turn "
                            "for the rest of their request."
                        )
            if note is None:
                kept.append(call)
                continue
            dropped.add(str(getattr(call, "tool_call_id", "")))
            notes.append(note)

        if not dropped:
            return HookResult.no_changes()

        try:
            parsed.tool_calls = kept
        except Exception:  # pragma: no cover - defensive
            return HookResult.no_changes()

        messages = _strip_calls(hook_input.messages, dropped)
        messages.append(
            _framework_note(
                str(len(dropped))
                + " redundant tool call(s) were NOT executed.\n"
                + "\n\n".join(notes)
            )
        )
        return HookResult.with_modifications(
            parsed_response=parsed,
            messages=messages,
            force_continue=True,
        )

    # ---------------------------------------------------------------- turn end

    def _continue_turn(self, hook_input: AfterModelHookInput, scan: _Scan) -> HookResult:
        total_notes = sum(scan.notes.values())
        if total_notes >= _MAX_NOTES:
            return HookResult.no_changes()
        text = hook_input.original_response or ""
        if not text.strip():
            return HookResult.no_changes()

        note = None

        claim = self._claim_note(text, scan, frozenset())
        if claim is not None:
            note = claim

        if note is None:
            approval = (scan.last_user_text or "").strip()
            if (
                scan.confirm_asks >= 2
                and approval
                and len(approval.split()) <= 8
                and _AFFIRM.match(approval)
                and _CONFIRM_ASK.search(text)
            ):
                key = "confirm"
                if scan.notes.get(key, 0) < 1:
                    note = (
                        _NOTE_MARK
                        + " "
                        + key
                        + "\nThe user just answered \""
                        + approval[:80]
                        + "\" to your confirmation request and you are asking for confirmation "
                        "again. If the action you described is the one the user approved, do not "
                        "ask a second time - call that tool now and report its result. Only ask "
                        "again if the user's approval clearly did not cover this action."
                    )

        if (
            note is None
            and scan.auth_user_id
            and not scan.profile_read
            and _ASK_ID.search(text)
        ):
            key = "ask:profile"
            if scan.notes.get(key, 0) < 1:
                note = (
                    _NOTE_MARK
                    + " "
                    + key
                    + "\nThe customer is already identified as "
                    + scan.auth_user_id
                    + ", so do not ask for an order id: `get_user_details` with user_id=\""
                    + scan.auth_user_id
                    + "\" returns this customer's orders (with their order ids, statuses and "
                    "items) and their payment methods. Call it now and take the order from its "
                    "result. Only ask the customer about something no tool can answer."
                )

        if note is None:
            return HookResult.no_changes()

        messages = list(hook_input.messages or [])
        messages.append(_framework_note(note))
        return HookResult.with_modifications(messages=messages, force_continue=True)

    # ------------------------------------------------------------------ tools

    def after_tool(self, hook_input: AfterToolHookInput) -> HookResult:
        output = hook_input.tool_output
        error = _first_error(output)
        if not error:
            return HookResult.no_changes()
        if _NOTE_MARK in error:
            return HookResult.no_changes()
        decorated = _decorate(
            output,
            _NOTE_MARK + " This call failed. An identical call cannot succeed: rebuild the arguments "
            "from the latest tool results, or explain what cannot be done.",
        )
        if decorated is output:
            return HookResult.no_changes()
        return HookResult.with_modifications(tool_output=decorated)


def _first_error(value: Any) -> str:
    if isinstance(value, str):
        return value if _is_error(value) else ""
    if isinstance(value, dict):
        for item in value.values():
            found = _first_error(item)
            if found:
                return found
        return ""
    if isinstance(value, list):
        for item in value:
            found = _first_error(item)
            if found:
                return found
        return ""
    return ""


def _decorate(value: Any, hint: str) -> Any:
    if isinstance(value, str):
        if _is_error(value):
            return value + "\n" + hint
        return value
    if isinstance(value, dict):
        return {key: _decorate(item, hint) for key, item in value.items()}
    if isinstance(value, list):
        return [_decorate(item, hint) for item in value]
    return value
