"""Deterministic guard that keeps the telecom support agent on the rails.

The first two evaluations of this environment exposed four failure modes that a
small (4B) model falls into very reliably.  None of them need an LLM to detect;
they are all visible in the plain conversation and tool traffic, so this
middleware fixes them with episode-local bookkeeping only:

1. *Hallucinated / customer-side tool calls.*  The policy describes many actions
   performed by the CUSTOMER on their own phone (``toggle_data``,
   ``reset_apn_settings``, ...).  The model sometimes emits them as tool calls;
   each one fails and burns a turn.
2. *Wrong / invented IDs.*  The model picks the first line of the account, or
   invents IDs such as ``C123456789``.
3. *Degenerate repetition.*  The model re-issues the exact same tool call dozens
   of times in a row (one observed run: 95 identical ``get_data_usage`` calls ->
   ``MAX_STEPS``), or repeats the exact same paragraph of instructions to the
   customer several turns in a row.  The simulated customer gets frustrated and
   stops, or the agent burns its whole iteration budget.
4. *Incomplete coverage / premature transfer.*  The model fixes one or two
   causes and then transfers the call while several checklist items (Wi-Fi
   Calling, messaging-app permissions, SIM seating, ...) or account-side fixes
   (refueling data, enabling roaming, paying an overdue bill) are still
   outstanding.

All logic is pure, synchronous, bounded and defensive: the guard only rewrites
the pending action or the system message and can never execute an action of its
own.  No LLM, network, filesystem or dynamic imports are used.
"""

from __future__ import annotations

import json
import re

from nexau.archs.main_sub.execution.hooks import HookResult, Middleware
from nexau.core.messages import Message, Role, TextBlock, ToolUseBlock


AGENT_TOOLS = frozenset({
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
})

# Tools that act on a specific line; a wrong line_id there silently fails the whole task.
LINE_TOOLS = frozenset({
    "get_data_usage",
    "refuel_data",
    "enable_roaming",
    "disable_roaming",
    "resume_line",
    "suspend_line",
})

# Tools that act on an account and therefore require the customer to be identified first.
ACCOUNT_TOOLS = frozenset({
    "get_details_by_id",
    "suspend_line",
    "resume_line",
    "get_bills_for_customer",
    "send_payment_request",
    "get_data_usage",
    "enable_roaming",
    "disable_roaming",
    "refuel_data",
})

# Tools that only REPORT account/line state.  Repeating one of these with the same arguments is
# only meaningful when the underlying state may have changed in between (a customer paying a
# bill, a resume, a refuel, ...), so their repetition counters are reset by every successful
# state-changing action (see WRITE_TOOLS below).
READ_TOOLS = frozenset({
    "get_customer_by_phone",
    "get_customer_by_id",
    "get_customer_by_name",
    "get_details_by_id",
    "get_bills_for_customer",
    "get_data_usage",
})

# Tools that CHANGE account/line state.  After any of them runs, previously-identical reads are
# informative again: the mandatory flow "send_payment_request -> get_bills_for_customer (verify
# 'Paid') -> resume_line" needs the bill to be re-read, and the loop breaker must not deadlock it.
WRITE_TOOLS = frozenset({
    "suspend_line",
    "resume_line",
    "send_payment_request",
    "refuel_data",
    "enable_roaming",
    "disable_roaming",
})

_CORRECTION_HEAD = (
    "STOP - the tool call(s) in your previous message used names that do not exist for you: "
)
_CORRECTION_BODY = (
    ". Those are actions the CUSTOMER performs on their own phone, not tools you can call. "
    "Never emit a tool call for them again. Respond now with a normal message: either use one "
    "of your 13 real tools, or ask the customer in plain language to perform ONE step on their "
    "phone and to report what they see. If you do not yet know their phone number, ask for it first."
)

_LOOP_HEAD = (
    "[system note - the customer cannot see this] You have already run this exact tool call "
    "several times with these exact arguments and repeating it changes nothing: "
)
_LOOP_BODY = (
    ". Do NOT call it again with the same arguments unless the account state has changed since "
    "(for example the customer has just paid a bill, or you have just changed or resumed a "
    "line) - in that case re-reading it is exactly right. Otherwise use the result you already "
    "have and take a DIFFERENT next step (ask the customer about a checklist item you have not "
    "covered yet)."
)

_REPEAT_HEAD = (
    "[system note - the customer cannot see this] Your last message repeats, almost word for "
    "word, an instruction you already gave in this conversation. The customer has already "
    "answered it and is frustrated by the repetition. Do NOT resend it."
)
_REPEAT_TAIL = (
    " Reply now with ONE short message that moves forward: ask about the next item from the "
    "'NOT covered yet' list, or perform the next account action listed."
)
_REPEAT_IDLE_TAIL = (
    " Reply now with ONE short message that moves forward: summarise what has been tried, confirm "
    "whether the customer's problem is now fixed, or check the customer's line details for any "
    "account action you have not performed yet."
)

_TRANSFER_HEAD = (
    "[system note - the customer cannot see this] Do NOT transfer yet: the issue is not "
    "resolved and these fixable items are still outstanding: "
)
_TRANSFER_TAIL = (
    ". Fix them first (ask the customer to perform the device step, or call the account tool "
    "yourself). Only transfer after every applicable fix has been attempted."
)

_TRANSFER_EARLY_HEAD = (
    "[system note - the customer cannot see this] STOP - you may NOT transfer this call yet. "
    "You have not finished helping the customer. A human agent is a LAST resort: with your 13 "
    "tools you can resolve almost everything, and transferring a customer whose problem you "
    "have not even looked at loses the case. "
)
_TRANSFER_EARLY_BODY = (
    " Your next message must NOT be a transfer. Do exactly this instead: "
)

_TRANSFER_UNKNOWN_CUSTOMER = (
    "ask the customer for their phone number, then call get_customer_by_phone with it."
)

_LINE_WRONG_HEAD = (
    "[system note - the customer cannot see this] STOP - that line_id is NOT the customer's "
    "own line, so that call would change the wrong line and fail the case. "
)
_LINE_UNKNOWN_TAIL = (
    " First work out which line belongs to the customer: call get_details_by_id once for each "
    "of the customer's line_ids and find the one whose phone_number equals the phone number the "
    "customer gave. Then retry the call with THAT line_id."
)

_ACCOUNT_NEEDS_ID_HEAD = (
    "[system note - the customer cannot see this] STOP - you tried to act on an account you "
    "have not identified yet. You cannot use any customer_id you did not receive from a tool. "
)
_ACCOUNT_NEEDS_ID_TAIL = (
    " Ask the customer for their phone number, call get_customer_by_phone with it, and copy "
    "the returned customer_id and line_ids exactly. Then retry."
)

_PAYMENT_HEAD = (
    "[system note - the customer cannot see this] STOP - this bill is not paid yet, so you "
    "cannot resume the line and you cannot tell the customer the bill is paid. "
)
_PAYMENT_TAIL = (
    " First check the bill yourself: call get_bills_for_customer and look at the status. If it "
    "still shows Overdue / Awaiting payment, tell the customer a payment request has been sent "
    "and ask them to ACCEPT it in their app - do NOT ask them whether the bill is paid, they "
    "cannot see the bill, only you can. As soon as the status reads 'Paid' and the customer's "
    "own line is Suspended, call resume_line for that line. Never ask the customer to confirm a "
    "bill status and never send the same request twice."
)

_PAYMENT_NOTE = (
    "# PAYMENT IN PROGRESS (do not skip this)\n"
    "A payment request has been sent. Tell the customer that the payment request has been sent "
    "and ask them to open their payment requests and ACCEPT it. The MOMENT the customer says "
    "they have accepted/paid - and any time the bill status is unclear - call "
    "get_bills_for_customer YOURSELF and read the status; the customer cannot see the bill, only "
    "you can. You may call get_bills_for_customer again after a payment; it is not a repeat. As "
    "soon as it shows 'Paid' and the customer's own line is Suspended, call resume_line for that "
    "line. Never ask the customer whether the bill is paid, never ask them to check their bills, "
    "and never say the bill is paid until a tool result shows 'Paid'."
)

_CONTRACT_HEAD = (
    "[system note - the customer cannot see this] STOP - this line's contract end date is in "
    "the past, so the policy does NOT allow you to resume it. Calling resume_line here is "
    "wrong. "
)
_CONTRACT_TAIL = (
    "Finish the other device-side fixes with the customer (airplane mode, SIM, APN), then "
    "explain that a line suspended because its contract has ended cannot be restored with your "
    "tools and transfer the customer to a human agent."
)

_LOOKUP_PHONE_HEAD = (
    "[system note - the customer cannot see this] STOP - the phone number in that call is not a "
    "number the customer gave you, so the lookup cannot succeed and no account will be found. "
)
_LOOKUP_PHONE_TAIL = (
    " Call get_customer_by_phone with EXACTLY the number the customer gave you, character for "
    "character - do not strip dashes, spaces or brackets, and never invent a number."
)

_LOOKUP_NAME_HEAD = (
    "[system note - the customer cannot see this] STOP - the customer never gave you that name "
    "or date of birth, so get_customer_by_name will not find them. "
)
_LOOKUP_NAME_TAIL = (
    " Do not guess identifiers. Look the account up with get_customer_by_phone using the exact "
    "phone number the customer gave you."
)

_REFUEL_HEAD = (
    "[system note - the customer cannot see this] STOP - the customer's line has NOT exceeded "
    "its data allowance, so refuelling it is wrong and would add an unnecessary charge. "
)
_REFUEL_TAIL = (
    " Do NOT call refuel_data. Move on to the next checklist item or account fix."
)

_RESUME_HEAD = (
    "[system note - the customer cannot see this] STOP - that line is not Suspended, so "
    "resume_line will fail. "
)
_RESUME_TAIL = (
    " Only a Suspended line can be resumed. Check the line's status with get_details_by_id and "
    "move on to the next outstanding item instead."
)

_LOCKED_SIM_NOTE = (
    "# REQUIRED ESCALATION (locked SIM)\n"
    "The customer has reported that their SIM is locked with a PIN/PUK. You cannot unlock a SIM "
    "with your tools - no PIN or PUK exists in your toolset. Finish any remaining device steps "
    "(airplane mode, APN) and account fixes, then call transfer_to_human_agents so technical "
    "support can unlock the SIM. Do not keep asking the customer for PINs or PUKs."
)

# Recent-message nudges. These are appended to the conversation (unlike the far-away system
# note) so the model's next turn sees them last. They stop the agent from merely TALKING about
# an account action or an escalation that it has not actually performed.
_ESCALATE_NUDGE = (
    "[system note - the customer cannot see this] You have just told the customer you are "
    "transferring the call, but you have NOT actually transferred it - only the tool call "
    "transfer_to_human_agents transfers a call, a sentence in a message does nothing and the "
    "case is lost. Your NEXT action must be the tool call transfer_to_human_agents with a "
    "one-line summary of the problem. Only AFTER that call succeeds, send the message "
    "'YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON.'"
)

_VERIFY_PAYMENT_NUDGE = (
    "[system note - the customer cannot see this] The customer says the payment is done, but no "
    "tool result has shown the bill as 'Paid' yet. Do NOT tell the customer the bill is paid, do "
    "NOT ask the customer whether the bill is paid or to check their bills - they cannot see it, "
    "only you can - and do NOT move on to another troubleshooting step. Your NEXT action must be "
    "the tool call get_bills_for_customer(customer_id) to check the bill status; this repeat is "
    "expected and allowed, the bill changes when the customer pays. If it shows 'Paid' and the "
    "customer's own line is Suspended, call resume_line(customer_id, line_id) for that line and "
    "then tell the customer to reboot."
)

_RESUME_LINE_NUDGE = (
    "[system note - the customer cannot see this] The bill is paid but the customer's own line "
    "is still Suspended, so the customer still has NO service. Do NOT move on to another step "
    "and do NOT tell the customer the line is active until you have called the tool. Your NEXT "
    "action must be the tool call resume_line(customer_id, line_id) for the customer's own line. "
    "Then tell the customer to reboot the phone."
)

# A message in which the agent announces an escalation but may not have made the tool call.
_TRANSFER_ANNOUNCE_RE = re.compile(
    r"(transfer(?:ring|red)?\s+(?:you|them|the customer|this call)?\s*(?:now\s*)?to\s+(?:a\s+)?human"
    r"|\bbeing transferred to a human\b|\bplease hold on\b)",
    re.IGNORECASE,
)

# The customer telling us the bill payment was accepted / made.
_PAYMENT_CONFIRM_RE = re.compile(
    r"(i(?:'ve| have| just| already)?\s*(?:made|completed|paid|accepted|sent|did)\b"
    r"|payment\s+(?:of\s+[\d.,]+\s*(?:usd|dollars?)?\s*)?(?:is\s+|has\s+|was\s+|had\s+)?"
    r"(?:been\s+)?(?:made|complete|completed|done|accepted|processed|sent|received|cleared)\b"
    r"|(?:already\s+)?paid\s+(?:the\s+|my\s+)?(?:bill|payment|it)\b"
    r"|bill[^.\n]{0,40}\b(?:has\s+been\s+|is\s+|was\s+|been\s+)?paid\b"
    r"|confirmation[^.\n]{0,60}(?:payment|paid)"
    r"|nothing\s+(?:left\s+)?pending)",
    re.IGNORECASE,
)


def _canon_key(name, params):
    try:
        return name + "|" + json.dumps(params, sort_keys=True, default=str)
    except Exception:
        return name + "|" + str(params)


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("true", "1", "yes", "on"):
            return True
        if text in ("false", "0", "no", "off"):
            return False
        return None
    if isinstance(value, (int, float)):
        return bool(value)
    return None


def _digits(value):
    """Digits-only form of a value (used to compare identifiers ignoring punctuation)."""
    return re.sub(r"[^0-9]", "", str(value or ""))


# A phone-looking run: at least 6 characters with a digit at each end.
_PHONE_CAND_RE = re.compile(r"\+?\d[\d\-\s\(\)\.]{5,}\d")


def _phone_candidates(text):
    """Raw substrings inside `text` that look like a phone number the customer gave."""
    found = []
    for match in _PHONE_CAND_RE.finditer(text or ""):
        raw = match.group(0).strip()
        raw = raw.rstrip(".,;:")
        digits = _digits(raw)
        # A real phone number carries 10-13 digits; shorter runs are dates or amounts.
        if 10 <= len(digits) <= 13:
            found.append(raw)
    return found


def _same_phone(digits_a, digits_b):
    """True when two digit strings denote the same number (equal, or suffix/country-code match)."""
    if not digits_a or not digits_b:
        return False
    if digits_a == digits_b:
        return True
    short, long_ = (digits_a, digits_b) if len(digits_a) <= len(digits_b) else (digits_b, digits_a)
    return len(short) >= 10 and long_.endswith(short)


def _real_user_texts(messages):
    """Plain text of the customer's own turns, ignoring synthetic guard notes."""
    texts = []
    for message in messages:
        if message.role != Role.USER:
            continue
        try:
            text = message.get_text_content()
        except Exception:
            continue
        if text and not _is_synthetic_note(text):
            texts.append(text)
    return texts


# Locked-SIM detection: both patterns must hit the SAME sentence.
_SIM_WORD_RE = re.compile(r"\bsim\b")
# Must also match the literal enum strings the environment/customer uses ("locked_pin"/"locked_puk"),
# where the underscore defeats a trailing \b.
_SIM_LOCK_WORD_RE = re.compile(r"\b(?:lock|locked|unlock|unlocked|pin|puk)\b|\blocked_?(?:pin|puk)\b")


def _iter_dicts(value, depth=0):
    """Yield every dict found inside a (possibly nested / JSON-encoded) value."""
    if depth > 8:
        return
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from _iter_dicts(item, depth + 1)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_dicts(item, depth + 1)
    elif isinstance(value, str):
        stripped = value.strip()
        if len(value) > 20000 or not (stripped.startswith("{") or stripped.startswith("[")):
            return
        try:
            decoded = json.loads(stripped)
        except Exception:
            return
        yield from _iter_dicts(decoded, depth + 1)


def _find_record(value, depth=0):
    """Return the first telecom record dict found inside a (possibly nested) tool output."""
    for record in _iter_dicts(value, depth):
        if "line_ids" in record or "line_id" in record or "customer_id" in record:
            return record
    return None


def _is_synthetic_note(text):
    return text.startswith("[system note") or text.startswith("STOP - the tool call(s)")


def _is_synthetic_note_message(message):
    """True when a Message is one of the guard's own injected corrective notes."""
    if message.role != Role.USER:
        return False
    try:
        text = message.get_text_content()
    except Exception:
        return False
    return bool(text) and _is_synthetic_note(text)


def _convo_text(messages):
    parts = []
    for message in messages:
        if message.role not in (Role.USER, Role.ASSISTANT):
            continue
        try:
            text = message.get_text_content()
        except Exception:
            continue
        if not text or _is_synthetic_note(text):
            continue
        parts.append(text)
    return "\n".join(parts).lower()


def _norm_tokens(text):
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _jaccard(tokens_a, tokens_b):
    set_a, set_b = set(tokens_a), set(tokens_b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


# --------------------------------------------------------------------------- #
# Symptom-specific, policy-derived troubleshooting checklists.
# Each entry is (id, human readable instruction, keyword tuple used to detect
# that the topic has already been raised in the conversation).
# --------------------------------------------------------------------------- #
_ITEM_AIRPLANE = ("airplane", "Airplane Mode is OFF (ask the customer to check the status bar)",
                  ("airplane",))
_ITEM_SIM = ("sim", "the SIM card is seated / not missing (ask the customer to reseat it if needed)",
             ("sim card", "sim status", "reseat", " sim "))
_ITEM_NETMODE = ("network mode", "the network mode preference is 4G/5G (ask the customer to switch "
                 "it to '4G/5G preferred')", ("network mode", "network preference", "2g", "3g", "4g", "5g"))
_ITEM_APN = ("APN/MMSC", "the APN / MMSC settings are correct (ask the customer to check the MMSC URL; "
            "reset the APN settings and reboot if it is wrong)", ("apn", "mmsc"))
_ITEM_WIFI_CALLING = ("Wi-Fi Calling", "Wi-Fi Calling is OFF (ask the customer to check Wi-Fi Calling "
                      "and turn it off if it is on)", ("wifi calling", "wi-fi calling"))
_ITEM_PERMISSIONS = ("app permissions", "the messaging app has BOTH storage and SMS permissions "
                     "(ask the customer to check and grant any missing permission)",
                     ("permission", "storage", "sms"))
_ITEM_ROAMING = ("data roaming",
                 "the customer's data roaming - ASK THE CUSTOMER DIRECTLY whether they are "
                 "currently abroad / outside their home network. If they are, their phone's "
                 "data roaming must be ON and you must call enable_roaming for their line",
                 ("roaming", "abroad", "overseas", "outside their home"))
_ITEM_MOBILE_DATA = ("mobile data", "mobile data is ON (ask the customer to switch it on)",
                     ("mobile data", "data enabled", "data is enabled", "data switch", "turn data"))
_ITEM_DATA_SAVER = ("Data Saver", "Data Saver mode is OFF (ask the customer to turn it off)",
                    ("data saver", "data saving", "data restriction"))
_ITEM_VPN = ("VPN", "any VPN is disconnected (ask the customer to disconnect it)", ("vpn",))
_ITEM_BILLING = ("billing", "the billing / suspension state (check overdue bills and resume a "
                 "suspended line once its bill is paid)", ("bill", "suspend", "overdue", "payment", "contract"))

_MMS_CHECKLIST = [_ITEM_AIRPLANE, _ITEM_SIM, _ITEM_NETMODE, _ITEM_MOBILE_DATA, _ITEM_APN,
                  _ITEM_WIFI_CALLING, _ITEM_PERMISSIONS, _ITEM_ROAMING]
_DATA_CHECKLIST = [_ITEM_AIRPLANE, _ITEM_NETMODE, _ITEM_MOBILE_DATA,
                   _ITEM_ROAMING, _ITEM_DATA_SAVER, _ITEM_VPN]
_SERVICE_CHECKLIST = [_ITEM_AIRPLANE, _ITEM_SIM, _ITEM_NETMODE, _ITEM_APN, _ITEM_BILLING]

# Items that describe the CUSTOMER's situation; only the customer's own words can settle them,
# so the guard must not treat the agent merely mentioning the topic as "already covered".
_USER_REPORT_ITEMS = frozenset({"data roaming"})

_MMS_KEYWORDS = ("mms", "picture message", "picture msg", "picture", "photo", "multimedia",
                 "send a picture", "send picture", "video message")
_DATA_KEYWORDS = ("mobile data", "internet", "data speed", "slow", "browsing", "browse",
                  "web page", "website", "stream")
_SERVICE_KEYWORDS = ("no service", "no signal", "can't call", "cannot call", "cant call",
                     "cell service", "cellular service", "make calls", "phone service")

_ABROAD_RE = re.compile(
    r"\b(abroad|overseas|travell?ing|travel|roaming|international|vacation|holiday|"
    r"outside (?:the|my) (?:country|network|home)|different country)\b",
    re.IGNORECASE,
)


class ProtocolGuardMiddleware(Middleware):
    """Keeps the agent inside its real toolset, on the customer's line, and moving forward."""

    def __init__(
        self,
        max_repairs=3,
        max_same_call=2,
        max_loop_breaks=40,
        max_repeat_breaks=4,
        max_transfer_blocks=8,
        max_line_blocks=4,
        max_account_blocks=8,
        max_lookup_blocks=6,
        max_refuel_blocks=3,
        max_resume_blocks=3,
        max_action_nudges=4,
        max_identity_nudges=2,
        min_turns_before_transfer=6,
        repeat_threshold=0.82,
    ):
        self.max_repairs = max_repairs
        self.max_same_call = max_same_call
        self.max_loop_breaks = max_loop_breaks
        self.max_repeat_breaks = max_repeat_breaks
        self.max_transfer_blocks = max_transfer_blocks
        self.max_line_blocks = max_line_blocks
        self.max_account_blocks = max_account_blocks
        self.max_lookup_blocks = max_lookup_blocks
        self.max_refuel_blocks = max_refuel_blocks
        self.max_resume_blocks = max_resume_blocks
        self.max_action_nudges = max_action_nudges
        self.max_identity_nudges = max_identity_nudges
        self.min_turns_before_transfer = min_turns_before_transfer
        self.repeat_threshold = repeat_threshold

        self.repairs = 0
        self.loop_breaks = 0
        self.repeat_breaks = 0
        self.transfer_blocks = 0
        self.line_blocks = 0
        self.account_blocks = 0
        self.lookup_blocks = 0
        self.refuel_blocks = 0
        self.resume_blocks = 0
        self.action_nudges = 0
        self.identity_nudges = 0

        self.base_system = None
        self.policy_date = None

        self.customer_id = ""
        self.customer_phone = ""
        self.query_phone = ""
        self.line_ids = []
        self.line_phones = {}
        self.line_roaming = {}
        self.line_status = {}
        self.line_contract_end = {}
        self.data_over = {}

        self.call_counts = {}
        self.called_tools = set()
        self.overdue_bill = False
        self.abroad = False

        # Values the CUSTOMER actually supplied (leftovers of invented arguments are rejected).
        self.user_texts = []
        self.user_phones = []

        # Payment flow (overdue-bill suspension).
        self.payment_requested = False
        self.bill_id = ""
        self.bill_unpaid = False

    # ------------------------------------------------------------------ model
    def before_model(self, hook_input):
        try:
            self._refresh_user_facts(hook_input.messages)
            self._refresh_abroad(hook_input.messages)
            return self._inject_state(hook_input)
        except Exception:
            return HookResult.no_changes()

    def after_model(self, hook_input):
        try:
            self._refresh_user_facts(hook_input.messages)
            return self._repair(hook_input)
        except Exception:
            return HookResult.no_changes()

    # ------------------------------------------------------------------- tool
    def after_tool(self, hook_input):
        try:
            self._observe(hook_input)
        except Exception:
            pass
        return HookResult.no_changes()

    def before_tool(self, hook_input):
        try:
            return self._repair_line(hook_input)
        except Exception:
            return HookResult.no_changes()

    # -------------------------------------------------------------- internals
    def _observe(self, hook_input):
        name = hook_input.tool_name
        self.called_tools.add(name)
        key = _canon_key(name, hook_input.tool_input)
        self.call_counts[key] = self.call_counts.get(key, 0) + 1

        # A state-changing action invalidates the "nothing has changed, stop repeating this read"
        # reasoning for every state-reading tool, so forget those repetition counts. Without this
        # the loop breaker blocked the very get_bills_for_customer call that must verify a payment,
        # while resume_line was itself blocked until the bill reads 'Paid' - a hard deadlock.
        if name in WRITE_TOOLS:
            for stale in [k for k in self.call_counts if k.split("|", 1)[0] in READ_TOOLS]:
                self.call_counts.pop(stale, None)

        if name == "get_customer_by_phone":
            wanted = hook_input.tool_input.get("phone_number")
            if isinstance(wanted, str) and wanted:
                self.query_phone = wanted

        output = hook_input.tool_output
        if name == "get_bills_for_customer":
            statuses = [
                str(entry.get("status", "")).strip().lower()
                for entry in _iter_dicts(output)
                if "bill_id" in entry
            ]
            if statuses:
                self.overdue_bill = "overdue" in statuses
                self.bill_unpaid = any(
                    status in ("overdue", "awaiting payment") for status in statuses
                )
        if name == "send_payment_request":
            self.payment_requested = True
            self.bill_unpaid = True
            wanted = hook_input.tool_input.get("bill_id")
            if isinstance(wanted, str) and wanted:
                self.bill_id = wanted

        record = _find_record(output)
        if record is not None:
            if isinstance(record.get("customer_id"), str):
                self.customer_id = record["customer_id"]
            if isinstance(record.get("phone_number"), str) and "line_id" not in record:
                self.customer_phone = record["phone_number"]
            lines = record.get("line_ids")
            if isinstance(lines, list):
                self.line_ids = [item for item in lines if isinstance(item, str)]
            line_id = record.get("line_id")
            if isinstance(line_id, str) and isinstance(record.get("phone_number"), str):
                self.line_phones[line_id] = record["phone_number"]

        for entry in _iter_dicts(output):
            if isinstance(entry.get("customer_id"), str):
                self.customer_id = entry["customer_id"]
            if isinstance(entry.get("line_ids"), list):
                self.line_ids = [item for item in entry["line_ids"] if isinstance(item, str)]
            lid = entry.get("line_id")
            if isinstance(lid, str):
                phone = entry.get("phone_number")
                if isinstance(phone, str):
                    self.line_phones[lid] = phone
                if "roaming_enabled" in entry:
                    roaming = _as_bool(entry["roaming_enabled"])
                    if roaming is not None:
                        self.line_roaming[lid] = roaming
                if isinstance(entry.get("status"), str):
                    self.line_status[lid] = entry["status"]
                if isinstance(entry.get("contract_end_date"), str):
                    self.line_contract_end[lid] = entry["contract_end_date"]
                if "data_used_gb" in entry and "data_limit_gb" in entry:
                    try:
                        used = float(entry["data_used_gb"])
                        limit = float(entry["data_limit_gb"])
                        refuel = float(entry.get("data_refueling_gb") or 0.0)
                        self.data_over[lid] = used >= limit + refuel
                    except Exception:
                        pass

        # A completed account fix must not be masked by a stale value in the raw output.
        acted_line = hook_input.tool_input.get("line_id")
        if isinstance(acted_line, str) and acted_line:
            if name == "resume_line":
                self.line_status[acted_line] = "Active"
            elif name == "suspend_line":
                self.line_status[acted_line] = "Suspended"
            elif name == "enable_roaming":
                self.line_roaming[acted_line] = True
            elif name == "disable_roaming":
                self.line_roaming[acted_line] = False
            elif name == "refuel_data":
                self.data_over[acted_line] = False

    def _refresh_user_facts(self, messages):
        """Remember what the customer actually said, so invented arguments can be rejected."""
        texts = _real_user_texts(messages)
        self.user_texts = texts
        phones = []
        for text in texts:
            for candidate in _phone_candidates(text):
                if candidate not in phones:
                    phones.append(candidate)
        self.user_phones = phones

    def _refresh_abroad(self, messages):
        for message in messages:
            if message.role != Role.USER:
                continue
            try:
                text = message.get_text_content()
            except Exception:
                continue
            if text and _ABROAD_RE.search(text):
                self.abroad = True
                return

    def _customer_line(self):
        """The customer's own line: the one whose phone number we were asked about."""
        for wanted in (self.query_phone, self.customer_phone):
            if not wanted:
                continue
            for line_id in self.line_ids or list(self.line_phones):
                if self.line_phones.get(line_id) == wanted:
                    return line_id
        return ""

    def _symptom(self, messages):
        text = _convo_text(messages)
        if any(k in text for k in _MMS_KEYWORDS):
            return "mms"
        if any(k in text for k in _SERVICE_KEYWORDS):
            return "service"
        if any(k in text for k in _DATA_KEYWORDS):
            return "mobile_data"
        return ""

    def _checklist(self, messages):
        symptom = self._symptom(messages)
        if symptom == "mms":
            items = list(_MMS_CHECKLIST)
        elif symptom == "mobile_data":
            items = list(_DATA_CHECKLIST)
        elif symptom == "service":
            items = list(_SERVICE_CHECKLIST)
        else:
            items = []

        text = _convo_text(messages)
        user_text = "\n".join(_real_user_texts(messages)).lower()
        covered, uncovered = [], []
        for item_id, human, keywords in items:
            source = user_text if item_id in _USER_REPORT_ITEMS else text
            if any(k in source for k in keywords):
                covered.append(item_id)
            else:
                uncovered.append((item_id, human))
        return symptom, covered, uncovered

    def _pending_actions(self):
        line = self._customer_line()
        pending = []
        if line:
            if (self.abroad and self.line_roaming.get(line) is False
                    and "enable_roaming" not in self.called_tools):
                pending.append(
                    "the customer is abroad but their line " + line + " is NOT roaming-enabled: "
                    "call enable_roaming(customer_id, line_id) for that line (free of charge)"
                )
            if self.data_over.get(line) and "refuel_data" not in self.called_tools:
                pending.append(
                    "line " + line + " has exceeded its data allowance: once the customer agrees, "
                    "call refuel_data (at most 2 GB) for that line"
                )
            if (self.line_status.get(line) == "Suspended"
                    and "resume_line" not in self.called_tools
                    and not self._contract_ended(line)
                    and (not self.overdue_bill or "send_payment_request" in self.called_tools)):
                pending.append(
                    "line " + line + " is Suspended: once its overdue bill is paid, call "
                    "resume_line for that line"
                )
        if self.overdue_bill and "send_payment_request" not in self.called_tools:
            pending.append(
                "there is an Overdue bill: call send_payment_request for it (then tell the "
                "customer to accept the payment request)"
            )
        return pending

    def _contract_ended(self, line):
        end = self.line_contract_end.get(line)
        if not end or not self.policy_date:
            return False
        try:
            return end[:10] < self.policy_date
        except Exception:
            return False

    # ------------------------------------------------------- transfer gating
    def _user_texts(self, messages):
        texts = []
        for message in messages:
            if message.role == Role.USER:
                try:
                    text = message.get_text_content()
                except Exception:
                    continue
                if text and not _is_synthetic_note(text):
                    texts.append(text)
        return texts

    def _user_reports_locked_sim(self, messages):
        """The customer only finds out their SIM is locked by checking it themselves."""
        for text in self._user_texts(messages):
            for sentence in re.split(r"[.!?\n;]", text.lower()):
                if _SIM_WORD_RE.search(sentence) and _SIM_LOCK_WORD_RE.search(sentence):
                    return True
        return False

    def _transfer_block_reason(self, messages):
        """Return a corrective note when a transfer must not be allowed yet, else ''."""
        locked_sim = self._user_reports_locked_sim(messages)
        line = self._customer_line()
        if not line:
            # The agent has not even identified the customer's line yet.
            return (
                _TRANSFER_EARLY_HEAD
                + "You do not know which customer or line this is."
                + _TRANSFER_EARLY_BODY + _TRANSFER_UNKNOWN_CUSTOMER
            )
        turns = sum(1 for m in messages if m.role == Role.ASSISTANT)
        if turns < self.min_turns_before_transfer:
            return (
                _TRANSFER_EARLY_HEAD
                + "The customer's problem has not been worked through yet."
                + _TRANSFER_EARLY_BODY + "ask the customer about the next item from the "
                "'NOT covered yet' checklist, one step at a time."
            )
        pending = self._pending_actions()
        if pending:
            # Outstanding account fixes always come first - even when the SIM is locked.
            return _TRANSFER_HEAD + "; ".join(pending) + _TRANSFER_TAIL
        if locked_sim:
            # The SIM cannot be unlocked with any tool, so escalation is the correct resolution
            # and the remaining (unanswerable) device checklist items must not hold the call back.
            return ""
        _symptom, _covered, uncovered = self._checklist(messages)
        if uncovered:
            return (
                _TRANSFER_EARLY_HEAD
                + "There are troubleshooting steps you have not raised with the customer yet."
                + _TRANSFER_EARLY_BODY + "ask the customer about the next item from the "
                "'NOT covered yet' checklist (" + "; ".join(h for _i, h in uncovered[:3]) + ")."
            )
        if self.payment_requested and self.bill_unpaid:
            return _TRANSFER_EARLY_HEAD + _PAYMENT_TAIL
        return ""

    # ---------------------------------------------------- action gating helpers
    def _account_block_reason(self, name):
        """Block account/line tools while the customer is still unknown."""
        if name not in ACCOUNT_TOOLS:
            return ""
        if self.customer_id or self.line_ids or self._customer_line():
            return ""
        if self.account_blocks >= self.max_account_blocks:
            return ""
        self.account_blocks += 1
        return _ACCOUNT_NEEDS_ID_HEAD + _ACCOUNT_NEEDS_ID_TAIL

    def _line_block_reason(self, name, params):
        """Block a line-level call aimed at a line we already know is not the customer's."""
        if name not in LINE_TOOLS or not isinstance(params, dict):
            return ""
        line_id = params.get("line_id")
        if not isinstance(line_id, str) or not line_id:
            return ""
        phone = self.query_phone or self.customer_phone
        if not phone:
            return ""
        target = self._customer_line()
        if target:
            # before_tool will rewrite the line_id to the customer's own line.
            return ""
        known = self.line_phones.get(line_id)
        if known is not None and known != phone:
            return _LINE_WRONG_HEAD + "Line " + line_id + " belongs to phone " + known + ", " \
                "but the customer's phone is " + phone + "." + _LINE_UNKNOWN_TAIL
        if self.line_ids:
            # We cannot tell yet which line is the customer's, so verify before changing anything.
            if self.line_blocks < self.max_line_blocks:
                return (
                    "[system note - the customer cannot see this] Do not act on a line before "
                    "you know it is the customer's own line." + _LINE_UNKNOWN_TAIL
                )
        return ""

    def _lookup_block_reason(self, name, params):
        """Reject lookup arguments the customer never supplied (invented phone / name / DOB)."""
        if not isinstance(params, dict) or not self.user_texts:
            return ""
        if self.lookup_blocks >= self.max_lookup_blocks:
            return ""

        if name == "get_customer_by_phone":
            wanted = _digits(params.get("phone_number"))
            if not wanted or not self.user_phones:
                return ""
            if any(_same_phone(_digits(candidate), wanted) for candidate in self.user_phones):
                return ""
            self.lookup_blocks += 1
            exact = self.user_phones[-1]
            return (_LOOKUP_PHONE_HEAD + "The only number the customer gave you is \"" + exact
                    + "\"." + _LOOKUP_PHONE_TAIL)

        if name == "get_customer_by_name":
            blob = "\n".join(self.user_texts).lower()
            full_name = str(params.get("full_name") or "").strip()
            dob = _digits(params.get("dob"))
            name_known = False
            if full_name:
                if full_name.lower() in blob:
                    name_known = True
                else:
                    tokens = set(_norm_tokens(full_name))
                    name_known = bool(tokens) and tokens.issubset(set(_norm_tokens(blob)))
            reason = ""
            if full_name and not name_known:
                reason = "The customer never gave that name."
            elif dob and dob not in _digits(blob):
                reason = "The customer never gave that date of birth."
            if not reason:
                return ""
            self.lookup_blocks += 1
            return _LOOKUP_NAME_HEAD + reason + _LOOKUP_NAME_TAIL

        return ""

    def _refuel_block_reason(self):
        """A line that is provably NOT over its allowance must not be refuelled."""
        line = self._customer_line()
        if not line:
            return ""
        if self.data_over.get(line) is not False:
            return ""
        if self.refuel_blocks >= self.max_refuel_blocks:
            return ""
        self.refuel_blocks += 1
        return _REFUEL_HEAD + _REFUEL_TAIL

    def _payment_block_reason(self, name):
        if name != "resume_line":
            return ""
        line = self._customer_line()
        if line and self._contract_ended(line):
            return _CONTRACT_HEAD + _CONTRACT_TAIL
        if line:
            status = self.line_status.get(line)
            if (status and status != "Suspended"
                    and self.resume_blocks < self.max_resume_blocks):
                self.resume_blocks += 1
                return _RESUME_HEAD + "Line " + line + " is currently '" + status + "'." + _RESUME_TAIL
        if self.payment_requested and self.bill_unpaid:
            return _PAYMENT_HEAD + _PAYMENT_TAIL
        return ""

    def _state_note(self, messages):
        parts = [self._identity_note()]
        checklist = self._coverage_note(messages)
        if checklist:
            parts.append(checklist)
        pending = self._pending_actions()
        if pending:
            parts.append(
                "# ACCOUNT ACTIONS STILL OUTSTANDING (only you can do these)\n"
                + "\n".join("- " + item for item in pending)
            )
        if self.payment_requested and self.bill_unpaid:
            parts.append(_PAYMENT_NOTE)
        own_line = self._customer_line()
        if own_line and self._contract_ended(own_line):
            parts.append(
                "# CONTRACT ENDED\n"
                "Line " + own_line + " is suspended because its contract end date has passed. You "
                "are NOT allowed to resume it. Do not call resume_line. Finish the other fixes, "
                "then transfer the customer to a human agent."
            )
        lock = self.line_status.get(self._customer_line())
        if lock in ("locked_pin", "locked_puk"):
            parts.append(
                "# REQUIRED ESCALATION\n"
                "The customer's line is '" + lock + "'. A locked SIM cannot be unlocked with your "
                "tools. After you have finished every other applicable fix, you MUST call "
                "transfer_to_human_agents so technical support can unlock it."
            )
        elif (self._user_reports_locked_sim(messages)
                and "transfer_to_human_agents" not in self.called_tools):
            parts.append(_LOCKED_SIM_NOTE)
        return "\n\n" + "\n\n".join(parts)

    def _identity_note(self):
        line = self._customer_line()
        if line:
            return (
                "# SESSION STATE\n"
                "Customer " + self.customer_id + " (phone " + (self.query_phone or self.customer_phone) + "). "
                "The customer's own line is " + line + ". Use customer_id " + self.customer_id
                + " and line_id " + line + " for every line-level tool call (data usage, refuel, roaming). "
                "Never invent an ID."
            )
        if self.line_ids:
            return (
                "# SESSION STATE\n"
                "Customer " + self.customer_id + " (phone " + (self.query_phone or self.customer_phone) + "). "
                "Their line_ids are " + ", ".join(self.line_ids) + ". The customer's own line is the one "
                "whose phone_number equals " + (self.query_phone or self.customer_phone) + " - call "
                "get_details_by_id on those line_ids to find it, then use that line_id for every "
                "line-level tool call."
            )
        if self.customer_id or self.customer_phone:
            return (
                "# SESSION STATE\n"
                "You have identified the customer but not their line yet. Work out which line_id belongs "
                "to the phone number the customer gave you before changing anything."
            )
        return (
            "# SESSION STATE\n"
            "You have not identified the customer yet. Ask the customer for their phone number, then "
            "call get_customer_by_phone with it before performing any account action."
        )

    def _coverage_note(self, messages):
        symptom, covered, uncovered = self._checklist(messages)
        if not uncovered:
            return ""
        label = {"mms": "MMS", "mobile_data": "MOBILE DATA", "service": "SERVICE"}.get(symptom, "GENERAL")
        lines = [
            "# TROUBLESHOOTING CHECKLIST (" + label + " issues)",
            "Raise these items with the customer ONE at a time, in order. Never repeat an item the "
            "customer has already answered or refused: if a step is done or refused, mark it done and "
            "move to the NEXT item.",
            "Items you have NOT raised yet:",
        ]
        for _item_id, human in uncovered[:4]:
            lines.append("- " + human)
        if covered:
            lines.append("Already raised: " + ", ".join(covered) + ".")
        return "\n".join(lines)

    def _inject_state(self, hook_input):
        messages = hook_input.messages
        index = -1
        for position in range(len(messages)):
            if messages[position].role == Role.SYSTEM:
                index = position
                break
        if index < 0:
            return HookResult.no_changes()
        if self.base_system is None:
            self.base_system = messages[index].get_text_content()
            match = re.search(r"current time is (\d{4}-\d{2}-\d{2})", self.base_system)
            if match:
                self.policy_date = match.group(1)
        updated = list(messages)
        updated[index] = Message(
            role=Role.SYSTEM,
            content=[TextBlock(text=self.base_system + self._state_note(messages))],
        )
        return HookResult.with_modifications(messages=updated)

    def _repair_line(self, hook_input):
        name = hook_input.tool_name
        params = hook_input.tool_input
        if not isinstance(params, dict):
            return HookResult.no_changes()

        # Forward the customer's phone number verbatim. Stripping dashes/spaces/brackets makes
        # the lookup fail even though the number is correct, so restore the customer's exact form.
        if name == "get_customer_by_phone":
            wanted = params.get("phone_number")
            if isinstance(wanted, str) and self.user_phones:
                digits = _digits(wanted)
                if digits:
                    for candidate in self.user_phones:
                        if _same_phone(_digits(candidate), digits) and candidate != wanted:
                            fixed = dict(params)
                            fixed["phone_number"] = candidate
                            return HookResult.with_modifications(tool_input=fixed)
            return HookResult.no_changes()

        if name not in ACCOUNT_TOOLS:
            return HookResult.no_changes()
        fixed = None

        # Never let an invented customer_id through when we know the real one.
        wanted_customer = params.get("customer_id")
        if (isinstance(wanted_customer, str) and self.customer_id
                and wanted_customer != self.customer_id):
            fixed = dict(fixed or params)
            fixed["customer_id"] = self.customer_id

        if name in LINE_TOOLS:
            line = self._customer_line()
            current = params.get("line_id")
            if (line and isinstance(current, str) and current != line
                    and (not self.line_ids or current in self.line_ids)):
                fixed = dict(fixed or params)
                fixed["line_id"] = line

        if fixed is None:
            return HookResult.no_changes()
        return HookResult.with_modifications(tool_input=fixed)

    # ------------------------------------------------- action-completion nudges
    def _payment_confirmed_by_customer(self, messages):
        """True when the customer's most recent turn says the payment was made/accepted."""
        texts = _real_user_texts(messages)
        if not texts:
            return False
        return bool(_PAYMENT_CONFIRM_RE.search(texts[-1]))

    def _action_nudge(self, hook_input, messages):
        """Force an outstanding account action the agent has only talked about.

        The advisory state note lives in the system prompt, far from the decision point; a small
        model frequently ignores it and either keeps troubleshooting or announces an action it
        never performs. This nudge is appended as the most recent message instead, so the next
        turn is pushed to actually execute the tool call. It is bounded and only fires in states
        where exactly one action is unambiguously owed.
        """
        if self.action_nudges >= self.max_action_nudges:
            return ""
        current = hook_input.original_response or ""

        line = self._customer_line()

        # (A) A locked SIM was reported and the agent announced an escalation without making the
        #     transfer_to_human_agents call - talking about a transfer does not transfer the call.
        lock = self.line_status.get(line) if line else None
        escalate_needed = bool(
            self._user_reports_locked_sim(messages) or lock in ("locked_pin", "locked_puk")
        )
        if ("transfer_to_human_agents" not in self.called_tools
                and _TRANSFER_ANNOUNCE_RE.search(current)):
            block = self._transfer_block_reason(messages)
            if escalate_needed or block:
                self.action_nudges += 1
                if not block:
                    return _ESCALATE_NUDGE
                # Announcing a transfer that is not allowed yet is doubly wrong: the call never
                # happens AND the customer is told something false. Repeat the real next step.
                return (
                    "[system note - the customer cannot see this] You have just told the customer "
                    "you are transferring the call, but you did NOT call transfer_to_human_agents, "
                    "so no transfer happened, and a transfer is not allowed at this point anyway. "
                    + block
                )

        # (B) The payment flow: verify the bill the customer says they paid, then resume the line.
        if self.payment_requested:
            if self.bill_unpaid and self._payment_confirmed_by_customer(messages):
                self.action_nudges += 1
                return _VERIFY_PAYMENT_NUDGE
            if (not self.bill_unpaid and line
                    and self.line_status.get(line) == "Suspended"
                    and "resume_line" not in self.called_tools
                    and not self._contract_ended(line)):
                self.action_nudges += 1
                return _RESUME_LINE_NUDGE

        # (C) The account was never identified and no tool has been called at all: the agent is
        #     chatting its way through a checklist it cannot complete without the account.
        if (not self.called_tools
                and not self.customer_id
                and not self.line_ids
                and self._symptom(messages)
                and self.identity_nudges < self.max_identity_nudges
                and not re.search(r"phone|number", current, re.IGNORECASE)):
            self.identity_nudges += 1
            if self.user_phones:
                return (
                    "[system note - the customer cannot see this] STOP - you have not looked up "
                    "the account yet, so no line-level action can be taken. The customer already "
                    "gave you their phone number. Your NEXT action must be the tool call "
                    "get_customer_by_phone with that exact number, copied character for character."
                )
            return (
                "[system note - the customer cannot see this] STOP - you have not identified the "
                "account yet and no tool call has been made, so you cannot fix anything. You do "
                "not know the customer's phone number. Your NEXT message must ONLY ask the "
                "customer for their phone number - do not describe any troubleshooting step "
                "before that, and do not move on to any other topic."
            )

        return ""

    # ------------------------------------------------------------- after_model
    def _repair(self, hook_input):
        parsed = hook_input.parsed_response
        messages = list(hook_input.messages)

        if parsed is not None and parsed.tool_calls:
            notes = []
            keep, bad_names = self._filter_calls(parsed.tool_calls, notes, messages)
            if len(keep) != len(parsed.tool_calls):
                parsed.tool_calls = keep
                parsed.is_parallel_tools = False
                self._filter_assistant_blocks(messages, keep)
                if keep:
                    return HookResult.with_modifications(messages=messages, parsed_response=parsed)
                if not notes and bad_names:
                    notes = [_CORRECTION_HEAD + ", ".join(sorted(bad_names)) + _CORRECTION_BODY]
                note_text = "\n".join(notes) or (
                    "[system note - the customer cannot see this] STOP - that action cannot be "
                    "performed right now. Take the next available step instead: ask the customer "
                    "about the next unchecked item, or call one of your 13 real tools."
                )
                self._append_note(messages, note_text)
                return HookResult.with_modifications(
                    messages=messages, parsed_response=parsed, force_continue=True,
                )
            return HookResult.no_changes()

        note = self._action_nudge(hook_input, messages)
        if note:
            self._append_note(messages, note)
            return HookResult.with_modifications(messages=messages, force_continue=True)

        note = self._repetition_note(hook_input, messages)
        if note:
            self._append_note(messages, note)
            return HookResult.with_modifications(messages=messages, force_continue=True)

        return HookResult.no_changes()

    def _append_note(self, messages, note):
        """Add a corrective note, replacing any earlier guard note rather than stacking them.

        A note is only ever a pointer to the single next action; older notes are stale and were
        observed to accumulate into dozens of contradictory instructions inside one conversation.
        """
        if not note:
            return
        kept = [
            message for message in messages
            if not _is_synthetic_note_message(message)
        ]
        messages[:] = kept
        messages.append(Message(role=Role.USER, content=[TextBlock(text=note)]))

    def _filter_calls(self, calls, notes, messages):
        keep = []
        bad_names = set()
        for call in calls:
            name = call.tool_name
            if name not in AGENT_TOOLS:
                bad_names.add(name)
                continue
            if name == "transfer_to_human_agents":
                blocked = False
                if self.transfer_blocks < self.max_transfer_blocks:
                    reason = self._transfer_block_reason(messages)
                    if reason:
                        self.transfer_blocks += 1
                        notes.append(reason)
                        blocked = True
                if blocked:
                    continue
            else:
                reason = self._account_block_reason(name)
                if reason:
                    notes.append(reason)
                    continue
                reason = self._lookup_block_reason(name, call.parameters)
                if reason:
                    notes.append(reason)
                    continue
                reason = self._line_block_reason(name, call.parameters)
                if reason:
                    if self.line_blocks < self.max_line_blocks and "[system note" in reason:
                        self.line_blocks += 1
                    notes.append(reason)
                    continue
                if name == "refuel_data":
                    reason = self._refuel_block_reason()
                    if reason:
                        notes.append(reason)
                        continue
                reason = self._payment_block_reason(name)
                if reason:
                    notes.append(reason)
                    continue
            key = _canon_key(name, call.parameters)
            if self.call_counts.get(key, 0) >= self.max_same_call and self.loop_breaks < self.max_loop_breaks:
                self.loop_breaks += 1
                notes.append(_LOOP_HEAD + name + " " + json.dumps(call.parameters, default=str)
                             + _LOOP_BODY)
                continue
            keep.append(call)
        return keep, bad_names

    def _filter_assistant_blocks(self, messages, keep):
        """Drop the tool-use blocks that were removed from the parsed response."""
        keep_ids = {call.tool_call_id for call in keep}
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if message.role != Role.ASSISTANT:
                continue
            kept = []
            for block in message.content:
                if isinstance(block, ToolUseBlock) and block.id not in keep_ids:
                    continue
                kept.append(block)
            if len(kept) != len(message.content):
                if not kept:
                    kept = [TextBlock(text="(that action is not available - picking a different step)")]
                messages[index] = Message(role=Role.ASSISTANT, content=kept)
            return

    def _repetition_note(self, hook_input, messages):
        if self.repeat_breaks >= self.max_repeat_breaks:
            return ""
        _symptom, _covered, uncovered = self._checklist(messages)
        pending = self._pending_actions()
        current = (hook_input.original_response or "").strip()
        if len(current) < 120:
            return ""
        current_tokens = _norm_tokens(current)
        if not current_tokens:
            return ""

        previous = []
        for message in messages:
            if message.role != Role.ASSISTANT:
                continue
            try:
                previous.append(message.get_text_content())
            except Exception:
                continue
        if previous and previous[-1].strip() == current:
            previous = previous[:-1]

        for text in previous:
            if len(text) < 120:
                continue
            if _jaccard(current_tokens, _norm_tokens(text)) >= self.repeat_threshold:
                self.repeat_breaks += 1
                if not uncovered and not pending:
                    return _REPEAT_HEAD + _REPEAT_IDLE_TAIL
                tail = ""
                if uncovered:
                    tail += " NOT covered yet: " + "; ".join(h for _i, h in uncovered[:3]) + "."
                if pending:
                    tail += " Account actions outstanding: " + "; ".join(pending) + "."
                return _REPEAT_HEAD + tail + _REPEAT_TAIL
        return ""
