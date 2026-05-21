"""Telecom domain harness rules and annotators.

All rules and annotators inspect only DB state (no conversation history),
ensuring replay safety during set_state evaluation.

Policy source: data/tau2/domains/telecom/main_policy.md

H2 Rule index
-------------
[Generic — applied to multiple write tools]
  CustomerIDValidationRule           – customer_id format check + existence guard (T8)
  LineOwnershipRule                  – line_id belongs to customer; lists lines with phone
                                       numbers when mismatch detected (T9)

get_customer_by_phone
  PhoneNumberFormatRule              – reject non-NXX-NXX-XXXX phone formats (T13)

send_payment_request
  CustomerIDValidationRule           (T8)
  BillOwnershipRule                  – bill_id must belong to customer_id (T11)
  SendPaymentExpiredContractRule     – do not collect payment when the suspended
                                       line cannot be resumed because contract expired
  SendPaymentRequestOverdueRule      – bill must be OVERDUE before sending (T1)
  SendPaymentOneAtATimeRule          – only one AWAITING_PAYMENT bill per customer (T6)

resume_line
  CustomerIDValidationRule           (T8)
  LineOwnershipRule                  (T9)
  ResumeLineStatusCheck              – line must be Suspended before resuming (T10)
  ResumeLineEligibilityRule          – all overdue bills paid + contract not expired (T2)

refuel_data
  CustomerIDValidationRule           (T8)
  LineOwnershipRule                  (T9)
  RefuelDataLimitRule                – max 2 GB per session (T3)
  RefuelDataActiveLineRule           – line must be Active to refuel (T5)

suspend_line
  CustomerIDValidationRule           (T8)
  LineOwnershipRule                  (T9)
  SuspendLineStatusRule              – line must be Active to suspend (T4)

enable_roaming
  CustomerIDValidationRule           (T8)
  LineOwnershipRule                  (T9)
  EnableRoamingAlreadyEnabledRule    – block when roaming already on; redirect to device
                                       toggle (T7)

disable_roaming
  CustomerIDValidationRule           (T8)
  LineOwnershipRule                  (T9)
  DisableRoamingAlreadyDisabledRule  – block when roaming already off (T12)

H4 Annotator index
------------------
get_customer_by_phone
  PhoneNumberFormatRule       – reject non-NXX-NXX-XXXX formats (e.g. +1234567890)
                                before lookup; guide agent to ask for correct format (T13)
  CustomerPhoneLineAnnotator  – when customer has multiple lines, identify which
                                line_id corresponds to the caller's phone number.

get_data_usage
  DataQuotaAnnotator          – flag when data quota is exhausted; prompt refuel.

get_details_by_id (Line)
  LineStatusAnnotator         – flag data quota exhaustion (vs plan limit);
                                flag roaming disabled; flag suspended line workflow.

get_bills_for_customer
  BillsOverdueAnnotator       – highlight OVERDUE bills; remind send_payment_request
                                → resume_line workflow.

send_payment_request
  PaymentWorkflowAnnotator    – remind agent of post-payment steps (make_payment →
                                resume_line → reboot).

enable_roaming
  EnableRoamingResultAnnotator – distinguish "already enabled" (device-side fix)
                                 from newly enabled (remind device toggle).
"""

import re
from typing import Any

from tau2.domains.telecom.data_model import BillStatus, Line, LineStatus, TelecomDB
from tau2.domains.telecom.tools import TelecomTools
from tau2.domains.telecom.utils import get_today
from tau2.harness.base import HarnessedToolKitMixin
from tau2.harness.h3_tools import H3TelecomToolDescriptionMixin

# ---------------------------------------------------------------------------
# T13 — get_customer_by_phone: phone number format guard
# ---------------------------------------------------------------------------


class PhoneNumberFormatRule:
    """Validate phone number format before get_customer_by_phone.

    Agents sometimes use country-code formats like '+1234567890' or omit
    dashes, causing guaranteed lookup failures.  Catch this early and prompt
    the agent to ask the customer for the correct 10-digit local format.
    """

    tool_name = "get_customer_by_phone"

    def check(self, db: Any, phone_number: str = "", **_: Any) -> None:
        if not phone_number:
            return
        if not re.match(r"^\d{3}-\d{3}-\d{4}$", phone_number):
            raise ValueError(
                f"Phone number '{phone_number}' is not in the expected format. "
                "Expected: 'NXX-NXX-XXXX' (10 digits, dashes, no country code — e.g. '555-123-4567'). "
                "Ask the customer: 'Could you provide your 10-digit phone number, "
                "for example starting with your area code, like 555-123-4567?'"
            )


# ---------------------------------------------------------------------------
# T8 — Generic: customer_id format + existence
# ---------------------------------------------------------------------------


class CustomerIDValidationRule:
    """Validate customer_id format and existence before any write operation.

    Catches hallucinated IDs like 'cust123', 'C123456' (non-existent) before
    the tool raises a generic "not found" error.  Redirects agent to use
    get_customer_by_phone to obtain the correct ID.
    """

    def check(self, db: TelecomDB, customer_id: str = "", **_: Any) -> None:
        if not customer_id:
            return
        if not re.match(r"^C\d+$", customer_id):
            raise ValueError(
                f"Invalid customer_id format '{customer_id}'. "
                "Customer IDs follow the pattern 'C1001'. "
                "Call get_customer_by_phone to look up the correct ID."
            )
        customer = next(
            (c for c in db.customers if c.customer_id == customer_id), None
        )
        if customer is None:
            raise ValueError(
                f"Customer '{customer_id}' not found. "
                "Call get_customer_by_phone to obtain the correct customer_id."
            )


# ---------------------------------------------------------------------------
# T9 — Generic: line_id belongs to customer, with phone-number guidance
# ---------------------------------------------------------------------------


class LineOwnershipRule:
    """Validate line_id format and ownership before any write operation.

    When the line doesn't belong to the customer, lists all the customer's
    lines with their phone numbers so the agent can identify the correct one.
    This addresses the common mistake of using L1001 instead of L1002 when
    the customer's phone number maps to a different line.
    """

    def check(
        self, db: TelecomDB, customer_id: str = "", line_id: str = "", **_: Any
    ) -> None:
        if not line_id:
            return
        if not re.match(r"^L\d+$", line_id):
            raise ValueError(
                f"Invalid line_id format '{line_id}'. "
                "Line IDs follow the pattern 'L1001'. "
                "Get the customer first to see their line IDs."
            )
        customer = next(
            (c for c in db.customers if c.customer_id == customer_id), None
        )
        if customer is None:
            return  # CustomerIDValidationRule handles unknown customer
        if line_id not in customer.line_ids:
            customer_lines = [ln for ln in db.lines if ln.line_id in customer.line_ids]
            line_details = ", ".join(
                f"{ln.line_id} (phone: {ln.phone_number})" for ln in customer_lines
            )
            raise ValueError(
                f"Line '{line_id}' does not belong to customer {customer_id}. "
                f"Customer {customer_id} has lines: {line_details}. "
                "Use the line_id that matches the customer's phone number."
            )


# ---------------------------------------------------------------------------
# T1 — send_payment_request: bill must be OVERDUE
# ---------------------------------------------------------------------------


class SendPaymentRequestOverdueRule:
    """Block send_payment_request if the bill is not in OVERDUE status.

    Policy: "The send payment request tool will not check if the bill is
    overdue.  You should always check that the bill is overdue before
    sending a payment request."
    """

    tool_name = "send_payment_request"

    def check(self, db: TelecomDB, bill_id: str, **_: Any) -> None:
        bill = next((b for b in db.bills if b.bill_id == bill_id), None)
        if bill is None:
            return  # let the tool raise its own "not found" error
        if bill.status != BillStatus.OVERDUE:
            raise ValueError(
                f"Bill {bill_id} has status '{bill.status.value}', not 'Overdue'. "
                "Per policy, a payment request can only be sent for overdue bills. "
                "Check the bill status with get_bills_for_customer before proceeding."
            )


class SendPaymentExpiredContractRule:
    """Block payment collection when the likely suspended line cannot be resumed.

    In service-restoration workflows, collecting an overdue payment is only useful
    if the line can be resumed afterwards.  If the customer's suspended line has an
    expired contract, policy requires transfer for contract renewal; sending a
    payment request first creates a harmful side effect without restoring service.
    """

    tool_name = "send_payment_request"

    def check(self, db: TelecomDB, customer_id: str, bill_id: str = "", **_: Any) -> None:
        customer = next(
            (c for c in db.customers if c.customer_id == customer_id), None
        )
        if customer is None:
            return

        today = get_today()
        suspended_lines = [
            line
            for line in db.lines
            if line.line_id in customer.line_ids
            and line.status == LineStatus.SUSPENDED
        ]
        if not suspended_lines:
            return

        expired_suspended_lines = [
            line
            for line in suspended_lines
            if line.contract_end_date is not None
            and line.contract_end_date < today
        ]
        has_resumable_or_unknown_suspended_line = any(
            line.contract_end_date is None or line.contract_end_date >= today
            for line in suspended_lines
        )
        if not expired_suspended_lines or has_resumable_or_unknown_suspended_line:
            return

        line_details = ", ".join(
            f"{line.line_id} ({line.phone_number}, contract ended {line.contract_end_date})"
            for line in expired_suspended_lines
        )
        raise ValueError(
            f"Do not send payment request {bill_id} yet. Customer {customer_id} has "
            f"suspended line(s) with expired contracts: {line_details}. Per policy, "
            "expired-contract lines cannot be resumed even after overdue bills are "
            "paid. Transfer the customer to a human agent for contract renewal "
            "instead of collecting payment."
        )


# ---------------------------------------------------------------------------
# T2 — resume_line: overdue bills paid + contract not expired
# ---------------------------------------------------------------------------


class ResumeLineEligibilityRule:
    """Block resume_line when the customer still has unpaid overdue bills or
    when the line's contract has expired.

    Policy:
    - "You are allowed to lift the suspension after the user has paid all
      their overdue bills."
    - "You are not allowed to lift the suspension if the line's contract end
      date is in the past, even if the user has paid all their overdue bills."
    """

    tool_name = "resume_line"

    def check(
        self, db: TelecomDB, customer_id: str, line_id: str, **_: Any
    ) -> None:
        customer = next(
            (c for c in db.customers if c.customer_id == customer_id), None
        )
        if customer is None:
            return

        # Check 1: outstanding overdue bills
        overdue_bills = [
            b
            for bill_id in customer.bill_ids
            for b in db.bills
            if b.bill_id == bill_id and b.status == BillStatus.OVERDUE
        ]
        if overdue_bills:
            bill_ids_str = ", ".join(b.bill_id for b in overdue_bills)
            raise ValueError(
                f"Customer {customer_id} still has outstanding overdue bills "
                f"({bill_ids_str}). All overdue bills must be paid before the "
                "line suspension can be lifted. "
                "Use send_payment_request to initiate payment first."
            )

        # Check 2: contract end date expired
        line = next((ln for ln in db.lines if ln.line_id == line_id), None)
        if line is None:
            return
        today = get_today()
        if line.contract_end_date is not None and line.contract_end_date < today:
            raise ValueError(
                f"Line {line_id} contract expired on {line.contract_end_date} "
                f"(today: {today}). "
                "Per policy, a line with an expired contract cannot be resumed, "
                "even if all overdue bills have been paid. "
                "Transfer the customer to a human agent for contract renewal."
            )


# ---------------------------------------------------------------------------
# T3 — refuel_data: max 2 GB per session
# ---------------------------------------------------------------------------


class RefuelDataLimitRule:
    """Block refuel_data when requested GB exceeds the policy maximum.

    Policy: "The maximum amount of data that can be refueled is 2GB."
    """

    tool_name = "refuel_data"

    def check(self, db: TelecomDB, gb_amount: float, **_: Any) -> None:
        if gb_amount > 2:
            raise ValueError(
                f"Requested refuel amount of {gb_amount} GB exceeds the "
                "per-session maximum of 2 GB. "
                "Please adjust the amount to 2 GB or less. "
                "If the customer needs more than 2 GB, you may call refuel_data "
                "a second time after the first refuel is confirmed."
            )


# ---------------------------------------------------------------------------
# T4 — suspend_line: line must be Active
# ---------------------------------------------------------------------------


class SuspendLineStatusRule:
    """Block suspend_line when the line is not in Active status.

    Provides a more informative error than the tool's generic 'Line must be
    active to suspend'.
    """

    tool_name = "suspend_line"

    def check(
        self, db: TelecomDB, customer_id: str, line_id: str, **_: Any
    ) -> None:
        line = next((ln for ln in db.lines if ln.line_id == line_id), None)
        if line is None:
            return  # let the tool raise its own "not found" error
        if line.status != LineStatus.ACTIVE:
            raise ValueError(
                f"Line {line_id} has status '{line.status.value}', not 'Active'. "
                "Only Active lines can be suspended. "
                f"{'Use resume_line to reactivate it first.' if line.status == LineStatus.SUSPENDED else ''}"
            )


# ---------------------------------------------------------------------------
# T5 — refuel_data: line must be Active
# ---------------------------------------------------------------------------


class RefuelDataActiveLineRule:
    """Block refuel_data when the target line is not Active.

    The tool silently accepts refuels on suspended lines (the status check
    was commented out).  This rule reinstates that guard with a clear message.
    """

    tool_name = "refuel_data"

    def check(
        self, db: TelecomDB, customer_id: str, line_id: str, **_: Any
    ) -> None:
        # Locate the customer to verify ownership
        customer = next(
            (c for c in db.customers if c.customer_id == customer_id), None
        )
        if customer is None:
            return
        if line_id not in customer.line_ids:
            return  # let the tool raise "line not found"

        line = next((ln for ln in db.lines if ln.line_id == line_id), None)
        if line is None:
            return
        if line.status != LineStatus.ACTIVE:
            raise ValueError(
                f"Line {line_id} has status '{line.status.value}'. "
                "Data can only be refueled on Active lines. "
                "If the line is suspended due to an overdue bill, the customer "
                "must pay the overdue bill and have the line resumed first."
            )


# ---------------------------------------------------------------------------
# T11 — send_payment_request: bill must belong to customer
# ---------------------------------------------------------------------------


class BillOwnershipRule:
    """Block send_payment_request when bill_id doesn't belong to customer_id.

    Prevents cross-customer billing errors and provides a redirect to
    get_bills_for_customer to list the correct bill IDs.
    """

    tool_name = "send_payment_request"

    def check(
        self, db: TelecomDB, customer_id: str, bill_id: str, **_: Any
    ) -> None:
        customer = next(
            (c for c in db.customers if c.customer_id == customer_id), None
        )
        if customer is None:
            return  # CustomerIDValidationRule handles unknown customer
        if bill_id not in customer.bill_ids:
            raise ValueError(
                f"Bill '{bill_id}' does not belong to customer {customer_id}. "
                f"Customer {customer_id} has bills: {', '.join(customer.bill_ids)}. "
                "Use get_bills_for_customer to list the correct bill IDs."
            )


# ---------------------------------------------------------------------------
# T6 — send_payment_request: one AWAITING_PAYMENT bill at a time
# ---------------------------------------------------------------------------


class SendPaymentOneAtATimeRule:
    """Block send_payment_request when another bill is already awaiting payment.

    Policy: "A user can only have one bill in the AWAITING PAYMENT status
    at a time."

    The tool enforces this but with a generic error; this rule provides a
    clearer message that names the already-pending bill.
    """

    tool_name = "send_payment_request"

    def check(
        self, db: TelecomDB, customer_id: str, bill_id: str, **_: Any
    ) -> None:
        customer = next(
            (c for c in db.customers if c.customer_id == customer_id), None
        )
        if customer is None:
            return

        awaiting = [
            b
            for b in db.bills
            if b.bill_id in customer.bill_ids
            and b.status == BillStatus.AWAITING_PAYMENT
            and b.bill_id != bill_id  # not the bill we're about to send
        ]
        if awaiting:
            existing_id = awaiting[0].bill_id
            raise ValueError(
                f"Customer {customer_id} already has bill {existing_id} in "
                "'Awaiting Payment' status. "
                "Only one bill can be awaiting payment at a time. "
                "Wait for the current payment request to be processed before "
                "sending a new one."
            )


# ---------------------------------------------------------------------------
# T10 — resume_line: line must currently be Suspended
# ---------------------------------------------------------------------------


class ResumeLineStatusCheck:
    """Block resume_line when the line is not in Suspended status.

    T2 (ResumeLineEligibilityRule) checks billing/contract conditions.
    This rule adds the complementary status check: you can only resume a
    line that is actually suspended.
    """

    tool_name = "resume_line"

    def check(
        self, db: TelecomDB, line_id: str = "", **_: Any
    ) -> None:
        line = next((ln for ln in db.lines if ln.line_id == line_id), None)
        if line is None:
            return  # let the tool raise "not found"
        if line.status == LineStatus.ACTIVE:
            raise ValueError(
                f"Line {line_id} is already Active — resume_line is not needed. "
                "If the customer is still experiencing issues, check for other causes "
                "(data quota, roaming settings, device configuration)."
            )
        if line.status != LineStatus.SUSPENDED:
            raise ValueError(
                f"Line {line_id} has status '{line.status.value}'. "
                "resume_line can only reactivate lines that are currently Suspended. "
                "Transfer to a human agent if the line is Closed or in an unexpected state."
            )


# ---------------------------------------------------------------------------
# T7 — enable_roaming: block when roaming is already enabled
# ---------------------------------------------------------------------------


class EnableRoamingAlreadyEnabledRule:
    """Block enable_roaming when the line's roaming is already enabled.

    When account-level roaming is already on, calling enable_roaming again
    wastes a turn and confuses the agent.  The fix is on the device side —
    the customer must toggle Data Roaming ON in their phone settings.

    Provides a redirect rather than a hard block so the agent understands
    what to do next.
    """

    tool_name = "enable_roaming"

    def check(self, db: TelecomDB, customer_id: str, line_id: str, **_: Any) -> None:
        line = next((ln for ln in db.lines if ln.line_id == line_id), None)
        if line is None:
            return
        if line.roaming_enabled:
            raise ValueError(
                f"Line {line_id} already has roaming_enabled=True — "
                "account-level roaming is already ON. "
                "Calling enable_roaming again will not help. "
                "The fix is on the DEVICE side: instruct the customer to toggle "
                "Data Roaming ON in their phone settings (toggle_roaming user action)."
            )


# ---------------------------------------------------------------------------
# T12 — disable_roaming: block when roaming is already disabled
# ---------------------------------------------------------------------------


class DisableRoamingAlreadyDisabledRule:
    """Block disable_roaming when the line's roaming is already off.

    Symmetric mirror of T7 (EnableRoamingAlreadyEnabledRule).  Prevents
    wasting a turn on a no-op call and provides a clear status message.
    """

    tool_name = "disable_roaming"

    def check(self, db: TelecomDB, customer_id: str, line_id: str, **_: Any) -> None:
        line = next((ln for ln in db.lines if ln.line_id == line_id), None)
        if line is None:
            return
        if not line.roaming_enabled:
            raise ValueError(
                f"Line {line_id} already has roaming_enabled=False — "
                "roaming is already OFF. No action is needed."
            )


# ---------------------------------------------------------------------------
# H4 Annotators
# ---------------------------------------------------------------------------


class DataQuotaAnnotator:
    """After get_data_usage, flag when the data quota is exhausted."""

    tool_name = "get_data_usage"

    def annotate(self, db: TelecomDB, result: Any, **_: Any) -> str | None:
        if not isinstance(result, dict):
            return None
        data_used: float = result.get("data_used_gb", 0.0)
        data_limit: float = result.get("data_limit_gb", 0.0)
        data_refueling: float = result.get("data_refueling_gb", 0.0)
        if data_limit <= 0:
            return None
        total_available = data_limit + data_refueling
        if data_used >= total_available:
            return (
                f"⚠️ QUOTA EXHAUSTED ({data_used:.1f}/{total_available:.1f} GB). "
                "This blocks all data. Offer refuel_data (max 2 GB, confirm price first)."
            )
        return None


class LineStatusAnnotator:
    """After get_details_by_id for a Line, flag roaming, quota, and suspension."""

    tool_name = "get_details_by_id"

    def annotate(self, db: TelecomDB, result: Any, **_: Any) -> str | None:
        if not isinstance(result, Line):
            return None
        notes: list[str] = []

        if not result.roaming_enabled:
            notes.append(
                "⚠️ ROAMING OFF (roaming_enabled=False). "
                "If abroad: call enable_roaming + remind toggle_roaming on device."
            )

        try:
            plan = next(p for p in db.plans if p.plan_id == result.plan_id)
            total_available = plan.data_limit_gb + result.data_refueling_gb
            if plan.data_limit_gb > 0 and result.data_used_gb >= total_available:
                notes.append(
                    f"⚠️ QUOTA EXHAUSTED ({result.data_used_gb:.1f}/{total_available:.1f} GB). "
                    f"Offer refuel_data({result.line_id}, max 2 GB, "
                    f"${plan.data_refueling_price_per_gb:.2f}/GB)."
                )
        except StopIteration:
            pass

        if result.status == LineStatus.SUSPENDED:
            today = get_today()
            if result.contract_end_date is not None and result.contract_end_date < today:
                notes.append(
                    f"⚠️ SUSPENDED — contract expired {result.contract_end_date}. "
                    "Call transfer_to_human_agents(summary=...) TOOL — not just text."
                )
            else:
                notes.append(
                    "⚠️ SUSPENDED. Fix: get_bills_for_customer → send_payment_request "
                    "→ make_payment → resume_line → reboot."
                )

        return "\n".join(notes) if notes else None


class CustomerPhoneLineAnnotator:
    """After get_customer_by_phone, flag which line_id matches the caller's number."""

    tool_name = "get_customer_by_phone"

    def annotate(self, db: TelecomDB, result: Any, phone_number: str = "", **_: Any) -> str | None:
        if not hasattr(result, "line_ids") or not phone_number:
            return None
        matching_line = next(
            (ln for ln in db.lines if ln.phone_number == phone_number), None
        )
        if matching_line is None or len(result.line_ids) <= 1:
            return None
        return (
            f"📌 Use line_id='{matching_line.line_id}' for ALL writes "
            f"(phone {phone_number} matches this line; customer has "
            f"{len(result.line_ids)} lines: {', '.join(result.line_ids)})."
        )


class BillsOverdueAnnotator:
    """After get_bills_for_customer, highlight OVERDUE bills."""

    tool_name = "get_bills_for_customer"

    def annotate(self, db: TelecomDB, result: Any, customer_id: str = "", **_: Any) -> str | None:
        if not isinstance(result, list):
            return None
        overdue = [b for b in result if hasattr(b, "status") and b.status == BillStatus.OVERDUE]
        if not overdue:
            return None
        if customer_id:
            today = get_today()
            customer = next(
                (c for c in db.customers if c.customer_id == customer_id), None
            )
            if customer is not None:
                suspended_lines = [
                    line
                    for line in db.lines
                    if line.line_id in customer.line_ids
                    and line.status == LineStatus.SUSPENDED
                ]
                has_expired_suspension = any(
                    line.contract_end_date is not None
                    and line.contract_end_date < today
                    for line in suspended_lines
                )
                has_resumable_or_unknown_suspension = any(
                    line.contract_end_date is None
                    or line.contract_end_date >= today
                    for line in suspended_lines
                )
                if has_expired_suspension and not has_resumable_or_unknown_suspension:
                    return (
                        "⚠️ OVERDUE bill exists, but the suspended line's contract "
                        "has expired. Do NOT collect payment for restoration; "
                        "call transfer_to_human_agents(summary=...) TOOL for "
                        "contract renewal."
                    )
        bill_strs = ", ".join(f"{b.bill_id} (${b.total_due:.2f})" for b in overdue)
        return (
            f"⚠️ OVERDUE: {bill_strs}. "
            "send_payment_request → make_payment → resume_line → reboot."
        )


class PaymentWorkflowAnnotator:
    """After send_payment_request, remind remaining steps."""

    tool_name = "send_payment_request"

    def annotate(self, db: TelecomDB, result: Any, **_: Any) -> str | None:
        return (
            "✓ Payment request sent. Next: customer uses check_payment_request → "
            "make_payment → verify PAID → resume_line → customer reboots."
        )


class EnableRoamingResultAnnotator:
    """After enable_roaming, flag 'already enabled' so agent knows to fix device side."""

    tool_name = "enable_roaming"

    def annotate(self, db: TelecomDB, result: Any, **_: Any) -> str | None:
        result_str = str(result).lower() if result else ""
        if "already enabled" in result_str:
            return (
                "ℹ️ Account roaming was already ON. Fix is device-side: "
                "tell customer to toggle Data Roaming ON in settings (toggle_roaming)."
            )
        return (
            "✓ Account roaming enabled. ALSO: customer must toggle Data Roaming ON "
            "in device settings (toggle_roaming). Then re-run speed test."
        )


# ---------------------------------------------------------------------------
# H4 Mixin and composite toolkit classes
# ---------------------------------------------------------------------------


class H4TelecomAnnotationMixin:
    """Mixin that adds H4 post-execution tool-response annotations to telecom tools."""

    harness_annotators: dict[str, list] = {
        "get_customer_by_phone": [CustomerPhoneLineAnnotator()],
        "get_data_usage": [DataQuotaAnnotator()],
        "get_details_by_id": [LineStatusAnnotator()],
        "get_bills_for_customer": [BillsOverdueAnnotator()],
        "send_payment_request": [PaymentWorkflowAnnotator()],
        "enable_roaming": [EnableRoamingResultAnnotator()],
    }


# ---------------------------------------------------------------------------
# Composite toolkit classes
# ---------------------------------------------------------------------------


class HarnessedTelecomTools(HarnessedToolKitMixin, TelecomTools):
    """TelecomTools with pre-execution harness validation (H2) enabled."""

    harness_rules: dict[str, list] = {
        "get_customer_by_phone": [
            PhoneNumberFormatRule(),
        ],
        "send_payment_request": [
            CustomerIDValidationRule(),
            BillOwnershipRule(),
            SendPaymentExpiredContractRule(),
            SendPaymentRequestOverdueRule(),
            SendPaymentOneAtATimeRule(),
        ],
        "resume_line": [
            CustomerIDValidationRule(),
            LineOwnershipRule(),
            ResumeLineStatusCheck(),
            ResumeLineEligibilityRule(),
        ],
        "refuel_data": [
            CustomerIDValidationRule(),
            LineOwnershipRule(),
            RefuelDataLimitRule(),
            RefuelDataActiveLineRule(),
        ],
        "suspend_line": [
            CustomerIDValidationRule(),
            LineOwnershipRule(),
            SuspendLineStatusRule(),
        ],
        "enable_roaming": [
            CustomerIDValidationRule(),
            LineOwnershipRule(),
            EnableRoamingAlreadyEnabledRule(),
        ],
        "disable_roaming": [
            CustomerIDValidationRule(),
            LineOwnershipRule(),
            DisableRoamingAlreadyDisabledRule(),
        ],
    }


class H4TelecomTools(H4TelecomAnnotationMixin, HarnessedToolKitMixin, TelecomTools):
    """TelecomTools with H4 post-execution annotations only (no H2 rules)."""

    harness_rules: dict[str, list] = {}


class H4HarnessedTelecomTools(H4TelecomAnnotationMixin, HarnessedTelecomTools):
    """TelecomTools with H2 harness rules and H4 annotations."""


class H3TelecomTools(H3TelecomToolDescriptionMixin, TelecomTools):
    """TelecomTools with H3 tool-description policy hints enabled."""


class H3HarnessedTelecomTools(H3TelecomToolDescriptionMixin, HarnessedTelecomTools):
    """TelecomTools with both H2 harness rules and H3 tool-description hints."""


class H3H4TelecomTools(H3TelecomToolDescriptionMixin, H4TelecomTools):
    """TelecomTools with H3 hints and H4 annotations (no H2 rules)."""


class H3H4HarnessedTelecomTools(
    H3TelecomToolDescriptionMixin, H4TelecomAnnotationMixin, HarnessedTelecomTools
):
    """TelecomTools with H2 rules, H3 hints, and H4 annotations."""
