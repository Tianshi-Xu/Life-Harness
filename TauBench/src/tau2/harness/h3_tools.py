"""H3: Tool-description policy embedding.

Unlike the system prompt (a message at position 0 that gets pushed down by
accumulating conversation turns), **tool descriptions are re-sent in every
LLM API request** as part of the ``tools`` parameter.  The model therefore
sees them at full attention weight regardless of how long the dialogue has
grown.

H3 appends concise, actionable policy constraints directly into the
``long_desc`` of key write tools for a domain.  This is complementary to H2
(runtime validation) and H5 (system-prompt injection):

- H2 catches bad calls *after* the agent decides to make them.
- H5 injects policy into the system prompt (visible at conversation start,
  but decays in salience as context grows).
- H3 keeps the most critical policy rules *always* in the model's immediate
  attention window, acting as a persistent reminder at the point of decision.

Usage
-----
Enable via ``harness_h3=True`` in the run config or ``--harness-h3`` on the
CLI.  H3 is independent of H2 and H5 and can be combined freely::

    # H3 only
    tau2 run --domain airline --harness-h3 ...

    # H2 + H3
    tau2 run --domain airline --harness-enabled --harness-h3 ...

    # H2 + H3 + full policy in prompt (H5)
    tau2 run --domain airline --harness-enabled --harness-h3 --harness-h5 ...

Implementation note
-------------------
``H3ToolDescriptionMixin.get_tools()`` wraps each target tool's underlying
function with a copy whose ``__doc__`` has the hint appended.  A new
``Tool`` object is built from the wrapper so that the ``openai_schema``
(which drives the LLM's tool-call decision) includes the extended description.
The ``ToolKitBase.use_tool()`` execution path is unaffected.
"""

from __future__ import annotations

import functools
from typing import Any

from tau2.environment.tool import Tool

# ---------------------------------------------------------------------------
# Core patching helper
# ---------------------------------------------------------------------------


def _append_hint_to_tool(tool: Tool, hint: str) -> Tool:
    """Return a new Tool identical to *tool* but with *hint* appended to its description.

    The function signature and parameter schema are preserved via
    :func:`functools.wraps`.  Only the docstring (and therefore
    ``short_desc`` / ``long_desc`` parsed from it) is extended.

    Args:
        tool: The original Tool object to patch.
        hint: Policy constraint text to append.

    Returns:
        A new Tool with the hint embedded in its description.
    """
    orig_func = tool._func
    orig_doc = orig_func.__doc__ or ""
    new_doc = (orig_doc.rstrip() + "\n\n" + hint).strip()

    @functools.wraps(orig_func)
    def _patched(*args: Any, **kwargs: Any) -> Any:
        return orig_func(*args, **kwargs)

    _patched.__doc__ = new_doc
    return Tool(
        func=_patched,
        use_short_desc=tool._use_short_desc,
        **tool._predefined,
    )


# ---------------------------------------------------------------------------
# Airline domain hints
# ---------------------------------------------------------------------------

_AIRLINE_H3_HINTS: dict[str, str] = {
    "search_direct_flight": """\
SELECTION HINT:
  • Args are exactly origin, destination, date. Do NOT pass cabin.
  • Results are candidates, not recommendations. For cheapest/fastest/time-window
    requests, compare all returned options using the requested cabin's price.
  • A candidate is usable only if available_seats for the requested cabin is at
    least the reservation's passenger count.""",
    "search_onestop_flight": """\
SELECTION HINT:
  • Args are exactly origin, destination, date. Do NOT pass cabin.
  • Results are candidates, not recommendations. For fastest, compare first
    departure to final arrival including layover; for cheapest, sum both legs
    using the requested cabin's prices.
  • A one-stop pair is usable only when BOTH legs have enough seats in the
    requested cabin. If an update fails for seat availability, choose the next
    valid candidate and retry the update before doing unrelated writes.""",
    "cancel_reservation": """\
POLICY CONSTRAINTS — verify ALL of the following before calling:

Eligibility: at least ONE condition must be true:
  1. Reservation was booked within the last 24 hours.
  2. The airline cancelled a flight in this reservation.
  3. Cabin is 'business'.
  4. Travel insurance was purchased. A health/weather reason is needed for
     insurance-covered full-refund reasoning, but do not skip a cancellation
     solely because an insured user gives a non-health/non-weather reason and
     explicitly still wants to proceed.

If any flight has status 'flying' or 'landed': DO NOT cancel — call \
transfer_to_human_agents instead.

Tip for basic_economy: call update_reservation_flights first to upgrade the \
cabin to 'business' (keeping the EXACT SAME flight numbers). Upgrading only \
to 'economy' does NOT satisfy cancellation condition 3.""",
    "update_reservation_flights": """\
POLICY CONSTRAINTS:
  • Cabin is reservation-wide; you cannot change cabin for only one segment or
    one direction. Upgrade/downgrade is allowed if no leg is flying/landed.
  • basic_economy: changing flight numbers is not allowed. To change both cabin
    and flights, cancel first, then book a new reservation.
    Updating a basic_economy reservation with different flight numbers is the
    wrong path even if the new cabin is economy/business. A same-flight cabin
    upgrade is allowed; a new itinerary requires cancel-and-rebook when the
    user accepts that fallback.
  • round_trip: include the entire connected itinerary in one call. If only one
    direction changes, copy the other direction exactly from get_reservation_details.
    Preserve the requested/original return date; do not accidentally reuse the
    outbound date for the return.
  • Origin, destination, trip type cannot change.
  • Payment must be gift_card or credit_card, never certificate.
  • Payment is for the NET fare difference (new itinerary minus current
    reservation fare). If the net charge is positive and the user prefers gift
    cards, choose the smallest gift card whose balance can cover the entire net
    charge; if no gift card can cover it, use a credit card. Do not choose the
    smallest balance overall when it is insufficient.""",
    "book_reservation": """\
POLICY CONSTRAINTS — check all limits before calling:

Before booking: collect or confirm all user constraints that affect the call:
  • departure-time preference, cabin, passenger identity/DOB, total checked bags,
    insurance preference, and payment allocation.
  • If the user wants certificates, use at most ONE certificate in this
    reservation; put any remaining cost on one credit card/gift card.

Payment (per reservation):
  • At most 1 travel certificate.
  • At most 1 credit card.
  • At most 3 gift cards.
  • To use multiple certificates: create separate reservations (1 cert each).

Passengers: maximum 5 per reservation; split into multiple bookings if needed.

Flights: ALL flights must have status 'available' (delayed / on time / flying / \
cancelled flights cannot be booked).

Free checked bags per passenger (membership × cabin):
  regular → basic_economy=0  economy=1  business=2
  silver  → basic_economy=1  economy=2  business=3
  gold    → basic_economy=2  economy=3  business=4
nonfree_baggages = max(0, total_baggages − free_allowance); each extra bag = $50.""",
    "update_reservation_baggages": """\
POLICY CONSTRAINTS:
  • Checked bags can only be ADDED, never removed.
  • Do NOT add bags the user has not explicitly requested.
  • If the user asks to add checked bags, you MUST call this tool even when
    all requested bags are free under membership/cabin allowance.
  • total_baggages is the new total bag count on the reservation.
    nonfree_baggages is only the count beyond free allowance.""",
    "update_reservation_passengers": """\
POLICY CONSTRAINTS:
  • Passenger COUNT is fixed at booking time and CANNOT be changed.
  • Not even a human agent can add or remove passengers from an existing \
reservation.
  • You MAY update/replace the details of an existing passenger slot when the
    count stays the same. If the user asks to change the passenger to themself,
    submit the same number of passengers with that slot's name and date of birth
    changed to the user's profile details.""",
    "send_certificate": """\
POLICY CONSTRAINTS — verify eligibility and amount before calling:

Eligibility — user must have at least ONE of:
  • silver or gold membership.
  • A reservation in business cabin.
  • A reservation with travel insurance purchased.

Do NOT offer compensation proactively; only when the user explicitly requests it.
For delayed flights: do NOT issue a certificate unless the user wants to \
change/cancel the reservation AND that change/cancel has already completed. \
If the user keeps the delayed flight unchanged, confirm the delay and explain \
that no certificate should be issued.

Amount rules:
  • Must be a positive multiple of $50.
  • Cancelled flight → $100 × number of passengers in the affected reservation.
  • Delayed flight   → $50  × number of passengers in the affected reservation.
  • Hard cap: $100 × passengers in the affected reservation (max 5 → $500).""",
    "transfer_to_human_agents": """\
TRANSFER POLICY:
  • Transfer only when tools cannot solve the request, or a flight already
    flying/landed must be cancelled.
  • Do not transfer for normal booking, flight/cabin changes, eligible
    cancellations, checked bags, passenger detail updates, or certificates.
  • If policy blocks an action, explain the reason and offer the valid tool path
    instead of transferring.""",
}


# ---------------------------------------------------------------------------
# Retail domain hints
# ---------------------------------------------------------------------------

_RETAIL_H3_HINTS: dict[str, str] = {
    "get_user_details": """\
HINT: Use this only as an order index. Before any write, call get_order_details \
on the relevant order(s) for exact status, item_id, product_id, payment method, \
address, and tracking. If the item/order is ambiguous, inspect all plausible \
orders; do not stop after the first partial match.""",
    "get_order_details": """\
HINT: Use exact item_id values from this result for writes. Route by status: \
pending -> cancel/modify; delivered -> return/exchange; requested/cancelled/\
processed usually block writes. If fulfillments.tracking_id is present, provide \
it even for cancelled/pending orders. Use payment_history for totals/refunds.""",
    "get_product_details": """\
HINT: Match every hard option exactly and preserve unchanged specs. Count \
"available/in stock" as variants with available=true only. For cheapest/most \
expensive, compare all variants; replacement new_item_ids for modify/exchange \
must be available=true.""",
    "find_user_id_by_email": """\
HINT: If the user provides email, use this directly. Do not require zipcode first.""",
    "find_user_id_by_name_zip": """\
HINT: Use only with first name, last name, and zipcode. If zipcode is missing, \
ask for email instead of transferring.""",
    "cancel_pending_order": """\
CONSTRAINTS: Status must be exactly 'pending'; 'pending (item modified)' \
cannot be cancelled. This cancels the ENTIRE order; it cannot remove one \
selected item from a multi-item order. Use it for item-scoped requests only \
after the user explicitly accepts full-order cancellation. reason is exactly \
'no longer needed' or 'ordered by mistake'. Refund goes to original payment \
method only. Use 'ordered by mistake' only when the user explicitly says it was \
a mistake/accident/duplicate order; otherwise use 'no longer needed' for \
changed plans or accessory orders no longer needed after a related return. \
Preserve the user's stated valid reason; do not replace 'no longer needed' with \
'ordered by mistake' or vice versa.""",
    "modify_pending_order_items": """\
CONSTRAINTS: Status exactly 'pending' only; one item-modification call per \
order. item_ids come from get_order_details; new_item_ids are same-product, \
different, available variants from get_product_details. This tool cannot remove \
line items or partially cancel an order. Use an exact saved payment_method_id \
from get_user_details/get_order_details; bare channel names like "paypal" are \
invalid. Gift-card balance only matters for a positive price increase, not for \
a cheaper replacement or refund/negative difference. If the user directly asks \
to change/exchange a pending item and the exact old/new item IDs are known, call \
this tool; do not add an extra confirmation turn unless the user requested a \
preview first or the final scope is ambiguous.""",
    "modify_pending_order_payment": """\
CONSTRAINTS: New payment method must differ from current one. Single payment \
method only. Gift card must cover the full order total.""",
    "modify_pending_order_address": """\
CONSTRAINTS: Status must be 'pending' or 'pending (item modified)'. Order \
shipping address and profile/default address are separate writes. If the user \
asks for all pending order addresses and the default/profile address, call the \
order-address tool once per eligible pending order and modify_user_address once. \
If the target address is in profile or any past/current order, copy exact fields \
from that record; never invent or submit blank street fields. After the user \
confirms the exact target address and scope, execute the write instead of asking \
for repeated confirmation.""",
    "exchange_delivered_order_items": """\
CONSTRAINTS: Status must be 'delivered'; each order can have one return OR \
exchange. item_ids from get_order_details; new_item_ids are same-product, \
different, available variants. Match hard option constraints exactly; \
directional requests compare to the current item ("less bright" means strictly \
lower brightness). Ranked preferences are priorities after hard constraints. \
If exchanging multiple items from the SAME delivered order, include all of that \
order's item_ids and new_item_ids in ONE call; a second exchange/return on that \
order will be blocked. \
Use an exact saved payment_method_id, not a bare channel name. Gift-card balance \
only matters when the exchange price difference is positive; do not refuse a \
cheaper replacement because the refund amount exceeds current gift-card balance.""",
    "return_delivered_order_items": """\
CONSTRAINTS: Status must be 'delivered'; one return OR exchange per order. \
item_ids from get_order_details. Refund to original payment method or user's \
gift card. Use an exact saved payment_method_id, not a bare channel name. \
Category requests ("gaming/office items") include only matching items, not the \
whole order unless user says everything in that order. available=false does not \
affect returns.""",
    "modify_user_address": """\
NOTE: Updates default profile address only, not order addresses. For \
"all/everywhere/orders too", also update eligible pending orders. If reverting \
or copying an address from profile/order history, use exact fields from earlier \
tool results; never invent or submit blank street fields. After the user \
confirms the exact target address and scope, execute the write instead of asking \
for repeated confirmation.""",
    "transfer_to_human_agents": """\
Transfer only for genuinely out-of-scope requests such as undoing a completed \
cancellation, placing a new order, or another user's account. Normal cancel, \
modify, return/exchange, address lookup/update are in scope. If a policy rule \
blocks the requested action, explain the valid alternative instead of transferring.""",
}

# Retail H3 is intentionally restricted to write/generic tools.  Hints on read
# tools made agents over-scan and over-explain on unseen tasks; read results
# should remain mostly unshaped evidence.
_RETAIL_READ_TOOL_HINTS = {
    "get_user_details",
    "get_order_details",
    "get_product_details",
    "find_user_id_by_email",
    "find_user_id_by_name_zip",
}
_RETAIL_H3_HINTS = {
    name: hint
    for name, hint in _RETAIL_H3_HINTS.items()
    if name not in _RETAIL_READ_TOOL_HINTS
}

# ---------------------------------------------------------------------------
# Generic mixin
# ---------------------------------------------------------------------------


class H3ToolDescriptionMixin:
    """Mixin that appends policy constraints to write-tool descriptions.

    Subclasses set ``_h3_hints`` to ``{tool_name: hint_text}``.
    ``get_tools()`` is overridden to patch each matching Tool's description
    before the OpenAI schema is exposed to the LLM.

    MRO note: place this mixin **before** the domain toolkit class so that
    its ``get_tools()`` runs first and calls ``super().get_tools()``
    downstream::

        class H3HarnessedAirlineTools(H3AirlineToolDescriptionMixin,
                                       HarnessedAirlineTools): ...
    """

    _h3_hints: dict[str, str] = {}

    def get_tools(self, include: list[str] | None = None) -> dict[str, Tool]:
        tools: dict[str, Tool] = super().get_tools(include=include)  # type: ignore[misc]
        for name, hint in self._h3_hints.items():
            if name in tools:
                tools[name] = _append_hint_to_tool(tools[name], hint)
        return tools


# ---------------------------------------------------------------------------
# Airline-specific H3 mixin
# ---------------------------------------------------------------------------


class H3AirlineToolDescriptionMixin(H3ToolDescriptionMixin):
    """H3 hint set for the airline domain."""

    _h3_hints: dict[str, str] = _AIRLINE_H3_HINTS


# ---------------------------------------------------------------------------
# Retail-specific H3 mixin
# ---------------------------------------------------------------------------


class H3RetailToolDescriptionMixin(H3ToolDescriptionMixin):
    """H3 hint set for the retail domain."""

    _h3_hints: dict[str, str] = _RETAIL_H3_HINTS


# ---------------------------------------------------------------------------
# Telecom domain hints
# ---------------------------------------------------------------------------

_TELECOM_H3_HINTS: dict[str, str] = {
    "send_payment_request": """\
• Only for OVERDUE bills. Verify status first.
• One AWAITING_PAYMENT bill at a time.
• After calling: customer uses check_payment_request → make_payment → \
you call resume_line → customer reboots.
• Locked SIM (locked_pin) does NOT block billing — handle billing if needed, \
then transfer for SIM PIN/in-store support.""",
    "resume_line": """\
• All overdue bills must be PAID first. Expired contract → transfer to human.
• After resuming: remind customer to reboot.""",
    "refuel_data": """\
• Line must be Active. Max 2 GB per call. Confirm price + consent first.
• When get_data_usage shows data_used_gb ≥ data_limit_gb + data_refueling_gb \
→ proactively offer refuel. Do NOT transfer instead.""",
    "suspend_line": "• Line must be Active. $5/month fee, max 6 months. Confirm before calling.",
    "enable_roaming": """\
• FREE for customers abroad. Check roaming_enabled via line details first.
• TWO parts: (1) call enable_roaming [account]; (2) customer toggles Data Roaming \
in device settings [device]. Both required.
• If blocked "already enabled" → account is fine; only device toggle needed.""",
    "transfer_to_human_agents": "• Must be an explicit TOOL CALL — do NOT just say 'transfer' in text. \
Use when: contract expired, SIM PIN needs in-store unlock, or issue is beyond tools.",
}


# ---------------------------------------------------------------------------
# Telecom-specific H3 mixin
# ---------------------------------------------------------------------------


class H3TelecomToolDescriptionMixin(H3ToolDescriptionMixin):
    """H3 hint set for the telecom domain."""

    _h3_hints: dict[str, str] = _TELECOM_H3_HINTS


# ---------------------------------------------------------------------------
# Banking-knowledge domain hints
# ---------------------------------------------------------------------------

_BANKING_KNOWLEDGE_H3_HINTS: dict[str, str] = {
    "unlock_discoverable_agent_tool": """\
IMPORTANT: agent_tool_name MUST come from a KB document — never guess it.
Workflow: KB_search → find document with tool_id → call this tool → \
call_discoverable_agent_tool.""",
    "call_discoverable_agent_tool": """\
IMPORTANT: You must call unlock_discoverable_agent_tool first with the same \
agent_tool_name. The tool_id must come from a KB document — never guess it.""",
    "give_discoverable_user_tool": """\
IMPORTANT: discoverable_tool_name must come from a KB document — never guess it.
Call KB_search to find the procedure that names this user-facing tool.""",
    "get_user_information_by_email": """\
If "No records found": try get_user_information_by_name next. \
There is NO phone-number lookup tool.""",
    "get_user_information_by_name": """\
After retrieval, compare the returned email against the email the customer \
stated in this conversation. If they differ, that is an identity verification \
discrepancy — use reason='account_ownership_dispute' if a transfer is needed.""",
    "transfer_to_human_agents": """\
Use the MOST SPECIFIC reason code available. Key codes:
  • kb_search_unsuccessful_customer_requests_transfer — KB_search returned no \
useful result AND customer explicitly asks to be transferred (use this even if \
search returned irrelevant docs — if you couldn't answer the question, it counts \
as unsuccessful)
  • customer_demands_after_unavailable_offer_refusal — customer asked about a \
specific offer/promotion not in the system, was told it isn't available, \
persisted/demanded human; NOT the same as general frustration
  • customer_frustrated_demands_human — customer is generally frustrated or \
demanding a human for reasons OTHER than a specific unavailable offer
  • fraud_or_security_concern — fraud, unauthorized transactions, security issues
  • account_ownership_dispute — identity verification failures, ownership conflicts
  • account_closure_request — customer explicitly requests account closure
  • complex_billing_dispute — billing disputes needing specialist review
  • technical_system_error — system error preventing task completion
Decision rule: if you did KB_search and it gave no useful answer → \
kb_search_unsuccessful_customer_requests_transfer. If customer is angry about \
a promotion you confirmed doesn't exist → \
customer_demands_after_unavailable_offer_refusal. \
When uncertain, KB_search the transfer reason guide before calling.""",
    "log_verification": """\
Use ONLY actual information the user provided in this conversation. \
Never use placeholder values (e.g. 'John Doe', 'usr_12345', 'unknown'). \
If a field was not provided by the user, pass an empty string.""",
    "get_credit_card_accounts_by_user": """\
"No records found" means no linked account was found — it does NOT confirm the \
user lacks a credit card. Ask the user to confirm before escalating or transferring.""",
}


class H3BankingKnowledgeToolDescriptionMixin(H3ToolDescriptionMixin):
    """H3 hint set for the banking_knowledge domain."""

    _h3_hints: dict[str, str] = _BANKING_KNOWLEDGE_H3_HINTS
