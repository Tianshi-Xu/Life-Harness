"""Retail domain harness rules and annotators.

All rules and annotators inspect only DB state (no conversation history),
ensuring replay safety during set_state evaluation.

Policy source: data/tau2/domains/retail/policy.md

H2 Rule index
-------------
cancel_pending_order
  CancelPendingOrderRule      – block cancel on non-pending orders (R1); explicit messages for
                                cancelled / processed / pending (item modified)
  CancelReasonRule            – reason must be 'no longer needed' or 'ordered by mistake' (R5)

modify_pending_order_items
  ModifyPendingOrderRule        – block modify on already-modified / non-pending orders (R2);
                                  explicit messages for cancelled / delivered / processed /
                                  exchange-or-return-in-progress
  NonEmptyItemIdsRule          – item_ids must be non-empty (R17)
  NonEmptyNewItemIdsRule       – new_item_ids must be non-empty when provided (R18)
  ItemIdsNewItemIdsParityRule  – len(item_ids) must equal len(new_item_ids) when both given (R19)
  ValidateModifyItemsExistRule  – all item_ids must actually be in the specified order (R12)
  NoDuplicateItemIdsRule        – item_ids must not contain duplicates beyond order quantity (R15)
  ValidateNewItemIdExistsRule   – all new_item_ids must exist in the product catalog; prevents
                                  hallucinated / fabricated item IDs like "X_new_expensive" (R16)
  ModifyItemsProductTypeRule    – new item must be a variant of the SAME product; cross-product
                                  changes are blocked with an explicit error message (R8)
  ModifySameItemRule            – new_item_id must differ from the old item_id being replaced (R10)

modify_pending_order_payment
  ModifyPaymentSameMethodRule    – new payment method must differ from the current one (R6)
  ModifyPaymentGiftCardBalRule   – gift card must cover the full order total (R7)

exchange_delivered_order_items
  ExchangeDeliveredOrderRule     – block exchange on non-delivered / already-exchanged orders (R3);
                                   pending orders → suggests modify_pending_order_items
  NonEmptyItemIdsRule            – item_ids must be non-empty (R17)
  NonEmptyNewItemIdsRule         – new_item_ids must be non-empty when provided (R18)
  ItemIdsNewItemIdsParityRule    – len(item_ids) == len(new_item_ids) when both given (R19)
  ValidateExchangeItemsExistRule – all item_ids must actually be in the specified order (R13)
  NoDuplicateItemIdsRule         – item_ids must not contain duplicates beyond order quantity (R15)
  ValidateNewItemIdExistsRule    – all new_item_ids must exist in the product catalog;
                                   prevents hallucinated / fabricated item IDs (R16)
  ExchangeItemsProductTypeRule   – new item must be a variant of the SAME product (R9)
  ExchangeSameItemRule           – new_item_id must differ from the old item_id being replaced (R11)

return_delivered_order_items
  ReturnDeliveredOrderRule       – block return on non-delivered orders; refund must go to
                                   original payment method or a gift card (R4);
                                   pending orders → suggests cancel_pending_order
  NonEmptyItemIdsRule            – item_ids must be non-empty (R17)
  ValidateReturnItemsExistRule   – all item_ids must actually be in the specified order (R14)
  NoDuplicateItemIdsRule         – item_ids must not contain duplicates beyond order quantity (R15)

H4 Annotator index
------------------
get_order_details
  OrderEligibilityAnnotator   – append eligibility note for tricky statuses: 'exchange requested',
                                'return requested', 'pending (item modified)'. Silent for normal
                                statuses (delivered / pending / processed / cancelled).

get_item_details
  ItemAvailabilityAnnotator   – when available=False, inject a note clarifying that this only
                                means out of stock for NEW orders; the item can still be
                                returned/exchanged by a customer who already owns it.

exchange_delivered_order_items
  ExchangePriceDiffAnnotator  – highlight the computed exchange_price_difference so the
                                agent remembers to communicate it.
  GiftCardBalanceAnnotator    – after gift-card writes, show the remaining gift card balance.

return_delivered_order_items
  ReturnRefundAnnotator       – highlight the actual refund payment method and amount.
  GiftCardBalanceAnnotator    – after gift-card writes, show the remaining gift card balance.

cancel_pending_order
  CancelRefundAnnotator       – highlight the actual refund payment method and amount.
  GiftCardBalanceAnnotator    – after gift-card writes, show the remaining gift card balance.
"""

import re
from typing import Any

from tau2.domains.retail.data_model import GiftCard, RetailDB
from tau2.domains.retail.tools import RetailTools
from tau2.harness.base import HarnessedToolKitMixin
from tau2.harness.h3_tools import H3RetailToolDescriptionMixin

# ---------------------------------------------------------------------------
# R1 — cancel_pending_order: status guard
# ---------------------------------------------------------------------------


class CancelPendingOrderRule:
    """Block cancel_pending_order when order is not in pending status.

    Provides a more informative error message than the tool's generic check.
    """

    tool_name = "cancel_pending_order"

    def check(self, db: RetailDB, order_id: str, **_: Any) -> None:
        order = db.orders.get(order_id)
        if order is None:
            return
        st = order.status
        if st == "cancelled":
            raise ValueError(
                f"Order {order_id} is already cancelled. It cannot be cancelled again."
            )
        if st == "processed":
            raise ValueError(
                f"Order {order_id} has status 'processed' (no longer a simple pending order). "
                "Cancellation via cancel_pending_order is only for status 'pending'. "
                "If the order is already delivered, use return_delivered_order_items instead."
            )
        if st == "pending (item modified)":
            raise ValueError(
                f"Order {order_id} has status 'pending (item modified)'. "
                "After items have been modified once, this order can no longer be cancelled "
                "with cancel_pending_order."
            )
        if st in ("delivered", "return requested", "exchange requested"):
            raise ValueError(
                f"Order {order_id} has status '{st}' and cannot be cancelled. "
                "Cancellation is only for 'pending' orders. "
                "For a delivered order, use return_delivered_order_items to refund it instead."
            )
        if st != "pending":
            raise ValueError(
                f"Order {order_id} has status '{st}'. "
                "Only orders with status 'pending' can be cancelled."
            )


# ---------------------------------------------------------------------------
# R2 — modify_pending_order_items: status + once-only guard
# ---------------------------------------------------------------------------


class ModifyPendingOrderRule:
    """Block modify_pending_order_items when order is not strictly pending.

    Policy: items can only be modified on pending orders, and only once
    (after modification status becomes "pending (item modified)").
    """

    tool_name = "modify_pending_order_items"

    def check(self, db: RetailDB, order_id: str, **_: Any) -> None:
        order = db.orders.get(order_id)
        if order is None:
            return
        if order.status == "pending (item modified)":
            raise ValueError(
                f"Order {order_id} has already been modified once "
                "(status: 'pending (item modified)'). "
                "Each order can only have its items modified once. "
                "If the user wants further changes, the whole order must be cancelled first."
            )
        if order.status == "cancelled":
            raise ValueError(
                f"Order {order_id} is cancelled. "
                "Line items cannot be modified on a cancelled order."
            )
        if order.status == "delivered":
            raise ValueError(
                f"Order {order_id} has status 'delivered'. "
                "modify_pending_order_items only applies to 'pending' orders. "
                "For delivered orders, use exchange_delivered_order_items or "
                "return_delivered_order_items instead."
            )
        if order.status == "processed":
            raise ValueError(
                f"Order {order_id} has status 'processed'. "
                "Items can only be modified while the order is 'pending'. "
                "If it is already delivered, use exchange or return tools instead."
            )
        if order.status == "exchange requested":
            raise ValueError(
                f"Order {order_id} already has an exchange in progress "
                "(status: 'exchange requested'). "
                "Item lines cannot be changed with modify_pending_order_items."
            )
        if order.status == "return requested":
            raise ValueError(
                f"Order {order_id} already has a return in progress "
                "(status: 'return requested'). "
                "Item lines cannot be changed with modify_pending_order_items."
            )
        if order.status != "pending":
            raise ValueError(
                f"Order {order_id} has status '{order.status}'. "
                "Items can only be modified on orders with status 'pending'."
            )


# ---------------------------------------------------------------------------
# R3 — exchange_delivered_order_items: status + once-only guard
# ---------------------------------------------------------------------------


class ExchangeDeliveredOrderRule:
    """Block exchange_delivered_order_items on non-delivered orders.

    Policy: exchange/return can only be performed on delivered orders,
    and only once per order.
    """

    tool_name = "exchange_delivered_order_items"

    def check(self, db: RetailDB, order_id: str, **_: Any) -> None:
        order = db.orders.get(order_id)
        if order is None:
            return
        if order.status == "exchange requested":
            raise ValueError(
                f"Order {order_id} already has an exchange in progress "
                "(status: 'exchange requested'). "
                "Each order supports EXACTLY ONE action (exchange or return) — "
                "no additional exchanges or returns are possible on this order. "
                "If the user wants to exchange more items, those items must be in "
                "a DIFFERENT order. Call get_user_details to see all orders, then "
                "get_order_details on each to locate the right order."
            )
        if order.status == "return requested":
            raise ValueError(
                f"Order {order_id} already has a return in progress "
                "(status: 'return requested'). "
                "Each order supports EXACTLY ONE action (exchange or return) — "
                "you cannot exchange items from an order that already has a return. "
                "If the user needs to exchange items, those items must be in a "
                "DIFFERENT order. Call get_user_details to see all orders, then "
                "get_order_details on each to find the order that contains those items."
            )
        if order.status in ("pending", "pending (item modified)"):
            raise ValueError(
                f"Order {order_id} has status '{order.status}' — this is a PENDING order. "
                "exchange_delivered_order_items is ONLY for delivered orders. "
                "For pending orders, use modify_pending_order_items instead to change items."
            )
        if order.status != "delivered":
            raise ValueError(
                f"Order {order_id} has status '{order.status}'. "
                "Exchanges can only be requested for orders with status 'delivered'."
            )


# ---------------------------------------------------------------------------
# R4 — return_delivered_order_items: status + refund-method guard
# ---------------------------------------------------------------------------


class ReturnDeliveredOrderRule:
    """Block return_delivered_order_items on non-delivered orders or invalid
    payment method.

    Policy: returns allowed only on delivered orders; refund must go to the
    original payment method or a gift card.
    """

    tool_name = "return_delivered_order_items"

    def check(
        self, db: RetailDB, order_id: str, payment_method_id: str, **_: Any
    ) -> None:
        order = db.orders.get(order_id)
        if order is None:
            return

        if order.status == "return requested":
            raise ValueError(
                f"Order {order_id} already has a return in progress "
                "(status: 'return requested'). Cannot submit another return. "
                "Each order supports EXACTLY ONE action (exchange or return)."
            )
        if order.status == "exchange requested":
            raise ValueError(
                f"Order {order_id} already has an exchange in progress "
                "(status: 'exchange requested'). "
                "Each order supports EXACTLY ONE action (exchange or return) — "
                "you cannot return items from an order that already has an exchange. "
                "If the user needs to return items, those items must be in a "
                "DIFFERENT order. Call get_user_details to see all orders, then "
                "get_order_details on each to find the order that contains those items."
            )
        if order.status in ("pending", "pending (item modified)"):
            raise ValueError(
                f"Order {order_id} has status '{order.status}' — this is a PENDING order. "
                "return_delivered_order_items is ONLY for delivered orders. "
                "For pending orders, use cancel_pending_order to cancel the whole order instead."
            )
        if order.status != "delivered":
            raise ValueError(
                f"Order {order_id} has status '{order.status}'. "
                "Returns can only be requested for orders with status 'delivered'."
            )

        # Validate payment method: must be gift card or original payment method
        user = db.users.get(order.user_id)
        if user is None:
            return
        pm = user.payment_methods.get(payment_method_id)
        if pm is None:
            return  # let the tool handle "payment method not found"

        original_pm_id = (
            order.payment_history[0].payment_method_id
            if order.payment_history
            else None
        )
        if not isinstance(pm, GiftCard) and payment_method_id != original_pm_id:
            raise ValueError(
                f"Refund payment method '{payment_method_id}' is not valid. "
                "Policy: refunds must be issued to the original payment method "
                f"('{original_pm_id}') or ANY gift card the user owns. "
                "A different credit card or PayPal account is NOT allowed."
            )


# ---------------------------------------------------------------------------
# R5 — cancel_pending_order: reason validation
# ---------------------------------------------------------------------------


class CancelReasonRule:
    """Validate that the cancellation reason is one of the two accepted values.

    Policy: the reason must be exactly 'no longer needed' or 'ordered by mistake'.
    """

    tool_name = "cancel_pending_order"

    _VALID_REASONS = frozenset({"no longer needed", "ordered by mistake"})

    def check(self, db: RetailDB, reason: str = "", **_: Any) -> None:
        if reason and reason not in self._VALID_REASONS:
            raise ValueError(
                f"Invalid cancellation reason: '{reason}'. "
                "The reason must be exactly one of:\n"
                "  • 'no longer needed'\n"
                "  • 'ordered by mistake'\n"
                "Ask the user which reason applies if it is not clear."
            )


# ---------------------------------------------------------------------------
# R6 — modify_pending_order_payment: same-method guard
# ---------------------------------------------------------------------------


class ModifyPaymentSameMethodRule:
    """Block modify_pending_order_payment when the new method equals the current one.

    Policy: the user can only switch to a DIFFERENT payment method.
    """

    tool_name = "modify_pending_order_payment"

    def check(
        self, db: RetailDB, order_id: str, payment_method_id: str = "", **_: Any
    ) -> None:
        order = db.orders.get(order_id)
        if order is None or not order.payment_history:
            return
        current = order.payment_history[0]
        if (
            current.transaction_type == "payment"
            and current.payment_method_id == payment_method_id
        ):
            raise ValueError(
                f"Payment method '{payment_method_id}' is already the current "
                f"payment method for order {order_id}. "
                "The new payment method must be DIFFERENT from the current one."
            )


# ---------------------------------------------------------------------------
# R7 — modify_pending_order_payment: gift card balance guard
# ---------------------------------------------------------------------------


class ModifyPaymentGiftCardBalRule:
    """Check that a gift card has sufficient balance before modifying payment.

    Policy: if the user wants to switch to a gift card, it must cover the full order total.
    """

    tool_name = "modify_pending_order_payment"

    def check(
        self, db: RetailDB, order_id: str, payment_method_id: str = "", **_: Any
    ) -> None:
        order = db.orders.get(order_id)
        if order is None or not order.payment_history:
            return
        if (
            len(order.payment_history) != 1
            or order.payment_history[0].transaction_type != "payment"
        ):
            return
        user = db.users.get(order.user_id)
        if user is None:
            return
        pm = user.payment_methods.get(payment_method_id)
        if pm is None or not isinstance(pm, GiftCard):
            return

        amount = order.payment_history[0].amount
        if pm.balance < amount:
            raise ValueError(
                f"Insufficient gift card balance: '{payment_method_id}' has "
                f"${pm.balance:.2f} but the order total is ${amount:.2f}. "
                "Please choose a different payment method or a gift card with "
                "enough balance."
            )


# ---------------------------------------------------------------------------
# R8 — modify_pending_order_items: same-product-type guard
# ---------------------------------------------------------------------------


class ModifyItemsProductTypeRule:
    """Block modify_pending_order_items when new item belongs to a different product.

    Policy: each item can only be changed to a variant of the SAME product type;
    cross-product modifications (e.g. shirt → shoes) are not allowed.
    """

    tool_name = "modify_pending_order_items"

    def check(
        self,
        db: RetailDB,
        order_id: str,
        item_ids: list | None = None,
        new_item_ids: list | None = None,
        **_: Any,
    ) -> None:
        if not item_ids or not new_item_ids:
            return
        order = db.orders.get(order_id)
        if order is None:
            return

        for old_item_id, new_item_id in zip(item_ids, new_item_ids):
            # Find the old order item
            old_order_item = next(
                (it for it in order.items if it.item_id == old_item_id), None
            )
            if old_order_item is None:
                continue  # let the tool raise its own "item not found"

            old_product_id = old_order_item.product_id
            old_product = db.products.get(old_product_id)
            if old_product is None:
                continue

            # If new item is NOT a variant of the same product → policy violation
            if new_item_id not in old_product.variants:
                new_product_name = _find_product_name(db, new_item_id)
                raise ValueError(
                    f"Cannot change item '{old_item_id}' "
                    f"({old_order_item.name}, product '{old_product_id}') "
                    f"to item '{new_item_id}' "
                    f"({new_product_name}): "
                    "modifications must stay within the SAME product type. "
                    "You can only swap between different variants/options of "
                    "the same product (e.g. different colour or size). "
                    "Changing to a completely different product is not allowed."
                )


# ---------------------------------------------------------------------------
# R9 — exchange_delivered_order_items: same-product-type guard
# ---------------------------------------------------------------------------


class ExchangeItemsProductTypeRule:
    """Block exchange_delivered_order_items when new item belongs to a different product.

    Policy: each exchanged item must be replaced by a variant of the SAME product type.
    """

    tool_name = "exchange_delivered_order_items"

    def check(
        self,
        db: RetailDB,
        order_id: str,
        item_ids: list | None = None,
        new_item_ids: list | None = None,
        **_: Any,
    ) -> None:
        if not item_ids or not new_item_ids:
            return
        order = db.orders.get(order_id)
        if order is None:
            return

        for old_item_id, new_item_id in zip(item_ids, new_item_ids):
            old_order_item = next(
                (it for it in order.items if it.item_id == old_item_id), None
            )
            if old_order_item is None:
                continue

            old_product_id = old_order_item.product_id
            old_product = db.products.get(old_product_id)
            if old_product is None:
                continue

            if new_item_id not in old_product.variants:
                new_product_name = _find_product_name(db, new_item_id)
                raise ValueError(
                    f"Cannot exchange item '{old_item_id}' "
                    f"({old_order_item.name}, product '{old_product_id}') "
                    f"for item '{new_item_id}' "
                    f"({new_product_name}): "
                    "exchanges must stay within the SAME product type. "
                    "You can only exchange for a different variant/option of "
                    "the same product (e.g. different colour or size). "
                    "Changing to a completely different product is not allowed."
                )


# ---------------------------------------------------------------------------
# R12 — modify_pending_order_items: validate item_ids exist in order
# ---------------------------------------------------------------------------


class ValidateModifyItemsExistRule:
    """Validate that all item_ids passed to modify_pending_order_items exist in the order.

    Provides a clear error listing the valid item IDs with names rather than
    the tool's generic 'item not found' message, helping the agent self-correct.
    """

    tool_name = "modify_pending_order_items"

    def check(
        self,
        db: RetailDB,
        order_id: str,
        item_ids: list | None = None,
        **_: Any,
    ) -> None:
        if not item_ids:
            return
        order = db.orders.get(order_id)
        if order is None:
            return
        order_item_ids = {it.item_id for it in order.items}
        bad = [iid for iid in item_ids if iid not in order_item_ids]
        if bad:
            valid = [
                f"{it.item_id} ({it.name})" for it in order.items
            ]
            raise ValueError(
                f"Item(s) {bad} not found in order {order_id}. "
                f"Items in this order: {valid}. "
                "Use the exact item_id from this list."
            )


# ---------------------------------------------------------------------------
# R13 — exchange_delivered_order_items: validate item_ids exist in order
# ---------------------------------------------------------------------------


class ValidateExchangeItemsExistRule:
    """Validate that all item_ids passed to exchange_delivered_order_items exist in the order."""

    tool_name = "exchange_delivered_order_items"

    def check(
        self,
        db: RetailDB,
        order_id: str,
        item_ids: list | None = None,
        **_: Any,
    ) -> None:
        if not item_ids:
            return
        order = db.orders.get(order_id)
        if order is None:
            return
        order_item_ids = {it.item_id for it in order.items}
        bad = [iid for iid in item_ids if iid not in order_item_ids]
        if bad:
            valid = [
                f"{it.item_id} ({it.name})" for it in order.items
            ]
            raise ValueError(
                f"Item(s) {bad} not found in order {order_id}. "
                f"Items in this order: {valid}. "
                "Use the exact item_id from this list."
            )


# ---------------------------------------------------------------------------
# R14 — return_delivered_order_items: validate item_ids exist in order
# ---------------------------------------------------------------------------


class ValidateReturnItemsExistRule:
    """Validate that all item_ids passed to return_delivered_order_items exist in the order."""

    tool_name = "return_delivered_order_items"

    def check(
        self,
        db: RetailDB,
        order_id: str,
        item_ids: list | None = None,
        **_: Any,
    ) -> None:
        if not item_ids:
            return
        order = db.orders.get(order_id)
        if order is None:
            return
        order_item_ids = {it.item_id for it in order.items}
        bad = [iid for iid in item_ids if iid not in order_item_ids]
        if bad:
            valid = [
                f"{it.item_id} ({it.name})" for it in order.items
            ]
            raise ValueError(
                f"Item(s) {bad} not found in order {order_id}. "
                f"Items in this order: {valid}. "
                "Use the exact item_id from this list."
            )


# ---------------------------------------------------------------------------
# R10 — modify_pending_order_items: same-item guard
# ---------------------------------------------------------------------------


class ModifySameItemRule:
    """Block modify_pending_order_items when a new_item_id equals the old item_id.

    Policy: the replacement variant must be a *different* item from the current one.
    """

    tool_name = "modify_pending_order_items"

    def check(
        self,
        db: RetailDB,
        item_ids: list | None = None,
        new_item_ids: list | None = None,
        **_: Any,
    ) -> None:
        if not item_ids or not new_item_ids:
            return
        for old_id, new_id in zip(item_ids, new_item_ids):
            if old_id == new_id:
                raise ValueError(
                    f"New item ID '{new_id}' is the same as the current item ID. "
                    "The replacement must be a DIFFERENT variant. "
                    "Use get_product_details to find other available variants of the same product."
                )


# ---------------------------------------------------------------------------
# R11 — exchange_delivered_order_items: same-item guard
# ---------------------------------------------------------------------------


class ExchangeSameItemRule:
    """Block exchange_delivered_order_items when a new_item_id equals the old item_id.

    Policy: the exchange target must be a *different* item from the current one.
    """

    tool_name = "exchange_delivered_order_items"

    def check(
        self,
        db: RetailDB,
        item_ids: list | None = None,
        new_item_ids: list | None = None,
        **_: Any,
    ) -> None:
        if not item_ids or not new_item_ids:
            return
        for old_id, new_id in zip(item_ids, new_item_ids):
            if old_id == new_id:
                raise ValueError(
                    f"New item ID '{new_id}' is the same as the current item ID. "
                    "The exchange target must be a DIFFERENT variant. "
                    "Use get_product_details to find other available variants of the same product."
                )


# ---------------------------------------------------------------------------
# R17 — non-empty item_ids (modify / exchange / return)
# ---------------------------------------------------------------------------


class NonEmptyItemIdsRule:
    """Reject empty item_ids before heavier checks."""

    tool_name: str = ""

    def check(
        self,
        db: RetailDB,
        item_ids: list | None = None,
        **_: Any,
    ) -> None:
        if item_ids is None:
            return
        if len(item_ids) == 0:
            raise ValueError(
                "item_ids cannot be empty. "
                "Specify at least one line item from the order "
                "(use get_order_details to list valid item IDs)."
            )


# ---------------------------------------------------------------------------
# R18 — non-empty new_item_ids (modify / exchange)
# ---------------------------------------------------------------------------


class NonEmptyNewItemIdsRule:
    """Reject empty new_item_ids when the parameter is supplied."""

    tool_name: str = ""

    def check(
        self,
        db: RetailDB,
        new_item_ids: list | None = None,
        **_: Any,
    ) -> None:
        if new_item_ids is None:
            return
        if len(new_item_ids) == 0:
            raise ValueError(
                "new_item_ids cannot be empty. "
                "Each position must name the replacement variant "
                "(use get_product_details to find valid item IDs)."
            )


# ---------------------------------------------------------------------------
# R19 — len(item_ids) == len(new_item_ids) (modify / exchange)
# ---------------------------------------------------------------------------


class ItemIdsNewItemIdsParityRule:
    """Ensure parallel lists have the same length (no silent zip truncation)."""

    tool_name: str = ""

    def check(
        self,
        db: RetailDB,
        item_ids: list | None = None,
        new_item_ids: list | None = None,
        **_: Any,
    ) -> None:
        if not item_ids or not new_item_ids:
            return
        if len(item_ids) != len(new_item_ids):
            raise ValueError(
                f"item_ids has length {len(item_ids)} but new_item_ids has length "
                f"{len(new_item_ids)} — they must match one-to-one. "
                "Each old item_id needs exactly one new_item_id in the same position."
            )


# ---------------------------------------------------------------------------
# R16 — modify / exchange: new_item_id must exist in product catalog
# ---------------------------------------------------------------------------


class ValidateNewItemIdExistsRule:
    """Block modify/exchange calls where any new_item_id does not exist in the DB.

    A common agent hallucination is fabricating item IDs (e.g. appending
    "_new_expensive" to an existing ID, or making up a numeric ID).
    This rule checks every new_item_id against the full product catalog and
    provides a helpful error message directing the agent to use
    get_product_details to find valid IDs.
    """

    tool_name: str = ""  # set per instance

    def check(
        self,
        db: RetailDB,
        new_item_ids: list | None = None,
        **_: Any,
    ) -> None:
        if not new_item_ids:
            return
        # Build a flat set of all valid item_ids across all products
        all_valid_ids: set[str] = {
            iid for prod in db.products.values() for iid in prod.variants
        }
        bad = [nid for nid in new_item_ids if nid not in all_valid_ids]
        if bad:
            raise ValueError(
                f"new_item_id(s) {bad} do not exist in the product catalog. "
                "You MUST call get_product_details on the relevant product to find "
                "valid variant item IDs before calling this tool. "
                "Never guess, fabricate, or modify an existing item ID."
            )


# ---------------------------------------------------------------------------
# R15 — exchange / return / modify: duplicate item_ids guard
# ---------------------------------------------------------------------------


class NoDuplicateItemIdsRule:
    """Block calls where item_ids contains duplicate entries that don't reflect
    the actual order contents.

    A common agent error is passing the same item_id twice (e.g. treating one
    physical item as if it were two). This rule validates that duplicates in
    item_ids only appear as many times as the item genuinely exists in the order.
    """

    tool_name: str = ""  # set per instance

    def check(
        self,
        db: RetailDB,
        order_id: str,
        item_ids: list | None = None,
        **_: Any,
    ) -> None:
        if not item_ids or len(item_ids) == len(set(item_ids)):
            return  # no duplicates at all → fine

        order = db.orders.get(order_id)
        if order is None:
            return

        # Count how many times each item_id genuinely appears in the order
        from collections import Counter

        order_counts = Counter(it.item_id for it in order.items)
        requested_counts = Counter(item_ids)

        bad = [
            iid
            for iid, cnt in requested_counts.items()
            if cnt > order_counts.get(iid, 0)
        ]
        if bad:
            raise ValueError(
                f"Duplicate item_id(s) in the request: {bad}. "
                "Each item_id should appear only as many times as it exists in the order. "
                f"Order {order_id} contains: "
                f"{dict(order_counts)}. "
                "If the user wants to process the same product from a DIFFERENT order, "
                "make a separate call for that order."
            )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _find_product_name(db: RetailDB, item_id: str) -> str:
    """Return '<product_name>' for the product that contains *item_id*, or 'unknown product'."""
    for prod in db.products.values():
        if item_id in prod.variants:
            return f"'{prod.name}'"
    return "unknown product"


# ---------------------------------------------------------------------------
# H2 Composite toolkit classes
# ---------------------------------------------------------------------------

_H2_RULES: dict[str, list] = {
    "cancel_pending_order": [
        CancelPendingOrderRule(),
        CancelReasonRule(),
    ],
    "modify_pending_order_items": [
        ModifyPendingOrderRule(),
        NonEmptyItemIdsRule(),
        NonEmptyNewItemIdsRule(),
        ItemIdsNewItemIdsParityRule(),
        ValidateModifyItemsExistRule(),
        NoDuplicateItemIdsRule(),
        ValidateNewItemIdExistsRule(),
        ModifyItemsProductTypeRule(),
        ModifySameItemRule(),
    ],
    "modify_pending_order_payment": [
        ModifyPaymentSameMethodRule(),
        ModifyPaymentGiftCardBalRule(),
    ],
    "exchange_delivered_order_items": [
        ExchangeDeliveredOrderRule(),
        NonEmptyItemIdsRule(),
        NonEmptyNewItemIdsRule(),
        ItemIdsNewItemIdsParityRule(),
        ValidateExchangeItemsExistRule(),
        NoDuplicateItemIdsRule(),
        ValidateNewItemIdExistsRule(),
        ExchangeItemsProductTypeRule(),
        ExchangeSameItemRule(),
    ],
    "return_delivered_order_items": [
        ReturnDeliveredOrderRule(),
        NonEmptyItemIdsRule(),
        ValidateReturnItemsExistRule(),
        NoDuplicateItemIdsRule(),
    ],
}


class HarnessedRetailTools(HarnessedToolKitMixin, RetailTools):
    """RetailTools with pre-execution harness validation (H2) enabled."""

    harness_rules: dict[str, list] = _H2_RULES


class H3RetailTools(H3RetailToolDescriptionMixin, RetailTools):
    """RetailTools with H3 tool-description policy hints enabled."""


class H3HarnessedRetailTools(H3RetailToolDescriptionMixin, HarnessedRetailTools):
    """RetailTools with both H2 harness rules and H3 tool-description hints."""


# ---------------------------------------------------------------------------
# H4 Annotators
# ---------------------------------------------------------------------------


class OrderEligibilityAnnotator:
    """H4: Append an eligibility note when get_order_details returns a 'tricky'
    order status that commonly causes agent mistakes.

    Only fires for: 'exchange requested', 'return requested',
    'pending (item modified)'.  Silent for all other statuses to avoid noise.
    """

    tool_name = "get_order_details"

    def annotate(self, db: RetailDB, result: Any, **_: Any) -> str | None:
        st = getattr(result, "status", None)
        oid = getattr(result, "order_id", "?")
        if st == "exchange requested":
            return (
                f"Note: Order {oid} already has an exchange in progress "
                "(status: 'exchange requested'). Each order allows exactly one "
                "exchange or return — no further exchange or return is possible "
                "on this order."
            )
        if st == "return requested":
            return (
                f"Note: Order {oid} already has a return in progress "
                "(status: 'return requested'). Each order allows exactly one "
                "exchange or return — no further exchange or return is possible "
                "on this order."
            )
        if st == "pending (item modified)":
            return (
                f"Note: Order {oid} items have already been modified once "
                "(status: 'pending (item modified)'). Items cannot be modified "
                "a second time. Address or payment method changes and full-order "
                "cancellation are still available."
            )
        return None


class OrderStatusRoutingAnnotator:
    """H4: Surface the valid write route for normal pending/delivered statuses.

    This is intentionally short and factual. It prevents status inversions such
    as treating a pending order as non-cancellable, without pushing the agent to
    act on orders the user did not request.
    """

    tool_name = "get_order_details"

    def annotate(self, db: RetailDB, result: Any, **_: Any) -> str | None:
        st = getattr(result, "status", None)
        oid = getattr(result, "order_id", "?")
        if st == "pending":
            return (
                f"Status route for {oid}: pending orders can be cancelled as a "
                "whole order with cancel_pending_order, or changed with "
                "modify_pending_order_items/address/payment if the user requested it."
            )
        if st == "delivered":
            return (
                f"Status route for {oid}: delivered orders can be returned with "
                "return_delivered_order_items or exchanged with "
                "exchange_delivered_order_items if the user requested it."
            )
        return None


class ExchangeReturnCompletionAnnotator:
    """H6: After a successful exchange or return, list the user's other delivered
    orders that are still eligible for exchange/return.

    Fires only when at least one such order exists.  Uses strong negative
    phrasing to prevent the agent from acting on orders the user did not
    explicitly mention (task-98-style false-positive interference).
    """

    tool_name: str  # set per instance

    def annotate(self, db: RetailDB, result: Any, **_: Any) -> str | None:
        user_id = getattr(result, "user_id", None)
        done_order = getattr(result, "order_id", None)
        if not user_id:
            return None
        eligible = [
            o.order_id
            for o in db.orders.values()
            if o.user_id == user_id
            and o.status == "delivered"
            and o.order_id != done_order
        ]
        if not eligible:
            return None
        ids = ", ".join(eligible)
        return (
            f"FYI: {len(eligible)} other delivered order(s) exist for this user: {ids}. "
            "Do NOT take any action on these orders unless the user has "
            "explicitly named items from them in their request."
        )


class AddressWriteCompletionAnnotator:
    """H6: After a successful address write, list the user's other pending
    orders that may also need an address update.

    Fires only when at least one such order exists.  Uses soft conditional
    phrasing so the agent decides whether to sync them.
    """

    tool_name: str  # set per instance

    def annotate(self, db: RetailDB, result: Any, **_: Any) -> str | None:
        user_id = getattr(result, "user_id", None)
        done_order = getattr(result, "order_id", None)  # None for modify_user_address
        if not user_id:
            return None
        pending = [
            f"{o.order_id} ({o.status})"
            for o in db.orders.values()
            if o.user_id == user_id
            and o.status in ("pending", "pending (item modified)")
            and o.order_id != done_order
        ]
        if not pending:
            return None
        return (
            f"This user has {len(pending)} other pending order(s): "
            f"{', '.join(pending)}. "
            "Update their addresses only if the user explicitly asked to sync "
            "all order addresses."
        )


class ExchangePriceDiffAnnotator:
    """H4 / H2-success-path: After a successful exchange, highlight the
    exchange_price_difference already stored on the order so the agent
    remembers to communicate it to the user.
    """

    tool_name = "exchange_delivered_order_items"

    def annotate(self, db: RetailDB, result: Any, **_: Any) -> str | None:
        diff = getattr(result, "exchange_price_difference", None)
        if diff is None:
            return None
        abs_diff = abs(diff)
        if diff > 0:
            direction = f"customer pays ${abs_diff:.2f} extra"
        elif diff < 0:
            direction = f"customer saves ${abs_diff:.2f}"
        else:
            direction = "no price change"
        return (
            f"Price difference for this exchange: ${diff:+.2f} ({direction}). "
            "Please communicate this amount to the user."
        )


class CancelCompletionAnnotator:
    """A7: After a successful cancel_pending_order, list other pending orders
    the user may also want to cancel.

    Fires only when at least one other pending order exists.  Uses soft
    conditional phrasing to avoid spurious cancellations.
    """

    tool_name = "cancel_pending_order"

    def annotate(self, db: RetailDB, result: Any, **_: Any) -> str | None:
        user_id = getattr(result, "user_id", None)
        done_order = getattr(result, "order_id", None)
        if not user_id:
            return None
        pending = [
            o.order_id
            for o in db.orders.values()
            if o.user_id == user_id
            and o.status in ("pending", "pending (item modified)")
            and o.order_id != done_order
        ]
        if not pending:
            return None
        ids = ", ".join(pending)
        return (
            f"This user has {len(pending)} other pending order(s): {ids}. "
            "Cancel them only if the user explicitly requested."
        )


class ReturnRefundAnnotator:
    """A8: After a successful return_delivered_order_items, inject the total
    refund amount so the agent communicates it to the user.

    Calculates refund by summing the purchase prices of the returned items
    directly from the order — no extra DB look-ups required.
    """

    tool_name = "return_delivered_order_items"

    def annotate(self, db: RetailDB, result: Any, **_: Any) -> str | None:
        return_item_ids = getattr(result, "return_items", None)
        if not return_item_ids:
            return None
        items = getattr(result, "items", [])
        refund = sum(
            item.price for item in items if item.item_id in return_item_ids
        )
        if refund == 0:
            return None
        payment_id = (
            getattr(result, "return_payment_method_id", None)
            or "original payment method"
        )
        return (
            f"Refund for returned item(s): ${refund:.2f} (via {payment_id}). "
            "Please communicate this refund amount to the user."
        )


class TrackingNumberAnnotator:
    """A9: When get_order_details returns a shipped order, surface the
    tracking number(s) so the agent can pass them to the user on request.

    Fires only when fulfillments contain at least one tracking ID (i.e., the
    order has actually been shipped).  Silent for pending/cancelled orders.
    """

    tool_name = "get_order_details"

    def annotate(self, db: RetailDB, result: Any, **_: Any) -> str | None:
        fulfillments = getattr(result, "fulfillments", None)
        if not fulfillments:
            return None
        tracking_ids = [
            tid
            for f in fulfillments
            for tid in (f.tracking_id if hasattr(f, "tracking_id") else [])
        ]
        if not tracking_ids:
            return None
        ids_str = ", ".join(tracking_ids)
        return (
            f"Tracking number(s) for this order: {ids_str}. "
            "Communicate to the user if they ask about delivery status or tracking."
        )


class OrderTotalAnnotator:
    """A10: When get_order_details returns an order that has been paid,
    surface the total amount paid so the agent can communicate it on request.

    Calculates the total from payment_history (sum of 'payment' transactions).
    Fires for any order with at least one payment entry.  Silent for orders
    with no payment history (e.g. free orders or edge-cases).
    """

    tool_name = "get_order_details"

    def annotate(self, db: RetailDB, result: Any, **_: Any) -> str | None:
        payment_history = getattr(result, "payment_history", None)
        if not payment_history:
            return None
        total = sum(
            p.amount
            for p in payment_history
            if getattr(p, "transaction_type", None) == "payment"
        ) - sum(
            p.amount
            for p in payment_history
            if getattr(p, "transaction_type", None) == "refund"
        )
        if total == 0:
            return None
        return (
            f"Total amount paid for this order: ${total:.2f}. "
            "Communicate to the user if they ask about the order total or costs."
        )


class ReturnRemainingItemsTotalAnnotator:
    """A12: After a successful return, surface the total paid for the
    non-returned items still in the order.

    Useful when the user asks "how much did I pay for what I kept?" or the
    task requires communicating the sub-total for remaining items.
    Fires only when at least one item is NOT being returned.
    """

    tool_name = "return_delivered_order_items"

    def annotate(self, db: RetailDB, result: Any, **_: Any) -> str | None:
        return_item_ids = list(getattr(result, "return_items", None) or [])
        items = list(getattr(result, "items", []))
        if not return_item_ids or not items:
            return None
        # Use a counter to handle duplicate item_ids correctly
        from collections import Counter

        returned_counts = Counter(return_item_ids)
        remaining: list[Any] = []
        for it in items:
            if returned_counts[it.item_id] > 0:
                returned_counts[it.item_id] -= 1
            else:
                remaining.append(it)
        if not remaining:
            return None
        total = sum(it.price for it in remaining)
        if total == 0:
            return None
        names = ", ".join(it.name for it in remaining[:3])
        if len(remaining) > 3:
            names += f" (+{len(remaining) - 3} more)"
        return (
            f"Total paid for remaining (non-returned) item(s) in this order: "
            f"${total:.2f} ({names}). "
            "Communicate if the user asks about remaining item costs."
        )


class GiftCardBalanceAnnotator:
    """A13: After any write that charges or refunds a gift card, surface the
    updated gift card balance so the agent can communicate it.

    Works for return_delivered_order_items, exchange_delivered_order_items,
    and cancel_pending_order — each uses a different field to identify the
    payment method, so we check in order of specificity.
    Fires only when the relevant payment method is actually a gift card.
    """

    tool_name: str  # set per instance

    def annotate(self, db: RetailDB, result: Any, **_: Any) -> str | None:
        user_id = getattr(result, "user_id", None)
        if not user_id:
            return None

        # Try to find the gift-card payment method used in this operation
        pm_id = (
            getattr(result, "return_payment_method_id", None)
            or getattr(result, "exchange_payment_method_id", None)
            or next(
                (
                    p.payment_method_id
                    for p in (getattr(result, "payment_history", None) or [])
                    if getattr(p, "transaction_type", None) == "payment"
                ),
                None,
            )
        )
        if not pm_id:
            return None

        user = db.users.get(user_id)
        if user is None:
            return None
        pm = user.payment_methods.get(pm_id)
        if pm is None or getattr(pm, "source", None) != "gift_card":
            return None

        balance = getattr(pm, "balance", None)
        if balance is None:
            return None
        return (
            f"Updated gift card balance ({pm_id}): ${balance:.2f}. "
            "Communicate this updated balance to the user if they ask."
        )


class ProductVariantSortAnnotator:
    """A15: After get_product_details, inject a price-sorted variant summary.

    Helps the agent correctly identify the cheapest / most expensive variant
    without manually parsing the unordered variants dict.  Fires whenever the
    product has ≥2 variants.
    """

    tool_name = "get_product_details"

    def annotate(self, db: RetailDB, result: Any, **_: Any) -> str | None:
        variants = getattr(result, "variants", None)
        if not variants or len(variants) < 2:
            return None

        sorted_v = sorted(variants.values(), key=lambda v: v.price)
        lines: list[str] = [
            f"Variants sorted by price (cheapest → most expensive):"
        ]
        for v in sorted_v:
            opts = ", ".join(f"{k}: {val}" for k, val in v.options.items())
            avail_tag = "" if v.available else " [out of stock for new orders]"
            lines.append(f"  ${v.price:.2f}  item_id={v.item_id}  ({opts}){avail_tag}")

        cheapest = sorted_v[0]
        priciest = sorted_v[-1]
        cheapest_avail = next((v for v in sorted_v if v.available), None)
        priciest_avail = next((v for v in reversed(sorted_v) if v.available), None)

        lines.append(
            f"Cheapest overall: ${cheapest.price:.2f} (item_id={cheapest.item_id})"
        )
        lines.append(
            f"Most expensive overall: ${priciest.price:.2f} (item_id={priciest.item_id})"
        )
        if cheapest_avail and cheapest_avail.item_id != cheapest.item_id:
            lines.append(
                f"Cheapest currently in stock: "
                f"${cheapest_avail.price:.2f} (item_id={cheapest_avail.item_id})"
            )
        if priciest_avail and priciest_avail.item_id != priciest.item_id:
            lines.append(
                f"Most expensive currently in stock: "
                f"${priciest_avail.price:.2f} (item_id={priciest_avail.item_id})"
            )
        lines.append(
            "Use price ranking only when the user asks for cheapest/most expensive. "
            "Otherwise, first match every user-stated option constraint exactly."
        )
        lines.append(
            "Note: 'out of stock for new orders' does NOT prevent exchange/return "
            "of an item the customer already owns."
        )
        return "\n".join(lines)


class ProductVariantExtremaAnnotator:
    """H4: Compact cheapest / most expensive available-variant summary."""

    tool_name = "get_product_details"

    def annotate(self, db: RetailDB, result: Any, **_: Any) -> str | None:
        variants = getattr(result, "variants", None)
        if not variants or len(variants) < 2:
            return None

        available = [
            v for v in variants.values() if getattr(v, "available", False)
        ]
        if len(available) < 2:
            return None

        def fmt_variant(v: Any) -> str:
            opts = ", ".join(f"{k}: {val}" for k, val in v.options.items())
            return f"item_id={v.item_id} ${v.price:.2f} ({opts})"

        cheapest = min(available, key=lambda v: v.price)
        priciest = max(available, key=lambda v: v.price)
        product_name = getattr(result, "name", "this product")
        return (
            f"Available price extrema for {product_name}: "
            f"cheapest available {fmt_variant(cheapest)}; "
            f"most expensive available {fmt_variant(priciest)}. "
            "Use these only for cheapest/most expensive requests; otherwise "
            "match hard user option constraints first."
        )


class ProductNumericOptionExtremaAnnotator:
    """H4: Compact min/max summary for numeric variant options."""

    tool_name = "get_product_details"

    _NUMERIC_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([a-zA-Z]+)?")

    def _numeric_value(self, raw: Any) -> float | None:
        text = str(raw).lower()
        match = self._NUMERIC_RE.search(text)
        if not match:
            return None

        value = float(match.group(1))
        unit = match.group(2) or ""
        if unit == "tb":
            return value * 1024
        if unit in {"gb", "mp", "x", "inch", "inches", "piece", "pieces"}:
            return value
        return value

    def annotate(self, db: RetailDB, result: Any, **_: Any) -> str | None:
        variants = getattr(result, "variants", None)
        if not variants or len(variants) < 2:
            return None

        available = [
            v for v in variants.values() if getattr(v, "available", False)
        ]
        if len(available) < 2:
            return None

        by_option: dict[str, list[tuple[float, Any, Any]]] = {}
        for variant in available:
            for key, raw in getattr(variant, "options", {}).items():
                value = self._numeric_value(raw)
                if value is None:
                    continue
                by_option.setdefault(key, []).append((value, raw, variant))

        lines: list[str] = []
        for key, entries in by_option.items():
            values = {value for value, _, _ in entries}
            if len(values) < 2:
                continue
            low = min(entries, key=lambda item: item[0])
            high = max(entries, key=lambda item: item[0])
            low_variant = low[2]
            high_variant = high[2]
            lines.append(
                f"{key}: min {low[1]} item_id={low_variant.item_id}; "
                f"max {high[1]} item_id={high_variant.item_id}"
            )
            if len(lines) >= 4:
                break

        if not lines:
            return None

        return (
            "Numeric option extrema among available variants: "
            + "; ".join(lines)
            + ". For max/min/larger/smaller requests, compare these option "
            "values directly and still preserve every hard user option constraint."
        )


class ProductAvailableCountAnnotator:
    """H4: Provide the exact available=true variant count after product lookup."""

    tool_name = "get_product_details"

    def annotate(self, db: RetailDB, result: Any, **_: Any) -> str | None:
        variants = getattr(result, "variants", None)
        if not variants:
            return None
        available = [
            v for v in variants.values() if getattr(v, "available", False)
        ]
        product_name = getattr(result, "name", "this product")
        return (
            f"Available variant count for {product_name}: {len(available)} "
            "variant(s) with available=true. Use this count when the user asks "
            "how many options are currently available."
        )


class ItemAvailabilityAnnotator:
    """A16: After get_item_details, clarify when available=False.

    Prevents agents from incorrectly refusing returns/exchanges on the grounds
    that the item is 'out of stock'.  Fires only when available=False.
    """

    tool_name = "get_item_details"

    def annotate(self, db: RetailDB, result: Any, **_: Any) -> str | None:
        if getattr(result, "available", True):
            return None
        item_id = getattr(result, "item_id", "?")
        return (
            f"Note: item {item_id} shows available=false, which means this "
            "product variant is currently OUT OF STOCK for NEW purchases. "
            "This does NOT affect returnability or exchangeability of an item "
            "the customer has already received. A customer can ALWAYS return "
            "or exchange an item in their delivered order regardless of current "
            "stock status."
        )


class CancelRefundAnnotator:
    """A14: After a successful cancel_pending_order, surface the refund amount
    (the original amount paid) so the agent communicates it to the user.

    Mirrors A8 (ReturnRefundAnnotator) for the cancel path.  The refund always
    goes to the original payment method and equals the original payment amount.
    """

    tool_name = "cancel_pending_order"

    def annotate(self, db: RetailDB, result: Any, **_: Any) -> str | None:
        payment_history = getattr(result, "payment_history", None)
        if not payment_history:
            return None
        refund = sum(
            p.amount
            for p in payment_history
            if getattr(p, "transaction_type", None) == "payment"
        ) - sum(
            p.amount
            for p in payment_history
            if getattr(p, "transaction_type", None) == "refund"
        )
        if refund == 0:
            return None
        payment_id = next(
            (
                p.payment_method_id
                for p in payment_history
                if p.transaction_type == "payment"
            ),
            "original payment method",
        )
        return (
            f"Refund for this cancellation: ${refund:.2f} (to {payment_id}). "
            "Please communicate this refund amount to the user."
        )


# ---------------------------------------------------------------------------
# H4 Mixin and composite toolkit classes
# ---------------------------------------------------------------------------

_EXCHANGE_RETURN_ANNOTATOR = ExchangeReturnCompletionAnnotator()
_EXCHANGE_RETURN_ANNOTATOR.tool_name = "exchange_delivered_order_items"

_RETURN_COMPLETION_ANNOTATOR = ExchangeReturnCompletionAnnotator()
_RETURN_COMPLETION_ANNOTATOR.tool_name = "return_delivered_order_items"

_ADDRESS_MODIFY_USER_ANNOTATOR = AddressWriteCompletionAnnotator()
_ADDRESS_MODIFY_USER_ANNOTATOR.tool_name = "modify_user_address"

_ADDRESS_MODIFY_ORDER_ANNOTATOR = AddressWriteCompletionAnnotator()
_ADDRESS_MODIFY_ORDER_ANNOTATOR.tool_name = "modify_pending_order_address"

_GC_RETURN_ANNOTATOR = GiftCardBalanceAnnotator()
_GC_RETURN_ANNOTATOR.tool_name = "return_delivered_order_items"

_GC_EXCHANGE_ANNOTATOR = GiftCardBalanceAnnotator()
_GC_EXCHANGE_ANNOTATOR.tool_name = "exchange_delivered_order_items"

_GC_CANCEL_ANNOTATOR = GiftCardBalanceAnnotator()
_GC_CANCEL_ANNOTATOR.tool_name = "cancel_pending_order"


class H4RetailAnnotationMixin:
    """Mixin that adds H4 post-execution tool-response annotations to retail tools.

    Must be placed before HarnessedToolKitMixin (or HarnessedRetailTools) in
    the MRO so that its ``harness_annotators`` dict takes precedence.

    Can be combined with or without H2 (HarnessedToolKitMixin is required for
    the annotation mechanism to run; use with empty harness_rules for H4-only).
    """

    harness_annotators: dict[str, list] = {
        "get_order_details": [
            OrderEligibilityAnnotator(),
            OrderStatusRoutingAnnotator(),
            TrackingNumberAnnotator(),
        ],
        "get_item_details": [
            ItemAvailabilityAnnotator(),
        ],
        "get_product_details": [
            ProductAvailableCountAnnotator(),
            ProductVariantExtremaAnnotator(),
            ProductNumericOptionExtremaAnnotator(),
        ],
        "exchange_delivered_order_items": [
            ExchangePriceDiffAnnotator(),
        ],
        "return_delivered_order_items": [
            ReturnRefundAnnotator(),
        ],
        "cancel_pending_order": [
            CancelRefundAnnotator(),
        ],
    }


class H4RetailTools(H4RetailAnnotationMixin, HarnessedToolKitMixin, RetailTools):
    """RetailTools with H4 post-execution annotations only (no H2 rules)."""

    harness_rules: dict[str, list] = {}


class H4HarnessedRetailTools(H4RetailAnnotationMixin, HarnessedRetailTools):
    """RetailTools with H2 harness rules and H4 post-execution annotations."""


class H3H4RetailTools(H3RetailToolDescriptionMixin, H4RetailTools):
    """RetailTools with H3 description hints and H4 annotations (no H2 rules)."""


class H3H4HarnessedRetailTools(
    H3RetailToolDescriptionMixin, H4RetailAnnotationMixin, HarnessedRetailTools
):
    """RetailTools with H2 rules, H3 description hints, and H4 annotations."""