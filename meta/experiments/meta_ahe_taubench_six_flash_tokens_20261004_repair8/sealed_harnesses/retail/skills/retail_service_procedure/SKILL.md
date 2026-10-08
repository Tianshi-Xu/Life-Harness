# Skill: retail service procedure

Run every conversation in this order: resolve, decide, confirm, execute, report.

## Resolve
- Authenticate with the customer's exact email (`find_user_id_by_email`) or their exact first name, last name
  and zip code (`find_user_id_by_name_zip`). This is required even if the customer gives a user id.
- `get_user_details` -> user id, order ids with statuses, payment methods. Read them instead of asking.
- `get_order_details` for every order you touch -> status, address, items (each item carries its own `item_id`
  and `product_id`), fulfilments and payment history. Item ids come from here; replacement ids come from
  `get_product_details` variants.

## Decide
- pending: `cancel_pending_order`, `modify_pending_order_address`, `modify_pending_order_payment`,
  `modify_pending_order_items`.
- delivered: `return_delivered_order_items`, `exchange_delivered_order_items`.
- processed / cancelled / return requested / exchange requested: nothing can be done with that order; say so.
- The product type never changes: an item can only be replaced by another variant of the same product.

## Confirm
- Name the order id, the items or fields that change, the payment method for the price difference and the
  resulting amount (use `calculate`), then ask a single yes/no question. Do not ask twice.

## Execute
- Collect every item of one action into a single call before sending it; an order accepts one return, exchange
  or modification call in its lifetime, so a second write on the same order is always refused.
- Settle a refund on the payment method inside that order's `payment_history`; a call that names any other
  method is rejected, and resending it does not help.
- The moment the customer agrees, the next output is the tool call - not another question, not another answer.
- After the call returns, read the new status and amounts out of the result.

## Report
- Only values a tool result contains: new status, amount charged or refunded, refund destination, balance.
