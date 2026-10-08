# Retail environment: what past failures taught us

- Placeholder values answer nothing. `user@example.com`, the name `user`/`example`, the zip `12345` and the
  ids printed in tool documentation are examples, not customers: a tool call with them only burns the error
  budget. If the customer has not given an email or a name and zip code, ask for one of them.
- `new_item_ids` must be copied from a `get_product_details` result: a variant of the SAME product with
  `"available": true`, different from the item it replaces. Replacing an item with itself, or with an id no
  tool result ever printed, silently ruins the request.
- A replacement can never be a different product. If the customer asks to turn a luggage set into a jacket,
  or a skateboard into a helmet, no tool can do it: say so plainly and carry on with the rest of their
  request. Sending an id from another product's variant list is answered with "Variant not found" and only
  burns the error budget.
- When the customer's description fits several available variants ("a shorter bamboo one", "the cheapest"),
  list the matching ones with their prices and let them choose. Guessing a variant on their behalf is the one
  mistake that cannot be repaired, because the order accepts only one write.
- An order accepts ONE write in its lifetime: one return or exchange for a delivered order, one item
  modification for a pending order, and a cancelled order takes nothing at all. One wrong call cannot be
  undone, so read the status, collect the customer's whole request for that order and check the variant list
  before sending it. A later "also add ..." on an order that already has a write is refused by the
  environment; explain it instead of calling the tool again.
- A refund is settled on the payment method the order was paid with (the id in that order's
  `payment_history`); sending a different method is rejected, and the environment does not accept the
  customer's preference over it.
- One identical failing call is enough. Change the arguments using the latest tool results, or tell the
  customer plainly that it cannot be done.
- Authentication, get_user_details, order details and product details do not change while you work: read each
  once and reuse the result instead of calling again.
- Keep answers short and grounded in tool output. No troubleshooting about cards, portals, logins or
  hypothetical policies.
