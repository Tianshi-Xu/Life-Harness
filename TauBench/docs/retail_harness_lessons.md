# Retail Harness Iteration Notes

These notes record experiment-backed harness development principles for retail.
Use them when analyzing new training-subset experiments, and prefer changes that
should generalize to the test split.

## 2026-05-03 sub30 iterations

- H5 should be sparse and high precision. Broad skill injection caused unrelated
  task drift, especially when a skill mentioned a secondary concern and pulled
  attention away from other requested actions.
- Prefer H3 for stable tool-contract rules that apply across tasks:
  status routing, one modification/exchange per order, exact item IDs, same
  product variants, and grouped same-order writes.
- Prefer H4 for compact facts computed from tool results:
  available variant counts, pending/delivered status route, refund or price
  difference amounts. Avoid H4 notes that infer mutable state incorrectly.
- Do not use training-task-specific item IDs, order IDs, names, or expected
  answers in harness logic. Harnesses should describe reusable failure modes.
- Regression tasks matter first. A harness that fixes a failure but creates
  new failures on previously passing tasks should be narrowed or moved to a
  more precise layer.
- Single-trial sub30 results are noisy. Treat one-off pass/fail flips as
  hypotheses, not proof. Prefer changes backed by repeated patterns, clear
  trajectory evidence, or deterministic tool-contract violations.
- For pending item-removal requests, full-order cancellation is valid only when
  the user accepts the fallback and its actual refund route. Cancellation refunds
  go to the original payment method.
- For delivered exchanges with multiple requested items in the same order, all
  items must be exchanged in one tool call because each delivered order supports
  only one return/exchange action.
- New-task failures should update generic patterns, not expand task-specific
  harnesses. The most useful additions in this round were compact product
  price extrema, category-scope discipline, pending replacement routing, and
  address source/target direction.
- For product variant selection, a compact H4 fact is safer than a broad skill:
  surface cheapest/most-expensive available variants after product lookup, while
  still requiring exact option matching when the user states hard constraints.

## Practical rules for the next retail iterations

- Compare experiments by overlap before reacting to the headline average.
  In the 20260503_124606 run, several old overlap tasks improved while one
  old task regressed, but most new tasks failed. That points to adding coverage
  for new generic failure modes, not reverting the previous changes wholesale.
- Keep H5 routing guards stricter than BM25. Retail task descriptions contain
  many generic words such as order, pending, address, return, and exchange; a
  skill should fire only when its distinctive trigger words are present. After
  adding a skill, sanity-check retrieval on both target tasks and regression
  sentinels.
- Use H4 when the missing information is a deterministic fact from a just-read
  object. Examples: available variant count, tracking number, cheapest available
  variant, most expensive available variant, refund amount, and price
  difference. Keep these annotations compact so they do not become a second
  policy document.
- Use H5 only for planning failures that require connecting multiple tool calls
  or user turns. Examples: copy a hidden address from the correct historical
  order, group multiple same-order exchanges into one call, keep a category-
  scoped request from absorbing unrelated items, and treat pending same-product
  replacement as an order modification rather than a new purchase.
- Avoid overfitting to train-subset artifacts. Task IDs may be listed in skill
  metadata for traceability, but prompt text and guards should not mention
  exact order IDs, item IDs, customer names, or expected answers.
- Prefer narrowing over deletion when a harness helps one cluster but hurts
  another. The product-count and pending-replacement skills are useful only
  with explicit trigger phrases; broad versions can steal attention from
  unrelated tracking, address, or bulk-return tasks.
- For single-trial runs, classify flips into likely deterministic fixes,
  likely deterministic regressions, and likely stochastic variation. Only the
  first two should usually drive code changes; stochastic flips should become
  watchlist tasks for the next sub30.
- New sub30 suites should mix known regression sentinels with unseen train
  tasks. A good default is half overlap and half new tasks, so improvements are
  tested against both stability and generalization.

## 2026-05-03 130732 sub30 findings

- Look at overlap wins before adding more harness. The 130732 run improved the
  previous failure cluster around cheapest/most-expensive variants, pending
  same-product replacement, and several confirmation-sensitive tasks. The next
  changes should be narrower and focused on remaining stable failures.
- Category-scoped tasks are vulnerable to broad bulk skills. If the user says
  everything associated with, not associated with, related to, or a named theme
  such as gaming, category scope must win over the word everything. Do not also
  inject generic bulk-order guidance for those tasks unless the task separately
  says all orders without a category filter.
- Privacy wording is not enough to trigger address guidance. Phrases like do
  not reveal it may refer to hidden item/category intent, not an address. Hidden
  address skills need explicit address/home/profile/shipping context.
- Product price extrema fixed a real class of failures, but some remaining
  variant errors are numeric-option errors rather than price errors. For max
  zoom, lower resolution, one size smaller, larger frame, more pieces, or named
  storage such as 32GB/1TB, the harness should surface numeric option extrema
  and remind the agent to preserve every hard option constraint.
- Same-order exchange failures often come from finding a similar item in a
  different order. The reusable instruction is: if the user says item B is in
  the same order as item A, pick the order containing both product names and
  perform one grouped exchange call for that order.

## 2026-05-07 qwen3.5-9B high-base comparison

- Strong retail agents can already solve many bulk/order-routing tasks. Harness
  should keep factual/tool-contract guidance but avoid turning fallback
  constraints into premature stops.
- Bulk returns: if the user initially requests an invalid refund route but later
  accepts original payment, the correct behavior is to proceed with all requested
  delivered-order returns using original payment methods, not to transfer.
- Expedited refund requests should not block completion. After a valid return,
  answer remaining arithmetic/informational requests and explain that refund
  timing cannot be guaranteed; transfer only if the user explicitly asks for a
  human after that.
- Cancellation reason hints must preserve user-provided valid reasons. When
  cancellation is only a fallback after an unavailable item change, ask for a
  reason if none was provided, then use the user's exact valid answer.
- Address-from-order-history guidance must distinguish source from target:
  a recent/new-home order is evidence to copy from, while the order described
  as old/wrong is the one to modify.
- For strong models, most fixes should be scope-preserving rather than
  procedure-expanding. Examples from this run: themed "everything" still means
  only matching category items; "received with" can be order-location context,
  not a request to return both items; and vague payment preferences should not
  override more reliable order/payment evidence.
- Later strong-model runs showed that payment-method hints must distinguish
  named preferences from vague confirmation-time preferences. Use a named card
  such as "visa" when the user requested it up front, but do not let a generic
  "other card" during confirmation override the original exchange payment
  method. Address fixes should also cover every pending order in scope, while
  preserving the order described as sent to the new home as the address source.
- Category scope should prefer conservative inclusion. For gaming, keyboard and
  mouse are clear peripherals; an action camera is not in scope just because it
  could be used to record gameplay. Avoid suggesting ambiguous items during the
  confirmation summary, because the user simulator may confirm the agent's
  mistaken expanded scope.
- Pending item modification calls should include only actual replacements. If
  an item is already the cheapest acceptable variant, leave it out of the
  modify call; do not choose a more expensive "next cheapest different" variant
  just to satisfy the different-item constraint.
- Hidden-address source selection must be tied to the product/order the user
  says was sent to the new home. Luggage-sent-new-home and tablet-sent-new-home
  templates are opposites even though the account and addresses are the same.
- When strong-model test results regress, prefer removing duplicated high-salience
  H5 detail over adding more rules. Keep H5 for task-level planning failures and
  leave narrow tool-argument constraints in H3/H2; duplicated H5+H3 guidance on
  payment, address source, or return scope can cause the model to overfit the
  hint instead of following the current dialogue.
