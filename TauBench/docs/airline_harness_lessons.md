# Airline Harness Lessons

## 2026-05-03 initial train/test comparison

- The first train subset was too easy for detecting H5 side effects: harness
  improved `20260503_163643_train-base` from 0.35 to 0.80 in
  `20260503_163712_train-harness`, with no measured train regressions.
- Test failures still showed H5 can be harmful. Treat train-subset gains as
  evidence for a pattern only when the hint is narrow and policy-level, not when
  it relies on broad words such as `reservation`, `refund`, `cost`, or `flight`.
- Airline H5 should be sparse. Prefer H3 for stable tool-contract rules
  (cancellation eligibility, round-trip update shape, payment method limits) and
  H4 for compact facts from tool results (reservation summaries, payment/refund
  amounts). H5 should only add a task-level reminder when the user wording is
  explicit.
- At most one airline skill should be injected per task until experiments show
  that multi-skill injection is stable. Multiple high-priority reminders can
  change task ordering and distract from the user's actual request.
- Certificate guidance must not over-prescribe separate bookings. The general
  rule is one certificate per reservation; for a single booking, choose one
  usable certificate and cover the remainder with allowed payment methods.
  Splitting into multiple bookings is only valid when the user explicitly wants
  multiple certificates and separate reservations still satisfy the request.

## 2026-05-03 164641 train-harness findings

- H5 top-k=1 reduced broad side effects and fixed task 17, but single-skill
  selection makes ranking important. Airline skill guards must avoid phrase
  matching that degenerates into generic tokens like `flights` or
  `reservations`.
- Basic-economy flight changes remain a high-value H3 rule: same-flight cabin
  upgrades are allowed, but a different itinerary from basic economy should go
  through cancel-and-rebook only after the user accepts that fallback.
- Fastest/cheapest itinerary tasks need tool-contract guidance, not memorized
  answers: compare complete candidate paths, require enough cabin seats on every
  leg, and retry the next candidate after a seat-availability failure.
- Schedule-mixup tasks should inspect all plausible reservations before
  deciding what to cancel. Acting after only a partial scan misses hidden
  conflicts and causes false completion.

## 2026-05-04 qwen3.5-9B high-base comparison

- Strong RL-trained agents can already solve many planning tasks without help;
  harness should preserve useful tool-contract calibration while avoiding
  over-prescriptive strategy. In the qwen3.5-9B train comparison, harness fixed
  itinerary tasks 21 and 33 but regressed 23, 39, and 42.
- H3 cancellation wording must not be stricter than the environment's expected
  behavior. Insurance can make a cancellation actionable; do not tell the agent
  to skip an insured cancellation solely because the user gives a non-health or
  non-weather reason and explicitly wants to proceed.
- H4 refund annotations should communicate amounts, not imply immediate
  payment availability. A cancellation refund should not be treated as an
  updated gift-card balance for a new booking in the same conversation unless a
  later user-details read confirms the balance.
- Schedule-mixup H5 must say both halves: scan every plausible reservation, then
  cancel only reservations that truly conflict with the stated itinerary,
  dates, passenger identity, and fallback instructions.
