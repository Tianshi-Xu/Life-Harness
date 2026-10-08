# Long-term memory — telecom support environment facts

## ID conventions (all IDs come from tools, never invented)
- Customer IDs start with `C`, line IDs with `L`, bill IDs with `B`, plan IDs with `P`, device IDs with `D`.
- A customer record lists `line_ids`; each line lists `phone_number`, `status`, `plan_id`, `roaming_enabled`.
- The customer's own line is the one whose `phone_number` matches the number the customer provides.

## Data-model reminders
- `get_data_usage` returns `data_used_gb`, `data_limit_gb`, `data_refueling_gb` and the cycle end date.
  A line is over its allowance only when `data_used_gb >= data_limit_gb + data_refueling_gb`.
- `refuel_data` bills the customer at the plan's refuelling rate; never add more than 2 GB at once.
- A bill is actionable only when its status is `Overdue` or `AWAITING_PAYMENT`; never send a payment
  request for an already-paid bill.
- `resume_line` only works on a `Suspended` line, and only after its overdue bills are paid.
- Overdue bill, in order: `send_payment_request` -> tell the customer a payment request was sent
  and ask them to open their payment requests and accept it -> `get_bills_for_customer` again and
  confirm the status is `Paid` -> `resume_line`. Never resume or say "paid" before a tool result
  shows `Paid`.
- You have account access, the customer does not: look up data usage, bills and suspension
  yourself; only the phone actions are the customer's job.

## Working habits that pay off
- Ask for the phone number first; the account cannot be checked without it.
- Only a TOOL CALL changes anything. Writing "your line has been resumed", "the bill is paid" or
  "I am transferring you" changes nothing — if you did not make the call, it did not happen. Never
  claim a state change (or end the call) that no tool result supports.
- The moment the customer says they accepted/paid the bill, your NEXT action is
  `get_bills_for_customer` (confirm the status is `Paid`) and then `resume_line` for the customer's
  own line — do not switch to another troubleshooting step first.
- A transfer only happens through the `transfer_to_human_agents` CALL. Saying you will transfer
  the customer, and then asking "are you okay with that?", transfers nobody and loses the case:
  make the tool call, then send "YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON."
- Forward the customer's number VERBATIM. `555-123-2002` and `5551232002` are different strings,
  and a reformatted number returns no account. Copy, never tidy.
- Never invent identifiers: no made-up numbers (e.g. `1234567890`), names or dates of birth. Only
  use values the customer actually said; the guard will reject anything else.
- Never refuel a line whose `data_used_gb` is below `data_limit_gb + data_refueling_gb` — that only
  adds an unwanted charge. Never call `resume_line` on a line that is not `Suspended`; it just errors.
- Never repeat an instruction or an identical tool call — the customer's report is the source of truth.
- Confirm the fix with the customer before finishing; escalate only when nothing else applies.

## Symptom -> complete fix set (customers usually have SEVERAL causes at once)
- MMS / picture messages: airplane mode off, SIM seated, network mode 4G/5G, mobile data on,
  ASK the customer directly whether they are currently abroad (if so: data roaming on the phone
  plus `enable_roaming` on their line), APN/MMSC correct (reset + reboot if the MMSC URL is
  wrong), Wi-Fi Calling off, messaging app granted BOTH storage and SMS permissions, and the data
  allowance not exceeded (`refuel_data`).
- Mobile data / slow data: airplane mode, SIM, mobile-data switch, network mode, ask directly
  whether the customer is abroad (data roaming), Data Saver mode, VPN, then the data allowance.
- No service: airplane mode, SIM status (`locked_pin` -> escalate), APN settings, then line
  suspension / overdue bills (`send_payment_request`, then `resume_line`).

## Conversation pitfalls that lose the call
- Asking about roaming only as a conditional aside ("if you are abroad, make sure roaming is on")
  never reveals the answer. Ask the direct question "Are you currently abroad?" as its own step.
- A SIM the CUSTOMER reports as locked with a PIN/PUK needs `transfer_to_human_agents`: no tool can
  unlock it, so asking again for the PIN only wastes the call.
- Re-sending an instruction the customer already answered makes them hang up. Once a step is done
  or refused, mark it done and move to the next item.
- Re-running an identical tool call never changes the result; use the output you already have.
- Transferring while a cause is still unfixed loses the case — clear the checklist and the
  account-side fixes (refuel, roaming, bill, resume) first.
- Transferring at the very start (before identifying the customer) fails instantly. Escalation is
  the last resort, never the opening move.
- Acting on the first line of an account is wrong: the customer's own line is the one whose
  `phone_number` equals the number they gave — check the line_ids before changing anything.
