# Results

All runs start from the identity implementation. Released Life-Harness H2-H5 flags are empty in every round. Candidate acceptance uses training score only; the complete fixed test split is evaluated after every round and hidden from subsequent proposals.

| Domain | Round | Batch | Train baseline | Train candidate | Accepted | Retained | Test |
|---|---:|---:|---:|---:|:---:|---|---:|
| airline | 1 | 0 | 0.3667 | 0.5333 | yes | write_policy_gate.py | 0.3000 (20) |
| airline | 2 | 0 | 0.5333 | 0.6167 | yes | bag_fee_normalizer.py | 0.4500 (20) |
| airline | 3 | 0 | 0.6167 | 0.5667 | no | bag_fee_normalizer.py | 0.3500 (20) |
| airline | 4 | 1 | 0.5500 | 0.5667 | yes | itinerary_continuity_gate.py | 0.4500 (20) |
| airline | 5 | 1 | 0.5667 | 0.5500 | no | itinerary_continuity_gate.py | 0.5500 (20) |
| airline | 6 | 2 | 0.6333 | 0.6333 | no | itinerary_continuity_gate.py | 0.5000 (20) |
| retail | 1 | 0 | 0.3800 | 0.4400 | yes | compact_read_annotator.py | 0.3500 (40) |
| retail | 2 | 0 | 0.4400 | 0.4400 | no | compact_read_annotator.py | 0.4000 (40) |
| retail | 3 | 1 | 0.4800 | 0.5400 | yes | user_order_map.py | 0.4750 (40) |
| retail | 4 | 1 | 0.5400 | 0.5200 | no | user_order_map.py | 0.4750 (40) |
| retail | 5 | 2 | 0.4600 | 0.4800 | yes | refund_payment_repair.py | 0.5500 (40) |
| retail | 6 | 2 | 0.4800 | 0.4600 | no | refund_payment_repair.py | 0.4750 (40) |
| telecom | 1 | 0 | 0.1800 | 0.0800 | no | baseline_zero.py | 0.3500 (40) |
| telecom | 2 | 1 | 0.1800 | 0.2200 | yes | device_intent_relay.py | 0.3500 (40) |
| telecom | 3 | 1 | 0.2200 | 0.3000 | yes | primary_line_binding.py | 0.4000 (40) |
| telecom | 4 | 1 | 0.3000 | 0.2800 | no | primary_line_binding.py | 0.3000 (40) |
| telecom | 5 | 2 | 0.3200 | 0.3200 | no | primary_line_binding.py | 0.4750 (40) |
| telecom | 6 | 3 | 0.3200 | 0.3400 | yes | caller_line_scope.py | 0.4250 (40) |
| alfworld | 1 | 0 | 0.2200 | 0.3200 | yes | loop_breaker.py | 0.3028 (109) |
| alfworld | 2 | 0 | 0.3200 | 0.5800 | yes | state_precondition_nudge.py | 0.4312 (109) |
| alfworld | 3 | 0 | 0.5800 | 0.6400 | yes | light_goal_hold_nudge.py | 0.4679 (109) |
| alfworld | 4 | 0 | 0.6400 | 0.6800 | yes | stuck_search_redirect.py | 0.5596 (109) |
| alfworld | 5 | 0 | 0.6800 | 0.6200 | no | stuck_search_redirect.py | 0.5872 (109) |
| alfworld | 6 | 1 | 0.4000 | 0.4600 | yes | transform_effect_nudge.py | 0.5321 (109) |
| dbbench | 1 | 0 | 0.4200 | 0.5200 | yes | identifier_quoting_normalizer.py | 0.4833 (300) |
| dbbench | 2 | 0 | 0.5200 | 0.5400 | yes | declared_identifier_surface_normalizer.py | 0.4900 (300) |
| dbbench | 3 | 0 | 0.5400 | 0.5600 | yes | text_toolcall_recovery.py | 0.4867 (300) |
| dbbench | 4 | 0 | 0.5600 | 0.5600 | no | text_toolcall_recovery.py | 0.4967 (300) |
| dbbench | 5 | 1 | 0.4600 | 0.5000 | yes | missing_table_repeat_probe.py | 0.4833 (300) |
| dbbench | 6 | 1 | 0.5000 | 0.4600 | no | missing_table_repeat_probe.py | 0.4800 (300) |
| webshop | 1 | 0 | 0.2400 | 0.3800 | yes | constraint_ledger.py | 0.2650 (200) |
| webshop | 2 | 0 | 0.3800 | 0.4000 | yes | progress_guard.py | 0.2550 (200) |
| webshop | 3 | 0 | 0.4000 | 0.4200 | yes | variant_code_check.py | 0.2950 (200) |
| webshop | 4 | 0 | 0.4200 | 0.4000 | no | variant_code_check.py | 0.2800 (200) |
| webshop | 5 | 1 | 0.2600 | 0.3200 | yes | option_alignment_check.py | 0.3050 (200) |
| webshop | 6 | 1 | 0.3200 | 0.3400 | yes | option_group_title_anchor.py | 0.3100 (200) |

## Domain summary

| Domain | Rounds | Accepted | Final retained | Final test | Best test |
|---|---:|---:|---|---:|---:|
| airline | 6 | 3 | itinerary_continuity_gate.py | 0.5000 | 0.5500 |
| retail | 6 | 3 | refund_payment_repair.py | 0.4750 | 0.5500 |
| telecom | 6 | 3 | caller_line_scope.py | 0.4250 | 0.4750 |
| alfworld | 6 | 5 | transform_effect_nudge.py | 0.5321 | 0.5872 |
| dbbench | 6 | 4 | missing_table_repeat_probe.py | 0.4800 | 0.4967 |
| webshop | 6 | 5 | option_group_title_anchor.py | 0.3100 | 0.3100 |
