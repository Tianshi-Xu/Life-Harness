# Six-round TRAIN comparison

All scores are TRAIN, 50 tasks per evaluation. Batches differ across methods, not a paired comparison.
Meta/Life: bN incumbent->candidate; accept/reject. AHE: post-evolve score, usually on the next batch.
Meta/Life new have API usage ledgers; old Life tokens missing; AHE full round tokens missing (Evolve lower bound only).

| Domain | Round | New Meta | Old Life | Restricted AHE | New Life | New Life round tokens |
|---|---:|---|---|---|---|---:|
| alfworld | 1 | b0 20%->32%; accept | b0 26%->38%; accept | b1 20% | b0 28%->26%; reject | 4161138 |
| alfworld | 2 | b0 32%->34%; accept | b0 38%->94%; accept | b2 64% | b1 22%->unmeasured; reject | 3940528 |
| alfworld | 3 | b0 34%->36%; accept | b0 94%->98%; accept | b3 98% | pending | pending |
| alfworld | 4 | b0 36%->36%; reject | b0 98%->98%; reject | b4 96% | pending | pending |
| alfworld | 5 | b1 36%->unmeasured; reject | b1 96%->98%; accept | b5 96% | pending | pending |
| alfworld | 6 | b2 34%->38%; accept | b1 98%->98%; reject | b5 98% | pending | pending |
| dbbench | 1 | b0 42%->50%; accept | b0 36%->54%; accept | b1 62% | b0 38%->60%; accept | 1440875 |
| dbbench | 2 | b0 50%->52%; accept | b0 54%->58%; accept | b2 52% | b1 58%->66%; accept | 1673309 |
| dbbench | 3 | b0 52%->54%; accept | b0 58%->56%; reject | b3 52% | b2 64%->72%; accept | 2238186 |
| dbbench | 4 | b0 54%->54%; reject | b1 52%->60%; accept | b4 40% | pending | pending |
| dbbench | 5 | b1 46%->50%; accept | b1 60%->58%; reject | b5 44% | pending | pending |
| dbbench | 6 | b1 50%->50%; reject | b2 64%->68%; accept | b5 40% | pending | pending |
| webshop | 1 | b0 26%->40%; accept | b0 26%->34%; accept | b1 38% | b0 26%->36%; accept | 3695868 |
| webshop | 2 | b0 40%->48%; accept | b0 34%->36%; accept | b2 36% | b1 44%->unmeasured; reject | 221260 |
| webshop | 3 | b0 48%->48%; reject | b0 36%->40%; accept | b3 42% | b2 46%->40%; reject | 3126448 |
| webshop | 4 | b1 46%->46%; reject | b0 40%->34%; reject | b4 34% | pending | pending |
| webshop | 5 | b2 42%->unmeasured; reject | b1 38%->38%; reject | b5 42% | pending | pending |
| webshop | 6 | b3 34%->38%; accept | b2 40%->36%; reject | b5 36% | pending | pending |
