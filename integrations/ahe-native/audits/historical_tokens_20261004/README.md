# Historical AHE token audit

Evolve-only known tokens are a lower-bound record, not full optimizer spend.
Debugger had no tracer; the old broker recorded request counts but not API usage.
Sensitivity scenarios below are explicitly assumptions, not recovered measurements.

| Domain | Round | Recorded Evolve tokens | Post-round TRAIN accuracy | Batch |
|---|---:|---:|---:|---:|
| alfworld | 1 | 14461543 | 20% | 1 |
| alfworld | 2 | 9242583 | 64% | 2 |
| alfworld | 3 | 8504957 | 98% | 3 |
| alfworld | 4 | 8043733 | 96% | 4 |
| alfworld | 5 | 16684715 | 96% | 5 |
| alfworld | 6 | 6594816 | 98% | 5 |
| dbbench | 1 | 17176526 | 62% | 1 |
| dbbench | 2 | 11676692 | 52% | 2 |
| dbbench | 3 | 14474235 | 52% | 3 |
| dbbench | 4 | 9437920 | 40% | 4 |
| dbbench | 5 | 9594355 | 44% | 5 |
| dbbench | 6 | 10064826 | 40% | 5 |
| webshop | 1 | 14180428 | 38% | 1 |
| webshop | 2 | 6737780 | 36% | 2 |
| webshop | 3 | 15047351 | 42% | 3 |
| webshop | 4 | 8569712 | 34% | 4 |
| webshop | 5 | 8590160 | 42% | 5 |
| webshop | 6 | 18618189 | 36% | 5 |

Known Evolve total: 207,700,521 tokens.
Current broker attempts: 3931; Evolve generations: 2005; difference: 1926.

| Assumed missing-call mean | Known plus scenario tokens |
|---:|---:|
| 20,000 | 246,220,521 |
| 50,000 | 304,000,521 |
| 100,000 | 400,300,521 |

Limitations:
- Assumed per-call means are not empirically measured, and this is not a confidence interval.
- Trace generations and request-gate attempts are not guaranteed one-to-one on failed calls.
- Reused first-round WebShop Debugger work predates the current broker and is outside its request count.
- Archived pre-repair evolution/startup costs are not included.
- Cannot assign missing spend to individual rounds; cannot claim these scenarios are audited full spend.
