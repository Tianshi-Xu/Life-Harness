# New Life: ONE-round TRAIN-only validation

Empty H2–H5 contents; all 50 current public training cases available on demand, no case-count quota.
Exactly one round, frozen b0. Candidate and incumbent are paired on the same 50 IDs; equality rejected. No TEST evaluation.
Historical training rows may use different batches and batch progression; not a continuous test curve.

| Domain | Round | Batch | Incumbent | Candidate | Accepted | Flash requests | Cases opened | Infra incumbent/candidate |
|---|---:|---:|---:|---:|---|---:|---:|---|
| dbbench | 1 | 0 | 20/50 | 28/50 | True | 29 | 9/50 | 0/0 |
| webshop | 1 | 0 | 12/50 | 16/50 | True | 32 | 18/50 | 0/0 |
