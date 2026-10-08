# New Life: six-round training

Empty H2–H5 contents; all 50 current public training cases available on demand, no case-count quota.
Round r uses frozen batch r-1. Candidate and incumbent are paired on the same 50 IDs; equality rejected.
Historical training rows may use different batches and batch progression; not a continuous test curve.

| Domain | Round | Batch | Incumbent | Candidate | Accepted | Flash requests | Cases opened | Infra incumbent/candidate |
|---|---:|---:|---:|---:|---|---:|---:|---|
| dbbench | 1 | 0 | 20/50 | 未测：context_limit | False | 8 | 2/50 | 0/None |
| dbbench | 2 | 1 | 15/50 | 未测：context_limit | False | 8 | 3/50 | 0/None |
| dbbench | 3 | 2 | 22/50 | 未测：context_limit | False | 8 | 6/50 | 0/None |
| dbbench | 4 | 3 | 25/50 | 未测：context_limit | False | 8 | 4/50 | 0/None |
| dbbench | 5 | 4 | 21/50 | 未测：context_limit | False | 7 | 5/50 | 0/None |
| dbbench | 6 | 5 | 22/50 | 未测：context_limit | False | 6 | 3/50 | 0/None |
| webshop | 1 | 0 | 12/50 | 未测：context_limit | False | 9 | 10/50 | 0/None |
| webshop | 2 | 1 | 13/50 | 未测：context_limit | False | 6 | 6/50 | 0/None |
| webshop | 3 | 2 | 11/50 | 未测：output_truncated | False | 6 | 4/50 | 0/None |
| webshop | 4 | 3 | 11/50 | 未测：context_limit | False | 6 | 5/50 | 0/None |
| webshop | 5 | 4 | 10/50 | 未测：context_limit | False | 6 | 5/50 | 0/None |
| webshop | 6 | 5 | 9/50 | 未测：context_limit | False | 6 | 5/50 | 0/None |
