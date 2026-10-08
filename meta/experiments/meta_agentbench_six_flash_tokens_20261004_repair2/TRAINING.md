# Meta: six-round API rerun

Measured optimizer tokens; no preset cumulative token ceiling. NOT an equal-token-budget comparison.

| Domain | Round | Batch | Incumbent /50 | Candidate /50 | Decision | Flash calls | Input tokens | Output tokens | Total tokens | Cumulative tokens |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| alfworld | 1 | 0 | 10 | 16 | accepted | 71 | 1777424 | 66998 | 1844422 | 1844422 |
| alfworld | 2 | 0 | 16 | 17 | accepted | 109 | 5999361 | 161394 | 6160755 | 8005177 |
| alfworld | 3 | 0 | 17 | 18 | accepted | 75 | 3477239 | 105226 | 3582465 | 11587642 |
| alfworld | 4 | 0 | 18 | 18 | rejected | 119 | 8726682 | 116874 | 8843556 | 20431198 |
| alfworld | 5 | 1 | 18 | unmeasured | candidate_unmeasured | 74 | 5579586 | 118750 | 5698336 | 26129534 |
| alfworld | 6 | 2 | 17 | 19 | accepted | 125 | 10833413 | 175362 | 11008775 | 37138309 |
| dbbench | 1 | 0 | 21 | 25 | accepted | 103 | 6059443 | 148078 | 6207521 | 6207521 |
| dbbench | 2 | 0 | 25 | 26 | accepted | 98 | 4767414 | 154556 | 4921970 | 11129491 |
| dbbench | 3 | 0 | 26 | 27 | accepted | 81 | 4215218 | 118223 | 4333441 | 15462932 |
| dbbench | 4 | 0 | 27 | 27 | rejected | 77 | 3844615 | 103678 | 3948293 | 19411225 |
| dbbench | 5 | 1 | 23 | 25 | accepted | 84 | 4432678 | 124311 | 4556989 | 23968214 |
| dbbench | 6 | 1 | 25 | 25 | rejected | 115 | 8707110 | 138277 | 8845387 | 32813601 |
| webshop | 1 | 0 | 13 | 20 | accepted | 64 | 2108447 | 84533 | 2192980 | 2192980 |
| webshop | 2 | 0 | 20 | 24 | accepted | 66 | 3199984 | 102784 | 3302768 | 5495748 |
| webshop | 3 | 0 | 24 | 24 | rejected | 89 | 5716132 | 136238 | 5852370 | 11348118 |
| webshop | 4 | 1 | 23 | 23 | rejected | 116 | 10778486 | 160187 | 10938673 | 22286791 |
| webshop | 5 | 2 | 21 | unmeasured | candidate_unmeasured | 95 | 7831489 | 116462 | 7947951 | 30234742 |
| webshop | 6 | 3 | 17 | 19 | accepted | 101 | 6358748 | 148101 | 6506849 | 36741591 |
