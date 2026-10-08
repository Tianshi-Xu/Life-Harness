# Identical-v3 continuation: rounds ten through twelve

Scope: DBBench and WebShop only; exactly three new proposal rounds per domain.
Seed: `../life_agentbench_v3_from_six_db_web_20261002`, sealed nine rounds,
SHA-256 `c05c921b40a4736e2b57cb1056985d5e8f7c089e7da9f76de7f807c581addd93`.
All previous training artifacts and test records stay unchanged.

Interface v3 and its policy-free empty templates are copied byte-for-byte and
hash-verified against the predecessor. No released policy is inserted. The
proposer remains the authorized ModelRouter `deepseek-v4-flash` API; the solver
remains Qwen3-4B, native environments, concurrency 8 and prior decoding settings.

DBBench starts on frozen training batch 5 and needs its first full 50-task
incumbent evaluation there. Batch rotation remains capped at 5; if a candidate
is rejected at the cap, the same cached unchanged incumbent evaluation is used
again, not rerun unnecessarily. WebShop starts on batch 3 and can reuse the
verified same-interface round-nine training incumbent (17/50). The provenance
of reused training evidence is recorded separately from new episode counts.

Candidates are compared on the complete matching 50-task batch. Only strict
improvement accepts. Invalid operations are filtered individually; valid sibling
operations are staged together. Episode infra errors count as zero, not as a
reason to discard every sibling result or rerun the entire domain.

Only existing evolved code, training history, training trajectories, interface
documentation and empty templates enter proposer workspaces. The latest test
scores have already been observed by the user, so prior test exposure is recorded;
test trajectories, labels and scores are not supplied to the proposer.

This queue is train-only; no final testing or baseline-policy mutation is
automatically invoked. After six new decisions, the queue verifies both histories,
seals round twelve, and writes `training_results.json` / `TRAINING_RESULTS.md`.

```bash
python meta/experiments/life_agentbench_v3_twelve_db_web_20261002/runner.py prepare
python -m unittest discover -s meta/experiments/life_agentbench_v3_twelve_db_web_20261002 -p test_continuation.py
python meta/experiments/life_agentbench_v3_twelve_db_web_20261002/runner.py smoke
python -u meta/experiments/life_agentbench_v3_twelve_db_web_20261002/workflow.py
python meta/experiments/life_agentbench_v3_twelve_db_web_20261002/status.py status
```
