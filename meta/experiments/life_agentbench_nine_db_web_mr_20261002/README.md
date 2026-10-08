# DBBench / WebShop three-round continuation

Inherit the sealed six-round run `../life_agentbench_six_resume_mr_20261001`
and add exactly rounds 7, 8 and 9 to DBBench and WebShop. ALFWorld and all
historical experiments remain unchanged. The predecessor seal SHA-256 is
`e5fb4fc69ffb641e07a9f6e51de3898035179b896bee3ce058a8b027a3a43ab0`.

Proposer: direct ModelRouter `deepseek-v4-flash` API. Solver: the existing
Qwen3-4B service at port 8400, temperature 0, max completion 4096, thinking
disabled. Native environments and concurrency 8 are unchanged. Accepted
artifacts are copied into independent directories, not writable symlinks.

DBBench starts at batch 2, reusing its sixth-round incumbent 34/50. WebShop
starts at batch 3 after rejecting round 6 and needs a new 50-task baseline.
Each candidate is evaluated on the full same 50-task batch. Strict improvement
accepts; tie/regression retains the incumbent and advances the frozen training
batch, capped at batch 5. Invalid operations are filtered individually before
valid siblings are staged together. Episode infra errors count as zero without
retrying the episode or voiding the domain.

Only training evidence enters proposer workspaces. Historical test results have
already been observed; this is a continuation, not untouched-holdout research.
Both domains must finish and seal round 9 before final testing. Snapshot-identical
final results may be reused; changed accepted snapshots get the full fixed test.
User-provided release references are DBBench 65% and WebShop 45.5%; they are not
new measurements or candidate acceptance criteria.

Commands from the repository root:

```bash
python meta/experiments/life_agentbench_nine_db_web_mr_20261002/runner.py prepare
python -m unittest discover -s meta/experiments/life_agentbench_nine_db_web_mr_20261002 -p 'test_*.py'
python meta/experiments/life_agentbench_nine_db_web_mr_20261002/runner.py smoke
python meta/experiments/life_agentbench_nine_db_web_mr_20261002/runner.py train
python meta/experiments/life_agentbench_nine_db_web_mr_20261002/runner.py status
```

New sources were derived from the six-round continuation, with the narrow
training evidence lookup/export fixes folded in before the formal run. No
frozen predecessor source was edited. Final reporting is a separate command;
do not use the old schema-1 `status.final_report` implementation.

## Task-boundary runtime amendment

The initial DBBench round-9 candidate hit the frozen Task session's legacy
120-word hint cap on all 50 tasks, despite the declared no-word-limit protocol.
The combination adapter had removed the cap, but the downstream Task bridge had
not. These invalid evaluation results, their log, and the pre-repair state are
preserved in `runtime_cap_archive/dbbench/`; they are not a valid candidate score.

The queue was stopped by its exact PID without touching model services or other
jobs. `repair_runtime.py` restored DBBench's unsealed pre-round-9 state. A separate
hash-bound adapter replaces only `FourHookSession._hints`'s word-count test;
list-of-at-most-one-string validation and ordinary trace semantics are preserved.
The frozen core/protocol/task files are unchanged. `runtime_amendment.json` binds
the correction sources; each corrected evaluation has a companion marker binding
its original config hash to this amendment. `unaffected_evidence.json` freezes
all pre-amendment evaluations with zero errors, including WebShop round 7.

DBBench round 9 reused its exact cached proposal and batch-4 baseline 31/50;
the corrected 50-task result is 27/50, zero infra errors, rejected. No fourth
proposal round was added. WebShop round 7 reused its existing 9/50→11/50
evaluation and accepted it. Nine continuation/runtime checks passed.

Use the corrected entrypoints for training and final reporting:

```bash
python meta/experiments/life_agentbench_nine_db_web_mr_20261002/corrected_workflow.py
python meta/experiments/life_agentbench_nine_db_web_mr_20261002/corrected_runner.py verify
python meta/experiments/life_agentbench_nine_db_web_mr_20261002/corrected_final.py
```

The resumed queue log is `corrected_workflow.log`. The initial `workflow.log`
and frozen sources remain unchanged; the final report includes the amendment
hash and excluded invalid-evaluation provenance.

## Completed results

Both domains completed and sealed nine rounds. DBBench rejected all three new
candidates and retains its six-round accepted snapshot; its hash-verified final
test is reused at 172/300 = 57.33%. WebShop accepted round 7, rejected rounds
8 and 9, and was newly tested on all 200 fixed tasks: 69/200 = 34.50%, zero
infra errors. The six-round WebShop reference was 71/200 = 35.50%.
The additional rounds did not improve the measured final test scores.

See `RESULTS.md` for the six new training decisions and user-provided release
references (DBBench 65%, WebShop 45.5%). Valid new training total: 500 episodes;
new test total: 200; two separate train-only smokes. Fifty initial invalid DB9
episodes remain archived and explicitly excluded from the corrected score.
`audit_final.py` verifies the sealed histories, evaluation provenance, reused
test, and runtime amendment without running any model calls.
