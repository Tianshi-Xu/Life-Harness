# Policy-free v3 continuation from sealed round six

Scope: DBBench and WebShop only, exactly rounds 7, 8 and 9. ALFWorld unchanged.
Seed: `../life_agentbench_six_resume_mr_20261001`, seal
`e5fb4fc69ffb641e07a9f6e51de3898035179b896bee3ce058a8b027a3a43ab0`.
This is independent of the previous seven-to-nine run; no old candidate or score
is silently substituted for a new v3 evaluation.

The complete, policy-free interface is copied and hash-frozen in `interface_v3/`.
H2/H3 templates are identity functions, H4/H5 are empty. Public context, shared
state, H3 message conditioning, H2 recovery/control and guidance persistence are
available, but no released rule, prompt, skill or retrieval content is inserted.
The proposer receives existing evolved artifacts, prior training decisions,
fresh baseline training trajectories, interface documentation and empty templates.
It does NOT receive released domain-policy source or historical test results.

Both domains discard inherited baseline **scores**, not their inherited code or
history. Re-evaluate the same inherited active snapshot on the starting train
batch under v3: DB batch 2, WebShop batch 3. Evaluate each new candidate on its
full 50-task training batch. Strict improvement accepts; ties/regressions retain
the incumbent and advance the frozen batch. Invalid operations are filtered
individually, with valid sibling operations staged simultaneously.

Proposer: authorized ModelRouter `deepseek-v4-flash` API; credentials remain
process-local. Solver: existing `qwen3-4b` on port 8400, temperature 0, completion
limit 4096, thinking disabled, concurrency 8. Native environments are unchanged.
An infra-error episode counts as zero without invalidating the entire domain.
WebShop's existing environment variation is left unchanged, per user preference.

New run/evaluation fingerprints bind interface/template hashes, base session
hash, protocol, runner sources and artifact manifest. Prior test exposure is
recorded. This authorized queue is **train-only**: no final-test command is
automatically invoked, and no old score is relabeled a v3 score.

From the repository root:

```bash
python meta/experiments/life_agentbench_v3_from_six_db_web_20261002/runner.py prepare
python -m unittest discover -s meta/experiments/life_agentbench_v3_from_six_db_web_20261002 -p test_continuation.py
python meta/experiments/life_agentbench_v3_from_six_db_web_20261002/runner.py smoke
python -u meta/experiments/life_agentbench_v3_from_six_db_web_20261002/workflow.py
python meta/experiments/life_agentbench_v3_from_six_db_web_20261002/status.py status
```

`workflow_state.json` records queue state and errors; `workflow.log` records the
queue output. After exactly six new domain-round decisions, `training_seal.json`,
`training_results.json` and `TRAINING_RESULTS.md` are produced. Resume uses the
same cached proposal and structurally valid episodes; histories remain untouched.

## Subsequently authorized final tests

After the training queue completed and sealed, the user explicitly requested
full tests of the latest effective DBBench/WebShop snapshots. `test_runner.py`
records that authorization separately, verifies the existing seal, and evaluates
DBBench 300 and WebShop 200 test tasks under v3. Even DBBench requires a fresh
test because its interface changed despite unchanged artifact code. No training,
proposer call or historical score overwrite occurs. `test_results.json` and
`TEST_RESULTS.md` record fresh scores and user-provided release references;
`test_workflow_state.json` and `test_workflow.log` record execution progress.
The original train-only completion report remains a historical record.
