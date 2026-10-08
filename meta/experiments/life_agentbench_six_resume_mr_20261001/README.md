# Six-round Life continuation

This run inherits the sealed five-round AgentBench experiment
`../life_agentbench_bundle_five_20261001` and adds exactly round 6 per domain.
The original experiment, its training seal, states, artifacts and episodes are
read-only. Inheritance is bound to SHA-256
`fb55372a76f89aac45d414e5dfc83a69582876286aae89ed7967a6e5af31f8c6`.

Only round 6 uses direct ModelRouter `deepseek-v4-flash` chat-completions tool
calling. Credentials are loaded from `/mnt/data/xts/skill/test_ap/env.sh` and
are never copied into proposer workspaces. Solver remains the historical
`qwen3-4b` endpoint at `127.0.0.1:8400`, with temperature 0, 4096 completion
tokens and thinking disabled. Domains run sequentially, with solver concurrency
at most 8. Existing services are reused; no GPU service is deployed.

ALFWorld retains training batch 1 and reuses its stored 49/50 incumbent result.
DBBench and WebShop advanced to batch 2 after round 5, so each needs one new
50-task incumbent evaluation before its 50-task round-6 candidate evaluation.
These are the only new formal training evaluations: 250 episodes in total.
Three separate one-task train-only native smokes do not enter round decisions.

Round 6 filters invalid operations individually, simultaneously stages valid
siblings, accepts only strict same-batch improvement, and otherwise retains
the previous accepted mechanisms. H4/H5 retain the one-string contract without
word limits. Infrastructure-error episodes are cached once as failures and
included in summaries. WebShop requests trim old history to fit solver context.

The predecessor's fixed test results were already observed. This is explicitly
a continuation with prior test exposure, not a new untouched-holdout experiment.
Neither predecessor test trajectories nor Meta scores enter the proposer
workspace. `runner.py train` only trains and seals all three six-round states;
it never starts final test or uses test performance to decide additional work.

From the repository root:

```bash
python meta/experiments/life_agentbench_six_resume_mr_20261001/resume_runner.py prepare
python meta/experiments/life_agentbench_six_resume_mr_20261001/resume_runner.py smoke
python meta/experiments/life_agentbench_six_resume_mr_20261001/resume_runner.py train
python meta/experiments/life_agentbench_six_resume_mr_20261001/resume_runner.py status
python -m unittest discover -s meta/experiments/life_agentbench_six_resume_mr_20261001 -p 'test_*.py'
```

`inheritance.py` validates predecessor training evidence without opening its
test outputs. Accepted artifacts are copied into a separate versions directory;
there are no writable symlinks into the predecessor. Imported decisions remain
unchanged, and no inherited run configuration or evaluation is relabeled as a
new six-round result. `inheritance.json` records per-domain provenance.

The smoke was run before a train-evidence path fallback correction. Its source
binding is preserved in `pretrain_source_revisions/`; formal training binds the
corrected source. No formal sixth-round evaluation was run before this revision.

ALFWorld's round-6 evaluation completed at 49/50 and tied its incumbent, so the
candidate was rejected and the original accepted snapshot retained. The queue
then stopped because verification looked for the imported baseline inside the
new directory. `resume_runner.py` applies a narrow amendment to resolve training
evidence and verification replay exports in the predecessor's read-only folder.
The amendment is hash-bound in `verification_amendment.json`. The original
round-6 evaluator, proposer, driver, fingerprints and 50 saved outcomes remain
unchanged; resumption verifies them and advances directly to DBBench.

## Completed training

| Domain | Sixth-round batch | Incumbent | Candidate | Decision |
|---|---:|---:|---:|---|
| ALFWorld | 1 | 49/50 = 0.98 | 49/50 = 0.98 | reject tie |
| DBBench | 2 | 32/50 = 0.64 | 34/50 = 0.68 | accept |
| WebShop | 2 | 20/50 = 0.40 | 18/50 = 0.36 | reject regression |

All three states completed six rounds and are bound by `training_seal.json`.
All 250 new formal train episodes have zero infrastructure errors. Only DBBench
changed its accepted snapshot. A subsequent user-authorized final evaluation
ran the 300 fixed DBBench test tasks once: 172/300 = 57.33%, infrastructure
errors 0. ALFWorld and WebShop reused the predecessor test outcomes for identical
accepted snapshots: 101/109 = 92.66% and 71/200 = 35.50%. WebShop's existing
index-42 infrastructure failure remains counted as zero, with no retry.
The five-round DBBench score is not relabeled as its six-round score.
The original five-round seal, states, and Meta results
remain unchanged. Existing solver backends were checked to use GPUs 4–7.
The 26 continuation/shared regressions and three verification amendment checks
passed. Sequential queue output is preserved in `train_resume.log`.

## Final comparison

`COMPARISON.md` contains six-round train decisions, all available per-round
test scores, and the final Meta/Life comparison. `round_comparison.json` records
exact values; `final_results.json` records reused versus newly measured outcomes
and their provenance hashes. No historical round-1-to-4 Life tests were added.
Use `python meta/experiments/life_agentbench_six_resume_mr_20261001/final_reporting.py`
to generate or validate this report. The previous `resume_runner.py report`
assumes fresh local test summaries in all three domains and is not compatible
with the new provenance-aware report schema. Final output is in
`final_reporting.log`; this reporting step added exactly 300 test episodes.
