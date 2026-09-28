# Meta-Harness from-scratch six-benchmark run

This experiment starts all six supported non-OS benchmarks from the identical
identity implementation in `*/workspace/agents/baseline_zero.py`. It invokes
the cloned upstream Meta-Harness `run_evolve` and its original skill with a
native interface addendum. Released Life-Harness H2-H5 implementations are
never loaded into a candidate, composed with a candidate, or enabled during
evaluation.

Every round proposes one free-form candidate and scores it against the retained
incumbent on the same training sample. Strict improvement is required. A failed
round changes to the next deterministic sample before the next proposal. Each
AgentBench sample contains 50 questions. Retail and telecom use 50. Airline's
public train split contains only 30 questions, so it uses all 30 with two trials
for 60 trajectories; no distinct 50-question airline sample exists.

After acceptance/rejection, the retained incumbent is evaluated on the complete
fixed public test/std split. Test output is stored outside the proposer workspace
and test fields are removed from proposer-visible history. Test scores therefore
measure each round but cannot select or guide candidates.

The frozen solver is eight local Qwen3-4B instances behind port 8400, used with
evaluation concurrency 8, temperature 0, and thinking disabled where the native
API supports it. TauBench initially used `openai/deepseek-flash` through the
configured ModelRouter route as user simulator and natural-language judge.
Telecom rounds 5--6 use `openai/deepseek-v4-flash` with a conservative 16-RPM
user-model cap after the original route and a GLM fallback repeatedly returned
rate-limit errors; [CORRECTIONS.md](CORRECTIONS.md) records the excluded attempts
and exact boundary. One trial is used except for airline training as described
above.

The original run used an external clone of upstream Meta-Harness. This branch
now includes the three exact upstream files at `upstream_reference/` and the
39 frozen AgentBench task/configuration files at `task_snapshot/`; both sets
match `manifest.json` SHA-256 values. `adapter.py` and `prepare.py` now resolve
the upstream files from this experiment directory. The source used during the
run is preserved by `frozen_run_20260915.json`; `frozen.json` hashes the two
post-run path edits. No candidate, evaluator, task, score, or result was
changed by this packaging step. See [CORRECTIONS.md](CORRECTIONS.md).

On the original run host, where ignored raw episodes and benchmark runtimes
still exist, run one resumable domain with:

```bash
source meta/deploy/env.sh
export PATH=/mnt/data/xts/tools/uv/bin:$PATH
python -u meta/experiments/meta_from_scratch_six_20260915/adapter.py airline
```

Replace `airline` with `retail`, `telecom`, `alfworld`, `dbbench`, or `webshop`.
On a clean clone, first run `check_local_setup.py` and install the external
assets described in the root README. A clean clone has the compact experiment
record but not ignored raw episodes; it cannot resume the old model calls from
their exact interruption point. Run `status.py` for a compact progress report.
`manifest.json`, the two frozen hash files, each `state.json`, per-round
candidates, public training summaries, and all train/test summaries form the
committed experiment record.

After completion, regenerate the compact tables and audit all frozen hashes,
round transitions, test cardinalities, and per-task score means with:

```bash
python meta/experiments/meta_from_scratch_six_20260915/summarize.py
python meta/experiments/meta_from_scratch_six_20260915/audit.py
```

The compact result table is in [RESULTS.md](RESULTS.md); the machine-readable
equivalents are `results.json`, `rounds.csv`, and `AUDIT.json`.
