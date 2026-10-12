# Life-Harness on TauBench

This suite contains **Airline, Retail and Telecom** tasks, native scoring, user simulation and Life-Harness H2–H5 runtime layers. See the root [README](../README.md) for complete Life-Harness/Meta-Harness/AHE evolution and comparison runs.

## Setup

```bash
cd TauBench
uv sync --frozen
uv run python scripts/eval_harness.py --help
```

Configure the solver and user-simulator endpoints and keys locally using the root [environment template](../evolution/deploy/env.example.sh). Set `LIFE_AGENT_MODEL`, `LIFE_USER_MODEL`, `AGENT_API_BASE` and `USER_API_BASE` to the actual served models/endpoints. Use the same model and simulator configuration for baseline and Life-Harness comparisons.

## Evaluate the provided Life-Harness

Run from `TauBench/`:

```bash
uv run python scripts/eval_harness.py \
  --domain retail --split test --enabled --h2 --h3 --h4 --h5 \
  --agent-llm "$LIFE_AGENT_MODEL" --agent-api-base "$AGENT_API_BASE" \
  --user-llm "$LIFE_USER_MODEL" --user-api-base "$USER_API_BASE" \
  --trials 1
```

Replace `retail` with `airline` or `telecom`. `--enabled` is the master switch: without it, all Life-Harness layers remain disabled even when their individual flags are present. Omit it for the baseline; omit selected H2/H3/H4/H5 flags for ablations.

`--split test` uses the held-out task split. `--trials` controls repeated trials and `--concurrency` controls parallel tasks. Model output budgets, thinking settings and simulator options are visible in `--help`; fix them across comparison arms. The solver defaults to temperature 0.

## Records

The evaluator saves `harness_summary.json` with accuracy and evaluation settings, native simulation results and exported task trajectories under its reported save directory. Preserve those files with the model/split/layer settings. `--user-api-key-env` names the environment variable holding the simulator key; no key should be stored in a published config.

## Newly evolved Life-Harnesses

The current bundle optimizer runs through `evolution.life_tau_run` or the root unified runner. It starts with empty H2–H5, selects on TRAIN, seals the accepted chain, and automatically evaluates final TEST. Its `FINAL_RESULTS.json` is the final evolved-harness record. The direct command above evaluates the provided built-in Life-Harness implementation, not an arbitrary run's accepted bundle. See [evolution/README.md](../evolution/README.md) for the separate single-file plugin interfaces.
