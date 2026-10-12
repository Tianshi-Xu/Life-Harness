# Life-Harness on AppWorld

This adapter applies Life-Harness's H2–H5 runtime layers to the upstream AppWorld simplified function-calling agent. Model weights, AppWorld tasks, environment transitions and scoring remain unchanged. It provides fixed-policy evaluation, a no-harness baseline, per-layer ablations and runtime records.

The release includes `policies/v009_r4/`, the frozen R4 policy selected during training, and `policies/v005/`, the earlier policy used by the layer regression tests. Choose the policy before evaluating held-out tasks. The AppWorld adapter is separate from the automatic six-round Meta-Harness/AHE comparison controllers; an AppWorld Meta-Harness/AHE evolution loop is not included.

## Setup

Install the upstream AppWorld engine, data and simplified agent package using the [official AppWorld instructions](https://github.com/StonyBrookNLP/appworld#-agent-experiments). `APPWORLD_ROOT` should point to that upstream checkout, which supplies the data, experiment prompts and simplified agent package. The adapter ships its own local-server configuration; a custom upstream `local/qwen3-4b/train` experiment is not required.

From the Life-Harness repository root, install this adapter into the upstream runtime:

```bash
export APPWORLD_ROOT=/absolute/path/to/appworld
uv pip install --python "$APPWORLD_ROOT/.venv/bin/python" -e "$PWD/AppWorld"
export PYTHONPATH="$PWD/AppWorld/src:$APPWORLD_ROOT/experiments"
export MODEL_SERVER_URL=http://127.0.0.1:8000
export NO_API_KEY=EMPTY
export OPENAI_API_KEY="$NO_API_KEY"
```

The upstream environment must already have `appworld` and `appworld-agents` with the `simplified` dependencies installed, and AppWorld data downloaded. Start a compatible model server and set `MODEL_SERVER_URL` to its endpoint. Record the upstream SDK/agent revision and model serving settings with the run; this adapter's lockfile does not pin the external AppWorld engine.

The shipped [local-server configuration](src/life_harness_appworld/configs/local_function_calling.json) uses the upstream function-calling prompts and demos, 50 native steps, and a 100,000 output-token task budget. It routes the main agent and API predictor to `MODEL_SERVER_URL` (adding `/v1` when needed), with credentials read from `NO_API_KEY`. Local monetary rates are set to zero; provider token usage is still recorded. Record these settings and configure real rates when monetary cost limits matter. To use your own installed upstream experiment instead, pass `--base-experiment EXPERIMENT_NAME`; the partitioned scripts accept the same setting through `BASE_EXPERIMENT`.

## Evaluate the frozen Life-Harness

From the Life-Harness root, run the frozen policy on `dev`:

```bash
"$APPWORLD_ROOT/.venv/bin/python" -m life_harness_appworld.runner \
  --appworld-root "$APPWORLD_ROOT" \
  --policy-directory "$PWD/AppWorld/policies/v009_r4" \
  --model-name Qwen3-4B --temperature 1.0 \
  --dataset dev --experiment-name life_harness/final/dev \
  --num-processes 1 --process-index 0

"$APPWORLD_ROOT/.venv/bin/appworld" evaluate \
  life_harness/final/dev dev --root "$APPWORLD_ROOT"
```

Use `test_normal` or `test_challenge` in both commands for an official TEST split, with a fresh experiment name. `--num-processes 1 --process-index 0` runs the complete selected dataset. For parallel execution, launch all indices `0..N-1` with `--num-processes N` and the same experiment name before evaluating the full split; the supplied shell scripts demonstrate this pattern.

For a no-harness baseline, repeat the runner command with `--baseline` and a separate experiment name, then evaluate that experiment. Keep model, temperature, predictor settings, task split and environment identical. Per-layer ablations use `--disable-h2`, `--disable-h3`, `--disable-h4` and `--disable-h5`.

The acting-model temperature defaults to `1.0`. The one-time API predictor inherits it unless `--api-predictor-temperature` is supplied. Qwen's native thinking behavior is preserved unless `--qwen-disable-thinking` is explicitly selected. Record these choices; they are separate ablations.

## Accuracy and inference-token records

The upstream output directory is:

```text
$APPWORLD_ROOT/experiments/outputs/EXPERIMENT_NAME/
  evaluations/           # Upstream dataset evaluation and scores
  tasks/TASK_ID/logs/     # Model calls, usage and Life-Harness events
```

Keep the upstream evaluation as the authoritative accuracy record. Produce a compact report for the same dataset:

```bash
"$APPWORLD_ROOT/.venv/bin/python" -m life_harness_appworld.analysis.summarize_experiment \
  --experiment-output "$APPWORLD_ROOT/experiments/outputs/life_harness/final/dev" \
  --dataset dev \
  --output "$APPWORLD_ROOT/experiments/outputs/life_harness/final/dev/life_summary.json"
```

The report retains native scores, task completion, layer events, context errors and known model input/output/total tokens from `lm_calls.jsonl`. Missing usage is counted and flagged; known totals exclude unknown consumption. If a provider supplies input/output but no total, their sum is explicitly marked as derived. Raw model-call logs remain available. These are **inference tokens**: fixed-policy evaluation performs no optimization and has no evolution-token cost.

`analysis/trajectory_analysis.py` and `analysis/replay_regression.py` are TRAIN-only diagnosis/replay utilities. Replay checks layer behavior on recorded prefixes; it is not a new AppWorld rollout or a substitute for native scoring.

## Offline layer tests

```bash
PYTHONPATH="$PWD/AppWorld/src" python -m pytest AppWorld/tests -q
```

The layer and report tests need no model server, AppWorld data or paid requests. Full evaluation requires the upstream environment and model service.
