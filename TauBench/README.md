# tau-bench Harness

This folder contains the tau-bench-style harness used in the paper. It is kept
as a separate subproject because its environment is managed by `uv` and is
independent from the Docker-based AgentBench setup.

The paper experiments in this folder use:

- Airline
- Retail
- Telecom

## Installation

```bash
cd my_tau_bench
uv sync
```

## Configure API Access

Copy the example environment file and fill it locally:

```bash
cp .env.example .env
```

The public repository only contains placeholder names. Do not commit private API
keys or private service URLs.

Common variables:

```bash
AGENT_API_BASE="<OPENAI_COMPATIBLE_AGENT_API_BASE>"
AGENT_API_KEY="<API_KEY_OR_EMPTY_FOR_LOCAL_SERVER>"
USER_API_BASE="<OPENAI_COMPATIBLE_USER_API_BASE>"
USER_API_KEY_ENV="OPENAI_API_KEY"
OPENAI_API_KEY="<USER_SIMULATOR_API_KEY>"
```

You can also pass these values explicitly with `--agent-api-base`,
`--user-api-base`, and `--user-api-key-env`.

## Run Evaluations

Start with a single trial to check that the environment is working. The paper
configuration uses 3 trials for final evaluation.

```bash
# Airline
uv run python scripts/eval_harness.py \
  --domain airline --split test --trials 3 \
  --enabled --h5 --h4 --h3 --h2 --h5-top-k 1 \
  --output airline/test-harness

# Retail
uv run python scripts/eval_harness.py \
  --domain retail --split test --num-trials 3 --nl \
  --enabled --h2 --h3 --h4 --h5 \
  --output retail/harness

# Telecom
uv run python scripts/eval_harness.py \
  --domain telecom --split train --trials 3 \
  --enabled --h2 --h3 --h4 --h5 --h5-top-k 1 \
  --concurrency 10 \
  --output telecom/harness
```

The script reports reward metrics and token usage. For paper tables, we report
the task success/reward metrics together with `agent_tokens`.

## Harness Switches

The harness CLI flags correspond to the four layers described in the paper:

- `--h2`: **Action Realization Layer**. Repairs malformed or recoverable actions, validates tool calls before execution, and blocks invalid actions when needed.
- `--h3`: **Environment Contract Layer**. Embeds environment-specific tool-use constraints into tool descriptions so the agent sees the correct contract when deciding how to call tools.
- `--h4`: **Trajectory Regulation Layer**. Monitors post-execution state, detects repeated failures or stagnation, and manages the remaining step budget.
- `--h5`: **Procedural Skill Layer**. Retrieves and injects task-relevant procedural skills or hints, controlled by settings such as `--h5-top-k`.

`--enabled` is the master switch. If `--enabled` is not passed, H2/H3/H4/H5 do
not take effect even if their individual flags are passed.

## Default Evaluation Settings

| Benchmark | Agent sampling | Agent max tokens | Max step |
| --- | --- | ---: | ---: |
| Airline | temperature = 0.0 | 2048 | 200 |
| Retail | temperature = 0.0 | 2048 | 200 |
| Telecom | temperature = 0.0 | 2048 | 200 |
