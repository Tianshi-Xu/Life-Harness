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

uv run python scripts/eval_harness.py --domain airline --split test --trials 3 \
  --agent-llm openai/claude-opus-4-8 --user-llm openai/deepseek-v4-pro \
  --concurrency 8 \
  --h2 --h3 --h4 --h5 --h5-top-k 1 --output airline/claude-harness
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

## Evolving the Harness

Harness evolution is an iterative code-editing loop. After each evaluation run,
give a CLI coding agent, such as Codex CLI, the current harness implementation,
the previous iteration's trajectories, and the harness design guide. The agent
should inspect recurring deterministic interface failures and directly modify the
harness code; it should not stop at producing an analysis report.

Run the CLI agent from this suite directory and point it to the local guide file
instead of pasting the full design rules into the prompt.

```bash
HARNESS_DIR=src/tau2/harness
TRAJECTORY_DIR=<previous-run-output-dir>
DESIGN_GUIDE=Harness.md

codex "
You are a coding agent responsible for improving a runtime harness for a
deterministic LLM-agent environment. Your goal is to improve task performance by
adapting the runtime interface between the frozen model and the environment,
without changing model weights, benchmark tasks, or environment evaluation logic.

Inputs:
- current harness implementation: ${HARNESS_DIR}
- trajectory directory from the previous iteration, including summary metrics: ${TRAJECTORY_DIR}
- harness design guide: ${DESIGN_GUIDE}

Inspect the previous iteration's trajectories and identify recurring failure
patterns. For each pattern, determine the earliest lifecycle point where it can
be reliably detected or prevented: before interaction, during task conditioning,
before environment execution, or after execution.

Focus on mechanically identifiable deterministic failures such as invalid action
formats, wrong tool conventions, missing required fields, repeated no-op actions,
loops, premature submissions, budget exhaustion, or recurring procedural
mistakes.

Directly implement targeted, minimal updates in the appropriate harness layer.
Do not only return an analysis report. Do not use hidden oracle information,
test labels, task modifications, environment transition changes, or
evaluation-criteria changes.

After editing, run or recommend the narrowest regression checks available.
Inspect cases where the harness may over-trigger, block a valid action, inject
misleading guidance, or reduce performance on previously successful
trajectories.

When finished, summarize:
1. dominant failure patterns found;
2. harness layer responsible for each update;
3. implemented code changes;
4. why each update is safe under the deterministic environment contract;
5. remaining failure modes to monitor next.
"
```

Then rerun the corresponding evaluation command and use the new trajectory
directory as `TRAJECTORY_DIR` for the next iteration. Keep each update local to
the harness layer that can detect the failure earliest.

## Default Evaluation Settings

| Benchmark | Agent sampling | Agent max tokens | Max step |
| --- | --- | ---: | ---: |
| Airline | temperature = 0.0 | 2048 | 200 |
| Retail | temperature = 0.0 | 2048 | 200 |
| Telecom | temperature = 0.0 | 2048 | 200 |
