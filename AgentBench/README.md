# AgentBench Harness

This folder contains the AgentBench-style harness used in the paper. It is kept
as a separate subproject because it uses a Docker-based task environment and a
Python 3.9 dependency stack.

The paper experiments in this folder use the following AgentBench tasks:

- ALFWorld
- DBBench
- OS
- WebShop

## Installation

```bash
cd AgentBench
conda create -n agent-bench python=3.9
conda activate agent-bench
pip install -r requirements.txt
```

Docker is required for the task workers:

```bash
docker ps
```

## Configure the Agent

Configure the OpenAI-compatible agent endpoint in:

```text
configs/agents/api_agents.yaml
```

The public repository uses local OpenAI-compatible defaults such as
`Bearer EMPTY` for self-hosted model servers. If you use a commercial provider,
replace the endpoint and authorization value locally before running experiments.
Do not commit private API keys or private service URLs.

For the Qwen agent profile used by our harness runs, edit:

```yaml
qwen3-4b-instruct:
  parameters:
    url: "http://localhost:30001/v1/chat/completions"
    headers:
      Authorization: "Bearer EMPTY"
    body:
      model: "openai/Qwen/Qwen3-4B-Instruct"
      max_tokens: 4096
      temperature: 0.0
```

You can test whether the agent configuration is valid with:

```bash
python -m src.client.agent_test --config configs/agents/api_agents.yaml --agent qwen3-4b-instruct
```

## Build Docker Images

The OS and DBBench environments require base images. Build them before starting
the Docker Compose services:

```bash
docker pull mysql:8
docker pull ubuntu
docker build -f data/os_interaction/res/dockerfiles/default data/os_interaction/res/dockerfiles --tag local-os/default
docker build -f data/os_interaction/res/dockerfiles/packages data/os_interaction/res/dockerfiles --tag local-os/packages
docker build -f data/os_interaction/res/dockerfiles/ubuntu data/os_interaction/res/dockerfiles --tag local-os/ubuntu
```

## Start Task Services

Task workers are launched through Docker Compose. Start Redis, the controller,
and the task worker needed for the benchmark you are running:

```bash
# ALFWorld
docker compose -f extra/docker-compose.yml up -d --force-recreate redis controller alfworld-std

# DBBench
docker compose -f extra/docker-compose.yml up -d --force-recreate redis controller dbbench-std

# OS
docker compose -f extra/docker-compose.yml up -d --force-recreate redis controller os-std

# WebShop
docker compose -f extra/docker-compose.yml up -d --force-recreate redis controller webshop-std
```

If you modify task configuration files, restart the corresponding Docker Compose
services so workers reload the new configuration.

WebShop can take several minutes to become ready after Docker reports that the
container has started. If the first run cannot find the WebShop worker, wait and
retry.

## Run Evaluations

The assignment files specify the task split, concurrency, output directory, and
number of trials.

```bash
# ALFWorld
python -m src.assigner --config configs/assignments/alfworld.yaml

# DBBench
python -m src.assigner --config configs/assignments/dbbench.yaml

# OS
python -m src.assigner --config configs/assignments/os.yaml

# WebShop
python -m src.assigner --config configs/assignments/webshop.yaml
```

Useful configuration locations:

- `configs/agents/api_agents.yaml`: model endpoint, model name, sampling, and max tokens.
- `configs/tasks/*.yaml`: task split, max steps/rounds, and harness switches.
- `configs/assignments/*.yaml`: selected task profile, concurrency, trials, and output directory.

## Harness Switches

The harness switches correspond to the four layers described in the paper:

- `h2`: **Action Realization Layer**. Repairs malformed or recoverable actions, validates tool calls before execution, and blocks invalid actions when needed.
- `h3`: **Environment Contract Layer**. Embeds environment-specific tool-use constraints into tool descriptions so the agent sees the correct contract when deciding how to call tools.
- `h4`: **Trajectory Regulation Layer**. Monitors post-execution state, detects repeated failures or stagnation, and manages the remaining step/round budget.
- `h5`: **Procedural Skill Layer**. Retrieves and injects task-relevant procedural skills or hints, controlled by settings such as `h5_top_k`.

`enabled` is the master switch. If `enabled=false`, the H2/H3/H4/H5 settings do
not take effect even if their individual flags are set to `true`. In AgentBench
task configs this master switch may appear either as top-level `enabled` or as
nested `harness.enabled`, depending on the task.

For harness ablations, edit the task config and enable or disable one harness
level at a time (`h2`, `h3`, `h4`, `h5`) while keeping the remaining levels fixed.

## Evolving the Harness

Harness evolution is an iterative code-editing loop. After each evaluation run,
give a CLI coding agent, such as Codex CLI, the current harness implementation,
the previous iteration's trajectories, and the harness design guide. The agent
should inspect recurring deterministic interface failures and directly modify the
harness code; it should not stop at producing an analysis report.

Run the CLI agent from this suite directory and point it to the local guide file
instead of pasting the full design rules into the prompt.

```bash
HARNESS_DIR=src/server/harness
TRAJECTORY_DIR=<previous-run-output-dir>
DESIGN_GUIDE=Harenss.md

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

| Benchmark | Agent sampling | Agent max tokens | Max step / rounds |
| --- | --- | ---: | ---: |
| ALFWorld | temperature = 0.0 | 4096 | 50 |
| DBBench | temperature = 0.0 | 4096 | 15 |
| OS | temperature = 0.0 | 4096 | 8 |
| WebShop | temperature = 0.0 | 4096 | 20 |
