# Life-Harness on AgentBench

This suite contains Life-Harness runtime harnesses and native task definitions for **ALFWorld, DBBench, OS and WebShop**. See the root [README](../README.md) for Life-Harness/Meta-Harness/AHE evolution; this guide evaluates the provided frozen Life-Harness implementation.

## Setup

For native environment setup, follow [NEW_MACHINE_SETUP.md](NEW_MACHINE_SETUP.md). For the Docker assignment path below, install `requirements.txt` in a compatible AgentBench environment and initialize the source submodules from the repository root.

From `AgentBench/`, configure the model endpoint, model name, authorization and sampling in `configs/agents/api_agents.yaml`. Keep credentials local. The Docker services are defined in `extra/docker-compose.yml`.

OS needs the task images:

```bash
docker build -f data/os_interaction/res/dockerfiles/default data/os_interaction/res/dockerfiles --tag local-os/default
docker build -f data/os_interaction/res/dockerfiles/packages data/os_interaction/res/dockerfiles --tag local-os/packages
docker build -f data/os_interaction/res/dockerfiles/ubuntu data/os_interaction/res/dockerfiles --tag local-os/ubuntu
```

Start the controller, Redis and the selected worker, for example:

```bash
docker compose -f extra/docker-compose.yml up -d redis controller alfworld-std
```

For the other environments, replace `alfworld-std` with `dbbench-std`, `webshop-std` or `os-std`. DBBench requires MySQL; WebShop requires its product data/index. Restart the corresponding worker after changing task configuration.

## Evaluate the provided Life-Harness

Check the selected profiles before evaluation:

| Setting | File |
| --- | --- |
| Model endpoint, name, temperature, output budget | `configs/agents/api_agents.yaml` |
| Split, native step budget, Life-Harness switches | `configs/tasks/DOMAIN.yaml` |
| Agent/task profiles, trials, concurrency, output directory | `configs/assignments/DOMAIN.yaml` |

Use `*-std` tasks for held-out scoring. Set `enabled: true` in the selected ALFWorld/WebShop/OS task profile, or `harness.enabled: true` for DBBench. The OS `os-std` master switch is disabled in the supplied config, so enable it explicitly for a Life-Harness run. Preserve the released per-layer settings or record the ablation you choose.

```bash
# Run from AgentBench/ after the selected services are ready.
python -m src.assigner --config configs/assignments/alfworld.yaml
python -m src.assigner --config configs/assignments/dbbench.yaml
python -m src.assigner --config configs/assignments/webshop.yaml
python -m src.assigner --config configs/assignments/os.yaml
```

Run only the commands for the environments you provisioned. For a baseline, disable the master switch, restart the worker and evaluate with the same tasks/model settings and a distinct output directory. H2/H3/H4/H5 are action realization, environment contracts, trajectory regulation and procedural skills.

## Results and newly evolved harnesses

The assignment's `output` field determines where native results, task histories and model usage are saved. Keep the exact config with the scores. The Docker assignment evaluator runs the provided task harness; newly evolved Life-Harness bundles are evaluated by `evolution.life_agentbench_run` and its TRAIN seal/final TEST workflow, documented in the root README. Those runtime bundles are not automatically installed by changing the assignment master switch.
