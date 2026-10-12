# Life-Harness, Meta-Harness and AHE implementations

Start with the root [README](../README.md) for the three methods, supported benchmarks, complete from-scratch runs and accuracy/token records.

| Method | Current entry points |
| --- | --- |
| **Life-Harness** | `life_evolve/` implements H2–H5 evolution; `life_tau_run.py` and `life_agentbench_run.py` run complete training, selection and final TEST evaluation |
| **Meta-Harness** | `reproduction/meta_transport/` prepares and runs fresh workspaces using the original optimizer in `reproduction/reference/upstream_reference/` |
| **AHE** | `../integrations/ahe-native/` adapts the pinned original AHE loop; `tau_comparison/` and `os_comparison/` integrate it with the other methods |
| **Comparison runner** | `python -m evolution.reproduce --help` prepares, launches and collects Life-Harness/Meta-Harness/AHE runs for TauBench and AgentBench; OS uses `os_comparison/` |

## Custom benchmark extensions

Use the [extension API](extensions/README.md) to connect another benchmark through a native adapter and run Life-Harness evolution, final TEST and accuracy/token accounting. A complete offline example is available with `python -m evolution.extensions demo --output evolution/experiments/extension_demo`.

## Lower-level Life-Harness proposals

For a single proposal, prepare a workspace from a complete incumbent TRAIN evaluation. `state.json` binds the active artifacts; `summary.json` and `episode_TASK_ID.json` bind the matching public TRAIN trajectories:

```bash
TauBench/.venv/bin/python -m evolution.life_evolve.profile prepare \
  --seed-run RUN/DOMAIN --eval-dir RUN/DOMAIN/evals/incumbent \
  --output RUN/DOMAIN/optimizer_workspace
TauBench/.venv/bin/python -m evolution.life_evolve.profile propose \
  --workspace RUN/DOMAIN/optimizer_workspace --output RUN/DOMAIN/candidate
TauBench/.venv/bin/python -m evolution.life_evolve.profile check \
  --workspace RUN/DOMAIN/optimizer_workspace --candidate RUN/DOMAIN/candidate
```

`prepare` and `check` are offline; `propose` calls the optimizer model. A proposal has a 10,000,000 provider input+output token ceiling, at most three preflight repairs and no interaction-count cap. Usage and working-memory checkpoints persist. See the [Life-Harness prompt](life_evolve/PROMPT.md), [candidate schemas](life_evolve/SCHEMAS.md) and [optimizer guide](life_evolve/README.md).

For complete Life-Harness runs, use the unified runner or:

```bash
TauBench/.venv/bin/python -m evolution.life_tau_run --name life_tau --launch

TauBench/.venv/bin/python -m evolution.life_agentbench_run --domain alfworld --name life_alfworld --launch
```

AgentBench domains are `alfworld`, `dbbench` and `webshop`. Use a fresh name with `--launch` to initialize and start training plus final evaluation. Omitting `--launch` prepares only; an already prepared workspace must be started with `--run`, since `--launch` itself initializes a new workspace.

## Separate plugin interfaces

`life_loop.py` and `agentbench_loop.py` retain plugin-based Life-Harness iteration; `meta_harness.py` retains a free-form Meta-Harness plugin interface. Their search spaces differ from the current Life-Harness bundle optimizer and native comparison controllers. Report the entry point when using them in experiments:

```bash
python evolution/life_loop.py --help
python evolution/meta_harness.py --help
python evolution/agentbench_loop.py --help
```

For plugin-based Tau runs, pass `--from-scratch` to the Life-Harness and Meta-Harness commands for identity initialization. For plugin-based AgentBench runs, use `--source evolution/reproduction/agentbench_source` with the matching baseline file.

`reproduction/` contains required runtime code and frozen task/source inputs. `experiments/` is reserved for ignored new run outputs. Public TRAIN evidence drives evolution; final TEST results never enter optimizer workspaces.
