# Native Life-Harness / Meta-Harness / AHE OS comparison

Set `LIFE_SOLVER_TOKENIZER` to the local Qwen3-4B model directory, provision the
native OS Docker image/workers and the AHE environment, and configure the solver
and provider endpoints. See the root README for common setup.

```bash
python -m evolution.os_comparison.protocol --help
python -m evolution.os_comparison.protocol --name NEW_RUN
python -m evolution.os_comparison.launch --root /absolute/path/to/prepared/run --smoke-only
```

The controller uses six disjoint TRAIN batches of 50 tasks, identity baselines,
current Life evolve, the pinned original Meta loop and the pinned AHE loop.
Tests run after all TRAIN states are sealed. Public optimizer evidence, native
scoring, candidate hashes and per-request token ledgers are stored in the new run.
The PTY transport fix preserves output frames without changing native graders.
