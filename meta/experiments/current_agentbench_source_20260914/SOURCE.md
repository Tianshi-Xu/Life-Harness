# Current AgentBench iteration source

This directory is the frozen source for AgentBench iterations started after the
four-hook migration. It contains the formal ALFWorld and DBBench Harness files,
the current Tasks and `FourHookSession`, and the same two 50-task training pools
used by `agentbench_migrated_20260913`.

The pools are reused training data, not a new held-out split. A derived run must
evaluate its chosen baseline again and accept a candidate only when the fixed
full-pool success count strictly increases. The frozen Task runs with its own
harness disabled; `EpisodeSession` applies the candidate hooks and marks their
actions as already H2-validated, which prevents a second legacy action repair.

Example preparation:

```bash
python meta/agentbench_loop.py \
  --source meta/experiments/current_agentbench_source_20260914 \
  --run-dir meta/experiments/next_alfworld_run \
  --domain alfworld \
  --baseline meta/experiments/current_agentbench_source_20260914/alfworld/baseline.py \
  --rounds 3 --screen-size 16 --prepare-only
```

Use the DBBench baseline path and domain for DBBench. The manifest freezes every
snapshot file and both baseline hashes. Historical source directories retain
their former Task and compatibility stub solely for replay.
