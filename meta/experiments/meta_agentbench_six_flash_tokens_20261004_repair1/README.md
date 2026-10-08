# Meta AgentBench: six-round Flash API rerun with token receipts

Repair continuation: the preceding startup is preserved at
`meta_agentbench_six_flash_tokens_20261004`. Its 90 Flash calls / 2,490,771 tokens
are explicitly included in `total_spent_including_startup`, separately from the
valid six-round protocol. No startup proposal was scored or accepted. Only the
45 baseline infrastructure-error episodes are recomputed; 105 completed baseline
episodes are copied with exact source hashes. The original three per-request Qwen
transport retries are restored. Existing shared Qwen services are not modified.
The host pauses before paid optimization if a 50-task batch has more than two
infra failures, while isolated failures remain zero in the fixed denominator.

This NEW experiment preserves the historical Meta run and starts ALFWorld,
DBBench and WebShop from its policy-free identity baseline. The original
`run_evolve`, frontier/summary logic and original Meta skill are copied byte-for-byte
from the frozen historical reference. Its required one analysis Agent and one
implementation Agent run in independent contexts. This is an API execution-shell
port, not a byte-for-byte reproduction of the official Claude/Qoder CLI backend.

Each scene schedules six optimization attempts, one free-form mechanism/candidate
per attempt, using the existing frozen 50-question TRAIN batches. A strictly higher
same-batch success count is required for acceptance. A rejection switches to the
next batch; an acceptance retains the batch and reuses the incumbent's evaluation.
All public TRAIN evidence is available on demand, with no ten-case quota.
All three first-round candidates must be generated and evaluated with complete
API usage before the remaining five rounds are allowed to start.

The optimizer is ModelRouter `deepseek-v4-flash`, one host-side global concurrency
gate of 8, output max 60000 per response and no artificial local context bound.
The original proposer wall-time safety limit of 5400 seconds is retained. No
cumulative optimizer token ceiling was requested and none is invented: label this
as a **six-round iteration-budget experiment with measured token usage**, NOT as
an equal-token-budget experiment. A later comparison can align checkpoints using
cumulative receipts; differing training batches and this backend port must be shown.

`*/flash_requests.jsonl` stores raw upstream usage for parent, analysis and
implementation calls, including retries and errors. `TOKEN_USAGE.json` groups
input/output/total/cache information by domain, round and role. Total is input
plus output; cache input is a subset, and reasoning is a subset of output. Missing
usage or timeouts are UNKNOWN, not zero, and stop further optimization. Credits
are not used as a token proxy. Qwen solver tokens are logged separately per episode
and in each TRAIN decision. Two-task setup smokes are not formal TRAIN rounds.

The solver remains local Qwen3-4B, temperature 0, output 4096, total context 32768,
50 native workers per domain. Public environment source and dataset hashes are
identical to the historical Meta partition. Released H2-H5 content is disabled and
unavailable in the isolated optimizer workspace. Context exhaustion is a task
failure; other single-episode errors remain zero in the fixed denominator, without
silently overwriting/retrying them. Setup failures stop before paid optimization.
ALFWorld uses a local tokenizer helper in the existing core environment because
its older native venv has no transformers; no shared environment is modified and
this helper makes no LLM/network calls.

After all three six-round TRAIN states are sealed, the optimizer API is closed and
the retained candidates receive one full final TEST (109 / 300 / 200 tasks).
TEST files never enter optimizer-visible history/evidence or influence acceptance.
The task partitions have prior exposure; this is not an unseen-test claim.

Real credentials are resolved in host broker memory only. Solver subprocesses
have credential variables removed. Optimizer tools run in a dedicated unprivileged
chroot with only public inputs. Agent shells/configs/logs contain no real key.

Preparation and offline tests:

```sh
python -m unittest discover -s meta/experiments/meta_agentbench_six_flash_tokens_20261004_repair1 -p 'test_*.py'
python meta/experiments/meta_agentbench_six_flash_tokens_20261004_repair1/prepare.py
```

Launch with the existing `ahe-native/.venv/bin/python` (httpx is installed there):

```sh
/mnt/data/xts/harness/ahe-native/.venv/bin/python meta/experiments/meta_agentbench_six_flash_tokens_20261004_repair1/runner.py --launch
```

Progress: `status.json`, `*/progress.json`, `*/state.json`, `first_round_gate.json`,
`flash_metrics.json`, `TRAINING.md`, `TOKEN_USAGE.json`, `runner.log`.
Completion: `training_seal.json`, `FINAL_RESULTS.json`.
Existing artifacts are hash-checked before/after; source/manifest drift fails closed.
