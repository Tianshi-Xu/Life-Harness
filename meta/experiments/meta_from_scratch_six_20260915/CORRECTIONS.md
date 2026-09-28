# Runtime corrections

## Post-run source packaging (2026-09-28)

This branch originally pointed to a Meta-Harness clone outside the repository
and omitted 39 task snapshot files under an ignored experiment directory. A
fresh checkout therefore could audit scores but lacked source files needed to
inspect or reuse the frozen method. The exact upstream files from commit
`44b9942127847f7421db70d8c7e48407f09a3c70` and the frozen task snapshot
were added to this experiment directory. Their hashes match `manifest.json`.
Only the upstream source path in `adapter.py` and `prepare.py` changed;
`frozen_run_20260915.json` preserves their original run-time hashes and
`frozen.json` hashes the packaged drivers. `AUDIT.json` was regenerated after
adding source integrity checks. No model call, candidate, score, or task content
was rerun or changed.

After ALFWorld round 1 finished all candidate episodes, the adapter wrote the
small compatibility summary expected by upstream Meta-Harness onto the native
evaluator's `summary.json`. This removed aggregate keys before the acceptance
record was committed. All 50 immutable episode files remained intact. The
compatibility summary now lives in a separate upstream job directory. The
native summary was deterministically rebuilt from those episode files, with no
new model requests, and the same candidate proceeded to its required test run.

## DBBench round-4 mutable pending file

The DBBench round-4 proposer correctly produced
`declared_surface_restoration.py` and the requested round-local
`logs/round_04/pending_eval.json`, but it also rewrote a stale workspace-root
`pending_eval.json`. The adapter's input immutability check stopped the run
before validation or evaluation. The round-local pending file and candidate
were copied unchanged to the adapter's resumable `pending_04.json`, then the
same round was resumed. No solver call or benchmark result existed to discard.

## Telecom batch-1 infrastructure retry

The telecom batch-1 incumbent evaluation ran all 50 requested tasks, but one
simulation ended with `infrastructure_error` and had no reward. The native
summary therefore contained only 49 scored task keys, and the adapter's strict
cardinality assertion stopped the run before any proposal. The 49 valid
simulations were retained, the single failed task was rerun with the identical
identity candidate and frozen evaluation settings, and the successful retry
replaced only the failed simulation before aggregate scores were rebuilt. The
repaired phase still contains exactly one valid result for each of the frozen
50 task IDs.

## Telecom batch-2 user-model rate limit

Telecom rounds 1--4 used `openai/deepseek-flash` as the user simulator. Its
logs contain ModelRouter HTTP 429 responses stating that the API key had
reached its RPM limit; TauBench retried them, but the new batch-2 incumbent
baseline continued to hit the same limit. The user requested switching models.
That incomplete run was stopped after 8 valid simulations, archived under
`telecom/recovery_batch_02_deepseek_429/`, and is not counted as an experiment
result. Starting with telecom training batch 2 (round 5), both training and
fixed-test phases use `openai/glm-5.3-flash`. The route was verified against
ModelRouter before restart. Earlier completed rounds were retained unchanged,
so cross-round test comparisons spanning this boundary also include a user
simulator change.

The first formal GLM batch-2 attempt was also stopped before completion after
8 rate-limit errors appeared at task concurrency 8; the route reported a
60-RPM key limit. Its six completed simulations are archived under
`telecom/recovery_batch_02_glm_429/` and are not an experiment result. The
rerun keeps task concurrency 8 but paces only user-simulator requests at one
request per 1.5 seconds (40 RPM), leaving all local Qwen agent calls concurrent.

That paced GLM rerun still produced 42 rate-limit errors, four transient
authentication errors, and one `infrastructure_error` among six completed
simulations. It is archived separately under
`telecom/recovery_batch_02_glm_throttled_429/` and is also excluded. A generic
concurrent route probe then produced 64/64 successful responses for
`mr.deepseek-v4-flash`; a real one-task telecom smoke test completed with a
valid reward and no 429, authentication, or infrastructure error. With the
user's authorization, the counted batch-2 rerun and all following telecom
phases use `openai/deepseek-v4-flash`, task concurrency 8, and a conservative
16-RPM user-model cap (one request every 3.75 seconds).
