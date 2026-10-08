# AHE native integration verification

Status: complete. Six native environment bridges and original
Debugger/Evolve/new-workspace/native-evaluation closure are verified. Both
configured iterations finished; no integration episode processes remain.
All numbers below are **training smoke results**, not test scores or comparative
effectiveness estimates. Baselines use local Qwen3-4B; original optimizer/debugger
and TauBench native user/judge use authorized DeepSeek V4 Flash.

## Six environment transport checks

| Environment | Training task | Success | Infrastructure error | Evidence job under smokes/ |
|---|---|---:|---|---|
| DBBench | 2491 | 1 | false | db_second/2026-10-02__14-39-39_e1e145d7 |
| WebShop | 321 | 0 | false | webshop_initial/2026-10-02__14-49-21_ac24a47d |
| ALFWorld | 503 | 0 | false | alfworld_initial/2026-10-02__14-49-53_1e63dca9 |
| airline | 0 | 0 | false | airline_initial/2026-10-02__14-55-21_809bf013 |
| retail | 73 | 1 | false | retail_initial/2026-10-02__14-55-52_1c152d46 |
| telecom | [service_issue]airplane_mode_on\|unseat_sim_card[PERSONA:None] | 1 | false | telecom_initial/2026-10-02__14-56-26_8f9aa3b1 |

Each job contains actual native records, real NexAU raw/cleaned traces and a
workspace receipt. A normal task failure is not an integration failure.

## Actual evolving-code loading and separation

- `smokes/runtime_variant/evaluation/2026-10-02__14-52-55_48ab1e63`
  records a real DB episode after a manual instrumentation-only fixture changed
  both a Python tool wrapper and a middleware. Both execution markers appear in
  `nexau.txt`; success=1, infrastructure_error=false. This is a runtime proof,
  not a hand-authored policy or an AHE-generated improvement.
- `smokes/isolation_probe/proof.json` records a nonroot solver UID and confirms
  host grader/credential paths are absent.
- Baseline policies/memory are blank. Public schemas/instructions are native,
  and no released Life mechanisms or H2-H5 constraints are imported.
- Final configuration isolates **all three** code-capable processes: solver,
  Debugger and Evolve. Actual ModelRouter credentials remain in a host-side
  loopback broker; isolated agent shells receive only EMPTY. Requests use the
  original OpenAI protocol, including streaming, with actual returned reasoning
  fields preserved at the transport boundary.

## Original optimization closure

`experiments/db_core_smoke_20261002` measured 3/3 on its first three train tasks.
Original AHE correctly stopped at its target, so this does not prove evolution.

`experiments/db_core_closure_20261002` measured 5/12 (41.7%), seven ordinary
failures and zero infrastructure-error episodes. The evaluation completed, but
subsequent analysis/evolution encountered integration dependency/path problems.
Those measurements are retained and reused rather than rerunning all 12 tasks.

`experiments/db_core_closure_permissions_20261003` is the isolated closure run.
It uses identical baseline/train IDs: its predecessor used original `skip_eval`
to reuse the baseline, and the final retry used original `start_iteration=2`.
Both continuation receipts preserve provenance. Original Debugger/Evolve and
the second evaluation are complete: 6/12 (50.0%), zero infrastructure errors,
compared with the reused baseline's 5/12 (41.7%). Task 1140 newly passes, all five
baseline successes remain successful. This is only a fixed training smoke.

| Original evaluation | Harness SHA256 | Success | Infra errors | Provenance |
|---|---|---:|---:|---|
| iteration 1 | 098b25c0ca66d03869e7435af736515aca35f7d17b8c54a71b8608612417f1f0 | 5/12 | 0 | unchanged completed baseline reused |
| iteration 2 | 52369da2f991a51f142ba81fd5e2ab8db0d4e36cb5d099d65d2b847d70e7c591 | 6/12 | 0 | actual AHE candidate, fresh native execution |

Measured candidate: runs/iteration_002/input/workspace under the latest
experiment. Original best-ever selects iteration 2. The subsequently generated
workspace candidate is not measured by this two-iteration loop.
Its SHA256 is db32976eb8ed8f109661203ff007d13bde36f70e62eefcaba56430a200d6cf3a,
and it is separately marked unmeasured, not assigned the 50% score.
Genuine native traces include the new `[SQL DIAGNOSTIC]` middleware feedback
(task 17: nine matching non-system messages; task 206: one), proving actual
candidate execution rather than config-only modification.

Final retry config samples one failed task for Debugger. Its three original
attempts return invalid `ask` output shape; the official fallback is preserved
and Evolve reads genuine raw/cleaned public traces. Earlier streamfix retry
recorded one successful original Debugger analysis and two fallbacks. Successful
native grades do not mean every Flash Debugger output satisfies its contract.
The preceding empty-reasoning run completed original Evolve and committed its
candidate, but all twelve second-round imports failed because private generated
file modes were copied to a different solver UID. No policy change was made to
repair this: the solver's private copy now has readable, root-owned, read-only
source. The latest run checks identical candidate bytes, archives the failed
evaluation/attribution in its new copy and uses original start_iteration=2.

## Preserved failed attempts and security limitation

Early smoke failures involved missing runtime devices/console-script paths,
Debugger AHE_HOME discovery, incompatible ag-ui-protocol 1.0, SOCKS dependency
connection-proxy compression handling and shared SDK temporary-directory
permissions. Fixes are dependency or infrastructure
changes, not edits to AHE's tracked optimizer sources. Each attempt remains in
its own directory; no historical Life/Meta result was overwritten.

During `db_core_closure_retry3_20261002`, before Debugger isolation was added,
an original Debugger shell initiated a broad host file search outside the public
training-trace directory. The exact verification process and all child processes
were stopped immediately on discovery. That attempt is not used as optimization
evidence. Original Debugger tracing was disabled, so the full model-visible tool
outputs from the stopped attempt cannot be reconstructed; **retrospective proof
of zero out-of-scope exposure is unavailable**. The subsequent filesystem
isolation and credential broker address those access paths; do not claim the
earlier attempt had the final security boundary.

A credential audit found ten generated Evolve YAML files containing the real
key: upstream resume serializes resolved connection settings. These ten derived
files were mechanically redacted to environment references, without changing
other fields or official/historical Life/Meta sources. Final launcher connection
resolution exposes only EMPTY/local broker URL to the original resume path.
The post-redaction audit result is recorded after final verification.
The final audit checked 23,656 derived text/config/trace files with zero real-key
matches; details are stored in verification_receipt.json. Changing the key is
recommended because historical exposure cannot be completely reconstructed.

## Fidelity and scope

- Official AHE pin: 8b2a55d97590363fe50c3cc6b5e833b020a4bb4c; tracked source clean.
- NexAU pin: 35ee1861546db3cb280a6e17e38a74060d7c96c3; tracked source clean.
  All 191 installed Python source files also match the pinned checkout; final
  receipt checks both host-installed and isolated-runtime copies.
- Frozen benchmark manifest SHA256:
  805d13dc3d07019cbd30e0cc50ac400526d3866aeaf8e6c96f0f807537732edd.
- Original search/statistics/selection remain unchanged. Original Debugger and
  Evolve function bodies execute inside isolated process wrappers, with path and
  API-connection transport adaptation only.
- Twenty-one offline tests pass; API broker nonstream and stream tool roundtrips pass.
  API-returned thinking content is restored at transport only; thinking remains enabled.
  A real NexAU streaming multi-turn/long-argument probe also passes:
  smokes/api_stream_probe/proof.json. Stream tool IDs use the same complete-value
  update semantics as pinned NexAU, not concatenation of repeated IDs.
  An additional loopback SSE regression verifies completed-response-only caching;
  partial reasoning chunks must not mark a genuine response ID as ambiguous.
  `smokes/api_stream_probe/restoration_proof.json` exercises actual NexAU with
  diagnostic-only deliberate reasoning stripping: completed=true, hits=3,
  misses=0. This fixture is never installed in the optimizer or solver.
  Response matching also supports unique actual call IDs after prefix rewriting
  or parallel-call splitting; conflicting IDs are rejected, never guessed.
  The unique-ID attempt's receipt isolated the remaining failure: observed API
  tool responses had reasoning_length=0, but old runtime serialization dropped
  that empty field. The broker now replays the actual zero-length value for those
  known responses, without inventing reasoning or disabling thinking.
  The completed final run records 1,025 historical-field restorations, zero
  misses, and 113 cached tool-call responses (86 with text, 27 zero-length).
  Restorations are not a count of API requests. Thinking remains enabled.
- This smoke has k=1, fixed subset, concurrency=2, optional Explore/Best-of-N off.
  Original Debugger max_tasks=1 for the latest closure retry (earlier attempts: 3)
  is a smoke sampling setting, not a new optimization constraint.
  Their upstream source is retained, but these branches are not validated here.
- AHE's evaluate-before-evolve convention is retained. Last generated candidates
  are not automatically measured; do not report them as improved scores.
- No full test evaluation or formal six-round AHE/Life/Meta comparison is claimed.
