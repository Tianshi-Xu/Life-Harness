# Current Life evolve entry point

User-requested interaction policy (2026-10-04): no total API-request/interaction
limit. The initial candidate can use as many read/reason/edit turns as needed.
After a failed preflight, grant at most **three repair opportunities**. A repair
can itself use any number of interactions; its next failed validation consumes
the next opportunity. Initial failure plus three unsuccessful repairs terminates
the proposal. Multiple failed checks within one assistant response are one cycle.
Read/write tool errors are not candidate preflight repair cycles.

Qualified candidate retention, public/approved-business-state H2–H5 interfaces, add/modify/delete,
same-complete-TRAIN-batch strict acceptance, on-demand evidence and token usage
accounting are retained. A failed proposal never authorizes promoting invalid code,
advancing an unevaluated round or resetting repair counters through checkpoint
resume. API output truncation and actual provider context exhaustion still stop
normally; an already qualified candidate can be restored. Unknown inflight request
outcomes must not be blindly retried.
Retention uses the latest qualified candidate, never the number of operations.

DeepSeek Flash output remains 100000 per request; no local Flash total-context or
token-spend cap is introduced. The outer launcher must retain global Flash
concurrency <=8. This module is synchronous and starts no background calls. The
local solver's 32768 total-context / 4096 per-output settings are not changed.

Use `python -m meta.life_evolve.profile prepare|check|propose` for **new** workspaces
and future drivers. `propose` is the explicitly paid operation; prepare/check and
the regression suite are offline. The paid CLI retains its public-only session,
token usage and repair counter beside the candidate output (`--checkpoint`
overrides that path), including failed attempts. A new driver should likewise
pass a persistent `checkpoint_dir`. This entry point does not switch any sealed
historical driver or resume its old workspace. Existing v2.5 sources and completed
six-round protocol hashes are unchanged; their validators, accounting and isolated
action-validation helpers are reused and included in new source fingerprints.

Current code-only preflight correction (2026-10-06):

- Required outputs are only `bundle.json` and referenced `artifacts/*.py`.
  Diagnosis/checks files are optional archival notes, copied unchanged. They have
  no required schema/order, are not executed, and do not affect candidate validity.
- Removed the mandatory per-case diagnosis, four-layer responsibility table,
  rule/test linkage graph and trigger/control/cross-layer fixture gates rather
  than adding more repairs or permissive schema adapters.
- The verified host workspace supplies domain/snapshot binding and a missing or
  invalid bundle ID. Mechanism revision CAS, safe paths/imports, H2–H5 signatures,
  tool contracts and isolated execution remain enforced.
- Public TRAIN replay is host-selected, independent of optional note content.
  Actual replay/runtime defects still use at most three repair opportunities.
  Qualified snapshot retention, token accounting and same-batch acceptance remain.
- This correction does not rewrite historical sources, paid artifacts, receipts,
  repair counters or seals. Existing workspaces require a new source-bound receipt;
  there is no automatic paid restart.

Methodology implemented after the old Life / Meta / AHE comparison:

- A shorter mechanism-first main prompt; exact candidate schemas are in the
  independently fingerprinted `SCHEMAS.md`. Only executable contracts are gates.
- Deterministic public symptom groups (including separate infra failures and
  possible successful controls) and per-mechanism bundle outcome history. These
  are observations, not paid diagnoses or operation-level causal attributions.
- `read_case` reconstructs deduplicated public history with the correct message
  indices or one actual recorded solver request/response. Reads remain unlimited,
  audited and allowlisted; no test answers or private evaluator data are added.
- Explicit evidence-first mechanism selection, replacing/deleting ineffective
  strategies, pivoting after repeated failures, preserving useful unrelated
  behavior and coordinating only the layers that need it. H2–H5 remain empty
  templates; no old/released/Meta/AHE domain strategies are hand-filled.
- Successful controls, state boundaries and cross-layer coordination guide the
  optimizer's reasoning, without compulsory fixture paperwork. Local replay still
  cannot infer downstream success or replace real batch evaluation.

Run order requested on 2026-10-04: implement and verify this module first, then
Meta/AHE TauBench airline/retail/telecom six-round experiments. Do **not** launch
Life's full AgentBench six-round run until that prerequisite is complete.

Offline regression: `python -m unittest discover -s meta/tests -p test_life_evolve.py`.

Latest implementation verification (2026-10-05): 220 passing meta regressions,
including restoration after provider context exhaustion with a qualified candidate
and normal failure without one. The existing snapshot restoration is reused;
no extra API retry or repair opportunity is introduced. The default paid transport
now calls the existing `verify_usage` check: missing or invalid usage fails closed,
not as zero cost. The
sealed previous six-round protocol and all 11 frozen
optimizer/interface source fingerprints still validate. No paid Life call or
AgentBench six-round experiment was launched for this implementation check.

Shared-interface completion (2026-10-06): `meta/harness_hooks.py` now supplies
policy-free public history, per-episode cross-layer state, per-call H3 context
conditioning and H2 repair/veto for all six domains. The current offline worker
is `meta/life_evolve/worker.py`; frozen v2.5/v3.1 source files are unchanged.
`source_view/INTERFACE.md` exposes the canonical contract and the released-Life
capability audit in `meta/HARNESS_INTERFACE.md`. Empty templates remain empty.

TauBench opt-in evaluation uses `python -m meta.tau_hooks --life-manifest PATH`
plus the native `eval_harness.py` flags, with `PYTHONPATH=TauBench/src:.` from the
repository root. This wrapper reuses native task selection, scoring, model arguments
and step budgets; the original CLI and historical Meta/AHE fingerprints are intact.
TRAIN evaluations additionally export public traces to `life_train/`, directly
usable by `prepare`; TEST evaluations never export optimization evidence.
Interface/artifact hashes accompany new results. No paid experiment is started by
prepare/check or the offline tests. Interface capacity is not a performance claim.

Offline verification for this completion: 259 Meta/Life tests, 16 existing TauBench
transport tests and 7 AgentBench credential tests passed (282 total). The TauBench
transport suite required local-loopback socket permission; all upstream requests
were mocked. Compilation and `git diff --check` passed. The native TauBench CLI
SHA-256 still matches the completed comparison's `frozen_sources.json` entry.

User-authorized business-state completion (2026-10-06): unified interface v2 adds
`bind_environment_state(self, state)` to ALL four layers. TauBench binds only the
current agent-side business DB as detached JSON, refreshing before model/action
boundaries; no live DB/write-back, evaluator or user-simulator state is passed.
No full DB is automatically injected into model prompts. `meta/environment_state.py`
records hash-checked initial state plus deduplicated leaf changes for offline replay.
Evolve can inspect selected TRAIN facts explicitly through existing `read_case`;
recorded snapshots refresh at the original model/action boundaries. State-bearing cases
are replayed in separate bounded worker processes to avoid stacking 50 full DBs
inside the sandbox. Historical state-free evidence remains supported as unavailable
(`None`), without forced reruns or changing old files. AgentBench adapters may bind
the same capability with the optional host `environment_state_provider`; no generic
filesystem/SQL/benchmark-private access is added. No paid experiment was launched.

Business-state completion verification: 271 Meta/Life + 16 Tau transport + 7
AgentBench credential tests passed (294 total). An offline size probe using the
actual Airline business DB, four state-reading layers and 20 successive changed
states passed in approximately 1.76 seconds under the unchanged 3-second CPU /
384-MiB worker bounds. Frozen views reuse unchanged branches and apply only changed
paths. Compilation and `git diff --check` passed; the historical native Tau CLI
hash is unchanged. These checks establish interface/transport behavior, not accuracy.

Telecom-only six-round launcher (2026-10-06):
`python -m meta.life_evolve.tau_run --name life_taubench_telecom_six_20261006 --launch`.
It reuses immutable bundle composition, the current optimizer, native Tau adapter,
local solver proxy and metered MR broker. Optimizer and Telecom user simulator share
one concurrency-8 gate; only `life_optimizer` receipts count as evolution expense.
Each candidate is selected against the incumbent on the same complete 50-task
TRAIN batch. The retained snapshot is then measured on AHE's corresponding batch;
that measurement is reused as the next incumbent evaluation. Starting from round
3, a single deficit of at least 30 percentage points stops the run; two successive
deficits of at least 20 points also stop it. Invalid/unmeasured proposals and
ambiguous accounting stop without advancing a round or blindly replaying a paid
session. TEST is run once only after six completed rounds and a training seal.
Native task IDs are hashed into stable public evidence identities, avoiding
Telecom fault labels and unsafe filename characters. Historical artifacts are untouched.

Code-gate correction offline audit (2026-10-06): the previously paid Telecom
round-1 candidate in `life_taubench_telecom_six_20261006` passes the new executable
preflight with all four operations retained: 50 TRAIN traces / 1,402 recorded
calls replayed, zero failed replay cases. Its diagnosis/checks notes were not
modified or used as gates. A temporary fresh workspace was prepared from the
preserved complete baseline; hashes of every original experiment file matched
before/after. No optimizer, solver or user-simulator API was called. This is not
an accuracy measurement or a resumed experiment.

Regression verification: 277 Meta/Life tests passed using `TauBench/.venv/bin/python
-m unittest discover -s meta/tests`; 7 AgentBench credential tests passed. The
updated prompt check, compilation and whitespace checks also passed. Coverage
includes all six runtime domains, real hook/schema errors, stale mechanism CAS,
unsafe-code filtering, qualified retention without required notes, optional-note
preservation, three repairs, unlimited interactions and optimizer-only accounting.

OS prompt comparison (2026-10-08; rolled back 2026-10-09): an experimental
prompt added investigator, skeptic and designer reasoning passes. Its fresh
six-round TEST result was 44/144, below the original Life 48/144. PROMPT.md is
restored byte-for-byte to the original experiment's prompt (SHA-256
2390c396e8151484037ff17e7b976485f807fa101777563d11f1398793bcce6c).
The experiment and its prompt snapshots remain archived; the shared seven-domain
report selects the original Life OS run. Tools and frozen-input preflight remain
unchanged. The launcher below uses whichever current prompt is explicitly frozen
for a new run; it does not automatically select the experimental prompt.

`python -m meta.os_comparison.life_prompt_run --name NEW_NAME --reference
/path/to/completed/os/reference --launch` creates an independent Life-only six-round
run from empty H2-H5 using the reference TRAIN schedule and solver configuration.
It archives the previous prompt, records both prompt hashes, and reuses the native
OS evaluation, immutable composition, strict same-batch selection and metered
optimizer. Meta/AHE are not paid or retrained. TEST144 is run after the Life
training seal; final JSON and CSV are written synchronously before controller exit.
The shared TEST has already been inspected, so protocol, input receipts and results
explicitly mark prior_test_exposure=true and describe this as a repeated comparison.
The archived experimental optimizer prompt contains no TEST examples or grader mismatches.
