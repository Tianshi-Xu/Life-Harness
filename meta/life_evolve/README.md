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

User-requested cost control (2026-10-10): new proposals have a **10000000 input +
output token ceiling**, below twice the old Telecom mean of 5076000. The host
reserves conservative encoded-input bytes plus the full 100000 output allowance
before each request; remaining budget may therefore be left unspent. Actual cost
uses provider input/output usage, including cached input. A budget stop restores
the latest qualified candidate, or stops without advancing a round if none exists.
The checkpoint binds budget and usage: restarting cannot reset spend or repairs.
Past paid receipts and experiment protocols are not rewritten.

To reduce spend rather than only stop at the ceiling, a long optimizer transcript
triggers a **same-agent working-memory checkpoint** at 65536 locally counted input
tokens. This threshold is not a provider-context limit and never truncates an API
response. The same Flash agent preserves evidence/code references, findings,
hypotheses, alternatives, regression risks, current candidate and next steps in
its own memory; no host-written domain strategy or auxiliary model is introduced.
The next request uses that memory and exact host validation/repair bookkeeping.
Full old messages (including reasoning and tool pairs) are hash-bound under
`context_history/`, readable/searchable with existing tools but not model-writable.
Candidate files, qualified snapshots, total usage and repairs persist unchanged.
The latest validation report is also saved and restored with the agent checkpoint,
so repeated memory refreshes after a safe pause retain exact host error feedback.
Checkpoint calls count as optimizer expense, with each checkpoint's own provider
usage saved alongside cumulative usage. Empty/malformed memory does not replace
the old context or terminate the proposal: record its cost and error, pair any
unadvertised calls with not-executed feedback, and continue a normal tool turn
before another optional refresh. This does not grant a preflight repair or reset
spend. Actual output truncation still stops or hands off a revalidated candidate.
`copy_file` seeds a candidate artifact from frozen active/history Python source;
`replace_file` then changes exact spans, avoiding reprinting unchanged functions.
Source/evidence reads remain unlimited. Actual cost
and accuracy effects need a future authorized TRAIN comparison; offline tests
cannot establish accuracy non-regression. Exact whole-string deduplication was
evaluated on the prior round-5 transcript (only 0.20% local input-token savings)
and discarded, rather than accumulating a low-value duplicate implementation.

DeepSeek Flash output remains 100000 per request; no local Flash total-context
cap is introduced. The outer launcher must retain global Flash
concurrency <=8. This module is synchronous and starts no background calls. The
local solver's 32768 total-context / 4096 per-output settings are not changed.

Use `python -m meta.life_evolve.profile prepare|check|propose` for **new** workspaces
and future drivers. `propose` is the explicitly paid operation; prepare/check and
the regression suite are offline. The paid CLI retains its public-only session,
token usage and repair counter beside the candidate output (`--checkpoint`
overrides that path), including failed attempts. A new driver should likewise
pass a persistent `checkpoint_dir`. This entry point does not switch any sealed
historical driver or resume its old workspace. Historical experiment artifacts,
receipts, counters and seals remain unchanged. Shared validator error reporting
was upgraded on 2026-10-10; new workspaces bind the new source fingerprints.
Replaying an old receipt requires its original source snapshot, not rewriting
old hashes to match current code.

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

Evolve diagnostics and development tooling (2026-10-10):

- Bundle rejections retain operation index/ID, mechanism, artifact and precise
  missing/extra fields, identifiers, revision/hash CAS or path failures, including
  when every operation is rejected. Imports and hook signatures identify the
  offending expression/line and expected contract. Full TRAIN replay identifies
  case, call index, hook and responsible or related artifacts; action-schema
  errors name nested argument paths. Worker termination reports signal/exit,
  affected cases and the existing CPU/RAM/wall bounds. Tool errors include the
  expected argument schema; read/state ranges, immutable input/checkpoint/cache
  changes and usage errors explain how to audit or repair them without resetting
  paid work. No safety predicate, native scoring or runtime budget is relaxed.
- Repeated replay failures are grouped in model feedback with every case/call
  occurrence retained. Full per-case reports remain in validation artifacts.
- Optional search_file provides paged literal-text byte offsets; replace_file
  replaces one unique exact span in candidate outputs. These reuse the existing
  read/write allowlists and bounds; there is no total search/read/edit quota.
- Optional preview_candidate replays one chosen TRAIN prefix in the same isolated
  worker, showing H3 description/model-view changes, H4/H5 guidance, H2 action and
  feedback, and scalar episode-state summaries. Long strings are explicitly
  marked truncated; containers are type summaries, not a full DB dump. Preview
  neither calls an LLM/environment nor qualifies a candidate, narrows full replay
  or consumes a preflight repair opportunity.
- Public history can expose hash-verified prior proposal code and paired TRAIN
  gain/regression case IDs when matching recorded summaries exist. Rejected code
  is read-only evidence, never activated; missing/unverifiable history is marked
  unavailable. Historical TEST details are not projected into these views.
- The shorter mechanism-first prompt uses these tools and regression evidence,
  asks for coverage/risk-aware fixes and coherent state/feedback, and preserves
  optional notes, unlimited interactions and at most three repair opportunities.
  H2–H5 interfaces/templates, model settings and external concurrency <=8 remain.

Before applying this upgrade, the completed Telecom branch's 24 optimizer source
fingerprints were verified and saved locally under the git-ignored
`meta/experiments/life_evolve_feedback_upgrade_20261010/source_before/` with
`source_audit.json`. Old experiments were not rewritten or paid sessions restarted.
Offline tests and frozen-input replay establish implementation behavior, not an
accuracy improvement; a new source-bound paid experiment is needed to measure it.

Verification on the merged source (2026-10-10): 267 offline regressions passed;
the unchanged real Telecom candidate passed all 817 calls across 50 TRAIN cases.
68 bundle, 14 static-source and 14 action-schema old/new acceptance comparisons
matched. Original 24 archived sources and completed-run receipts were verified;
this verification made no external LLM calls and started no native evaluation.

The round-two branch launcher now accepts an explicit --source-upgrade-audit
with exact archived-before/current-after fingerprints for these optimizer and
diagnostic sources; native scoring, tasks and solver changes remain disallowed.
Optional --history-from exposes a closed, sealed TRAIN branch as additional
read-only history. Inherited rows are deduplicated, different past code remains
separate, and complete ancestor exploration cost is recorded separately from
new spend. This does not activate rejected code or authorize a TEST rerun.

First paid tooling-upgrade probe (2026-10-10): the Telecom round-two continuation
produced a qualified candidate with zero preflight repairs, but scored 39/50 TRAIN
against the same-batch incumbent's 39/50. Eight tasks improved and eight regressed;
the candidate was not accepted, with zero infrastructure failures and no TEST run.
All 101 optimizer requests have measured usage: 23,015,966 new tokens, 64,713,337
including complete ancestor exploration. This is evidence that tooling works,
not that accuracy improved. That probe exposed a feedback gap: historical code
and task flips were readable, but read_case only selected incumbent trajectories.

Regression-aware evidence upgrade (2026-10-10, optimizer only): read_case now
accepts a zero-based history_index for hash-bound public TRAIN trajectories from
past candidates, including rejected ones. Optional metadata pins existing sealed
files without copying large DB logs; missing evidence is reported explicitly,
never replaced with TEST or incumbent evidence. Existing privacy filtering and
explicit model/action-boundary business-state reads are reused. Public setup is
available via include_setup=true instead of being repeated in every response;
recorded-call messages can be paged independently of episode history. Full
evidence access remains unlimited. The compact prompt prioritizes observed
regressions and encourages handoff once a coherent qualified change is ready,
without imposing any total interaction/case/token cap. Multiple --history-from
branches retain distinct failed attempts and count their measured costs once.
H2–H5, native evaluation, solver settings and acceptance gates are unchanged.
Previous paid-run sources were archived under
`meta/experiments/life_evolve_feedback_traces_upgrade_20261010/source_before/`;
those paid runs, results and receipts remain immutable.
Verification for this evidence upgrade: 273 offline regressions passed; all 817
calls across the same 50 real TRAIN cases replayed. Eight historical attempts
have readable candidate traces, 21 prior code files remain separate, and each of
the latest eight regressions returns the actual success/failure pair. This is
implementation evidence, not proof of a better candidate's accuracy.

Paid regression-feedback probe completed: 45/50 TRAIN (90%) against the reused
39/50 incumbent (78%); accepted, 10 gains and 4 regressions, zero infrastructure
errors. All nine preflights passed with zero repairs. New optimizer spend is
26,123,863 measured tokens across 103 requests, 90,837,200 including all ancestor
exploration. The model read 15 TRAIN cases but never selected historical candidate
traces; soft handoff guidance did not reduce interaction/spend. No new TEST was
run and original six-round records stay unchanged. After verifying its sources,
receipts, seal and exited processes, all 24 source files were preserved under
`meta/experiments/life_telecom_evolve_regression_probe_20261010/source_before/`.

Follow-up evidence correction allows valid H3 description-only changes between
recorded calls and returns the selected call's descriptions with include_setup.
Native names/parameter contracts remain immutable; no solver, runtime or scoring
change is made. This offline correction happened after the paid probe was sealed,
so its accuracy gain cannot be attributed to the follow-up fix.

Evidence-first continuation: the initial optimizer message now includes measured
latest-primary-attempt outcomes and paired gain/regression IDs, without generating
a diagnosis or inserting policy content. read_case's history_variant='incumbent'
allows the matching historical baseline to be compared with its candidate even
after batch rotation. Copied ancestor records gain missing code and trace indexes
from registered sealed sources without duplication. Case metadata includes
history_length for targeted reads. Qualified feedback recommends handoff to real
evaluation; cases, interactions and total tokens remain unlimited.

Follow-up after round four: preview_candidate can inspect the retained chain
before a bundle exists or explicitly with chain='incumbent'. A malformed working
candidate still fails candidate preview and full preflight, never silently falling
back. Rotated-case errors route to paired historical reads, and the prompt names
both historical variants explicitly. These are diagnostic corrections, not new
qualification gates or manually filled domain policies.

After round five's tied 42/50 candidate, Evolve now binds omitted modify/delete
version guards from the frozen inventory; explicit wrong guards still fail.
Startup feedback includes hash-verified public paired regression tail excerpts,
not diagnoses or policies. Full case access remains unlimited and unavailable
history is explicit. These changes do not alter H2–H5, the native budget/scorer
or the already sealed paid runs.

--next-round-from creates one new rotated TRAIN round from any sealed incumbent
before round six, preserving accepted versions/state and all recorded ancestor
cost once. Sources and ancestry receipts are pinned; no bulk logs or TEST data
are copied, and no TEST is run. This does not resume a paid session or edit old
receipts. Each one-round continuation is inspected before authorizing another.

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

Implementation verification milestone (2026-10-05): 220 passing meta regressions,
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
