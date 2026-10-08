# AHE native AgentBench / TauBench integration

This independent adapter runs genuine NexAU workspaces against native DBBench,
WebShop, ALFWorld, airline, retail and telecom environments. It never invokes a
frozen Life/Meta policy or replaces AHE's optimizer.

Pinned official AHE: ../agentic-harness-engineering, commit
8b2a55d97590363fe50c3cc6b5e833b020a4bb4c.
Pinned NexAU: vendor/NexAU, tag v0.3.9, commit
35ee1861546db3cb280a6e17e38a74060d7c96c3.
Both official checkouts retain unchanged tracked sources.

## Fidelity boundary

This is a benchmark port, not a reproduction of published Terminal-Bench scores.
New CLI/batch launches use the user-requested `h2345-budget-restricted-v1`
capability profile, so they are **restricted-search AHE**, not unrestricted
official AHE reproduction. Historical unrestricted runs retain their original
configuration, labels, source files and results.
Original search, statistics, history, best-ever selection, attribution, change
manifests, Agent Debugger and Evolve Agent are retained. Infrastructure wrappers:

- launch_harbor / wait_for_harbor launch native environments and wait for complete
  Harbor-shaped jobs; incomplete backend runs cannot enter optimization.
- get_llm_config resolves process-only environment references at consumption time.
- run_evolve_agent moves execution into a private filesystem, invokes the
  unchanged original function, appends the host capability contract to its task,
  and validates actual workspace changes before copying them back.
- run_parallel_adb_ask likewise invokes the unchanged Debugger batch in a
  public-evidence-only filesystem; neither agent shell can read host datasets.

No single-mechanism rule, released Life/Meta policy or score acceptance rule is
added. Baselines contain public native instructions, schemas and mechanical
RPC wrappers only; candidate content still comes from iteration. LLM/tracing
and native grading remain fixed infrastructure. The restricted component and
evidence contracts below are an explicit search-space deviation.

### Budget-restricted component contract

- Allowed: prompts, descriptions of existing native tools, pure synchronous
  middleware for episode-local state/planning, action argument repair or
  replacement, and public-message/feedback conditioning. Validated pure helper
  modules and the SDK's public `Message`, `ToolCall` and `ParsedResponse` data
  interfaces remain available; full agent/executor/LLM imports do not. Frozen Markdown skills
  and `LongTermMEMORY.md` may evolve between iterations; they cannot write
  cross-episode runtime state. They are available through the `frozen_skills`
  and `frozen_memory` Jinja context variables, without registering a loader tool.
- Disabled: new tools, native tool implementation changes, additional LLM
  calls from candidate components, subagents, multiple environment calls hidden
  inside one tool, and cross-episode writable memory. Explore/Best-of-N remain
  off. Native TauBench user simulation/scoring are unchanged benchmark services,
  not candidate-created auxiliary LLM calls.
- Enforcement is not prompt-only: host validation freezes tool implementations,
  schemas, bindings, registry, model/tracer/iteration budgets and all config
  keys except the middleware list. Candidate Python is restricted to pure imports
  and hook interfaces; execution wrappers, network/RPC/filesystem/executor access
  are rejected. Trusted runtime guards require the ordinary model execution path
  and at most one native RPC invocation per registered tool dispatch. Normal
  framework transport retries remain intact. Illegal candidates stay private,
  receive a rejection receipt, and never replace the host incumbent. Evaluation
  validates the workspace again before any task/model call.
- This is a compatibility/cost guard plus the existing filesystem isolation,
  not a general adversarial Python/network security sandbox. Host evidence is
  never rewritten to pretend an older unrestricted run used these restrictions.

Historical tool-augmenting candidates may fail this new contract. They are not
silently continued or edited; any later port/restart requires a new experiment
name and explicit provenance. The stopped nine-round batch remains stopped;
implementing this profile did not launch any new paid experiments.

Upstream evaluates before evolving. Two iterations normally measure baseline
and the first changed harness, then generate a second candidate. The last new
candidate is unmeasured until another evaluation; do not count it as an improvement.

## Runtime and data separation

Native execution/grading use existing environment interpreters. Evolving
solvers run genuine NexAU in a dedicated Python 3.13 environment. RPC carries
ordinary public actions/feedback. Genuine raw/cleaned NexAU traces and scalar
training outcomes are exported, not fabricated runtime spans.

Each solver/Debugger/Evolve process enters a chroot with an unprivileged episode-specific
UID. Datasets, grader source, host credential files and other experiments are
absent. Solver processes do not receive proposer/user credentials.
Generated private source permissions are normalized only on the solver's copy;
the enclosing episode stays private and source remains root-owned/read-only.
Candidate workspace symlinks are refused before host-side copies, avoiding
host resolution of paths created inside the isolated process.
Host CAP_SYS_CHROOT is required; bwrap user namespaces are unavailable here.
This is a filesystem boundary, not a general network security sandbox.

Only frozen training IDs may enter the backend. Test split and oracle/private
result fields are rejected, and the manifest is SHA256-pinned. Infra-error
episodes remain success=0 in the fixed denominator; incomplete jobs fail closed.
Historical Life/Meta artifacts and failed smoke attempts remain untouched.
Final tests must stay outside optimizer workspace/history/selection.

## Connections and smoke settings

- Solver: existing local Qwen3-4B, http://127.0.0.1:8400/v1, temperature 0,
  output limit 4096, thinking disabled, context budget 40000.
- Original Debugger/Evolve: deepseek-v4-flash via existing ModelRouter.
- TauBench native user simulation/NL scoring: the authorized Flash route.
- Smoke: k=1, concurrency=2, target 95%, fixed training subset. Optional Explore
  and Best-of-N are disabled for this configuration; those unchanged upstream
  branches are not end-to-end validated by this smoke.
  CLI Debugger analyzes at most ten tasks by default (or all tasks if the smoke
  has fewer than ten). `--debugger-tasks` accepts 1–10 in the restricted profile.
  Its original configurable retry count is set to one to prevent a failed
  ten-task analysis from silently tripling its cost. The legacy `config_for`
  helper retains the old defaults when no capability profile is supplied.

Credentials are loaded without displaying the key. A host loopback API broker
holds the real ModelRouter key; isolated Debugger/Evolve environments receive
only EMPTY and a local URL, so agent shell tools cannot expose real credentials.
The broker forwards the original request body and supports streaming; it rejects
any request body containing the real key. Saved configuration uses environment
references or EMPTY/local broker settings. User authorization covers public training evidence and
candidate code only, not test tasks, hidden answers or credential disclosure.
For Flash tool calls, the broker restores only reasoning_content actually
returned by the API and omitted by old NexAU serialization, matched to the
conversation prefix or globally unique, unambiguous actual response call IDs.
Conflicting IDs are rejected. It does not disable thinking or invent
reasoning. Ordinary and streaming tool roundtrips passed live preflight.
Actual zero-length reasoning on known API tool responses is preserved as an
empty field: absence of text must not be confused with an unobserved response.

## Usage

Use .venv/bin/python (Python 3.13.13), not system Python 3.12. Dependencies are
recorded in requirements.lock. Keep ag-ui-protocol==0.1.10: version 1.0 removed
event types used by pinned NexAU. socksio supports the existing API proxy.
Original Agent Debugger is installed from the upstream bundled _source.

```bash
cd /mnt/data/xts/harness/ahe-native
.venv/bin/python -m unittest discover -s tests -v
# Host execution with chroot and authorized model/network access:
.venv/bin/python -m ahe_native.cli dbbench \
  --name ahe_db_restricted_new_smoke --iterations 2 --smoke-tasks 12 \
  --debugger-tasks 10
```

Other domain names: webshop, alfworld, airline, retail, telecom. Existing
experiment names are refused. IDs are the first N frozen first-batch training
IDs. This convenience launcher is a smoke tool, not a six-round comparison
protocol. Adapter code can be bound around the original config-driven loop.

For an infrastructure-only failure before any workspace change, retry in a new
directory while reusing the measured baseline training job:

```bash
.venv/bin/python -m ahe_native.cli dbbench \
  --name ahe_db_restricted_retry --iterations 2 --smoke-tasks 12 \
  --reuse-evaluation-from ahe_db_restricted_new_smoke
```

Reuse checks exact native domain/IDs and unchanged baseline workspace, preserves
the source experiment, archives failed analysis in the new copy and delegates
metadata rollback/skip_eval to original AHE. This is not arbitrary resume for
already-evolved experiments.

For the separately verified all-import-error second-round case, an unchanged
evolved snapshot can be resumed in a new copy with --resume-evolved-from NAME.
This invokes original start_iteration=2, archives failed evaluation/attribution
and verifies candidate hashes; it does not reuse the failed score or change policy.

## Verification scope

All six domains have completed real native training episodes with genuine
NexAU traces and no infrastructure error. An explicit instrumentation fixture
also verified modified tool and middleware execution under the earlier
unrestricted port; it is not an AHE-evolved policy improvement or permission to
edit native tools under the new profile. Offline adapter regression includes an
actual loopback SSE test that caches complete reasoning once rather than treating
partial stream chunks as conflicting responses. New tests cover the six frozen
baselines, actual NexAU model/tool execution with fabricated offline responses,
ten-task evidence projection, policy propagation, forbidden component rejection
before promotion/evaluation, and ordinary retry preservation. No paid API or
benchmark episode is needed by these tests; the restricted profile has not yet
been validated by a new paid optimization run.

Original Debugger -> Evolve -> next-iteration evaluation closure is validated
separately; see experiment artifacts and INTEGRATION_REPORT.md for final status.
Small smoke scores are not effectiveness estimates or comparable published scores.

## Formal three-domain rotating training

`ahe_native.batch` starts independent ALFWorld, DBBench and WebShop experiments
around the original AHE loop. For the nine-iteration configuration, each native
train split is shuffled with a recorded seed and split into nine disjoint batches
of 50 tasks. This is 450 unique tasks per domain, not a whole-training-set scan.
The schedule is SHA256-pinned before launch and checked at every evaluation;
WebShop IDs below 200 are forbidden. Earlier smoke task restrictions remain intact.

```bash
.venv/bin/python -u -m ahe_native.batch \
  --name ahe_restricted_agentbench_nine_flash8_NEW \
  --iterations 9 --training-tasks 50 --debugger-tasks 10 \
  --flash-concurrency 8 --native-concurrency 50 --seed 20261003
```

Flash Debugger/Evolve requests share one host-side broker and a global limit of
8 concurrent API requests, held through the complete response stream. Local
Qwen requests go directly to the existing service and do not consume this limit.
Each domain may run all 50 native episodes concurrently. Existing model services
are not restarted or reconfigured.

Only configuration changes disable score-based early exit: target_pass_rate=1.01
ensures the requested fixed iteration budget. Original statistics, search,
Debugger/Evolve bodies, attribution and best-ever selection remain unchanged.
Explore and Best-of-N remain off as in the verified integration configuration.
Each round still evaluates all 50 native tasks, but original Debugger `max_tasks`
is capped at ten with one CLI attempt per selected task. Selection uses the
unchanged original failure-first `_build_adb_jobs`, not a new sampling algorithm.
Debugger and Evolve receive disposable experiment copies exposing raw traces
only for those same at-most-ten current-round tasks. Previous raw traces and
detailed analyses/CLI archives are removed only from those private copies;
previous score/history/change summaries and all host results remain intact.
This prevents Evolve from bypassing the ten-task cap by reading the other forty
traces or archived raw evidence. Ten tasks is not ten model requests: each
Debugger/Evolve interaction may still require multiple Flash generations.

Different batches are not a continuous same-task improvement curve. The original
best-ever selector still compares the measured per-batch scores; record that
sampling limitation when interpreting the chosen snapshot. The last evolved
candidate remains unmeasured under the original evaluate-before-evolve convention.
No final test is launched by this training command.

Batch configuration, schedule, source fingerprints and live status are under
`batches/NAME/`. `status.json` records actual completed episodes, measured training
outcomes, domain states and global Flash request concurrency. Generated workspaces
and genuine trajectories are under `experiments/NAME_DOMAIN/`. New names are
required; historical experiments cannot be overwritten by this launcher.

### API RPM guard and first-baseline recovery

Live ModelRouter diagnostics confirmed this key's additional limit of 60 requests
per minute. Concurrency 8 alone is not a rate limit. The formal launcher therefore
paces Flash starts globally by 1.25 seconds (48 RPM, leaving quota headroom),
while keeping the original eight in-flight slots. A 429 response defers all callers
for 60 seconds and retries the identical request at the transport boundary; model
prompts, reasoning, original Debugger/Evolve code and local Qwen requests are not
changed. `status.json` records pacing and observed 429 response counts.

The interrupted unpaced batch is retained in its original directory. Its complete
150 baseline episodes are reused by
`ahe_agentbench_nine_rotating_flash8_rpm48_20261003`. `--resume-from` requires
identical frozen schedules, unchanged baseline bytes, matching native job receipts
and complete first-batch task IDs. It copies to new experiments, archives the
unfinished Debugger stage, and delegates metadata rollback plus `skip_eval` to
original AHE. It cannot resume an already evolved or incomplete baseline.

The user also requested per-optimization-step Meta/Life comparison. The historical
now-stopped batch was configured with `--measure-final-candidate`: one additional 50-task training evaluation
after each domain's unchanged nine-iteration loop measures its last new candidate
on the last batch, without changing best-ever selection. This is not a test run.
Nine disjoint batches still contain 450 unique tasks per domain; the extra final
evaluation reuses the last 50 IDs. There are 500 measured train episodes per
domain including baseline and this final evaluation, with the first 50 reused.

`COMPARISON.md` and `round_comparison.json` update from actual completed jobs.
They align optimization step zero as baseline, step N with original evaluation
N+1, and the final step with the separate post-loop evaluation. Historical Meta
only has six rounds; ALFWorld Life also only has six. Missing rounds stay null.
DBBench/WebShop Life rounds 7–9 use the explicitly identified v3 continuation.
Comparison reports remain host-only, outside every optimizer's isolated workspace.
