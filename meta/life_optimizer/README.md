# Life on-demand diagnosis + replay optimizer

Opt-in profile: `life-h2345-diagnose-replay-v2.4`. This absorbs AHE's concrete failure
diagnosis, coordinated component edits and local behavioral checks, **not** its
broader runtime capabilities or its per-task paid Debugger workflow.

Nothing switches an old driver, rewrites a sealed experiment, fills empty H2–H5
templates with published policy, or starts an experiment automatically. The
existing shared v3 interface and bundle-v2 add/modify/delete contract stay intact.

## Budget and gates

DeepSeek diagnosis/evolve uses the user-requested `max_tokens=60000`
per request, including provider reasoning. There is no locally imposed 32768 total-context cap, reserved-output
guard, or 4096-token output cap on this optimizer. Actual ModelRouter/provider
limits still apply. The optional token-accounting helpers are not on the default
paid completion path and do not reject a request based on the solver's budget.
Qwen's task-execution context/output settings are separate and unchanged.

v2.3 removed the mistakenly applied solver limits; v2.4 changes only Flash's
per-request output budget from 8192 to the explicitly requested 60000.
Evidence access, diagnosis, repair feedback, checks, acceptance and
request/concurrency budgets are unchanged.
The frozen v2.2 experiment retains its original configuration, receipts and results
as historical evidence, with its optimizer sources byte-preserved in
`../experiments/life_on_demand_six_db_web_ctx32768_out4096_20261003/optimizer_source_snapshot/`.
Do not rewrite or resume that sealed experiment with this changed profile: source
fingerprints intentionally require a fresh workspace. This restoration starts no
model call or experiment. The completed 8192-output one-round run is also preserved,
including its v2.3 optimizer sources under
`../experiments/life_on_demand_one_db_web_flash_restored_20261003/optimizer_source_snapshot/`.

- All current TRAIN cases are available, with **no case-count or total-character
  limit**. `evidence.json` is a metadata index; whole detailed cases live in
  `evidence/case_ID.json` and are read on demand. Existing frequent/rare failure
  representatives, matched successes and spot checks determine only index order,
  never availability. The optimizer chooses which cases need inspection/diagnosis;
  it is not forced to analyze every task. Infra cases remain accessible with an
  explicit annotation; their outcomes stay zero in the full score denominator.
- Earlier scores, accepted/rejected decisions and operation hypotheses are retained;
  earlier raw trajectories and hidden labels are not exposed. Detailed evidence
  includes public request tools/messages and raw assistant outputs, not provider
  config, private task results, rewards or credentials. Deduplicated message indices
  avoid quadratic repeated request-history payloads. An audit ledger records which
  detailed files/ranges were opened; it does not reject later reads or impose a quota.
- One `deepseek-v4-flash` agent, at most 32 requests including diagnosis, code edits
  and corrective feedback. No separate Debugger calls per case. The existing agent
  turn budget and per-read chunk size remain; there is no aggregate evidence quota.
  These limits do **not** cap money or all context tokens. Local checks themselves have
  no model calls. Future experiment orchestration must preserve the global Flash
  concurrency cap of 8; the CLI issues synchronous requests, one at a time.
- A mandatory diagnosis sidecar records observed first error (or uncertainty),
  public trigger, causal hypothesis, confidence, change mapping and regression risk
  only for cited cases, not every available training task. Replay follows these
  cited cases; every surviving operation still needs cited evidence and tested rules.
- Every rule declares all four layer responsibilities, including unchanged reasons,
  plus state ownership/type/reset when used. Trigger/non-trigger fixtures are
  mandatory; changes spanning layers also need sequential cross-layer fixtures.
  This tests declared behavior; it does not automatically prove that prose is
  semantically consistent or that the diagnosis is causally correct.
- The only execution tool is `check_candidate`: static bundle/CAS validation,
  independent invalid-sibling filtering, frozen-input replay and declarative fixtures.
  Candidate code never executes in the controller. A disposable Python worker gets
  a scrubbed environment, CPU/memory/file-descriptor limits, and an audit guard
  denying file, network and process access. There is no Bash or arbitrary test code.
  This is defense-in-depth, **not an OS/container security boundary**. Only use it
  for the authorized generated harness; do not treat it as a hostile-code sandbox.
- Replay uses recorded model inputs/outputs (including incumbent hints), does not
  execute rewritten actions, and cannot produce replacement environment feedback.
  Thus it is a hook-level preflight, not a complete bridge replay or success estimate.
- `strict_improvement` permits acceptance only on the **same complete training task
  keys**, strictly higher binary score, same domain, no hidden test selection.
  Offline pass alone never accepts, promotes, or changes the incumbent.

## Use (from repository root)

Prepare from an existing incumbent state and a matching complete TRAIN evaluation.
Use a **new directory outside the old experiment**:

```bash
python -m meta.life_optimizer.profile prepare \
  --seed-run /path/to/life/run/dbbench \
  --eval-dir /path/to/matching/incumbent/train/evaluation \
  --output /path/to/new/run/optimizer_inputs/round_01
```

The evidence must match the incumbent's exact artifact content hashes. DBBench
loads only allowlisted public context fields for these training task IDs from the
local dataset (`--db-source` overrides its location); no oracle SQL/labels are copied.
No model call is made by `prepare` or `check`.

An explicit paid optimization request uses `MR_API_KEY` from process environment
only; optional `MR_HOST` sets the router. Never put keys in files, arguments, fixtures
or outputs:

```bash
python -m meta.life_optimizer.profile propose \
  --workspace /path/to/new/run/optimizer_inputs/round_01 \
  --output /path/to/new/run/candidates/round_01

python -m meta.life_optimizer.profile check \
  --workspace /path/to/new/run/optimizer_inputs/round_01 \
  --candidate /path/to/new/run/candidates/round_01
```

The candidate directory contains a normal bundle-v2 `bundle.json`, surviving
`artifacts/*.py`, `diagnosis.json`, `checks.json`, and `validation.json` (usage,
request count, files/ranges read, preflight result, rejected siblings and immutable input/output hashes).
Validated proposals are cached: repeating the exact call performs no LLM request;
tampered inputs or outputs fail closed.

For a **new** outer-loop driver, use `prepare(...)`, `propose(...)`, stage the bundle
with the existing bundle-v2 compositor, evaluate candidate/incumbent on matching
50-task training batches, then call
`strict_improvement(candidate, incumbent, expected_task_ids=predeclared_batch_ids)`
before committing. The gate requires all 50 predeclared keys, not just matching
candidate/incumbent screen subsets.
Bind both `receipt.json` and `validation.json` hashes into the new run's fingerprint
and audit trail. This profile does not monkeypatch old drivers: their reconstruction
and cache formats stay frozen. Task counts/batches, solver settings, global router
limits, episode isolation and accepted snapshot history remain outer-loop concerns.

## Offline regression

```bash
python -m unittest discover -s meta/tests -p test_life_optimizer.py
```

Test fixtures are synthetic policies for testing machinery, not optimizer priors
or manually inserted benchmark strategy. No tests invoke ModelRouter or a solver.

## Original v1 verification (2026-10-03, preserved)

- Shared `meta/tests` regression: 54 tests passed (21 new optimizer tests, including
  mocked proposer completion, CAS add/modify/delete, invalid-sibling filtering,
  state/reset, native schema preservation, budgets and worker defenses).
- Sealed v3 twelve-round continuation regression: 8 tests passed.
- Real TRAIN frozen-input smoke, reusing unchanged incumbent code: ALFWorld
  7 cases / 102 recorded calls; DBBench 10 / 51; WebShop 10 / 38; all passed.
  ALFWorld has seven whole cases because of the character cap. These are **not**
  new optimization rounds, fresh rollouts or measured improvements. No optimizer
  or solver request was made. Receipts:
  [`alfworld`](../experiments/life_optimizer_absorption_offline_20261003/alfworld/verification.json),
  [`dbbench`](../experiments/life_optimizer_absorption_offline_20261003/dbbench/verification.json),
  [`webshop`](../experiments/life_optimizer_absorption_offline_20261003/webshop/verification.json).
- Before/after hashes of 57 historical driver/state/seal files were identical.
  Published policies, historical accepted artifacts and the official AHE checkout
  were not edited. Only the new optimizer, its tests/docs and isolated offline
  verification artifacts were added; no paid run was started.

The user subsequently removed the Life evidence quota: v2 makes every current
training case readable on demand and requires diagnosis only of cited cases.
The earlier v1 receipts above remain unchanged and do not attest to the v2 profile.
Use a new v2 workspace; changing profile/source hashes invalidates old caches.

## On-demand v2 verification (2026-10-03)

v2.1 retains the same unrestricted on-demand TRAIN evidence and hook contracts.
It preserves actual provider `reasoning_content` on tool continuations, supports
host-only public session checkpoints for completed-request resume, and stops on
output truncation instead of spending more requests on blind preflight feedback.
An interrupted in-flight API request has an unknown outcome and is not silently
reissued. Historical v2 receipts remain immutable; they do not attest to v2.1.

- Shared regression: 55 tests passed; the sealed continuation's 8 tests passed.
  The 22 optimizer tests include reading more than ten distinct cases, reading an
  entire case larger than the old 160,000-character cap, and diagnosing/replaying
  just two cited cases while all other cases remain accessible.
- Every domain exposes all 50 current training cases, without truncation or an
  aggregate quota. Byte-identical incumbent smoke passed on ALFWorld's 535,
  DBBench's 244 and WebShop's 182 recorded calls (961 total). This offline smoke
  explicitly chose all cases; the optimizer is not required to do so. No new model
  call, training episode, optimization round, acceptance or score was produced.
- New receipts are separate from v1:
  [`alfworld`](../experiments/life_optimizer_on_demand_offline_20261003/alfworld/verification.json),
  [`dbbench`](../experiments/life_optimizer_on_demand_offline_20261003/dbbench/verification.json),
  [`webshop`](../experiments/life_optimizer_on_demand_offline_20261003/webshop/verification.json).
- The 57 historical driver/state/seal files again have identical before/after
  hashes; old v1 verification outputs and AHE's separate budget profile were not
  overwritten. Only Life's opt-in profile and its documentation/tests changed.
