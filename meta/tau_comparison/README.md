# Meta / AHE native Tau comparison

Six optimization rounds on airline, retail and telecom, followed by one final
fixed test per method/domain. Life is not launched by this controller.

Meta runs the pinned original `run_evolve` and official skill through the existing
audited API coding shell (parent, analysis Agent, implementation Agent). AHE runs
the pinned original `run_single_experiment`, Debugger/Evolve and best-ever selection;
the previously agreed restricted capability profile and ten detailed analysis
tasks remain in force. Neither official optimizer source is edited.

Airline has 30 native training tasks and two trials; retail/telecom use 50 tasks
and one trial. Schedules/IDs are reused from the frozen historical manifest and
checked against native train/test splits. These small pools necessarily overlap
across rounds. Meta changes batches after non-improvement; AHE retains its original
evaluate-before-evolve and best-ever logic. Do not present different-batch scores
as paired improvements. AHE step 6 receives an extra final TRAIN measurement,
not fed into optimizer or original selection. Tests wait for all six completed
optimizations and the training seal and never enter optimizer workspaces.

Airline/retail user simulator and retail NL judge use `qwen3.8-flash` at the user's
Aliyun gateway with thinking enabled. Telecom uses MR `deepseek-v4-flash`.
All optimization, Debugger and telecom simulator requests share one host MR gate,
concurrency <=8 and global pacing. Aliyun/local Qwen are outside this gate. Native
task concurrency is 50 per domain/method. A shared local request admission proxy
enforces solver total context 32768 and output 4096; no such cap is applied to
DeepSeek optimization. The AHE internal soft context value remains its original
configuration; the shared solver admission boundary enforces the effective cap.
Telecom's DeepSeek user simulator has output 100000, independently of the solver.
Its previous adapter value 4096 produced empty user replies when reasoning
consumed the full output budget. The diagnosed first-round run is preserved,
including the matched API receipts and affected episodes; it is not a sealed
six-round result. Airline/retail simulator settings are unchanged.

Real credentials are loaded from environment files into the host broker only.
Worker configurations use loopback URLs and `EMPTY`. Detailed native scoring
artifacts remain host-side; optimizer evidence exports scalar TRAIN reward and
public agent dialogue/tools, never hidden goals/answers or full judge reward info.
Both providers have durable per-attempt ledgers. Optimization totals contain only
Meta parent/analysis/implementation or AHE Debugger/Evolve (including nested
compaction and retries), excluding solver/user simulator/NL judge costs. Missing
usage is reported unknown, not replaced by an estimate or zero.
Explicit HTTP 429 rejections are separately counted and retain their missing-usage
markers. As in the Meta ledger, these rejections do not fail the inference-usage
gate; a successful/ambiguous optimization call without usage still stops it.

Integration guards: six one-task native setup checks, then all first candidates
must be valid, measured and have complete optimizer accounting before optimization
steps 2-6. For AHE, candidate 1 is measured by native evaluation 2; its Debugger/
Evolve 2 wait at the gate. Native setup failures do not spend optimization tokens.
No blind replay of a paid session is supported. The same runner accepts
`--resume-from ANCESTOR`: it waits for the ancestor to exit, audits the verified
pre-API UID failure, reuses its exact Meta baseline batch and original AHE
resume/skip-eval path. Completed telecom episodes are hash-checked and reused;
only missing episodes rerun. Ancestor artifacts and all raw expenses remain intact.
Tau workers use the native evaluator's six-hour wall timeout, not a solver budget.
The same resume path also audits the recorded first-round simulator-budget incident.
Paid, hash-matched Meta proposals are reused without API replay; measured
airline/retail rounds are retained. Telecom Meta remeasures baseline and candidate
under the repaired simulator. AHE reuses valid baseline episodes and remeasures
only the eight diagnosed errors; original metadata/scoring functions then correct
the baseline before the original iteration-2 resume. Its first paid optimization
was informed by pre-repair TRAIN and is labeled as such, not regenerated.
Upstream resume also materializes Evolve connection values in YAML. The isolation
boundary rebinds only that private copy's key/URL to the scoped host broker.
Telecom's first resumed Evolve had 140 unscoped calls: its original tracer's
generation count, exact token total and time interval match all 140 API receipts.
Their scope is corrected only in optimizer aggregation, with the matching trace
hash/request IDs archived in the continuation receipt; raw ledgers are unchanged.
An incomplete/ambiguous match or any later unscoped call stops aggregation.
Later transport continuations retain all committed Meta rounds, including failed
proposals with real usage and no measured candidate. Uncommitted paid rounds stop
recovery. Complete iteration-2 Debugger reports are hash-checked and reused by
AHE's existing overview cache, without repeating its paid analysis. Repaired
historical jobs expose only aggregate evidence; the current ten-task cap is unchanged.

The public pure-stdlib interface includes `datetime` in both integrations. An
explicitly approved continuation may add `--repair-date-imports` to reevaluate
audited, date-only rejected Meta candidates without changing their bytes or paying
for optimization again. It uses the original `run_evolve`; only a changed strict
acceptance archives and regenerates later optimization branches. Original request
ledgers and superseded branch expenses remain included, never subtracted. Other
private/runtime restrictions are unchanged. This flag can also restart the dead,
zero-API `stopped_pending_import_contract_choice` launcher; old status/launch
metadata are archived, and a started/paid destination still cannot be overwritten.

The public accessor interface also includes AgentState context get/set and
ParsedResponse/ToolCall fields (`tool_calls`, `parameters`, `tool_call_id`).
Completed original Evolve 2 results rejected only by that accessor omission can
be reused after complete usage, original input, candidate and file-hash audits.
The isolation wrapper performs its normal validation/promotion and returns the
original result to the unchanged AHE loop; Debugger/Evolve 2 are not paid again.
If Evolve 2 was already promoted, its complete evaluation 3 is reused when present
and the original loop resumes at 3. Partial/ambiguous paid artifacts fail closed.

Offline checks: use the AHE virtualenv with the Life repository on `PYTHONPATH`:
`python -m unittest meta.tests.test_tau_recovery meta.tau_comparison.test_transport -q`.

Launch a new isolated run with `python -m meta.tau_comparison.run --name NAME --launch`.
Use `--smoke-only` for setup validation only. State, full provider ledgers,
`OPTIMIZER_TOKEN_USAGE.json`, training seal and final results are saved in
`meta/experiments/NAME`; the launcher refuses to overwrite historical work.
