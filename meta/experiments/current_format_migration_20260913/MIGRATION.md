# Seven-benchmark current-format migration

Completed on 2026-09-14. The release paths for all seven benchmarks now expose
the current iteration boundary while retaining each benchmark's evaluated
policy.

| Benchmark | Current execution form | Loss-preservation evidence |
| --- | --- | --- |
| TauBench airline | H2/H3/H4 registries, H5 skill registry, `register()` plugin | all 8 H2/H3/H4 switch combinations select the expected class; no-op plugin preserves rule, annotator and skill state |
| TauBench retail | same | same |
| TauBench telecom | same | same |
| AgentBench ALFWorld | `Harness.h2/h3/h4/h5` through `FourHookSession` | frozen 109-task replay remains 104/109; outcome, status, turns and normalized model inputs match |
| AgentBench DBBench | same, with host-bound public task context | frozen 300-task replay remains 190/300; outcome, status, turns, normalized model inputs and tool inputs match |
| AgentBench WebShop | same | public requirement parsing and H5 output match the released Runtime on six representative task forms |
| AgentBench OS Interaction | same | public task parsing and H5 output match the released Runtime on all 1,000 local training tasks |

ALFWorld has 21 raw request differences in one failed episode because the old
Task emitted recovery and step guidance as two adjacent user messages. The
current hook contract permits one H4 hint, so the bridge joins the same two
texts with a newline. Text and ordering are unchanged. DBBench has four raw
request differences caused only by its random per-session database name; after
normalizing that ephemeral identifier, the request histories match.

The migration fixed three execution defects found by replay:

- ALFWorld no longer runs a legacy fuzzy action matcher after H2 has already
  validated or repaired the action. That second pass could silently change a
  deliberately allowed invalid action into a different environment action.
- ALFWorld's container-loop recovery no longer selects a target through set
  iteration. It deterministically chooses the most frequent recent container,
  breaking ties by recency.
- DBBench retains the original assistant/tool history around H2 repairs and
  blocks, while executing the repaired SQL privately. Public task type, table,
  evidence and additional description are bound before H3/H5; oracle SQL,
  labels and expected answers are rejected.
- The local `dbbench-std` profile had still disabled H2/H4 and the harness as a
  whole after the prior release. The canonical and native standard profiles now
  enable all four hooks; the restarted worker reports `harness_enabled=True`.

`agentbench_evaluate.py`, preflight and the bounded-evidence loop use the same
message prefixes, persistent-H4 mode, H2 raw-history mode, blocked-tool control,
no-tool recovery and public DB context as the Tasks. New ALFWorld/DBBench runs
must use [`current_agentbench_source_20260914`](../current_agentbench_source_20260914/SOURCE.md).
Its frozen Tasks are the migrated Tasks; old source snapshots remain historical
inputs and cannot be silently resumed because the implementation hashes differ.

Validation artifacts:

- [`verification.json`](verification.json): seven-domain contract/context report and source hashes.
- [`tau_verification.json`](tau_verification.json): 24 TauBench switch selections, H5 counts and no-op plugin state check.
- [`replay_alfworld.json`](replay_alfworld.json): complete 109-task frozen-response replay.
- [`replay_dbbench.json`](replay_dbbench.json): complete 300-task frozen-response replay.
- [`http_smoke.json`](http_smoke.json): restarted native workers exercised through the controller with real tool calls.

WebShop cannot currently run a score replay because the local checkout lacks its
usable product index. OS Interaction cannot safely run its benchmark process
isolation on this Docker-free VM. Their migration claim is therefore limited to
hook contract, public context and H5 equivalence; no new score is attributed to
them. TauBench code already used the current registry form, so its release rules
were verified rather than rewritten.
