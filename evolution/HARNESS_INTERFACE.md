# Shared Life H2--H5 interface

This is a policy-free contract for new AgentBench, TauBench and registered
custom benchmark runs. The [extension API](extensions/README.md) registers
custom public-context schemas and connects native episodes to these hooks. The
four empty templates and static validator are reused from
`evolution/agentbench_hooks_v31`; historical interfaces, runs and scores are unchanged.
No published Life/Meta/AHE rules, plans, checklists or skills are supplied.

| Hook | Signature | Capability |
| --- | --- | --- |
| H2 | `h2(self, message, history)` | Repair/replace the pending assistant action, or veto it before execution |
| H3 | `h3(self, tools)` | Change tool descriptions only; names, schemas, order and dispatch stay fixed |
| H4 | `h4(self, messages, remaining)` | Observe public history, update episode-local state and return next-step guidance |
| H5 | `h5(self, messages)` | Supply evolved cold-start skills/instructions on the first solver turn |

Each artifact declares one `TARGET_LAYER` and one `Harness` class with that
primary hook. Use the existing bundle-v2 add/modify/delete and CAS contract.
H4/H5 return `[]` or `[str]`, without a separate word-count limit. Normal solver
context/output/turn budgets remain enforced. No new model, tool or hidden action.

Every layer may implement `bind_shared_state(self, state)` and
`bind_task_context(self, context)`. State starts empty, is shared across the four
layers, persists within ONE episode and is never shared across episodes. State
keys, progress tracking and clearing policies are evolved content, not host rules.
History includes public user messages, assistant actions, actual tool results
and clearly identified harness feedback. No evaluator state, hidden answers,
task fault labels, test IDs, user-simulator instructions or credentials are exposed.

## Read-only environment state

Any H2--H5 artifact may implement `bind_environment_state(self, state)`. The
host supplies `None` when unavailable, otherwise `{"db": BUSINESS_DB_JSON}`.
This is a separate capability from public task context and shared progress state.
It includes current business records even if the solver has not queried them.
The TauBench adapter reads ONLY `environment.tools.db`, not `user_tools.db`, task
objects, scoring criteria, target/reference states or simulator instructions.

The host refreshes at initial binding, incoming-message conditioning, before each
solver call (before H4/H5/H3), and again before H2 action control. Each artifact
receives its own detached, read-only JSON view, never a native DB object, connection,
tool handle or setter. Ordinary dict/list writes raise TypeError; there is no
write-back and one layer cannot modify another layer's binding. Unchanged frozen
views are reused; binders are called when state changes, not as a turn counter.
All actual DB changes still use existing environment tools. Binders should replace
their snapshot and handle `None`; cross-layer progress
belongs in `bind_shared_state`, not in this detached DB copy.
Views support normal dict/list reads and JSON serialization. To construct separate
mutable derived data, select/copy the needed fields, not the environment binding.
Immutable unchanged branches are reused internally; replay retains the existing
CPU/memory limits rather than repeatedly cloning the whole DB on each hook.

The full DB is never automatically included in solver prompts. The evolved
artifact decides which checks/derived annotations to emit. TRAIN traces record
initial, model-boundary and action-boundary state references, deduplicating equal
snapshots and storing changed leaves rather than a full DB per call. Offline
replay uses these exact recorded points; it does not replay proposed writes or
infer their success. Old records without DB state yield `None`, not invented
facts, and are not silently rerun.

Evolve's `read_case` leaves DB contents out by default. To inspect approved TRAIN
facts explicitly, use `call_index`, `environment_state="model" | "action"` and
optionally a JSON `state_path`, e.g. `["db", "users", "U1"]`. These are ordinary
audited evidence reads; no auxiliary model/tool call or new quota is introduced.
Declarative tests can supply `environment_state` at the test level and override
it on a hook step to exercise a business-state transition.

H3 may implement `h3_message(self, message)` at message injection and
`h3_messages(self, messages)` immediately before EVERY solver call (after H4/H5).
The latter can read shared state to update the current model-visible context.
Only content may change; message count/order, roles, tool calls and metadata must
remain unchanged. These payload changes never rewrite actual episode history.
H2 may implement
`h2_no_tool_message(self, message, remaining)` returning a corrective string or
`None`, and `drain_h2_messages(self)` returning public user messages. Returning
`_harness_control = {'action': 'suppress_tool', 'messages': [...]}` from H2 vetoes
the pending action. Feedback must be nonempty; tool feedback must match proposed
call IDs. Missing suppressed-call responses receive neutral host feedback stating
that no action executed. Later H2 mechanisms cannot erase an earlier veto.

Lifecycle: fresh instances/state -> public context -> H3 tool/message conditioning
-> H5 -> solver -> H2 -> environment -> H4 -> solver -> H2 -> ... . A veto or
no-tool recovery executes no environment action and consumes a normal turn/step.
It does not reach the user simulator. Empty hooks change neither requests nor
environment actions. Raw and repaired actions are retained in the hook trace;
by default model history contains the action actually dispatched, so subsequent
tool results have matching call IDs. H4 guidance persists by default and is marked
`[H4]`; H5 is payload-only. All persistence options remain explicit and fingerprinted.

Adapters normalize messages/actions, bind ONLY native public context and connect
the hooks at these boundaries. They do not implement domain strategies:

- AgentBench: `evolution.harness_hooks.install_agentbench(...)`. Public context remains
  the existing DB description/type/table/evidence, WebShop instruction or ALFWorld
  instruction. SQL, clicks and environment actions retain native semantics.
  OS uses domain `os` and binds only `{"description": PUBLIC_TASK_DESCRIPTION}`.
  Its native `bash_action`, `finish_action` and `answer_action` schemas, Docker
  execution, 16-turn budget and grading remain unchanged. No filesystem snapshot,
  container handle, init/check script, answer/match or evaluator state is bound.
  A host adapter may supply `environment_state_provider` to bind approved business
  state through the SAME interface; no arbitrary environment object is inferred.
- TauBench airline/retail/telecom: `python -m evolution.tau_hooks --life-manifest PATH`
  followed by native evaluation CLI flags. Only the native agent-visible policy is
  bound as `policy`; user requests arrive through public history. No task
  metadata/evaluator is passed to hooks. Current agent-side business DB is bound
  separately through `bind_environment_state`.
  H2 recovery is charged to the existing orchestrator step budget, not an unbounded
  inner LLM loop. Agent-local state does not alter environment replay/set_state.

New runs must record interface, adapter, base-session and artifact hashes before
evaluating. Never reuse an old score after switching interface. Legacy CLI/plugin
paths remain unchanged when the new interface is not explicitly selected.
TauBench's opt-in summary records canonical binary `accuracy`, `n`, `successes`
and `infra_errors`, retaining infrastructure failures as zeros in the denominator.
Native legacy reward aggregates are retained separately, not used as this accuracy.

## Released Life capability audit (2026-10-06)

Audit targets are the released `AgentBench/src/server/harness/{alfworld,dbbench,webshop}.py`,
`TauBench/src/tau2/harness/{base,airline,retail,telecom,skills}.py` and native
agent/environment integration. This table describes mechanism capacity, not
supplied strategies or a measured accuracy result.

| Released mechanism | Shared interface mapping | Status |
| --- | --- | --- |
| Validate eligibility, ownership and argument consistency before a write | H2 checks/repairs/vetoes using current business DB and public history | Expressible with direct business-state access |
| Embed policy in tool descriptions and model-visible context | H3 description conditioning, initial `h3_message`, per-call `h3_messages` | Expressible without changing schemas |
| Tool-result status summaries, remaining work and recovery guidance | H4 current business DB/history plus guidance/shared state, consumed by H2/H3 | Expressible with direct business-state access |
| Retrieve procedural skills from user intent and task context | H5 evolved in-file skill content and standard-library ranking/routing | Expressible, no extra model/tool |
| Couple progress tracking, next-action control and completion | Episode-local shared state across H2--H5; native step budget and isolated reset | Expressible |
| ALFWorld admissible-action repair and budget-aware recovery | Public observations/admissible actions, H4 state and remaining turns, H2 action replacement | Expressible |
| DBBench schema cards, SQL/answer formatting, evidence-based recovery | Public description/type/table/evidence, H3 context, H4 result tracking, H2 repair/veto/side messages | Expressible |
| WebShop requirement/page/option tracking, purchase guards and search recovery | Public instruction/pages/tool calls, H4 state and H2 action control | Expressible |

The earlier business-DB capability gap is now closed for TauBench's three domains:
like released H2 rules/H4 annotators, evolved artifacts can inspect agent-side
business records without extra solver queries. Airline reservation-ID expansion
and comparable state predicates are therefore expressible. This does not expose
the evaluator, reference/target states or Telecom's separate user-device DB.
It supplies data access, NOT any published validation/annotation policy.

H4 guidance is a separate clearly marked message, not an overwrite of an actual
tool result. Dynamic H3 is a model-only content view. These provide the relevant
conditioning capacity but do not claim byte-identical released prompt construction.
The released Life reference scores (Airline 60%, Retail 63%, Telecom 57.5%, supplied
by the user) remain unverified performance targets for a fresh end-to-end experiment.
AgentBench's released adapters consume the same public context/history categories
available here. Their domain algorithms are not copied, and reproducing them may
require coordinating multiple evolved artifacts, not changing the hook API.
