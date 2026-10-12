# Life AgentBench H2–H5 interface v3

This is a **policy-free scaffold**, not a released harness or a new experiment.
All domain rules, schema rendering, parsing, repairs, guards, monitors, prompts,
skill content and retrieval strategies must be produced by iteration. The four
files in `templates/` are identity/empty hooks: they insert no content, block no
action and perform no domain-specific transformation. The original six-round
artifacts and later nine-round records are immutable and remain reusable as
historical evidence. A new interface does not silently upgrade their results.

## Primary hooks and ownership

Each artifact declares `TARGET_LAYER = 'h2' | 'h3' | 'h4' | 'h5'` and one class
`Harness` with exactly ONE corresponding primary hook. Bundle-v2 add/modify/delete,
revision/CAS validation and independent sibling filtering remain the outer-loop
protocol; this module does not replace that protocol. Valid sibling operations
must be staged together. Safe synchronous helper methods are allowed.

| Hook | Signature | Return / responsibility |
| --- | --- | --- |
| H2 | `h2(self, message, history)` | Assistant dictionary; may repair/replace action or veto through host control |
| H3 | `h3(self, tools)` | Tool list with descriptions changed only; names, schemas and order unchanged |
| H4 | `h4(self, messages, remaining)` | `[]` or `[str]`; monitor public trajectory, update shared state, give guidance |
| H5 | `h5(self, messages)` | `[]` or `[str]`; cold-start skills/instructions, called on first solver turn |

H4/H5 have **no word-count limit**, including at the actual Task/session boundary.
Multiple mechanisms' strings are joined in manifest order into one host hint.
Hints remain subject to the solver's normal context/token limit, not a hidden
120-word gate. Hint contents are NOT provided by the scaffold.

## Context and state: available to EVERY layer

Optional `bind_shared_state(self, state)` receives the SAME mutable dictionary in
all artifacts in one episode, before context binding. Store it as `self.state`.
It starts empty and is newly allocated for each episode. Each artifact instance
also persists for the whole episode and may keep its own counters/caches.
Choose shared key names and value formats consistently across your bundle.
An H4 mechanism can record a proposed action; an H2 mechanism can read it and
construct the action. The host does NOT prescribe a queue schema or execute
anything merely because a key appears in the dictionary.

Optional `bind_task_context(self, context)` then receives an independent deep
copy of the public task context (or `None`), before H3 tool conditioning:

- DBBench: `description`, `type`, `table`, and optional `evidence`,
  `add_description`, exactly as provided by the Task. `table` is the Task's public
  table structure, not a guessed text schema. These fields allow a candidate
  to derive task intent, exact column names, sample rows and a schema card.
- WebShop: `{"instruction": PUBLIC_INSTRUCTION}`. Pages, selectable options and
  tool results are obtained from public message history, not private goal data.
- ALFWorld: the current Task binds `None`; read the initial instruction and
  public observations/admissible actions from message history. An explicitly
  supplied public `instruction` field is also supported.

No oracle SQL, labels, answer hashes, hidden goals, reward state or test results
may be supplied. Unknown context fields are rejected. No filesystem, network,
environment, evaluator introspection or released-policy imports are permitted.

## Layer-specific optional extensions

H3 may implement `h3_message(self, message) -> dict`. It runs on public messages
injected through the session and may change **content only**. Role and other
metadata must be preserved. This permits evolving system-message conditioning
and task/schema cards, without changing the tool schema. Track once-only
conditioning in the instance if needed; the host does not invent a schema card.

H2 may implement:

- `h2_no_tool_message(self, message, remaining) -> str | None`: when the repaired
  assistant has no tool call, return a public nudge to suppress this empty action
  and continue, or `None` to leave it alone.
- `drain_h2_messages(self) -> list[str | dict]`: return public user messages to
  append after a normal H2 action. Dictionary messages require `role='user'` and
  string `content`. The candidate owns queue storage and clearing.
- Veto through its primary return value: an assistant dictionary with
  `_harness_control = {'action': 'suppress_tool', 'messages': [...]}`. The Task
  executes no tool that turn, preserves the raw assistant message, injects the
  supplied public user/tool messages and consumes a turn. Tool feedback must
  refer to actual call IDs. Later H2 mechanisms cannot erase an earlier veto.

H2 may otherwise replace the assistant's `tool_calls` to implement a recovery
action. A shared-state request from H4 needs an evolved H2 consumer; there is
no prefilled forced-action policy. Task H2-prevalidation prevents a second
legacy fuzzy-matching pass over the returned action.

## Lifecycle / host options

`new instances + empty shared state -> bind state -> bind public context -> H3
tools -> inject initial messages through H3-message -> H5 -> solver -> H2 ->
environment -> H4 -> solver -> H2 -> ...`

H4 is called before solver turns 2+, with all public history and remaining
action turns INCLUDING the impending turn. Environment tool results are in
history; observe new results using a candidate-owned cursor. H2 receives public
history captured before that turn's H4 hint is injected, as in the native
released bridge; shared state is the reliable same-turn H4-to-H2 channel.

Default host options match released persistence capabilities:
`h4_persistent=True`, `h2_preserve_raw_history=True`, `h4_message_prefix=''`,
`h5_message_prefix='\n\nProcedural guidance:\n'`. H4 hints persist in public
history; H5 guidance appears in solver payloads without rewriting base history.
The evaluator may explicitly choose other options, but must record them in its
fingerprint. All four switches remain available for ablation. Persistence and
transport options are infrastructure, not evolved policy content.

## Integration and provenance (new runs only)

From the repository root, import `evolution.agentbench_hooks`. In EACH evaluator
worker, after importing the selected Task module and before constructing Task,
call `receipt = install(task_module, domain, chain_manifest, options=None)`.
This installs the complete composite and an uncapped subclass of that Task's
FourHookSession, locally in the Task module. It does not modify the source of
the published bridge or any frozen historical snapshot.

Include the full returned receipt in the new run/evaluation/cache fingerprint:
interface version, all interface/template source hashes, base-session hash,
options, domain and manifest hash. Never reuse an old score merely because the
artifact hash is unchanged: the interface changes evaluation semantics.

The next proposer workspace MUST include this entire file as
`source_view/contract.txt` (alongside the bundle schema), all four empty templates,
the safe-import allowlist in `contract.py`, the actual base session source and
public Task-context construction. Use this module's `load_artifact` validator,
not the frozen experiment's old validator. The old round driver/evaluator are
NOT switched automatically. No benchmark or paid model call is launched here.
