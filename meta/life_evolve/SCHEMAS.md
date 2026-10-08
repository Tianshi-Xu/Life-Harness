# Candidate file contracts

This reference specifies file formats, not domain strategies. Read before writing
candidate outputs. Runtime boundaries are in source_view/INTERFACE.md and contract.py.

Only bundle.json and its referenced artifacts/*.py are required. diagnosis.json
and checks.json are optional archival notes, with no schema or write-order gate.
They are copied unchanged if present, never executed and never used to select
replay cases or determine candidate validity.

Inspect relevant cases and successful counterexamples on demand; no per-task
diagnosis is required. Each detailed case
has a deduplicated message_pool, history_indices and call.message_indices;
resolve history_indices into message_pool to recover the original history.
Signals are observations, not root causes. Keep uncertain hypotheses explicit.
Shared state is episode-local and reset by the runtime. Coordinate writers and
readers in code; no ownership table or explanations for unchanged layers are required.
Choose coherent contracts: public goal → progress/state → next action → feedback.
When a rule affects several layers, modify them together or explicitly explain
why a layer remains unchanged. Do not turn uncertain heuristics into unconditional
action overrides. Counterexamples must exercise the non-trigger/ambiguity path.
Consistency declarations alone do not prove semantic correctness.

Write bundle.json using bundle-v2 operations:
{"schema_version":2,"hypothesis":"nonempty",
"operations":[{"operation_id":"identifier","mechanism_id":"identifier",
"op":"add","target_layer":"h2|h3|h4|h5","hypothesis":"nonempty",
"artifact":"artifacts/name.py"}]}
The host supplies domain, base_snapshot_sha256 and a bundle_id when missing or
invalid; do not copy hashes by hand. Previously supplied bundle IDs remain valid
when matching `[a-z][a-z0-9_]{0,63}`. Operation/mechanism identifiers use that same
pattern. Host binding uses the verified frozen workspace, never model-supplied
domain or snapshot values.
modify has the add fields plus expected_revision and expected_sha256 from inventory.
delete omits artifact and has expected_revision and expected_sha256. There is no
one-operation limit: valid add/modify/delete siblings are staged simultaneously.
Each artifact has TARGET_LAYER and class Harness with one matching primary hook.
Use ONLY the safe imports in source_view/contract.py. All rules/skills must be
evolved; the interface supplies no domain policy or published solution.

check_candidate validates frozen inputs, filters unsafe/invalid operations,
and runs isolated H2–H5 replay against the available public TRAIN traces. No
trigger/control/cross_layer test count, category or linkage is required. Passing
replay means interface safety, not environment success; acceptance still requires
strict improvement on the same complete TRAIN batch.
