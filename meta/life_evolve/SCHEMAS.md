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
invalid; do not generate or copy the base snapshot hash. Previously supplied
bundle IDs remain valid
when matching `[a-z][a-z0-9_]{0,63}`. Operation/mechanism identifiers use that same
pattern. Host binding uses the verified frozen workspace, never model-supplied
domain or snapshot values.
modify has the add fields; delete omits artifact. The host fills omitted
expected_revision and expected_sha256 from that mechanism's frozen inventory
record. Supplied values must match it exactly; stale values are still rejected.
Unknown mechanisms are not inferred or created by this binding. There is no
one-operation limit: valid add/modify/delete siblings are staged simultaneously.
Each artifact has TARGET_LAYER and class Harness with one matching primary hook.
Use ONLY the safe imports in source_view/contract.py. All rules/skills must be
evolved; the interface supplies no domain policy or published solution.

check_candidate validates frozen inputs, filters unsafe/invalid operations,
and runs isolated H2–H5 replay against the available public TRAIN traces.
rejection detail is retained when all operations fail: rejected_operations
identifies each operation, artifact and exact format or source validation error. No
trigger/control/cross_layer test count, category or linkage is required. Passing
replay means interface safety, not environment success; acceptance still requires
strict improvement on the same complete TRAIN batch.

For inexpensive local diagnosis, search_file returns literal-match UTF-8 byte
offsets and context in pages; replace_file edits one exact unique span in candidate
outputs only. These tools do not change candidate checks or impose a total read
quota. preview_candidate(case_id, call_index, chain=...) uses the same sandbox to
replay one TRAIN prefix and expose its guidance, model-view changes, actions and
feedback. chain='incumbent' inspects retained behavior before coding or provides
a control. Omit chain to inspect the candidate when bundle.json exists, otherwise
the incumbent. Explicit chain='candidate' requires the working bundle/artifacts;
an invalid candidate is never silently replaced with the incumbent.
It cannot qualify a candidate or select replay cases for full check_candidate.
history.json may reference hash-verified historical_code_path files, including
rejected attempts. They are read-only evidence, never part of the active chain;
missing or unverifiable historical code is marked unavailable, not invented.

read_case defaults to the current incumbent. For a history.json row with
candidate_evidence, pass its zero-based history_index and a case_id from that
TRAIN attempt. history_variant defaults to 'candidate'; 'incumbent' selects its
matching past TRAIN baseline when incumbent_evidence is available. This works
when the current batch has rotated; it does not substitute one view for another.
Both historical views need history_index; omitting it selects the current batch,
not the past baseline. Historical IDs are not files in current evidence/.
origin identifies current incumbent, historical_candidate or historical_incumbent; rejected
candidate evidence never becomes active code or counterfactual replay input.
Historical files are hash-bound, read lazily, and exclude private evaluation data.
Missing historical TRAIN evidence is an explicit error, never a TEST fallback.
Context and tool definitions are omitted by default; include_setup=true retrieves
the public setup. This does not change the recorded request messages. Without
call_index, start_message/message_count index the episode history; with call_index,
they index that call's messages, and full_message_count/next_message/eof describe
the page. Omit ranges to read the complete recorded call. Business state remains
explicit: call_index plus environment_state='model'|'action', optional state_path.
There is no total case, page or interaction quota.
evidence.json includes history_length and calls for targeted reads. Startup
feedback identifies the primary source's latest measured attempt, not the last
appended source, and shows paired public tail excerpts for its regressions.
Excerpts are observations, not a diagnosis or a case-read quota; unavailable
history is explicit and never substituted with current or TEST evidence.
Copied history rows can gain missing code/traces from registered
ancestors without duplication or activating old policies. Successful preflight
feedback explicitly recommends real evaluation; it imposes no interaction cap.
H3 may change tool descriptions between calls without changing native tool names
or parameter schemas. include_setup=true with call_index returns that call's
recorded descriptions; evidence reading must not reject this valid H3 behavior.
