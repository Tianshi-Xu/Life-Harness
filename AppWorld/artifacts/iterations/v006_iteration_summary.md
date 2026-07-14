# v006 Qwen3-4B harness iteration summary

All runs used native tool calls, `temperature=1.0`, and the unchanged AppWorld
train evaluator. Scores from different subsets are not directly comparable.

| Round | Tasks | Passed | Primary change or check |
|---|---:|---:|---|
| R00 | 12 | 3 | mixed baseline |
| R01 | 8 | 3 | global auth contract (rejected as too broad) |
| R02 | 8 | 2 | scoped file-system auth contract |
| R03 | 5 | 1 | post-execution create-conflict advice |
| R04 | 5 | 1 | cross-app procedural skill conditioning |
| R05 | 4 | 2 | scoped phone/note/payment auth contracts |
| R06 | 4 | 2 | per-app post-execution auth recovery |
| R07 | 4 | 2 | delay terminal submission batched with actions |
| R08 | 4 | 2 | block literal token placeholders |
| R09 | 4 | 2 | high-threshold no-action private-text compaction |
| R10 | 4 | 2 | completeness and missing-comparison-field skills |
| R11 | 4 | 3 | page exhaustion and auth-aware tool prediction |
| R12 | 4 | 3 | item metadata discovery for collection tasks |
| R13 | 4 | 2 | bounded-batch collection guidance |
| R14 | 12 | 5 | mixed regression; best same-subset sample |
| R15 | 4 | 2 | numeric rating versus binary-like skill |
| R16 | 4 | 2 | rating pagination and distinct update API |
| R17 | 4 | 2 | drop irrelevant token on public schemas |
| R18 | 12 | 4 | final mixed regression |
| R19 | 9 | 2 | common-failure H3/H5 refinements; guards only |
| R20 | 8 | 2 | parameter-local pagination/auth/submission contracts; guards only |
| R21 | 5 | 1 | near-miss skill specialization; one guard only |
| R22 | 5 | 2 | predictor-temperature-0 ablation; guards only |

## Same-subset comparison

The directly comparable 12-task mixed runs scored:

- R00: 3/12 (25.0%)
- R14: 5/12 (41.7%)
- R18: 4/12 (33.3%)

The three file-system compression variants (`7d7fbf6_1`, `7d7fbf6_2`, and
`7d7fbf6_3`) failed in R00 and all passed in both R14 and R18. `287e338_1`
passed in all three. Other tasks flipped between rounds, so their single-sample
changes are treated as temperature-1 sampling variance unless a trajectory
mechanism was reproduced independently.

## Reproduced mechanism improvements

- Scoped authentication contracts plus post-error advice recovered all three
  file-system compression variants in both later mixed runs.
- The aggregate skill made `82e2fac_2` enumerate every page and retrieve every
  song detail, passing in R11 and R12. It remains sampling-sensitive.
- Metadata guidance changed `34d9492_2` from no useful mutations to 22 correct
  moves in R12; the remaining failure was the 50-turn budget. Batch guidance
  was not reliably followed and was not strengthened further.
- Numeric-rating guidance changed `692c77d_1` from choosing a binary like API
  to selecting review read/create APIs and passing 3/7 evaluator checks in
  R15. Pagination, distinct update selection, and later model sampling remain
  unstable.
- H2's public-schema token repair is narrowly unit-tested. It was added after
  a reproduced 45-turn loop of calls carrying an undeclared `access_token`;
  it did not trigger in the R18 mixed run.

## Final independent layer boundaries

- **H5 task conditioning:** at most one generic procedural skill selected from
  the task instruction; it may condition API prediction but cannot force APIs.
- **H3 environment contract:** one-time description augmentation on tools
  already selected by the upstream predictor; it cannot add or remove tools.
- **H2 action realization:** current-call schema validation, integer-string
  coercion, removal of an irrelevant token only when the schema omits it,
  placeholder blocking, and same-batch terminal delay. It has no trajectory
  state.
- **H4 trajectory regulation:** bounded post-execution advice for auth errors,
  create conflicts, repeated executed actions, error streaks, and budget; it
  does not block actions or send state to another layer.

Post-R22 findings:

- Routing temperature `0.0` consistently omitted important search/login/update
  APIs and is not the recommended default. The main agent remains at
  `temperature=1.0`; predictor temperature continues to inherit it unless an
  explicit ablation value is passed.
- Parameter-local pagination contracts made `afc0fce_2` enumerate to an empty
  page and pass 7/9 evaluator checks, but it over-mutated non-contact
  transactions. The current H5 skill makes contact-set construction a hard
  procedural precondition before mutation.
- A specialized extremum-action skill caused API-predictor regressions in R21
  and R22 and was removed. Extremum actions again use the broader enumeration
  skill, with a common conditional action-only submission convention.

Current unit regression: 29 tests. The next external regression is pending
because the R23 launch was rejected by the Codex-side usage limit, not by
AppWorld or vLLM.
