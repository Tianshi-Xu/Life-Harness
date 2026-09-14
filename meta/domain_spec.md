# Domain spec: Life-Harness TauBench domains (meta-harness onboarding)

## Domain Summary

- Evaluation unit: one tau2-bench task = a multi-turn dialogue between the
  frozen agent model and an LLM user simulator, with tool calls against a
  domain database; reward 0/1 (or partial) from the tau2 evaluator.
- Domains: airline (30 search / 20 test tasks), retail (74/40),
  telecom (74/40). Search split = tau2 `train` split; held-out test = tau2
  `test` split (matches the paper's reported split).
- Fixed: base model (OpenAI-compatible endpoint), user simulator model,
  environment, task data, H2–H5 layer semantics. Variable: harness code.
- Budget: propose-evaluate iterations, `--iterations` N ×
  `--candidates-per-iter` candidates × domains × trials. Cost scales with
  user-simulator + agent tokens (~130k tokens/simulation on average per the
  paper's RESULTS.md); use `--num-tasks` and `--domains` to control spend.

## Harness and Search Plan

- Candidate interface: single-file Python plugin imported by
  `TauBench/scripts/eval_harness.py --harness-plugin` before environment
  construction; optional `register()` hook. Plugins patch the H2/H4
  rule/annotator dicts, H3 tool descriptions, or the H5 skill bank.
- Interface compliance test: py_compile + import inside the TauBench uv env
  (outer loop, `meta_harness.smoke_check`), plus a forbidden-reference scan
  (no file/network/env access, no task-data references).
- Anchors: `baseline` (no harness) and `h2345` (paper's full harness).
  Proposer hypotheses should target the dominant failure mode of the current
  frontier, one mechanism per candidate.
- Out of bounds: modifying tau2 package code, model weights/prompts outside
  the harness layers, task data, the evaluator, the user simulator.

## Evaluation Plan

- Entry: `meta/benchmark.py --domain D --split search|test ...` wrapping
  `uv run python scripts/eval_harness.py` in TauBench/.
- Primary metric: `average_reward` on the split (written as `accuracy` in
  the score JSON). Secondary: `pass@k`/`pass^k` (trials>1), token counts.
- Score path: `meta/runs/<run>/evals/<domain>/<candidate>/val.json`;
  test results isolated under `meta/runs/<run>/test/.../test.json`.
- Noise: temperature 0 for the agent; user simulator is not temperature
  controlled by default — expect ±0.05 pass variation across 3 trials
  (RESULTS.md). Prefer hypotheses with a clear mechanism over chasing noise.
- Contamination controls: candidates are substring-scanned for data/split/
  log references; test split is only ever touched by `--test` finalization;
  a finalized run refuses further evolution.

## Experience and Logging

- `evolution_summary.jsonl`: one row per candidate — iteration, hypothesis,
  flags, per-domain scores, mean_accuracy, outcome.
- `frontier_val.json`: per-domain best + `_overall` best (full domain
  coverage only).
- `prompts/iter_NN.md` + `qoder_logs/iter_NN.json`: exact proposer prompt
  and raw qoder output, for reproducibility.
- Per-candidate `val.json` carries `save_dir` → TauBench
  `data/simulations/meta_harness/<run>/<domain>/<candidate>/` with
  `results.json` and per-task `task.log` artifacts for failure analysis.

## Open Questions and Unknowns

- Optimal `--num-tasks` per iteration (full search split vs cheap subset)
  given per-eval wall time — unknown until first real run.
- Whether telecom's long-horizon tasks need `--max-steps` capping during
  search to keep iteration cost bounded.
- Agent/user simulator endpoints: supplied via environment
  (`AGENT_API_BASE`, `AGENT_API_KEY`, `USER_API_BASE`, `USER_API_KEY_ENV`);
  the loop inherits the caller's environment.
