# Life Flash 100k repair continuation

This branch repairs preflight/retention, without overwriting historical implementations or results.

- Predecessor: `../life_agentbench_six_flash_tokens_20261004`, paused at a settled API/checkpoint boundary.
- Inherited qualified decisions: ALFWorld round 1, DBBench rounds 1–3, WebShop round 1.
- Retry ALFWorld and WebShop round 2 first. Only qualified candidates with complete paired 50-task TRAIN evaluations pass the retry gate.
- An unqualified or unmeasured attempt remains pending at the same round; it does not increment the six-round count.
- After the retry gate, finish exactly six qualified measured decisions per domain, then seal training and evaluate the final TRAIN-selected incumbent on the frozen full TEST split.
- DeepSeek-v4-flash: agentic tool loop, max 32 requests per proposal, per-request max output 100000, no local context cap, global concurrency at most 8.
- Local Qwen3-4B: 50 native episode workers per domain, total context 32768, output 4096.
- Case evidence is read on demand; Life has no ten-case cap.
- Qualified candidate snapshots are retained separately from the writable proposal workspace and fully revalidated before fallback/export.
- Safe explicitly imported `re.compile` is allowed; builtin code `compile`, shadowing, network and external LLM calls remain forbidden.
- Cross-layer tests may be shared only by related surviving layers. References to filtered shared-state owners are normalized without silently accepting unknown owner names.
- Recovered old drafts are checked against a fresh v2.5 input workspace. Fully qualified drafts need zero new API calls; invalid drafts receive explicit preflight feedback.
- Original paid requests, rejected candidates, unqualified attempts and abandoned branches remain charged in `ancestor_usage.json` and `TOTAL_SPEND.json`.
- `TOKEN_USAGE.json` records new requests only. Do not treat it as total historical experiment spend.
- Historical TRAIN/TEST exposure is explicitly true. This is a repair continuation, not a fresh equal-budget experiment.

Provenance is in manifests, code hashes and experiment artifacts. Current repair regression tests are consolidated in `meta/tests/test_life_repair.py`; no per-version test copies are needed.

