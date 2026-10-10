# Native OS six-round comparison

This adapter runs the original Meta optimizer, pinned official AHE loop, and current Life Evolve against AgentBench's native OS Docker tasks. Each method starts from its own identity baseline and completes six optimizer rounds before any final TEST evaluation. TRAIN batches contain 50 task IDs and are pinned in the experiment protocol; Meta retains its current batch after acceptance and advances after rejection. AHE uses the original best-ever selection. Life uses immutable bundle staging and strict same-batch improvement. The final TEST contains all 144 native TEST tasks.

The deployed solver is Qwen3-4B with thinking disabled, 32,768 total context tokens, 4,096 output tokens, temperature zero and the native 16-turn budget. Ten local nodes run two per GPU on GPUs 1–5 behind port 8400. Solver usage is separate from optimizer usage. Native images, descriptions, tools, checks and graders are unchanged. The adapter fixes persistent PTY transport so stdout is preserved when the controller splits the prompt and command output across frames.

Preparation and launch:

```bash
AgentBench/.native/core/bin/python -m meta.os_comparison.protocol --help
AgentBench/.native/core/bin/python -m meta.os_comparison.launch --root /absolute/path/to/prepared/experiment
```

The launcher creates a dedicated controller container with the original OS image, native Docker socket and AHE chroot capabilities. Existing runs are refused by default. The experiment records native source hashes, TRAIN IDs, public optimizer evidence, candidate hashes, task outcomes, provider receipts and optimization-only token totals.

## Timeout repair and explicitly authorized continuation

The initial OS run used the broker's historical 600-second read timeout, while Life waited 2,100 seconds. OS requests now use a 1,800-second broker read timeout. An OS Life completion retries the same request at most once, and only when the provider receipt confirms a finished `ReadTimeout` before any upstream HTTP response. Context errors, HTTP failures with a response, partial streams and other failures are not automatically retried. Successful calls are never replayed.

The broker also previously initialized `delivered_to_client=True` before receiving response headers. It now initializes that flag to false. The original timeout receipt is preserved; its null HTTP status and absent transport receipt show that the old true flag did not establish delivery.

On 2026-10-08 the user explicitly authorized timeout retries and approximate accounting when gateway billing is delayed. This overrides the otherwise strict unknown-usage stop rule for these OS Life timeout calls. A SHA-bound `*.estimates.json` sidecar stores the authorization and approximation method. Raw API receipts remain unchanged, and `accounting_complete` remains false for unknown usage. `accounting_usable_with_estimates` permits the authorized continuation; estimated tokens and budget totals are reported separately from measured input/output tokens.

The timeout estimate uses the latest measured same-round input-token count and the median previous output-token/wall-second rate times the timeout duration. Queueing, reasoning and actual gateway charges are uncertain. The recorded broad ranges are 0–262,144 input tokens and 0–100,000 output tokens; the input range is a rough uncertainty bracket, not a verified model context limit; the output ceiling comes from the 100,000-token request cap. Neither is a statistical confidence interval or official billing receipt.

The audited continuation in `os_meta_ahe_life_six_20261008_setup3` archives the original checkpoint, retains all 17 successful Life calls, retries the interrupted request, preserves the exact paid Meta candidate, and reuses all 50 original AHE native baseline records. The audit records each permitted adapter source correction and preserves the original frozen source receipt. No original upstream optimizer or historical result is changed. Original Meta's cached-proposal branch performs candidate selection without another paid proposal.

Run an audit only for the supported, fully stopped first-round state:

```bash
# Inside the dedicated trusted controller container, with its original mounts:
/data/home/xts/harness/ahe-native/.venv/bin/python -m meta.os_comparison.recovery_audit --root /absolute/path/to/stopped/experiment
# Then launch meta.os_comparison.run in a new dedicated controller container.
```

`status.json`, `TRAINING_RESULTS.json`, `OPTIMIZER_TOKEN_USAGE.json`, and the per-request ledgers describe progress. `training_seal.json` is written only after all eighteen optimizer rounds finish. `FINAL_RESULTS.json` and phase `complete` indicate that the final held-out TEST evaluations have finished.
