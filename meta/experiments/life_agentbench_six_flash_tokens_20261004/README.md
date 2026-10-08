# Life six-round Flash token audit

Fresh Life v2.4, policy-free H2-H5 interface v3. ALFWorld, DBBench, WebShop: six rounds each, rotating frozen 50-task training batches. Flash API concurrency at most 8; no local context cap; output 60000. Local Qwen total context 32768 and output 4096, 50 workers per domain. All current train evidence is available on demand.

First validate native setup and the first scored proposal/whole-round API ledger in all domains. Only then run rounds 2-6. Close optimizer broker and seal every training state before final full fixed tests. Test results do not select checkpoints or enter optimizer inputs. Historical datasets/tests have prior exposure; not claimed newly unseen.

Usage: `runner.py --preflight`, `runner.py --launch`. Host ledger: `flash_requests.jsonl` and `TOKEN_USAGE.json`; proposer receipts and row totals per domain; `TRAINING.md` and `COMPARISON.md`; final `FINAL_RESULTS.json`.

No original or historical experiment files are overwritten; old broker source is retained byte-identical for prior fingerprint checks.

