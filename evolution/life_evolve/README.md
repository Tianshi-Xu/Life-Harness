# Current Life evolve

Use `python -m evolution.life_evolve.profile prepare|propose|check --help`.
The root [README](../../README.md) describes setup and the TRAIN input contract.

Required candidate outputs are `bundle.json` and referenced `artifacts/*.py`.
Diagnosis/check files are optional notes. Mechanism revision CAS, source hashes,
safe imports, H2–H5 signatures, tool contracts and isolated TRAIN replay are
validated before scoring. Acceptance requires strict improvement on the complete
same TRAIN batch. The latest qualified candidate can be restored on a normal
budget stop; invalid or unmeasured proposals never advance the round.

Each proposal allows unlimited read/reason/edit interactions, at most three
preflight repair opportunities, and at most 10,000,000 provider input+output
tokens including cached input. The Flash per-request output allowance is 100000.
Working-memory checkpoints start at 65536 locally counted input tokens and retain
source/evidence references, exact validation feedback, repair counters and usage.
Unknown inflight paid calls cannot be blindly retried. Keep global Flash
concurrency at or below eight in the outer controller.
