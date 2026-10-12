# Native Meta / AHE Tau comparison

```bash
python -m evolution.tau_comparison.run --help
python -m evolution.tau_comparison.run --name NEW_RUN --smoke-only
python -m evolution.tau_comparison.run --name NEW_RUN --launch
```

Requires the AHE environment, native TauBench environment, local solver and
configured host credentials. Six rounds use identity initialization and native
splits from `evolution/reproduction/reference/manifest.json`. Airline uses 30 tasks
with two trials; Retail and Telecom use 50 tasks with one trial. Global Flash
concurrency is at most eight. The solver uses total context 32768/output 4096.
Original Meta `run_evolve` and AHE Debugger/Evolve selection remain upstream.
AHE uses the adapter's restricted component/evidence profile. Final TEST waits
until TRAIN states are frozen. New results and provider usage remain local.

Plugin-based Life Tau runs use `evolution/life_loop.py`; current H2–H5 bundle proposals
use `evolution.life_evolve.profile`. Report entry points and match experimental settings
when comparing methods. The controller above starts Meta and AHE only.
