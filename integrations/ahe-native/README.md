# Native AHE adapter

Runs the original AHE Debugger/Evolve loop against native ALFWorld, DBBench,
WebShop, OS, Airline, Retail and Telecom. Upstream AHE is pinned by the sibling
`agentic-harness-engineering` Git submodule at
`8b2a55d97590363fe50c3cc6b5e833b020a4bb4c`; its tracked source must remain unchanged.
NexAU is pinned in `requirements.lock` at
`35ee1861546db3cb280a6e17e38a74060d7c96c3`.

From the repository root:

The native optimizer runtime uses Linux chroot, UID isolation and device nodes.
Run its setup and trusted controllers with the required privileges, typically
as root inside a dedicated container. The OS launcher provisions `SYS_CHROOT`
and `MKNOD` capabilities. This requirement also applies to the META transport
that reuses this runtime.

```bash
git submodule update --init --recursive
cd integrations/ahe-native
uv venv --python 3.13
uv pip install --python .venv/bin/python -r requirements.lock
export PYTHONPATH="$PWD:$PWD/../.."
.venv/bin/python -m ahe_native.isolation
.venv/bin/python -m ahe_native.cli --help
.venv/bin/python -m ahe_native.batch --help
```

Install benchmark workers and services separately (see
[AgentBench setup](../../AgentBench/NEW_MACHINE_SETUP.md)). Keep `MR_API_KEY`,
`MR_HOST` and endpoint credentials in the process environment. Native workers
use public baseline instructions and mechanical RPC wrappers in `baselines/`.

The adapter retains original AHE statistics, search, best-ever selection and
Debugger/Evolve bodies. The `h2345-budget-restricted-v1` profile restricts evolved
components, additional tools/model calls and evidence exposure; identify it as
restricted-search AHE when reporting comparisons. Optimization runs use TRAIN
only. Native scoring and model budgets remain environment infrastructure.
The original evaluate-before-evolve convention leaves the last candidate
unmeasured unless the extra final TRAIN measurement is requested.

```bash
PYTHONPATH="$PWD:$PWD/../.." .venv/bin/python -m unittest discover -s tests -q
```

Fresh results remain under ignored `batches/`, `experiments/` and `heldout_tests/`.
Historical comparison reports are optional; their absence must not block a fresh
AHE run or substitute scores for newly measured results.
