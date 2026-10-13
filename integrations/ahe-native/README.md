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
and `MKNOD` capabilities. This requirement also applies to the Meta-Harness transport
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

## Setup from a source ZIP

For a freshly extracted source ZIP, restore the four pinned public dependencies from the Life-Harness root. This works without Git metadata in the parent repository:

```bash
set -e
while read -r directory repository revision; do
  git init "$directory"
  git -C "$directory" fetch --depth 1 "$repository" "$revision"
  git -C "$directory" checkout --detach FETCH_HEAD
done <<'SOURCES'
AgentBench/data/alfworld_repo https://github.com/alfworld/alfworld.git a5d3476e65e3effc2ac9b7f2493979ba6794e2a6
AgentBench/data/webshop_repo https://github.com/princeton-nlp/WebShop.git 64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd
integrations/agentic-harness-engineering https://github.com/china-qijizhifeng/agentic-harness-engineering.git 8b2a55d97590363fe50c3cc6b5e833b020a4bb4c
integrations/ahe-native/vendor/NexAU https://github.com/nex-agi/NexAU.git 35ee1861546db3cb280a6e17e38a74060d7c96c3
SOURCES
```

Continue with the runtime installation above, starting at `cd integrations/ahe-native`; omit `git submodule update` for a ZIP snapshot. The remaining TauBench and native benchmark setup steps are unchanged. These commands initialize only the public dependency repositories, without importing the Life-Harness repository's history.
