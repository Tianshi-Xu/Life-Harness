# New-machine setup for the six-domain branch

This is a checklist for a **new checkout**, not a claim that the current host has
these services running. It covers the three AgentBench domains in the recorded
six-domain experiment. Airline, Retail, and Telecom use TauBench; OS remains
outside this experiment. Nothing below starts a GPU model service or runs a
scored evaluation.

## Get the tracked source and check the record

```bash
git clone --branch meta-six-round-baseline-20260916 --recurse-submodules \
  https://github.com/Tianshi-Xu/Life-Harness.git Life-Harness
cd Life-Harness
python meta/experiments/meta_from_scratch_six_20260915/audit.py
python meta/experiments/meta_from_scratch_six_20260915/check_local_setup.py
```

The branch includes benchmark source, task lists, ALFWorld game archives,
submodule gitlinks, frozen Meta-Harness source, and the compact six-round
results. It does **not** include virtual environments, extracted ALFWorld games,
MySQL binaries or data, WebShop's multi-gigabyte product data/index, the
AgentRL controller binary, model weights, credentials, or raw episode logs.
The audit needs none of those external assets. The setup check reports local
files and Python imports; a passing row still requires a service smoke test.

## TauBench: Airline, Retail, Telecom

```bash
cd TauBench
uv sync --frozen
cd ..
python meta/experiments/meta_from_scratch_six_20260915/check_local_setup.py
```

The lockfile includes `httpx[socks]` because a SOCKS proxy on the 2026-09-28
host made plain `httpx` fail on import. Task JSON is tracked. Before a new
evaluation, configure the Qwen endpoint and a local teacher-model route in an
ignored `meta/deploy/env.sh`, then test both endpoints without using the
historical experiment directory as a fresh run. See the root README for the
iteration commands and the frozen experiment's `EXPERIMENT.md` for its exact
protocol.

## AgentBench common runtime

Commands below run from `AgentBench/`. A new host needs `uv`, `unzip`, and a
Python 3.12 interpreter. `uv` may download the selected interpreter. Keep all
generated files under ignored `.native/`.

```bash
cd AgentBench
uv venv --python 3.12 .native/core
uv pip install --python .native/core/bin/python \
  -r scripts/native/requirements-core.txt
.native/core/bin/python scripts/native/prepare_configs.py
```

Install the Linux amd64 `controller-v0.2.0` binary from the
[AgentRL release](https://github.com/THUDM/AgentRL/releases/tag/controller-v0.2.0)
as `.native/bin/agentrl` and make it executable. The original host's deployment
record gives the Linux amd64 release SHA-256 as
`0c0ec387385100b317b491d5f4ad1e0ec8bd24751aa41669ad38ec4604a209c7`;
verify the corresponding downloaded release asset against that value before
relying on the old smoke-test record. AgentRL also documents the
[controller and worker deployment](https://github.com/THUDM/AgentRL/blob/main/docs/deployment.md).

The controller and native workers use ports 15020–15023 on loopback. Start
them only after the domain-specific runtime below is ready. The previously
recorded no-model smoke tests are in [NATIVE_ENVIRONMENT.md](NATIVE_ENVIRONMENT.md).

## ALFWorld

The tracked archives share one `json_2.1.1/` directory. Extract all three
into `data/alfworld/`, install the isolated text environment, and run the
no-model smoke test:

```bash
for archive in data/alfworld/json_2.1.1_{json,pddl,tw-pddl}.zip; do
  unzip -n -q "$archive" -d data/alfworld
done
uv venv --python 3.10 .native/alfworld-venv
uv pip install --python .native/alfworld-venv/bin/python \
  -r scripts/native/requirements-alfworld.txt
PYTHONPATH=. .native/alfworld-venv/bin/python scripts/native/smoke_alfworld.py
```

Do not compare scores to the historical run until the smoke test verifies real
game reset, step, and close. This native stack is ALFWorld 0.4.2/TextWorld 1.7.0;
it was not established as trajectory-equivalent to the old Docker stack.

## DBBench

The train/test JSONL files are tracked. The existing `scripts/native/mysql.sh`
expects a **MySQL 8** installation at `.native/sysroot/usr/sbin/mysqld` with
its libraries under `.native/sysroot/usr/lib/x86_64-linux-gnu`. That installation
was local to the original host and is not in Git. Provision it on the new host
before starting the script. It creates a dedicated loopback-only instance on
port 13306 with its data under `/tmp/life-agentbench-native/mysql` by default.
Do not point it at a business database.

```bash
bash scripts/native/mysql.sh
# In another terminal, from AgentBench/:
PYTHONPATH=. .native/core/bin/python scripts/native/smoke_dbbench.py
```

The smoke test exercises real SQL, isolation, cleanup, and three scripted
interactions; it does not measure model accuracy.

## WebShop

The pinned WebShop source submodule is tracked, but the three raw product JSON
files, 100k derived data, and Lucene index are not. Install a Java 11 runtime
and set `WEBSHOP_JAVA_HOME` if it is not in the old `.native/sysroot/` layout.
Then run:

```bash
bash scripts/native/prepare_webshop.sh
```

The script downloads about 5.8 GB of raw data, checks its SHA-256 values,
installs the native Python environment, builds the 100k index, and runs the
no-model WebShop smoke test. Leave sufficient disk space for both downloads
and generated data. Its data source can be changed with `WEBSHOP_DATA_BASE`,
but the hash checks still apply.

After all three native smoke tests pass, use the start commands in
[NATIVE_ENVIRONMENT.md](NATIVE_ENVIRONMENT.md#启动) and its HTTP smoke test to
verify controller-to-worker calls. Only then start model services and new
evaluations. Keep the historical six-round experiment directory unchanged;
create a new run directory for further iteration.

## October source/results handoff and AHE adapter

The latest shared Life implementation is in `meta/life_evolve/`,
`meta/harness_hooks.py`, and `meta/HARNESS_INTERFACE.md`. Historical versioned
implementations are immutable experiment dependencies, not alternative defaults.
Compact results, per-round optimizer accounting, sealed state and committed
harness artifacts are tracked in the corresponding `meta/experiments/` directories.
Raw episodes, request ledgers, local environments, models and credentials are
not included. These archives are evidence, not relocatable live checkpoints:
their absolute paths and original source hashes must not be rewritten to resume
them on a different host. Always create a new directory for a new experiment.
Meta/AHE live workspaces contain nested Git repositories. Their code is archived
as ordinary files under `sealed_agents/` and `sealed_harnesses/` inside the
completed experiment directories, with candidate bytes/fingerprints unchanged;
the nested Git databases and raw test runs are not part of this handoff.

The existing aggregate reports are also archived inside the checkout:

- `meta/reports/BENCHMARK_RESULTS_20261007.md`
- `meta/reports/AHE_META_SIX_ROUND_TRAIN_TOKENS_20261004.md`

The latter explicitly labels historical AHE token estimates. Do not relabel
them as API-measured usage. Qwen2.5 compatibility results are in
`meta/experiments/life_six_qwen25_compat_20261007/FINAL_RESULTS.json`;
the 14B DBBench regression is retained alongside the improvements.

The adapter used by Life/Meta/AHE comparisons originally lived outside this
repository. Its source-only copy is now in `integrations/ahe-native/`.
From the Life-Harness repository root, restore the original sibling layout
on a **fresh host** (the destinations must not already exist). The full AHE
test suite includes a native-data hash audit: prepare the ALFWorld extraction
and WebShop data described above before that test command. Those large assets
are deliberately not added to the source archive.

```bash
test ! -e ../ahe-native && cp -a integrations/ahe-native ../ahe-native
# Git does not retain the original empty scaffold directories.
for domain in alfworld dbbench webshop airline retail telecom; do
  mkdir -p "../ahe-native/baselines/$domain"/{middleware,skills,sub_agents}
done
git clone https://github.com/china-qijizhifeng/agentic-harness-engineering.git \
  ../agentic-harness-engineering
git -C ../agentic-harness-engineering checkout --detach \
  8b2a55d97590363fe50c3cc6b5e833b020a4bb4c
git clone https://github.com/nex-agi/NexAU.git ../ahe-native/vendor/NexAU
git -C ../ahe-native/vendor/NexAU checkout --detach \
  35ee1861546db3cb280a6e17e38a74060d7c96c3
cd ../ahe-native
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements.lock
.venv/bin/python -m unittest discover -s tests -q
```

Life's offline suite can be run from the Life-Harness repository root using
`PYTHONPATH=.:TauBench/src:../ahe-native TauBench/.venv/bin/python -m unittest discover -s meta/tests -q`.
Three optional Qwen2.5 historical replay checks require raw episodes/runtime
manifests retained on the original host and are skipped when absent. Synthetic
tool-envelope/history tests still run. Java/environment readiness is verified
by the native smoke tests above, not by requiring a local Java installation
for these unit tests.

The lockfile's Debugger dependency is relative to this sibling layout; the
adapter Python files and baseline workspaces are copied unchanged. Existing
adapter destinations must be reconciled manually, not overwritten by these
fresh-host commands. The isolated AHE runtime also requires `CAP_SYS_CHROOT`;
having Docker installed does not itself grant this capability to the launcher.
Export `MR_API_KEY` and, for Airline/Retail, `DASHSCOPE_API_KEY` in the host
process or a local ignored `env.sh`. Never commit that file. The historical MR
key previously present in Git should be rotated; replacing its current value
with an environment reference does not remove it from old commits.

## OS on a Docker-capable host

The original OS task, train/test profiles, data, Dockerfiles and Compose services
are tracked. See [README.md](README.md#build-docker-images) for image build
commands. `extra/docker-compose.yml` provides separate `os-env_train` and
`os-std` workers; `configs/tasks/os.yaml` defines their splits. Use the training
worker for optimization and keep the test worker outside optimizer evidence.
Before model evaluation, verify Docker access, image builds, worker registration,
and an actual environment reset/action/close cycle.

OS is **not yet supported by the latest shared six-round comparison adapters**:
`meta/harness_hooks.py` currently has six domain bindings excluding OS, and
`ahe_native/native_worker.py` likewise has no OS binding. The WebShop-only
`meta.life_agentbench_run` entry point must not be used as an OS launcher.
After the Docker smoke passes, add the OS transport/hook binding and a fresh
train-only six-round protocol, retaining the same optimizer accounting and
global DeepSeek concurrency limit of eight. Only then run the held-out test.
This source handoff does not start or claim completion of that missing experiment.
