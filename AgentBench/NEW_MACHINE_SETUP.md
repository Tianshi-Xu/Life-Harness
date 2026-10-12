# Native benchmark setup

Initialize all pinned Git submodules from the repository root first.

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
them only after the domain-specific runtime below is ready. The domain-specific commands below provide no-model smoke checks.

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


Use `scripts/native/worker.sh` for native workers. Configure solver endpoints locally;
then use the evolution commands in the root README.
