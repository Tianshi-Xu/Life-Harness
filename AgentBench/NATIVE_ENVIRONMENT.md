# AgentBench 无 Docker 部署

本机部署记录：2026-09-12。运行时、下载件和生成配置均放在忽略的 `.native/`；
MySQL 数据目录在 `/tmp/life-agentbench-native/mysql`。没有修改系统 Python、
安装系统服务，也没有修改 H2–H5、题目或评分标准。

## 任务依赖与边界

| 任务 | 原生路径 | 数据检查 |
|---|---|---|
| ALFWorld | Python 3.10 + ALFWorld/TextWorld，文本环境不需要 Unity | 本地三个数据 ZIP 已展开；new_std 109，train_valid 3150 |
| DBBench | 独立 MySQL 8.0.46；`env_driver: native_mysql` | standard 300，训练 4803；当前两者全部使用 MySQL |
| OS | 已验证 bubblewrap 独立根目录，但私有 `/proc` 挂载失败 | 禁止将模型命令直接交给宿主 shell；暂不暴露完整 OS worker |
| WebShop | 可原生运行 Python + Java 11/Lucene，但原资源下载受阻 | 源码已固定原 Dockerfile commit；100k 商品、属性和预建索引缺失 |

DBBench 的 `db_file: data/dbbench/db_train` 路径不存在，但当前数据没有
`user_sqlite: true`，不会使用该路径。若以后换成 SQLite 数据，需要另行补齐数据库。
原 MySQL Docker 配置的 32 GB buffer pool 在原生环境下改为 128 MB；这是服务资源配置，
仍使用 MySQL 8，未用 SQLite/MariaDB 替代 SQL 引擎。

## 已验证结果

- DBBench：真实 MySQL 查询、两个并发数据库隔离和清理通过；原 `start_sample` 跑通 3 题。
- ALFWorld：六种任务类型各一个真实游戏，reset / step / close 全部通过。
- HTTP：controller → DBBench 的 SQL / 提交路径通过；controller → ALFWorld 的动作 / 取消路径通过。
- 两个 worker 均为 `ALIVE`，冒烟结束后活动会话数均为 0；assigner CLI 可正常导入。
- 没有调用实验模型，没有启动训练、方法迭代或计分实验。

原始检查输出在 `.native/logs/{dbbench-smoke,alfworld-smoke,http-smoke}.json`，
在线状态在 `.native/logs/worker-status.json`，OS 失败证据在 `.native/logs/os-probe.log`。

## 启动

从 AgentBench 目录执行。下面命令是前台进程，建议分别放到终端或进程管理器中。

```bash
.native/core/bin/python scripts/native/prepare_configs.py
bash scripts/native/mysql.sh
.native/bin/agentrl controller --host 127.0.0.1 --port 15020 --dashboard=false --long-timeout
bash scripts/native/worker.sh dbbench
bash scripts/native/worker.sh alfworld
```

训练 profile 用第二个参数，例如 `bash scripts/native/worker.sh dbbench dbbench-env_train`。
生成配置继承原 profile 的 H2–H5 设置，仅调整本机路径、连接方式和并发为 1。
当前 DBBench 标准 profile 已明确启用 Harness 与 H2/H3/H4/H5；生成配置后仍建议用
`scripts/native/smoke_http.py` 检查 worker 实际加载的版本。

controller 使用 15020，ALFWorld worker 15021，DBBench worker 15022，MySQL 13306。
`prepare_configs.py` 还生成了 `.native/configs/assign-{alfworld,dbbench}.yaml`，
已调整 controller 地址和并发，保留原 agent 端点配置。配置好模型端点后可运行：

```bash
.native/core/bin/python -m src.assigner -c .native/configs/assign-dbbench.yaml
.native/core/bin/python -m src.assigner -c .native/configs/assign-alfworld.yaml
```

以上两个命令会调用模型并执行评测，本次配置工作未执行它们。
原生 MySQL 是专用回环测试实例，root 无密码，不能拿这个配置连接已有业务数据库。
每个 episode 仍由原 `MySQLDatabase` 创建独立数据库并在 finally 路径清理。
此路径不需要 Redis。

## 无模型冒烟检查

```bash
PYTHONPATH=. .native/core/bin/python scripts/native/smoke_dbbench.py
PYTHONPATH=. .native/alfworld-venv/bin/python scripts/native/smoke_alfworld.py
.native/core/bin/python scripts/native/smoke_http.py
bash scripts/native/probe_os.sh
```

DBBench 检查真实查询、两个数据库之间隔离、清理，以及原 `start_sample` 的三次脚本化
交互。脚本固定提交 `1`，不代表模型能力或准确率。ALFWorld 检查各任务类型的一次真实
reset / step / close。OS 探针在当前机器上应在私有 procfs 阶段失败，不能据此启动正式实验。

## OS 限制

本机能运行 user namespace 和独立根目录，模型看不到宿主工作目录；但
`bwrap --unshare-all ... --proc /proc` 返回 `Operation not permitted`。
只把 `/proc` 留空，或挂入宿主 `/proc`，都会改变进程相关题目的行为，后者还暴露宿主信息。
因此不提供这种退化环境用于和原 OS 结果比较。`scripts/native/probe_os.sh` 可在更换运行环境后复查。
根文件系统依赖安装还遇到部分包的多用户属主配置失败（例如 dbus 的 chown 返回
`Invalid argument`）；`.native/os-rootfs` 是诊断产物，不是已完成的 OS 基准镜像。

## WebShop 缺失资源

源码：`data/webshop_repo`，固定 `64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd`。
仓库的 `webshop.patch` 已通过 apply check；未在数据缺失时假装启动成功。
当前 Dockerfile 使用 100k 商品，与旧 `WEBSHOP_HANDOFF.md` 的全量/16GB 机器说明不同。
本机约 700 GiB RAM，当前障碍是数据和下载连接，不是内存。
Java 11.0.32 已在 `.native/sysroot/usr/lib/jvm/java-11-openjdk-amd64` 验证可运行；
WebShop Python 全依赖和搜索服务尚未完成，不宣称 worker 已就绪。

原始资源来自 `longinyu/agentbench-webshop:latest` 的以下数据层：
`sha256:bc12f0ab4b0cfe48d5e999036bb1b967476909cfc68fc20e6ff7dc121c72ac63`（约 11.5 GB）。
Docker Hub blob 下载连接重置，DaoCloud 镜像返回 403；官方 Google Drive 在此机不可达。
需要该层中的 `root/webshop/data`、`search_engine/resources_100k/documents.jsonl`、
`search_engine/indexes_100k`，按原 Dockerfile 的转换方法生成相同 100k 商品与属性文件。
没有下载第三方小商品集冒充原来的实验集。

## 上游来源

本机 core 使用 Python 3.12.3 的独立 venv（可读取已有系统包），ALFWorld 使用完全独立的
Python 3.10.21 venv。安装清单分别记录在 `scripts/native/core-installed.txt` 与
`scripts/native/alfworld-installed.txt`。ALFWorld 0.4.2 + TextWorld 1.7.0 +
fast-downward-textworld 20.6.4 + NumPy 1.26.4 已通过上述检查。
仓库附带的旧 TextWorld 源码是 1.3.2；本次采用官方可安装的新文本栈，尚未证明与旧
Docker 环境逐轨迹等价。之后比较 baseline/Life 时应固定使用同一份原生运行时，
不要直接把它与旧 Docker 分数混在一起。

- [AgentRL 原生 controller/worker 部署](https://github.com/THUDM/AgentRL/blob/main/docs/deployment.md)
- [ALFWorld 文本环境安装](https://github.com/alfworld/alfworld)
- [WebShop 原数据下载脚本](https://github.com/princeton-nlp/WebShop/blob/master/setup.sh)

controller 版本 `controller-v0.2.0`，Linux amd64 发布件 SHA256：
`0c0ec387385100b317b491d5f4ad1e0ec8bd24751aa41669ad38ec4604a209c7`。

## 2026-09-14 四钩子发布

15021 的 `alfworld-std` 与 15022 的 `dbbench-std` 均加载当前四接口 Harness，
H2/H3/H4/H5 全开；Task、`FourHookSession` 与 `src/client/task.py` 的完整历史兼容
需配套使用。冻结回放保持 ALFWorld 104/109、DBBench 190/300，真实 HTTP 工具调用
通过。完整验证记录见
[`current_format_migration_20260913`](../meta/experiments/current_format_migration_20260913/MIGRATION.md)。
