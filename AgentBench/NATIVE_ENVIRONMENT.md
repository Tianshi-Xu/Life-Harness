# AgentBench 无 Docker 部署

本机部署记录：2026-09-12，WebShop 于 2026-09-15 补齐。运行时、下载件和生成配置均放在忽略的 `.native/`；
MySQL 数据目录在 `/tmp/life-agentbench-native/mysql`。没有修改系统 Python、
安装系统服务，也没有修改 H2–H5、题目或评分标准。

## 任务依赖与边界

| 任务 | 原生路径 | 数据检查 |
|---|---|---|
| ALFWorld | Python 3.10 + ALFWorld/TextWorld，文本环境不需要 Unity | 本地三个数据 ZIP 已展开；new_std 109，train_valid 3150 |
| DBBench | 独立 MySQL 8.0.46；`env_driver: native_mysql` | standard 300，训练 4803；当前两者全部使用 MySQL |
| OS | 已验证 bubblewrap 独立根目录，但私有 `/proc` 挂载失败 | 禁止将模型命令直接交给宿主 shell；暂不暴露完整 OS worker |
| WebShop | Python 3.12 文本环境 + Java 11/Lucene | 三份原始 JSON 校验通过；100k 商品、100k 属性、99,995 条可索引文档、1,021 个目标 |

DBBench 的 `db_file: data/dbbench/db_train` 路径不存在，但当前数据没有
`user_sqlite: true`，不会使用该路径。若以后换成 SQLite 数据，需要另行补齐数据库。
原 MySQL Docker 配置的 32 GB buffer pool 在原生环境下改为 128 MB；这是服务资源配置，
仍使用 MySQL 8，未用 SQLite/MariaDB 替代 SQL 引擎。

## 已验证结果

- DBBench：真实 MySQL 查询、两个并发数据库隔离和清理通过；原 `start_sample` 跑通 3 题。
- ALFWorld：六种任务类型各一个真实游戏，reset / step / close 全部通过。
- HTTP：controller → DBBench 的 SQL / 提交路径通过；controller → ALFWorld 的动作 / 取消路径通过。
- WebShop：100k 环境 reset / search / close 通过；controller → WebShop 的
  `start_sample` / `search_action` / `cancel` 路径通过，标准 profile 暴露 200 题。
- 三个 worker 均可注册为 `ALIVE`，冒烟结束后活动会话数均为 0；assigner CLI 可正常导入。
- 没有调用实验模型，没有启动训练、方法迭代或计分实验。

原始检查输出在 `.native/logs/{dbbench-smoke,alfworld-smoke,http-smoke}.json`，
在线状态在 `.native/logs/worker-status.json`，OS 失败证据在 `.native/logs/os-probe.log`。

## 启动

从 AgentBench 目录执行。下面命令是前台进程，建议分别放到终端或进程管理器中。

```bash
.native/core/bin/python scripts/native/prepare_configs.py
# 首次安装 WebShop；会下载约 5.8 GB 原始 JSON、校验哈希、生成 100k 数据和 Lucene 索引
bash scripts/native/prepare_webshop.sh
bash scripts/native/mysql.sh
.native/bin/agentrl controller --host 127.0.0.1 --port 15020 --dashboard=false --long-timeout
bash scripts/native/worker.sh dbbench
bash scripts/native/worker.sh alfworld
bash scripts/native/worker.sh webshop
```

训练 profile 用第二个参数，例如 `bash scripts/native/worker.sh dbbench dbbench-env_train`。
生成配置继承原 profile 的 H2–H5 设置，仅调整本机路径、连接方式和并发为 1。
当前 DBBench 标准 profile 已明确启用 Harness 与 H2/H3/H4/H5；生成配置后仍建议用
`scripts/native/smoke_http.py` 检查 worker 实际加载的版本。

controller 使用 15020，ALFWorld worker 15021，DBBench worker 15022，WebShop worker
15023，MySQL 13306。`prepare_configs.py` 还生成了
`.native/configs/assign-{alfworld,dbbench,webshop}.yaml`，
已调整 controller 地址和并发，保留原 agent 端点配置。配置好模型端点后可运行：

```bash
.native/core/bin/python -m src.assigner -c .native/configs/assign-dbbench.yaml
.native/core/bin/python -m src.assigner -c .native/configs/assign-alfworld.yaml
.native/core/bin/python -m src.assigner -c .native/configs/assign-webshop.yaml
```

以上命令会调用模型并执行评测，本次配置工作未执行它们。
原生 MySQL 是专用回环测试实例，root 无密码，不能拿这个配置连接已有业务数据库。
每个 episode 仍由原 `MySQLDatabase` 创建独立数据库并在 finally 路径清理。
此路径不需要 Redis。

## 无模型冒烟检查

```bash
PYTHONPATH=. .native/core/bin/python scripts/native/smoke_dbbench.py
PYTHONPATH=. .native/alfworld-venv/bin/python scripts/native/smoke_alfworld.py
JAVA_HOME=.native/sysroot/usr/lib/jvm/java-11-openjdk-amd64 \
  PATH="$PWD/.native/sysroot/usr/lib/jvm/java-11-openjdk-amd64/bin:$PATH" \
  PYTHONPATH="$PWD/.native/webshop-src:$PWD" \
  .native/webshop-venv/bin/python scripts/native/smoke_webshop.py
.native/core/bin/python scripts/native/smoke_http.py
bash scripts/native/probe_os.sh
```

DBBench 检查真实查询、两个数据库之间隔离、清理，以及原 `start_sample` 的三次脚本化
交互。脚本固定提交 `1`，不代表模型能力或准确率。ALFWorld 检查各任务类型的一次真实
reset / step / close。WebShop 检查 100k 商品、1,021 个目标以及一次真实 Lucene 搜索。
HTTP 检查要求相应的三个 worker 都已启动。OS 探针在当前机器上应在私有 procfs 阶段失败，
不能据此启动正式实验。

## OS 限制

本机能运行 user namespace 和独立根目录，模型看不到宿主工作目录；但
`bwrap --unshare-all ... --proc /proc` 返回 `Operation not permitted`。
只把 `/proc` 留空，或挂入宿主 `/proc`，都会改变进程相关题目的行为，后者还暴露宿主信息。
因此不提供这种退化环境用于和原 OS 结果比较。`scripts/native/probe_os.sh` 可在更换运行环境后复查。
根文件系统依赖安装还遇到部分包的多用户属主配置失败（例如 dbus 的 chown 返回
`Invalid argument`）；`.native/os-rootfs` 是诊断产物，不是已完成的 OS 基准镜像。

## WebShop 原生构建

源码：`data/webshop_repo`，固定 `64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd`。
`prepare_webshop.sh` 从该 gitlink 导出干净运行副本、应用仓库的 `webshop.patch`，下载
WebShop 原始数据的字节级镜像，再用与 Dockerfile 相同的前 100k 选择规则生成商品、
属性、JSONL 和 Lucene 索引。下载地址可用 `WEBSHOP_DATA_BASE` 覆盖；无论来自何处，
`build_webshop_100k.py` 都会先验证以下 SHA-256：

- `items_shuffle.json`: `2ef591d65df3af89e972ab72468eb82cbf124d876552d9f3678667edd620a6c8`
- `items_ins_v2.json`: `1d36af476bdb8f82a5da62bd8acdabe54cd8de2fa84010d37da5c4890feb447e`
- `items_human_ins.json`: `cf78667548a71786e1d9049c24b802e48e1084ad4bb021cae56ce1f6d96954a3`

生成结果为 100,000 件商品和 100,000 条属性；Lucene 接受 99,995 条文档并跳过
5 条空文本商品，这与 Pyserini 的索引语义一致。运行时位于 `.native/webshop-src`，
Python 环境位于 `.native/webshop-venv`，原始和生成数据保留在被 git 忽略的 submodule
数据目录。Java 11.0.32 已在本机验证；也可通过 `WEBSHOP_JAVA_HOME` 指向其他 Java 11。
`webshop-std` 继续使用 0:200，训练池使用互不重叠的 200:1021，并按配置最多抽样 100 题。

## 上游来源

本机 core 使用 Python 3.12.3 的独立 venv（可读取已有系统包），WebShop 使用单独的
Python 3.12 venv，ALFWorld 使用完全独立的 Python 3.10.21 venv。安装清单和固定依赖
分别记录在 `scripts/native/{core,alfworld}-installed.txt` 与
`scripts/native/requirements-webshop.txt`。ALFWorld 0.4.2 + TextWorld 1.7.0 +
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

15021 的 `alfworld-std`、15022 的 `dbbench-std` 与 15023 的 `webshop-std` 均加载
当前四接口 Harness 及各自发布开关；Task、`FourHookSession` 与 `src/client/task.py` 的
完整历史兼容需配套使用。冻结回放保持 ALFWorld 104/109、DBBench 190/300；三个任务的
真实 HTTP 工具调用均通过。完整迁移记录见
[`current_format_migration_20260913`](../meta/experiments/current_format_migration_20260913/MIGRATION.md)。
