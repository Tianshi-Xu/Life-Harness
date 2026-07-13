# WebShop 交接文档

## 状态
**未完成** — 当前机器 16GB RAM 不足以运行 WebShop 全量产品数据集。需要在
**32GB+ RAM** 的机器上运行。

## 原因
WebShop 的 `webshop.patch` 将产品文件路径设为 `items_shuffle.json`（5.5GB，
全量产品）。`load_products()` 用 `json.load()` 一次性加载整个文件到 Python
堆内存，5.5GB JSON 在 Python 中膨胀到 10+GB，加上 JVM、Lucene 索引等，
总计需要 15GB+ 内存。当前机器 Docker VM 仅 12GB，OOM。

## 仓库中已就绪的文件

以下文件均已在仓库中，`git clone` 后直接可用：

| 文件 | 说明 |
|------|------|
| `src/server/tasks/webshop/Dockerfile` | 全量版本，使用 aliyun/tsinghua 镜像加速 |
| `src/server/tasks/webshop/task.py` | 标准 task.py（无简化改动） |
| `src/server/tasks/webshop/webshop.patch` | 原版 patch |
| `configs/tasks/webshop.yaml` | 含 `webshop-std`(harness) 和 `webshop-std-baseline` |
| `configs/assignments/webshop_{model}.yaml` | 3个 harness assignment |
| `configs/assignments/webshop_{model}_baseline.yaml` | 3个 baseline assignment |
| `extra/docker-compose.yml` | 含 controller/redis/alfworld/webshop 服务 |
| `.dockerignore` | 排除 .venv/outputs/.git 等 |

## 在新机器上运行的步骤

### 0. 前置条件
- Docker Desktop（分配至少 **16GB** 内存给 Docker VM，推荐 24GB）
- Python 3.9+（用于运行 assigner）
- 网络可访问 `routify-pub.alibaba-inc.com`（ModelRouter API）

### 1. Clone 仓库
```bash
git clone <repo-url> Life-Harness
cd Life-Harness/AgentBench
```

### 2. 准备 WebShop 数据
```bash
# 克隆 WebShop 仓库（Dockerfile 需要 COPY data/webshop_repo）
git clone https://github.com/princeton-nlp/WebShop.git data/webshop_repo
cd data/webshop_repo && git checkout 64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd && cd ../..

# 拉取 longinyu 镜像（含预构建搜索索引和产品数据）
docker pull longinyu/agentbench-webshop:latest
```

### 3. 配置 Docker 内存
```bash
# macOS: 修改 Docker Desktop 设置为 16GB+（推荐 24GB）
# Settings → Resources → Memory → 16 GB (or higher)
```

### 4. 构建镜像
```bash
# 构建时间约 10-20 分钟（apt + pip install）
DOCKER_BUILDKIT=0 docker build \
  -f src/server/tasks/webshop/Dockerfile \
  -t agentbench-webshop-std . 2>&1 | tee logs_webshop_build.log
```

> **注意**: 如果 Dockerfile 中的 aliyun/tsinghua 镜像在新机器上不需要
> （例如在海外），可以移除 `sed` 镜像替换行，使用官方源。

### 5. 启动基础设施
```bash
docker compose -f extra/docker-compose.yml up -d controller redis
```

### 6. 启动 WebShop 容器（harness 模式）
```bash
docker run -d --name agentbench-fc-webshop-std-1 \
  --network agentbench-fc_default \
  -v "$(pwd)/configs:/app/configs" \
  -v "$(pwd)/src/server/tasks/webshop:/app/src/server/tasks/webshop" \
  -v "$(pwd)/src/server/harness:/app/src/server/harness" \
  agentbench-webshop-std \
  --controller http://controller:5020/api webshop-std

# 等待 2-3 分钟，检查日志确认初始化完成
docker logs agentbench-fc-webshop-std-1 2>&1 | tail -10
# 应看到 "Task worker initialized" 和 "Uvicorn running"
```

### 7. 运行 harness 实验（3模型）
```bash
source .venv/bin/activate  # 或 python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt  # 如果还没安装

python -m src.assigner -c configs/assignments/webshop_gpt-5.5.yaml
python -m src.assigner -c configs/assignments/webshop_gemini-3.1-pro-preview.yaml
python -m src.assigner -c configs/assignments/webshop_claude-opus-4-8.yaml
```

### 8. 运行 baseline 实验（3模型）
```bash
# 停止 harness 容器，启动 baseline 容器
docker rm -f agentbench-fc-webshop-std-1
docker run -d --name agentbench-fc-webshop-baseline-1 \
  --network agentbench-fc_default \
  -v "$(pwd)/configs:/app/configs" \
  -v "$(pwd)/src/server/tasks/webshop:/app/src/server/tasks/webshop" \
  -v "$(pwd)/src/server/harness:/app/src/server/harness" \
  agentbench-webshop-std \
  --controller http://controller:5020/api webshop-std-baseline

# 等待初始化完成
docker logs agentbench-fc-webshop-baseline-1 2>&1 | tail -5

python -m src.assigner -c configs/assignments/webshop_gpt-5.5_baseline.yaml
python -m src.assigner -c configs/assignments/webshop_gemini-3.1-pro-preview_baseline.yaml
python -m src.assigner -c configs/assignments/webshop_claude-opus-4-8_baseline.yaml
```

### 9. 计算结果
```python
import json, os, glob

for model in ["gpt-5.5", "gemini-3.1-pro-preview", "claude-opus-4-8"]:
    for mode in ["run", "baseline"]:
        dirs = glob.glob(f"outputs/{model}/webshop/*-{mode}")
        if not dirs:
            continue
        d = sorted(dirs)[-1]  # 最新一次
        task = "webshop-std" if mode == "run" else "webshop-std-baseline"
        path = f"{d}/{model}/{task}/runs.jsonl"
        if not os.path.exists(path):
            continue
        total = success = 0
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                total += 1
                res = r.get("output", {}).get("result", {})
                if isinstance(res, dict) and res.get("reward", 0) > 0:
                    success += 1
        rate = success / total * 100 if total > 0 else 0
        print(f"{model:30s} {mode:10s}: {success}/{total} = {rate:.1f}%")
```

## 注意事项

1. **WebShop 初始化时间**: 容器启动后需 2-3 分钟加载搜索索引，等日志出现
   "Uvicorn running" 后再运行 assigner。

2. **Claude temperature**: Claude Opus 4.8 不接受 `temperature` 参数，已在
   `configs/agents/api_agents.yaml` 中设为 `null`。

3. **Docker 网络**: macOS 上 `network_mode: host` 不工作，已改为端口映射 +
   service name（`controller:5020`）。

4. **ALFWorld TextWorld**: ALFWorld 的 TextWorld 依赖 `setup.sh` 从
   `emshort.com` 下载 Inform7 CLI，该域名在中国可能被拦截。ALFWorld
   Dockerfile 已改为使用本地 patch 过的 TextWorld 源码（跳过 Inform7
   下载，ALFWorld 使用预编译数据不需要 Inform7）。

5. **Docker Base 镜像**: 如果新机器在中国，可能需要从 daocloud 拉取基础镜像:
   ```bash
   docker pull docker.m.daocloud.io/library/python:3.9-bullseye
   docker tag docker.m.daocloud.io/library/python:3.9-bullseye python:3.9-bullseye
   ```
