# HANDOVER — Life-Harness × meta-harness 进化实验交接文档

> 写给接手者：本文档记录当前实验的全部上下文——目标、基础设施、两套进化循环、
> 已测数据、已踩过的坑、待决策事项。读这份 + `meta/README.md` 即可继续工作。
> 日期：2026-09-12。
>
> 本文是 2026-09-12 的历史快照。当前七域迁移状态、冻结起点和运行命令以
> `meta/README.md` 最后一节及仓库根 `README.md` 为准。

## 1. 研究目标（论文实验）

对比两种"自动进化 agent harness"的方法，挂在 Life-Harness 论文的 7 个 benchmark
上（当前只做 TauBench 轨：airline / retail；telecom 因成本暂缓；AgentBench 四题
等 TauBench 验证后再做）：

- **meta-harness**（别人的方法）：proposer agent 写**自由形式**的单文件插件，
  候选之间竞争 frontier。搜索空间无结构。
- **Life-Harness（我们的方法）**：同样的插件工程，但搜索空间**结构化**——每个
  假设必须挂到预定义的 H2-H5 生命周期钩子（H2 执行前规则 / H3 工具描述提示 /
  H4 执行后标注 / H5 技能注入），接受后形成有序插件链。

**核心科学问题**：在相同 proposer、相同评估、相同预算下，结构化搜索空间是否比
自由搜索空间收敛更快、held-out test 泛化更好？正式对比要求**双方都从零 harness
内容起步**（`--from-scratch`），不是从写好的 h2345 起步。

## 2. 基础设施

- 冻结 agent 模型：本地 Qwen3-4B（`/mnt/data2/xts/models/Qwen3-4B`），8 个 vLLM
  实例（GPU 2-5，每卡 2 个，端口 8401-8408，hermes tool parser，max-model-len
  40960），经 round-robin 代理 `http://127.0.0.1:8400/v1`。
  启停：`bash meta/deploy/start_vllm.sh` / `stop_vllm.sh`；健康检查
  `curl http://127.0.0.1:8400/health`。
- user simulator：`openai/deepseek-flash`，通过 ModelRouter 的 OpenAI-compatible
  endpoint；本机 `meta/deploy/env.sh` 从 `/mnt/workspace/xts/skill/test_ap/env.sh`
  读取凭据。跑之前 `source meta/deploy/env.sh`。
- proposer（写代码的模型）：DeepSeek-Flash，经 headless `qoder` CLI 调用
  （`meta/proposer.py` 包装）。
- 评估纪律：**进化只用 tau2 train split**；test split 只在 `--test` finalize 时
  碰，之后 run 冻结。telecom 很贵，默认不跑。
- 成本意识：单题几十轮交互，15 题 retail 一轮全量评估约 10 分钟量级。

## 3. 代码结构（Life-Harness/meta/）

- `meta_harness.py` — meta-harness 循环：anchors（baseline=无 harness；
  h2345=全部层）+ 每轮多候选竞争 frontier。状态在 `runs/<name>/`
  （evolution_summary.jsonl、frontier_val.json、candidates/、evals/）。
- `life_loop.py` — 我们的循环（**2026-09-12 重写为插件链架构**）：
  - proposer 每轮只写 `runs_life/<name>/candidates/iter_NN.py` 一个自包含插件
    （`register()` 钩子挂 H2-H5 注册表）；
  - 接受 = 复制进 `chain/` 并追加 `chain/manifest.json`；拒绝 = 丢弃；
    回滚 = 截断链。**不再编辑任何源码**（git commit/revert 机制已退役）；
  - 评测经生成的 `chain/loader.py`（按序执行整条链）以 `--harness-plugin` 加载；
  - tiered 评估：screen（失败题+哨兵抽样）只做**快速否决**；接受必须全量确认
    （`--accept-on full`，默认）；`--accept-on screen` 可临时接受+批量确认
    （`--full-every N`）；
  - 候选评测前过反泄漏扫描（复用 meta_harness.FORBIDDEN_REFERENCES）+ 链式
    冒烟（worktree venv 里 import + register 整条 staged 链）。
- `zero_layer.py` — `--from-scratch` 的第 0 层插件：清空所有已发布 H2-H5 内容
  （rules/annotators/H3 hints/技能库），钩子机制保留。
- `benchmark.py` — 评测适配器（包 `TauBench/scripts/eval_harness.py`）。
  `META_MOCK_EVAL=1` 时走确定性 mock，不花真钱，用于管线冒烟。
- `skills/tau2-harness/SKILL.md` — 给 proposer 的先验文档（插件契约、真实类名、
  两种 run 模式的挂载点、反泄漏硬约束）。
- `runs/` = meta_harness 的 run；`runs_life/` = life_loop 的 run；
  `worktrees/` = 每个 life run 的隔离 git 检出。

## 4. 已测数据（全部在 train split，retail 子集为前 15 题）

### meta_retail（retail 15题，DeepSeek-Flash proposer，2 轮迭代）
| 系统 | 分数 |
|---|---|
| baseline（无 harness） | 0.667 |
| h2345 anchor | 0.733 |
| iter1 order_index_lookup | 0.600（拒） |
| iter1 delivered_order_money_map | 0.533（拒） |
| iter2 confirm_before_write | 0.600（拒） |
| **iter2 lookup_before_asking** | **0.867（frontier 最优）** |

获胜候选：`runs/meta_retail/candidates/lookup_before_asking.py`（106 行），
= H3 工具提示加 "LOOKUP DISCIPLINE"（订单号必须查工具、绝不问用户）+ H4
轻量 annotator（查用户后提示账户订单数）。它显式避开了 iter1 的"过度扫描"
副作用——失败反馈环路有效。

### life_retail（旧架构：就地改源码；retail 15题）
- 初始 h2345 = 0.800。
- iter1 候选 screen 估计 0.867 被接受，但全量确认实测 0.667（真退步）→ 回滚。
  **这次"screen 幻觉"直接促成了 accept-on-full 重构**：screen 重测的正是被针对的
  失败题，估计系统性偏乐观，只能用于否决、不能用于接受。

### life_retail_chain（新插件链架构首跑；retail 15题，1 轮迭代）
- iter0 初始 h2345 = **0.867**（与上轮 0.800 是同一子集同一配置！）
- iter1 候选 retail_order_discovery_annotators：screen 估计 0.667 →
  便宜否决（screen 只花 10 题就拒掉，符合设计）。
- 值得记录的细节：该候选的方案（H4 annotator 给 get_user_details 结果附加
  订单清单：id/状态/商品）与 meta iter1 **失败**候选 order_index_lookup 的
  思路几乎相同——我们的 proposer 独立提出了"失败变体"并被 screen 拒掉。
  meta 获胜候选恰好是吸取教训后的轻量版（只加入口指引、不动高流量读结果）。
  完整 hypothesis 见 `runs_life/life_retail_chain/state.json`。

### pilot_airline（更早期，airline 子集，学习用）
train：baseline 0.40 / h2345 0.533 / evolved 0.60；test：0.35 / 0.50 / 0.50
（小规模下训练集增益未迁移到测试集）。

## 5. 关键经验（务必读）

1. **噪声警告（最重要）**：h2345 同一 15 题子集三次测量 = 0.733 / 0.800 /
   0.867。噪声带宽 ±0.07，与我们要检测的提升同量级。meta 的 0.867"获胜"很可能
   含相当运气成分。**15 题单 trial 不足以支撑论文结论**——正式实验需要全量
   train（retail ~74 题）或多 trial 配对比较。这是当前最重要的待决策点。
2. 两个循环的 proposer 独立收敛到同一失败模式（retail 的 order discovery：
   agent 问用户要订单号而不是查工具），但"诊断容易用药难"——多数候选修复反而
   引入副作用。meta 唯一成功的那次胜在"轻挂载 + 明确规避前序副作用"。
3. meta 工程优势（已被我们吸收）：自包含插件文件提案成本低、候选之间不互相
   污染、回滚免费、可复现。life_loop 重写后双方工程同构，差异只剩搜索空间
   结构（H2-H5 钩子 vs 自由形式）+ tiered 评估——这正是论文要 ablate 的东西。
4. H2-H5 分层得到独立验证：meta 获胜候选的内容本质上就是一条 H3 提示 +
   一个 H4 annotator，挂在最早可检测的生命周期点。

## 6. 已踩的坑（别重蹈）

- `TauBench/scripts/eval_harness.py` 的 `--harness-plugin` 特性已随当前版本提交。
  `life_loop` 仍会按冻结源校验并同步入口，供旧 run 与隔离 worktree 兼容。
- retail 评估必须带 `--nl`（NL assertion judge 是 reward 基础），否则大量
  infrastructure_error。life_loop 默认 `--nl-domains retail`。
- tau2 的 `try_resume` 会在已有 save dir 时交互式提问 → 非 stdin 环境 EOFError；
  benchmark.py 会删掉缺 harness_summary.json 的残留 save dir。
- vLLM(PPU fork）不认 `--disable-log-requests`；双实例需错峰启动
  （slot1 util 0.44 → 健康后再起 slot2 util 0.50，fork 要求 util×total ≤ 空闲显存）。
- Qwen3-4B 工具调用必须 `--enable-auto-tool-choice --tool-call-parser hermes`。
- qoder headless 写 run 目录需要 `--add-dir` 授权（proposer.run 的 add_dirs）。
- pkill 匹配自身命令行会自杀，用括号技巧（如 `qwen3-4[b]`）。

## 7. 怎么跑（速查）

```bash
cd /mnt/data2/xts/harness/Life-Harness/meta && source deploy/env.sh

# 管线冒烟（不花钱，mock 评估+mock proposer）：
META_MOCK_EVAL=1 python3 life_loop.py --run-name smoke --fresh \
    --iterations 2 --domains airline --mock-proposer

# 我们的循环（layered 学习臂，retail 15题）：
python3 life_loop.py --run-name life_retail_chain --fresh --iterations 1 \
    --domains retail --num-tasks 15 \
    --agent-llm openai/qwen3-4b --user-llm openai/deepseek-flash

# meta 循环（对照）：
python3 meta_harness.py --run-name meta_retail --fresh --iterations 2 \
    --domains retail --num-tasks 15 \
    --agent-llm openai/qwen3-4b --user-llm openai/deepseek-flash

# 正式对比臂（双方从零起步）——评估规模请先解决噪声问题再上：
python3 life_loop.py   --run-name life_fs --fresh --from-scratch ...
python3 meta_harness.py --run-name meta_fs --fresh --from-scratch ...

# held-out test finalize（冻结 run）：
python3 life_loop.py --run-name <name> --test
```

## 8. 待决策 / 下一步

1. **评估规模**（最优先）：全量 train 还是多 trial？15 题单 trial 的噪声
   （±0.07）盖过信号，当前所有"谁赢谁输"都不能当真。
2. 正式对比实验（双方 `--from-scratch`、同预算、同 proposer）：等评估规模
   定了再起跑。
3. 把 meta 获胜候选的 lookup-discipline 思想作为一条 H3/H4 规则移植进
   life harness 源码（学习臂的"取长补短"）。
4. AgentBench 适配器（ALFWorld/DBBench/OS/WebShop）：TauBench 验证后再做。
5. 提交 `eval_harness.py` 的 `--harness-plugin` 改动，删除 worktree 同步
   workaround。


## 9. 接手更新（2026-09-12，Life retail 全量 train 一轮已完成）

该轮运行是 `runs_life/life_retail_full_20260912`，不是旧 15 题运行。
从已发布 h2345 出发，74 题、单 trial，唯一候选 H4 查单入口提示被全量确认接受：
初始 31/74 → 候选 42/74；同期未改动对照复测 38/74。
相对同期对照净增 4 题（+5.41pp），16 题改善、12 题退步；尚无统计可靠性证据。
没有运行 Meta，没有使用 test。详情与逐题差异见该运行的 EXPERIMENT.md / comparison.json。

现有 Life 循环会拒绝无指纹的旧缓存或配置/代码不匹配续跑；扩大题量应另建运行。
本次修复还覆盖候选中断复用、失败缓存恢复、原子状态写入、批量确认失败处理、
零层内置 stuck-loop 提示以及 test 开始即冻结和失败不标完成。
Qoder 1.1.47 的 --allowed-tools 不是工具可见性白名单；proposer.py 已改用 --tools，
并禁止 Bash/Agent/Task。首个未受正确限制的提案会话在写候选前终止，修复后重开，
日志已归档，不能把其消耗遗漏在后续成本分析中。

复现命令见运行 EXPERIMENT.md。临时 uv 位于 /tmp/life-run-tools/bin；如目录不再存在，
需重新准备 uv。请勿对这个运行使用 --fresh。下一轮应沿用配置继续或另建复评运行，
优先检查同期对照中的 12 个回归，并增加重复配对测量；不要声称已获得稳定增益。

## 10. 接手更新（2026-09-12，Meta 同起点一轮 + Life 迁移验证已完成）

用户随后授权 Meta 也从 h2345 迭代一次，并探索 Life 能否继续进化。
该轮两阶段为：

1. `runs/meta_retail_matched_20260912`：复用先前 Life 的同一份 31/74 初始
   h2345 轨迹，Meta 独立产出一个候选后，同期复测冻结 Life 和未改动对照。
   全部 retail/train 74 题、单 trial：h2345 **40/74**、Life H4 **45/74**、
   Meta H3+额外 H5 **35/74**。Life 比 Meta 多 10 题，配对 18 胜 8 负，
   描述性 p=0.0755；并不证明稳定方法优势。Meta 虽超过复用低起点 31/74，
   却低于同期对照，不能因进入 frontier 就称有效。候选数相同，评估成本和
   实际注入提示数不同。Meta 未被强制限定 H2–H5 挂载，但自行选择了 H3/H5。
2. `runs_life/life_retail_transfer_20260912`：以同期 Life 45/74 为确认起点，
   看过 Meta train 轨迹后仅追加一个 H3 候选，保留 H4 和 H5 top_k=1。
   screen 17/37、合并估计 72.97%，全量却为 **40/74**，13 改善/18 退步。
   最终 **reject(full)**，state iteration=2，since_full=0；仅保留
   `identity_resolved_order_lookup_pointer`，确认分数仍为 0.6081。
   本次未能进一步进化。候选是失败实验留档，不是最新推荐插件。

两个阶段新增 333 个最终计分 episode（222 + 111），不含复用锚点和先前 Life
运行。均无最终 infrastructure_error；第一阶段 control 第 109 题发生自动重试。
没有使用 test，没有 finalize。每个运行的 EXPERIMENT.md / comparison.json /
audit/ 记录证据、逐题差异、成本口径和可复现命令。主程序回归测试在实验前 8 项通过。
Meta 路径已修复固定工作树传递、保存提案续跑和 register() 冒烟检查。

下一步应以当前 H4 incumbent 为基准，先做重复对照再下稳定增益结论。
失败题 screen 用于筛选，不用于报告全量准确率；保留 full confirmation，
本轮它成功阻止了一次退步。可以研究按实际 product result 触发的短提示，
但这是待验证假设，尚未实现，不应把静态 H3 检查清单继续无条件叠加。
请勿对上述运行使用 --fresh，也不要将迁移轮包装成与 Meta 等输入、等预算实验。

## 11. 接手更新（2026-09-12，原仓库从零对照一轮已完成）

最新用户要求：双方从零内容起步；Life 仅给 H2–H5 四层设计原则，Meta 使用已
克隆的原始仓库。已完成 `experiments/zero_retail_20260912`，retail/train 74 题，
单 trial，各一个候选，直接全量评估。共同新基线 31/74；Life **35/74**；原始 Meta
**31/74**。Life 对 Meta 配对 12 胜 8 负，p=0.5034；不能据此证明稳定超越。
Meta 候选的计分 token 比 Life 少约 3.51%。未使用 test，没有追加迭代。

Meta 直接 import 克隆仓库 `reference_examples/terminal_bench_2/meta_harness.py`
并调用原 `run_evolve`；统计、frontier 和 history 核心函数也来自未修改的原文件。
commit `44b9942127847f7421db70d8c7e48407f09a3c70`。适配器替换 Harbor/接口、
既有 Qoder proposer 和模型配置，保留原始 skill 的一个分析 Agent 与一个实现
Agent。不要再把先前 `meta/meta_harness.py` 的四层共用提示称为本轮 Meta 实现。
原仓库干净未改动；准确适配边界与来源哈希见该实验协议和 upstream_provenance.json。

Life 运行 `runs_life/life_zero_retail_20260912`：接受 H3 的 7 个操作提示，并在
候选中修复 H3 提示插入导致参数 schema 描述退化的问题；全量 74 份实际 schema
均检查通过。链是 zero_state + retail_h3_lookup_and_selection_hints，确认分数
0.4730，iteration=1、since_full=0。主 checkout 未单独合入该候选中的 H3 helper 修复。

Meta 运行在 `experiments/zero_retail_20260912/native/logs/meta_zero_retail_20260912`，
候选在同实验 `native/agents/grounded_case_card.py`。它在 agent 层每轮从公开对话
重建最多 4000 字符事实卡，临时附加到模型输入。原生准确率 frontier 在平分时
保留 baseline-zero；候选以更低计分 tokens 达到相同成功数，仍值得留作效率对照。
动态卡不保存在普通对话日志中，注册/生成路径已用独立模拟调用验证，不能声称
逐条读取过实际网络请求。模型/工具/解码参数转发检查通过。

两边合计 222 个新计分 episode，27,078,704 agent/user tokens，最终 infra 错误 0。
Life proposer 约 598.5 秒/62 模型请求；Meta 1131.4 秒/121 请求，工具工作流不同，
不是等 proposer compute 或纯设计原则消融。原 skill/请求和审计日志均归档；事件
元数据有长参数截断，审计不是完整安全沙箱证明。离线适配/空接口/schema 检查
见实验目录下脚本及 audit/，不会计作额外模型实验。

复现与续跑使用 joint adapter；隔离工作树特意去掉旧零层内容与 H5 领域启发式，
不应换成主 checkout。不要同时重复启动 launch.py，不要用 --fresh。下一步如要
判断“超越”，应预先固定更多迭代预算、重复测量并保留 held-out 评估；本轮尚未
测试 Meta 依赖多轮完整历史的核心优势。
