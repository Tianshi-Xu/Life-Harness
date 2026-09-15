# meta/ — meta-harness evolution loop for Life-Harness

Migrates the [meta-harness](../../meta-harness) method (frozen base model +
automatic search over harness code) onto the Life-Harness benchmarks. This
directory adapts the TauBench track (airline / retail). An isolated native
AgentBench pilot (ALFWorld / DBBench) is documented in
[`experiments/agentbench_zero_20260912/EXPERIMENT.md`](experiments/agentbench_zero_20260912/EXPERIMENT.md);
it uses its own four-hook adapter and does not replace the standard workers.

## Two loops, two methods

- `meta_harness.py` — meta-harness style: candidate single-file plugins
  compete against a frontier (`runs/<run>/evolution_summary.jsonl` +
  `frontier_val.json`). Free-form search space (any patch the proposer
  writes).
- `life_loop.py` — the Life-Harness method: same plugin engineering, but a
  **structured search space** — every hypothesis must mount onto the
  predefined H2-H5 lifecycle hooks, and accepted plugins form an ordered
  chain (`runs_life/<run>/chain/manifest.json` + generated `loader.py`).
  Rejection discards the candidate; rollback truncates the chain. No source
  file is ever edited.
- `zero_layer.py` — chain layer 0 for `--from-scratch` runs: clears all
  released H2-H5 content (rules, annotators, H3 hints, skill bank) while
  keeping the hook machinery active.

## Experiment arms

- Learning runs (absorb meta-harness design ideas): start FROM the released
  h2345 harness (anchor), default flags all on — `life_loop.py` default mode,
  `meta_harness.py` default mode.
- Formal comparison (both arms start from ZERO harness content):
  `meta_harness.py --from-scratch` (baseline anchor only, candidates are pure
  free-form plugins with no H2–H5 layers) and `life_loop.py --from-scratch`
  (chain starts at `zero_layer.py`: all H2-H5 hook machinery active but
  empty; all harness content is evolved via plugins).

## Layout

- `meta_harness.py` — deterministic outer loop (propose → validate →
  evaluate → frontier), resume-safe, with `--test` finalization.
- `life_loop.py` — chain-of-plugins loop on H2-H5 hooks with tiered
  evaluation (screen fast-reject, full-split confirm before accept;
  `--accept-on screen` for cheaper batched confirmation).
- `zero_layer.py` — from-scratch chain layer 0 (clears released H2-H5
  content).
- `proposer.py` — headless `qoder` CLI wrapper (the proposer agent).
- `benchmark.py` — TauBench adapter: wraps `scripts/eval_harness.py`,
  converts `harness_summary.json` into the score contract (`accuracy` field).
- `skills/tau2-harness/SKILL.md` — proposer prior: plugin contract, harness
  architecture pointers, hypothesis strategy, hard constraints.
- `domain_spec.md` — onboarding spec (meta-harness ONBOARDING.md format).
- `runs/<run_name>/` — all state: evolution_summary.jsonl, frontier_val.json,
  candidates/, evals/, prompts/, qoder_logs/, test/, finalized.json.
- `runs_life/<run_name>/` — life_loop state: state.json (accepted chain,
  task-score cache, history), chain/ (accepted layers + loader), candidates/
  (all proposals incl. rejected), evals/, prompts/, qoder_logs/.

## How it works

A candidate harness is a single Python file that the eval entry imports via
the new `--harness-plugin` flag on `TauBench/scripts/eval_harness.py`
(imported before environment construction; optional `register()` hook).
Evolution uses the tau2 `train` split as the search set; the `test` split is
only touched by `--test` finalization, after which the run is frozen.
Candidate code is scanned for forbidden references (task data, splits, prior
simulation logs, file/network/env access) and import-smoke-checked before
every evaluation.

## 实验记录（溯源）

每个数字都标注来源文件；`val.json.accuracy` 为该次评估原始分。
更详细的背景见 HANDOVER.md。

### 结论速览（retail train 前 15 题，DeepSeek-Flash proposer）

| 方法 | 起点 → 终点 | 说明 |
| --- | --- | --- |
| Ours（life_loop 插件链） | 0.867 → 0.867 | 1 次提案被 screen 否决，无接受 |
| Meta（meta_harness） | 0.733 → 0.867 | 4 次提案，第 4 次接受 |

**警告**：同一 h2345 在相同 15 题上三次测量 = 0.733 / 0.800 / 0.867
（来源见下），噪声带宽 ±0.07 与提升同量级——以上对比暂不具备统计意义。

### Run: meta_retail（2026-09-11，retail 15题，2 轮迭代）

溯源：`runs/meta_retail/evolution_summary.jsonl`（逐候选分数）、
`runs/meta_retail/frontier_val.json`、`runs/meta_retail/candidates/*.py`

| 系统 | 分数 | 结果 |
| --- | --- | --- |
| baseline（无 harness） | 0.667 | anchor |
| h2345 | 0.733 | anchor |
| iter1 order_index_lookup | 0.600 | 拒（过度扫描副作用） |
| iter1 delivered_order_money_map | 0.533 | 拒 |
| iter2 confirm_before_write | 0.600 | 拒 |
| iter2 lookup_before_asking | 0.867 | **frontier 最优**（H3 提示 + 轻量 H4 annotator） |

### Run: life_retail（2026-09-11，旧就地改源码架构，retail 15题）

溯源：`runs_life/life_retail/state.json`、
`runs_life/life_retail/evals/retail/iter_{000,001_screen,001_full}/val.json`

- iter_000 初始 h2345 = 0.800
- iter1 候选 screen 估计 0.867（screen val.json 原始 0.818）→ 临时接受
- 全量确认 iter_001_full = **0.667**（真退步）→ 批量回滚，终态 0.800
- 教训（screen 幻觉）：screen 重测的正是被针对的失败题，估计系统性偏乐观。
  此后规则：screen 只否决，接受必须全量确认（`--accept-on full`）。

### Run: life_retail_chain（2026-09-11，新插件链架构首跑，retail 15题，1 轮）

溯源：`runs_life/life_retail_chain/state.json`、
`runs_life/life_retail_chain/evals/retail/iter_{000,001_screen}/val.json`、
`runs_life/life_retail_chain/candidates/iter_01.py`

- iter_000 初始 h2345 = 0.867（同一 h2345 第三次测量值）
- iter1 retail_order_discovery_annotators（H4 订单清单标注）：
  screen 原始 0.500 / 合并估计 0.667 < 0.867 → 便宜否决，链保持为空
- 注意：该方案与 meta iter1 失败候选 order_index_lookup 思路几乎相同——
  诊断易、用药难的交叉验证

### Run: pilot_airline（更早期学习用，airline 子集）

溯源：`runs/pilot_airline/evolution_summary.jsonl`、
`runs/pilot_airline/test/airline/{baseline,h2345,safe_upgrade_rebook_guard}/test.json`

- train：baseline 0.400 / h2345 0.533 / evolved(safe_upgrade_rebook_guard) 0.600
- test（held-out）：baseline 0.350 / h2345 0.500 / evolved 0.500
- 小规模下 train 增益未迁移到 test

### 噪声证据（同一 h2345，retail 前 15 题，同配置三次测量）

| 测量 | 分数 | 来源 |
| --- | --- | --- |
| meta_retail anchor | 0.733 | `runs/meta_retail/evolution_summary.jsonl` |
| life_retail iter_000 | 0.800 | `runs_life/life_retail/evals/retail/iter_000/val.json` |
| life_retail_chain iter_000 | 0.867 | `runs_life/life_retail_chain/evals/retail/iter_000/val.json` |

结论：正式对比前必须先扩大评估规模（全量 train 或多 trial），见
HANDOVER.md 第 8 节。

### Run: life_retail_full_20260912（2026-09-12，retail 全量 train，Life 一轮）

溯源：`runs_life/life_retail_full_20260912/EXPERIMENT.md`、
`comparison.json`、`state.json`、`evals/retail/*/val.json`。
仅运行 Life 学习臂，从已发布 h2345 起步；74 题，单 trial，未运行 Meta 或 test。

| 评估 | 成功题数 | 分数 |
| --- | ---: | ---: |
| 初始 h2345（iter_000） | 31/74 | 0.4189 |
| 候选 screen（iter_001_screen） | 28/51 | 0.5490 |
| 候选全量确认（iter_001） | 42/74 | **0.5676** |
| 未改动 h2345 同期复测（control_repeat_001） | 38/74 | 0.5135 |

- 接受插件：`identity_resolved_order_lookup_pointer`，仅 H4：身份查询成功后
  提示通过 `get_user_details` 的 orders 字段查订单号，不附加完整订单清单。
- 相对首次起点 +14.86 个百分点；相对同期未改动对照 **+5.41 个百分点**
  （净增 4 题；16 题改善、12 题退步）。单 trial 的描述性配对检验 p=0.5716，
  **观察到提升，但尚未证明稳定收益**。不能将全部起点差值归因于插件。
- 同一旧 15 题子集在本次起点仅 7/15，说明历史 ±0.07 不是噪声上界。
- screen 合并估计 0.6892，不是全量确认成绩。4 次评估均无 infrastructure_error。
- 修复了 Life 配置/代码/插件链缓存校验、中断续跑、失败分数恢复、零层残留
  stuck-loop 提示和 test 冻结状态；另修复 Qoder `--allowed-tools` 不能限制
  工具集合的问题，改用 `--tools`。6 项回归测试通过。
- 首次提案因工具边界失效在产出前终止；修复后重新提案，仍只评估了一个候选。
  事件日志与执行偏差记录位于 `audit/`。完整 train 轨迹精简导出在 `train_evidence/`。

### Run: meta_retail_matched_20260912（2026-09-12，Meta 从 h2345 迭代一次）

溯源：`runs/meta_retail_matched_20260912/EXPERIMENT.md`、`comparison.json`、
`audit/` 与 `evals/retail/*/val.json`。仍是 retail/train 全部 74 题，单 trial。
Meta 与先前 Life 使用相同的初始 h2345 31/74 轨迹、模型和一候选预算；
Meta 提案结束后才引入已冻结的 Life 插件，并同期复测三组。

| 本轮全量评估 | 成功题数 | 分数 |
| --- | ---: | ---: |
| 未改动 h2345 | 40/74 | 0.5405 |
| 已接受 Life H4 订单查找提示 | **45/74** | **0.6081** |
| Meta `variant_match_check`（H3 + 额外 H5 技能） | 35/74 | 0.4730 |

- 当前 Life 比 Meta 多 10 题（+13.51 个百分点），配对 18 胜、8 负；
  描述性精确 McNemar p=0.0755。单次 train 结果尚不能证明方法稳定优越。
- Meta 超过复用的初始 31/74，成为其 frontier 最优，却低于同期未改动对照
  40/74。因此不能把“胜过起点、被接受”等同于真实收益。
- Meta 自行选择 H3 规格检查和 H5 额外注入，未强制它遵循 H2–H5 挂载。
  H5 实际可能返回两条技能，与配置 top_k=1 的 Life 不同；不能称提示预算相等。
- 三组新增 222 个最终计分 episode，无最终基础设施错误；control 第 109 题
  有自动重试。初始 31/74 是明确记录来源的复用，未重跑、未重复计入本轮预算。
- 候选数相同不等于总评估成本相同：Life 原先另有 51 题 screen 和一次中断提案。
  Meta 续跑现在复用已保存提案，smoke 调用 register，proposer 固定五工具；
  本轮开始前相关 8 项回归测试通过。未使用 test。

### Run: life_retail_transfer_20260912（2026-09-12，Meta 经验引导的 Life 追加一轮）

溯源：`runs_life/life_retail_transfer_20260912/EXPERIMENT.md`、`comparison.json`、
`state.json` 与 `audit/`。复用同期 Life 45/74 为起点，保留 H4，只尝试一份
精简 H3 换货规格约束检查；不复制 Meta 额外 H5 注入，H5 top_k 仍为 1。
这轮使用了 Meta 已完成的 train 轨迹，因此是额外迁移实验，不是独立等预算比较臂。

- screen：17/37，恢复 10 道失败题、丢失 1 道哨兵题，合并估计 **72.97%**。
- 全量确认：**40/74 = 54.05%**，低于当前 Life 45/74；13 题改善、18 题退步，
  净减 5 题。描述性精确 McNemar p=0.4731，单 trial 不支持稳定收益结论。
- 最终 **reject(full)**，已恢复原任务分数，仍仅保留已接受的 H4 订单查找提示。
  本次未实现进一步提升；候选保留在 candidates/ 供分析，不进入 active chain。
- 新增 111 个计分 episode，screen/full 无最终基础设施错误，未运行 test。
  74 份实际工具 schema 均验证 H3 生效；无 Meta 的额外 H5 清单。
- 流程经验：失败题 screen 的估计不代表全量成绩，必须保留 full confirmation；
  局部修复不能代替回归检查，后续应区分查单、规格选择和写入参数的错误。

### Run: zero_retail_20260912（2026-09-12，从零四层 Life vs 原仓库 Meta）

用户要求双方不基于已发布 h2345，Life 仅提供四层设计准则，Meta 使用克隆原仓库。
溯源：`experiments/zero_retail_20260912/EXPERIMENT.md`、`comparison.json`。
Meta 直接执行原仓库 Terminal-Bench 外循环（commit `44b9942`），原代码未改；
retail/模型/proposer 接口由独立 adapter 适配。双方共享一份新测零基线轨迹，
各一候选、全量 74 题 train、单 trial；本轮 Life 关闭失败题 screen，未用 test。

| 系统 | 成功题数 | 成功率 | 计分 agent/user tokens |
| --- | ---: | ---: | ---: |
| 新零起点 | 31/74 | 41.89% | 9,748,317 |
| Life：H3 操作提示 + schema 保真修复 | **35/74** | **47.30%** | 8,819,830 |
| 原始 Meta：动态公开事实卡 | 31/74 | 41.89% | **8,510,557** |

- Life 对 Meta 配对 12 胜/8 负，净增 4 题（+5.41pp），描述性 p=0.5034；
  这是正面探索信号，尚不能说已稳定超越。Meta 比 Life 少用约 3.51% 计分 tokens。
- Life 接受 H3 候选；新链为 zero_state + retail_h3_lookup_and_selection_hints。
  Meta 原始准确率 frontier 平分时保留 baseline-zero；其候选仍有本次用量优势。
- 初始 retail H2/H4 内容、全部 H3 文本、H5 内容及领域路由/加分启发式均移除；
  空接口检查通过。双方只见本轮轨迹及隔离源码，不复用旧插件/实验经验。
- Life 提案约 10 分钟、62 次模型请求；Meta 按原始 skill 两阶段 Agent 工作流，
  约 19 分钟、121 次模型请求。因此候选/评估预算相同，不等于 proposer 成本相同。
- 本轮共新增 222 个计分 episode、27,078,704 agent/user tokens，最终 infra 错误 0。
  Meta 核心函数来源、schema 检查、注册/动态消息检查和工具访问日志均已归档。
- 单轮没有检验 Meta 利用多轮完整历史的优势；这也不是纯四层原则消融。
  原有 h2345/H4 运行与该零起点实验分别保留。

### Run: agentbench_zero_20260912（2026-09-12，原生 ALFWorld / DBBench 从零两轮）

用户要求前两个数据集从头跑两轮，每个训练集最多抽样 50 题。本轮只迭代 Life，
每域固定抽样 50 道 train，H2–H5 从空接口开始；不继承发布规则、零售插件或旧实验候选。
冻结 qwen3-4b、任务/评分代码、样本和预算，每轮全量复测同一批题，成功数严格提高才接受。
实验适配器与原生环境配置独立，未替换标准 worker。
溯源：[`EXPERIMENT.md`](experiments/agentbench_zero_20260912/EXPERIMENT.md)、
[`comparison.json`](experiments/agentbench_zero_20260912/comparison.json)，各域另有逐题 `paired_results.csv`。

| 数据集 | 空四层基线 | 第 1 轮 H2 | 第 2 轮 H2 + H4 | 最终相对基线 |
| --- | ---: | ---: | ---: | ---: |
| ALFWorld | 16/50 (32%) | 17/50 (34%) | **20/50 (40%)** | +8pp，4 胜 / 0 负 |
| DBBench | 23/50 (46%) | 25/50 (50%) | **28/50 (56%)** | +10pp，5 胜 / 0 负 |

- 四个候选均接受。ALFWorld：H2 可用动作校验，随后 H4 跟踪公开目标状态、提醒遗漏步骤和循环。
  DBBench：H2 修复 SQL 标识符/调用格式，随后 H4 核验空结果及更新后的状态。
- 最终 50 题 tokens 比基线减少 ALFWorld 21.6%、DBBench 55.5%；DB 第二轮比第一轮更贵，
  换取另外 3 题成功。三阶段共 32,646,708 计分 agent tokens，不含 proposer。
- 共 300 个正式计分 episode、另 2 个样本内链路检查；无最终基础设施错误，无 episode 重跑，未用 test。
- 原生 ALFWorld 0.4.2 / TextWorld 1.7.0、MySQL 8.0.46；各阶段环境一致，
  未证明与旧 Docker 镜像轨迹等价。预检修订、DB 第二轮计分前接口修复及工具审计单独归档。
- 同一训练样本上的自适应单 trial 改善，不等于泛化提升。描述性配对 p 为 0.125 / 0.0625；
  下一步应冻结候选做独立留出验证。本轮没有 Meta 对照，不能用于宣称超越 Meta。

### Supplement: agentbench_zero_20260912 公开版对照（2026-09-13）

沿用同一模型、原生环境、固定 50 题，补测公开原实现 H2345 全开及 H3+H5，
不迭代、不重新抽样。记录：[`PUBLISHED_COMPARISON.md`](experiments/agentbench_zero_20260912/PUBLISHED_COMPARISON.md)、
[`published_comparison.json`](experiments/agentbench_zero_20260912/published_comparison.json)。

| 数据集 | 空基线 | 从零 Life 两轮 | 公开 H2345 | 公开 H3+H5 |
| --- | ---: | ---: | ---: | ---: |
| ALFWorld | 16/50 | 20/50 | **44/50** | 15/50 |
| DBBench | 23/50 | 28/50 | 29/50 | **35/50** |

- 从零两轮版尚未超过各域最佳公开配置。ALFWorld 对公开 H2345 配对 0 胜 / 24 负；
  DBBench 对 H2345 为 4 胜 / 5 负，对 H3+H5 为 2 胜 / 9 负。
- ALFWorld 公开 H5 技能映射为空，H3+H5 是开关名称，实际未触发 H5；H3 描述修改已验证。
  公开 H4 有逐步状态提示和预算强制动作，而从零候选接口更受限；不是纯四层原则消融。
  DBBench 公开 H0 读取结构化任务类型/表定义，信息接口差异详见实验记录。
- 新增 200 个 episode、15,889,652 agent tokens，最终基础设施错误 0，无 episode 重跑。
  公开 H2345 在 ALFWorld 用量为 3,215,284 tokens，比新迭代版 9,590,031 更低。
- 后续应以各域最佳公开配置作为强基线；若继续研究从零搜索，应先审视接口限制、
  逐步状态指导及配置选择，再做独立留出验证。当前结果不能支持自动搜索已替代公开规则。

### Continuation: agentbench_zero_20260912 再迭代三轮（2026-09-13）

用户要求继续追赶公开版；两域各从第 2 轮接受版继续第 3–5 轮，仍固定原 50 题。
提案器只见本方法历史轨迹和公开总分目标，不见公开源码/公开运行轨迹，接口限制不变。
记录：[`CONTINUATION.md`](experiments/agentbench_zero_20260912/CONTINUATION.md)、
[`continued_comparison.json`](experiments/agentbench_zero_20260912/continued_comparison.json)。

| 数据集 | 追加前 | 第 3 轮 | 第 4 轮 | 第 5 轮 | 最终接受 | 公开最佳 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ALFWorld | 20/50 | 27/50 接受 | 26/50 拒绝 | 28/50 接受 | **28/50** | 44/50 H2345 |
| DBBench | 28/50 | 29/50 接受 | 30/50 接受 | 29/50 拒绝 | **30/50** | 35/50 H3+H5 |

- 最终接受文件：ALFWorld `candidate_05.py`，DBBench `candidate_04.py`，只保留为实验产物。
  相对追加前，ALFWorld 10 胜 / 2 负，净 +8；DBBench 3 胜 / 1 负，净 +2。
  仍未追上各域最佳公开配置，分别少 16 题和 5 题。
- ALFWorld 主要收益来自第 3 轮 H2 所需处理状态校验（7 胜 / 0 负）；第 4 轮扩大循环拦截退步，
  第 5 轮收紧停滞条件后净增 1 题，但仍有 3 题回退。DBBench 接受 H3 提交格式和 H4 输出解释提示。
- ALFWorld 最终 tokens 比追加前减少 13.8%；DBBench 增加 15.5%。新增 300 个计分 episode、
  25,451,161 agent tokens；无最终基础设施错误、无 episode 重跑。6 个提案及接口修复共记录
  359 次 proposer 模型请求，proposer tokens 无可靠计量；可见显式访问审计未发现越界请求。
- ALFWorld 第 4 轮计分前修复计数器/方法重名；DBBench 第 5 轮实际联合修改 H2/H4，
  不符合单一集中假设的严格解释，已明确记录且候选最终拒绝。
- DBBench 两道回退题在首轮请求完全相同、温度为 0 时已产生不同助手正文。
  因而不能将所有逐题变化归因于代码，尤其 1–2 题的小幅变化应复测。这里仍是自适应训练单 trial，
  未做留出验证；后续优先确认稳定性，并检验逐步状态指导和搜索接口限制的影响。

### 2026-09-13：公开 H2345 迁入当前格式，再迭代三轮（完成）

用户要求原始 H2345 的实现形式也升级。先前 `agentbench_native_20260913` 的原生 API 方案已停止，仅保留基线记录；本轮使用 [`agentbench_migrated_20260913`](experiments/agentbench_migrated_20260913/EXPERIMENT.md)。公开规则已迁入自包含 `Harness.h2/h3/h4/h5`，旧 Runtime 不参与本轮执行。H3 仅改工具描述，H5 只冷启动，H4 从公开回执更新状态并返回限长临时提示，H2 承接门控与正常预算内的动作修复。DB 不读取数据集类型或参考 SQL。

结果见 [`RESULTS.md`](experiments/agentbench_migrated_20260913/RESULTS.md) 与 [`comparison.json`](experiments/agentbench_migrated_20260913/comparison.json)。固定原 50 道训练题，每域三轮，严格分数增加才接受：

| 数据集 | 迁移基线 | 第 1 轮 | 第 2 轮 | 第 3 轮/最终 | 历史 H2345 | 历史最佳 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ALFWorld | 43/50 | 47/50 | 48/50 | **49/50** | 44/50 | 44/50 |
| DBBench | 24/50 | 25/50 | 27/50 | **29/50** | 29/50 | 35/50 H3+H5 |

- 六轮均接受。ALFWorld 依次补充关闭容器搜索、处理状态恢复、目的容器打开指导（均 H4），相对迁移基线 6 胜 / 0 负，相对历史 H2345 5 胜 / 0 负；tokens 比迁移基线少 16.2%。
- DB 依次修正按组统计的单值误判、表头 Total 的求和误判、多行元组提交格式（均 H2），相对迁移基线 5 胜 / 0 负；与历史 H2345 同分但为 4 胜 / 4 负，与历史 H3+H5 为 1 胜 / 7 负；tokens 比迁移基线少 9.0%。
- 两域最终文件均为各域 `candidate_03.py`，四个公开 hook、公开状态更新和门控回归检查通过。旧 Runtime 使用 fail-fast 占位，未恢复到本轮执行链。标准 worker 的部署未改变。
- DB 初版迁移 23/50，修正 `Correct ...` 更新指令的公开类型解析后正式基线 24/50；初版代码与 50 个诊断 episode 单独保留。当前目录共 450 个 live episode、12,363,632 agent tokens，无最终基础设施错误；旧形式中止实验的另 100 个 baseline episode 不混计。
- 冻结源代码/数据、候选、样本、提案器只读输入哈希核对通过，见 `integrity.json`；六次提案共 202 次可见模型请求，未发生格式修复。DB 第 3 轮有一次父目录 Glob 和一次不可用 Bash 请求，匹配执行结果均为错误，未见成功越界读取或执行；详见实验文档及 audit 文件。
- 新旧接口和可见信息有明确差异；这些是训练集上的自适应单 trial，尚无独立留出验证。DB 相同首轮请求出现不同 SQL，见 `db_nondeterminism.json`，不能将所有逐题变化归因于代码。ALF 剩余一题仍计为失败，未验证其是否不可解。

### 2026-09-13：冻结最优版的测试集泛化评估（完成）

用户授权比较发布版本与三轮训练最优版。测试前冻结两域 candidate_03.py；ALFWorld 全部 new_std 109 个 valid_unseen 场景，DBBench 全部 standard 300 题，与本轮 50 道训练题的游戏路径/规范化题面交集为零。固定原模型与预算，单 trial，无 proposer、无测试调参。

结果：[`RESULTS.md`](experiments/agentbench_test_20260913/RESULTS.md)、[`ANALYSIS.md`](experiments/agentbench_test_20260913/ANALYSIS.md)、[`comparison.json`](experiments/agentbench_test_20260913/comparison.json)。

| 数据集 | 发布 H2345 | 三轮最优 | 变化 | 新版胜 / 负 |
| --- | ---: | ---: | ---: | ---: |
| ALFWorld | 100/109（91.7%） | **103/109（94.5%）** | +2.8 个百分点 | 3 / 0 |
| DBBench | **189/300（63.0%）** | 148/300（49.3%） | -13.7 个百分点 | 10 / 51 |

DB 预先指定的发布 H3+H5 补充对照为 187/300（62.3%），也高于新版。ALF 仅 3 个配对胜例，描述性 McNemar p=0.25，尚不能确认稳定优势；DB 对 H2345 p≈9.62e-8，单 trial 下观察到的下降明显。新版 tokens 相对 H2345 分别少 13.6%、3.5%。

DB 下降集中于 INSERT（54→33/100）和 UPDATE（86→72/100）。只读诊断发现，迁移后的公开题面解析把 71 道 INSERT、21 道 UPDATE 判为查询类；发布版仍使用结构化类型和原 schema 样例。此处同时包含形式迁移与三轮改进，不能把差异全归因于迭代或过拟合。具体诊断见 db_diagnostic.json；未修改候选验证因果。训练仅 6 道 INSERT，测试有 100 道，覆盖与分布变化也是重要限制。

共 1118 个 live episode、14,561,347 agent tokens；无环境错误或重跑。候选、样本、任务代码与数据哈希一致，见 integrity.json。当前结论是 **ALF 有小幅正向泛化迹象，DB 未保住发布版能力，整体尚不能称为超越**。std 已用于此次诊断，后续若据此修改方法，须另设独立测试；不应继续拿本次 std 调参后宣称全新泛化。

### 2026-09-13：ALFWorld 发布替换与 DB 新批广覆盖训练

用户授权 ALFWorld 最新版替换发布版，已完成并重启空闲的标准 worker：正式 Harness 与三轮 candidate_03.py 相同，旧 Runtime 移出调用链，标准 H2345 全开。新 Task/Session/客户端完整历史处理已集成；109 条环境回放的 1540 个模型输入和 103/109 结果相等，实际 HTTP 完整一题回放通过。记录见 [`RELEASE.md`](experiments/alfworld_release_20260913/RELEASE.md)。

DB 新实验 [`dbbench_broad_20260913`](experiments/dbbench_broad_20260913/EXPERIMENT.md) 已启动：17 INSERT + 17 UPDATE + 16 道多种查询，按公开表达/长度及词汇差异扩大覆盖，排除旧 50 题和 std/dev，仍固定 50 题。起点为上一轮 candidate_03.py，新批基线 19/50；三轮候选依次 19/50（拒绝）、19/50（拒绝）、20/50（接受）。最终 candidate_03.py 相对新基线 1 胜 / 0 负，收益来自 SELECT 提交文本的内嵌引号恢复；INSERT 仍 5/17、UPDATE 仍 12/17，写操作短板尚未改善。共 200 个计分 episode、1,138,155 agent tokens，零基础设施错误；三次提案 154 次可见模型请求，无显式越界请求，无格式修复，冻结输入哈希通过。结果见 [`RESULTS.md`](experiments/dbbench_broad_20260913/RESULTS.md)。未部署 DB 候选，未新增独立测试；不能将这批 20/50 与旧批 29/50 直接比较。std 已用于诊断，不自动拿它再次充当独立测试。

关于更大训练集：当前 AgentBench loop 仍导出全部历史轨迹供提案器自行检索，没有自动的失败统计/代表案例索引。要扩到几百上千题，应分开评估预算和阅读预算，先汇总全量信号、限量选择代表失败及成功对照，再按需读轨迹；必要时用分阶段评估控制执行成本。本轮没有在运行中改变协议。后续设计建议见 [`EVIDENCE_SCALING.md`](EVIDENCE_SCALING.md)，尚未实现自动证据组织。

### 2026-09-13：四步证据流程已接入后续 AgentBench 迭代

新增 [`agentbench_loop.py`](agentbench_loop.py)：全池统计、限量代表失败与成功对照、宿主按需返回轨迹、筛查后全池确认。默认 12 案例 / 24k 初始字符 / 80k 整轮证据字符、两批检索、16 题筛查、每三轮全量探索；只接受全池成功数严格增加。支持预注册独立留出并冻结后测起点/最终版。训练池不再写死 50；旧实验与四层设计保持原样，当前尚无新 live 成绩。详见 [`AGENTBENCH_ITERATION.md`](AGENTBENCH_ITERATION.md)。

### 2026-09-13：新证据流程三轮实跑完成，未提高 DB 成功数

从广覆盖 50 题最优 candidate_03（20/50）重新基线，三轮结果：20/50 拒绝；16 题筛查 0 胜 / 1 负拒绝（无全量分数）；20/50 拒绝。最终仍用起点，INSERT 5/17、UPDATE 12/17 均未提高。正式运行 166 个 episode、1,127,614 agent tokens，零基础设施错误/transport 重试。三轮提案 162 次模型请求，独立证据分别 79,055 / 72,614 / 74,950 字符，累计实际证据读取 525,002 字符。第一轮一次 Bash 请求返回 error，其余无显式越界请求；只读输入与全部冻结哈希核对通过。

实跑先暴露并修复“组检索回合越界中断”和“最终 JSON 包装字符漏计导致整批回退”两个问题；旧尝试保留，不改写协议。包含这两次排错基线，总计 266 个 episode、1,707,466 agent tokens。修复后 17 项测试通过。完整记录见 [`dbbench_evidence_20260913_v3/RESULTS.md`](experiments/dbbench_evidence_20260913_v3/RESULTS.md)。本轮证明流程可跑通、预算和筛查生效，不能声称方法成绩提高；未测试独立留出，也未替换 DB 发布版。

### 2026-09-13：原始 Meta 从发布 H2345 各迭代六轮，停滞换 50 题

实验 [`meta_agentbench_six_20260913`](experiments/meta_agentbench_six_20260913/EXPERIMENT.md) 已完成。起点为原始发布 H2345 快照（ALF 替换发布前、DB 迁移前），直接使用克隆仓库的原始 Meta loop/skill，移植到 AgentBench 原生接口；无 Life 四层限制、无有界证据阅读或筛查。每轮一候选、真实评测 50 道训练题，严格同批提高才接受；未提高则下轮换互斥新批，并重测 incumbent。

| 轮次 | ALFWorld：批次、同批起点→候选 | DBBench：批次、同批起点→候选 |
|---|---|---|
| 1 | 批 0，47→49，接受 | 批 0，31→34，接受 |
| 2 | 批 0，49→49，拒绝并换题 | 批 0，34→36，接受 |
| 3 | 批 1，45→48，接受 | 批 0，36→36，拒绝并换题 |
| 4 | 批 1，48→49，接受 | 批 1，27→28，接受 |
| 5 | 批 1，49→50，接受 | 批 1，28→28，拒绝并换题 |
| 6 | 批 1，50→50，拒绝 | 批 2，29→30，接受 |

以上均为 /50。ALF 共 100 道不同训练题，最终保留 `look_at_frontier.py`；DB 共 150 道，保留 `tool_call_desync_recovery.py`。总计 850 个计分 episode、24,094,545 solver tokens，可见 proposer 请求 1,597 次。ALF 第六轮 `info_first_search.py` 虽持平未接受，但同批 tokens 降低约 64%、模型回合降低约 36%，另保留为效率候选。DB 第六轮的唯一新增成功来自修复上一保留候选的控制字符解析 KeyError；原错误仍计 0 分，没有重跑抹去。

完整分数、候选机制与 CSV 见 [`RESULTS.md`](experiments/meta_agentbench_six_20260913/RESULTS.md)。逐题分数/接受/轮换、原始 frontier、冻结输入/源码/数据核验通过。工具审计发现 ALF Bash 曾列父目录、向父目录复制候选描述、向 `/tmp` 写离线输出，不能声称完全无越界；详细记录见 [`AUDIT.md`](experiments/meta_agentbench_six_20260913/AUDIT.md)。没有独立测试、没有部署。不同批分数不能直接相减，本轮也不是与 Life 同题同预算的对照，因此不能据此宣布任一方法整体胜出。

### 2026-09-13：Meta 六轮候选标准集评测完成

用户授权补测后，在测试前冻结 ALF 第五轮保留版、第六轮效率候选及 DB 第六轮保留版；全量 ALF new_std 109 题、DB standard 300 题，原始 H2345 同时重新实跑，模型/回合预算不变，无 proposer。

| 数据集 | 本次原始 H2345 | Meta 保留版 | 另测效率候选 |
|---|---:|---:|---:|
| ALFWorld | 100/109（91.7%） | 103/109（94.5%），3 胜 / 0 负 | 第六轮 104/109（95.4%），4 胜 / 0 负 |
| DBBench | 186/300（62.0%） | 190/300（63.3%），9 胜 / 5 负 | — |

ALF 第六轮相对第五轮多成功 1 题、无回退，tokens 降 21.2%、模型回合降 12.1%；训练时效率收益部分迁移。DB INSERT 54→57/100，UPDATE 86/100 不变，查询 46→47/100，增益较小。历史 ALF Life 发布版是 103/109，但本轮未同期重测；DB 历史原始基线为 189/300，本次少 3 题，不能忽略单次波动。

共 927 个计分 episode、17,844,204 成功响应 solver tokens。12 次模型 HTTP 500 经原有请求重试机制恢复，0 个未恢复基础设施错误、0 个候选异常；冻结哈希和逐题计分核验通过。完整分数、类型拆分、CSV 与解释见 [`meta_agentbench_test_20260913/RESULTS.md`](experiments/meta_agentbench_test_20260913/RESULTS.md)。标准集此前已评估/诊断，不是全新独立留出；未根据测试继续迭代，未替换发布版本。

### 2026-09-13：Meta 标准集最优版已替换本地发布 Harness

用户授权发布后，ALFWorld 使用第六轮 `info_first_search`，DBBench 使用第六轮 `tool_call_desync_recovery`。DB 正式文件与冻结评测候选字节一致；ALF 保留候选 Runtime 全文，并追加当前统一 `Harness.h2/h3/h4/h5` 适配层，没有退回旧 Task。旧发布文件、哈希、重启状态及验证见 [`meta_release_20260913/RELEASE.md`](experiments/meta_release_20260913/RELEASE.md)。15021/15022 新 worker 均已注册 ALIVE，controller→worker 冒烟通过。

ALF 109 条冻结回复经正式四层入口回放仍为 104/109、成功位无差异；1126/1290 个模型输入逐字一致，剩余差异限于 11 个 episode 的多提示及无工具/阻断分支。live HTTP 完整回放一条成功任务时 5/5 个输入逐字一致并正常结束。四层适配后尚未二次全量实测，记录中没有把原候选 104/109 偷换成适配层的新实测分数。

Meta 看似更强的主要原因已拆分记录在 [`WHY_META_LOOKED_STRONGER.md`](experiments/meta_release_20260913/WHY_META_LOOKED_STRONGER.md)：ALF 对 Life 历史发布版实际仅 1/109；DB 的大差距主要来自 Life 统一接口迁移丢失结构化任务类型/样例入口。Meta 同时使用约 8 倍的 proposer 请求、两角色委派、完整轨迹/脚本分析、更宽代码搜索空间、更多训练批次以及每轮全量评测。应吸收这些诊断能力与停滞换批策略，同时保留四层发布边界、证据隔离与严格同批接受。

### 2026-09-14：七个 benchmark 统一到当前迭代形式

TauBench airline/retail/telecom 保持 H2/H3/H4 注册表、H5 技能注册表及 `register()` 插件入口；AgentBench ALFWorld/DBBench/WebShop/OS Interaction 的正式 Task 均改走 `Harness.h2/h3/h4/h5` 与共享 `FourHookSession`。DB 由宿主提前绑定公开任务类型和表结构，拒绝参考 SQL、标签和答案字段；H2 修复动作与公开历史分开传递，保留旧 Runtime 的消息时序。

无模型冻结回放显示：ALFWorld 仍为 **104/109**，逐题结果、状态、轮数和归一化模型输入均一致；DBBench 仍为 **190/300**，逐题结果、状态、轮数、归一化模型输入和工具输入均一致。ALF 有 21 次原始消息边界变化，来自把旧 Task 的两条相邻 H4 提示合成当前契约允许的一条，文字与顺序不变；DB 有 4 次原始字符串变化，仅为随机临时数据库名。迁移同时修复 ALF 的 H2 后二次模糊动作映射，以及容器循环提示依赖集合遍历顺序的问题。

部署检查还发现旧 `dbbench-std` 配置实际上仍把 Harness 整体及 H2/H4 关闭；正式配置与本机 native 配置均已改为 H2–H5 全开。ALF/DB 两个空闲 worker 已重启并加载迁移代码。

Tau 三域完成 24 种开关选择和无操作插件状态等价检查。WebShop 在 6 类代表指令、OS Interaction 在本地 1,000 条训练题上的公开上下文解析与 H5 输出均和旧 Runtime 一致；由于 WebShop 缺可用商品索引、OS 在无 Docker 虚拟机上无法提供原隔离语义，本轮没有为这两域声称新的分数等价。完整记录见 [`MIGRATION.md`](experiments/current_format_migration_20260913/MIGRATION.md)。后续 ALF/DB 迭代必须从 [`current_agentbench_source_20260914`](experiments/current_agentbench_source_20260914/SOURCE.md) 新建 run，历史 source 不再作为当前起点。

## Usage

Prerequisites: TauBench env ready (`uv sync`), `qoder` on PATH, and model
endpoints. For the current pilot setup:

- Agent (frozen): local Qwen3-4B — `bash deploy/start_vllm.sh` launches 8
  vLLM instances (GPUs 2-5, 2 per GPU) behind a round-robin proxy at
  `http://127.0.0.1:8400/v1`; `deploy/stop_vllm.sh` tears it down.
- User simulator: `deepseek-flash` through the configured OpenAI-compatible
  ModelRouter route.
- `source deploy/env.sh` exports `AGENT_API_BASE`/`AGENT_API_KEY`/
  `USER_API_BASE`/`OPENAI_API_KEY`; the flag values are in that file's
  `AGENT_FLAGS`/`USER_FLAGS` comments.

```bash
cd Life-Harness/meta

# cheap plumbing smoke test (no LLM calls, no proposer):
META_MOCK_EVAL=1 python meta_harness.py --run-name smoke --fresh \
    --iterations 1 --domains airline --num-tasks 2 --mock-proposer

# real link smoke test: 2 airline tasks, anchors only:
source deploy/env.sh
python meta_harness.py --run-name linkcheck --fresh --iterations 0 \
    --domains airline --num-tasks 2 \
    --agent-llm openai/qwen3-4b \
    --user-llm openai/deepseek-flash

# real pilot, airline only, small search subset:
python meta_harness.py --run-name pilot_airline --fresh \
    --iterations 5 --domains airline --num-tasks 15 \
    --agent-llm openai/qwen3-4b \
    --user-llm openai/deepseek-flash

# full run:
python meta_harness.py --run-name tau2_full --iterations 10

# finalize on the held-out test split (freezes the run):
python meta_harness.py --run-name tau2_full --test
```

Interrupted runs resume automatically (completed evals are cached as
`val.json`); `--fresh` restarts a run name from scratch.

Useful cost knobs: `--domains`, `--num-tasks`, `--candidates-per-iter`,
`--trials` (>1 enables pass@k), `--max-steps`, `--agent-llm`, `--user-llm`,
`--proposer-model`, `--propose-timeout`, `--eval-timeout`.

## 新版本 Life-Harness 与 Meta-Harness baseline

2026-09-14 版本把可迭代对象统一为可冻结的单文件候选，同时保留 Life 的 H2–H5
设计边界。TauBench 的 airline/retail/telecom 通过注册表和 `register()` 加载候选；
AgentBench 的 ALFWorld/DBBench/WebShop/OS Interaction 通过共享 `FourHookSession`
调用 `Harness.h2/h3/h4/h5`。ALFWorld 和 DBBench 另有可直接创建新 run 的冻结源；
WebShop 的 100k 商品、Lucene 索引和原生 worker 已补齐，可以运行发布版本评测，但尚未
制作独立的冻结迭代源；OS Interaction 仍等待满足原语义的隔离环境。

先在仓库根目录配置环境。`env.sh` 含本机凭据且不会被 git 跟踪：

```bash
cd Life-Harness
cd TauBench && uv sync && cd ..
cp meta/deploy/env.example.sh meta/deploy/env.sh
# 编辑 meta/deploy/env.sh 后：
source meta/deploy/env.sh
```

TauBench 上运行新 Life 方法。默认从当前发布 H2–H5 开始；每轮先筛查，候选只有在
完整固定训练池成功数严格增加后才接受。`--fresh` 只在首次创建 run 时使用：

```bash
python meta/life_loop.py \
  --run-name life_retail_current --fresh \
  --iterations 3 --domains retail --num-tasks 50 \
  --accept-on full --screen-sentry 8 \
  --agent-llm "$LIFE_AGENT_MODEL" --user-llm "$LIFE_USER_MODEL" \
  --proposer-model DeepSeek-Flash
```

AgentBench 上从当前冻结版本运行新 Life 方法：

```bash
python meta/agentbench_loop.py \
  --source meta/experiments/current_agentbench_source_20260914 \
  --run-dir meta/runs_agentbench/life_alfworld_current \
  --domain alfworld \
  --baseline meta/experiments/current_agentbench_source_20260914/alfworld/baseline.py \
  --rounds 3 --screen-size 16 --proposer-model DeepSeek-Flash

python meta/agentbench_loop.py \
  --source meta/experiments/current_agentbench_source_20260914 \
  --run-dir meta/runs_agentbench/life_dbbench_current \
  --domain dbbench \
  --baseline meta/experiments/current_agentbench_source_20260914/dbbench/baseline.py \
  --rounds 3 --screen-size 16 --proposer-model DeepSeek-Flash
```

TauBench 上运行自由形式 Meta-Harness baseline。公平的从零对照需要 Life 和 Meta
双方都带 `--from-scratch`，并保持模型、任务数、split、trial、并发与回合预算一致：

```bash
python meta/meta_harness.py \
  --run-name meta_retail_from_scratch --fresh --from-scratch \
  --iterations 3 --candidates-per-iter 1 \
  --domains retail --num-tasks 50 \
  --agent-llm "$LIFE_AGENT_MODEL" --user-llm "$LIFE_USER_MODEL" \
  --proposer-model DeepSeek-Flash
```

去掉 `--from-scratch` 时，Meta run 会同时评测无 Harness 和发布 H2–H5 两个 anchor，
并从发布版本继续自由形式迭代。断点续跑时复用同一命令并移除 `--fresh`。冻结后只做
一次 held-out 评测；应重复训练命令中的模型和评测参数，再把 `--fresh` 换成 `--test`。
AgentBench 的无 Docker 服务配置见
[`AgentBench/NATIVE_ENVIRONMENT.md`](../AgentBench/NATIVE_ENVIRONMENT.md)，完整迁移证据见
[`current_format_migration_20260913/MIGRATION.md`](experiments/current_format_migration_20260913/MIGRATION.md)。
