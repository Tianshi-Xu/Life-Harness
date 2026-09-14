# 当前 H2345 的有界证据迭代流程

入口是 `agentbench_loop.py`，支持 ALFWorld / DBBench 的当前 `Harness.h2/h3/h4/h5` 接口。它替代**后续 AgentBench 实验**中手工复制固定 50 题 loop 的做法；旧实验目录和 TauBench 的 `life_loop.py` 保留原协议。DBBench 在四层 Hook 之外由宿主绑定公开题面、任务类型和表结构，候选不会看到参考 SQL、标签或期望答案；消息级 H3、H4 历史方式和 H5 前缀也与正式 Task 使用同一语义。

2026-09-14 之后的新实验必须从 `experiments/current_agentbench_source_20260914` 建立。该 source 冻结的是迁移后的 Task、四层桥和正式 Harness，并设置 `current_task_snapshot=true`；旧实验 source 中的 Task/`legacy_disabled.py` 只用于历史复现。实现哈希会阻止已有 run 静默切换到新执行语义。

## 四步

1. 基线对预先固定的训练任务池做真实全量评估。统计成功数、状态、公开指令开头类别、错误信号、hook 次数、turns 和 tokens。信号可重叠，`unclassified` 不冒充根因。正确 SQL、内部评分详情、reward 对象不进入证据包。
2. 提案器先读 `evidence/index.json`。默认最多 12 个案例、24,000 字符，交替考虑大组和小组，配公开题面词汇相似的成功案例，并留随机抽查位置。每个案例的预览明确截短；完整轨迹保留在宿主目录。证据携带 candidate / manifest / evaluator 哈希。
3. 默认两个检索阶段。提案器只写 `retrieval_request.json`，宿主按任务 ID 或失败组返回整条轨迹或指定**模型回合**片段。整轮证据暴露上限默认 80,000 字符；超限拒绝并要求更窄范围，不静默截断详情。账本与检索日志可恢复。模型回合由执行历史和调用记录共同校验，无法对应时禁用回合切片。
4. 候选先通过全部基线历史的无模型四层契约检查，再真实评估默认 16 题筛查。筛查净胜负 >= 0 才全量确认；每第三轮即使筛查下降也做全量探索，以免持续漏掉长尾收益。只有固定全量训练池成功数严格增加才接受。筛查成绩不会混入主线分数。全量阶段复用同一候选、同一配置刚完成的筛查 episode，不重复计费，不复用旧候选成绩。

这里限制的是**不同证据的可见字符数**，不是 Qoder 重复读取次数、模型上下文或全部推理 token 的硬上限。源码、设计准则、最多五轮的简短结果记录不计入轨迹证据预算。更大池的全量评估和契约回放成本仍会增加。尚未证明这一流程能提高真实模型成绩，需要之后在固定预算下与旧流程比较。

## 启动与恢复

从已有、确认只含训练任务的冻结实验导入任务快照和 manifest；显式选择起点。任务量来自 manifest 的 indices，不再写死 50。当前示例沿用上一批广覆盖 50 题，未扩大用户此前的训练上限。

```bash
cd /mnt/data2/xts/harness/Life-Harness
python meta/agentbench_loop.py \
  --source meta/experiments/dbbench_broad_20260913 \
  --run-dir meta/experiments/dbbench_evidence_next \
  --domain dbbench \
  --baseline meta/experiments/dbbench_broad_20260913/dbbench/candidate_03.py \
  --rounds 3 --screen-size 16 --prepare-only
```

去掉 `--prepare-only` 才会启动真实模型评估和提案。相同参数再次运行可恢复；候选、模型、样本、源码快照、迭代实现或预算变化会拒绝混用，须新建 run。不要改写历史实验 manifest 来改变题量，应准备一个具有独立抽样溯源的新 source。入口不会自行把已用于诊断的 std 认定为独立测试。

可选：首次准备时提供 `--heldout-indices /path/to/unused_ids.json`，注册**同一个底层数据集**里从未用于开发的额外留出任务；校验非空、唯一、与训练不重叠。其原始数据来源与独立性由实验设计者确认，索引不重叠本身无法证明跨实验无泄漏。此选项不是切换官方 std split 的开关。完成搜索后用相同参数加 `--finalize`，先冻结候选，再单独评估起点和最终版，之后禁止继续提案。留出轨迹不会导出到 proposer 工作区。

## 验证

`python -m unittest discover -s meta/tests` 覆盖公私字段隔离、模型回合映射、1,000 题预算上限、检索越界/过大请求/重入记账、配对筛查、周期探索、严格接受、同候选筛查续算、缓存错配拒绝、断点恢复、留出隔离和冻结。

真实 DB 50 题历史的离线证据检查在 `experiments/evidence_pipeline_checks_20260913/checks.json`；无新增模型调用。这验证证据组织可用，不是新的训练成绩。

## 首次三轮实跑

2026-09-13 实跑完成，起点 20/50，第一轮 20/50 拒绝，第二轮筛查 0 胜 / 1 负拒绝，第三轮 20/50 拒绝。保留原起点。见 [结果](experiments/dbbench_evidence_20260913_v3/RESULTS.md)。实跑修复了组检索回合越界中断及预算整批回退两个问题，前两个冻结尝试与成本已保留；目前 17 项测试通过。
