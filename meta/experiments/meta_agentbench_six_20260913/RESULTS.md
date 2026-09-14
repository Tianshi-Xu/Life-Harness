# Meta-Harness：六轮与停滞换题

起点为原始发布 H2345；每批 50 道训练题。同批严格提高才接受；未提高则下一轮换题并复测 incumbent。不同批分数不能直接比较。

## alfworld

状态：complete；当前保留：look_at_frontier.py。

| 轮次 | 批次 | 同批起点 | 候选 | 胜/负 | 接受 | 下轮换题 |
|---|---|---:|---:|---:|---|---|
| 1 | 0 | 47/50 | 49/50 | 2/0 | True | False |
| 2 | 0 | 49/50 | 49/50 | 0/0 | False | True |
| 3 | 1 | 45/50 | 48/50 | 3/0 | True | False |
| 4 | 1 | 48/50 | 49/50 | 1/0 | True | False |
| 5 | 1 | 49/50 | 50/50 | 1/0 | True | False |
| 6 | 1 | 50/50 | 50/50 | 0/0 | False | False |

已完成计分 400 episodes，22,170,151 agent tokens（不含 proposer）。

## dbbench

状态：complete；当前保留：tool_call_desync_recovery.py。

| 轮次 | 批次 | 同批起点 | 候选 | 胜/负 | 接受 | 下轮换题 |
|---|---|---:|---:|---:|---|---|
| 1 | 0 | 31/50 | 34/50 | 3/0 | True | False |
| 2 | 0 | 34/50 | 36/50 | 2/0 | True | False |
| 3 | 0 | 36/50 | 36/50 | 0/0 | False | True |
| 4 | 1 | 27/50 | 28/50 | 1/0 | True | False |
| 5 | 1 | 28/50 | 28/50 | 0/0 | False | True |
| 6 | 2 | 29/50 | 30/50 | 1/0 | True | False |

已完成计分 450 episodes，1,924,394 agent tokens（不含 proposer）。

未测独立测试集、未部署候选；最终不同批次分数不代表固定测试集泛化或六轮累计净提升。原始 Meta loop 的单轮日志另存各域 workspace/logs/round_NN。

## 完成核验与成本

两域均完成六轮，每轮候选实际评测 50 题。ALF 使用两批、100 道不同训练题；DB 使用三批、150 道不同训练题。加上每批 incumbent 基线，共 850 个计分 episode、24,094,545 solver tokens。已有同批 incumbent 成绩复用，不重复评测；没有额外第七轮。

[integrity.json](integrity.json) 通过：12 轮逐题胜负、严格接受规则、停滞换批、各批题目互斥、原始 Meta frontier、冻结提案输入、原仓库/驱动/任务快照/数据哈希。前置回放使用历史训练轨迹验证 ALF 103 / DB 16 个模型输入，均与原生发布版本相同；三个离线适配器测试通过。最终核验没有调用模型。

可见 proposer 请求 ALF 777、DB 820，共 1,597 次；每轮均有一个分析 Agent 和一个实现 Agent。原始 provider token 统计不可用，不把它写成零成本。完整工具审计与执行偏差见 [AUDIT.md](AUDIT.md)：包括一条候选异常按 0 分计入，以及 ALF 提案的父目录/临时文件操作；不能将本次称为完全无越界的隔离实验。

## 候选机制与保留结果

以下机制是代码改动概括；预测收益以实际同批分数为准。

| 轮次 | ALF 候选及改动 | DB 候选及改动 |
|---|---|---|
| 1 | `grounded_placement`：对象/地点完整名称匹配与放置完成状态核验；接受 | `terminal_evidence_gate`：终止答案的查询证据门控；接受 |
| 2 | `open_obligation`：搜索期间打开容器的指导；持平拒绝 | `insert_literal_fidelity`：INSERT 字面值解析与格式保真；接受 |
| 3 | `no_progress_redirect`：无进展循环检测与行动转向；接受 | `escape_literal_fidelity`：转义恢复；持平拒绝 |
| 4 | `text_action_rescue`：将明确对应可用动作的纯文本回复恢复为工具调用；接受 | `tool_call_escape_fidelity`：损坏的工具调用 JSON/XML 恢复；接受 |
| 5 | `look_at_frontier`：寻找光源时提示尚未访问的位置；接受 | `grounded_evidence_gate`：将提交答案与公开查询结果匹配；持平拒绝 |
| 6 | `info_first_search`：优先搜索可直接观察的表面，并提示打开未搜索容器；持平拒绝 | `tool_call_desync_recovery`：修复控制字符 KeyError，并恢复引号失配的调用；接受 |

正式保留：ALF [`look_at_frontier.py`](alfworld/workspace/agents/look_at_frontier.py)，DB [`tool_call_desync_recovery.py`](dbbench/workspace/agents/tool_call_desync_recovery.py)。文件只作为实验产物，未替换发布版本。

ALF 第六轮虽不满足严格成功数增加，但同批仍为 50/50，tokens 从 2,698,747 降到 971,447（约 −64.0%），模型回合从 695 降到 442（约 −36.4%）。[`info_first_search.py`](alfworld/workspace/agents/info_first_search.py) 保留为效率候选，未事后更改接受规则。这是单次同批测量，尚未验证其他批次与测试集。

DB 第六轮唯一新增成功是 task 4050：该题原先触发候选代码异常；新候选成功完成，其余 49 题成功位不变。未通过重新抽样或重跑消除原始错误。

这些结果说明原始发布 H2345 仍有可被 Meta 找到的改进空间，尤其是动作/工具调用解析、证据门控和搜索引导。当前实验没有同题同预算的 Life 对照，也没有留出评估，因此不能据此判断 Meta 或 Life 谁整体更强。
