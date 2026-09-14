# 实验审计与解释边界

本记录保留实际执行偏差，不把分数完整性检查等同于严格沙箱合规证明。

## 候选错误计分

DB 第三批基线的 task 4050 在已接受候选 `tool_call_escape_fidelity.py` 中触发 `KeyError`。原生任务包装最初将它归为基础设施错误。依据候选文件 traceback 改为候选失败，成功位仍为 0；没有重跑该题或由宿主修改候选。原始 episode、summary、归因及哈希见 [failure_attribution/dbbench_batch02_4050](failure_attribution/dbbench_batch02_4050)。第六轮 Meta 自行提出 `tool_call_desync_recovery.py`，同批 29/50 → 30/50，唯一新增成功为此题。错误归因记录作为公开训练证据提供给提案程序。

## 提案工具边界

可见日志统计见 [proposer_audit.json](proposer_audit.json)，逐轮 Bash 命令与执行状态见各域 `audit/shell_NN.json`。仅扫描 Read/Grep 路径会漏掉 Bash 内的路径操作，故不能据前者声称零越界。

- ALF 第一轮有一次指向工作区父目录的 Grep 请求被拒绝；但 Bash 成功列出了父目录，并将工作区内的 `pending_eval.json` 复制到父目录后读回。父目录副本仍保留，未作为宿主评分输入。
- ALF 第五轮将自己生成的离线检查输出写到 `/tmp/verify_laf_out.txt` 和 `/tmp/verify_laf_out_final.txt`。这是工作区外临时输出，不是隐藏数据来源。
- 还存在被拒绝的父目录查看和 `/tmp` 重定向尝试，逐项执行状态见 shell 日志。工具层边界并非严格封闭。
- 已审阅的上述命令涉及目录项、提案描述及离线输出，未见通过这些命令读取隐藏答案。这个结论限于可见记录，不是任意代码行为的形式化证明。

因此，本次分数可作为探索性同批实验记录，但不应描述为“所有提案完全无越界”的严格隔离实验。未来复跑应在启动前统一实施文件系统隔离，不能在本次中途改规则后掩盖已发生的偏差。

## 可比较范围

- 每轮 50 题，在当前同批 incumbent 与候选之间比较；只有成功数严格增加才接受。换批后重新测 incumbent，分数不可跨批直接相减。
- 同批已完成的 incumbent 结果复用；单 trial、temperature 0 仍不代表统计显著或完全确定性。
- 换批后未复测更早批次；最终保留候选不保证保有旧批所有成功。ALF 达到某批 50/50 也不代表整个训练集满分。
- 未进行独立留出测试、未部署候选，也不是与 Life 流程使用同一批题的受控比较。
- 原始 Meta loop/skill 保留，但执行环境、模型、接口已移植；不等于原论文 TB2/Opus 配置的复现。
- 求解器 tokens 来自实际计分记录；proposer 的 provider token 计数不可用，可见模型请求数包含日志中可见的委派请求，不能当作完整 token 成本。
