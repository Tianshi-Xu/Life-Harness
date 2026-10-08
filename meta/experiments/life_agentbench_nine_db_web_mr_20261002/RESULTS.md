# DBBench / WebShop：Life 六轮续跑到九轮

两域各追加三轮，API proposer 为 `deepseek-v4-flash`，solver 沿用 Qwen3-4B。
未追加 ALFWorld；原五轮、六轮、Meta 实验保持不变。

## 最终测试集

| 环境 | 六轮 test | 九轮 test | 九轮 − 六轮 | 发布版参照 | 九轮 − 发布版 |
|---|---:|---:|---:|---:|---:|
| DBBench | 172/300 = 57.33% | 172/300 = 57.33% | 0.00 pp | 65.0% | −7.67 pp |
| WebShop | 71/200 = 35.50% | 69/200 = 34.50% | −1.00 pp | 45.5% | −11.00 pp |

DB 三轮均未接受新机制，accepted snapshot 与六轮相同，因此复用原测试并逐题核验来源。
WebShop 接受了第 7 轮机制，第 8、9 轮拒绝；在两域训练封存后新测完整 200 题。
本次有效训练及新测 WebShop test 的 infra_errors 均为 0。发布版数值为用户提供参照，
不是本次同期重测。六轮 WebShop 的一个历史基础设施错误仍按 0 计入原 200 题分母。

结论：本次追加三轮未改善最终测试表现，未追平发布版。WebShop 第 7 轮的训练提升
没有在这次固定测试集测量中表现为提升。此前 test 已观察，且仅单次测量，不代表
统计显著性，也不能将分差全部归因于新增规则。

## 第 7–9 轮训练决策

每次同批评测 50 题；严格提高才接受。不同 batch 的分数不能拼成连续改善曲线。

| 环境 | 轮次 | batch | incumbent → candidate | 接受 |
|---|---:|---:|---:|---|
| DBBench | 7 | 2 | 34/50 → 34/50 | 否，持平 |
| DBBench | 8 | 3 | 30/50 → 28/50 | 否，退步 |
| DBBench | 9 | 4 | 31/50 → 27/50 | 否，退步 |
| WebShop | 7 | 3 | 9/50 → 11/50 | 是 |
| WebShop | 8 | 3 | 11/50 → 10/50 | 否，退步 |
| WebShop | 9 | 4 | 18/50 → 11/50 | 否，退步 |

新增有效 train 共 500 episodes（各域 250）；新 test 200 episodes；另有 2 个训练题冒烟。
没有为补表重测过去轮次，也没有按 test 成绩回退或另选机制。

## 运行时修订与保留证据

初次 DB 第 9 轮的 50 题全部被冻结 Task 会话层的遗留 120-word 检查阻断，与本轮
声明的“无字数上限”协议冲突。旧评测、日志和未封存 state 保留于
`runtime_cap_archive/dbbench/`，不把这项协议冲突当作候选效果 0/50。
独立、哈希绑定的适配修订只移除字数检查，保留单字符串形状、trace 和其他会话行为；
冻结 core/protocol/task 源文件不变。修订后复用原第 9 轮提案及 31/50 基线，
仅重新评测受影响的 50 题，得到上述 27/50。没有增加第四轮提案。

`runtime_amendment.json` 绑定修订源码，修订后的每个评测有 companion marker；
`unaffected_evidence.json` 绑定修订前无需重测的全部评测。九项续跑/运行时回归通过。
`training_seal.json` 封存两域九轮状态；`final_results.json` 记录精确分数、复用来源、
修订哈希和无效评测归档。完整队列日志为 `corrected_workflow.log`。

只读核验：`python meta/experiments/life_agentbench_nine_db_web_mr_20261002/audit_final.py`。
