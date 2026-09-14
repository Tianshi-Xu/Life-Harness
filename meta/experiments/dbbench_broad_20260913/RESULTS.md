# DB 新批广覆盖训练结果

基线 19/50 → 最终 20/50；最终文件 dbbench/candidate_03.py，对基线 1 胜 / 0 负。

| 轮次 | 候选 | 分数 | 胜 / 负 | 接受 |
|---|---|---:|---:|---|
| 1 | repair_plaintext_toolcall_json_escapes | 19/50 | 0 / 0 | 否 |
| 2 | rewrite_imported_identifiers | 19/50 | 0 / 0 | 否 |
| 3 | recover_single_commit_item_with_inner_quotes | 20/50 | 1 / 0 | 是 |

| 类型 | 题数 | 基线 | 最终 |
|---|---:|---:|---:|
| INSERT | 17 | 5 | 5 |
| SELECT | 2 | 1 | 2 |
| UPDATE | 17 | 12 | 12 |
| aggregation-AVG | 1 | 0 | 0 |
| aggregation-MAX | 1 | 0 | 0 |
| aggregation-MIN | 1 | 0 | 0 |
| aggregation-SUM | 1 | 1 | 1 |
| comparison | 3 | 0 | 0 |
| counting | 2 | 0 | 0 |
| other | 2 | 0 | 0 |
| ranking | 3 | 0 | 0 |

共 200 个计分 episode，1,138,155 agent tokens；最终基础设施错误 0。源代码、数据、候选、样本和提案器只读输入哈希检查通过。

该批使用新题面/类型覆盖策略，仍只有固定 50 道训练题；不能与上一批 29/50 直接比较。本轮无新的独立测试，也未部署 DB 候选。训练改善不自动证明泛化。
