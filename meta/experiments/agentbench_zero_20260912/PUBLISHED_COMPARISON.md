# 公开版对照补测（2026-09-13）

本次只补评估，不进行提案或迭代。两域各沿用 manifest.json 固定的 50 道训练题；模型、原生环境、任务快照、温度、输出长度、并发及任务预算与前次一致。

- published_h2345：公开实现，enabled=true，H2/H3/H4/H5 全开。
- published_h35：公开实现，enabled=true，H3/H5 开，H2/H4 关，对应 RESULTS.md 所述发布评估层组合。DBBench 的 std YAML 中 enabled=false，因此本次显式开启，避免误测成无 harness。
- life_iter_02：直接引用已完成的从零两轮迭代结果，不再训练或重测。
- baseline：直接引用同样本空四层基线。

published_evaluate.py 复制冻结 evaluate.py，仅调整原 Task 的 harness 开关、配置记录及说明；额外插件使用空 baseline.py，因此不叠加新迭代的 H2/H4。任务快照逐文件与当前仓库一致，运行时仍校验快照与数据哈希。公开规则按原实现执行，不套用新候选的 120 词提示限制。

每域两个公开配置顺序运行，两域并行；每配置 50 题单 trial，合计新增 200 个计分 episode，无新增抽样、无 test。本轮训练样本已用于新方法的自适应选择，因此此处是同样本对照，不能替代独立留出验证。

解释边界：本次比较完整公开实现与受限的从零搜索产物，并非完全相同能力空间的四层原则消融。公开 ALFWorld 的 H4 含逐步状态/子目标提示及预算分支的强制完成，H5 提供冷启动技能；公开 DBBench 的 H5 还含逐步提示。新候选 H4 只能提供有限临时文本，H5 仅在首轮调用。公开 DBBench 的 H0 读取数据条目的 type 元数据与结构化表定义；INSERT 解析还读取参考 SQL 的列数到 expected_insert_cols，但该字段在当前文件中只有定义/赋值，没有后续消费点。保留这些原实现细节以满足“公开版”对照，不把本次称为信息接口完全一致的比较。

公开 ALFWorld H2345 的 H4 轨迹中，有 477 条逐步状态提示、1 个 episode 记录非空预算强制动作；不能把其全部收益解释成强制完成。

有效开关审计：公开 ALFWorld 的 `_H5_CONTRACT_MAP` 当前为空，所以 H5 虽开启但本批 50 题全部未注入技能；表中的 H3+H5 是配置名称，实际表现为 H3。H3 在两域全部 50 题均确认修改工具描述且保留参数 schema；DBBench 两配置均在 50 题触发 H5。没有为补测修改这一公开实现细节。

结果见下表，逐题对照见各域 published_paired_results.csv，完整机器可读记录见 published_comparison.json。

## 完成结果

| 数据集 | 空基线 | 从零 Life 两轮 | 公开 H2345 | 公开 H3+H5 |
|---|---:|---:|---:|---:|
| alfworld | 16/50 | 20/50 | 44/50 | 15/50 |
| dbbench | 23/50 | 28/50 | 29/50 | 35/50 |

| 数据集 / 配置 | agent tokens | 助手回合 |
|---|---:|---:|
| alfworld / baseline | 12,229,497 | 1847 |
| alfworld / iter_02 | 9,590,031 | 1618 |
| alfworld / published_h2345 | 3,215,284 | 805 |
| alfworld / published_h35 | 12,203,834 | 1850 |
| dbbench / baseline | 386,190 | 287 |
| dbbench / iter_02 | 171,744 | 171 |
| dbbench / published_h2345 | 229,014 | 145 |
| dbbench / published_h35 | 241,520 | 139 |

alfworld 新 Life 相对 published_h2345：0 胜 / 24 负，净 -24 题；描述性 exact McNemar p=1.19209e-07。
alfworld 新 Life 相对 published_h35：5 胜 / 0 负，净 +5 题；描述性 exact McNemar p=0.0625。

dbbench 新 Life 相对 published_h2345：4 胜 / 5 负，净 -1 题；描述性 exact McNemar p=1。
dbbench 新 Life 相对 published_h35：2 胜 / 9 负，净 -7 题；描述性 exact McNemar p=0.0654297。

新增 200 个计分 episode，共 15,889,652 agent tokens，最终基础设施错误 0。
该结果不支持从零两轮版已经超过公开版。公开 H2345 与 H3+H5 的差异是联合开关对照，不能单独归因于 H2 或 H4。
