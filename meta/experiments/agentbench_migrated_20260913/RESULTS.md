# 迁移与三轮迭代结果

| 数据集 | 迁移基线 | 第 1 轮 | 第 2 轮 | 第 3 轮 | 最终接受 | 历史 H2345 | 历史最佳 |
|---|---:|---:|---:|---:|---:|---:|---:|
| alfworld | 43/50 | 47/50 接受 | 48/50 接受 | 49/50 接受 | 49/50 | 44/50 | 44/50 |
| dbbench | 24/50 | 25/50 接受 | 27/50 接受 | 29/50 接受 | 29/50 | 29/50 | 35/50 |

## alfworld

最终文件 `alfworld/candidate_03.py`；对迁移基线 6 胜 / 0 负。

- 第 1 轮 container_aware_search_guidance（h4）：区分到达关闭容器与搜索其内部，补充打开容器和返回未搜索容器的 H4 指导；候选 47/50，对当时接受版 4 胜 / 0 负。
- 第 2 轮 transform_state_recovery_guidance（h4）：从工具回执记录清洗/冷热状态，在提前放下或反向处理后给出取回和重新处理指导；候选 48/50，对当时接受版 1 胜 / 0 负。
- 第 3 轮 closed_destination_open_hint（h4）：在目的容器关闭、缺少可用放入动作时，明确提示先打开目的容器；候选 49/50，对当时接受版 1 胜 / 0 负。

## dbbench

最终文件 `dbbench/candidate_03.py`；对迁移基线 5 胜 / 0 负。

- 第 1 轮 fix_grouped_scalar_shape_misclassification（h2）：修正按组统计问题被 H2 单值门控误拦截；候选 25/50，对当时接受版 1 胜 / 0 负。
- 第 2 轮 header_word_total_type_misclassification（h2）：区分公开表头中的 Total 与真正的求和请求；候选 27/50，对当时接受版 2 胜 / 0 负。
- 第 3 轮 canonicalize_multi_row_tuple_rendering（h2）：按公开 SQL 回执统一多行答案的元组空格与单元格类型表示；候选 29/50，对当时接受版 3 胜 / 1 负。

本目录 live episode 共 450，含 DB 迁移诊断 50 题；agent tokens 共 12,363,632，不含 proposer。最终基础设施错误 0。此前已停止的旧形式实验 100 baseline episode 不计入此数。

这是固定训练集上的自适应单 trial。新旧接口存在明确语义差异，模型温度 0 仍有波动，不能将微小分差全部归因于代码，也不能据此声称泛化超越。迁移细节见 EXPERIMENT.md，逐题配对见各域 paired_results.csv。
