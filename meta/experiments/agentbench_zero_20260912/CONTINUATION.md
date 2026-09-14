# 继续迭代三轮（2026-09-13）

用户授权在已接受的第二轮基础上继续三轮，即总轮次 3、4、5。每域固定原 50 题、同一模型和环境、同一四层接口，不添加训练样本。公开最佳成绩仅以总分目标告知提案器，不提供公开源码或公开版轨迹；证据只来自本方法 baseline / iter_*。不修改旧候选或公开配置。

沿用单轮一个集中假设、完整替换候选、50 题全量评估、严格成功数增长才接受，平分拒绝；无 test、无 Meta。预计新增 300 个计分 episode。保留 state_after_round02.json；loop_continued.py 负责第 3–5 轮，原 loop.py 与二轮结果文件保留。接口格式修复仍只在首次计分前进行并保留日志。

H4 仍限每次一条 120 词临时提示，H5 仍只在冷启动调用；本轮不会暗中放宽接口以追赶公开版。公开版的能力空间差异见 PUBLISHED_COMPARISON.md。

协议偏离：DBBench 第 5 轮候选同时修改 H2 保留字引用与 H4 是非结论提交，虽然元数据 target_layer=h2，但实际是跨层联合候选，不满足原定单一集中假设的严格解释。该轮仍只评估一份候选，未追加搜索，并按 29 < 30 拒绝；不对其收益作单层归因。

运行波动：DBBench 第 5 轮的两道回退题（2063、3168），其第一轮请求与第 4 轮逐字段完全相同，但助手正文已不同（不仅是 tool-call ID）。温度 0 在当前模型服务上未实现严格重现；证据见 continuation_nondeterminism.json。按预定单 trial 规则保留观察分数和拒绝决定，但不能把这些回退纯粹归因于候选修改，1–2 题的改进同样需要重复评估确认。本轮不额外加评估次数。

预检修复：ALFWorld 第 4 轮新增 `_pg_ntt` 整数计数器与同名方法冲突，导致 `'int' object is not callable`。计分前由提案器将计数器改名，保留失败源文件和 repair 日志；修复后通过公开轨迹回放才评估，50 题结果 26/50，低于接受版 27/50，因此拒绝。该修复不构成新搜索轮次。

审计与完整性：continuation_integrity.json 校验冻结源代码/数据、最终接受文件和 300 个新增计分 episode；各域 continued_paired_results.csv 保存逐题比较。新增提案及修复共 359 次可见 proposer 模型请求，无显式越界/未允许工具请求；该审计不能证明自动上下文附件的完整隔离。未部署候选到标准 worker。

## 完成结果

| 数据集 | 第 2 轮接受版 | 第 3 轮候选 | 第 4 轮候选 | 第 5 轮候选 | 最终接受 | 公开最佳 |
|---|---:|---:|---:|---:|---:|---:|
| alfworld | 20/50 | 27/50 (接受) | 26/50 (拒绝) | 28/50 (接受) | 28/50 | 44/50 |
| dbbench | 28/50 | 29/50 (接受) | 30/50 (接受) | 29/50 (拒绝) | 30/50 | 35/50 |

### alfworld

- 第 3 轮：h2_transform_state_gate（h2）；27/50，相对当时接受版 7 胜 / 0 负；接受；8,229,542 agent tokens。
- 第 4 轮：h2_progress_loop_gate（h2）；26/50，相对当时接受版 2 胜 / 3 负；拒绝；8,353,200 agent tokens。
- 第 5 轮：h2_stall_search_redirect（h2）；28/50，相对当时接受版 4 胜 / 3 负；接受；8,265,197 agent tokens。
- 最终对 iter_02：10 胜 / 2 负，净 +8；描述性 exact McNemar p=0.0385742。
- 最终对 published_h2345：1 胜 / 17 负，净 -16；描述性 exact McNemar p=0.000144958。

### dbbench

- 第 3 轮：h3_atomic_value_commit_format（h3）；29/50，相对当时接受版 2 胜 / 1 负；接受；218,970 agent tokens。
- 第 4 轮：h4_execution_output_semantics_repair（h4）；30/50，相对当时接受版 1 胜 / 0 负；接受；198,299 agent tokens。
- 第 5 轮：h2_reserved_word_headers_and_h4_yesno_commit（h2）；29/50，相对当时接受版 1 胜 / 2 负；拒绝；185,953 agent tokens。
- 最终对 iter_02：3 胜 / 1 负，净 +2；描述性 exact McNemar p=0.625。
- 最终对 published_h35：4 胜 / 9 负，净 -5；描述性 exact McNemar p=0.266846。

新增 300 个最终计分 episode，共 25,451,161 agent tokens，不含 proposer。最终基础设施错误 0。
同一训练样本上的自适应单 trial 结果，未做独立留出验证；配置/接口差异仍适用。
