# Meta / AHE TauBench 六轮实验留档

最终逐轮表共 36 行（Meta/AHE × 三场景 × 六轮），本轮实际 token、TRAIN 成绩及分母已逐项与 repair8/TRAINING_RESULTS.json 核对一致；历史快照另留，不覆盖。

状态（2026-10-05 23:46 UTC）：全部完成。Meta/AHE 三场景各六轮已封存，六组最终 TEST 全部结束且均无基础设施错误或缺失。Meta TEST：Airline 4/20 = 20%、Retail 16/40 = 40%、Telecom 20/40 = 50%；AHE TEST：7/20 = 35%、12/40 = 30%、37/40 = 92.5%。Meta 是六次原版优化尝试，不代表六个合格候选；AHE 六轮候选均有完整 TRAIN 测量。repair8 控制器 PID 1092553 及最后 TEST 子进程均已退出，最终全局 DeepSeek 闸门 active=0、limit=8、peak=8；旧废弃 runner/僵尸子进程不存在，本地 Qwen、代理、数据库和其他用户服务保留。此次恢复没有新增付费优化，Retail 原付费 Evolve 6 回执为 `optimizer_replayed=false`，两组末轮共 110 条评测直接复用。最终审计确认训练封存、当前及 TEST 候选哈希、固定任务/分母、源文件指纹、优化用量和闭合账本一致，TEST 未用于选优。320 项全套离线回归及 16 项 Tau 接入测试通过，最后解析器权限复核 31 项通过。下方逐轮表为 TRAIN，最终 TEST 单列；新版 Life 已实现，但未启动其 AgentBench 全量六轮。

本目录是经过产物与费用审计的接入层修复续跑，不是全新实验。Meta 已在前序目录 `meta_ahe_taubench_six_flash_tokens_20261004_repair3` 完成全部 18 次原版优化尝试；其主进程 PID 2887067 已自然退出，终态为 `stopped`，`errors.json` 仅记录三组 AHE 的旧接入故障。本目录此前的零 API 等待器 PID 465802 在发现导入契约缺口后停止等待选择；用户批准后，旧 launch/status 已另外归档。付费产物和原始费用没有改动或删除。现有 `audit_parent` 通过源文件/历史修复档案、候选、固定批次、完整回执、费用及 AHE 缓存核验后，才执行复制与续跑。禁止盲目重放付费任务。

## 已批准的最小修复续跑（2026-10-05）

用户答复“我用你推荐的吧”，选择补齐 Meta/AHE 的纯标准库 `datetime` 接口，复用付费候选补测；只有接收决定改变才重做受影响的后续优化。没有扩大私有属性、工具、模型调用或文件访问权限，也不修改官方优化器或候选内容。

启动前审计已通过：Airline 第 3～6 轮四个原始候选哈希匹配，现有全套预检查仅补日期接口后全部通过。Retail/Telecom 原六轮完整复用。AHE 三场景从原版 iteration 2 续跑，复用 160 条完整候选评测与 Airline/Retail 共 22 个已付费 Debugger 分析文件；Telecom 的 Debugger 2 尚未付费。MR 仍共享单一并发 8 闸门。

沿用现有 `MetaPort.round()`、`run_evolve` 与 `--resume-from`。若某个日期候选严格提升，则归档其后旧分支并运行剩余原版优化；否则继续补测下一个原始候选，不重付优化费用。原始费用账本全部保留；即使后续分支作废，其已花费 token 仍计入对应轮及累计预算，不能用重跑成本覆盖或扣除。

离线回归：Meta/Life 222 项通过，AHE 接入 76 项通过；包含原版 Meta 工作流的零优化 API 缓存补测、首个改变接收结果处截断后续分支、末轮接收、日期正常导入与私有/执行能力继续拒绝。修复前四份接入源代码在前序 `source_archive/*before_date_import.py` 归档，旧运行源哈希可核验。尚未启动 Life 的 AgentBench 全量六轮。

11:31 UTC 运行核验：六组 first-round gate 全部通过；Meta Airline 正在补测原第 3 轮候选，正式预检查通过，19/60 条已完成，尚无最终候选分数。Meta Retail/Telecom 均保留完整六轮，不重放优化。AHE Airline/Retail 正在原版 Evolve 2，原 Debugger 2 报告已被官方缓存复用；Telecom 正在 Debugger 2。MR 闸门实际峰值 4、上限 8，没有控制器错误、训练封存或 TEST 结果。进行中请求费用仍由原账本持续记录。

## 公开访问器最小修复与付费缓存续跑（2026-10-05）

用户核对后明确批准“请修复，然后继续”。仅在接入层补齐 `get_context_value`、`set_context_value`、`tool_calls`、`parameters`、`tool_call_id`。实际 NexAU 源码和离线对象核验表明，它们分别服务于单任务内状态与普通工具调用处理，属于已声明的允许能力；不开放 global storage、变量、executor、sandbox、私有属性或新模型调用。

回归通过：Meta/Life 223 项、AHE 接入 78 项。缓存续跑测试确认无新 Evolve 子进程/API；输入、候选或原结果被更改时拒绝复用；候选仍须通过全部能力校验。官方 AHE 源码保持固定 commit `8b2a55d97590363fe50c3cc6b5e833b020a4bb4c`，无 tracked 修改。四份修复前接入源码在本目录 `source_archive/*before_public_accessors.py` / `*before_completed*_cache.py` 归档。

两份真实原版 Evolve 2 结果通过完整费用、输入快照和文件哈希审计：Airline 原 Evolve 2 已付费 7,873,970 token，126 个文件哈希，候选 workspace SHA-256 `3e58f25ba09b20171e3e42550b0f31e87a9203e825de59561df2fb80a66553c0`；Retail 原 Evolve 2 已付费 9,712,413 token，156 个文件哈希，候选 SHA-256 `b8bc27a3382ac3da3011f6bddf40af2d93d034ee8a551de783c54ada2366d15f`。原候选未编辑，原始费用不扣减。新进程从原版 iteration 2 恢复、复用 Debugger 报告，在接入边界完成原有校验/同步并返回原 Evolve 结果，后续 summary、commit、history 与选择仍由官方流程完成，不重新付费 Evolve 2。

Telecom 的原 Evolve 2 已通过能力校验并被提升，现有进程正在原版 iteration 3 测量该候选。新续跑等待这一评测完成；完整结果通过任务/试验身份、输入快照和费用审计后复用，随后从原版 iteration 3 优化继续。不会终止正在进行的付费请求，也不会启动两个 MR 闸门。所有 Meta 完成轮次直接复用。部分、被篡改或费用不明的缓存仍然停止，不估算或盲目重放。

### Meta 当前完整六轮：TRAIN

| 场景 | 轮次 | batch | incumbent | candidate | 接收 | 本轮实际 token | 累计实际 token |
| --- | ---: | ---: | --- | --- | --- | ---: | ---: |
| Airline | 1 | 0 | 16/60 | 18/60 = 30.00% | 是 | 2,622,885 | 2,622,885 |
| Airline | 2 | 0 | 18/60 | 15/60 = 25.00% | 否 | 5,212,757 | 7,835,642 |
| Airline | 3 | 1 | 15/60 | 18/60 = 30.00% | 是；原候选补测 | 8,064,329 | 15,899,971 |
| Airline | 4 | 1 | 18/60 | 18/60 = 30.00% | 否；新分支 | 9,981,132 | 25,881,103 |
| Airline | 5 | 2 | 17/60 | 18/60 = 30.00% | 是；新分支 | 13,479,032 | 39,360,135 |
| Airline | 6 | 2 | 18/60 | 16/60 = 26.67% | 否；新分支 | 13,728,217 | 53,088,352 |
| Retail | 1 | 0 | 16/50 | 16/50 = 32.00% | 否 | 1,661,897 | 1,661,897 |
| Retail | 2 | 1 | 19/50 | 未测：仅完成 analysis | 否 | 1,892,981 | 3,554,878 |
| Retail | 3 | 2 | 19/50 | 18/50 = 36.00% | 否 | 1,441,959 | 4,996,837 |
| Retail | 4 | 3 | 24/50 | 17/50 = 34.00% | 否 | 4,092,266 | 9,089,103 |
| Retail | 5 | 4 | 18/50 | 18/50 = 36.00% | 否 | 1,539,914 | 10,629,017 |
| Retail | 6 | 5 | 20/50 | 17/50 = 34.00% | 否 | 3,734,309 | 14,363,326 |
| Telecom | 1 | 0 | 15/50 | 12/50 = 24.00% | 否 | 2,324,074 | 2,324,074 |
| Telecom | 2 | 1 | 16/50 | 未测：私有 __setattr__ 被拒绝 | 否 | 2,751,245 | 5,075,319 |
| Telecom | 3 | 2 | 14/50 | 未测：私有属性被拒绝 | 否 | 2,169,234 | 7,244,553 |
| Telecom | 4 | 3 | 14/50 | 未测：私有属性被拒绝 | 否 | 4,572,550 | 11,817,103 |
| Telecom | 5 | 4 | 14/50 | 未测：object.__setattr__ 被拒绝 | 否 | 5,341,751 | 17,158,854 |
| Telecom | 6 | 5 | 15/50 | 未测：object.__setattr__ 被拒绝 | 否 | 4,163,937 | 21,322,791 |

最终训练保留 `error_reground_recovery.py`。第 4～6 轮 token 含被作废原分支的真实费用加新分支费用，未覆盖或扣除旧支出。Retail / Telecom 仍分别为 14,363,326 / 21,322,791 token；修正后 Meta 三场景六轮总优化支出为 88,774,469 token。18 轮已完成的状态、固定批次、候选和费用回执通过现有 `audit_meta_cached`；这仍不是 TEST 分数。

当前 Meta 18 次优化尝试中，12 个候选完整测量、5 个因私有接口预检查被拒绝、1 次未完成原版两代理流程；接收 Airline 第 1、3、5 轮，Retail/Telecom 保留 `baseline_zero.py`。以下“历史完成轮次”段落保留修复前快照，不代表当前结果。实际支出包含修复后分支费用，不能把它当作全新干净六轮实验的纯方法预算，或据此夸大另一方法的 token 优势。

## 实验条件

两种方法各跑 Airline / Retail / Telecom 六轮优化，随后对训练保留版本做一次固定 TEST，不用测试集选版本。

| 场景 | 每轮 TRAIN | 最终 TEST | 用户模拟器 |
| --- | --- | --- | --- |
| Airline | 30 个任务 × 2 次 = 60 条 | 20 个任务 × 1 次 | 用户阿里云 API，qwen3.8-flash，thinking 开启 |
| Retail | 50 个任务 × 1 次 | 40 个任务 × 1 次 | 用户阿里云 API，qwen3.8-flash，thinking 开启 |
| Telecom | 50 个任务 × 1 次 | 40 个任务 × 1 次 | ModelRouter deepseek-v4-flash，推理配置保持提供方默认 |

训练批次和 TEST 身份固定于历史 manifest，并与原生 train/test split 校验。不同轮训练批次可能重叠；分数必须连同 batch 和分母展示，不解释为同批次连续改善。

做题模型为本地 qwen3-4b，temperature=0，总上下文 32768、单次输出 4096；任务并发 50。优化模型为 API deepseek-v4-flash。所有 MR 优化、Debugger、Telecom 用户模拟器调用共享一个全局闸门，并发不超过 8；阿里云和本地 Qwen 不受此闸门限制。没有给 DeepSeek 优化套用 Qwen 的 32768/4096 上限，也没有累计 token 上限。

原有优化器参数保持不变：Meta API shell 单次输出 60000、提案时限 5400 秒；AHE Evolve 单次输出 32000、`max_iterations=500`、`max_context_tokens=200000`，压缩中间件的 `max_context_tokens=220000`。这些是各自已有配置，不代表两种优化器拥有相同的单次输出或内部压缩预算；跨方法比较使用实际累计优化 token。

Meta 执行固定原版 `run_evolve` 和官方 skill，通过 API shell 使用 parent → analysis → implementation；按完整同批次严格提升接收，未提升时更换批次。提案失败保留为实际消耗的一轮，候选分数记“未测”，不伪造为 0% 或重新优化来替换失败记录。

AHE 执行固定原版 `run_single_experiment`、Debugger/Evolve 和 best-ever 逻辑；采用用户批准的受控能力版本，每轮最多精读 10 个训练任务，不开放额外运行时 LLM、Explore、子代理或 best-of-N。它不是无限制官方 AHE 的复现。候选 N 在原版评测 N+1 中、通常在下一训练批次测量；候选 6 额外测 batch 5，这次测量不反馈进选择。AHE 与 Meta 的初始模板、选择机制及测量批次并不完全相同。

2026-10-05 20:15 UTC 原版源码核验澄清：AHE 的 `best_ever.json` 用于历史记录和 Evolve 的回滚参考，`run_single_experiment` 结束时不会自动把 workspace 替换为最高 TRAIN 快照。本次末轮额外 TRAIN 和最终 TEST 均使用六轮结束时的 workspace；没有人工改选历史最高 TRAIN，更没有按 TEST 选优。冻结协议中“original best-ever”的描述指沿用这一原版逻辑，不应解读为最终强制挑最高训练分版本。实际运行与官方核心未改变，无需重跑。

## 实际优化 token 口径

只累计 Meta 的 parent / analysis / implementation，或 AHE 的 Debugger / Evolve，包含其压缩、重试等实际调用。不包含 qwen 做题、用户模拟器、Retail 自然语言 judge。实测总量为 API 实际 input + output；cached input 已包含在 input 中，不重复加算。仅下文明确授权、绑定原回执的 Telecom Evolve 2 单笔另列估算，绝不混入实测列。

显式 HTTP 429 且未提供 usage 的拒绝单独登记，原始缺失标记保留；除上述单笔历史授权外，任何成功或结果模糊的优化调用缺少 usage 都会中断核验，不按 0 或平均值补齐。下表只汇总已结束的轮次，进行中提案的费用仍保存在逐请求账本，不能将下表小计当作实时总开销。

## Meta 历史完成轮次：TRAIN（日期修复补测前快照）

| 场景 | 轮次 | batch | incumbent | candidate | 接收 | 本轮 token | 累计 token |
| --- | ---: | ---: | --- | --- | --- | ---: | ---: |
| Airline | 1 | 0 | 16/60 = 26.67% | 18/60 = 30.00% | 是 | 2,622,885 | 2,622,885 |
| Airline | 2 | 0 | 18/60 = 30.00% | 15/60 = 25.00% | 否 | 5,212,757 | 7,835,642 |
| Airline | 3 | 1 | 15/60 = 25.00% | 未测：导入 datetime 被接入层白名单拒绝 | 否 | 8,064,329 | 15,899,971 |
| Airline | 4 | 2 | 17/60 = 28.33% | 未测：日期导入被同一白名单拒绝 | 否 | 5,959,179 | 21,859,150 |
| Airline | 5 | 3 | 17/60 = 28.33% | 未测：日期导入被同一白名单拒绝 | 否 | 7,489,792 | 29,348,942 |
| Airline | 6 | 4 | 19/60 = 31.67% | 未测：日期导入被同一白名单拒绝 | 否 | 9,557,502 | 38,906,444 |
| Retail | 1 | 0 | 16/50 = 32.00% | 16/50 = 32.00% | 否 | 1,661,897 | 1,661,897 |
| Retail | 2 | 1 | 19/50 = 38.00% | 未测：仅完成 analysis，未执行 implementation | 否 | 1,892,981 | 3,554,878 |
| Retail | 3 | 2 | 19/50 = 38.00% | 18/50 = 36.00% | 否 | 1,441,959 | 4,996,837 |
| Retail | 4 | 3 | 24/50 = 48.00% | 17/50 = 34.00% | 否 | 4,092,266 | 9,089,103 |
| Retail | 5 | 4 | 18/50 = 36.00% | 18/50 = 36.00% | 否 | 1,539,914 | 10,629,017 |
| Retail | 6 | 5 | 20/50 = 40.00% | 17/50 = 34.00% | 否 | 3,734,309 | 14,363,326 |
| Telecom | 1 | 0 | 15/50 = 30.00% | 12/50 = 24.00% | 否 | 2,324,074 | 2,324,074 |
| Telecom | 2 | 1 | 16/50 = 32.00% | 未测：预检查拒绝 __setattr__ 访问 | 否 | 2,751,245 | 5,075,319 |
| Telecom | 3 | 2 | 14/50 = 28.00% | 未测：预检查拒绝原生类私有属性访问 | 否 | 2,169,234 | 7,244,553 |
| Telecom | 4 | 3 | 14/50 = 28.00% | 未测：预检查拒绝原生类私有属性访问 | 否 | 4,572,550 | 11,817,103 |
| Telecom | 5 | 4 | 14/50 = 28.00% | 未测：预检查拒绝 object.__setattr__ 访问 | 否 | 5,341,751 | 17,158,854 |
| Telecom | 6 | 5 | 15/50 = 30.00% | 未测：预检查拒绝 object.__setattr__ 访问 | 否 | 4,163,937 | 21,322,791 |

修复前历史快照：当时 Meta 三场景各六轮优化尝试均已结束并通过续跑审计。18 次尝试中，8 个候选完整测量、9 个候选被预检查拒绝、1 次未完成原版两代理流程；只接收 Airline 第一轮，不能把“六轮尝试完成”称作“六个合格候选全部测过”。当时优化 token 合计 74,592,561（Airline 38,906,444、Retail 14,363,326、Telecom 21,322,791）；当前修正后结果以文首“Meta 当前完整六轮”表为准。预检查拒绝计入实际预算，但没有候选成绩，不能解释成性能为 0%。Retail 第 3、4 轮的完整 50 条评测均无缺失或基础设施错误，未达到同批次严格提升，按原版流程拒绝；源码、两代理回执、预检查、成绩和费用均已核验，不重新提案。第 4 轮候选 SHA-256 为 `b1dcbe640feb91b8b3363f09f2c9f15e897b380abdd723ef95e9d2addd080459`，保留 `baseline_zero.py`。

Retail 第 5 轮也已通过现有续跑审计器：原版 analysis/implementation、候选与 pending 哈希、预检查、完整配对测量及全部优化 usage 一致。候选 `general_operating_checklist` 的 SHA-256 为 `25908bf9f38a728d5ba54325409539071c7a67984ab7bf687fab6556944124ef`，实际优化开销 1,539,914 token（parent 618,266、analysis 787,586、implementation 134,062）。完整 batch 4、50 条 TRAIN 无缺失或基础设施错误；候选与 incumbent 均为 18/50 = 36.00%，按严格提升规则拒绝，保留 `baseline_zero.py` 并进入 batch 5。费用已计入表内累计，不重放候选。

Retail 第 6 轮已完成并通过完整六轮续跑审计：原版两代理流程、候选及 pending 哈希、通过的预检查、配对任务身份和完整优化 usage 一致。候选 `post_auth_id_grounding` 的 SHA-256 为 `7bcf3dadd4acaf36ce31ad51e706048614265f96779ad434634dfca9b3050dd4`，实际优化开销 3,734,309 token（parent 2,479,009、analysis 1,072,379、implementation 182,921）。完整 batch 5、50 条 TRAIN 无缺失或基础设施错误；候选 17/50 = 34.00%，低于 incumbent 20/50 = 40.00%，按原版规则拒绝。六轮累计 14,363,326 token，最终保留 `baseline_zero.py`；未封存或 TEST，不重放提案。

Telecom 第 3 轮已通过现有续跑审计器：batch 2 incumbent 为 14/50 = 28.00%，无缺失或基础设施错误；任务/试验身份、源码和做题配置一致，原版 analysis/implementation 和费用完整。候选 `customer_device_tool_boundary` 的 SHA-256 为 `006340221568831425be9f18b50c3ffbb86bd05eab2152c92c7bc7cf57f46422`。原预检查再次只读复现 `forbidden private/released runtime reference`，源码访问原生类的 `__init__`（第 144、159 行）、`__doc__`（第 168 行）；未测候选，不等于准确率 0%。未放宽这类访问，保留 `baseline_zero.py` 并进入 batch 3，不重放已付费优化。

Telecom 第 4 轮已完成并通过四轮续跑审计：batch 3 incumbent 为 14/50 = 28.00%，完整 50 条无缺失或基础设施错误。候选 `identifier_grounding_rule.py` 的 SHA-256 为 `3a545650d2b6832f8162b9838ba146d81cf53ca0a46058c10d677410e1ee8ab2`，导入原生 `tau2.agent.llm_agent`，访问 `__init__`（第 267、276 行）、`__doc__`（第 285 行）；原校验器只读复现 `forbidden private/released runtime reference`。原版两代理流程、候选/pending 哈希与实际优化费用完整，无未知 usage。本轮 4,572,550 token（parent 2,698,868、analysis 1,454,767、implementation 418,915），累计 11,817,103；候选未测、未接收，不记作 0%，保留 `baseline_zero.py` 进入 batch 4。没有改白名单、候选或原始决定，也没有重发优化请求。

Telecom 第 5 轮已完成并通过五轮续跑审计：batch 4 baseline 为 14/50 = 28.00%，完整 50 条无缺失或基础设施错误；固定任务身份、唯一任务/试验对、源码哈希与 32768/4096 做题配置一致。候选 `off_schema_action_routing.py` 的 SHA-256 为 `87ba66a091b711bea8f932044cde6cf16f8d3fd5630b9525727dd156dbe0a4a1`，仅导入 `__future__`、`json`、`re`；第 159 行 `_set` 辅助函数使用 `object.__setattr__` 作为设置字段的后备路径，原校验器只读复现 `forbidden private/released runtime reference`。原版两代理流程、候选/pending 哈希和费用完整，无未知 usage。本轮 5,341,751 token（parent 2,504,637、analysis 2,090,908、implementation 746,206），累计 17,158,854。候选未测、未接收，不等于准确率 0%；保留 `baseline_zero.py` 进入最后的 batch 5。没有更改名单、候选、原决定或重发优化请求。

Telecom 第 6 轮已完成并通过完整六轮续跑审计：batch 5 baseline 为 15/50 = 30.00%，完整 50 条无缺失或基础设施错误；固定任务身份、唯一任务/试验对、源码哈希与 32768/4096 做题配置一致。候选 `grounded_identifier_gate.py` 的 SHA-256 为 `99d44340caf4155eda6d6fd0f88e5bde25e3ad8e05e0a404881d5f6c22b6f647`，仅导入 `__future__`、`json`、`re`；第 127 行 `_set` 辅助函数使用 `object.__setattr__`，原校验器只读复现 `forbidden private/released runtime reference`。75 次实际调用和原版两代理回执完整，无未知 usage。本轮 4,163,937 token（parent 1,539,434、analysis 2,455,719、implementation 168,784），六轮累计 21,322,791。候选未测、未接收，最终保留 `baseline_zero.py`；未将未知成绩记作 0%，未改名单、候选或重发优化。

已确认接口说明缺口：Meta 提示允许标准库、AHE 提示允许普通纯标准库，但两者白名单不含 `datetime`；Life 的接口允许 `from datetime import datetime` 等受控日期导入。Airline 第 3 轮候选只使用 `datetime.datetime.strptime`，仍被拒绝；该候选的原版两代理流程和 130 个实际 API 回执完整，不能将其未测结果归因于模型性能差。是否修正名单并复用候选补测正在等待用户选择；目前没有改白名单、候选、接收结果或冻结协议，亦没有重发该轮优化请求。Telecom 的私有属性访问是另一个限制，未随日期导入问题放宽。

2026-10-05 02:57 UTC 的独立只读探针：仅在进程内临时给现有 Meta 预检查允许 `datetime`，该 Airline 候选即通过全部现有预检查（含 AST 规则及导入/导出检查）。真实校验器和候选文件 SHA-256 均不变，未调用付费 API、未进行 TRAIN 测量。此探针证明现有预检查拒绝点，不证明候选能提升准确率。

第 4 轮 `blocked_record_guard.py` 再次命中同一缺口：只导入 `json`、`re` 和 `from datetime import datetime`，没有禁用的私有属性或名称。原版两代理回执及实际费用完整，现有续跑审计通过；候选 SHA-256 为 `2d622ed9150cf4fc8a1b8d0d408cd694f3052f845f1b23aea00d068802474bc3`。03:35 UTC 的独立只读探针仅临时允许 `datetime` 后通过全部现有预检查，真实校验器和候选文件均未改变，未调用付费 API 或补测。

第 5 轮 `cancel_eligibility_gate.py` 同样只导入 `datetime`、`json`、`re`，没有私有属性访问；候选 SHA-256 为 `bb143c88030f3baffc375752e9a3b508b9caad46a250a4980eeaf73bf21d6348`。原版两代理回执、完整费用和历史决定已通过现有续跑审计；04:11 UTC 的独立只读探针仅临时允许 `datetime` 后通过全部现有预检查，真实文件不变，付费 API 和补测调用均为 0。其未知候选成绩不能当作性能下降；当前仍等待接口选择，没有改动接收分支。

Airline 第 6 轮也已结束并通过完整六轮续跑审计，原版两代理回执和实际费用完整，无未知优化 usage。候选 `cancel_hold_guard.py` 的 SHA-256 为 `11958b7c1a730d9c933e7858fe4a91d1f19de90a2f83b307d5594f12c45d2ed1`，仅导入 `datetime`、`json`、`re`，无私有属性访问；原校验器只读复现 `forbidden candidate import`。本轮 9,557,502 token（parent 4,074,794、analysis 4,968,675、implementation 514,033）计入累计 38,906,444；候选未测，仍保留第一轮的 `preaction_verify_reminder.py`。不把接口拒绝解释为准确率下降，也不更改原始协议或重放已付费提案。

2026-10-05 07:31 UTC 的第 6 轮只读探针：仅在独立进程内给现有校验函数的标准库集合增加 `datetime`，该候选即通过全部现有 AST 及原生导入/导出检查；真实校验器、候选及该场景付费账本的哈希均不变，未调用模型或进行 TRAIN 测量。至此第 3～6 轮的四个日期候选均已独立确认仅被日期导入拦截，但准确率仍未知。未改变冻结协议、接收分支或启动续跑。

## AHE 已完成轮次：TRAIN

baseline 与 candidate 不是同一批次的配对比较，不能直接归因其分数变化。

| 场景 | 优化轮次 | 诊断 batch | 候选测量 batch | 初始 baseline | 候选成绩 | 本轮 token | 累计 token |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| Airline | 1 | 0 | 1 | 22/60 = 36.67% | 6/60 = 10.00% | 11,435,194 | 11,435,194 |
| Retail | 1 | 0 | 1 | 21/50 = 42.00% | 20/50 = 40.00% | 17,352,611 | 17,352,611 |
| Telecom | 1 | 0 | 1 | 9/50 = 18.00% | 23/50 = 46.00% | 18,510,187 | 18,510,187 |
| Airline | 2 | 1 | 2 | 6/60 = 10.00% | 18/60 = 30.00% | 8,392,342 | 19,827,536 |
| Retail | 2 | 1 | 2 | 20/50 = 40.00% | 20/50 = 40.00% | 10,731,847 | 28,084,458 |
| Telecom | 2 | 1 | 2 | 23/50 = 46.00% | 27/50 = 54.00% | 16,397,916 实测 + 135,307 估算 | 34,908,103 实测 + 135,307 估算 |
| Telecom | 3 | 2 | 3 | 27/50 = 54.00% | 43/50 = 86.00% | 14,144,198 实测 | 49,052,301 实测 + 135,307 历史估算 |
| Airline | 3 | 2 | 3 | 18/60 = 30.00% | 27/60 = 45.00% | 13,269,875 实测 | 33,097,411 实测 |
| Retail | 3 | 2 | 3 | 20/50 = 40.00% | 25/50 = 50.00% | 13,405,282 实测 | 41,489,740 实测 |
| Airline | 4 | 3 | 4 | 27/60 = 45.00% | 25/60 = 41.67% | 12,667,408 实测 | 45,764,819 实测 |
| Telecom | 4 | 3 | 4 | 43/50 = 86.00% | 44/50 = 88.00% | 14,086,441 实测 | 63,138,742 实测 + 135,307 历史估算 |
| Retail | 4 | 3 | 4 | 25/50 = 50.00% | 22/50 = 44.00% | 28,289,115 实测 | 69,778,855 实测 |
| Airline | 5 | 4 | 5 | 25/60 = 41.67% | 25/60 = 41.67% | 10,350,181 实测 | 56,115,000 实测 |
| Telecom | 5 | 4 | 5 | 44/50 = 88.00% | 44/50 = 88.00% | 12,609,327 实测 | 75,748,069 实测 + 135,307 历史估算 |
| Airline | 6 | 5 | 5；额外末轮测量 | 25/60 = 41.67% | 25/60 = 41.67% | 8,382,756 实测 | 64,497,756 实测 |
| Retail | 5 | 4 | 5 | 22/50 = 44.00% | 16/50 = 32.00% | 11,475,061 实测 | 81,253,916 实测 |
| Telecom | 6 | 5 | 5；额外末轮测量 | 44/50 = 88.00% | 50/50 = 100.00% | 12,465,222 实测 | 88,213,291 实测 + 135,307 历史估算 |
| Retail | 6 | 5 | 5；额外末轮测量 | 16/50 = 32.00% | 10/50 = 20.00% | 11,887,185 实测 | 93,141,101 实测 |

六轮优化开销已封存：

| 方法 | Airline 实测 token | Retail 实测 token | Telecom 实测 token | 三场景实测合计 | 单列历史估算 | 含估算合计 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Meta | 53,088,352 | 14,363,326 | 21,322,791 | 88,774,469 | 0 | 88,774,469 |
| AHE | 64,497,756 | 93,141,101 | 88,213,291 | 245,852,148 | 135,307 | 245,987,455 |

以上不含 Qwen 做题、用户模拟器和 judge；最终 TEST 仅新增这些排除角色，不再增加优化开销。Meta Airline 含日期修复后旧分支与新分支的真实支出，不能直接当作全新实验的干净预算。AHE Retail 最后一条 TRAIN 的 judge 思考较长，但已正常完成评分并有用量；没有因此重跑优化或改动固定分母。

2026-10-05 21:36 UTC 最小恢复：Retail 原付费 Evolve 6 使用 `ast.literal_eval` 解码历史工具结果的 Python 字面量，不执行代码、不读取文件、不新增工具/模型。仅开放此纯解析器，`ast.parse`、`ast.main`、私有属性、动态导入和执行路径仍被拒绝；原候选未编辑，完整 workspace 校验及其 JSON/repr 解析、可执行字符串拒绝检查通过。源文件修复前快照保存在 repair7/source_archive。真实续跑审计确认 Airline/Telecom 六轮完成，从 iteration 7 空后缀返回，复用末轮 60/50 条评测；Retail 复用已完成 iteration 6 的 50 条测量、Debugger 与 Evolve 缓存，只提升原产物后补末轮测量。恢复逻辑验证任务/试验身份、固定分母、完整分数、workspace 和文件哈希，拒绝部分、篡改或跨场景结果，复制时仅更新 job 目录引用。新增运行测试证明不再调用已完成末轮的 run_harbor。没有重放优化，也未修改官方 AHE 核心和选择规则。

2026-10-05 14:40 UTC 新增完整测量：Telecom 第 2 轮候选在 batch 2、50 条 TRAIN 上为 **27/50 = 54%**，基础设施错误 0；这不是 TEST，也不是与前一 batch 的配对提升。该轮实测优化 token 16,397,916，单笔估算 135,307，含估算 16,533,223；累计实测 34,908,103，含估算 35,043,410。完整 native job、任务/试验身份及 workspace receipt 均经 `audit_parent` 核验，原版 iteration 3 从 `skip_eval=True` 接续，无重新测量或优化第 2 轮。

2026-10-05 16:52 UTC 完整核验：Telecom 第 3 轮候选在 batch 3、50 条 TRAIN 上为 **43/50 = 86%**，其中 1 条基础设施错误按失败计入固定分母，整批未剔除或重跑。这不是 TEST，不能直接与 Life 发布版 TEST 的 57.5% 作同条件比较。该轮真实优化费用 14,144,198 token；单笔 135,307 估算仍只属于历史第 2 轮，未重复计入第 3 轮。

2026-10-05 17:16 UTC 完整测量：Airline 第 3 轮候选在 batch 3、60 条 TRAIN 上为 **27/60 = 45%**，基础设施错误 0。第 3 轮已完成实测优化 13,269,875 token，累计前三轮 33,097,411；缓存候选未修改、未重放付费优化。仍不是同 batch 的配对提升或 TEST 成绩。

2026-10-05 17:21 UTC 完整测量：Retail 第 3 轮候选在 batch 3、50 条 TRAIN 上为 **25/50 = 50%**。第 3 轮已完成实测优化 13,405,282 token，累计前三轮 41,489,740；没有重新生成原付费候选。最终 TEST 仍未开始。

2026-10-05 18:10 UTC 完整测量：Airline 第 4 轮候选在 batch 4、60 条 TRAIN 上为 **25/60 = 41.67%**，费用 12,667,408 实测 token，累计前四轮 45,764,819。与上一轮不同批次，不据此声称同批次性能下降；第 5 轮继续使用原版优化器。

2026-10-05 18:43 UTC 完整测量：Telecom 第 4 轮候选在 batch 4、50 条 TRAIN 上为 **44/50 = 88%**，其中 1 条基础设施错误按失败保留在分母 50，不剔除或重跑整批。该错误为原生用户模拟消息校验 `UserMessage must have either content or tool_calls`，不是 harness 预检查；未保留该条原始响应，不能进一步断定由模型输出还是文本处理触发。费用 14,086,441 实测 token；累计前四轮 63,138,742 实测，另保留第 2 轮单笔历史估算 135,307。这是 TRAIN，不与 Life 发布版 57.5% TEST 作直接同条件比较；继续原版第 5 轮。

2026-10-05 19:00 UTC 完整测量：Retail 第 4 轮候选在 batch 4、50 条 TRAIN 上为 **22/50 = 44%**，基础设施错误 0。该轮真实优化费用 28,289,115 token，累计前四轮 69,778,855；保留全部原调用、压缩与重试费用，不因训练未优于不同 batch 的上一轮而扣除。三组前四轮的 TRAIN 与优化费用均已完整列出，尚不是六轮终态。

2026-10-05 19:04 UTC 完整测量：Airline 第 5 轮候选在 batch 5、60 条 TRAIN 上为 **25/60 = 41.67%**，费用 10,350,181 实测 token，累计前五轮 56,115,000。第 6 轮诊断已启动；Retail 尚在第 5 轮，Telecom 正在测量第 5 轮候选。训练尚未封存，TEST 尚未启动。

2026-10-05 20:07 UTC 完整测量：Telecom 第 5 轮候选在 batch 5、50 条 TRAIN 上为 **44/50 = 88%**，基础设施错误 0，费用 12,609,327 实测 token，累计前五轮 75,748,069；仍另列第 2 轮单笔历史估算 135,307，不重复估算新轮次。原版第 6 轮诊断继续，最终 TEST 未启动。

2026-10-05 20:19 UTC 完整测量：Airline 六轮优化和末轮额外 TRAIN 均结束。第 6 轮最终 workspace 在 batch 5、60 条 TRAIN 上为 **25/60 = 41.67%**，基础设施错误 0，`used_for_selection=false`。第 6 轮优化费用 8,382,756 实测 token，六轮累计 64,497,756，优化费用已完整核验；这次末轮 TRAIN 不反馈原版优化器，也不是 TEST。Retail/Telecom 尚未全部结束，暂不封存或测试。

2026-10-05 20:27 UTC 完整测量：Retail 第 5 轮候选在 batch 5、50 条 TRAIN 上为 **16/50 = 32%**，费用 11,475,061 实测 token，累计前五轮 81,253,916。该结果原样保留，不为提高分数重跑；原版第 6 轮诊断继续，TEST 尚未开始。

2026-10-05 21:09 UTC 完整测量：Telecom 六轮及末轮额外 TRAIN 均完成。第 6 轮最终 workspace 在 batch 5、50 条 TRAIN 上为 **50/50 = 100%**，基础设施错误 0，`used_for_selection=false`。本轮 12,465,222 实测 token，六轮累计 88,213,291 实测；另列第 2 轮历史估算 135,307，含估算预算 88,348,598。这不是 TEST，不与 Life 发布版 57.5% TEST 直接等同。仍等 Retail 六轮及末轮测量完成后统一封存并测试。

历史快照（第二轮续跑前）：Airline Debugger 已消耗 518,372 token、Retail 已消耗 1,019,434 token，两者各有 10 份完整分析；原版 AHE 的 overview 缓存将直接复用，不重复付费。彼时第二轮 Evolve 尚未调用，第二轮候选未产出/未测。Telecom 第二轮 Debugger 在接入层证据投影处失败前未发起 API 调用；该历史档案检查已修复。

AHE 已付费优化 token 合计 48,835,798，其中完整第一轮合计 47,297,992、第二轮已完成 Debugger 合计 1,537,806；不能当作六轮总费用。主进程退出后，现有续跑审计器确认三场景均可从原版第 2 轮、`skip_eval=True` 接续，复用已完整测量的 60/50/50 条候选 episode；Airline/Retail 各 11 个 analysis 文件（overview + 10 个 detail）的哈希和完整付费回执均通过核验。原账本不重标、不估算、不重付 Debugger。当前尚未调用复制或启动续跑，等待接口选择。

## 修复与复用说明

Telecom 用户模拟器的旧输出上限 4096 曾产生 8 条“推理耗尽输出、无最终用户文本”的错误训练 episode。已提高该模拟器输出到 100000，并仅重测确诊错误：AHE 初始 baseline 复用 42 条有效 episode，补测 8 条，得到完整 9/50；当时已完成的第 1、2 轮候选测量均无缺失 episode 或基础设施错误。后续第 3 轮的 1 条基础设施错误另按失败保留，不以此宣称模拟器从不失败。两种方法第一次 Telecom 优化均读取修复前 TRAIN，不重新生成，必须保留这个条件差异。Meta 第一次提案的 `proposer_01/frozen_inputs.json` 中训练 summary 哈希与保留的旧 14/50 summary 一致，不是修复后 15/50 summary；表中是候选复用后的完整重测成绩。

Telecom 第一次续跑 Evolve 的 140 个请求因接入层地址重绑定问题缺少 scope。其原版 tracer 的请求数、时间区间及 14,734,324 个实际 token 与全部回执一一匹配，故仅在汇总时明确归到第一次 Evolve；原账本不修改、不重标签、不估算。后续私有隔离副本的 key/URL 已按轮次绑定。

修复后的历史 baseline 保留两个评测档案，曾被证据投影误判为“当前轮重复 job”。修复只对当前轮执行单 job 检查；历史档案只开放汇总、不开放详细轨迹，当前轮 10 个精读任务限制不变。AHE 核心未修改。

## 流式用量修复与单笔历史估算（2026-10-05）

用户明确批准“赶紧修复，然后继续；这次找不回来就用估算值先顶着”。接入层现仅用有效完整且计数不回退的 usage 更新累计回执，空、不完整、负数、不一致及下降至零/较小值的尾包不再覆盖此前有效值；一条请求多次累计回执只取最新有效值，不相加。账本另记录流式 `[DONE]`、正常 EOF、有效/无效用量包计数、错误事件及结束原因；不保存提示、响应正文或错误消息中的凭据。断流时保留已收到的有效回执。

完整离线回归：AHE 86 项、Meta/Life 224 项通过。新增覆盖空/不完整/无效尾包、累计更新、无 DONE 与无 usage 的区分、读取异常、错误事件、8 路并发、估算与实测分离、估算回执身份/哈希/重复检查以及后续未知/待完成请求仍拒绝自动续跑。官方优化器和候选源码没有更改。四份修复前源文件另存于本目录 `source_archive/*before_usage_*.py`。

旧 Telecom Evolve 2 请求 `e18eb75c714448f48b6b40180dac0c21` 的实际用量无法恢复，不能据此次 bug 复现断定它一定曾返回有效 usage。只对该请求添加独立 `ledgers/mr_requests.estimates.json`；原始 started/finished 回执均未修改，仍为 unknown。估算依据是同 experiment/domain/round/role/model 的前后最近两条有实际用量调用：前一条 132,522 输入 + 136 输出，后一条 134,709 输入 + 3,246 输出。分别取向上舍入算术平均：**133,616 输入 + 1,691 输出 = 135,307 token（估算，非实测）**。不能由持续 61 分钟推导实际计费。

Telecom 第 2 轮实测优化 16,397,916 token；另估算 135,307，含估算预算为 16,533,223。当前该场景已知累计优化 34,908,103，含该笔估算 35,043,410。汇总保留 `total_tokens`（实测）、`estimated_total_tokens` 与 `budget_tokens_including_estimates` 三列，`inference_accounting_complete` 仍为 false，不冒充完整实测；只有绑定确切原回执哈希、明确授权的这一笔估算可放行，未来缺失回执不会被自动估算或忽略。模拟器和评分费用仍不进入优化预算。

13:53 UTC：只重启了零 API 等待器，旧 PID 1358478 的 launch/status 另外归档，新 PID 1763176 加载修复后代码；仍等待旧 Telecom TRAIN 评测完成（12/50），不终止在途任务、不重跑已付费 Debugger/Evolve、不重开 Meta 六轮、不重叠 MR 闸门。Life AgentBench 全量六轮未启动。

14:38 UTC 最终边界复核又增加零值/回退计数尾包用例，AHE 86 项与 Meta/Life 224 项完整回归再次全部通过。仅重启零 API 等待器，先前等待 launch/status 另按 PID 归档，未中断旧训练。14:39 UTC 真实 `audit_parent` 通过：Meta 三场景各六轮、Airline/Retail 原付费缓存、Telecom 完整第三次训练评测和单笔估算回执均一致，所有未知优化请求只剩这笔明确授权估算；旧控制器已退出，没有在途请求。14:40 UTC PID 2147656 接续，14:41 UTC `status.phase=training`，MR 新闸门尚为 0 请求、上限 8，原始 Meta 请求重放数为 0。

15:12 UTC 真实流式回执核验：续跑后的 51 条已结束 Evolve 流式请求中，50 条 HTTP 200 均有有效实测 usage，另 1 条明确 HTTP 429。成功请求保留 `[DONE]`、正常 EOF 及结束原因诊断；未再出现空尾包覆盖此前有效回执。Airline / Retail 原 Evolve 2 的 `completed_cache_receipt.json` 均明确 `optimizer_replayed=false`；完整 iteration 3 测量分别 18/60、20/50，基础设施错误均为 0，原候选哈希未改。

### 全局并发与废弃进程清理（2026-10-05）

当前唯一 TauBench 控制器 PID 2147656 的 Meta optimizer、AHE Debugger/Evolve 与 Telecom 用户模拟器共用同一个 `RequestGate(8)` 实例；不是三个场景各限 8。实测峰值 6。阿里云模拟器、本地 Qwen 服务不纳入 Flash 闸门。

已只读核验旧 TauBench 控制器 PID 544803、等待器 PID 1358478 / 1763176 均退出。另发现旧 Life AgentBench 控制器 PID 2890786 停在 `T` 状态，保留旧监听端口，但无外部 API 连接；其两个子进程 PID 3365663 / 3368071 均为零文件描述符的 `Z` 状态，不能占用 API 并发。按用户清理废弃进程的要求，核对启动时间、精确 runner 路径及 cwd 后仅终止该暂停控制器，没有先恢复运行，三个 PID 均已消失。所有历史状态、账本、候选和 episode 文件保留；本地 Qwen/vLLM、数据库、原生环境服务及用户会话未动。清理回执在续跑目录 `process_cleanup.json`。

## 参数说明校验与后续轮次缓存修复（2026-10-05）

Retail 第 3 轮原 Evolve 已完成，但接入校验只允许工具顶层 `description`，误把 JSON Schema 参数的 `description` 文本变化当成结构改动。真实差异只有 3 个工具的说明文字：`exchange_delivered_order_items`、`modify_pending_order_items`、`return_delivered_order_items` 的 `item_ids` / `new_item_ids` 说明；名称、参数集合、类型、required、enum/default 和 dispatch 都未改变。只读独立探针替换比较用的文档注释后，完整原校验已通过，候选字节未变。

最小修复只在 schema 节点忽略文本 `description` 注释；属性名 `description`、默认值/枚举里的同名键仍参与结构比较，不删除或新增参数，不开放执行权限。原版 AHE commit `8b2a55d97590363fe50c3cc6b5e833b020a4bb4c` 无 tracked 修改。现有 `recovery.py` 在固定六轮范围内审计首个未完成阶段，支持第 3 轮原付费 Debugger/Evolve 缓存；已完成原版优化不重放，部分/不同批次/改动过/费用不明的产物仍拒绝。没有新增运行器或给算法添加搜索策略。

旧控制器把单个 Future 的异常推迟到其余场景完成后才写 `errors.json`，因此总状态曾仍显示 `training`。新入口会在发现失败后立即记录异常，同时让已在途的付费阶段结束；没有强杀当前 Telecom 评测或未完成的 Airline Evolve。源代码修复前副本在 repair5 的 `source_archive/capabilities.before_parameter_descriptions.py`、`recovery.before_general_round_resume.py`、`run.before_failure_reporting.py`，恢复审计核对这些原指纹。

离线全量回归：AHE 88 项、Meta/Life 228 项，共 316 项通过。AHE 首次运行的 18 个 socket 错误来自沙箱禁止 loopback，随后在仅本地模拟 HTTP、清空会打印的凭据变量的环境重验全部通过；未调用付费模型。新增验证六个环境参数注释可编辑、实际参数名/类型/枚举/required/default/dispatch 不可改、后续轮次缓存边界、付费第 3 轮缓存、未来未知/部分结果拒绝以及错误即时报告。

真实 Retail 第 3 轮缓存审计通过：100 条完整历史候选 episode、11 份原 Debugger 文件、188 个原 Evolve 缓存文件哈希，候选 SHA-256 `74386d20a11fa930d8909f697a201db3ccff97e526ff0bf7761e083e23a71f89`。第 3 轮实测优化费用 13,405,282 token，原始费用和候选均未修改，`optimizer_replayed=false`。待旧控制器退出并完成整体审计，原版从 iteration 3 / skip_eval=True 复用缓存、继续后续评测；不会重新付费生成候选。

16:22 UTC：旧 PID 2147656 的 Flash 闸门峰值 6、上限 8；Telecom 第 4 次训练评测约 19/50，测量第 3 轮候选，整批分数尚未知。新 PID 3036812 仅零 API 等待，不创建新 MR broker，当前仍只有一个实际调用 Flash 的实验控制器。测试成绩继续保持未测，不用部分训练结果代替。

### 普通消息字段补齐与原付费第 3 轮复用（2026-10-05 17:10 UTC）

旧控制器在 16:48 UTC 自然退出；零 API 等待入口 repair6 随后因 Airline 原候选的普通 `TextBlock.text` / `Role.value` 字段仍被拒而停止审计，未创建费用账本、未调用 API。NexAU 实际类型注解核验后，接入层补齐公开消息、内容块、响应及 hook 的数据字段。`sandbox`、executor、私有属性、新模型/工具/子代理和跨 episode 状态仍禁止；读取响应中已有的 sub-agent-call 数据不代表允许创建子代理。候选代码没有人工修改，官方 AHE 核心无 tracked 修改。

修复后的全量离线回归为 AHE 89 项、Meta/Life 228 项，共 **317 项通过**；新增实际 NexAU 类型注解清单测试，避免仅修一个字段而遗漏其余普通数据接口。真实整体续跑审计通过：Meta 18 轮不重放；AHE Airline/Retail/Telecom 分别复用 120/100/150 条完整 episode。Airline 第 3 轮原 Evolve 缓存 157 个文件、候选 SHA-256 `c794e3768525f57ca7803af9a66c3855b6c47b25ba73d03a8dc1a9a9fafded20`；Retail 第 3 轮 188 个文件、候选 SHA-256 `74386d20a11fa930d8909f697a201db3ccff97e526ff0bf7761e083e23a71f89`，均完整核对费用和输入身份。

repair7 的 PID 3388623 在确认旧进程退出后启动。17:10 UTC 两份 `iteration_003/evolve/completed_cache_receipt.json` 均确认 `optimizer_replayed=false`，原版流程已提升原候选并进入 iteration 4 评测；Telecom 复用完整 iteration 4 评测，从 Debugger/Evolve 4 继续。新控制器只创建一个共享 `RequestGate(8)`，当前峰值 3。前述旧控制器、零 API 等待器及废弃 Life runner/僵尸子进程均已再次只读确认不存在；历史文件全部保留。

17:43 UTC 再核验本次续跑新回执：MR 672 条已结束请求中 657 条 HTTP 200，均有实际 usage；另 15 条为明确 HTTP 429。253 条成功流式请求全部保留有效 usage、`[DONE]` 与正常 EOF，没有新增成功但费用缺失的请求。共享闸门已实际达到峰值 8，未超限。三个场景均正在原版第 4 轮 Evolve；最终六轮与 TEST 仍未完成。

## 最终固定 TEST

训练已封存，六组 TEST 全部完成；下表为完整最终保留版本的 TEST，不是部分进度、逐轮 TRAIN 或按 TEST 选出的最佳版本。

Life 发布版参考成绩由用户于 2026-10-05 明确提供：Airline 60%、Retail 63%、Telecom 57.5%。这是发布版参照，不是本次重新测量；其批次、试验次数和模拟器条件未在本次核对，不据此声称完全同条件比较。Meta/AHE 使用六轮训练封存后的保留版本，不能按 TEST 选优。最终同时列出相对发布版的百分点差值。

| 场景 | 本次 TEST 分母 | Meta 六轮后 | AHE 六轮后 | Life 发布版参考 | Meta / AHE 相对发布版百分点 |
| --- | ---: | --- | --- | ---: | --- |
| Airline | 20 | 4/20 = 20.00% | 7/20 = 35.00% | 60% | -40 / -25 |
| Retail | 40 | 16/40 = 40.00% | 12/40 = 30.00% | 63% | -23 / -33 |
| Telecom | 40 | 20/40 = 50.00% | 37/40 = 92.50% | 57.5% | -7.5 / +35 |

2026-10-05 22:18 UTC：Airline/Retail 四组 TEST 全部完整结束，均无基础设施错误、缺失或分母变动。两组 Telecom 正在继续。已结束分数只反映本次固定条件；发布版数字为用户参考，不是重新测量，也不能据此推断新版 Life 的实测表现。测试阶段原 progress.json 只保留封存时快照，实际在途 MR 请求改从逐请求 started/finished 账本核对；封存后 Debugger/Evolve 调用为 0，未开放 TEST 反馈给优化器。

2026-10-05 23:22 UTC：AHE Telecom 完整 TEST 为 37/40 = 92.50%，infra_errors=0、missing_episodes=0；三个 AHE 场景的最终 TEST 均结束。Telecom 高于用户发布版参考 35 个百分点，不能宣称所有场景 AHE 均弱于发布版。Meta Telecom 仍在评测，已返回 11 条原生记录，不据部分任务估分。当前账本在途 DeepSeek 请求为 8，全部属于用户模拟；优化费用未增加。全 harness 路径已复查：仅当前控制器、活跃原生 TEST 子进程和保留的本地 Qwen/代理/数据库/AgentRL/用户会话等常驻服务，无旧优化器；废弃 runner 和僵尸子进程不存在。另一个未标识的常驻 Python 进程无 TCP 监听或已建立连接，未误删。

用于后续 token 曲线时，若纵轴表示每轮最终保留版本：Meta 接收时使用 candidate TRAIN，拒绝/失败时使用 incumbent TRAIN，不能把“候选未测”画成保留版本 0%；AHE 使用本表对应候选测量，明确标注其测量批次。TEST 点单独展示，不能混作逐轮 TRAIN 或按 TEST 挑选版本。预算横轴需保留实测/历史估算及修复分支费用的区别。

2026-10-05 23:46 UTC 最终核验通过：六组 TEST 都为固定 task_ids × 1 次，Airline 分母 20、Retail/Telecom 分母 40，均 infra_errors=0、missing_episodes=0。Meta 原六轮、完整候选/费用回执和最终保留源码再次通过 `audit_meta_cached`；AHE 三份末轮 TRAIN 再次通过任务/分数/workspace 审计。当前 Meta/AHE workspace、AHE TEST 快照与 training_seal 哈希一致，冻结源文件指纹未变，最终 OPTIMIZER_TOKEN_USAGE 与 seal 中费用完全相同。Meta 实测优化 88,774,469 token；AHE 实测 245,852,148，加单笔历史估算 135,307，含估算 245,987,455。全部 API 开始记录已闭合，除显式 429 和已注明的历史估算/排除角色旧缺失之外无未核清优化费用；封存后无 Debugger/Evolve 新调用。官方 AHE commit 仍为 `8b2a55d97590363fe50c3cc6b5e833b020a4bb4c`，无 tracked 修改。最终控制器及 TEST 子进程退出、相关临时监听关闭；本地 Qwen 代理 8400 和正常服务保留，未启动新版 Life AgentBench 实验。

2026-10-05 13:33 UTC 核验：Meta 三场景均六轮结束；AHE Telecom 正在 iteration 3 TRAIN 测量，11/50 条完成，续跑器仍零 API 等待旧控制器退出。Telecom Evolve 2 有一条成功状态但无用量回执的流式请求（`e18eb75c714448f48b6b40180dac0c21`，HTTP 200、raw_usage 为空）。该组 129 条有用量调用合计 12,427,963 token，与原版 tracer 的总 token 恰好一致，因此 tracer 也不能补出这条缺失用量；另有 3 条明确 429。原账本保持未知，不记零、不估算、不重跑已付费 Evolve。严格续跑审计会在旧控制器退出后拒绝不完整优化费用，已向用户询问是否明确保留未知费用并以已知下界继续。

最终证据位于 `../meta_ahe_taubench_six_flash_tokens_20261004_repair8/` 的 `protocol.json`、`continuation_receipt.json`、`TRAINING_RESULTS.json`、`OPTIMIZER_TOKEN_USAGE.json`、`training_seal.json`、`FINAL_RESULTS.json`、`status.json`、`progress.json` 及完整逐请求账本，全部已生成且终态为 complete。本 Markdown 作为统一留档，前序目录和原始费用均保留。

Life 新版位于 `meta/life_evolve/`：短主提示、按需不限案例读取、机制/策略历史、H2–H5 空接口、允许增改删与转向、最多 3 次返修、无总交互次数上限、保留最新合格候选。2026-10-05 补齐“已有合格候选后 API 报上下文耗尽”的恢复分支，仅复用现有快照恢复逻辑；无合格候选仍失败，不额外重试或消耗返修机会。新增两项测试先复现一项失败，修复后两项通过。默认付费 API 入口另补调用现有 `verify_usage`：缺失、空或无效费用回执正常报错，不记零费用；新增一项测试的三种模拟响应均先复现失败再通过。Meta/Life 全量离线回归 220 项通过。修改不在 Tau 冻结运行源范围内；未启动付费 Life 优化或新版 Life 的 AgentBench 全量六轮，不据此宣称其新版本性能优势。
