# 七场景 Meta / AHE / Life 六轮实验结果汇总

汇总时间：2026-10-07 08:02:43 UTC；OS 原版六轮结果并入：2026-10-09。

范围：AgentBench 的 ALFWorld、DBBench、WebShop、OS，以及 TauBench 的 Airline、Retail、Telecom。仅汇总已有结果，不新增或重跑付费实验，不修改历史实验产物。

## 1. 完成情况与比较口径

- 七个场景 × 三种方法，共21组实验，均已有六次优化尝试的记录和最终 TEST 结果；以下保留126个逐轮训练／开销数据点。
- “六轮完成”不等于六轮都生成并评测了新候选。Meta AgentBench 的 ALFWorld/WebShop 第5轮，以及 Meta TauBench 的 Retail 第2轮、Telecom 第2～6轮，候选未测或没有有效候选；表中标记为“†”，保留当轮现任版本的实测 TRAIN 分数，不能将缺失候选分数写成0%。
- TEST 表使用各实验依据 TRAIN 选择并冻结的最终测试版本；不是用 TEST 从六轮中挑最高分。AHE 保留原版选择逻辑，不人为套用 Meta/Life 的严格提升接纳规则。
- Meta/Life 的 TRAIN 表列出当轮保留版本分数：接受候选则取候选分数，否则取当轮现任分数。AHE 列出该轮优化产物后续测量的分数，未必是原版 best-ever 选中的版本。
- AgentBench 每批50条 TRAIN；TauBench Retail/Telecom 每批50条 TRAIN；TauBench Airline 为完整30条 TRAIN × 2次试验，共60条 episode。TEST 每题1次；完整分母含基础设施失败，失败计0。
- 多数场景会换训练批次；Airline 即使使用同一30题池也存在模型／用户对话随机性。跨轮准确率不是固定 TEST 集的单调学习曲线，六轮相同也不代表 token 预算相同。
- 外部优化模型为 DeepSeek V4 Flash API，本地求解器为 Qwen3-4b。优化 token 只计输入＋输出；缓存输入不重复相加，不包含求解器、用户模拟器、NL judge 或测试做题开销，也不等同于人民币费用。
- 当前 Life 汇总并非七域同一代码版本的联合实验：ALFWorld/DBBench 取10月4日修复续跑；WebShop、Telecom、Airline/Retail 取10月6～7日各自最新完成版本；OS 取10月8日原版 Life 与 Meta/AHE 的六轮实验，Life 保留第4轮产物，不纳入后续新版 prompt 对照实验。ALFWorld/DBBench 记录 `prior_test_exposure=true`、`test_used_for_optimization=false`，不是项目首次盲测。
- AHE AgentBench 四场景均使用 `h2345-budget-restricted-v1` 限制版：每轮最多精读10条详细案例，禁用新增工具、修改工具实现、额外运行时LLM、子代理、单工具隐藏多次环境调用、跨任务可写记忆。这里不把该限制版冒称为无限制原版AHE，也不把此案例限额推广为 Meta/Life 限额。
- OS 使用原生 AgentBench 任务、工具和判分，回合上限16；三方法完整 TEST144 在六轮 TRAIN 选择冻结后运行。AHE 的 best-ever 取原版 iteration004 的输入工作区（第3轮优化产物），不能用第6轮额外 TRAIN 测量替代最终选中版本。

## 2. 最终 TEST 准确率

发布版参考是用户提供值，不是本次重新测试，也没有对应的六轮优化 token 账本；ALFWorld 发布版参考未提供。

| 套件 | 场景 | TEST分母 | Meta | AHE | Life（当前所选六轮版） | 发布版参考 | TEST基础设施失败：Meta / AHE / Life |
|---|---|---:|---:|---:|---:|---:|---|
| AgentBench | ALFWorld | 109 | 35.78%（39/109） | 95.41%（104/109） | 88.07%（96/109） | 未提供 | 1 / 0 / 1 |
| AgentBench | DBBench | 300 | 50.33%（151/300） | 52%（156/300） | 65%（195/300） | 65% | 0 / 0 / 0 |
| AgentBench | WebShop | 200 | 42.50%（85/200） | 40.50%（81/200） | 46%（92/200） | 45.50% | 0 / 0 / 0 |
| AgentBench | OS | 144 | 25.69%（37/144） | 37.50%（54/144） | 33.33%（48/144） | 未提供 | 4 / 2 / 4 |
| TauBench | Airline | 20 | 20%（4/20） | 35%（7/20） | 40%（8/20） | 60% | 0 / 0 / 0 |
| TauBench | Retail | 40 | 40%（16/40） | 30%（12/40） | 50%（20/40） | 63% | 0 / 0 / 0 |
| TauBench | Telecom | 40 | 50%（20/40） | 92.50%（37/40） | 65%（26/40） | 57.50% | 0 / 0 / 0 |

注意：AHE 历史 AgentBench `COMPARISON.md` 中的 Meta/Life 列属于更早的实验。本报告没有沿用那些旧列；Meta 换成10月4日有实测 token 的六轮重跑，Life 换成上文明确列出的当前版本。ALFWorld/DBBench/WebShop 的 AHE 测试结果仍来自该历史限制版的固定最终测试；OS 三方法来自10月8日原实验。

## 3. 六轮外部优化 token 总量

单位：输入＋输出 token，整数。AgentBench ALFWorld/DBBench/WebShop 的 AHE“作图预算”是约定估算情景，其“已记录”只是 Evolve 已知下界；OS 与 TauBench 的 AHE 已记录项包含 Debugger＋Evolve。Life OS 的单次超时估算独立列出，作图预算列均为已记录＋明确标注的估算。

| 场景 | Meta实测 | Life实测（含本轮系谱的失败／返修） | Life补充估算 | Life作图预算：实测＋估算 | AHE已记录 | AHE补充估算 | AHE作图预算：已记录＋估算 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ALFWorld | 37,138,309 | 24,288,432 | 0 | 24,288,432 | 63,532,347 | 32,100,000 | 95,632,347 |
| DBBench | 32,813,601 | 11,204,889 | 0 | 11,204,889 | 72,424,554 | 32,100,000 | 104,524,554 |
| WebShop | 36,741,591 | 37,750,993 | 0 | 37,750,993 | 71,743,620 | 32,100,000 | 103,843,620 |
| OS | 26,787,235 | 25,566,110 | 143,658 | 25,709,768 | 93,994,363 | 0 | 93,994,363 |
| Airline | 53,088,352 | 36,142,791 | 0 | 36,142,791 | 64,497,756 | 0 | 64,497,756 |
| Retail | 14,363,326 | 25,706,750 | 0 | 25,706,750 | 93,141,101 | 0 | 93,141,101 |
| Telecom | 21,322,791 | 30,455,997 | 0 | 30,455,997 | 88,213,291 | 135,307 | 88,348,598 |
| 七场景合计 | 222,255,205 | 191,115,962 | 143,658 | 191,259,620 | 547,547,032 | 96,435,307 | 643,982,339 |

### 3.1 估算约定与用量缺口

- AgentBench ALFWorld/DBBench/WebShop 沿用原留档的缺失调用平均50,000 token情景，不把所有调用重估为50,000：已记录 Evolve 207,700,521；broker attempts 3,931；tracer generations 2,005；计数差1,926 × 50,000＝96,300,000。
- 缺失调用不能恢复到真实场景／轮次；仅为估算情景均摊到18个单元，每场景每轮补5,350,000，每场景六轮补32,100,000。逐轮均摊不是恢复出的实测分配，也不是置信区间。
- attempts 与 generations 在失败／重试时未必一一对应，差值不能断言全部来自 Debugger；WebShop 首轮复用的更早 Debugger 和修复前归档启动成本不在完整计数内。因此这三个历史 AgentBench 场景的 AHE 全历史开销仍未完整恢复；OS 不套用该50k均摊估算，其六轮93,994,363全部实测（Debugger 11,952,415＋Evolve 82,041,948）。
- TauBench AHE Airline/Retail 的已记录推理用量无额外估算；分别另有23/19次429拒绝请求未返回 usage。原账本标记 `inference_accounting_complete=true`，不将这些拒绝请求假装成已消耗推理 token。
- TauBench AHE Telecom 第2轮有1次缺失推理 usage，原账本补估135,307（输入133,616＋输出1,691）；另有28次429拒绝无 usage。其六轮已记录88,213,291，含估算预算88,348,598，不能标为全部实测。

- Life OS 第1轮有1次未返回 usage 的超时调用，按用户授权及归档估算规则补143,658（输入104,091＋输出39,567）。六轮实测25,566,110、含估算预算25,709,768；账本 `accounting_complete=false`、`accounting_usable_with_estimates=true`，不能写成全部实测。

### 3.2 历史启动与续跑费用

- Meta AgentBench 六轮表外另有前置失败／启动成本 2,490,771；其三场景六轮106,693,501，加启动109,184,272。未将这笔无法按本表轮次归并的启动费偷偷删为0；主曲线按既有六轮口径，全部已知支出另记。
- Life ALFWorld/DBBench 不是只统计修复后的新请求：合并 `ancestor_usage.json` 与新 `TOKEN_USAGE.json`，同一旧调用只计一次，废弃／失败分支也保留。
- Life ALFWorld 的旧run在新run成功完成第2轮前，已经花掉旧第3轮失败提案的3,900,206 token。下方为了作图不漏掉提前支出，将这笔费用计入当前合格第2轮的成本里程碑：第2轮8,029,314、第3轮1,886,019；累计第2轮12,190,452。原ledger按round标签归并则是第2轮4,129,108、第3轮5,786,225；两种归并六轮总计均24,288,432。
- Life DBBench 第4轮包括祖先失败第4轮1,924,480＋修复后396,779＝2,321,259。不能仅用新run的396,779作为全部第4轮成本。
- 旧 Life AgentBench 三域 `TOTAL_SPEND.json` 为49,292,551，其中旧WebShop13,799,230不属于本报告选用的新WebShop六轮。新WebShop从空内容单独重跑，费用37,750,993；不混用旧WebShop分数／费用，也不声称本表涵盖所有项目研发和历次重跑成本。
- OS 首轮 AHE Debugger 子进程因隔离环境缺少 `agent_debugger_core` 而未成功运行；首轮 Evolve 仍读取10条轨迹并保留已付费用量。后续修复隔离依赖后继续官方流程，未重放首轮付费 Evolve；该启动限制保留为实验边界。Life 的超时、重试与续跑费用按原账本归并到第1轮，不混入新版 prompt 对照实验的开销。

## 4. 逐轮 TRAIN 准确率与作图数据

每个表中的累计 token 只在同场景、同方法内累计。标记 M＝实测；E50＝AHE AgentBench缺失调用50k、跨场景跨轮均摊情景；M+E＝AHE Telecom或Life OS含单次usage估算；†＝新候选未测或未产生，TRAIN分数来自现任版本。

AHE 通常在下一训练批次测量上一轮优化产物，第6轮产物在最后一批额外测量且不反馈优化器。批次字段是测分批次，不代表三方法同轮使用同一题组。下方不得标成 TEST 学习曲线。

### 4.1 AgentBench / ALFWorld

| 轮次 | 方法 | 测分训练批次 | TRAIN准确率 | 当轮优化token | 累计优化token | 用量／候选标记 |
|---:|---|---|---:|---:|---:|---|
| 1 | Meta | b0 | 32% | 1,844,422 | 1,844,422 | M；接收 |
| 1 | AHE | b1 | 20% | 19,811,543 | 19,811,543 | E50；当轮产物 |
| 1 | Life | b0 | 28% | 4,161,138 | 4,161,138 | M；不接收 |
| 2 | Meta | b0 | 34% | 6,160,755 | 8,005,177 | M；接收 |
| 2 | AHE | b2 | 64% | 14,592,583 | 34,404,126 | E50；当轮产物 |
| 2 | Life | b1 | 28% | 8,029,314 | 12,190,452 | M；接收；见3.2成本归并 |
| 3 | Meta | b0 | 36% | 3,582,465 | 11,587,642 | M；接收 |
| 3 | AHE | b3 | 98% | 13,854,957 | 48,259,083 | E50；当轮产物 |
| 3 | Life | b2 | 72% | 1,886,019 | 14,076,471 | M；接收；见3.2成本归并 |
| 4 | Meta | b0 | 36% | 8,843,556 | 20,431,198 | M；不接收 |
| 4 | AHE | b4 | 96% | 13,393,733 | 61,652,816 | E50；当轮产物 |
| 4 | Life | b3 | 82% | 5,157,572 | 19,234,043 | M；接收 |
| 5 | Meta | b1 | 36%† | 5,698,336 | 26,129,534 | M；†候选未测，沿用 |
| 5 | AHE | b5 | 96% | 22,034,715 | 83,687,531 | E50；当轮产物 |
| 5 | Life | b4 | 86% | 3,848,217 | 23,082,260 | M；接收 |
| 6 | Meta | b2 | 38% | 11,008,775 | 37,138,309 | M；接收 |
| 6 | AHE | b5 | 98% | 11,944,816 | 95,632,347 | E50；当轮产物 |
| 6 | Life | b5 | 88% | 1,206,172 | 24,288,432 | M；接收 |

### 4.2 AgentBench / DBBench

| 轮次 | 方法 | 测分训练批次 | TRAIN准确率 | 当轮优化token | 累计优化token | 用量／候选标记 |
|---:|---|---|---:|---:|---:|---|
| 1 | Meta | b0 | 50% | 6,207,521 | 6,207,521 | M；接收 |
| 1 | AHE | b1 | 62% | 22,526,526 | 22,526,526 | E50；当轮产物 |
| 1 | Life | b0 | 60% | 1,440,875 | 1,440,875 | M；接收 |
| 2 | Meta | b0 | 52% | 4,921,970 | 11,129,491 | M；接收 |
| 2 | AHE | b2 | 52% | 17,026,692 | 39,553,218 | E50；当轮产物 |
| 2 | Life | b1 | 66% | 1,673,309 | 3,114,184 | M；接收 |
| 3 | Meta | b0 | 54% | 4,333,441 | 15,462,932 | M；接收 |
| 3 | AHE | b3 | 52% | 19,824,235 | 59,377,453 | E50；当轮产物 |
| 3 | Life | b2 | 72% | 2,238,186 | 5,352,370 | M；接收 |
| 4 | Meta | b0 | 54% | 3,948,293 | 19,411,225 | M；不接收 |
| 4 | AHE | b4 | 40% | 14,787,920 | 74,165,373 | E50；当轮产物 |
| 4 | Life | b3 | 58% | 2,321,259 | 7,673,629 | M；不接收 |
| 5 | Meta | b1 | 50% | 4,556,989 | 23,968,214 | M；接收 |
| 5 | AHE | b5 | 44% | 14,944,355 | 89,109,728 | E50；当轮产物 |
| 5 | Life | b4 | 56% | 2,417,307 | 10,090,936 | M；不接收 |
| 6 | Meta | b1 | 50% | 8,845,387 | 32,813,601 | M；不接收 |
| 6 | AHE | b5 | 40% | 15,414,826 | 104,524,554 | E50；当轮产物 |
| 6 | Life | b5 | 70% | 1,113,953 | 11,204,889 | M；不接收 |

### 4.3 AgentBench / WebShop

| 轮次 | 方法 | 测分训练批次 | TRAIN准确率 | 当轮优化token | 累计优化token | 用量／候选标记 |
|---:|---|---|---:|---:|---:|---|
| 1 | Meta | b0 | 40% | 2,192,980 | 2,192,980 | M；接收 |
| 1 | AHE | b1 | 38% | 19,530,428 | 19,530,428 | E50；当轮产物 |
| 1 | Life | b0 | 38% | 6,778,022 | 6,778,022 | M；接收 |
| 2 | Meta | b0 | 48% | 3,302,768 | 5,495,748 | M；接收 |
| 2 | AHE | b2 | 36% | 12,087,780 | 31,618,208 | E50；当轮产物 |
| 2 | Life | b1 | 42% | 3,513,241 | 10,291,263 | M；接收 |
| 3 | Meta | b0 | 48% | 5,852,370 | 11,348,118 | M；不接收 |
| 3 | AHE | b3 | 42% | 20,397,351 | 52,015,559 | E50；当轮产物 |
| 3 | Life | b2 | 48% | 8,290,227 | 18,581,490 | M；不接收 |
| 4 | Meta | b1 | 46% | 10,938,673 | 22,286,791 | M；不接收 |
| 4 | AHE | b4 | 34% | 13,919,712 | 65,935,271 | E50；当轮产物 |
| 4 | Life | b3 | 36% | 8,405,844 | 26,987,334 | M；不接收 |
| 5 | Meta | b2 | 42%† | 7,947,951 | 30,234,742 | M；†候选未测，沿用 |
| 5 | AHE | b5 | 42% | 13,940,160 | 79,875,431 | E50；当轮产物 |
| 5 | Life | b4 | 38% | 5,193,504 | 32,180,838 | M；接收 |
| 6 | Meta | b3 | 38% | 6,506,849 | 36,741,591 | M；接收 |
| 6 | AHE | b5 | 36% | 23,968,189 | 103,843,620 | E50；当轮产物 |
| 6 | Life | b5 | 42% | 5,570,155 | 37,750,993 | M；接收 |

### 4.4 AgentBench / OS

| 轮次 | 方法 | 测分训练批次 | TRAIN准确率 | 当轮优化token | 累计优化token | 用量／候选标记 |
|---:|---|---|---:|---:|---:|---|
| 1 | Meta | b0 | 54% | 4,498,792 | 4,498,792 | M；接收 |
| 1 | AHE | b1 | 36% | 9,228,392 | 9,228,392 | M；当轮产物 |
| 1 | Life | b0 | 54% | 2,155,531 | 2,155,531 | M+E；不接收 |
| 2 | Meta | b0 | 54% | 1,718,914 | 6,217,706 | M；不接收 |
| 2 | AHE | b2 | 56% | 11,083,628 | 20,312,020 | M；当轮产物 |
| 2 | Life | b1 | 54% | 2,313,269 | 4,468,800 | M；接收 |
| 3 | Meta | b1 | 40% | 7,753,235 | 13,970,941 | M；不接收 |
| 3 | AHE | b3 | 60% | 14,171,680 | 34,483,700 | M；当轮产物 |
| 3 | Life | b2 | 52% | 4,280,113 | 8,748,913 | M；接收 |
| 4 | Meta | b2 | 38% | 1,314,114 | 15,285,055 | M；不接收 |
| 4 | AHE | b4 | 60% | 17,517,322 | 52,001,022 | M；当轮产物 |
| 4 | Life | b3 | 66% | 5,779,790 | 14,528,703 | M；接收 |
| 5 | Meta | b3 | 44% | 8,515,180 | 23,800,235 | M；不接收 |
| 5 | AHE | b5 | 54% | 16,598,896 | 68,599,918 | M；当轮产物 |
| 5 | Life | b4 | 56% | 5,287,817 | 19,816,520 | M；不接收 |
| 6 | Meta | b4 | 44% | 2,987,000 | 26,787,235 | M；不接收 |
| 6 | AHE | b5 | 56% | 25,394,445 | 93,994,363 | M；当轮产物 |
| 6 | Life | b5 | 62% | 5,893,248 | 25,709,768 | M；不接收 |

### 4.5 TauBench / Airline

| 轮次 | 方法 | 测分训练批次 | TRAIN准确率 | 当轮优化token | 累计优化token | 用量／候选标记 |
|---:|---|---|---:|---:|---:|---|
| 1 | Meta | b0 | 30% | 2,622,885 | 2,622,885 | M；接收 |
| 1 | AHE | b1 | 10% | 11,435,194 | 11,435,194 | M；当轮产物 |
| 1 | Life | b0 | 56.67% | 17,892,199 | 17,892,199 | M；接收 |
| 2 | Meta | b0 | 30% | 5,212,757 | 7,835,642 | M；不接收 |
| 2 | AHE | b2 | 30% | 8,392,342 | 19,827,536 | M；当轮产物 |
| 2 | Life | b1 | 56.67% | 3,840,269 | 21,732,468 | M；不接收 |
| 3 | Meta | b1 | 30% | 8,064,329 | 15,899,971 | M；接收 |
| 3 | AHE | b3 | 45% | 13,269,875 | 33,097,411 | M；当轮产物 |
| 3 | Life | b2 | 55% | 3,799,763 | 25,532,231 | M；不接收 |
| 4 | Meta | b1 | 30% | 9,981,132 | 25,881,103 | M；不接收 |
| 4 | AHE | b4 | 41.67% | 12,667,408 | 45,764,819 | M；当轮产物 |
| 4 | Life | b3 | 58.33% | 3,458,893 | 28,991,124 | M；接收 |
| 5 | Meta | b2 | 30% | 13,479,032 | 39,360,135 | M；接收 |
| 5 | AHE | b5 | 41.67% | 10,350,181 | 56,115,000 | M；当轮产物 |
| 5 | Life | b4 | 55% | 3,992,155 | 32,983,279 | M；接收 |
| 6 | Meta | b2 | 30% | 13,728,217 | 53,088,352 | M；不接收 |
| 6 | AHE | b5 | 41.67% | 8,382,756 | 64,497,756 | M；当轮产物 |
| 6 | Life | b5 | 60% | 3,159,512 | 36,142,791 | M；接收 |

### 4.6 TauBench / Retail

| 轮次 | 方法 | 测分训练批次 | TRAIN准确率 | 当轮优化token | 累计优化token | 用量／候选标记 |
|---:|---|---|---:|---:|---:|---|
| 1 | Meta | b0 | 32% | 1,661,897 | 1,661,897 | M；不接收 |
| 1 | AHE | b1 | 40% | 17,352,611 | 17,352,611 | M；当轮产物 |
| 1 | Life | b0 | 40% | 3,725,478 | 3,725,478 | M；不接收 |
| 2 | Meta | b1 | 38%† | 1,892,981 | 3,554,878 | M；†候选未测，沿用 |
| 2 | AHE | b2 | 40% | 10,731,847 | 28,084,458 | M；当轮产物 |
| 2 | Life | b1 | 32% | 3,529,400 | 7,254,878 | M；不接收 |
| 3 | Meta | b2 | 38% | 1,441,959 | 4,996,837 | M；不接收 |
| 3 | AHE | b3 | 50% | 13,405,282 | 41,489,740 | M；当轮产物 |
| 3 | Life | b2 | 48% | 2,817,392 | 10,072,270 | M；不接收 |
| 4 | Meta | b3 | 48% | 4,092,266 | 9,089,103 | M；不接收 |
| 4 | AHE | b4 | 44% | 28,289,115 | 69,778,855 | M；当轮产物 |
| 4 | Life | b3 | 30% | 10,124,346 | 20,196,616 | M；不接收 |
| 5 | Meta | b4 | 36% | 1,539,914 | 10,629,017 | M；不接收 |
| 5 | AHE | b5 | 32% | 11,475,061 | 81,253,916 | M；当轮产物 |
| 5 | Life | b4 | 50% | 2,511,307 | 22,707,923 | M；接收 |
| 6 | Meta | b5 | 40% | 3,734,309 | 14,363,326 | M；不接收 |
| 6 | AHE | b5 | 20% | 11,887,185 | 93,141,101 | M；当轮产物 |
| 6 | Life | b5 | 42% | 2,998,827 | 25,706,750 | M；接收 |

### 4.7 TauBench / Telecom

| 轮次 | 方法 | 测分训练批次 | TRAIN准确率 | 当轮优化token | 累计优化token | 用量／候选标记 |
|---:|---|---|---:|---:|---:|---|
| 1 | Meta | b0 | 30% | 2,324,074 | 2,324,074 | M；不接收 |
| 1 | AHE | b1 | 46% | 18,510,187 | 18,510,187 | M；当轮产物 |
| 1 | Life | b0 | 36% | 8,365,576 | 8,365,576 | M；接收 |
| 2 | Meta | b1 | 32%† | 2,751,245 | 5,075,319 | M；†候选未测，沿用 |
| 2 | AHE | b2 | 54% | 16,533,223 | 35,043,410 | M+E；当轮产物 |
| 2 | Life | b1 | 78% | 3,089,408 | 11,454,984 | M；接收 |
| 3 | Meta | b2 | 28%† | 2,169,234 | 7,244,553 | M；†候选未测，沿用 |
| 3 | AHE | b3 | 86% | 14,144,198 | 49,187,608 | M；当轮产物 |
| 3 | Life | b2 | 78% | 5,298,445 | 16,753,429 | M；不接收 |
| 4 | Meta | b3 | 28%† | 4,572,550 | 11,817,103 | M；†候选未测，沿用 |
| 4 | AHE | b4 | 88% | 14,086,441 | 63,274,049 | M；当轮产物 |
| 4 | Life | b3 | 82% | 4,733,975 | 21,487,404 | M；不接收 |
| 5 | Meta | b4 | 28%† | 5,341,751 | 17,158,854 | M；†候选未测，沿用 |
| 5 | AHE | b5 | 88% | 12,609,327 | 75,883,376 | M；当轮产物 |
| 5 | Life | b4 | 88% | 5,644,578 | 27,131,982 | M；不接收 |
| 6 | Meta | b5 | 30%† | 4,163,937 | 21,322,791 | M；†候选未测，沿用 |
| 6 | AHE | b5 | 100% | 12,465,222 | 88,348,598 | M；当轮产物 |
| 6 | Life | b5 | 76% | 3,324,015 | 30,455,997 | M；不接收 |

## 5. 当前结论与使用边界

- 当前 Life 的最终 TEST 准确率七域均高于本次 Meta；高于AHE的四域为 DBBench、WebShop、Airline、Retail；ALFWorld、Telecom与OS仍低于AHE。不应只挑有利域汇报。
- 相对用户发布版参考，Life DBBench持平，WebShop高0.5个百分点，Telecom高7.5个百分点；Airline低20个百分点，Retail低13个百分点。ALFWorld/OS缺少用户提供的发布版参考，不能从旧训练smoke推断其发布版全TEST结果。
- Life 的七域优化 token（OS包含单次超时估算）都低于AHE的已记录量；但并非每域低于Meta：Life的WebShop、Retail、Telecom高于Meta，ALFWorld、DBBench、Airline、OS低于Meta。
- 后期仍有同轮有效改进：Airline第4～6轮、Retail/WebShop第5～6轮接收；OS第2～4轮也接收新候选；但DBBench第4～6轮、Telecom第3～6轮、OS第5～6轮没有接收新候选。不能声称所有场景一直上涨，或保证继续增加轮次就达到发布版。
- 横轴建议使用本报告明确标注口径的累计外部优化 token；纵轴若使用上述逐轮点必须标注TRAIN及可变训练批次。本报告每方法每域只收录一份所选版本最终TEST，不能将TRAIN点伪装成逐轮TEST点。
- 主实验是六次优化尝试预算下的结果，非等token预算实验；不同框架、候选接纳、能力限制、代码版本和部分测试条件不同。单次测试与估算曲线不支持统计显著性声明。
- Meta候选缺失与AHEusage缺口已保留原样；本文不补造候选成绩、不偷偷重跑、不从最终TEST反向选择更好检查点。

## 6. 原始记录位置与可追溯性

- 旧AgentBench留档：[AHE/Meta逐轮记录](/mnt/data/xts/harness/AHE_META_SIX_ROUND_TRAIN_TOKENS_20261004.md)，其AHE估算情景已在本报告继承。
- Meta AgentBench：[最终测试](/mnt/data/xts/harness/Life-Harness/meta/experiments/meta_agentbench_six_flash_tokens_20261004_repair2/FINAL_RESULTS.json)、[六轮与启动usage](/mnt/data/xts/harness/Life-Harness/meta/experiments/meta_agentbench_six_flash_tokens_20261004_repair2/TOKEN_USAGE.json)、[逐轮训练](/mnt/data/xts/harness/Life-Harness/meta/experiments/meta_agentbench_six_flash_tokens_20261004_repair2/TRAINING.md)。
- Life ALFWorld/DBBench：[最终测试](/mnt/data/xts/harness/Life-Harness/meta/experiments/life_agentbench_flash100k_repair_20261004/FINAL_RESULTS.json)、[祖先usage](/mnt/data/xts/harness/Life-Harness/meta/experiments/life_agentbench_flash100k_repair_20261004/ancestor_usage.json)、[新usage](/mnt/data/xts/harness/Life-Harness/meta/experiments/life_agentbench_flash100k_repair_20261004/TOKEN_USAGE.json)；各域state包含完整六轮记录。
- Life WebShop：[最终测试](/mnt/data/xts/harness/Life-Harness/meta/experiments/life_webshop_six_current_20261007_binding_resume/FINAL_RESULTS.json)、[逐轮训练与usage](/mnt/data/xts/harness/Life-Harness/meta/experiments/life_webshop_six_current_20261007_binding_resume/TRAINING_RESULTS.json)。
- Meta/AHE TauBench：[最终测试](/mnt/data/xts/harness/Life-Harness/meta/experiments/meta_ahe_taubench_six_flash_tokens_20261004_repair8/FINAL_RESULTS.json)、[逐轮训练](/mnt/data/xts/harness/Life-Harness/meta/experiments/meta_ahe_taubench_six_flash_tokens_20261004_repair8/TRAINING_RESULTS.json)、[优化usage](/mnt/data/xts/harness/Life-Harness/meta/experiments/meta_ahe_taubench_six_flash_tokens_20261004_repair8/OPTIMIZER_TOKEN_USAGE.json)。
- Life Telecom：[最终测试](/mnt/data/xts/harness/Life-Harness/meta/experiments/life_taubench_telecom_six_20261006_codegate_resume/FINAL_RESULTS.json)、[逐轮训练与usage](/mnt/data/xts/harness/Life-Harness/meta/experiments/life_taubench_telecom_six_20261006_codegate_resume/TRAINING_RESULTS.json)。
- Life Airline/Retail：[最终测试](/mnt/data/xts/harness/Life-Harness/meta/experiments/life_taubench_airline_retail_six_20261007_trace_resume/FINAL_RESULTS.json)、[总usage](/mnt/data/xts/harness/Life-Harness/meta/experiments/life_taubench_airline_retail_six_20261007_trace_resume/OPTIMIZER_TOKEN_USAGE.json)、[Airline逐轮](/mnt/data/xts/harness/Life-Harness/meta/experiments/life_taubench_airline_retail_six_20261007_trace_resume/airline/TRAINING_RESULTS.json)、[Retail逐轮](/mnt/data/xts/harness/Life-Harness/meta/experiments/life_taubench_airline_retail_six_20261007_trace_resume/retail/TRAINING_RESULTS.json)。
- AHE AgentBench：[固定最终TEST报告](/mnt/data/xts/harness/ahe-native/heldout_tests/ahe_restricted_six_rotating_flash8_20261003_073330_repair1_finaltest/COMPARISON.md)、[原始token审计](/mnt/data/xts/harness/ahe-native/audits/historical_tokens_20261004/TOKEN_AUDIT.json)。
- OS Meta/AHE/原版Life：[统一结果](/data/home/xts/harness/Life-Harness/meta/experiments/os_meta_ahe_life_six_20261008_setup3/BENCHMARK_RESULTS.json)、[逐轮训练与费用CSV](/data/home/xts/harness/Life-Harness/meta/experiments/os_meta_ahe_life_six_20261008_setup3/SIX_ROUND_RESULTS.csv)、[最终TEST](/data/home/xts/harness/Life-Harness/meta/experiments/os_meta_ahe_life_six_20261008_setup3/FINAL_RESULTS.json)、[训练原记录](/data/home/xts/harness/Life-Harness/meta/experiments/os_meta_ahe_life_six_20261008_setup3/TRAINING_RESULTS.json)、[优化usage](/data/home/xts/harness/Life-Harness/meta/experiments/os_meta_ahe_life_six_20261008_setup3/OPTIMIZER_TOKEN_USAGE.json)、[冻结选择](/data/home/xts/harness/Life-Harness/meta/experiments/os_meta_ahe_life_six_20261008_setup3/training_seal.json)。

### 源文件快照 SHA-256

历史六场景路径相对于 `/mnt/data/xts/harness`；新增 OS 快照使用本机 `/data/home/xts/harness` 绝对路径。原六场景已校验108个逐轮数据点和18份最终结果；并入 OS 后共126个数据点、21份最终结果，新增18行的合计、完整 TEST144 成功数及实验complete状态均已核对。本文仅更新汇总，未变更原始实验证据。

```text
b41bea803dacd4698ec6c80f831a2051ba6e6f08c15032d8ef7ea13d1141c74a  Life-Harness/meta/experiments/meta_agentbench_six_flash_tokens_20261004_repair2/FINAL_RESULTS.json
5fc40100632a75406085ec94ead870927f4fc10f36cf743bc3bd285765856d00  Life-Harness/meta/experiments/meta_agentbench_six_flash_tokens_20261004_repair2/TOKEN_USAGE.json
22e968460eb64050aca6233d0d13145954a62930e0f91737f0744a9ade08a0b3  Life-Harness/meta/experiments/meta_agentbench_six_flash_tokens_20261004_repair2/training_seal.json
807d27e9aedab9211a6ec1891514457b638905d8c61a6d603c298638d5f81de0  Life-Harness/meta/experiments/life_agentbench_flash100k_repair_20261004/FINAL_RESULTS.json
6b2ed9753b65b1b97219c89208f28904cdf860df975a02eeeb81197fceba4c38  Life-Harness/meta/experiments/life_agentbench_flash100k_repair_20261004/TOTAL_SPEND.json
831df2a120fbd8c9b87697ba8f093d6bf82c860c69863e170d2da6d1ac54d6c5  Life-Harness/meta/experiments/life_agentbench_flash100k_repair_20261004/ancestor_usage.json
b1cfd22654fb5fea89f3eb78e77c40c9ee3d5216270b3aa4b532361790fca7cd  Life-Harness/meta/experiments/life_agentbench_flash100k_repair_20261004/TOKEN_USAGE.json
54db7037fccbd63431ffbe1cc77ab4ff3d41723d4ffbedc8ba7694cb51483cb2  ahe-native/audits/historical_tokens_20261004/TOKEN_AUDIT.json
ef8a470801991f1c22c06cb7de007f0ff2bad29187dcb01a14209c1f13de1047  Life-Harness/meta/experiments/meta_ahe_taubench_six_flash_tokens_20261004_repair8/FINAL_RESULTS.json
d1917f1256f0ef990745f611312efe57887c10eaf5bc291d981b29888a33f4c5  Life-Harness/meta/experiments/meta_ahe_taubench_six_flash_tokens_20261004_repair8/OPTIMIZER_TOKEN_USAGE.json
26855e0a0e32c7c1196c867db5aa6e30f07539a8ccecac85f6bfee53cf485e46  Life-Harness/meta/experiments/meta_ahe_taubench_six_flash_tokens_20261004_repair8/TRAINING_RESULTS.json
c6a3e135acd75fd5d8898f224435c7fbec77fffdfc712059935cd81cb904455d  Life-Harness/meta/experiments/life_webshop_six_current_20261007_binding_resume/FINAL_RESULTS.json
75e1b5dc9692a38f7d2522407c1518dcc6a5c4436abdf505a70a8c1cc7d31341  Life-Harness/meta/experiments/life_webshop_six_current_20261007_binding_resume/TRAINING_RESULTS.json
7a4938f132252c9ea86674429a1c5155c264562c35f2b2dc9b148711b9399703  Life-Harness/meta/experiments/life_webshop_six_current_20261007_binding_resume/OPTIMIZER_TOKEN_USAGE.json
8bb5783f8e54e677cbab9e69bd4e9578ed3e7f7b7777a7cfea67f14dd2be9469  Life-Harness/meta/experiments/life_taubench_telecom_six_20261006_codegate_resume/FINAL_RESULTS.json
5a254a87c53212e1706add4c9bec89d63612dc186f973c57c21779d27bf2d89d  Life-Harness/meta/experiments/life_taubench_telecom_six_20261006_codegate_resume/TRAINING_RESULTS.json
77c561cb74e0b87d8b55accef399a6739e6d2fb0db2f2d7e8a949da604013270  Life-Harness/meta/experiments/life_taubench_telecom_six_20261006_codegate_resume/OPTIMIZER_TOKEN_USAGE.json
40b76a5d722abbbe6e54772bb35183609a5203aa7b21dae632f9310c3bce42cd  Life-Harness/meta/experiments/life_taubench_airline_retail_six_20261007_trace_resume/FINAL_RESULTS.json
9c4a6333d95716df12b4d00a2f5bb6b68bc7269ba693734c5d6efefa4b4f256d  Life-Harness/meta/experiments/life_taubench_airline_retail_six_20261007_trace_resume/OPTIMIZER_TOKEN_USAGE.json
a5b5ac56e2e93e309b4850f8e0888b89e3d28a5b634aad0fea97f608ccc059ae  Life-Harness/meta/experiments/life_taubench_airline_retail_six_20261007_trace_resume/training_seal.json
804faffb957294debf92ce06557773c93d62d721f0103aded29bbddbc5285387  Life-Harness/meta/experiments/meta_agentbench_six_flash_tokens_20261004_repair2/alfworld/state.json
e53e9f5e582c67b66a04095dc866f33a5f097f39687936c93e5b6f7f4dc2fe00  ahe-native/heldout_tests/ahe_restricted_six_rotating_flash8_20261003_073330_repair1_finaltest/alfworld/summary.json
2aa07b90fec50b781078cdc07d8233ded3b11b8f54df025a4c25939a3ed674b6  Life-Harness/meta/experiments/meta_agentbench_six_flash_tokens_20261004_repair2/dbbench/state.json
63452418898bfb11b5eb7b572e2d687ab70bab99fe739a24b77bdaf050566786  ahe-native/heldout_tests/ahe_restricted_six_rotating_flash8_20261003_073330_repair1_finaltest/dbbench/summary.json
01c41a34c01d580ba8c3d8113d12f480678fca5e85885012d57b6e6432049c1f  Life-Harness/meta/experiments/meta_agentbench_six_flash_tokens_20261004_repair2/webshop/state.json
5121b48cde9bf152b77fb60ccf3e065cc4ad96aa8facd5aebe18dbe265de0d15  ahe-native/heldout_tests/ahe_restricted_six_rotating_flash8_20261003_073330_repair1_finaltest/webshop/summary.json
01ac6583264a1198fd760b4a0e3dfce1bf3b7444448787232acb4aaf8b04fa65  Life-Harness/meta/experiments/life_agentbench_flash100k_repair_20261004/alfworld/state.json
2db908b1ebd74c0d2acb91a7c40077b44e2ee3cfdd00409fc4ff742a4a1378e3  Life-Harness/meta/experiments/life_agentbench_flash100k_repair_20261004/dbbench/state.json
9ae2eb58800c93dd7d96ca6fe0b2374e5eff99a041b0ee50809ff4aa69f82440  Life-Harness/meta/experiments/life_taubench_airline_retail_six_20261007_trace_resume/airline/TRAINING_RESULTS.json
cbcdbe074d028e7446129528146e9cb50ecb27c0384d958f692db26c71dbad7a  Life-Harness/meta/experiments/life_taubench_airline_retail_six_20261007_trace_resume/airline/OPTIMIZER_TOKEN_USAGE.json
77b8b78b98d92595adaf543bfb9e0cdb4a2586bfc02750fee45f1c25b488c708  Life-Harness/meta/experiments/life_taubench_airline_retail_six_20261007_trace_resume/airline/state.json
39602d517caa9717a454e57d0a74b9671e298ccdb2520553c9eff7802cb4a05e  Life-Harness/meta/experiments/life_taubench_airline_retail_six_20261007_trace_resume/retail/TRAINING_RESULTS.json
2db9f82ea5785f730450dbce09a73f2352c4314858cde952bacceac2556e4a70  Life-Harness/meta/experiments/life_taubench_airline_retail_six_20261007_trace_resume/retail/OPTIMIZER_TOKEN_USAGE.json
8a450888ed1c6c6c31a5a1cbcf8e11e7b58d9fb2540f008b6f0ea0d0454281f1  Life-Harness/meta/experiments/life_taubench_airline_retail_six_20261007_trace_resume/retail/state.json
b3436b7882583f788067241aca944953efaf2de6819773e3a064676558346ced  /data/home/xts/harness/Life-Harness/meta/experiments/os_meta_ahe_life_six_20261008_setup3/BENCHMARK_RESULTS.json
31c34b529967cb9d6c88579be54c10df3e092b42b84341d4d049396bc32b597d  /data/home/xts/harness/Life-Harness/meta/experiments/os_meta_ahe_life_six_20261008_setup3/SIX_ROUND_RESULTS.csv
b3e7b55cc1c66ed853754c661dd28bb26dfc40df9b051cd398f7978dd5d88145  /data/home/xts/harness/Life-Harness/meta/experiments/os_meta_ahe_life_six_20261008_setup3/FINAL_RESULTS.json
0569a8323fb514fc69d77b79cdcf243d4f2e890ece96aa0c3163872e65c2d145  /data/home/xts/harness/Life-Harness/meta/experiments/os_meta_ahe_life_six_20261008_setup3/TRAINING_RESULTS.json
d4a20cd15542c468123460ae3893ee5eb3b90f54090fb1104afdadceffc3183b  /data/home/xts/harness/Life-Harness/meta/experiments/os_meta_ahe_life_six_20261008_setup3/OPTIMIZER_TOKEN_USAGE.json
7c836b91530ee7d45282c3f0a3dbe5efd5b84f6112a3f9b75d04b6a17cb0cb3c  /data/home/xts/harness/Life-Harness/meta/experiments/os_meta_ahe_life_six_20261008_setup3/training_seal.json
```
