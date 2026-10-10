# 六场景 Meta / AHE / Life 六轮实验结果汇总

汇总时间：2026-10-07 08:02:43 UTC。

范围：AgentBench 的 ALFWorld、DBBench、WebShop，以及 TauBench 的 Airline、Retail、Telecom。OS 不在本报告中。仅汇总已有结果，不新增或重跑付费实验，不修改历史实验产物。

## 1. 完成情况与比较口径

- 六个场景 × 三种方法，共18组实验，均已有六次优化尝试的记录和最终 TEST 结果；以下保留108个逐轮训练／开销数据点。
- “六轮完成”不等于六轮都生成并评测了新候选。Meta AgentBench 的 ALFWorld/WebShop 第5轮，以及 Meta TauBench 的 Retail 第2轮、Telecom 第2～6轮，候选未测或没有有效候选；表中标记为“†”，保留当轮现任版本的实测 TRAIN 分数，不能将缺失候选分数写成0%。
- TEST 表使用各实验依据 TRAIN 选择并冻结的最终测试版本；不是用 TEST 从六轮中挑最高分。AHE 保留原版选择逻辑，不人为套用 Meta/Life 的严格提升接纳规则。
- Meta/Life 的 TRAIN 表列出当轮保留版本分数：接受候选则取候选分数，否则取当轮现任分数。AHE 列出该轮优化产物后续测量的分数，未必是原版 best-ever 选中的版本。
- AgentBench 每批50条 TRAIN；TauBench Retail/Telecom 每批50条 TRAIN；TauBench Airline 为完整30条 TRAIN × 2次试验，共60条 episode。TEST 每题1次；完整分母含基础设施失败，失败计0。
- 多数场景会换训练批次；Airline 即使使用同一30题池也存在模型／用户对话随机性。跨轮准确率不是固定 TEST 集的单调学习曲线，六轮相同也不代表 token 预算相同。
- 外部优化模型为 DeepSeek V4 Flash API，本地求解器为 Qwen3-4b。优化 token 只计输入＋输出；缓存输入不重复相加，不包含求解器、用户模拟器、NL judge 或测试做题开销，也不等同于人民币费用。
- 当前 Life 汇总并非六域同一代码版本的联合实验：ALFWorld/DBBench 取10月4日修复续跑；WebShop、Telecom、Airline/Retail 取10月6～7日各自最新完成版本。ALFWorld/DBBench 记录 `prior_test_exposure=true`、`test_used_for_optimization=false`，不是项目首次盲测。
- AHE AgentBench 使用 `h2345-budget-restricted-v1` 限制版：每轮最多精读10条详细案例，禁用新增工具、修改工具实现、额外运行时LLM、子代理、单工具隐藏多次环境调用、跨任务可写记忆。这里不把该限制版冒称为无限制原版AHE，也不把此案例限额推广为 Meta/Life 限额。

## 2. 最终 TEST 准确率

发布版参考是用户提供值，不是本次重新测试，也没有对应的六轮优化 token 账本；ALFWorld 发布版参考未提供。

| 套件 | 场景 | TEST分母 | Meta | AHE | Life（当前所选六轮版） | 发布版参考 | TEST基础设施失败：Meta / AHE / Life |
|---|---|---:|---:|---:|---:|---:|---|
| AgentBench | ALFWorld | 109 | 35.78%（39/109） | 95.41%（104/109） | 88.07%（96/109） | 未提供 | 1 / 0 / 1 |
| AgentBench | DBBench | 300 | 50.33%（151/300） | 52%（156/300） | 65%（195/300） | 65% | 0 / 0 / 0 |
| AgentBench | WebShop | 200 | 42.50%（85/200） | 40.50%（81/200） | 46%（92/200） | 45.50% | 0 / 0 / 0 |
| TauBench | Airline | 20 | 20%（4/20） | 35%（7/20） | 40%（8/20） | 60% | 0 / 0 / 0 |
| TauBench | Retail | 40 | 40%（16/40） | 30%（12/40） | 50%（20/40） | 63% | 0 / 0 / 0 |
| TauBench | Telecom | 40 | 50%（20/40） | 92.50%（37/40） | 65%（26/40） | 57.50% | 0 / 0 / 0 |

注意：AHE 历史 AgentBench `COMPARISON.md` 中的 Meta/Life 列属于更早的实验。本报告没有沿用那些旧列；Meta 换成10月4日有实测 token 的六轮重跑，Life 换成上文明确列出的当前版本。AHE 的测试结果本身仍来自该历史限制版的固定最终测试。

### 2.1 TauBench Meta / AHE 三次 TEST 与 pass^3（2026-10-09 补测）

Meta / AHE 三个场景的三次 TEST 均已完成（2026-10-10），完整复用200个、新增400个 episode，六组均无缺失或基础设施失败。原上表是每题一次的 TEST 准确率，不是 pass^3。下表使用完成后的固定分母结果；Life 当前只有单次 TEST，不能当作 pass^3 对比。

| 方法 | 场景 | 每次题数 | 第1次成功数 | 第2次成功数 | 第3次成功数 | 三次平均准确率 | 三次均正确题数 | pass^3 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Meta | Airline | 20 | 4 | 2 | 1 | 11.67% | 0 | 0% |
| AHE | Airline | 20 | 7 | 7 | 7 | 35% | 2 | 10% |
| Meta | Retail | 40 | 16 | 13 | 11 | 33.33% | 6 | 15% |
| AHE | Retail | 40 | 12 | 12 | 10 | 28.33% | 4 | 10% |
| Meta | Telecom | 40 | 20 | 19 | 22 | 50.83% | 16 | 40% |
| AHE | Telecom | 40 | 37 | 38 | 39 | 95% | 35 | 87.5% |

- 固定原六轮 TRAIN 选择并封存的 Meta/AHE harness，不再 evolve，不按测试结果重新选版本；Life 不在本次补测范围。
- 复用原每题第1次 TEST，在完全相同的测试题目上补第2、3次，共复用200个、新增400个 episode。三个 TauBench 场景的题数分别为20、40、40。
- `pass^3 = 三次均正确的题数 / 固定测试题数`。不是“三次中至少一次正确”的 pass@3，也不是平均准确率的三次方。基础设施失败仍在固定分母中计0；未完成的 trial 不伪装成已完成结果。
- Meta 使用原生 checkpoint 的 seed=300 序列补齐后两个 trial；AHE 延续原 adapter 的逐 trial seed=42、43、44。两方法原始 seed 策略不同，保留原协议，不改训练产物。
- 新目录：`meta/experiments/meta_ahe_tau_pass3_20261009/`。`reuse_receipt.json` 记录旧封存与逐记录哈希，`FINAL_RESULTS.json` 已保存逐题三次结果、逐次准确率、三次平均准确率与 pass^3。完成后复核了三次均成功计数、完整分母、原始 TEST 记录和所选 harness 的封存哈希。
- 仅用户模拟器／必要评分器产生外部 TEST API 开销，独立存入该目录 `ledgers/`；本次 evolve 开销为0，不加入下方六轮优化 token。DeepSeek 所有补测请求共享并发上限8；Airline/Retail 仍使用用户 Aliyun `qwen3.8-flash`。
- 补测 DeepSeek 实测并发峰值6，未超过8。Meta Telecom 部分任务因连续环境工具错误达到原生上限而结束，按原生评分计0并保留；这是做题失败，不是基础设施故障，也没有因此重跑或剔除任务。

### 2.2 Life Telecom 第2轮产物的单轮续跑（2026-10-09）

本次仅小幅强化 evolve 对已有 H2 普通文本拦截、H3 动态上下文、H4 进度／完成证据的按需使用，以及避免把提醒／承诺当成完成；没有填入 AHE 策略、领域答案、强制组件或新增校验。保留原六轮实验不变，从已封存的第2轮产物建立新分支，仅新增第3轮 TRAIN 提案和同批50题候选评测，不进行 TEST。原第3轮基线为39/50（78%），通过任务、产物、摘要及接口哈希核对后复用。新候选于2026-10-10完成第3次返修并通过817次公开 TRAIN 调用回放；完整候选评测为37/50（74%），基础设施错误为0，未达到严格提升条件，**未接收**，保留第2轮产物。相同题目的结果中，候选新增正确7题、使原本正确9题退步；这是整包的观察差异，不是单组件因果归因。

新目录为 `meta/experiments/life_telecom_r2_prompt_continue_20261009/`。旧3～6轮的失败尝试保留为优化器可读的历史 TRAIN 记录，不冒充新分支的迭代或成本。`OPTIMIZER_TOKEN_USAGE.json` 只计本次新增优化开销；旧1～2轮开销另存协议。为避免两个独立进程的并发门合计超过8，Life 付费调用等待当前 Meta/AHE 三次 TEST 进程退出，不暂停或改动补测。

预算说明：上述“本次新增开销”不是完整祖先实验费用。由于优化器仍可读取旧3～6轮尝试，若画包含历史探索的累计预算曲线，应使用旧六轮30,455,997 token＋本次新增优化 token，不能仅取协议里的前两轮 `seed_optimizer_tokens` 再加本轮。原六轮主表暂不替换成这个单轮分支。

本轮新增 Evolve 实测为11,241,374 token（输入10,955,264＋输出286,110），49次请求全部有 usage，缺失／待结用量均为0；包含初次提案与3次返修机会，交互请求没有总次数上限。完整祖先探索预算为41,697,371 token。候选评测的 Qwen solver／DeepSeek user simulator 不计入 Evolve 预算。

本轮已封存完成，MR峰值并发为5（上限8），进程已退出。此后才实施通用错误反馈、搜索／局部编辑／单次hook预览、历史代码和逐题增退反馈的 Evolve 升级；上述74%不是这些新能力的实验结果。实现验证不能冒充新训练／测试提升，原六轮主表和最终 TEST 不作替换。

2026-10-10升级已合入当前源码：267项离线回归通过；同一个真实候选在新源码绑定的工作区中，50条 TRAIN 任务的817次调用全部回放通过，bundle内容不变。68组bundle、14组静态源码、14组动作schema的新旧接受／拒绝对照一致。旧24份源文件已本地归档，完成实验的receipt、state、seal和候选bundle校验未变。上述验证没有调用外部LLM，也没有启动新训练／测试评测；升级说明见 `meta/life_evolve/README.md`。

### 2.3 新版 Evolve 工具单轮验证（2026-10-10，已完成，未接收）

目录：`meta/experiments/life_telecom_evolve_tools_probe_20261010/`。从原 Telecom 保留的第2轮产物建立新源码绑定分支，只新增第3轮 TRAIN 提案和50题候选评测；复用同批基线39/50（78%），不自动续跑六轮，也不运行 TEST。新版保留H2～H5接口，增加精确搜索／局部编辑／轨迹预览、具体校验反馈、历史候选代码和逐题得失信息。原六轮历史和上一轮失败提案共同作为历史证据，继承记录去重；上一轮7题增益、9题退步均可查阅，拒绝的代码不会激活。

启动前269项离线回归通过，817次真实 TRAIN 调用回放通过；两份父实验的旧receipt、seal和源码指纹不改写，显式源码升级审计仅允许已记录的优化器／诊断变化，不允许native评分、任务或solver变化。祖先探索开销41,697,371 token与本次新增 Evolve usage分开记录；solver和user simulator开销不计入Evolve。DeepSeek共用8并发门，确认此前两个付费进程已退出后启动。回放通过本身不表示性能提升，准确率以完整候选评测为准。

完整 TRAIN 结果已核对：候选39/50（78%），基线39/50（78%），基础设施错误0，**未接收**，保留第2轮产物。相同50题中8题由错变对、8题由对变错、3题两版均错；这是整包的观察差异，不能据此单独归因某个组件。模型修改H2／H4／H5、保留H3，增加公开证据驱动的未完成工作跟踪和未实际调用工具的承诺／转人工纠正。十次完整预检查全部通过，返修为0；模型实际使用了搜索、局部编辑和轨迹预览，故这次不是校验误拦导致没有合格候选。

本次新增 Evolve 用量23,015,966 token（输入22,887,087＋输出128,879），101次请求均有usage，缺失／待结用量0。包含全部祖先探索的累计预算为64,713,337 token，不能仅计算保留前两轮的费用。模型合格后仍进行较多局部编辑和离线检查，没有临时添加请求上限。MR峰值并发4（上限8），最后active=0且broker已关闭；控制与评测进程均已退出，source、seal、同批题目和保留快照已核验。详情见该目录的 `RESULT_AUDIT.json`。

本轮没有运行TEST，原六轮主表与原保留版本65%单次TEST记录不变。AHE历史第3轮TRAIN为86%、第6轮为100%，但使用其各自历史批次；本轮78%不能声称追上AHE，更不能推断新的TEST提升。下一步应读取实际退步轨迹并改进演化反馈与泛化风险判断，而不是无修改地续跑更多轮。

### 2.4 历史退步轨迹反馈单轮验证（2026-10-10，已完成并接收）

目录：`meta/experiments/life_telecom_evolve_regression_probe_20261010/`。保留第2轮 H2～H5 产物，只新增一轮 TRAIN 提案与50题候选评测，继续复用39/50（78%）同批基线，不运行TEST。新增 `read_case(history_index=...)` 可按需读取历史候选真实公开 TRAIN 轨迹；既有隐私过滤与显式model/action边界业务状态读取复用，历史文件绑定hash，不复制大体积DB日志。重复context/tools改为 `include_setup=true` 按需获取，记录调用的messages可分页，但完整内容仍可读、读取和总交互不限；返修仍至多3次。Prompt优先比较真实退步、确认触发范围，并在合格机制准备好后交给真实评测，没有引入硬请求上限或手填策略。

启动前273项离线回归通过，真实50题817次调用回放通过；8个历史提案的候选轨迹均可读取，21份历史代码可查，上一轮8个退步案例已离线逐一验证能区分基线成功与历史候选失败。保存两代原源码各24份，旧实验receipt与seal不改写。全部祖先探索64,713,337 token另记，本轮新增usage单独统计，solver/user simulator不计入Evolve开销。离线回放本身不是新准确率结果，实际分数以完整候选评测为准。

Evolve 已完成并送入50题真实 TRAIN 评测：103次请求共26,123,863 token（输入25,943,632＋输出180,231），103次usage均齐全，无未知／待结 optimizer usage；含全部祖先探索累计90,837,200 token。9次完整预检查均通过，返修0、工具错误0，模型最终修改H2／H4／H5、保留H3。模型通过32次read_case读取15个不同案例，但没有使用新增的history_index读取历史候选轨迹；这与启动前宿主离线验证的16次读取应区分。30次轨迹预览、14次局部替换、56次搜索说明软性结束提示没有减少交互，本次新增token反而高于上一轮23,015,966，不能宣称节省预算。

最终完整 TRAIN：基线39/50（78%），候选45/50（90%），提高12个百分点，**已接收第3轮产物**，基础设施错误0；相同50题10题由错变对、4题由对变错、1题两版均错。MR峰值3（上限8），9次模拟器429均有成功重试，optimizer无429；结束时active=0、broker已关闭，控制／代理／评测进程全部退出。seal、summary receipt、保留快照和运行时源码指纹已核对，24份本轮源码另存该目录的 `source_before/`。这些是整包的观察差异，不是组件单独因果归因或统计显著性证明。AHE历史第3轮TRAIN为86%、第6轮为100%，其批次不同；本轮90%不能冒充原六轮曲线或最终TEST，更不能宣称追上AHE 95%三次TEST平均／87.5% pass³。原六轮和65%单次TEST记录保持不变。详情见 `RESULT_AUDIT.json`。

封存后另修复一个证据接口误判：H3合法地改变工具描述曾被当作native契约变化拒绝。现在保留每次调用的公开描述差异，并让include_setup按需返回指定调用的真实描述；仍检查工具名／参数schema／类型一致，不改solver、runtime或评分。这是后续离线修复，不能归因到已经完成的90%结果；不会重写本轮receipt或自动重跑付费评测。

### 2.5 证据优先的第4轮续跑（2026-10-10，已完成、不接收）

目录：`meta/experiments/life_telecom_r4_evidence_first_20261010/`。继承2.4中已接收的第3轮 H2～H5 产物，按冻结的六轮训练计划换下一批50题，只新增第4轮TRAIN，不运行TEST，也不重复已经完成的第3轮。新批次先测同一保留版本的基线，再同批测候选并严格决定是否接收；不能复用旧批次的90%作本轮基线。

Evolve开轮消息直接提供最新真实增益／退步和准确的历史索引；`read_case(history_index=..., history_variant='candidate'|'incumbent')` 可配对读取过去同题的两版公开轨迹，即使当前50题已更换。继续保留历史代码、任意按需读取、不限总交互、至多3次返修和100000单次输出。只有事实反馈和通用证据工具变化，不手填业务机制、不改变H2～H5、solver、评分或本地32768／4096设置。历史继承记录去重并补齐原始来源的缺失轨迹，不激活未接收候选；启动提示和合格后交接建议不是新合规门槛。

启动前281项离线回归通过；已接收第3轮产物在其真实50条TRAIN轨迹796次调用中回放通过，9个历史提案的基线与候选证据均可读，最近4个退步案例的0／1配对已验证。首次离线脚本直接把50个含DB的案例送入单个底层worker触发资源限制，核对后复用已有逐案例生产路径通过；没有为此添加重复实现、放宽限制或付费重试。24份改动前源码已保存，4个封存祖先来源、保留快照、操作日志与来源指纹核验通过。全部祖先探索90,837,200 token单独继承，新付费Evolve用量从零记录，solver／user simulator另记；旧receipt、seal和主表不改写。启动前未发现旧付费runner，MR共用8并发门。

单轮runner已完成：第3轮保留产物基线46/50（92%），新候选43/50（86%），两版基础设施错误均为0；同题3题由错变对、6题由对变错、1题两版均错。因此第4轮不接收，继续保留第3轮产物，快照和操作日志与父来源一致。这是整包观察，不足以单独归因某一组件。本批与上一批共有29题、新增21题；跨批次90%→92%不是同批提升证据，也不是TEST。主表及旧TEST记录不变。

新批次的保留产物在50条真实轨迹816次调用中离线回放通过，仍用原逐案例CPU／RAM／wall边界，没有额外付费调用或混入模型读取日志。Evolve共59次请求，新增9,462,331 token（输入9,332,562＋输出129,769），59次usage均齐全，无未知／待结optimizer用量；全部祖先探索加本轮累计100,299,531 token。两次显式完整预检查与最终自动检查均通过，返修0，没有回退到早期快照。这次是准确率未提升，不是被预检查误拦截。模型read_case共21次、10个不同案例，其中历史候选1次、历史基线0次；文件搜索触及50个ID不能写成精读50题。另观察到未写bundle就请求preview、把已轮换的历史案例当当前文件搜索；后续修正保留版本预览及跨批次读取说明，不改本轮已绑定付费源码、不加逐案分析门槛。

本轮实际绑定的24份源码已复制到运行目录的 `source_before/`，逐份核对hash，并记录 `SOURCE_BEFORE_MANIFEST.json`；完整结果核验见 `RESULT_AUDIT.json`。MR共用门limit8、peak5，结束closed=true／active0，controller、tokenizer proxy和评测进程均退出。17次simulator HTTP429全部在第二次尝试成功，其费用不计Evolve；没有其他未知用量。快照与大体积实验结果均不进入Git，旧收据、源码绑定、计数器与已完成基线不改写。

### 2.6 第5轮：配对历史证据与保留版本预览（2026-10-10，已完成、不接收）

目录：`meta/experiments/life_telecom_r5_paired_diagnostics_20261010/`。第4轮结束后补齐通用诊断接口：无bundle时默认预览保留版本，或显式 `chain='incumbent'`；已有坏候选不会静默回退，完整预检查仍只检查候选。当前批次之外的案例ID／文件错误说明如何读取历史候选及匹配历史基线，prompt明确两者都需要history_index，并强调用观测结果而非承诺清除待办。没有新增合规门槛、强制逐案分析或手填业务内容，H2～H5核心、native评分和solver设置不改。

启动前285项离线回归全部通过；真实50题／816次调用的保留链回放及上一轮候选完整预检查均通过，10个历史提案可索引，最新索引3的6个退步案例均验证候选0／历史基线1配对，保留版本预览无需候选或付费调用。明确保留原付费源码与receipt，改动前后24份源码和祖先归档由上一目录的 `FOLLOWUP_SOURCE_AUDIT.json` 绑定。

仅续跑第5轮TRAIN，使用固定计划的batch4共50题（较batch3新增15题），当前状态、最佳第3轮快照和操作日志与第4轮封存状态完全一致，5个祖先来源已核验。祖先探索成本100,299,531 token独立继承，本轮新Evolve费用从零记录，不混入solver／user simulator。DeepSeek共用并发8，本地Qwen3-4B workers50、32768总上下文／4096输出；优化器无总交互／token上限、至多3次返修、100000单次输出。未启动新TEST，不声称该接口改动已提升准确率。

batch4完整基线为保留第3轮产物42/50（84%），基础设施错误0。本批与上批不同，84%不能与上批92%直接解释为代码退步；当前保留快照没有变化。基线summary receipt、runtime manifest和完整分母已核验，开轮主历史索引3准确返回上一轮3增益／6退步；10个历史提案可索引，没有宿主read_case混入付费模型账本，详见BASELINE_RESULT_AUDIT.json。候选必须在同一batch4完整50题上严格超过42/50才接收；最终结果见下。

Evolve已完成并送入同批50题候选评测：102次请求、22,168,910实测token（输入21,984,057＋输出184,853），102次usage全部齐全，无未知／待结优化用量；全部祖先探索加本轮累计122,468,441 token。本轮有1次失败预检查、1次返修：三个modify操作漏了expected_revision和expected_sha256，反馈逐操作明确给出缺项，模型补齐后完整50题／778次调用回放通过，最终自动检查也通过，没有回退。不是H2～H5被误拦截。候选修改H4进展状态／账户待办、H2关闭与转人工边界、H5启动约定，保留H3；这些是模型生成的机制，尚无本轮准确率提升证据。提案来源、输出hash和费用核验见PROPOSER_RESULT_AUDIT.json。

付费模型read_case共36次、20个不同案例；文件读取／搜索触及49个ID，不能写成精读49题。详细历史候选／历史基线读取均为0，已有配对历史接口此次没有被模型调用，不将本轮候选效果归因于该工具。没有给Life添加10案例上限或其他总交互／token上限。

最终同批50题候选42/50（84%），与基线相同，故不接收，继续保留第3轮产物。同题4增益、4退步、4题两版均错，两版基础设施错误均为0；这是整包观察，不是单组件因果归因。完整结果和收据核验见RESULT_AUDIT.json。MR门limit8、peak4，结束closed=true／active0，controller、tokenizer和候选评测进程均退出。24份本轮绑定源码已归档到source_before并逐一校验，frozen_sources一致。旧receipt、seal和读取账本不改写，模拟器费用不混入Evolve预算。仍无新TEST，不声称已追上AHE。

### 2.7 第6轮：版本记账自动绑定与成对退步轨迹（2026-10-10，用户要求停止）

第5轮封存后仅改Evolve：modify/delete省略的expected_revision／expected_sha256由宿主按冻结库存补齐；显式错误或过期值仍报错，不改写模型声明文件。开轮直接提供上一轮4个退步案例两版公开轨迹尾部摘录和整包假设；摘录是观测锚点而非原因判断，完整历史按需读取仍无限制。没有手填业务策略、照搬AHE代码、新增合规门槛或改变H2～H5／native评分／solver预算。

289项离线测试全部通过；真实保留链、上一轮候选和仅删除版本字段的未付费候选副本，完整50题／778次调用回放均通过。11个历史提案可读、最新主索引4的4个退步案例均为基线1／候选0，宿主反馈不混入模型读取账本。6份历史源码归档逐项hash核验，无遗留付费runner。前后源码和验证记录在第5轮目录FOLLOWUP_SOURCE_AUDIT.json，旧receipt、seal与读取账本不改写。

新目录meta/experiments/life_telecom_r6_paired_briefing_20261010，原runner PID1460240。继承封存状态和122,468,441祖先优化token，仅从保留第3轮快照续跑固定batch5的50题TRAIN；新费用从0记录，未重复启动旧付费任务。24份本轮源码已冻结归档；原付费协议仍为DeepSeek共用并发8，本地workers50／32768总上下文／4096输出，优化器100000单次输出、至多3次返修、无总交互／token限制，不改写历史协议。

用户明确要求停止第6轮、直接测试已接收的TRAIN90%版本。控制器已SIGINT退出，MR broker关闭且active=0，无遗留付费runner。完整batch5基线为39/50（78%），基础设施失败0。第6轮未完成提案／候选评测，不标成完成一轮。账本48次优化请求usage全部齐全，新增Evolve为6,636,485 token（输入6,515,532＋输出120,953），无未知／待结用量，累计探索费用129,104,926 token。保留所有源码、基线、输出、账本和checkpoint，详见CANCELLATION.json；checkpoint在工具处理期间仍标inflight=true，未重置或盲目续跑。旧六轮主表不替换。

### 2.8 固定已接收第3轮产物的 Telecom 三次 TEST（2026-10-10，完成）

目录：`meta/experiments/life_telecom_r3_fixed_test3_20261010/`。固定第3轮TRAIN45/50（90%）已接收快照`25d3163e49016d239dc7ea4ec09e21af5bf1d1a3c00fed056e0c7fdfe78b8c4a`，没有按TEST选版本。原生40题、trials=3、固定120个episode全部完成。只测这一份产物，不再Evolve。DeepSeek只作用户模拟，单broker并发8；Qwen3-4B solver仍32768／4096、workers50。测试runtime/data/产物单独冻结，后续修改未使用的Evolve代码不会改变此TEST。

| 方法／固定版本 | Trial 1 | Trial 2 | Trial 3 | 三次均值 | 严格 pass^3（三次都正确） |
|---|---:|---:|---:|---:|---:|
| Life 本次TRAIN选定第3轮产物 | 35/40（87.5%） | 37/40（92.5%） | 36/40（90%） | 108/120（90%） | 31/40（77.5%） |
| AHE 原六轮固定产物 | 37/40（92.5%） | 38/40（95%） | 39/40（97.5%） | 114/120（95%） | 35/40（87.5%） |
| Meta 原六轮固定产物 | 20/40（50%） | 19/40（47.5%） | 22/40（55%） | 61/120（50.83%） | 16/40（40%） |

Life本次均值距AHE仍差5个百分点，pass^3差10个百分点，不声称已经追上。AHE／Meta数值复用`meta/experiments/meta_ahe_tau_pass3_20261009/FINAL_RESULTS.json`，不是本次重跑；旧Life单次TEST65%和发布版57.5%参考值不替换、不当作三次指标。这是后续续跑分支结果，不覆盖原六轮主表。

独立核验：40题各有trial0／1／2，120条无重复、无缺失、基础设施失败0；326份冻结源码／数据／产物hash全部一致，原生结果hash与summary一致。账本3058次请求均有实测usage：用户模拟输入17,683,286＋输出370,422＝18,053,708 token，未知／待结／估算均为0；仅user_simulator角色，新增Evolve费用0。MR门limit8／peak5／active0，broker已关闭，原控制器／评测器及停止的第6轮控制器已退出。核验记录见本目录`RESULT_AUDIT.json`。

该产物生成于本次降耗改造之前：其成功提案26,123,863 Evolve token，包含此前失败提案的完整祖先费用90,837,200；计入后续第4／5轮失败探索和停止的第6轮，共花费129,104,926。TEST用户模拟费用不混入Evolve曲线，不能将这一90%测试结果标为“新1000万预算内生成”，也不能用此次固定旧产物测试证明新工作记忆机制不损害优化效果。

### 2.9 后续 Evolve 成本控制（2026-10-10，仅实现／离线验证，不启动新TRAIN）

用户要求不超过旧版平均开销的两倍。旧Telecom六轮均值5,075,999.5 token；新的单次提案输入＋输出总上限10,000,000 token，包含初次提案、返修及工作记忆总结，按provider实际用量累计，含缓存输入、不混入solver或simulator。保留每请求100000输出，无Flash本地总上下文／API次数／案例读取上限，最多3次返修。下一请求前保守预留输入与完整输出，预算不足时可交付已校验候选；没有合格候选则停止，不晋升无效产物或重置成本。

总预算只是保险线，实际降耗改为同一agent主动整理工作记忆，避免后续每次都重新携带完整分析、工具输出及reasoning。65536输入token仅触发记忆检查点，不是上下文限制或截断：模型自行保留证据定位、假设、风险、当前候选和下一步；宿主保留准确的校验反馈／返修计数。原始完整上下文hash归档可随时读取，候选、合格快照、费用计数不重置；总结费用也计入Evolve。空白／格式错误总结不会丢弃旧上下文或暗中重试。首次尝试的整段原文去重在第5轮记录上仅节省0.20%本地输入token，已删除，未堆积低收益实现。实际节省比例和准确率不退步仍须后续获准TRAIN验证，当前不作保证、不启动新付费优化。

首次边界修复后303项离线回归全部通过，覆盖同agent记忆切换、原始历史可恢复／不可模型改写、错误反馈保留、候选与合格快照保留、空摘要回退、暂停恢复后不重置返修／用量、总结费用累计及单次provider用量、预算预留和超预算禁止盲目恢复，以及原有六域接口／评分适配。进一步发现并修复安全暂停后未恢复宿主最近校验报告的问题：报告随checkpoint保存／恢复，连续两次记忆切换仍保留原始错误反馈；新回归先复现缺失字段失败，修复后通过，不增加API调用或更改H2～H5。

另用新版`prepare`／`verify_inputs`在临时目录离线验证停止的第6轮完整50题基线可复用：39/50（78%）、基础设施失败0，新输入绑定当前源码及1000万预算；54份历史状态／协议／取消记录／训练证据hash不变，临时验证未调用优化器或模拟器。因此后续获准的一轮TRAIN验证无需重复付费跑基线；候选仍须评完整同批50题。仍未启动新付费Evolve／TRAIN；三次已完成TEST的326份冻结runtime／数据／产物hash仍一致。实际Evolve省幅与优化效果是否保持，仍需用户确认的一轮训练验证，不能由离线通过或旧产物TEST代替。

### 2.10 新工作记忆机制的一轮 TRAIN 验证（2026-10-10，运行中）

用户明确同意仅新增一轮验证，目录`meta/experiments/life_telecom_memory_budget_probe_20261010/`，controller PID1786279。继承第5轮封存状态（仍保留已接收第3轮快照），复用停止的第6轮完整batch5基线39/50（78%），不恢复其旧提案／checkpoint、不重复付费测基线。只新生成一份第6轮候选并评同批完整50题TRAIN；严格提升才接收，无新增TEST、额外对齐评测或后续轮。

单次提案输入＋输出上限10,000,000 token，包含工作记忆整理和最多3次返修；保留单次100000输出、无限交互／案例读取、全局MR并发8和本地solver50并发／32768总上下文／4096输出。新账本从0记录；原封存祖先122,468,441及停止第6轮6,636,485分别保留，历史已知总支出129,104,926，不将其清零或混入新提案预算。准备过程API调用0，476份源码／训练基线／父记录等冻结，当前24份接口与Evolve源码已归档`source_used/`；只变Evolve，不改native运行／评分或现有H2～H5产物。准备核验见`PREPARATION_AUDIT.json`。已核实原进程存活并收到前三次完整API usage，共44,621 token；实际成本与训练效果仍待本轮完整结果。

初次执行在请求129被新增记忆格式门误终止：总结请求仅提供checkpoint工具，但模型返回两项正常`search_file`，宿主将其视为整份提案失败。这是Evolve宿主流程缺陷，不是H2～H5校验失败或准确率下降。未产生候选／未开始评测；全部129次usage实测齐全，输入4,985,073＋输出320,862＝5,305,935 token，未知／待结0，broker关闭／active0／peak1，PID1786279已退出。旧状态、源码、读取记录、账本和错误checkpoint不改写。

修复后，失败的可选记忆整理保留完整上下文，明确反馈未执行的工具调用，下一步回到正常工具交互；不消耗返修机会、不丢失合格快照或重置费用。另新增通用`copy_file`，仅把冻结active/history Python源文件原样复制到候选artifacts，再用已有`replace_file`精确修改；不手填业务策略、不变H2～H5或评分。Prompt仍少于5000字符，不给模型增加分析表或案例读取配额。305项全量离线回归通过。

仅续接**同一份第6轮提案**：`meta/experiments/life_evolve_cost_control_20261010/resumed/life_telecom_memory_budget_probe_20261010/`，新PID1866921。原129次请求、全部上下文／13份hash绑定原始记忆档案、实际用量5,305,935及0次返修原样携带；两项已知只读搜索在离线审计中完成并回给模型，不产生API调用。冻结公开TRAIN证据、现有H2～H5内容逐项hash一致；完整50题基线复用39/50。新输入只重新绑定修复源码；总预算仍10,000,000，剩余4,694,065，不另开新轮或新预算、不自动恢复未知请求。审计见`SAME_PROPOSAL_RESUME_AUDIT.json`。续接账本包含原129次调用，作图必须只计一次，不能再把初次失败目录的5,305,935重复相加。本轮仍未完成，不报候选准确率。

2026-10-10 04:18 UTC进度补记：本次Evolve已完成，共182次请求，输入7,079,419＋输出436,280＝**7,515,699 token**，全部为实测usage，未知／待结／估算均为0，已包含初次中断的5,305,935，不重复相加。模型仅修改H4进度监控组件，H2／H3／H5保留；50条公开TRAIN轨迹815次调用回放通过，返修0次，17次工作记忆检查点成功，1次旧格式失败已保留并续接。候选已交付同批完整50题真实TRAIN评测，仍在运行；预检查通过不能代替最终准确率。新用量约为旧Telecom六轮均值的1.48倍，低于用户允许的两倍上限；比最近第5轮22,168,910约减少66.1%，但不同轮次并非降耗机制的严格因果对照。现阶段尚不能声称新候选无性能退步或有新的TEST结果。并发门仍为8，当前已观测峰值4；没有追加轮次或TEST。

## 3. 六轮外部优化 token 总量

单位：输入＋输出 token，整数。AgentBench AHE 的“作图预算”是约定估算情景；其“已记录”只是 Evolve 已知下界。TauBench AHE 已记录项包含 Debugger＋Evolve。

| 场景 | Meta实测 | Life实测（含本轮系谱的失败／返修） | AHE已记录 | AHE补充估算 | AHE作图预算：已记录＋估算 |
|---|---:|---:|---:|---:|---:|
| ALFWorld | 37,138,309 | 24,288,432 | 63,532,347 | 32,100,000 | 95,632,347 |
| DBBench | 32,813,601 | 11,204,889 | 72,424,554 | 32,100,000 | 104,524,554 |
| WebShop | 36,741,591 | 37,750,993 | 71,743,620 | 32,100,000 | 103,843,620 |
| Airline | 53,088,352 | 36,142,791 | 64,497,756 | 0 | 64,497,756 |
| Retail | 14,363,326 | 25,706,750 | 93,141,101 | 0 | 93,141,101 |
| Telecom | 21,322,791 | 30,455,997 | 88,213,291 | 135,307 | 88,348,598 |
| 六场景合计 | 195,467,970 | 165,549,852 | 453,552,669 | 96,435,307 | 549,987,976 |

### 3.1 AHE 的估算约定与缺口

- AgentBench 沿用原留档的缺失调用平均50,000 token情景，不把所有调用重估为50,000：已记录 Evolve 207,700,521；broker attempts 3,931；tracer generations 2,005；计数差1,926 × 50,000＝96,300,000。
- 缺失调用不能恢复到真实场景／轮次；仅为估算情景均摊到18个单元，每场景每轮补5,350,000，每场景六轮补32,100,000。逐轮均摊不是恢复出的实测分配，也不是置信区间。
- attempts 与 generations 在失败／重试时未必一一对应，差值不能断言全部来自 Debugger；WebShop 首轮复用的更早 Debugger 和修复前归档启动成本不在完整计数内。因此 AHE AgentBench 全历史开销仍未完整恢复。
- TauBench AHE Airline/Retail 的已记录推理用量无额外估算；分别另有23/19次429拒绝请求未返回 usage。原账本标记 `inference_accounting_complete=true`，不将这些拒绝请求假装成已消耗推理 token。
- TauBench AHE Telecom 第2轮有1次缺失推理 usage，原账本补估135,307（输入133,616＋输出1,691）；另有28次429拒绝无 usage。其六轮已记录88,213,291，含估算预算88,348,598，不能标为全部实测。

### 3.2 历史启动与续跑费用

- Meta AgentBench 六轮表外另有前置失败／启动成本 2,490,771；其三场景六轮106,693,501，加启动109,184,272。未将这笔无法按本表轮次归并的启动费偷偷删为0；主曲线按既有六轮口径，全部已知支出另记。
- Life ALFWorld/DBBench 不是只统计修复后的新请求：合并 `ancestor_usage.json` 与新 `TOKEN_USAGE.json`，同一旧调用只计一次，废弃／失败分支也保留。
- Life ALFWorld 的旧run在新run成功完成第2轮前，已经花掉旧第3轮失败提案的3,900,206 token。下方为了作图不漏掉提前支出，将这笔费用计入当前合格第2轮的成本里程碑：第2轮8,029,314、第3轮1,886,019；累计第2轮12,190,452。原ledger按round标签归并则是第2轮4,129,108、第3轮5,786,225；两种归并六轮总计均24,288,432。
- Life DBBench 第4轮包括祖先失败第4轮1,924,480＋修复后396,779＝2,321,259。不能仅用新run的396,779作为全部第4轮成本。
- 旧 Life AgentBench 三域 `TOTAL_SPEND.json` 为49,292,551，其中旧WebShop13,799,230不属于本报告选用的新WebShop六轮。新WebShop从空内容单独重跑，费用37,750,993；不混用旧WebShop分数／费用，也不声称本表涵盖所有项目研发和历次重跑成本。

## 4. 逐轮 TRAIN 准确率与作图数据

每个表中的累计 token 只在同场景、同方法内累计。标记 M＝实测；E50＝AHE AgentBench缺失调用50k、跨场景跨轮均摊情景；M+E＝AHE Telecom含单次usage估算；†＝新候选未测或未产生，TRAIN分数来自现任版本。

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

### 4.4 TauBench / Airline

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

### 4.5 TauBench / Retail

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

### 4.6 TauBench / Telecom

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

- 当前 Life 的最终 TEST 准确率六域均高于本次 Meta；高于AHE的四域为 DBBench、WebShop、Airline、Retail；ALFWorld与Telecom仍低于AHE。不应只挑有利域汇报。
- 相对用户发布版参考，Life DBBench持平，WebShop高0.5个百分点，Telecom高7.5个百分点；Airline低20个百分点，Retail低13个百分点。ALFWorld缺少用户提供的发布版参考，不能从旧训练smoke推断其发布版全TEST结果。
- Life 的六域优化 token 都低于AHE的已记录下界；但并非每域低于Meta：Life的WebShop、Retail、Telecom高于Meta，ALFWorld、DBBench、Airline低于Meta。
- 后期仍有同轮有效改进：Airline第4～6轮、Retail/WebShop第5～6轮接收；但DBBench第4～6轮、Telecom第3～6轮没有接收新候选。不能声称所有场景一直上涨，或保证继续增加轮次就达到发布版。
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

### 源文件快照 SHA-256

路径相对于 `/mnt/data/xts/harness`。校验了108个逐轮数据点的六轮合计、18份最终结果的成功数／完整分母，并确认所选实验状态均已complete。本文仅创建汇总文件，未变更原始证据。

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
```
