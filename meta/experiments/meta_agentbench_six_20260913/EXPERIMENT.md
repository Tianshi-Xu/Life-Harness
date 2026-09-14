# Meta-Harness：ALFWorld / DBBench，各六轮，停滞换样本

用户要求两套 AgentBench 各跑六轮，每次至少 50 道训练任务，一旦没有提升就换样本，记录每轮结果。起点按讨论上下文采用**原始发布 H2345**（ALF 发布替换前与 DB 迁移前的冻结快照），不是当前迁移候选。起点选择已询问，未收到更改意见，按已说明的默认项推进。

- 每批固定 50 题，分层抽样。ALF 六类任务 9/9/8/8/8/8；DB INSERT 17、UPDATE 17、多类查询 16。
- 每域预先生成六批互不重复训练题，排除之前两批训练样本；DB 还排除与 std/dev 规范化题面重合。不做测试集评估。
- 每轮仅一个候选、50 道真实评估，不使用 Life 的有界阅读包和筛查规则。Meta 可访问本次所有公开训练轨迹与历史候选。
- 严格同批成功数增加才接受，并继续用这批题；持平/下降时保留 incumbent，下一轮换新批并重新真实评估 incumbent。不同批分数不可串成一条直接比较的学习曲线。
- Qwen3-4B、temperature 0、thinking false、4096 tokens；ALF 50 回合，DB 15 回合；每域四 worker。两个域可同时进行。

## 原版来源与适配边界

直接导入克隆仓库 reference_examples/terminal_bench_2/meta_harness.py，逐轮调用原始 run_evolve，compute_pass_rates / update_frontier / update_evolution_summary 原函数不变。原始 SKILL.md 原文提供，保留一个分析 Agent 与一个实现 Agent。Qoder/DeepSeek-Flash 替代原 Claude CLI/Opus；Bash 仅准许在工作区作离线原型与 import smoke。原仓库不修改，commit 和哈希见 manifest.json。

Harbor 被替换为 AgentBench 原生环境计分。每个候选是原发布 runtime 的完整单文件副本，导出 AgentHarness 类/别名，保留原生同步调用兼容；也可通过 transform_request / transform_response 改模型可见请求和输出。允许任意 runtime/helper 改写，不给四层设计准则，不限制只能改某一层。固定模型调用、任务预算、原生工具语义/schema、环境状态转移与评分；不支持 TB2 终端/新增外部模型调用/任意外部库。已有 native task-context 接口字段按原版保留，但禁止新增 oracle/隐藏数据访问。原版已有 eval 解析表达式可原样保留，新 eval 站点禁止。

因此这是**原始 Meta 方法的 AgentBench 原生接口移植**，不是 TB2/Opus 原实验的逐项复现。原版任务级 frontier 仅供分析，不被当作可部署组合分数。为执行用户换题规则，每轮用独立 native 日志和当前批 incumbent 重建 frontier，跨批历史另记 history.json；不会混合不同批次的 _best 分数。

## 验证与结果位置

6 条 ALF 历史训练回放的 103 次模型输入、6 条 DB 回放的 16 次模型输入，与原始 native H2345 完全一致，终止状态/成功相同；无新模型调用。原始外循环集成、样本轮换与完整 evaluator 入口/缓存的三个离线测试通过。首个启动尝试在模型调用前暴露旧测试守卫残留，修复后重新冻结才开始真实计分，原记录见 prelaunch_fix。

每域 state.json 记录真实外层六轮编号、batch、起点/候选成绩、配对胜负、是否接受及下轮是否换题；workspace/logs/round_NN 保存原版 frontier 与 evolution_summary；evals 保存宿主原始结果；workspace/evidence 只有公开输入/输出及成功位，不含正确 SQL、内部 reward 或世界状态。每轮后更新 state，实验最终统一汇总。

## DB 第三批候选异常归因

已接受的 tool_call_escape_fidelity 在第三批基线 task 4050 的 _lenient_json_loads 触发 KeyError。原生任务捕获为 task error，宿主最初统归基础设施错误，导致暂停。依据明确指向候选文件的 traceback，将该 episode 标记为候选运行失败（仍为 0 分），而非环境故障；没有重跑、没有改候选、没有改成功位。原始 episode/summary 与归因记录保存在 failure_attribution/dbbench_batch02_4050。对未来任何可明确归于候选代码的异常同样计失败；真正环境/传输错误仍阻止接受。错误诊断作为公开训练证据供第六轮分析。

## 完成状态

2026-09-13 两域各六轮完成，最终完整性检查通过。逐轮分数、候选机制、成本与效率候选见 [RESULTS.md](RESULTS.md)；工具边界偏差及可比较范围见 [AUDIT.md](AUDIT.md)。ALF 最终保留第五轮，DB 最终保留第六轮。没有新测试集评估或部署。
