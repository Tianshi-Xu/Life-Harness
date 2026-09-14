# Meta 六轮最优版发布替换

用户授权将刚完成标准集评测的 Meta 版本替换当前发布版本。

ALFWorld 采用第六轮 `info_first_search`，标准集 104/109。候选 Runtime 源码原样置入正式 `src/server/harness/alfworld.py`，并在文件末尾追加统一 `Harness.h2/h3/h4/h5` 适配层；没有退回旧 Task。适配层从公开完整历史重建 admissible actions 和已执行步骤，H2 同时承接候选的纯文本动作恢复。候选正文是评测文件的字节级完整前缀，哈希和适配后发布哈希见 release.json。

DBBench 采用第六轮 `tool_call_desync_recovery`，标准集 190/300。正式 `src/server/harness/dbbench.py` 与冻结评测候选字节一致；正式 Task/interaction 与评测快照哈希一致，继续使用原生 Runtime 接口。

旧正式 Harness 均保存在 `before/src/server/harness/`，历史实验文件未更改。两个原 worker 在 controller 显示 idle 后停止；新 worker 分别在 15021/15022 启动，均为 ALIVE、idle。controller、MySQL 和模型服务未重启。

验证：Python import/compile 通过；FourHookSession 的完整历史、H2 持久修复、默认临时 H4 及 H5 生命周期测试通过。Meta ALF 适配器显式启用与旧 Runtime 一致的持久 H4 传递模式，其他 Life Harness 的默认临时 H4 不变。ALF 109 条冻结回复经过正式四层 Task 回放仍为 104/109、成功位零差异；1126/1290 个请求逐字一致，剩余 164 个差异集中在 11 个 episode，来自合并多条 H4 提示及无工具/阻断分支；4 个失败 episode 的结束状态不同，不影响成功位。另经 live controller→最终 ALF worker 完整回放 1 条成功 episode，5 个模型输入逐字一致并正常完成；两域 controller→worker 脚本冒烟通过，DB 完成两次工具调用，ALF 完成一次环境动作。以上回放没有新模型调用。

没有因发布继续读取标准集轨迹或修改候选规则；没有重新跑标准集选择另一个版本。ALF 四层适配后尚未做第二次全量真实模型评测，因此 104/109 是原 Meta Runtime 的冻结评测成绩，适配层的全量保证限于冻结回复成功位一致。
