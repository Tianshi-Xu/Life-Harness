# ALFWorld 当前四层版发布（2026-09-13）

用户授权将三轮最优版替换发布版。正式 src/server/harness/alfworld.py 与迁移实验 candidate_03.py 字节一致，SHA256 见 release.json。标准 alfworld-std 开启 H2/H3/H4/H5；旧 Runtime 已移出实际 Task 调用链，旧发布文件保留在 before/，历史实验快照不变。

Task 通过 FourHookSession 调用四接口：H3 设置工具描述，H5 冷启动，H4 每轮临时提示，H2 修复动作并更新持久历史。AgentRL 0.4 的 full_history 需同步更新客户端；src/client/task.py 现在能辨认保留原 system+task 前缀的完整消息快照并替换历史，同时继续追加普通增量消息，避免累积重复提示和原始未修复动作。自定义客户端也必须支持完整历史快照。旧 ALFWorldHarnessRuntime/Config 导出改为 ALFWorldHarness（即 Harness），DB/OS/WebShop 的导出不变。

验证：原 109 条固定测试回复经新发布 Task 回放，1540 个模型输入（含工具、提示、公开历史）逐个相等，结果 103/109 相等；这是集成回放，不是新增模型泛化试验。Session 生命周期及客户端增量/完整历史兼容检查通过。实际 controller→worker HTTP 回放完整一题，5 个模型输入一致、正常结束。最初 HTTP 检查脚本误以为 controller 返回 history/history_ptr 和每轮 tools，已按实际 messages + 首轮 tools 协议修正；未更改 Harness 规则。

空闲旧 worker 已停止并在 15021 启动新版，标准入口已生效。DB worker 未重启。历史测试本身是 103/109 vs 100/109，3 胜/0 负但单 trial 未证明稳定优势。
