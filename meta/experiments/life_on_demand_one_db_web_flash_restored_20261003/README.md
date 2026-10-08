# Life：DBBench / WebShop 各一轮验证

严格只跑一轮（b0），从空 H2～H5 内容开始；每域完整 50 条训练题，候选与基线同题比较，仅严格提高接收。正常提案失败结束该域本次尝试，不自动另起提案，不推进第二轮。不运行测试集。

DeepSeek Flash 使用此前的 `max_tokens=8192`，不设置本地 32768 上下文拦截，不调用 FlashCounter/solver-budget guard。单域最多 32 次代理请求，包括诊断、修改和本地预检查返修；两个域共用并发上限 8 的 broker。本地 Qwen 保持总上下文 32768、单次输出 4096、每域最多 50 worker。

复用原首轮两域 100 条空基线之前，核验完整 episode/summary 哈希、相同 b0 题号、相同空快照及接口，且原调用满足 Qwen 长度要求。不继承此前提案的诊断或代码；新提案使用共享优化器 v2.3。

公开训练案例全部按需可读，不设 10 条案例配额。预检查工具反馈保留在同一 DeepSeek 会话；不改变优化器核心逻辑或添加压缩/检索策略。测试数据、隐藏答案和凭据不进入优化器。

`protocol.py`、`driver.py`、`runner.py` 分别检查只允许一轮；driver/evaluator 独立拒绝 TEST。源码/配置/输入/缓存由指纹绑定，重启已完成运行也不会追加轮次。

监控 `progress.json`、`flash_started.jsonl`、`flash_requests.jsonl` 和 `runner.log`。请求日志只含元数据，不含真实 key 或提示内容。首批请求即核验模型、输出设置、实际返回及工具进展；一轮完成后结果在 `round_results.json` 和 `TRAINING.md`。历史对比仅为训练记录并列，见 `COMPARISON.md`，不是测试结果或严格同条件消融。
