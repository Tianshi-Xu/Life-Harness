# Life：60000 输出预算，DBBench / WebShop 各重跑一轮

按用户明确要求，DeepSeek Flash 单次 `max_tokens=60000`（包含 reasoning），无本地 32768 上下文限制。该预算是输出上限，不是必须生成的长度；真实 ModelRouter/provider 能力以实际回包为准，不静默降回 8192 或 4096。

严格只跑一轮（b0），每域完整 50 条训练题，从空 H2～H5 开始。有效候选与基线同题比较，只接收严格提高。提案失败结束该域此次尝试，不自动另起提案，不推进第二轮，不运行测试集。

本地 Qwen 保持总上下文 32768、单次输出 4096、每域最多 50 worker。Flash 全局并发上限 8；每域一个代理会话，最多 32 次请求，包含按需诊断、代码修改和本地检查返修。不新增逐任务付费 Debugger，不改变 H2～H5 接口、预检查、接收规则或证据读取方式。

首轮空基线 DB 20/50、WebShop 12/50 已由哈希、题号、接口和实际调用长度核验，复用而不重跑本地基线。不继承此前截断提案的诊断或代码。此前 8192-output 的一轮运行和其 v2.3 优化器源码完整保留；本次使用 v2.4，独立目录、独立指纹。

`protocol.py`、`driver.py`、`runner.py` 明确锁定每域一轮，driver/evaluator 独立拒绝 TEST。首批真实 API 回包将核对模型、60000 输出参数和工具进展；如 API 拒绝该预算或提案失败，明确记录并停止，不静默改配置再重跑。

过程见两域 `progress.json`、`flash_started.jsonl`、`flash_requests.jsonl`、`runner.log`。最终训练结果见 `round_results.json`、`TRAINING.md`、`COMPARISON.md`。日志仅含 API 元数据，真实 key 只在宿主 broker 内存中。
