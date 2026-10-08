# Meta 与 Life 六轮对比

逐轮表为 train 成功率（百分比），每次 50 题。b 表示冻结 batch；✓ 接受，× 拒绝。箭头表示 incumbent → candidate；拒绝时保留 incumbent。不同 batch 的数值不可作为连续提升曲线。

| 环境 | 轮次 | Meta train | Life train |
|---|---:|---|---|
| alfworld | 1 | b0：22 → 32 ✓ | b0：26 → 38 ✓ |
| alfworld | 2 | b0：32 → 58 ✓ | b0：38 → 94 ✓ |
| alfworld | 3 | b0：58 → 64 ✓ | b0：94 → 98 ✓ |
| alfworld | 4 | b0：64 → 68 ✓ | b0：98 → 98 × |
| alfworld | 5 | b0：68 → 62 × | b1：96 → 98 ✓ |
| alfworld | 6 | b1：40 → 46 ✓ | b1：98 → 98 × |
| dbbench | 1 | b0：42 → 52 ✓ | b0：36 → 54 ✓ |
| dbbench | 2 | b0：52 → 54 ✓ | b0：54 → 58 ✓ |
| dbbench | 3 | b0：54 → 56 ✓ | b0：58 → 56 × |
| dbbench | 4 | b0：56 → 56 × | b1：52 → 60 ✓ |
| dbbench | 5 | b1：46 → 50 ✓ | b1：60 → 58 × |
| dbbench | 6 | b1：50 → 46 × | b2：64 → 68 ✓ |
| webshop | 1 | b0：24 → 38 ✓ | b0：26 → 34 ✓ |
| webshop | 2 | b0：38 → 40 ✓ | b0：34 → 36 ✓ |
| webshop | 3 | b0：40 → 42 ✓ | b0：36 → 40 ✓ |
| webshop | 4 | b0：42 → 40 × | b0：40 → 34 × |
| webshop | 5 | b1：26 → 32 ✓ | b1：38 → 38 × |
| webshop | 6 | b1：32 → 34 ✓ | b2：40 → 36 × |

## 逐轮 test

Life 第 1–4 轮未运行 test；空白不是失败，也未为补表追加历史轮次测试。第 5 轮 test 属于此前封存结果。第 6 轮 ALFWorld/WebShop 复用完全相同机制的旧评测，DBBench 新测。

| 轮次 | ALFWorld Meta / Life | DBBench Meta / Life | WebShop Meta / Life |
|---:|---|---|---|
| 1 | 30.28% / 未测 | 48.33% / 未测 | 26.50% / 未测 |
| 2 | 43.12% / 未测 | 49.00% / 未测 | 25.50% / 未测 |
| 3 | 46.79% / 未测 | 48.67% / 未测 | 29.50% / 未测 |
| 4 | 55.96% / 未测 | 49.67% / 未测 | 28.00% / 未测 |
| 5 | 58.72% / 92.66% | 48.33% / 53.67% | 30.50% / 35.50% |
| 6 | 53.21% / 92.66% | 48.00% / 57.33% | 31.00% / 35.50% |

## 六轮最终 test

| 环境 | Meta | Life | Life − Meta | Life 来源 | infra errors |
|---|---:|---:|---:|---|---:|
| alfworld | 58/109 = 53.21% | 101/109 = 92.66% | +39.45 pp | 复用第五轮 | 0 |
| dbbench | 144/300 = 48.00% | 172/300 = 57.33% | +9.33 pp | 第六轮新测 | 0 |
| webshop | 62/200 = 31.00% | 71/200 = 35.50% | +4.50 pp | 复用第五轮 | 1 |

WebShop 旧 index 42 的基础设施失败仍按 0 计分，保留完整 200 题分母，没有重试或剔除。原 DBBench 第五轮的两项错误仅保留在旧结果中，不迁入第六轮新评测。

Life 的前五轮来自旧运行，第六轮 proposer 改为 API deepseek-v4-flash。三域六轮训练全部封存后才执行本次 final test。此前第五轮 test 已观察，因此该比较明确是追加实验；单次结果不表示统计显著性。

来源：历史 Meta `state.json`、各轮 test summaries；Life 六轮 `state.json`、五轮原始 test episodes、DBBench 六轮 final summary；精确分数与复用哈希见本目录 `final_results.json` 和 `round_comparison.json`。
