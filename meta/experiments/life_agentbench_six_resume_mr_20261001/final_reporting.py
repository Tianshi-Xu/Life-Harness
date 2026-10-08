"""Complete only changed-domain test evaluation and report explicit result reuse."""
from __future__ import annotations

import fcntl
import json
from pathlib import Path

import agentbench_chain
import agentbench_evaluate
import continuation_support
import driver
import inheritance
import protocol
import status


def read(path):
    return json.loads(Path(path).read_text())


def reuse_test(domain: str, state: dict) -> dict:
    seed = inheritance.verified_seed(domain)
    if state["active_snapshot"] != seed["active_snapshot"]:
        raise RuntimeError("cannot reuse test results for a changed accepted snapshot")
    root = inheritance.base_root()
    folder = root / domain / "evals/final_test"
    config = read(folder / "config.json")
    body = {k: v for k, v in config.items() if k != "config_sha256"}
    expected = protocol.task_ids(protocol.validate_protocol(check_files=False)["domains"][domain], 0, "test")
    if (config["config_sha256"] != protocol.object_digest(body)
            or config["split"] != "test" or config["domain"] != domain
            or config["phase"] != "final_test" or config["ids"] != expected
            or config["snapshot_sha256"] != protocol.object_digest(seed["active_snapshot"])):
        raise RuntimeError("predecessor test configuration mismatch")
    for name, expected_hash in config["source_sha256"].items():
        if protocol.digest(root / name) != expected_hash:
            raise RuntimeError("predecessor test source changed")
    chain = root / domain / "runtime/final/manifest.json"
    artifact_root, records = agentbench_chain.manifest_records(chain)
    if records != seed["active_snapshot"] or config["chain_hashes"] != agentbench_evaluate.chain_hashes(chain):
        raise RuntimeError("predecessor test chain mismatch")
    outcomes = {}
    infra = []
    hashes = {}
    for index in expected:
        path = folder / f"episode_{index}.json"
        row = read(path)
        if (row.get("episode_schema_version") != 1 or row["index"] != index
                or row["config_sha256"] != config["config_sha256"]
                or row["success"] not in (0, 1) or not isinstance(row["infrastructure_error"], bool)):
            raise RuntimeError("predecessor test episode mismatch")
        if row["infrastructure_error"]:
            if row["success"] != 0:
                raise RuntimeError("infrastructure-error episode must be counted as failure")
            infra.append(index)
        outcomes[str(index)] = row["success"]
        hashes[path.name] = protocol.digest(path)
    return {"domain": domain, "split": "test", "n": len(outcomes),
            "successes": sum(outcomes.values()), "accuracy": sum(outcomes.values()) / len(outcomes),
            "infra_errors": len(infra), "infra_error_indices": infra, "per_task": outcomes,
            "provenance": "reused predecessor final test because accepted snapshot is identical",
            "new_test_episodes": 0, "source_run": str(root), "source_config_sha256": protocol.digest(folder / "config.json"),
            "source_episode_sha256": hashes, "source_evaluator_sha256": config["source_sha256"],
            "snapshot_sha256": state["active_snapshot_sha256"], "prior_test_exposure": True}


def fresh_test(domain: str, state: dict) -> dict:
    folder = protocol.HERE / domain
    chain = driver.write_active_chain(folder, state, "final")
    summary_path = folder / "evals/final_test/summary.json"
    if summary_path.exists():
        summary = read(summary_path)
    else:
        summary = driver.phase(domain, "test", "final_test", chain, state["batch"])
    agentbench_evaluate.validate_cached_summary(summary_path.parent, summary)
    driver._assert_active_chain_config(folder, "final_test", state["active_snapshot"], state["active_snapshot_sha256"])
    expected = protocol.task_ids(protocol.validate_protocol(check_files=False)["domains"][domain], 0, "test")
    expected_sources = {name: protocol.digest(protocol.HERE / name) for name in
                        ("protocol.py", "candidate_contract.py", "driver.py", "agentbench_chain.py", "agentbench_evaluate.py")}
    if (summary["ids"] != expected or summary["split"] != "test"
            or summary["phase"] != "final_test" or summary["domain"] != domain
            or summary["source_sha256"] != expected_sources
            or summary["protocol_hash"] != protocol.digest(protocol.HERE / "protocol.json")):
        raise RuntimeError("six-round final test is not bound to the sealed evaluator")
    driver.score(summary)
    errors = [index for index in expected if read(summary_path.parent / f"episode_{index}.json")["infrastructure_error"]]
    return {"domain": domain, "split": "test", "n": summary["n"], "successes": summary["successes"],
            "accuracy": summary["accuracy"], "infra_errors": summary["infra_errors"],
            "infra_error_indices": errors, "per_task": summary["per_task"],
            "provenance": "new fixed final test of sealed sixth-round accepted snapshot",
            "new_test_episodes": len(expected), "summary_path": str(summary_path),
            "summary_sha256": protocol.digest(summary_path),
            "snapshot_sha256": state["active_snapshot_sha256"], "prior_test_exposure": True}


def comparison(report: dict) -> dict:
    meta_root = protocol.HISTORICAL_ROOT
    rounds = []
    tests = []
    for domain in protocol.DOMAINS:
        meta = read(meta_root / domain / "state.json")
        life = read(protocol.HERE / domain / "state.json")
        if len(meta["rounds"]) != 6 or len(life["rounds"]) != 6:
            raise RuntimeError("comparison requires all six round decisions")
        for m, l in zip(meta["rounds"], life["rounds"]):
            n = l["round"]
            if m["iteration"] != n:
                raise RuntimeError("comparison round order mismatch")
            meta_test = read(meta_root / domain / "evals" / m["test_phase"] / "summary.json")
            if meta_test["accuracy"] != m["test_score"]:
                raise RuntimeError("Meta round test disagrees with recorded decision")
            rounds.append({"domain": domain, "round": n,
                "meta_batch": m["train_batch"], "meta_incumbent": m["baseline_score"],
                "meta_candidate": m["candidate_score"], "meta_accepted": m["accepted"],
                "life_batch": l["train_batch"], "life_incumbent": l["incumbent_score"],
                "life_candidate": l["candidate_score"], "life_accepted": l["accepted"]})
            life_test = report["evaluations"][domain] if n == 6 else reuse_test(domain, inheritance.verified_seed(domain)) if n == 5 else None
            tests.append({"domain": domain, "round": n, "meta_test": m["test_score"],
                          "life_test": life_test["accuracy"] if life_test else None,
                          "life_provenance": life_test["provenance"] if life_test else "not evaluated"})
    return {"train_rounds": rounds, "test_rounds": tests,
            "notes": ["all train evaluations use 50 tasks; frozen batch is explicitly identified",
                      "round rejection retains incumbent; candidate score is not the retained score",
                      "Life rounds 1-4 have no test measurements; no retrospective test runs were added",
                      "Life round-6 proposer is ModelRouter deepseek-v4-flash; first five rounds are inherited",
                      "predecessor final tests were already observed; results are a continuation with prior test exposure"]}


def markdown(report: dict, data: dict) -> str:
    lines = ["# Meta 与 Life 六轮对比", "", "逐轮表为 train 成功率（百分比），每次 50 题。b 表示冻结 batch；✓ 接受，× 拒绝。箭头表示 incumbent → candidate；拒绝时保留 incumbent。不同 batch 的数值不可作为连续提升曲线。", "",
             "| 环境 | 轮次 | Meta train | Life train |", "|---|---:|---|---|"]
    for r in data["train_rounds"]:
        cells = []
        for arm in ("meta", "life"):
            cells.append(f"b{r[arm+'_batch']}：{r[arm+'_incumbent']*100:.0f} → {r[arm+'_candidate']*100:.0f} {'✓' if r[arm+'_accepted'] else '×'}")
        lines.append(f"| {r['domain']} | {r['round']} | {cells[0]} | {cells[1]} |")
    lines += ["", "## 逐轮 test", "", "Life 第 1–4 轮未运行 test；空白不是失败，也未为补表追加历史轮次测试。第 5 轮 test 属于此前封存结果。第 6 轮 ALFWorld/WebShop 复用完全相同机制的旧评测，DBBench 新测。", "",
              "| 轮次 | ALFWorld Meta / Life | DBBench Meta / Life | WebShop Meta / Life |", "|---:|---|---|---|"]
    for number in range(1, 7):
        cells = []
        for domain in protocol.DOMAINS:
            row = next(r for r in data["test_rounds"] if r["domain"] == domain and r["round"] == number)
            value = f"{row['life_test']*100:.2f}%" if row["life_test"] is not None else "未测"
            cells.append(f"{row['meta_test']*100:.2f}% / {value}")
        lines.append(f"| {number} | {' | '.join(cells)} |")
    lines += ["", "## 六轮最终 test", "", "| 环境 | Meta | Life | Life − Meta | Life 来源 | infra errors |", "|---|---:|---:|---:|---|---:|"]
    for domain in protocol.DOMAINS:
        c = report["comparisons"][domain]
        e = report["evaluations"][domain]
        source = "复用第五轮" if e["new_test_episodes"] == 0 else "第六轮新测"
        lines.append(f"| {domain} | {c['meta_successes']}/{e['n']} = {c['meta_test']*100:.2f}% | {e['successes']}/{e['n']} = {e['accuracy']*100:.2f}% | {c['delta']*100:+.2f} pp | {source} | {e['infra_errors']} |")
    lines += ["", "WebShop 旧 index 42 的基础设施失败仍按 0 计分，保留完整 200 题分母，没有重试或剔除。原 DBBench 第五轮的两项错误仅保留在旧结果中，不迁入第六轮新评测。", "",
              "Life 的前五轮来自旧运行，第六轮 proposer 改为 API deepseek-v4-flash。三域六轮训练全部封存后才执行本次 final test。此前第五轮 test 已观察，因此该比较明确是追加实验；单次结果不表示统计显著性。", "",
              "来源：历史 Meta `state.json`、各轮 test summaries；Life 六轮 `state.json`、五轮原始 test episodes、DBBench 六轮 final summary；精确分数与复用哈希见本目录 `final_results.json` 和 `round_comparison.json`。", ""]
    return "\n".join(lines)


def main():
    continuation_support.bind_amendment()
    continuation_support.install()
    with (protocol.HERE / ".final_reporting.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        seal_path, seal = status.seal_training()
        agentbench_evaluate.validate_test_gate(protocol.HERE)
        meta_hash, meta_scores = status.load_meta_results()
        evaluations = {}
        for domain in protocol.DOMAINS:
            state = read(protocol.HERE / domain / "state.json")
            seed = inheritance.verified_seed(domain)
            unchanged = state["active_snapshot"] == seed["active_snapshot"]
            print(json.dumps({"domain": domain, "action": "reuse_prior_test" if unchanged else "evaluate_sixth_round_final_test"}), flush=True)
            evaluations[domain] = reuse_test(domain, state) if unchanged else fresh_test(domain, state)
        comparisons = {}
        for domain, evaluation in evaluations.items():
            meta = read(protocol.HISTORICAL_ROOT / domain / "evals/test_round_06/summary.json")
            if meta["accuracy"] != meta_scores[domain] or meta["n"] != evaluation["n"]:
                # RESULTS.md rounds historical values to four decimals.
                if abs(meta["accuracy"] - meta_scores[domain]) > 0.00005 or meta["n"] != evaluation["n"]:
                    raise RuntimeError("historical Meta final test disagrees with results")
            comparisons[domain] = {"life_six_test": evaluation["accuracy"], "meta_test": meta["accuracy"],
                "meta_successes": meta["successes"], "delta": evaluation["accuracy"] - meta["accuracy"],
                "beats_meta": evaluation["accuracy"] > meta["accuracy"], "n": evaluation["n"]}
        report = {"schema_version": 2, "reporting_source_sha256": protocol.digest(Path(__file__)),
                  "training_seal_sha256": protocol.digest(seal_path), "state_sha256": seal["state_sha256"],
                  "meta_results_sha256": meta_hash, "prior_test_exposure": True,
                  "new_test_episodes": sum(e["new_test_episodes"] for e in evaluations.values()),
                  "evaluations": evaluations, "comparisons": comparisons}
        path = protocol.HERE / "final_results.json"
        if path.exists() and read(path) != report:
            raise RuntimeError("final report changed; refusing to overwrite")
        if not path.exists():
            protocol.atomic_json(path, report)
        data = comparison(report)
        protocol.bind_fingerprint(protocol.HERE / "round_comparison.json", data)
        text = markdown(report, data)
        print(json.dumps({"complete": True, "new_test_episodes": report["new_test_episodes"], "comparisons": comparisons}), flush=True)
        return text


if __name__ == "__main__":
    main()
