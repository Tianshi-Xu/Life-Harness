"""After round-nine sealing, test changed snapshots and reuse unchanged tests."""
import fcntl
import json
from pathlib import Path

import agentbench_chain
import agentbench_evaluate
import driver
import inheritance
import protocol
import status


def read(path):
    return json.loads(Path(path).read_text())


def reuse(domain, state):
    seed = inheritance.verified_seed(domain)
    if state["active_snapshot"] != seed["active_snapshot"]:
        raise RuntimeError("cannot reuse a changed snapshot")
    report_path = inheritance.base_root() / "final_results.json"
    previous = read(report_path)["evaluations"][domain]
    if previous["snapshot_sha256"] != state["active_snapshot_sha256"]:
        raise RuntimeError("predecessor report snapshot mismatch")
    if previous["new_test_episodes"]:
        folder = Path(previous["summary_path"]).parent
        if protocol.digest(folder / "summary.json") != previous["summary_sha256"]:
            raise RuntimeError("predecessor test summary changed")
        agentbench_evaluate.validate_cached_summary(folder, read(folder / "summary.json"))
    else:
        folder = Path(previous["source_run"]) / domain / "evals/final_test"
        if protocol.digest(folder / "config.json") != previous["source_config_sha256"]:
            raise RuntimeError("predecessor test config changed")
        for name, expected in previous["source_episode_sha256"].items():
            if protocol.digest(folder / name) != expected:
                raise RuntimeError("predecessor test episode changed")
    config = read(folder / "config.json")
    expected_ids = protocol.task_ids(protocol.validate_protocol(check_files=False)["domains"][domain], 0, "test")
    if (config["split"] != "test" or config["ids"] != expected_ids
            or config["snapshot_sha256"] != state["active_snapshot_sha256"]
            or protocol.object_digest({k: v for k, v in config.items() if k != "config_sha256"}) != config["config_sha256"]):
        raise RuntimeError("reused test configuration mismatch")
    source_root = folder.parents[2]
    for name, expected in config["source_sha256"].items():
        if protocol.digest(source_root / name) != expected:
            raise RuntimeError("reused test evaluator changed")
    chain = source_root / domain / "runtime/final/manifest.json"
    _, records = agentbench_chain.manifest_records(chain)
    if records != state["active_snapshot"] or agentbench_evaluate.chain_hashes(chain) != config["chain_hashes"]:
        raise RuntimeError("reused test chain mismatch")
    hashes, outcomes, errors = {}, {}, []
    for index in expected_ids:
        path = folder / f"episode_{index}.json"
        if not agentbench_evaluate._episode_valid(path, index, config["config_sha256"]):
            raise RuntimeError("invalid reused test episode")
        row = read(path)
        if row["infrastructure_error"]:
            if row["success"] != 0:
                raise RuntimeError("infra error not counted as zero")
            errors.append(index)
        outcomes[str(index)] = row["success"]
        hashes[path.name] = protocol.digest(path)
    if outcomes != previous["per_task"] or sum(outcomes.values()) != previous["successes"]:
        raise RuntimeError("predecessor outcomes disagree with report")
    return {"domain": domain, "split": "test", "n": len(outcomes),
            "successes": sum(outcomes.values()), "accuracy": sum(outcomes.values()) / len(outcomes),
            "per_task": outcomes, "infra_errors": len(errors), "infra_error_indices": errors,
            "snapshot_sha256": state["active_snapshot_sha256"], "new_test_episodes": 0,
            "provenance": "reused six-round final result for identical accepted snapshot",
            "source_report_sha256": protocol.digest(report_path), "source_run": str(source_root),
            "source_config_sha256": protocol.digest(folder / "config.json"),
            "source_episode_sha256": hashes, "prior_test_exposure": True}


def fresh(domain, state):
    folder = protocol.HERE / domain
    chain = driver.write_active_chain(folder, state, "final")
    summary = driver.phase(domain, "test", "final_test", chain, state["batch"])
    path = folder / "evals/final_test/summary.json"
    agentbench_evaluate.validate_cached_summary(path.parent, summary)
    driver._assert_active_chain_config(folder, "final_test", state["active_snapshot"], state["active_snapshot_sha256"])
    expected = protocol.task_ids(protocol.validate_protocol(check_files=False)["domains"][domain], 0, "test")
    if summary["ids"] != expected or summary["protocol_hash"] != protocol.digest(protocol.HERE / "protocol.json"):
        raise RuntimeError("new final test protocol mismatch")
    return {"domain": domain, "split": "test", "n": summary["n"],
            "successes": summary["successes"], "accuracy": driver.score(summary),
            "per_task": summary["per_task"], "infra_errors": summary["infra_errors"],
            "snapshot_sha256": state["active_snapshot_sha256"], "new_test_episodes": len(expected),
            "provenance": "new fixed test of sealed ninth-round accepted snapshot",
            "summary_path": str(path), "summary_sha256": protocol.digest(path),
            "prior_test_exposure": True}


def main():
    with (protocol.HERE / ".final_reporting.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        seal_path, seal = status.seal_training()
        agentbench_evaluate.validate_test_gate(protocol.HERE)
        previous_path = inheritance.base_root() / "final_results.json"
        previous = read(previous_path)
        evaluations, comparisons, rounds = {}, {}, []
        anchors = {"dbbench": 0.65, "webshop": 0.455}
        for domain in protocol.DOMAINS:
            state = read(protocol.HERE / domain / "state.json")
            seed = inheritance.verified_seed(domain)
            unchanged = state["active_snapshot"] == seed["active_snapshot"]
            print(json.dumps({"event": "final_test", "domain": domain, "reused": unchanged}), flush=True)
            evaluation = reuse(domain, state) if unchanged else fresh(domain, state)
            evaluations[domain] = evaluation
            comparisons[domain] = {"six_round_test": previous["evaluations"][domain]["accuracy"],
                "nine_round_test": evaluation["accuracy"],
                "delta": evaluation["accuracy"] - previous["evaluations"][domain]["accuracy"],
                "release_reference": anchors[domain], "release_reference_source": "user provided 2026-10-02",
                "delta_vs_release": evaluation["accuracy"] - anchors[domain]}
            rounds.extend(dict(row, domain=domain) for row in state["rounds"][6:])
            print(json.dumps({"event": "final_test_complete", "domain": domain,
                              "n": evaluation["n"], "successes": evaluation["successes"],
                              "infra_errors": evaluation["infra_errors"]}), flush=True)
        report = {"schema_version": 2, "training_seal_sha256": protocol.digest(seal_path),
                  "predecessor_final_report_sha256": protocol.digest(previous_path),
                  "prior_test_exposure": True, "evaluations": evaluations,
                  "comparisons": comparisons, "additional_train_rounds": rounds}
        amendment = protocol.HERE / "runtime_amendment.json"
        if amendment.exists():
            report["runtime_amendment_sha256"] = protocol.digest(amendment)
            report["excluded_invalid_evaluation"] = "runtime_cap_archive/dbbench/candidate_09: 50 episodes blocked by a legacy cap, preserved separately"
        path = protocol.HERE / "final_results.json"
        protocol.bind_fingerprint(path, report)
        print(json.dumps(report["comparisons"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
