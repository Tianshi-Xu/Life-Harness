"""Read-only status and separately gated final reporting command."""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Callable

import driver
from protocol import (DOMAINS, HERE, HISTORICAL_ROOT, ROUNDS, atomic_json,
                      digest, object_digest, protocol)


def _process_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def rows(root: Path = HERE) -> list[dict[str, Any]]:
    result = []
    for domain in DOMAINS:
        path = root / domain / "state.json"
        state = json.loads(path.read_text()) if path.exists() else {}
        lock = root / domain / ".lock"
        try:
            pid = int(lock.read_text().strip()) if lock.exists() else None
        except ValueError:
            pid = None
        rounds = state.get("rounds", [])
        result.append({"domain": domain, "status": state.get("status", "not_started"),
                       "running": _process_alive(pid), "pid": pid, "rounds": len(rounds),
                       "batch": state.get("batch", 0),
                       "active_mechanisms": len(state.get("active_snapshot", [])),
                       "last": rounds[-1] if rounds else None})
    return result


def seal_training(root: Path = HERE) -> tuple[Path, dict[str, Any]]:
    states = {}
    run_configs = {}
    for domain in DOMAINS:
        path = root / domain / "state.json"
        if not path.exists():
            raise RuntimeError(f"final reporting refused: {domain} has no state")
        state = json.loads(path.read_text())
        if state.get("status") != "complete" or state.get("round") != ROUNDS or len(state.get("rounds", [])) != ROUNDS:
            raise RuntimeError(f"final reporting refused: {domain} has not completed twelve rounds")
        run_config = root / domain / "run_config.json"
        if not run_config.is_file() or json.loads(run_config.read_text()) != driver.run_fingerprint(domain):
            raise RuntimeError(f"final reporting refused: {domain} run fingerprint mismatch")
        driver.verify_active(root / domain, state)
        driver.verify_training_history(root / domain, state)
        states[domain] = digest(path)
        run_configs[domain] = digest(run_config)
    value = {"experiment": "life_agentbench_v3_twelve_db_web_20261002",
             "protocol_sha256": digest(root / "protocol.json"),
             "inherited_training_seal_sha256": protocol()["inherited_training_seal_sha256"],
             "prior_test_exposure": True,
             "state_sha256": states, "run_config_sha256": run_configs}
    path = root / "training_seal.json"
    if path.exists() and json.loads(path.read_text()) != value:
        raise RuntimeError("sealed training state changed")
    if not path.exists():
        atomic_json(path, value)
    return path, value


def load_meta_results() -> tuple[str, dict[str, float]]:
    cfg = protocol()
    path = HISTORICAL_ROOT / "RESULTS.md"
    if digest(path) != cfg["historical_results_sha256"]:
        raise RuntimeError("historical Meta RESULTS.md changed")
    text = path.read_text()
    summary = text.split("## Domain summary", 1)[-1]
    scores = {}
    for domain in DOMAINS:
        match = re.search(rf"^\| {domain} \|[^\n]*\| ([0-9.]+) \| ([0-9.]+) \|$", summary, re.MULTILINE)
        if not match:
            raise RuntimeError(f"cannot parse Meta result for {domain}")
        scores[domain] = float(match.group(1))
    return digest(path), scores


def final_report(root: Path = HERE,
                 evaluate: Callable[..., dict[str, Any]] = driver.phase) -> dict[str, Any]:
    seal_path, seal = seal_training(root)
    # This is deliberately after seal_training: Meta results cannot affect training state.
    meta_sha, meta_scores = load_meta_results()
    report_path = root / "final_results.json"
    if report_path.exists():
        evaluator = __import__("agentbench_evaluate")
        evaluator.validate_test_gate(root)
        current_evaluations = {}
        comparisons = {}
        manifest = __import__("protocol").validate_protocol(check_files=False)
        for domain in DOMAINS:
            summary_path = root / domain / "evals/final_test/summary.json"
            if not summary_path.is_file():
                raise RuntimeError("final report cache is missing an evaluation summary")
            summary = json.loads(summary_path.read_text())
            if (summary.get("domain") != domain or summary.get("split") != "test"
                    or summary.get("phase") != "final_test"):
                raise RuntimeError("final report cache has invalid evaluation metadata")
            evaluator.validate_cached_summary(summary_path.parent, summary)
            state = json.loads((root / domain / "state.json").read_text())
            driver._assert_active_chain_config(root / domain, "final_test",
                                               state["active_snapshot"],
                                               state["active_snapshot_sha256"])
            expected = __import__("protocol").task_ids(manifest["domains"][domain], 0, "test")
            expected_source = {name: digest(root / name) for name in
                               ("protocol.py", "candidate_contract.py", "driver.py",
                                "agentbench_chain.py", "agentbench_evaluate.py")}
            if (summary.get("evaluator_schema_version") != 1
                    or summary.get("batch") is not None
                    or summary.get("ids") != expected
                    or summary.get("protocol_hash") != digest(root / "protocol.json")
                    or summary.get("source_sha256") != expected_source
                    or summary.get("concurrency") != 8
                    or summary.get("task_limit") is not None):
                raise RuntimeError("final report cache is not bound to the sealed evaluator")
            driver._assert_expected_keys(summary, expected)
            value = driver.score(summary)
            current_evaluations[domain] = object_digest(summary)
            comparisons[domain] = {"life_six_test": value, "meta_test": meta_scores[domain],
                                   "delta": value - meta_scores[domain],
                                   "beats_meta": value > meta_scores[domain], "n": len(expected)}
        expected_report = {"training_seal_sha256": digest(seal_path),
                           "state_sha256": seal["state_sha256"],
                           "meta_results_sha256": meta_sha,
                           "evaluation_sha256": current_evaluations,
                           "comparisons": comparisons}
        report = json.loads(report_path.read_text())
        if report != expected_report:
            raise RuntimeError("final report cache mismatch")
        return report
    comparisons = {}
    evaluation_sha256 = {}
    for domain in DOMAINS:
        state = json.loads((root / domain / "state.json").read_text())
        chain = driver.write_active_chain(root / domain, state, "final")
        summary = evaluate(domain, "test", "final_test", chain, state["batch"])
        expected = __import__("protocol").task_ids(
            __import__("protocol").validate_protocol(check_files=False)["domains"][domain], 0, "test")
        driver._assert_expected_keys(summary, expected)
        score = driver.score(summary)
        evaluation_sha256[domain] = object_digest(summary)
        comparisons[domain] = {"life_six_test": score, "meta_test": meta_scores[domain],
                               "delta": score - meta_scores[domain],
                               "beats_meta": score > meta_scores[domain], "n": len(expected)}
    report = {"training_seal_sha256": digest(seal_path),
              "state_sha256": seal["state_sha256"], "meta_results_sha256": meta_sha,
              "evaluation_sha256": evaluation_sha256, "comparisons": comparisons}
    atomic_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("status", "report"), default="status")
    args = parser.parse_args()
    if args.command == "report":
        print(json.dumps(final_report(), indent=2, sort_keys=True))
        return
    for item in rows():
        activity = f"running pid={item['pid']}" if item["running"] else "idle"
        print(f"{item['domain']:9} {item['status']:11} {activity:20} rounds={item['rounds']} "
              f"batch={item['batch']} active={item['active_mechanisms']}")


if __name__ == "__main__":
    main()
