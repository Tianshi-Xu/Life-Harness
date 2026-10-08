"""Explicitly authorized DB-only full test of the sealed twelve-round snapshot."""
import fcntl
import json
import os
import time
import traceback

import agentbench_evaluate as evaluator
import driver
import protocol


def event(state, **extra):
    value = {"status": state, "pid": os.getpid(), "updated_at": time.time(), **extra}
    protocol.atomic_json(protocol.HERE / "db_test_workflow_state.json", value)
    print(json.dumps(value), flush=True)


def authorization():
    seal = evaluator.validate_test_gate(protocol.HERE)
    historical = protocol.validate_protocol()
    targets = {}
    for domain in ("dbbench",):
        state = json.loads((protocol.HERE / domain / "state.json").read_text())
        driver.verify_active(protocol.HERE / domain, state)
        ids = protocol.task_ids(historical["domains"][domain], 0, "test")
        expected_n = {"dbbench": 300, "webshop": 200}[domain]
        if len(ids) != expected_n:
            raise RuntimeError("unexpected full-test denominator")
        targets[domain] = {"snapshot_sha256": state["active_snapshot_sha256"],
                           "n": len(ids), "task_ids_sha256": protocol.object_digest(ids)}
    value = {"user_request": "db再测一次测试集吧",
             "scope": "fresh full DBBench test only under the sealed v3 interface; no WebShop, training or proposer calls",
             "training_seal_sha256": protocol.digest(protocol.HERE / "training_seal.json"),
             "sealed_state_sha256": seal["state_sha256"],
             "runner_sha256": protocol.digest(protocol.HERE / "db_test_runner.py"), "targets": targets}
    protocol.bind_fingerprint(protocol.HERE / "db_test_authorization.json", value)
    return value


def validate_result(domain, summary):
    evaluator.validate_test_gate(protocol.HERE)
    folder = protocol.HERE / domain
    evaluator.validate_cached_summary(folder / "evals/final_test", summary)
    state = json.loads((folder / "state.json").read_text())
    expected_ids = protocol.task_ids(protocol.validate_protocol(check_files=False)["domains"][domain], 0, "test")
    expected_interface = {key: protocol.protocol()[key] for key in
                          ("interface_version", "interface_source_sha256", "base_session_sha256")}
    if (summary.get("split") != "test" or summary.get("phase") != "final_test"
            or summary.get("ids") != expected_ids or summary.get("task_limit") is not None
            or summary.get("batch") is not None or summary.get("interface") != expected_interface
            or summary.get("snapshot_sha256") != state["active_snapshot_sha256"]):
        raise RuntimeError("final evaluation not bound to the full sealed v3 target")
    driver._assert_expected_keys(summary, expected_ids)
    driver._assert_active_chain_config(folder, "final_test", state["active_snapshot"], state["active_snapshot_sha256"])
    driver.score(summary)


def main():
    with (protocol.HERE / ".db_final_test.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock.write(str(os.getpid()) + "\n")
        lock.flush()
        try:
            receipt = authorization()
            summaries = {}
            for domain in ("dbbench",):
                folder = protocol.HERE / domain
                state = json.loads((folder / "state.json").read_text())
                chain = driver.write_active_chain(folder, state, "final")
                event("testing", domain=domain, n=receipt["targets"][domain]["n"])
                summary = driver.phase(domain, "test", "final_test", chain, 0)
                validate_result(domain, summary)
                summaries[domain] = {key: summary[key] for key in
                                     ("n", "successes", "accuracy", "infra_errors", "config_sha256", "snapshot_sha256")}
                summaries[domain]["summary_sha256"] = protocol.digest(folder / "evals/final_test/summary.json")
                event("domain_complete", domain=domain, **summaries[domain])
            authorization()  # Training state and runner must still be identical.
            value = {"experiment": protocol.protocol()["name"],
                     "interface_version": protocol.protocol()["interface_version"],
                     "test_authorization_sha256": protocol.digest(protocol.HERE / "db_test_authorization.json"),
                     "training_seal_sha256": receipt["training_seal_sha256"],
                     "fresh_full_tests": True, "domains": summaries,
                     "release_reference_source": "user-provided, not a contemporaneous release evaluation",
                     "release_reference": {"dbbench": 0.65, "webshop": 0.455}}
            protocol.atomic_json(protocol.HERE / "db_test_results.json", value)
            lines = ["# Sealed twelve-round DBBench full test", "",
                     "Fresh evaluations under interface v3; historical tests were not reused.",
                     "Release references are user-provided, not a contemporaneous control.", "",
                     "| Domain | Successes | Accuracy | Infra errors | Release reference | Gap (pp) |",
                     "| --- | ---: | ---: | ---: | ---: | ---: |"]
            for domain, summary in summaries.items():
                reference = value["release_reference"][domain]
                lines.append(f"| {domain} | {summary['successes']}/{summary['n']} | "
                             f"{summary['accuracy']:.2%} | {summary['infra_errors']} | "
                             f"{reference:.2%} | {(summary['accuracy'] - reference) * 100:+.2f} |")
            (protocol.HERE / "DB_TEST_RESULTS.md").write_text("\n".join(lines) + "\n")
            event("complete", domains=summaries, training_seal_sha256=receipt["training_seal_sha256"])
        except Exception as exc:
            event("failed", error=f"{type(exc).__name__}: {exc}")
            traceback.print_exc()
            raise


if __name__ == "__main__":
    main()
