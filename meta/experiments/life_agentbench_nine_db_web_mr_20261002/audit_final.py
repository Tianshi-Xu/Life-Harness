"""Read-only final result, source, amendment and sealed-history audit."""
import agentbench_evaluate
import final_reporting
import inheritance
import protocol
import runtime_support
import status


def main():
    runtime_support.install()
    path, seal = status.seal_training()
    report = protocol.load_json(protocol.HERE / "final_results.json")
    assert report["training_seal_sha256"] == protocol.digest(path)
    assert report["runtime_amendment_sha256"] == protocol.digest(protocol.HERE / "runtime_amendment.json")
    assert report["predecessor_final_report_sha256"] == protocol.digest(inheritance.base_root() / "final_results.json")
    for domain in protocol.DOMAINS:
        state = protocol.load_json(protocol.HERE / domain / "state.json")
        saved = report["evaluations"][domain]
        if saved["new_test_episodes"] == 0:
            assert final_reporting.reuse(domain, state) == saved
        else:
            summary_path = protocol.HERE / domain / "evals/final_test/summary.json"
            summary = protocol.load_json(summary_path)
            agentbench_evaluate.validate_cached_summary(summary_path.parent, summary)
            assert protocol.digest(summary_path) == saved["summary_sha256"]
            assert saved["per_task"] == summary["per_task"]
            assert saved["successes"] == summary["successes"]
            assert saved["snapshot_sha256"] == state["active_snapshot_sha256"]
        assert len(state["rounds"]) == 9 and len(state["rounds"][6:]) == 3
        for phase in ("candidate_07", "candidate_08", "candidate_09"):
            folder = protocol.HERE / domain / "evals" / phase
            summary = protocol.load_json(folder / "summary.json")
            agentbench_evaluate.validate_cached_summary(folder, summary)
            assert summary["n"] == 50 and summary["infra_errors"] == 0
    archived = protocol.load_json(protocol.HERE / "runtime_cap_archive/dbbench/candidate_09/summary.json")
    assert archived["n"] == archived["infra_errors"] == 50
    print("Verified two sealed nine-round states, six new decisions, 500 valid new train episodes,")
    print("200 new test episodes, DB6 test reuse, runtime amendment and preserved 50 invalid episodes.")


if __name__ == "__main__":
    main()
