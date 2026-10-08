"""Exactly three train-only rounds per domain, with a durable completion record."""
import fcntl
import json
import os
import subprocess
import sys
import time
import traceback

import driver
import protocol
import status


def event(state, **extra):
    value = {"status": state, "pid": os.getpid(), "updated_at": time.time(), **extra}
    protocol.atomic_json(protocol.HERE / "workflow_state.json", value)
    print(json.dumps(value), flush=True)


def report():
    seal_path, seal = status.seal_training()
    rows = []
    counts = {}
    for domain in protocol.DOMAINS:
        folder = protocol.HERE / domain
        state = json.loads((folder / "state.json").read_text())
        new_rounds = state["rounds"][9:]
        if [row["round"] for row in new_rounds] != [10, 11, 12]:
            raise RuntimeError("not exactly three new rounds")
        rows.extend({"domain": domain, **row} for row in new_rounds)
        evaluations = {}
        for path in sorted((folder / "evals").glob("*/summary.json")):
            summary = json.loads(path.read_text())
            if summary["split"] != "train":
                raise RuntimeError("this continuation is train-only")
            __import__("agentbench_evaluate").validate_cached_summary(path.parent, summary)
            evaluations[path.parent.name] = {key: summary[key] for key in
                                            ("n", "successes", "infra_errors", "config_sha256")}
        seed = __import__("inheritance").verified_seed(domain)
        inherited_baseline = seed.get("baseline_summary")
        if inherited_baseline:
            value = driver._phase_summary_path(domain, seed["baseline_phase"])
            evaluations["reused_" + seed["baseline_phase"]] = {
                **{key: inherited_baseline[key] for key in ("n", "successes", "infra_errors", "config_sha256")},
                "reused": True, "source_summary": str(value),
                "source_sha256": protocol.digest(value)}
        counts[domain] = evaluations
    value = {"experiment": protocol.protocol()["name"],
             "training_seal_sha256": protocol.digest(seal_path),
             "interface": protocol.protocol()["interface_version"],
             "inherited_rounds": 9, "new_rounds": rows,
             "train_evaluations": counts, "final_tests_run": False,
             "historical_artifacts_preserved": True}
    protocol.atomic_json(protocol.HERE / "training_results.json", value)
    lines = ["# Three-round v3 continuation (train only)", "",
             "Inherited nine rounds unchanged; identical v3 interface, API DeepSeek Flash proposals.",
             "No released policy was supplied. No new final test was run.", "",
             "| Domain | Round | Batch | Incumbent | Candidate | Decision |",
             "| --- | ---: | ---: | ---: | ---: | --- |"]
    for row in rows:
        lines.append(f"| {row['domain']} | {row['round']} | {row['train_batch']} | "
                     f"{round(row['incumbent_score'] * 50)}/50 | "
                     f"{round(row['candidate_score'] * 50)}/50 | "
                     f"{'accepted' if row['accepted'] else 'rejected'} |")
    (protocol.HERE / "TRAINING_RESULTS.md").write_text("\n".join(lines) + "\n")
    return value


def main():
    with (protocol.HERE / ".workflow.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock.write(str(os.getpid()) + "\n")
        lock.flush()
        try:
            protocol.validate_protocol()
            # Configuration check only: never print or persist the credential.
            key, host = driver._model_router_config()
            if not key or host != protocol.protocol()["proposer_host"]:
                raise RuntimeError("authorized API configuration unavailable")
            del key
            for domain in protocol.DOMAINS:
                path = protocol.HERE / domain / "evals/smoke_inherited/summary.json"
                summary = json.loads(path.read_text())
                __import__("agentbench_evaluate").validate_cached_summary(path.parent, summary)
                if summary["n"] != 1 or summary["infra_errors"]:
                    raise RuntimeError("train-only interface smoke did not pass")
            event("training", domains=list(protocol.DOMAINS), rounds=[10, 11, 12])
            subprocess.run([sys.executable, "-u", str(protocol.HERE / "runner.py"), "train"],
                           cwd=protocol.ROOT, check=True)
            value = report()
            event("complete", training_seal_sha256=value["training_seal_sha256"],
                  new_decisions=len(value["new_rounds"]), final_tests_run=False)
        except Exception as exc:
            event("failed", error=f"{type(exc).__name__}: {exc}")
            traceback.print_exc()
            raise


if __name__ == "__main__":
    main()
