"""One fixed held-out evaluation, outside AHE's optimizer and selection history."""
import argparse
import concurrent.futures
import hashlib
import json
import os
import resource
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from .backend import HERE, LIFE, MANIFEST, MANIFEST_SHA256, interpreter, native_env, workspace_hash
from .capabilities import PROFILE, restricted_policy, validate_workspace
from .records import write_json
from .training import AGENTBENCH
from .upstream import source_fingerprint


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identifier(value):
    if not value or not value.replace("_", "").replace("-", "").isalnum():
        raise ValueError("use a simple isolated name")
    return value


def completed_training(batch_dir):
    """Fail closed until ALL domains finish six evolutions and the last train check."""
    batch_dir = Path(batch_dir)
    protocol = json.loads((batch_dir / "protocol.json").read_text())
    status = json.loads((batch_dir / "status.json").read_text())
    if protocol["iterations_per_domain"] != 6 or protocol.get("capability_profile") != PROFILE:
        raise ValueError("final test requires the declared restricted six-round protocol")
    if status["state"] != "complete" or set(status["domains"]) != set(AGENTBENCH):
        raise ValueError("all three training domains must be complete before any test")
    if source_fingerprint() != json.loads((batch_dir / "source_receipt.json").read_text()):
        raise ValueError("official AHE source changed since training")
    if digest(batch_dir / "training_schedule.json") != protocol["training_schedule_sha256"]:
        raise ValueError("training schedule changed since launch")
    result = {}
    for domain in AGENTBENCH:
        state = status["domains"][domain]
        if state["state"] != "complete" or state["current_iteration"] != 6:
            raise ValueError("all three training domains must finish six rounds")
        evaluations = state["evaluations"]
        if ([row["iteration"] for row in evaluations] != list(range(1, 7))
                or any(row["tasks"] != protocol["train_tasks_per_iteration"] for row in evaluations)):
            raise ValueError("incomplete fixed training evaluation budget")
        experiment_name = identifier(protocol["experiments"][domain])
        experiment = HERE / "experiments" / experiment_name
        summaries = [experiment / f"runs/iteration_{i:03d}/evolve/evolve_summary.md" for i in range(1, 7)]
        if any(not path.is_file() for path in summaries):
            raise ValueError("six completed Evolve receipts are required")
        workspace = experiment / "workspace"
        validate_workspace(workspace, restricted_policy(domain))
        receipt_path = batch_dir / f"{domain}_final_candidate_train.json"
        receipt = json.loads(receipt_path.read_text())
        measured_hash = workspace_hash(workspace)
        if (receipt.get("split") != "train" or receipt.get("optimization_step") != 6
                or receipt.get("tasks") != protocol["train_tasks_per_iteration"]
                or receipt.get("workspace_sha256") != measured_hash
                or receipt.get("selection_modified") is not False
                or receipt.get("used_for_optimization") is not False):
            raise ValueError("last candidate must match its separate final train measurement")
        result[domain] = {"workspace": str(workspace), "workspace_sha256": measured_hash,
            "final_train_receipt_sha256": digest(receipt_path),
            "evolve_summary_sha256": {str(p.relative_to(experiment)): digest(p) for p in summaries}}
    return {"batch": protocol["name"], "optimization_step": 6, "domains": result,
        "source_receipt_sha256": digest(batch_dir / "source_receipt.json"),
        "protocol_sha256": digest(batch_dir / "protocol.json"),
        "status_sha256": digest(batch_dir / "status.json"),
        "training_schedule_sha256": protocol["training_schedule_sha256"]}


def fixed_test_spec():
    if digest(MANIFEST) != MANIFEST_SHA256:
        raise ValueError("frozen benchmark manifest changed")
    manifest = json.loads(MANIFEST.read_text())
    snapshot = MANIFEST.parent / "task_snapshot"
    for relative, expected in manifest["snapshot_hashes"].items():
        if digest(snapshot / relative) != expected:
            raise ValueError(f"frozen grader source changed: {relative}")
    # Verify all pinned native data, including test ordering and item catalogs.
    # Only hashes/IDs enter this host-side receipt; no oracle is passed to solvers.
    for relative, expected in manifest["data_hashes"].items():
        if digest(LIFE / "AgentBench" / relative) != expected:
            raise ValueError(f"frozen native data changed: {relative}")
    result = {}
    for domain in AGENTBENCH:
        ids = manifest["domains"][domain]["test_indices"]
        if not ids or ids != list(range(len(ids))):
            raise ValueError("fixed complete native test indices required")
        result[domain] = list(ids)
    return {"split": "test", "task_ids": result, "manifest_sha256": MANIFEST_SHA256,
        "native_data_hashes": manifest["data_hashes"],
        "native_grader_hashes": manifest["snapshot_hashes"]}


def check_record(record, task_id):
    if (record.get("task_id") != str(task_id) or record.get("trial") != 0
            or type(record.get("success")) is not int or record["success"] not in (0, 1)
            or type(record.get("infrastructure_error")) is not bool
            or (record["infrastructure_error"] and record["success"])):
        raise ValueError("invalid native test record or fixed-denominator outcome")


def run_episode(domain, task_id, workspace, folder):
    folder.mkdir(parents=True)
    request = {"domain": domain, "split": "test", "task_id": str(task_id), "trial": 0,
        "workspace": str(workspace), "agent_config": "code_agent.yaml",
        "trace_dir": str(folder / "agent"), "capability_policy": restricted_policy(domain)}
    write_json(folder / "request.json", request)
    env = native_env(domain)
    # AgentBench testing uses local Qwen only. Do not load a router key or broker.
    env.update(LLM_API_KEY="EMPTY", LLM_BASE_URL="http://127.0.0.1:8400/v1", LLM_MODEL="qwen3-4b")
    start = time.monotonic()
    try:
        with (folder / "environment.log").open("w") as log:
            proc = subprocess.run([str(interpreter(domain)), "-m", "ahe_native.native_worker",
                "--request", str(folder / "request.json"), "--out", str(folder / "record.json")],
                cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
        if proc.returncode or not (folder / "record.json").is_file():
            raise RuntimeError("native test worker failed; see its environment.log")
        record = json.loads((folder / "record.json").read_text())
        check_record(record, task_id)
    except (subprocess.TimeoutExpired, RuntimeError) as exc:
        record = {"task_id": str(task_id), "trial": 0, "success": 0,
            "infrastructure_error": True, "status": type(exc).__name__,
            "wall_seconds": time.monotonic() - start}
        # Preserve any partial diagnostic file, never retry a failed test episode.
        original = folder / "record.json"
        if original.exists():
            original.rename(folder / "worker_partial_record.json")
        write_json(original, record)
    return record


def evaluate_domain(domain, ids, workspace, output, concurrency):
    expected_hash = workspace_hash(workspace)
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(run_episode, domain, task, workspace,
                              output / "episodes" / str(task)) for task in ids]
        records = [future.result() for future in futures]
    if workspace_hash(workspace) != expected_hash:
        raise ValueError("frozen test workspace changed during evaluation")
    if len(records) != len(ids) or {r["task_id"] for r in records} != {str(i) for i in ids}:
        raise ValueError("incomplete fixed test split")
    summary = {"domain": domain, "split": "test", "optimization_step": 6,
        "n": len(ids), "ids": ids, "successes": sum(r["success"] for r in records),
        "infra_errors": sum(r["infrastructure_error"] for r in records),
        "accuracy": sum(r["success"] for r in records) / len(ids),
        "workspace_sha256": expected_hash, "used_for_optimization": False,
        "candidate_selection": "final sixth-round workspace, not chosen by test score",
        "modelrouter_test_requests": 0, "per_task": {r["task_id"]: r["success"] for r in records},
        "episode_sha256": {f"episodes/{i}/record.json": digest(output / "episodes" / str(i) / "record.json") for i in ids}}
    write_json(output / "summary.json", summary)
    return summary


def comparison_report(results, output):
    reference = LIFE / "evolution/experiments/life_agentbench_six_resume_mr_20261001/final_results.json"
    if not reference.is_file():
        return {"rows": {}, "historical_reference_available": False}
    historical = json.loads(reference.read_text())["comparisons"]
    # User-provided release references, not measurements made by this runner.
    released = {"dbbench": 0.65, "webshop": 0.455}
    rows = {}
    lines = ["# AHE fixed TEST results after six optimization steps", "",
        "AHE uses the h2345-budget-restricted-v1 profile. Each domain's final sixth-step candidate is evaluated without selecting versions by TEST scores.",
        "Meta-Harness/Life-Harness scores are historical six-round TEST results. Release reference values were supplied by the user and were not remeasured in this run.", "",
        "| Environment | AHE six-round test | Meta-Harness six-round test | Life-Harness six-round test | Release reference |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for domain in AGENTBENCH:
        measured = results[domain]
        old = historical[domain]
        if old["n"] != measured["n"]:
            raise ValueError("historical test reference has a different denominator")
        rows[domain] = {"ahe": measured["accuracy"], "meta": old["meta_test"],
            "life_six": old["life_six_test"], "release_user_reference": released.get(domain),
            "ahe_below_life_six": measured["accuracy"] < old["life_six_test"],
            "ahe_below_release": (measured["accuracy"] < released[domain]) if domain in released else None}
        release = f"{released[domain]:.2%}" if domain in released else "Not provided"
        lines.append(f"| {domain} | {measured['successes']}/{measured['n']} = {measured['accuracy']:.2%}"
            f" | {old['meta_test']:.2%} | {old['life_six_test']:.2%} | {release} |")
    lines.extend(["", "Training procedures, interfaces and some evaluation settings differ across methods. Sharing a fixed TEST split does not imply identical reproduction conditions or establish statistical significance.",
        "All TEST infrastructure failures count as zero in the full denominator. There are no retries, optimization from TEST feedback or extra Flash TEST calls.",
        "See COMPARISON.md in the original training batch for per-round TRAIN comparisons; TRAIN and TEST scores are not combined into one curve.", ""])
    (output / "COMPARISON.md").write_text("\n".join(lines))
    return {"rows": rows, "historical_six_test_source": str(reference),
        "historical_six_test_sha256": digest(reference), "release_reference_source": "user-provided"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--wait", action="store_true", help="wait without model calls until ALL training domains complete")
    args = parser.parse_args()
    batch = HERE / "batches" / identifier(args.batch)
    output = HERE / "heldout_tests" / identifier(args.name)
    if output.exists():
        raise FileExistsError("never overwrite a held-out evaluation")
    if args.concurrency < 1:
        raise ValueError("positive local native concurrency required")
    output.mkdir(parents=True)
    status = {"state": "waiting_for_training", "batch": args.batch, "pid": os.getpid(),
        "created_at": now(), "domains": {}, "split": "test", "modelrouter_requests": 0}
    write_json(output / "status.json", status)
    try:
        while True:
            training_status = json.loads((batch / "status.json").read_text())
            if training_status["state"] == "complete":
                break
            if training_status["state"] != "running" or not args.wait:
                raise ValueError("training is not complete; no test episode was started")
            failed = [d for d, v in training_status["domains"].items() if v["state"] == "failed"]
            if failed:
                raise ValueError("training domain failed; no test episode was started")
            status.update(updated_at=now())
            write_json(output / "status.json", status)
            time.sleep(10)
        seal = completed_training(batch)
        spec = fixed_test_spec()
        for domain, value in seal["domains"].items():
            source = Path(value["workspace"])
            destination = output / domain / "workspace"
            shutil.copytree(source, destination, ignore=shutil.ignore_patterns(".git", "__pycache__"))
            if workspace_hash(destination) != value["workspace_sha256"]:
                raise ValueError("frozen test snapshot differs from final train snapshot")
            value["frozen_test_workspace"] = str(destination)
        write_json(output / "training_seal.json", seal)
        write_json(output / "test_spec.json", spec)
        write_json(output / "protocol.json", {"split": "test", "training_batch": str(batch),
            "optimization_step": 6, "task_counts": {d: len(ids) for d, ids in spec["task_ids"].items()},
            "native_concurrency_per_domain": args.concurrency, "solver": "qwen3-4b",
            "capability_profile": PROFILE, "modelrouter_requests": 0,
            "test_feedback_to_optimizer": False, "test_based_selection": False,
            "episode_retries": 0, "infra_errors_count_as_zero": True})
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(max(soft, 8192), hard), hard))
        status["state"] = "running"
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = {d: pool.submit(evaluate_domain, d, spec["task_ids"][d],
                output / d / "workspace", output / d, args.concurrency) for d in AGENTBENCH}
            while True:
                for domain, future in futures.items():
                    status["domains"][domain] = {"state": "running",
                        "completed": len(list((output / domain / "episodes").glob("*/record.json"))),
                        "total": len(spec["task_ids"][domain])}
                    if future.done():
                        status["domains"][domain]["state"] = "failed" if future.exception() else "complete"
                status.update(updated_at=now())
                write_json(output / "status.json", status)
                if all(f.done() for f in futures.values()):
                    break
                time.sleep(5)
            results = {d: f.result() for d, f in futures.items()}
        # Verify that reporting did not alter the original trained harnesses.
        for value in seal["domains"].values():
            if workspace_hash(value["workspace"]) != value["workspace_sha256"]:
                raise ValueError("original final training workspace changed during test")
        write_json(output / "final_results.json", {"split": "test", "evaluations": results,
            "training_seal_sha256": digest(output / "training_seal.json"),
            "test_spec_sha256": digest(output / "test_spec.json"), "used_for_optimization": False,
            "comparisons": comparison_report(results, output)})
        status.update(state="complete", updated_at=now())
        write_json(output / "status.json", status)
    except Exception as exc:
        status.update(state="failed", updated_at=now(), error_type=type(exc).__name__, error=str(exc))
        write_json(output / "status.json", status)
        raise


if __name__ == "__main__":
    main()
