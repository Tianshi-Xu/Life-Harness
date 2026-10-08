"""Formal three-domain launcher around the unchanged original AHE loops."""
import argparse
import concurrent.futures
import hashlib
import json
import os
import resource
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .metered_broker import Broker, RequestGate
from .backend import HERE, MANIFEST_SHA256, workspace_hash
from .cli import config_for
from .capabilities import MAX_ANALYSIS_TASKS, PROFILE, restricted_policy
from .connection import bind_connections, credentials
from .debugger_isolation import bind_isolated_debugger
from .evolve_isolation import bind_isolated_evolve
from .isolation import ensure_shell_utilities, BASE
from .records import write_json
from .training import AGENTBENCH, make_schedule
from .upstream import AHE, bind_native_transport, load_upstream, source_fingerprint


def now():
    return datetime.now(timezone.utc).isoformat()


def progress(experiment):
    runs = sorted((experiment / "runs").glob("iteration_*"))
    result = {"current_iteration": 0, "evaluations": []}
    for run in runs:
        if not run.name[10:].isdigit():
            continue
        iteration = int(run.name[10:])
        result["current_iteration"] = max(result["current_iteration"], iteration)
        jobs = sorted((run / "input/benchmark").glob("20*/result.json"))
        if jobs:
            records = list(jobs[-1].parent.glob("native_*/native_record.json"))
            values = [json.loads(path.read_text()) for path in records]
            result["evaluations"].append({"iteration": iteration, "tasks": len(values),
                "successes": sum(value["success"] for value in values),
                "infrastructure_errors": sum(value["infrastructure_error"] for value in values)})
    if runs:
        latest = runs[-1]
        records = list((latest / "input/benchmark/.native_pending").glob("*/*/record.json"))
        result["current_completed_episodes"] = len(records)
        if (latest / "evolve/evolve_summary.md").is_file():
            result["phase"] = "iteration_complete"
        elif (latest / "evolve/isolated_process.log").is_file():
            result["phase"] = "evolve"
        elif (latest / "debugger/isolated_process.log").is_file():
            result["phase"] = "debugger"
        else:
            result["phase"] = "evaluation"
    return result


def reuse_baseline(module, source, destination, ids, capability_policy=None):
    source, destination = Path(source), Path(destination)
    snapshot = source / "runs/iteration_001/input/workspace"
    if (source / "runs/iteration_002").exists() or (source / "runs/iteration_001/evolve/evolve_summary.md").exists():
        raise ValueError("baseline reuse is only permitted before first evolution")
    if workspace_hash(source / "workspace") != workspace_hash(snapshot):
        raise ValueError("source baseline changed; cannot reuse its evaluation")
    jobs = sorted((source / "runs/iteration_001/input/benchmark").glob("20*/result.json"))
    if len(jobs) != 1:
        raise ValueError("resume requires exactly one completed baseline job")
    receipt = json.loads((jobs[0].parent / "workspace_receipt.json").read_text())
    if receipt["workspace_sha256"] != workspace_hash(snapshot):
        raise ValueError("measured baseline receipt differs from preserved snapshot")
    records = [json.loads(path.read_text()) for path in jobs[0].parent.glob("native_*/native_record.json")]
    if len(records) != len(ids) or {int(row["task_id"]) for row in records} != set(ids):
        raise ValueError("baseline evaluation differs from frozen first-round tasks")
    preserve_analysis = False
    if capability_policy:
        from .capabilities import validate_workspace
        validate_workspace(source / "workspace", capability_policy)
        previous_config = yaml.safe_load((source / "config_snapshot.yaml").read_text())
        if previous_config.get("native_capability_policy") == capability_policy:
            adb = previous_config["agent_debugger"]
            stats = module.compute_stats(jobs[0].parent, k=1)
            selected = module._build_adb_jobs(stats["task_results"], jobs[0].parent, adb,
                                             timeout_tasks=stats.get("timeout_tasks"))
            names = {job.task_name for job in selected}
            analysis = source / "runs/iteration_001/input/analysis"
            details = list((analysis / "detail").glob("*.md"))
            preserve_analysis = ((analysis / "overview.md").is_file() and bool(details)
                and 1 <= adb.get("max_tasks", 0) <= MAX_ANALYSIS_TASKS and adb.get("retry_attempts") == 1
                and len(details) <= adb["max_tasks"] and all(path.stem in names for path in details))
    shutil.copytree(source, destination)
    first = destination / "runs/iteration_001"
    for relative in ("debugger", "evolve", *([] if preserve_analysis else ["input/analysis"])):
        path = first / relative
        if path.exists():
            path.rename(path.with_name(path.name + "_before_rpm_guard_archive"))
    if preserve_analysis:
        for path in (first / "input/analysis").rglob("*.md"):
            path.write_text(path.read_text().replace(str(source), str(destination)))
    module.rollback_experiment_metadata(destination, 1)
    write_json(destination / "continuation_receipt.json", {"source_experiment": str(source),
        "source_workspace_sha256": workspace_hash(snapshot), "original_skip_eval": True,
        "completed_native_episodes_reused": len(records),
        "restricted_selected_analysis_reused": preserve_analysis,
        "reason": "continue pre-evolve integration/transport interruption; preserve already completed baseline and validated bounded analysis",
        "optimizer_source_modified": False})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--iterations", type=int, default=9)
    parser.add_argument("--training-tasks", type=int, default=50)
    parser.add_argument("--debugger-tasks", type=int, default=MAX_ANALYSIS_TASKS,
        help="restricted port: analyze/expose at most ten detailed tasks, not the entire evaluation batch")
    parser.add_argument("--flash-concurrency", type=int, default=8)
    parser.add_argument("--native-concurrency", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--flash-min-interval", type=float, default=1.25)
    parser.add_argument("--resume-from", help="reuse completed unchanged first evaluations from an interrupted pre-evolve batch")
    parser.add_argument("--measure-final-candidate", action="store_true",
        help="after the original loop, measure its final candidate on the last 50-task train batch without changing selection")
    args = parser.parse_args()
    if not args.name.replace("_", "").replace("-", "").isalnum():
        raise ValueError("batch name must be a simple isolated identifier")
    if args.flash_concurrency < 1 or args.native_concurrency < 1:
        raise ValueError("positive concurrency required")
    if not 1 <= args.debugger_tasks <= MAX_ANALYSIS_TASKS:
        raise ValueError("restricted Debugger analyzes at most ten tasks")
    batch_dir = HERE / "batches" / args.name
    names = {domain: f"{args.name}_{domain}" for domain in AGENTBENCH}
    if batch_dir.exists() or any((HERE / "experiments" / name).exists() for name in names.values()):
        raise FileExistsError("formal batch never overwrites an existing experiment")
    schedule = make_schedule(args.iterations, args.training_tasks, args.seed)
    source_protocol = None
    if args.resume_from:
        if not args.resume_from.replace("_", "").replace("-", "").isalnum():
            raise ValueError("resume source must be a simple batch name")
        source_dir = HERE / "batches" / args.resume_from
        source_protocol = json.loads((source_dir / "protocol.json").read_text())
        previous_schedule = json.loads((source_dir / "training_schedule.json").read_text())
        if previous_schedule != schedule:
            raise ValueError("resume must preserve the exact frozen training schedule")
    module, source_receipt = load_upstream()
    if not (BASE / ".ready").is_file():
        raise RuntimeError("isolated runtime must be prepared before the formal batch")
    ensure_shell_utilities()
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(max(soft, 8192), hard), hard))
    key, router_url = credentials()
    gate = RequestGate(args.flash_concurrency, min_interval=args.flash_min_interval)
    broker = Broker(key, router_url, request_gate=gate, rate_limit_retries=3,
                    ledger_path=batch_dir / "flash_requests.jsonl")
    os.environ.update(AHE_EVOLVE_API_KEY="EMPTY", AHE_EVOLVE_BASE_URL=broker.url,
        AHE_LOCAL_BROKER_URL=broker.url, QA_API_KEY="EMPTY", QA_BASE_URL=broker.url,
        QA_MODEL_NAME="deepseek-v4-flash", AHE_HOME=str(AHE),
        GIT_AUTHOR_NAME="AHE native integration", GIT_COMMITTER_NAME="AHE native integration",
        GIT_AUTHOR_EMAIL="ahe-native@localhost", GIT_COMMITTER_EMAIL="ahe-native@localhost")
    os.environ["PATH"] = str(HERE / ".venv/bin") + ":" + os.environ.get("PATH", "")
    batch_dir.mkdir(parents=True)
    schedule_path = batch_dir / "training_schedule.json"
    write_json(schedule_path, schedule)
    schedule_sha = hashlib.sha256(schedule_path.read_bytes()).hexdigest()
    bind_connections(module)
    bind_native_transport(module)
    bind_isolated_evolve(module)
    bind_isolated_debugger(module)
    module.EXPERIMENTS_DIR = HERE / "experiments"
    protocol = {"schema_version": 1, "name": args.name, "created_at": now(),
        "domains": list(AGENTBENCH), "iterations_per_domain": args.iterations,
        "train_tasks_per_iteration": args.training_tasks, "train_trials": 1,
        "training_schedule_sha256": schedule_sha, "historical_manifest_sha256": MANIFEST_SHA256,
        "training_policy": "pre-frozen disjoint 50-task batches from each native train split",
        "flash_api_global_concurrency": args.flash_concurrency,
        "flash_min_interval_seconds": args.flash_min_interval,
        "observed_api_key_rpm_limit": 60,
        "api_rate_policy": "global request pacing; on 429 defer all callers for 60 seconds and retry unchanged request",
        "native_episode_concurrency_per_domain": args.native_concurrency,
        "solver": "qwen3-4b", "solver_url": "http://127.0.0.1:8400/v1",
        "debugger_and_evolve": "deepseek-v4-flash", "target_pass_rate": 1.01,
        "fixed_budget_policy": "target 1.01 disables score-based early exit by configuration only",
        "explore_agent": False, "best_of_n": False,
        "optimizer": "original run_single_experiment, statistics, selection and Debugger/Evolve",
        "external_llm_token_accounting": "durable API-side attempt ledger; Debugger/Evolve/compaction/retries; input+output, cached input counted once; missing usage explicitly unknown",
        "token_ledger": str(batch_dir / "flash_requests.jsonl"),
        "round_semantics": "evaluate before evolve; last newly generated candidate is unmeasured",
        "cross_batch_scores": "not a continuous same-task improvement curve; original best-ever selection retained",
        "test_policy": "no test tasks, hidden answers or test results enter optimization",
        "initialization": "new blank native baselines; do not resume smoke or Life/Meta policies",
        "experiments": names}
    protocol.update(capability_profile=PROFILE, unrestricted_ahe_reproduction=False,
        debugger_tasks_per_iteration=args.debugger_tasks, debugger_retry_attempts=1,
        evidence_policy="at most ten current-iteration detailed tasks; previous raw traces hidden, decision summaries retained; full host evaluation preserved",
        disabled_capabilities=restricted_policy("dbbench")["disabled"])
    if args.measure_final_candidate:
        protocol.update(final_candidate_train_evaluation=True,
            final_candidate_policy="one extra same-last-batch train evaluation after original loop; not fed into optimization or best-ever selection",
            round_semantics=f"baseline is iteration 1; optimization step N uses evaluation N+1; step {args.iterations} gets a separate post-loop train evaluation",
            unique_train_tasks_per_domain=args.iterations * args.training_tasks,
            measured_train_episodes_per_domain=(args.iterations + 1) * args.training_tasks)
    if source_protocol:
        protocol.update(predecessor_batch=args.resume_from,
            continuation_policy="reuse three complete unchanged baseline evaluations; original skip_eval at iteration 1")
    write_json(batch_dir / "protocol.json", protocol)
    write_json(batch_dir / "source_receipt.json", source_receipt)
    configs = {}
    for domain, name in names.items():
        ids = schedule["domains"][domain]["batches"][0]["task_ids"]
        config = config_for(domain, ids, args.iterations, broker.url,
            debugger_tasks=args.debugger_tasks, concurrency=args.native_concurrency,
            debugger_concurrency=args.flash_concurrency, target_pass_rate=1.01,
            capability_policy=restricted_policy(domain))
        config["native_backend"].update(training_schedule_file=str(schedule_path),
            training_schedule_sha256=schedule_sha)
        config_path = batch_dir / f"{domain}.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        if source_protocol:
            source = HERE / "experiments" / source_protocol["experiments"][domain]
            destination = HERE / "experiments" / name
            reuse_baseline(module, source, destination, ids, config["native_capability_policy"])
        configs[domain] = (config, config_path)
    status = {"name": args.name, "pid": os.getpid(), "started_at": now(),
              "state": "running", "domains": {}}

    def run(domain):
        config, config_path = configs[domain]
        module.run_single_experiment(config, str(config_path), experiment_name=names[domain],
            skip_eval=bool(source_protocol))
        if source_fingerprint() != source_receipt:
            raise RuntimeError("official source changed during experiment")
        experiment = HERE / "experiments" / names[domain]
        measured = progress(experiment)["evaluations"]
        if len(measured) != args.iterations or any(row["tasks"] != args.training_tasks for row in measured):
            raise RuntimeError("original loop returned before completing the frozen training budget")
        write_json(experiment / "core_fidelity.json", {"official_commit": source_receipt["commit"],
            "upstream_source_modified": False, "smoke_only": False,
            "restricted_search_space": True, "capability_profile": PROFILE,
            "unrestricted_ahe_reproduction": False,
            "search_loop": "original run_single_experiment", "statistics": "original compute_stats",
            "selection": "original update_best_ever", "optimizer": "original isolated Debugger/Evolve",
            "batch_protocol": str(batch_dir / "protocol.json"), "training_schedule_sha256": schedule_sha})
        if args.measure_final_candidate:
            final_dir = batch_dir / "final_candidate_evaluations" / domain / f"iteration_{args.iterations:03d}"
            final_dir.mkdir(parents=True)
            job = module.run_harbor(config, experiment / "workspace", "code_agent.yaml", final_dir)
            records = [json.loads(path.read_text()) for path in job.glob("native_*/native_record.json")]
            if len(records) != args.training_tasks:
                raise RuntimeError("incomplete final candidate train evaluation")
            write_json(batch_dir / f"{domain}_final_candidate_train.json", {
                "domain": domain, "optimization_step": args.iterations, "split": "train",
                "tasks": len(records), "successes": sum(row["success"] for row in records),
                "infrastructure_errors": sum(row["infrastructure_error"] for row in records),
                "workspace_sha256": workspace_hash(experiment / "workspace"), "job": str(job),
                "batch": args.iterations - 1, "selection_modified": False,
                "used_for_optimization": False})

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(AGENTBENCH)) as pool:
            futures = {domain: pool.submit(run, domain) for domain in AGENTBENCH}
            reported_errors = set()
            while True:
                for domain, future in futures.items():
                    value = progress(HERE / "experiments" / names[domain])
                    value["experiment"] = names[domain]
                    value["state"] = "running"
                    if future.done():
                        error = future.exception()
                        value["state"] = "failed" if error else "complete"
                        if error and domain not in reported_errors:
                            reported_errors.add(domain)
                            value["error_type"] = type(error).__name__
                            print(f"[batch] {domain} failed: {type(error).__name__}: {error}", flush=True)
                    status["domains"][domain] = value
                status.update(updated_at=now(), flash_api=gate.snapshot(),
                              token_usage=broker.ledger.summary()["overall"])
                status["flash_rate_limit_responses"] = broker.rate_limit_responses
                if all(future.done() for future in futures.values()):
                    status["state"] = "failed" if any(future.exception() for future in futures.values()) else "complete"
                write_json(batch_dir / "status.json", status)
                from .comparison import publish_comparison
                publish_comparison(batch_dir)
                if status["state"] != "running":
                    break
                time.sleep(5)
    finally:
        broker.close()
        write_json(batch_dir / "api_transport_receipt.json", {
            "real_credentials_in_isolated_agents": False, "flash_api": gate.snapshot(),
            "thinking_disabled": False, "reasoning_restoration_hits": broker.reasoning.hits,
            "reasoning_restoration_misses": broker.reasoning.misses,
            "responses_with_reasoning": broker.reasoning.with_reasoning,
            "responses_without_reasoning": broker.reasoning.without_reasoning,
            "token_usage": broker.ledger.summary(),
            "token_ledger": str(broker.ledger.path)})
    if status["state"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
