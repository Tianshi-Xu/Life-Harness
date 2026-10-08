"""Run upstream AHE with native evaluation transport, without optimizer replacement."""
import argparse
import json
import os
import shutil
from pathlib import Path
import yaml

from .backend import HERE, MANIFEST
from .connection import bind_connections, configure_environment
from .evolve_isolation import bind_isolated_evolve
from .debugger_isolation import bind_isolated_debugger
from .metered_broker import Broker
from .records import write_json
from .upstream import AHE, bind_native_transport, load_upstream, source_fingerprint


def config_for(domain, task_ids, iterations, router_url, debugger_tasks=3,
               concurrency=2, debugger_concurrency=2, target_pass_rate=0.95,
               capability_policy=None):
    config = {"_meta": {"_name": f"ahe_native_{domain}"},
        "target_pass_rate": target_pass_rate, "max_iterations": iterations,
        "source_config_dir": str(HERE / "baselines" / domain),
        "agent_config_filename": "code_agent.yaml", "dataset": f"native-{domain}-train",
        "harbor": {"agent": "nexau", "env": "native", "k": 1, "n_concurrent": concurrency},
        "llm": {"api_key": "EMPTY", "base_url": "http://127.0.0.1:8400/v1", "model": "qwen3-4b"},
        "evolve_agent": {"llm_config": {"api_key": "${env.AHE_EVOLVE_API_KEY}",
            "base_url": "${env.AHE_EVOLVE_BASE_URL}", "model": "deepseek-v4-flash"}},
        "agent_debugger": {"enabled": True, "max_concurrent": debugger_concurrency, "max_tasks": min(debugger_tasks, len(task_ids)),
            "timeout_per_task": 600, "retry_attempts": 3, "retry_backoff_seconds": 2,
            "llm": {"model": "deepseek-v4-flash", "base_url": router_url}},
        "explore_agent": {"enabled": False}, "best_of_n": {"enabled": False},
        "notify": {"enabled": False}, "post_evolve": {"enabled": False},
        "native_backend": {"command": [str(HERE / ".venv/bin/python"), "-m", "ahe_native.backend"],
            "domain": domain, "split": "train", "task_ids": task_ids}}
    if capability_policy is not None:
        from .capabilities import MAX_ANALYSIS_TASKS, check_policy
        check_policy(capability_policy)
        if not 1 <= debugger_tasks <= MAX_ANALYSIS_TASKS:
            raise ValueError("restricted Debugger analyzes at most ten tasks")
        config["native_capability_policy"] = capability_policy
        config["native_backend"]["capability_policy"] = capability_policy
        config["agent_debugger"]["retry_attempts"] = 1
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("domain", choices=["dbbench","webshop","alfworld","airline","retail","telecom"])
    parser.add_argument("--name", required=True)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--smoke-tasks", type=int, default=2)
    parser.add_argument("--debugger-tasks", type=int, default=10)
    parser.add_argument("--reuse-evaluation-from", help="copy a failed pre-evolve smoke into a NEW experiment and reuse its measured training job")
    parser.add_argument("--resume-evolved-from", help="copy a second-round infrastructure failure and resume original iteration 2 from its unchanged candidate snapshot")
    args = parser.parse_args()
    if not args.name.replace("_", "").replace("-", "").isalnum():
        raise ValueError("experiment name must be an isolated simple identifier")
    if args.iterations < 1 or args.smoke_tasks < 1 or args.debugger_tasks < 1:
        raise ValueError("positive smoke sizes required")
    if args.reuse_evaluation_from and args.resume_evolved_from:
        raise ValueError("choose one continuation mode")
    experiment = HERE / "experiments" / args.name
    if experiment.exists():
        raise FileExistsError("smoke launcher never overwrites or resumes a previous experiment")
    module, receipt = load_upstream()
    router_url = configure_environment()
    broker = Broker(os.environ["AHE_EVOLVE_API_KEY"], router_url,
                    ledger_path=HERE / "token_ledgers" / args.name / "flash_requests.jsonl")
    os.environ["AHE_LOCAL_BROKER_URL"] = broker.url
    # Original resume patches serialize get_llm_config output. Ensure even that
    # original path can see only the harmless local transport credentials.
    os.environ.update(AHE_EVOLVE_API_KEY="EMPTY", AHE_EVOLVE_BASE_URL=broker.url,
                      QA_API_KEY="EMPTY", QA_BASE_URL=broker.url)
    os.environ["PATH"] = str(HERE / ".venv/bin") + ":" + os.environ.get("PATH", "")
    os.environ["QA_MODEL_NAME"] = "deepseek-v4-flash"
    os.environ["AHE_HOME"] = str(AHE)
    # Dedicated per-launch subprocess HOME; never modifies the user's home state.
    runtime_home = HERE / "runtime_homes" / args.name
    runtime_home.mkdir(parents=True)
    os.environ["HOME"] = str(runtime_home)
    os.environ["GIT_AUTHOR_NAME"] = os.environ["GIT_COMMITTER_NAME"] = "AHE native integration"
    os.environ["GIT_AUTHOR_EMAIL"] = os.environ["GIT_COMMITTER_EMAIL"] = "ahe-native@localhost"
    historical = json.loads(MANIFEST.read_text())["domains"][args.domain]
    first = historical["batches"][0]
    ids = first.get("indices", first.get("task_ids"))[:args.smoke_tasks]
    from .capabilities import restricted_policy
    config = config_for(args.domain, ids, args.iterations, broker.url, args.debugger_tasks,
                        capability_policy=restricted_policy(args.domain))
    bind_connections(module)
    bind_native_transport(module)
    bind_isolated_evolve(module)
    bind_isolated_debugger(module)
    module.EXPERIMENTS_DIR = HERE / "experiments"
    reused = False
    start_iteration = 1
    if args.resume_evolved_from:
        if not args.resume_evolved_from.replace("_", "").replace("-", "").isalnum():
            raise ValueError("resume source must be an isolated experiment name")
        source = HERE / "experiments" / args.resume_evolved_from
        previous = yaml.safe_load((source / "config_snapshot.yaml").read_text())
        if previous["native_backend"] != config["native_backend"]:
            raise ValueError("resume requires identical native training tasks")
        from .backend import workspace_hash
        snapshot = source / "runs/iteration_002/input/workspace"
        if workspace_hash(source / "workspace") != workspace_hash(snapshot):
            raise ValueError("candidate changed after second-round snapshot; audit before resuming")
        if not (source / "runs/iteration_001/evolve/evolve_summary.md").is_file():
            raise ValueError("no completed original first evolution")
        job = module.find_latest_job_dir(source / "runs/iteration_002/input/benchmark")
        records = [json.loads(p.read_text()) for p in job.glob("native_*/native_record.json")] if job else []
        if len(records) != len(ids) or not all(r["infrastructure_error"] for r in records):
            raise ValueError("resume mode is only for a full second-round infrastructure failure")
        shutil.copytree(source, experiment)
        second = experiment / "runs/iteration_002"
        for relative in ("input/benchmark", "input/analysis", "debugger", "input/change_evaluation.json"):
            path = second / relative
            if path.exists():
                path.rename(path.with_name(path.name + "_infra_failed_previous"))
        start_iteration = 2
        write_json(experiment / "candidate_continuation_receipt.json", {
            "source_experiment": str(source), "source_workspace_sha256": workspace_hash(snapshot),
            "reason": "rerun unchanged candidate after fixing solver copy permissions",
            "original_resume_start_iteration": 2, "prior_infrastructure_job_archived": str(job),
            "policy_manually_modified": False})
    if args.reuse_evaluation_from:
        if not args.reuse_evaluation_from.replace("_", "").replace("-", "").isalnum():
            raise ValueError("reuse source must be an isolated experiment name")
        source = HERE / "experiments" / args.reuse_evaluation_from
        previous = yaml.safe_load((source / "config_snapshot.yaml").read_text())
        if previous["native_backend"] != config["native_backend"]:
            raise ValueError("reuse requires exactly the same native domain and training tasks")
        if (source / "runs/iteration_002").exists():
            raise ValueError("reuse is limited to a pre-evolve first-round infrastructure failure")
        snapshot = source / "runs/iteration_001/input/workspace"
        from .backend import workspace_hash
        if workspace_hash(source / "workspace") != workspace_hash(snapshot):
            raise ValueError("failed source workspace has already changed; cannot reuse baseline results")
        if module.find_latest_job_dir(source / "runs/iteration_001/input/benchmark") is None:
            raise ValueError("no completed training job to reuse")
        shutil.copytree(source, experiment)
        # Re-export genuine raw spans with the same public-context options used
        # by upstream Evolve traces. Older smoke exports omitted system/user
        # turns by the converter's defaults; native scores/raw spans are intact.
        from trace_converter import extract_trace_data_from_inmemory_dump
        converted = []
        for raw_path in (experiment / "runs/iteration_001/input/benchmark").glob("20*/native_*/agent/nexau_in_memory_tracer.json"):
            cleaned = extract_trace_data_from_inmemory_dump(json.loads(raw_path.read_text()),
                include_system_prompt_message=True, include_user_message=True,
                capture_errors=True, jsonable_output=True)
            if not any(m.get("role") == "user" for m in cleaned.get("messages", [])):
                raise ValueError("genuine raw trace lacks public user context")
            write_json(raw_path.with_name("nexau_in_memory_tracer.cleaned.json"), cleaned)
            converted.append(str(raw_path.relative_to(experiment)))
        analysis = experiment / "runs/iteration_001/input/analysis"
        if analysis.exists() and "[adb error]" in (analysis / "overview.md").read_text():
            analysis.rename(analysis.with_name("analysis_infra_failed_previous"))
        module.rollback_experiment_metadata(experiment, 1)
        write_json(experiment / "continuation_receipt.json", {
            "source_experiment": str(source), "source_workspace_sha256": workspace_hash(snapshot),
            "reason": "retry infrastructure-only analysis/evolve failure; retain original artifacts",
            "public_context_reexport_from_genuine_raw": converted,
            "test_results_reused": False})
        reused = True
    try:
        module.run_single_experiment(config, str(HERE / "README.md"), experiment_name=args.name,
            start_iteration=start_iteration, skip_eval=reused)
        if source_fingerprint() != receipt:
            raise RuntimeError("upstream source changed during smoke")
        write_json(experiment / "core_fidelity.json", {
            **receipt, "upstream_source_modified": False,
            "boundary_wrappers": ["launch_harbor", "wait_for_harbor", "get_llm_config", "run_evolve_agent process transport", "run_parallel_adb_ask process transport"],
            "search_loop": "original run_single_experiment", "statistics": "original compute_stats",
            "analysis": "original Agent Debugger", "evolve_invocation": "original run_evolve_agent in chroot",
            "smoke_only": True, "restricted_search_space": True,
            "capability_profile": config["native_capability_policy"]["profile"],
            "unrestricted_ahe_reproduction": False,
            "prior_training_evaluation_reused": reused or start_iteration > 1,
            "original_resume_start_iteration": start_iteration,
            "configuration_deviations": {"explore_agent": "disabled for smoke", "k": 1,
                "solver": "qwen3-4b", "proposer_debugger": "deepseek-v4-flash",
                "debugger_max_tasks": config["agent_debugger"]["max_tasks"],
                "target_benchmarks": "native public training subsets"}})
    finally:
        broker.close()
        write_json(experiment / "api_transport_receipt.json", {
            "real_credentials_in_isolated_agents": False,
            "token_usage": broker.ledger.summary(),
            "token_ledger": str(broker.ledger.path),
            "thinking_disabled": False,
            "reasoning_restoration_hits": broker.reasoning.hits,
            "reasoning_restoration_misses": broker.reasoning.misses,
            "responses_with_reasoning": broker.reasoning.with_reasoning,
            "responses_without_reasoning": broker.reasoning.without_reasoning,
            "last_api_response_tool_ids": broker.reasoning.response_ids,
            "last_missing_tool_ids": broker.reasoning.miss_ids,
            "missing_reasoning_diagnostics": broker.reasoning.miss_details,
            "restoration": "only actual API responses matched to prefix or globally unique unambiguous tool-call IDs"})


if __name__ == "__main__":
    main()
