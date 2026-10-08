"""Ten-task evidence projection on disposable isolation copies only."""
import re
import shutil
from pathlib import Path

from .capabilities import MAX_ANALYSIS_TASKS, check_policy


def project_public_evidence(experiment, config, module, current_iteration):
    experiment = Path(experiment)
    if (experiment.name != "experiment" or experiment.parent.parent.name != "episodes"
            or not re.fullmatch(r"[0-9a-f]{32}", experiment.parent.name)):
        raise ValueError("evidence projection accepts only a disposable episode experiment copy")
    check_policy(config["native_capability_policy"])
    adb = config["agent_debugger"]
    if not 1 <= adb.get("max_tasks", 0) <= MAX_ANALYSIS_TASKS or adb.get("retry_attempts") != 1:
        raise ValueError("restricted Debugger cap/retry policy changed")
    receipt = {"current_iteration": current_iteration, "detailed_tasks_max": MAX_ANALYSIS_TASKS,
               "selection": "unchanged original _build_adb_jobs, failure-first", "iterations": []}
    for run in sorted((experiment / "runs").glob("iteration_*")):
        if not re.fullmatch(r"iteration_\d+", run.name):
            continue
        current = int(run.name.split("_")[1]) == current_iteration
        selected_names = set()
        retained_folders = set()
        jobs = list((run / "input/benchmark").glob("20*/result.json"))
        # Historical repairs keep aggregate-only archives; detailed evidence
        # is selected only from the current iteration's single job.
        if current and len(jobs) > 1:
            raise ValueError("restricted projection requires one native job per iteration")
        for complete in jobs:
            job = complete.parent
            stats = module.compute_stats(job, k=config["harbor"].get("k", 1))
            selected = module._build_adb_jobs(stats["task_results"], job, adb,
                                             timeout_tasks=stats.get("timeout_tasks")) if current else []
            selected_names.update(item.task_name for item in selected)
            retained_folders.update(path.parent.parent for item in selected for path in item.trace_paths)
            for folder in job.glob("native_*"):
                if folder.is_dir() and folder not in retained_folders:
                    for child in list(folder.iterdir()):
                        if child.name in {"result.json", "native_record.json"}:
                            continue  # task metadata contains no messages (export_job contract)
                        if child.name == "verifier":
                            for verifier_file in list(child.iterdir()):
                                if verifier_file.name != "reward.txt":
                                    if verifier_file.is_dir():
                                        shutil.rmtree(verifier_file)
                                    else:
                                        verifier_file.unlink()
                        elif child.is_dir():
                            shutil.rmtree(child)
                        else:
                            child.unlink()
        details = list((run / "input/analysis/detail").glob("*.md"))
        if not current and details:
            shutil.rmtree(run / "input/analysis/detail")
            details = []
        if len(details) > MAX_ANALYSIS_TASKS or any(p.stem not in selected_names for p in details):
            raise ValueError("existing analysis exceeds the restricted selected-task evidence budget")
        # Never expose failed/before-policy analysis archives or Debugger CLI
        # diagnostics as an alternate path around the raw evidence limit.
        for folder in list(run.iterdir()):
            if folder.is_dir() and folder.name.startswith(("debugger", "evolve_")):
                shutil.rmtree(folder)
        for folder in list((run / "input").glob("analysis_*")):
            if folder.is_dir():
                shutil.rmtree(folder)
        # Retain previous decisions, not optimizer transcripts that repeat raw
        # episode text and invite expensive re-reading of the whole context.
        evolution = run / "evolve"
        if evolution.is_dir():
            for child in list(evolution.iterdir()):
                if child.name not in {"evolve_summary.md", "change_manifest.json"}:
                    if child.is_dir():
                        shutil.rmtree(child)
                    else:
                        child.unlink()
        receipt["iterations"].append({"iteration": run.name,
            "selected_tasks": sorted(selected_names), "detailed_tasks": len(selected_names)})
    return receipt
