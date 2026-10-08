"""Filesystem transport wrapper; optimization executes the original AHE function."""
import json
import os
import shutil
import subprocess
from pathlib import Path

import yaml

from .isolation import BASE, case_root, enter, require_regular_workspace, copy_unchanged_or_atomic
from .records import write_json
from .upstream import AHE
from .api_broker import isolated_connection
from .token_accounting import scoped_url


def own_tree(root, uid):
    for path in [root, *root.rglob("*")]:
        if not path.is_symlink():
            os.chown(path, uid, uid)


def sync_files(source, destination):
    source, destination = Path(source), Path(destination)
    require_regular_workspace(source)
    new_files = {p.relative_to(source) for p in source.rglob("*") if p.is_file()}
    old_files = {p.relative_to(destination) for p in destination.rglob("*") if p.is_file()}
    for path in old_files - new_files:
        (destination / path).unlink()
    shutil.copytree(source, destination, dirs_exist_ok=True)


def validate_completed_cache(cache, workspace):
    """Fail closed before promoting an audited original paid Evolve result."""
    from .backend import workspace_hash
    from .capabilities import sha
    experiment = Path(cache["private_experiment"])
    case = experiment.parent
    require_regular_workspace(experiment / "workspace")
    if (workspace_hash(workspace) != cache["input_workspace_sha256"] or
            workspace_hash(experiment / "workspace") != cache["candidate_workspace_sha256"] or
            any(sha(case / name) != expected for name, expected in cache["files"].items()) or
            not isinstance(json.loads((case / "evolve_result.json").read_text()).get("result"), str) or
            not json.loads((case / "evolve_result.json").read_text())["result"].strip()):
        raise RuntimeError("completed Evolve cache changed; refuse paid replay")
    return experiment


def promote_completed(exp_dir, iteration_dir, experiment, policy):
    if policy:
        from .capabilities import validate_workspace
        try:
            validation = validate_workspace(experiment / "workspace", policy)
        except Exception as exc:
            write_json(iteration_dir / "evolve/capability_rejection.json", {
                "profile": policy["profile"], "candidate_promoted": False,
                "error": str(exc), "private_candidate": str(experiment / "workspace")})
            raise
        write_json(iteration_dir / "evolve/capability_validation.json", validation)
    sync_files(experiment / "workspace", exp_dir / "workspace")
    shutil.copytree(experiment / "runs" / iteration_dir.name / "evolve",
                    iteration_dir / "evolve", dirs_exist_ok=True)
    if (experiment / "change_manifest.json").exists():
        shutil.copy2(experiment / "change_manifest.json", exp_dir / "change_manifest.json")
    return json.loads((experiment.parent / "evolve_result.json").read_text())["result"]


def bind_isolated_evolve(module):
    original = module.run_evolve_agent

    def run(config, exp_dir, iteration, query, job_dir, iteration_dir):
        policy = config.get("native_capability_policy")
        if policy:
            from .capabilities import validate_workspace, constraint_text
            validate_workspace(exp_dir / "workspace", policy)
            query += constraint_text(policy)
        cache = config.get("native_backend", {}).get("reuse_completed_evolve")
        if cache and cache["iteration"] == iteration:
            experiment = validate_completed_cache(cache, exp_dir / "workspace")
            result = promote_completed(exp_dir, iteration_dir, experiment, policy)
            write_json(iteration_dir / "evolve/completed_cache_receipt.json", {
                "iteration": iteration, "candidate_sha256": cache["candidate_workspace_sha256"],
                "optimizer_replayed": False, "original_private_experiment": str(experiment)})
            return result
        root, case, logical, uid = case_root(exp_dir / "workspace", iteration_dir / "evolve")
        copy_unchanged_or_atomic(AHE / "evolve.py", root / "upstream/evolve.py")
        experiment = case / "experiment"
        shutil.copytree(exp_dir, experiment, ignore=shutil.ignore_patterns(".native_pending", "__pycache__"))
        if policy:
            from .capability_evidence import project_public_evidence
            projection = project_public_evidence(experiment, config, module, iteration)
            write_json(iteration_dir / "evolve/evidence_projection.json", projection)
        own_tree(experiment, uid)
        # Runs remain read-only by instruction, as in original AHE; hidden grader
        # data and other experiments do not exist in this process's filesystem.
        logical_exp = logical + "/experiment"
        for path in (experiment / "runs").glob("*/input/analysis/**/*.md"):
            path.write_text(path.read_text().replace(str(exp_dir), logical_exp))
        ahe_home = case / "ahe/agents"
        ahe_home.mkdir(parents=True)
        (ahe_home / "evolve_agent").symlink_to(logical_exp + "/evolve_agent")
        broker_url = scoped_url(os.environ["AHE_LOCAL_BROKER_URL"],
            experiment=exp_dir.name, domain=config["native_backend"]["domain"],
            round=iteration, role="evolve")
        # Upstream resume materializes connection values in this YAML, so
        # process environment alone cannot restore the scoped accounting URL.
        agent_path = experiment / "evolve_agent/evolve_agent.yaml"
        agent_config = yaml.safe_load(agent_path.read_text())
        agent_config["llm_config"].update(api_key="EMPTY", base_url=broker_url)
        agent_path.write_text(yaml.safe_dump(agent_config, sort_keys=False))
        payload = {"config": isolated_connection(config, broker_url), "iteration": iteration, "query": query.replace(str(exp_dir), logical_exp),
            "experiment": logical_exp,
            "job_dir": logical_exp + "/" + str(job_dir.relative_to(exp_dir)),
            "iteration_dir": logical_exp + "/" + str(iteration_dir.relative_to(exp_dir))}
        write_json(case / "evolve_request.json", payload)
        env = {"AHE_EVOLVE_API_KEY": "EMPTY", "AHE_EVOLVE_BASE_URL": broker_url,
            "QA_API_KEY": "EMPTY", "QA_BASE_URL": broker_url, "QA_MODEL_NAME": "deepseek-v4-flash",
            "NO_PROXY": "localhost,127.0.0.1"}
        env.update(PYTHONPATH="/adapter:/upstream", PATH="/runtime/bin:/usr/bin:/bin",
            AHE_HOME=logical + "/ahe",
            HOME=logical + "/workspace/.home", TMPDIR="/tmp",
            GIT_AUTHOR_NAME="AHE native integration", GIT_AUTHOR_EMAIL="ahe-native@localhost",
            GIT_COMMITTER_NAME="AHE native integration", GIT_COMMITTER_EMAIL="ahe-native@localhost")
        log_path = iteration_dir / "evolve/isolated_process.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as log:
            proc = subprocess.run(["/runtime/bin/python", "-u", "-m", "ahe_native.evolve_worker",
                "--request", logical + "/evolve_request.json", "--out", logical + "/evolve_result.json"],
                preexec_fn=lambda: enter(str(root), logical_exp, uid), env=env,
                stdout=log, stderr=subprocess.STDOUT, timeout=7200)
        if proc.returncode or not (case / "evolve_result.json").exists():
            raise RuntimeError(f"original AHE Evolve Agent failed; inspect {log_path}")
        return promote_completed(exp_dir, iteration_dir, experiment, policy)

    run.__wrapped__ = original
    module.run_evolve_agent = run
