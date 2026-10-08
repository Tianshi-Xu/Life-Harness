"""Run the original Debugger batch inside a public-evidence-only filesystem."""
import json
import os
import shutil
import subprocess

import yaml

from .isolation import case_root, enter, copy_unchanged_or_atomic
from .evolve_isolation import own_tree
from .records import write_json
from .upstream import AHE
from .api_broker import isolated_connection
from .token_accounting import scoped_url


def evidence_config(config, exp_dir, k):
    """Upstream passes only agent_debugger, not the full experiment config."""
    if "agent_debugger" in config:
        full = dict(config)
    else:
        snapshot = exp_dir / "config_snapshot.yaml"
        full = yaml.safe_load(snapshot.read_text()) if snapshot.is_file() else {}
        if full.get("native_capability_policy"):
            full["agent_debugger"] = config
    if full.get("native_capability_policy"):
        full["harbor"] = {**full.get("harbor", {}), "k": k}
        return full
    return None


def bind_isolated_debugger(module):
    original = module.run_parallel_adb_ask

    def run(config, job_dir, task_results, iteration_dir, iteration,
            timeout_tasks=None, k=1):
        exp_dir = iteration_dir.parent.parent
        projection_config = evidence_config(config, exp_dir, k)
        root, case, logical, uid = case_root(exp_dir / "workspace", iteration_dir / "debugger")
        copy_unchanged_or_atomic(AHE / "evolve.py", root / "upstream/evolve.py")
        experiment = case / "experiment"
        shutil.copytree(exp_dir, experiment, ignore=shutil.ignore_patterns(".native_pending", "__pycache__"))
        if projection_config:
            from .capability_evidence import project_public_evidence
            projection = project_public_evidence(experiment, projection_config, module, iteration)
            write_json(iteration_dir / "debugger/evidence_projection.json", projection)
        own_tree(experiment, uid)
        logical_exp = logical + "/experiment"
        ahe_home = case / "ahe/agents"
        ahe_home.mkdir(parents=True)
        (ahe_home / "evolve_agent").symlink_to(logical_exp + "/evolve_agent")
        scope_config = projection_config
        if scope_config is None:
            scope_config = yaml.safe_load((exp_dir / "config_snapshot.yaml").read_text())
        broker_url = scoped_url(os.environ["AHE_LOCAL_BROKER_URL"],
            experiment=exp_dir.name, domain=scope_config["native_backend"]["domain"],
            round=iteration, role="debugger")
        payload = {"mode": "debugger", "config": isolated_connection(config, broker_url, debugger=True), "iteration": iteration,
            "task_results": task_results, "timeout_tasks": sorted(timeout_tasks or []), "k": k,
            "job_dir": logical_exp + "/" + str(job_dir.relative_to(exp_dir)),
            "iteration_dir": logical_exp + "/" + str(iteration_dir.relative_to(exp_dir))}
        write_json(case / "debugger_request.json", payload)
        env = {"QA_API_KEY": "EMPTY", "QA_BASE_URL": broker_url,
               "QA_MODEL_NAME": "deepseek-v4-flash", "NO_PROXY": "localhost,127.0.0.1"}
        env.update(PYTHONPATH="/adapter:/upstream", PATH="/runtime/bin:/usr/bin:/bin",
            AHE_HOME=logical + "/ahe", HOME=logical + "/workspace/.home", TMPDIR="/tmp")
        log_path = iteration_dir / "debugger/isolated_process.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as log:
            proc = subprocess.run(["/runtime/bin/python", "-u", "-m", "ahe_native.evolve_worker",
                "--request", logical + "/debugger_request.json", "--out", logical + "/debugger_result.json"],
                preexec_fn=lambda: enter(str(root), logical_exp, uid), env=env,
                stdout=log, stderr=subprocess.STDOUT, timeout=7200)
        result_path = case / "debugger_result.json"
        if proc.returncode or not result_path.exists():
            raise RuntimeError(f"original isolated Debugger failed; inspect {log_path}")
        source = experiment / "runs" / iteration_dir.name / "input/analysis"
        if source.exists():
            shutil.copytree(source, iteration_dir / "input/analysis", dirs_exist_ok=True)
            for path in (iteration_dir / "input/analysis").rglob("*.md"):
                path.write_text(path.read_text().replace(logical_exp, str(exp_dir)))
        diagnostics = experiment / "runs" / iteration_dir.name / "debugger/subprocess_diagnostics"
        if diagnostics.exists():
            shutil.copytree(diagnostics, iteration_dir / "debugger/subprocess_diagnostics", dirs_exist_ok=True)
        result = json.loads(result_path.read_text())["result"]
        return result.replace(logical_exp, str(exp_dir)) if result else result

    run.__wrapped__ = original
    module.run_parallel_adb_ask = run
