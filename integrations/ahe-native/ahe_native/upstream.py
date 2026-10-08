"""Load unmodified AHE and bind only its evaluation transport boundary."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
AHE = HERE.parent / "agentic-harness-engineering"
PIN = "8b2a55d97590363fe50c3cc6b5e833b020a4bb4c"


def source_fingerprint(root=AHE):
    root = Path(root)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                     text=True).strip()
    if commit != PIN:
        raise RuntimeError("upstream AHE commit differs from the audited pin")
    dirty = subprocess.check_output(["git", "diff", "HEAD", "--"], cwd=root)
    if dirty:
        raise RuntimeError("tracked upstream source changes are not allowed")
    paths = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).split(b"\0")
    return {"commit": commit, "tracked_sha256": {
        os.fsdecode(p): hashlib.sha256((root / os.fsdecode(p)).read_bytes()).hexdigest()
        for p in paths if p}}


def load_upstream(root=AHE):
    receipt = source_fingerprint(root)
    root = Path(root)
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("ahe_upstream_evolve", root / "evolve.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, receipt


def bind_native_transport(module):
    """Bind launch/wait transport; all statistics/search/selection stay upstream.

    The configured executable must load the actual evolving NexAU workspace,
    execute the native environment, and publish a Harbor-shaped job. There is
    deliberately NO fallback to a Life/Meta plugin or a fake benchmark.
    """
    original = module.launch_harbor
    original_wait = module.wait_for_harbor

    def launch(config, workspace_dir, agent_config_filename, iteration_dir,
               label="", n_concurrent_override=None):
        cfg = config.get("native_backend")
        if cfg is None:
            return original(config, workspace_dir, agent_config_filename,
                            iteration_dir, label, n_concurrent_override)
        if cfg.get("split") != "train":
            raise ValueError("optimizer transport accepts train evidence only")
        if label and not re.fullmatch(r"[a-zA-Z0-9_-]+", label):
            raise ValueError("unsupported evaluation label")
        command = cfg.get("command")
        if not isinstance(command, list) or not command or not all(
                isinstance(x, str) for x in command):
            raise ValueError("native_backend.command must be an argv list")
        previous = module.find_latest_job_dir(iteration_dir)
        request = {"schema_version": 1, "domain": cfg["domain"], "split": "train",
                   "workspace": str(Path(workspace_dir).resolve()),
                   "agent_config": agent_config_filename,
                   "output_root": str(Path(iteration_dir).resolve()),
                   "k": int(config["harbor"].get("k", 1)),
                   "concurrency": n_concurrent_override or config["harbor"]["n_concurrent"],
                   "task_ids": cfg["task_ids"]}
        if cfg.get("capability_policy"):
            request["capability_policy"] = cfg["capability_policy"]
        if cfg.get("training_schedule_file"):
            from .training import scheduled_batch
            match = re.search(r"(?:^|/)iteration_(\d+)(?:/|$)", str(iteration_dir))
            if not match:
                raise ValueError("rotating training requires an original iteration directory")
            iteration = int(match.group(1))
            request.update(training_schedule_file=cfg["training_schedule_file"],
                training_schedule_sha256=cfg["training_schedule_sha256"], iteration=iteration)
            request["task_ids"] = scheduled_batch(request)
        if request.get("iteration", 1) == 1 and cfg.get("reuse_completed_episodes"):
            request["reuse_completed_episodes"] = cfg["reuse_completed_episodes"]
        # request must never contain API keys; connection settings are process-only.
        request_file = Path(iteration_dir) / ("native_request" + ("_" + label if label else "") + ".json")
        from .records import write_json
        write_json(request_file, request)
        env = os.environ.copy()
        llm = module.get_llm_config(config, role="agent")
        for key, env_key in (("api_key", "LLM_API_KEY"), ("base_url", "LLM_BASE_URL"),
                             ("model", "LLM_MODEL")):
            if llm.get(key):
                env[env_key] = llm[key]
        proc = subprocess.Popen([*command, "--request", str(request_file)],
                                cwd=HERE, env=env)
        proc.ahe_native_backend = True
        return proc, previous.name if previous else ""

    def wait(proc, iteration_dir, started_after, timeout_minutes=0, label=""):
        if getattr(proc, "ahe_native_backend", False):
            try:
                returncode = proc.wait(timeout=timeout_minutes * 60 if timeout_minutes else None)
            except subprocess.TimeoutExpired:
                proc.terminate()
                raise module.HarborJobTimeoutError("native evaluation transport timed out")
            if returncode:
                raise RuntimeError(f"native evaluation transport exited with {returncode}; do not optimize from incomplete results")
        return original_wait(proc, iteration_dir, started_after, timeout_minutes, label)

    module.launch_harbor = launch
    module.wait_for_harbor = wait
    return {"replaced_functions": ["launch_harbor", "wait_for_harbor"], "core_source_modified": False}
