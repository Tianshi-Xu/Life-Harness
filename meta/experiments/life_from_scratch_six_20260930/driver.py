"""Resume-safe six-domain Life-Harness experiment driver.

This module defines the approved run. Importing it never starts a model or benchmark.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import py_compile
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from candidate_contract import (load_candidate, prompt, public_history,
                                validate_proposal, validate_source)
from protocol import (AGENTBENCH_DOMAINS, DOMAINS, HERE, ROOT, TAU_DOMAINS,
                      atomic_json, bind_fingerprint, digest, task_ids,
                      task_trial_keys, validate_protocol)

ZERO_LAYER = ROOT / "meta" / "zero_layer.py"


def score(summary: dict[str, Any]) -> float:
    return float(summary["accuracy"])


def accept(candidate: dict[str, Any], incumbent: dict[str, Any]) -> bool:
    if task_trial_keys(candidate) != task_trial_keys(incumbent):
        raise RuntimeError("Candidate and incumbent task/trial keys differ")
    return score(candidate) > score(incumbent)


def next_batch(batch: int, accepted: bool) -> int:
    return batch if accepted else min(batch + 1, 5)


def _files_hash(paths: list[Path]) -> dict[str, str]:
    return {str(path.resolve()): digest(path) for path in paths}


def write_chain(directory: Path, layers: list[Path], family: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    atomic_json(manifest_path, {"family": family,
                                "layers": [str(path.resolve()) for path in layers],
                                "hashes": _files_hash(layers)})
    if family == "agentbench":
        return manifest_path
    loader = directory / "loader.py"
    source = '''"""Generated ordered register() chain."""
import importlib.util
import json
from pathlib import Path


def register():
    manifest = json.loads(Path(__file__).with_name("manifest.json").read_text())
    for index, item in enumerate(manifest["layers"]):
        spec = importlib.util.spec_from_file_location(f"life_tau_layer_{index}", item)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        hook = getattr(module, "register", None)
        if callable(hook):
            hook()
'''
    temporary = loader.with_suffix(".py.tmp")
    temporary.write_text(source)
    temporary.replace(loader)
    return loader


def _accepted_paths(run_dir: Path, state: dict[str, Any]) -> list[Path]:
    return [run_dir / "chain" / item["file"] for item in state["accepted"]]


def _initial_state(domain: str, family: str, run_dir: Path) -> dict[str, Any]:
    accepted = []
    if family == "tau":
        zero = run_dir / "chain" / "layer_000_zero.py"
        zero.parent.mkdir(parents=True, exist_ok=True)
        if not zero.exists():
            shutil.copy2(ZERO_LAYER, zero)
        elif digest(zero) != digest(ZERO_LAYER):
            raise RuntimeError("Tau zero layer changed")
        accepted.append({"round": 0, "name": "zero", "target_layer": "zero",
                         "file": zero.name, "sha256": digest(zero)})
    return {"status": "running", "domain": domain, "family": family,
            "round": 0, "batch": 0, "accepted": accepted, "rounds": [],
            "baseline_phase": None, "baseline_summary": None}


def _phase_summary_path(domain: str, split: str, phase: str) -> Path:
    if domain in TAU_DOMAINS:
        name = "val.json" if split == "train" else "test.json"
    else:
        name = "summary.json"
    return HERE / domain / "evals" / phase / name


def phase(domain: str, split: str, phase_name: str, chain: Path, batch: int) -> dict[str, Any]:
    if domain in TAU_DOMAINS:
        command = [sys.executable, "-u", str(HERE / "tau_evaluate.py"), domain,
                   split, phase_name, str(chain), "--batch", str(batch)]
    else:
        python = {
            "alfworld": ROOT / "AgentBench/.native/alfworld-venv/bin/python",
            "dbbench": ROOT / "AgentBench/.native/core/bin/python",
            "webshop": ROOT / "AgentBench/.native/webshop-venv/bin/python",
        }[domain]
        command = [str(python), "-u", str(HERE / "agentbench_evaluate.py"), domain,
                   split, phase_name, str(chain), "--batch", str(batch)]
    env = dict(os.environ)
    env["PATH"] = "/mnt/data/xts/tools/uv/bin:" + env.get("PATH", "")
    if domain == "webshop":
        java = ROOT / "AgentBench/.native/sysroot/usr/lib/jvm/java-11-openjdk-amd64"
        env.update(JAVA_HOME=str(java), PATH=str(java / "bin") + ":" + env["PATH"],
                   _JAVA_OPTIONS="-Xmx4g",
                   PYTHONPATH=str(ROOT / "AgentBench/.native/webshop-src") + ":" +
                              str(ROOT / "AgentBench"))
    log_path = HERE / domain / f"{phase_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                stderr=subprocess.STDOUT, timeout=21600)
    if result.returncode:
        raise RuntimeError(f"evaluation failed; see {log_path}")
    return json.loads(_phase_summary_path(domain, split, phase_name).read_text())


def _export_evidence(domain: str, phase_name: str, workspace: Path,
                     summary: dict[str, Any]) -> None:
    destination = workspace / "evidence" / phase_name
    destination.mkdir(parents=True, exist_ok=True)
    safe_summary = {key: value for key, value in summary.items()
                    if key not in {"save_dir", "cmd"} and not key.startswith("test_")}
    atomic_json(destination / "summary.json", safe_summary)
    source = HERE / domain / "evals" / phase_name
    if domain in AGENTBENCH_DOMAINS:
        for episode in source.glob("episode_*.json"):
            row = json.loads(episode.read_text())
            atomic_json(destination / episode.name, {
                key: row.get(key) for key in
                ("index", "success", "status", "turns", "history", "calls", "rewards")
            })
    else:
        results_path = Path(summary["save_dir"]) / "results.json"
        rows = []
        for row in json.loads(results_path.read_text()).get("simulations", []):
            rows.append({key: row.get(key) for key in
                         ("task_id", "trial", "reward_info", "termination_reason",
                          "messages", "agent_system_prompt", "agent_tools_schema")})
        atomic_json(destination / "public_trajectories.json", rows)


def _workspace_hashes(workspace: Path) -> dict[str, str]:
    excluded = {workspace / "candidate.py", workspace / "proposal.json"}
    return {str(path.relative_to(workspace)): digest(path)
            for path in sorted(workspace.rglob("*"))
            if path.is_file() and path not in excluded and "__pycache__" not in path.parts}


def _copy_frozen_source(source: Path, destination: Path) -> None:
    if destination.exists():
        if digest(destination) != digest(source):
            raise RuntimeError(f"Proposer source snapshot changed: {destination}")
        return
    shutil.copy2(source, destination)


def _prepare_workspace(domain: str, round_number: int, run_dir: Path,
                       state: dict[str, Any]) -> Path:
    workspace = run_dir / "proposer" / f"round_{round_number:02d}"
    workspace.mkdir(parents=True, exist_ok=True)
    accepted = workspace / "accepted"
    accepted.mkdir(exist_ok=True)
    for source in _accepted_paths(run_dir, state):
        target = accepted / source.name
        if not target.exists():
            shutil.copy2(source, target)
        elif digest(target) != digest(source):
            raise RuntimeError("Proposer accepted-layer snapshot changed")
    atomic_json(workspace / "history.json", public_history(state["rounds"]))
    source_view = workspace / "source_view"
    source_view.mkdir(exist_ok=True)
    contract = source_view / "contract.txt"
    contract.write_text(
        "Implement one incremental Life-Harness layer. TARGET_LAYER must match the "
        "single H2/H3/H4/H5 hook implemented. Accepted layers run first in filename "
        "order. H2 repairs an assistant message; H3 changes tool descriptions only; "
        "H4/H5 return zero or one short public hint. Runtime I/O, hidden data, task IDs, "
        "and per-instance solution tables are forbidden. Use only public names from this "
        "source view; vars(), globals(), locals(), and dunder attributes are forbidden. "
        "Allowed standard-library imports: ast, collections, copy, dataclasses, datetime, "
        "difflib, functools, itertools, json, math, re, statistics, string, typing.\n"
    )
    if state["family"] == "tau":
        sources = {
            "SKILL.md": ROOT / "meta/skills/tau2-harness/SKILL.md",
            "base.py": ROOT / "TauBench/src/tau2/harness/base.py",
            f"{domain}.py": ROOT / f"TauBench/src/tau2/harness/{domain}.py",
            "h3_tools.py": ROOT / "TauBench/src/tau2/harness/h3_tools.py",
            "skills.py": ROOT / "TauBench/src/tau2/harness/skills.py",
        }
    else:
        snapshot = ROOT / "meta/experiments/meta_from_scratch_six_20260915/task_snapshot/src/server/harness"
        sources = {
            "session.py": snapshot / "session.py",
            f"{domain}.py": snapshot / f"{domain}.py",
        }
    for name, source in sources.items():
        _copy_frozen_source(source, source_view / name)
    return workspace


def propose(domain: str, family: str, round_number: int, workspace: Path,
            run_dir: Path) -> tuple[Path, dict[str, Any]]:
    saved_candidate = run_dir / "candidates" / f"candidate_{round_number:02d}.py"
    saved_proposal = run_dir / "proposals" / f"proposal_{round_number:02d}.json"
    if saved_candidate.exists() and saved_proposal.exists():
        return saved_candidate, validate_proposal(json.loads(saved_proposal.read_text()), domain)
    candidate = workspace / "candidate.py"
    proposal_path = workspace / "proposal.json"
    frozen = _workspace_hashes(workspace)
    atomic_json(run_dir / "proposals" / f"inputs_{round_number:02d}.json", frozen)
    text = prompt(domain, family, round_number, workspace)
    prompt_path = run_dir / "proposals" / f"prompt_{round_number:02d}.txt"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(text)
    tools = ["Read", "Write", "Edit", "Glob", "Grep"]
    command = ["qoder", "-p", text, "-w", str(workspace), "-o", "json",
               "--permission-mode", "accept_edits", "--tools", *tools,
               "--disallowed-tools", "Bash", "Agent", "Task", "-m", "DeepSeek-Flash"]
    for tool in tools:
        command += ["--allowed-tools", tool]
    for attempt in range(1, 4):
        candidate.unlink(missing_ok=True)
        proposal_path.unlink(missing_ok=True)
        result = subprocess.run(command, cwd=workspace, capture_output=True, text=True,
                                timeout=5400)
        record = {"attempt": attempt, "exit_code": result.returncode,
                  "stdout": result.stdout, "stderr": result.stderr}
        atomic_json(run_dir / "proposals" /
                    f"qoder_{round_number:02d}_attempt_{attempt:02d}.json", record)
        atomic_json(run_dir / "proposals" / f"qoder_{round_number:02d}.json", record)
        if _workspace_hashes(workspace) != frozen:
            raise RuntimeError("Proposer modified a frozen input")
        if result.returncode == 0 and candidate.exists() and proposal_path.exists():
            break
    else:
        raise RuntimeError("Proposer failed to produce candidate.py and proposal.json after 3 attempts")
    proposal = validate_proposal(json.loads(proposal_path.read_text()), domain)
    saved_candidate.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidate, saved_candidate)
    atomic_json(saved_proposal, proposal)
    return saved_candidate, proposal


def preflight(candidate: Path, proposal: dict[str, Any], family: str,
              staged_chain: Path) -> dict[str, Any]:
    tree = validate_source(candidate, family)
    py_compile.compile(str(candidate), doraise=True)
    if family == "agentbench":
        module = load_candidate(candidate, family)
        if module.TARGET_LAYER != proposal["target_layer"]:
            raise ValueError("proposal target does not match candidate TARGET_LAYER")
        # Loading every staged layer catches composition/import problems.
        from agentbench_chain import CompositeHarness, layer_paths
        CompositeHarness(layer_paths(staged_chain))
    else:
        # Tau imports require the TauBench environment; host-side validation stays
        # structural, then the staged loader is imported and registered below.
        text = candidate.read_text()
        double_quoted = f'TARGET_LAYER = "{proposal["target_layer"]}"'
        single_quoted = f"TARGET_LAYER = '{proposal['target_layer']}'"
        if "def register" not in text or not (double_quoted in text or single_quoted in text):
            raise ValueError("Tau candidate must declare matching TARGET_LAYER and register()")
        code = ("import importlib.util;"
                f"s=importlib.util.spec_from_file_location('staged',r'{staged_chain}');"
                "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.register()")
        env = dict(os.environ)
        env["UV_CACHE_DIR"] = "/tmp/uv-cache"
        result = subprocess.run(["uv", "run", "python", "-c", code],
                                cwd=ROOT / "TauBench", env=env,
                                capture_output=True, text=True, timeout=180)
        if result.returncode:
            raise ValueError("Tau staged register() validation failed: " + result.stderr[-500:])
    return {"candidate_sha256": digest(candidate), "chain_sha256": digest(staged_chain),
            "validation": "pass"}


def _assert_expected_keys(summary: dict[str, Any], expected_ids: list[Any], trials: int) -> None:
    expected = {(str(task), trial) for task in expected_ids for trial in range(trials)}
    if task_trial_keys(summary) != expected:
        raise RuntimeError("Evaluation task/trial keys do not match frozen protocol")


def run_domain(domain: str, evaluate: Callable[..., dict[str, Any]] = phase,
               propose_candidate: Callable[..., tuple[Path, dict[str, Any]]] = propose,
               do_preflight: Callable[..., dict[str, Any]] = preflight) -> dict[str, Any]:
    manifest = validate_protocol()
    spec = manifest["domains"][domain]
    family = spec["family"]
    run_dir = HERE / domain
    run_dir.mkdir(parents=True, exist_ok=True)
    bind_fingerprint(run_dir / "run_config.json", {
        "protocol_sha256": digest(HERE / "protocol.json"), "domain": domain,
        "family": family, "rounds": 6, "proposer": "DeepSeek-Flash"})
    state_path = run_dir / "state.json"
    state = (json.loads(state_path.read_text()) if state_path.exists()
             else _initial_state(domain, family, run_dir))
    if state["status"] == "complete":
        return state
    for round_number in range(state["round"] + 1, 7):
        batch = state["batch"]
        current_layers = _accepted_paths(run_dir, state)
        incumbent_chain = write_chain(run_dir / "runtime" / "incumbent", current_layers, family)
        if state["baseline_summary"] is None:
            baseline_phase = f"batch_{batch:02d}_incumbent"
            baseline = evaluate(domain, "train", baseline_phase, incumbent_chain, batch)
            _assert_expected_keys(baseline, task_ids(spec, batch, "train"),
                                  spec.get("train_trials", 1))
            state.update(baseline_phase=baseline_phase, baseline_summary=baseline)
            atomic_json(state_path, state)
        else:
            baseline = state["baseline_summary"]
        workspace = _prepare_workspace(domain, round_number, run_dir, state)
        _export_evidence(domain, state["baseline_phase"], workspace, baseline)
        candidate, proposal = propose_candidate(domain, family, round_number, workspace, run_dir)
        staged_layers = current_layers + [candidate]
        staged_chain = write_chain(run_dir / "runtime" / f"staged_{round_number:02d}", staged_layers, family)
        check = do_preflight(candidate, proposal, family, staged_chain)
        atomic_json(run_dir / "proposals" / f"preflight_{round_number:02d}.json", check)
        candidate_phase = f"candidate_{round_number:02d}"
        candidate_summary = evaluate(domain, "train", candidate_phase, staged_chain, batch)
        _assert_expected_keys(candidate_summary, task_ids(spec, batch, "train"),
                              spec.get("train_trials", 1))
        accepted = accept(candidate_summary, baseline)
        if accepted:
            layer_name = f"layer_{round_number:03d}_{proposal['name']}.py"
            layer_path = run_dir / "chain" / layer_name
            layer_path.parent.mkdir(parents=True, exist_ok=True)
            if layer_path.exists() and digest(layer_path) != digest(candidate):
                raise RuntimeError("Accepted layer path contains different code")
            if not layer_path.exists():
                shutil.copy2(candidate, layer_path)
            state["accepted"].append({"round": round_number, "name": proposal["name"],
                                      "target_layer": proposal["target_layer"],
                                      "hypothesis": proposal["hypothesis"], "file": layer_name,
                                      "sha256": digest(layer_path)})
            state.update(baseline_phase=candidate_phase,
                         baseline_summary=candidate_summary)
        else:
            state.update(batch=next_batch(batch, False) if round_number < 6 else batch,
                         baseline_phase=None, baseline_summary=None)
        retained_chain = write_chain(run_dir / "runtime" / "incumbent",
                                     _accepted_paths(run_dir, state), family)
        test_phase = f"test_round_{round_number:02d}"
        test = evaluate(domain, "test", test_phase, retained_chain, batch)
        _assert_expected_keys(test, task_ids(spec, batch, "test"), 1)
        row = {"round": round_number, "train_batch": batch,
               "incumbent_score": score(baseline), "candidate_score": score(candidate_summary),
               "accepted": accepted, "next_batch": state["batch"],
               "batch_rotated": not accepted and state["batch"] != batch,
               "proposal": proposal, "retained_layers": len(state["accepted"]),
               "test_phase": test_phase, "test_score": score(test),
               "test_n": len(test["per_task"])}
        state["rounds"].append(row)
        state["round"] = round_number
        atomic_json(state_path, state)
        print(json.dumps({key: row[key] for key in
                          ("round", "train_batch", "accepted", "next_batch")}), flush=True)
    state["status"] = "complete"
    atomic_json(state_path, state)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domain", choices=DOMAINS)
    args = parser.parse_args()
    run_dir = HERE / args.domain
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / ".lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit(f"{args.domain} already has a running driver") from exc
        lock.write(str(os.getpid()) + "\n")
        lock.flush()
        run_domain(args.domain)


if __name__ == "__main__":
    main()
