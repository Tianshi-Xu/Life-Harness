"""Read-only nine-round seed verification and train-only evidence lookup."""
from __future__ import annotations

import copy
import json
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

from protocol import HERE, atomic_json, digest, protocol

BASE_ROUNDS = 9


def base_root() -> Path:
    return (HERE / protocol()["inherited_run"]).resolve()


def read(path):
    return json.loads(Path(path).read_text())


@lru_cache(maxsize=1)
def _audit_predecessor():
    root = base_root()
    subprocess.run([sys.executable, str(root / "runner.py"), "verify"],
                   cwd=root, capture_output=True, text=True, check=True, timeout=300)
    return True


def verified_seed(domain: str) -> dict:
    root = base_root()
    seal_path = root / "training_seal.json"
    if digest(seal_path) != protocol()["inherited_training_seal_sha256"]:
        raise RuntimeError("inherited training seal changed")
    seal = read(seal_path)
    if digest(root / "protocol.json") != seal["protocol_sha256"]:
        raise RuntimeError("inherited protocol changed")
    for name in seal["state_sha256"]:
        if digest(root / name / "state.json") != seal["state_sha256"][name]:
            raise RuntimeError("inherited state changed")
        config = root / name / "run_config.json"
        if digest(config) != seal["run_config_sha256"][name]:
            raise RuntimeError("inherited run configuration changed")
        for filename, expected in read(config)["source_sha256"].items():
            if digest(root / filename) != expected:
                raise RuntimeError("inherited evaluated source changed")
    old_protocol = read(root / "protocol.json")
    for key in ("interface_version", "interface_source_sha256", "base_session_sha256"):
        if old_protocol[key] != protocol()[key]:
            raise RuntimeError("cannot reuse seed with a different interface")
    for name, expected in old_protocol["interface_source_sha256"].items():
        if digest(root / "interface_v3" / name) != expected:
            raise RuntimeError("inherited interface changed")
    _audit_predecessor()
    state = read(root / domain / "state.json")
    if (state["status"] != "complete" or state["round"] != BASE_ROUNDS
            or len(state["rounds"]) != BASE_ROUNDS):
        raise RuntimeError("predecessor did not finish nine rounds")
    return state


def initialize(domain: str, run_dir: Path) -> dict:
    state = copy.deepcopy(verified_seed(domain))
    origin = base_root() / domain
    for entry in state["operation_log"]:
        for op in entry["operations"]:
            if op["op"] == "delete":
                continue
            relative = Path("versions") / entry["bundle_id"] / f"{op['operation_id']}.py"
            source, target = origin / relative, run_dir / relative
            if digest(source) != op["artifact_sha256"]:
                raise RuntimeError("inherited accepted artifact changed")
            if target.exists():
                if digest(target) != digest(source):
                    raise RuntimeError("copied inherited artifact changed")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
    certificate = {"source_run": str(base_root()),
                   "training_seal_sha256": protocol()["inherited_training_seal_sha256"],
                   "source_state_sha256": digest(origin / "state.json"),
                   "inherited_rounds": BASE_ROUNDS, "prior_test_exposure": True,
                   "new_proposer": "ModelRouter deepseek-v4-flash"}
    path = run_dir / "inheritance.json"
    if path.exists() and read(path) != certificate:
        raise RuntimeError("inheritance certificate changed")
    if not path.exists():
        atomic_json(path, certificate)
    state["status"] = "running"
    state["interface_version"] = "life-agentbench-h2345-v3"
    return state


def phase_summary_path(domain: str, phase_name: str) -> Path:
    local = HERE / domain / "evals" / phase_name / "summary.json"
    if local.exists():
        return local
    seed = verified_seed(domain)
    allowed = {f"candidate_{number:02d}" for number in range(1, BASE_ROUNDS + 1)}
    allowed.update(f"batch_{row['train_batch']:02d}_incumbent" for row in seed["rounds"])
    if phase_name in allowed:
        roots = (base_root(), base_root().parent / "life_agentbench_six_resume_mr_20261001",
                 base_root().parent / "life_agentbench_bundle_five_20261001")
        for root in roots:
            path = root / domain / "evals" / phase_name / "summary.json"
            if path.exists():
                return path
    return local
