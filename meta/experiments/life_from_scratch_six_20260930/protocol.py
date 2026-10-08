"""Frozen protocol loader and integrity checks for the six-domain Life run."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
HISTORICAL_ROOT = HERE.parent / "meta_from_scratch_six_20260915"
TAU_DOMAINS = ("airline", "retail", "telecom")
AGENTBENCH_DOMAINS = ("alfworld", "dbbench", "webshop")
DOMAINS = TAU_DOMAINS + AGENTBENCH_DOMAINS


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def object_digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")
    temporary.replace(path)


def protocol() -> dict[str, Any]:
    return load_json(HERE / "protocol.json")


def manifest() -> dict[str, Any]:
    return load_json(HISTORICAL_ROOT / "manifest.json")


def task_ids(spec: dict[str, Any], batch: int, split: str) -> list[Any]:
    if split == "train":
        row = spec["batches"][batch]
        return list(row.get("task_ids", row.get("indices", [])))
    return list(spec.get("test_task_ids", spec.get("test_indices", [])))


def task_trial_keys(summary: dict[str, Any]) -> set[tuple[str, int]]:
    trials = summary.get("per_task_trials")
    if trials is not None:
        return {(str(task), trial) for task, values in trials.items()
                for trial in range(len(values))}
    return {(str(task), 0) for task in summary.get("per_task", {})}


def validate_protocol(check_files: bool = True) -> dict[str, Any]:
    cfg = protocol()
    manifest_path = HISTORICAL_ROOT / "manifest.json"
    if digest(manifest_path) != cfg["historical_manifest_sha256"]:
        raise RuntimeError("Historical manifest changed")
    source = load_json(manifest_path)
    if tuple(source["domains"]) != DOMAINS or tuple(cfg["domains"]) != DOMAINS:
        raise RuntimeError("Six-domain order changed")
    if source["iterations"] != cfg["rounds_per_domain"] or source["concurrency"] != cfg["concurrency"]:
        raise RuntimeError("Historical round or concurrency protocol changed")
    if object_digest(source["snapshot_hashes"]) != cfg["snapshot_hashes_sha256"]:
        raise RuntimeError("Task snapshot hash map changed")
    if object_digest(source["data_hashes"]) != cfg["data_hashes_sha256"]:
        raise RuntimeError("Benchmark data hash map changed")
    for domain in DOMAINS:
        spec = source["domains"][domain]
        if len(spec["batches"]) != 6:
            raise RuntimeError(f"{domain} no longer has six frozen batches")
        expected = 30 if domain == "airline" else 50
        if any(len(task_ids(spec, i, "train")) != expected for i in range(6)):
            raise RuntimeError(f"{domain} frozen batch size changed")
    if source["domains"]["airline"].get("train_trials") != 2:
        raise RuntimeError("Airline must remain 30 tasks x 2 trials")
    if check_files:
        for name, expected in cfg["historical_evaluators"].items():
            if digest(HISTORICAL_ROOT / name) != expected:
                raise RuntimeError(f"Historical evaluator changed: {name}")
        snapshot = HISTORICAL_ROOT / "task_snapshot"
        agentbench = ROOT / "AgentBench"
        for relative, expected in source["snapshot_hashes"].items():
            if digest(snapshot / relative) != expected:
                raise RuntimeError(f"Frozen task snapshot changed: {relative}")
        for relative, expected in source["data_hashes"].items():
            if digest(agentbench / relative) != expected:
                raise RuntimeError(f"Frozen benchmark data changed: {relative}")
    return source


def bind_fingerprint(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        if load_json(path) != value:
            raise RuntimeError(f"Resume fingerprint mismatch: {path}")
    else:
        atomic_json(path, value)
