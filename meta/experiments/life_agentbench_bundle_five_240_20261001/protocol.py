"""Frozen protocol and integrity checks for the five-round AgentBench bundle run."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
HISTORICAL_ROOT = HERE.parent / "meta_from_scratch_six_20260915"
SOURCE_ROOT = HERE.parent / "life_from_scratch_six_20260930"
DOMAINS = ("alfworld", "dbbench", "webshop")
ROUNDS = 5


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def object_digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def protocol() -> dict[str, Any]:
    return load_json(HERE / "protocol.json")


def task_ids(spec: dict[str, Any], batch: int, split: str) -> list[Any]:
    if split == "train":
        row = spec["batches"][batch]
        return list(row.get("task_ids", row.get("indices", [])))
    return list(spec.get("test_task_ids", spec.get("test_indices", [])))


def task_trial_keys(summary: dict[str, Any]) -> set[tuple[str, int]]:
    return {(str(task), 0) for task in summary.get("per_task", {})}


def validate_protocol(check_files: bool = True) -> dict[str, Any]:
    cfg = protocol()
    manifest_path = HISTORICAL_ROOT / "manifest.json"
    if digest(manifest_path) != cfg["historical_manifest_sha256"]:
        raise RuntimeError("Historical manifest changed")
    source = load_json(manifest_path)
    if tuple(cfg["domains"]) != DOMAINS or cfg["rounds_per_domain"] != ROUNDS:
        raise RuntimeError("Bundle protocol domain or round count changed")
    if cfg["schema_version"] != 2 or cfg["initial_active_snapshot"] != []:
        raise RuntimeError("Bundle protocol schema or initial snapshot changed")
    if object_digest(source["snapshot_hashes"]) != cfg["snapshot_hashes_sha256"]:
        raise RuntimeError("Task snapshot hash map changed")
    if object_digest(source["data_hashes"]) != cfg["data_hashes_sha256"]:
        raise RuntimeError("Benchmark data hash map changed")
    for domain in DOMAINS:
        spec = source["domains"][domain]
        if spec.get("family") != "agentbench" or len(spec["batches"]) != 6:
            raise RuntimeError(f"{domain} historical task protocol changed")
        if any(len(task_ids(spec, index, "train")) != 50 for index in range(6)):
            raise RuntimeError(f"{domain} frozen training batch changed")
    if check_files:
        # RESULTS.md is intentionally not touched here. Final reporting verifies and
        # reads it only after every training state has been sealed.
        for name, expected in cfg["historical_sources"].items():
            if digest(SOURCE_ROOT / name) != expected:
                raise RuntimeError(f"Historical source changed: {name}")
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
