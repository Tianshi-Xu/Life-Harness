"""Frozen protocol and integrity checks for the six-round continuation AgentBench bundle run."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
HISTORICAL_ROOT = HERE.parent / "meta_from_scratch_six_20260915"
SOURCE_ROOT = HERE.parent / "life_from_scratch_six_20260930"
DOMAINS = ("dbbench", "webshop")
ROUNDS = 9


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
    if (cfg.get("inherited_rounds") != 6 or cfg.get("prior_test_exposure") is not True
            or cfg.get("inherited_run") != "../life_agentbench_six_resume_mr_20261001"
            or cfg.get("inherited_training_seal_sha256") != "e5fb4fc69ffb641e07a9f6e51de3898035179b896bee3ce058a8b027a3a43ab0"):
        raise RuntimeError("continuation inheritance protocol changed")
    source = load_json(manifest_path)
    if tuple(cfg["domains"]) != DOMAINS or cfg["rounds_per_domain"] != ROUNDS:
        raise RuntimeError("Bundle protocol domain or round count changed")
    if cfg["schema_version"] != 2 or cfg["initial_active_snapshot"] != []:
        raise RuntimeError("Bundle protocol schema or initial snapshot changed")
    if (cfg.get("proposer") != "ModelRouter deepseek-v4-flash"
            or cfg.get("proposer_model") != "deepseek-v4-flash"
            or cfg.get("proposer_protocol") != "openai_chat_completions_tool_calling"
            or cfg.get("proposer_host") != "https://routify-pub.alibaba-inc.com"
            or cfg.get("proposer_env") != "/mnt/data/xts/skill/test_ap/env.sh"):
        raise RuntimeError("ModelRouter proposer protocol changed")
    if cfg.get("operation_validation") != (
            "reject invalid operations individually before simultaneously staging the remaining bundle operations"):
        raise RuntimeError("operation filtering protocol changed")
    if cfg.get("agentbench_hint_policy") != (
            "one optional string per H4/H5 mechanism; no word-count limit"):
        raise RuntimeError("AgentBench hint policy changed")
    if cfg.get("episode_error_policy") != (
            "an infrastructure-error episode is cached once as success=0 and included in the domain summary without retrying or invalidating sibling episodes"):
        raise RuntimeError("episode error policy changed")
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
