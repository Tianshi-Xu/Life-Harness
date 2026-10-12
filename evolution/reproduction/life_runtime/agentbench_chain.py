"""Hash-verified bundle-v2 AgentBench chain composition."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterable

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from protocol import atomic_json, digest, object_digest


def _without_descriptions(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    value = copy.deepcopy(tools)
    for tool in value:
        function = tool.get("function")
        if isinstance(function, dict):
            function.pop("description", None)
    return value


def _hints(layer: str, value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > 1 or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{layer} must return zero or one string")
    return [item.strip() for item in value if item.strip()]


def _merge(groups: Iterable[list[str]]) -> list[str]:
    pieces = [piece for group in groups for piece in group]
    return ["\n".join(pieces)] if pieces else []


def write_manifest(path: Path, records: list[dict[str, Any]], artifact_root: Path,
                   snapshot_sha256: str) -> Path:
    rows = []
    for record in records:
        artifact = (artifact_root / record["artifact_path"]).resolve()
        if digest(artifact) != record["sha256"]:
            raise RuntimeError(f"accepted artifact tampered: {record['mechanism_id']}")
        rows.append({key: record[key] for key in
                     ("mechanism_id", "revision", "target_layer", "artifact_path", "sha256")})
    body = {"schema_version": 2, "snapshot_sha256": snapshot_sha256,
            "artifact_root": str(artifact_root.resolve()), "records": rows}
    atomic_json(path, {**body, "manifest_sha256": object_digest(body)})
    return path


def manifest_records(manifest_path: Path) -> tuple[Path, list[dict[str, Any]]]:
    value = json.loads(Path(manifest_path).read_text())
    supplied = value.pop("manifest_sha256", None)
    if value.get("schema_version") != 2 or supplied != object_digest(value):
        raise RuntimeError("chain manifest hash mismatch")
    root = Path(value["artifact_root"]).resolve()
    records = value.get("records")
    if not isinstance(records, list):
        raise RuntimeError("chain manifest records missing")
    seen = set()
    for record in records:
        if record["mechanism_id"] in seen:
            raise RuntimeError("duplicate mechanism in chain manifest")
        seen.add(record["mechanism_id"])
        path = (root / record["artifact_path"]).resolve()
        if root not in path.parents or digest(path) != record["sha256"]:
            raise RuntimeError(f"chain artifact hash mismatch: {record['mechanism_id']}")
    if object_digest(records) != value["snapshot_sha256"]:
        raise RuntimeError("chain snapshot hash mismatch")
    return root, records


def layer_paths(manifest_path: Path) -> list[Path]:
    root, records = manifest_records(manifest_path)
    return [(root / item["artifact_path"]).resolve() for item in records]


from evolution.agentbench_hooks_v31.runtime import CompositeHarness as CompleteHarness, install


class CompositeHarness(CompleteHarness):
    def __init__(self, records_or_paths, artifact_root=None, domain="dbbench"):
        values = list(records_or_paths)
        if values and isinstance(values[0], dict):
            if artifact_root is None:
                raise ValueError("artifact_root required for records")
            pairs = [(artifact_root / row["artifact_path"], row["target_layer"]) for row in values]
        else:
            pairs = []
            for path in values:
                path = Path(path)
                import ast
                tree = ast.parse(path.read_text())
                target = next(node.value.value for node in tree.body
                              if isinstance(node, ast.Assign) and any(
                                  isinstance(item, ast.Name) and item.id == "TARGET_LAYER"
                                  for item in node.targets))
                pairs.append((path, target))
        super().__init__(pairs, domain)


def inject_task_module(domain, task_module, manifest_path):
    from protocol import protocol
    receipt = install(task_module, domain, manifest_path)
    cfg = protocol()
    if (receipt["interface_version"] != cfg["interface_version"]
            or receipt["sources"] != cfg["interface_source_sha256"]
            or receipt["base_session_sha256"] != cfg["base_session_sha256"]):
        raise RuntimeError("worker interface differs from frozen protocol")
    return receipt
