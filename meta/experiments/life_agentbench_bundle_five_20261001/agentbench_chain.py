"""Hash-verified bundle-v2 AgentBench chain composition."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterable

from candidate_contract import load_artifact
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


class CompositeHarness:
    """Fresh ordered instances implementing the FourHookSession contracts."""

    h4_persistent = False
    h4_message_prefix = "Execution guidance: "
    h5_message_prefix = "\n\nProcedural guidance:\n"

    def __init__(self, records_or_paths: Iterable[Any], artifact_root: Path | None = None):
        values = list(records_or_paths)
        if values and isinstance(values[0], dict):
            if artifact_root is None:
                raise ValueError("artifact_root required for records")
            pairs = [((artifact_root / row["artifact_path"]).resolve(), row["target_layer"])
                     for row in values]
        else:
            pairs = [(Path(path), None) for path in values]
        self.layers = []
        for path, target in pairs:
            if target is None:
                text = path.read_text()
                target = next(layer for layer in ("h2", "h3", "h4", "h5")
                              if f"TARGET_LAYER='{layer}'" in text or f'TARGET_LAYER="{layer}"' in text)
            module = load_artifact(path, target)
            self.layers.append(module.Harness())
        self._h4_words = 0
        self._h5_words = 0

    def bind_task_context(self, context: Any) -> None:
        for layer in self.layers:
            bind = getattr(layer, "bind_task_context", None)
            if callable(bind):
                bind(copy.deepcopy(context))

    def h3(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        current = copy.deepcopy(tools)
        for layer in self.layers:
            hook = getattr(layer, "h3", None)
            if callable(hook):
                before = copy.deepcopy(current)
                current = hook(copy.deepcopy(current))
                if not isinstance(current, list) or _without_descriptions(before) != _without_descriptions(current):
                    raise ValueError("H3 layers may change tool descriptions only")
        return current

    def h5(self, messages: list[dict[str, Any]]) -> list[str]:
        result = _merge(_hints("H5", layer.h5(copy.deepcopy(messages)))
                        for layer in self.layers if callable(getattr(layer, "h5", None)))
        self._h5_words = len(result[0].split()) if result else 0
        if self._h4_words + self._h5_words > 120:
            raise ValueError("composed H4/H5 hint exceeds 120 words")
        return result

    def h4(self, messages: list[dict[str, Any]], remaining: int) -> list[str]:
        result = _merge(_hints("H4", layer.h4(copy.deepcopy(messages), remaining))
                        for layer in self.layers if callable(getattr(layer, "h4", None)))
        self._h4_words = len(result[0].split()) if result else 0
        if self._h5_words + self._h4_words > 120:
            raise ValueError("composed H4/H5 hint exceeds 120 words")
        return result

    def h2(self, message: dict[str, Any], history: list[dict[str, Any]]) -> dict[str, Any]:
        current = copy.deepcopy(message)
        for layer in self.layers:
            hook = getattr(layer, "h2", None)
            if callable(hook):
                current = hook(copy.deepcopy(current), copy.deepcopy(history))
                if not isinstance(current, dict) or current.get("role") != "assistant":
                    raise ValueError("H2 layers must return an assistant dictionary")
        if not isinstance(current, dict) or current.get("role") != "assistant":
            raise ValueError("H2 layers must return an assistant dictionary")
        return current


def factory(manifest_path: Path):
    root, records = manifest_records(manifest_path)

    def create(*_args: Any, **_kwargs: Any) -> CompositeHarness:
        return CompositeHarness(records, root)

    return create


def inject_task_module(domain: str, task_module: Any, manifest_path: Path) -> None:
    symbol = {"alfworld": "Harness", "dbbench": "DBBenchHarness",
              "webshop": "WebShopHarness"}[domain]
    setattr(task_module, symbol, factory(manifest_path))
