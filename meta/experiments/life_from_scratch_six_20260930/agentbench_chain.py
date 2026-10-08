"""Structured incremental H2-H5 composition for AgentBench."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from typing import Any, Iterable

from candidate_contract import load_candidate


def _load_layer(path: Path, index: int) -> Any:
    module = load_candidate(path, "agentbench")
    harness = module.Harness()
    harness._life_layer_index = index
    return harness


def _without_descriptions(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    value = copy.deepcopy(tools)
    for tool in value:
        function = tool.get("function")
        if isinstance(function, dict):
            function.pop("description", None)
    return value


def _validate_hints(layer: str, value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > 1 or not all(isinstance(x, str) for x in value):
        raise ValueError(f"{layer} must return zero or one string")
    return value


def _merge_hints(layer: str, groups: Iterable[list[str]]) -> list[str]:
    pieces = [hint.strip() for group in groups for hint in group if hint.strip()]
    if not pieces:
        return []
    combined = "\n".join(pieces)
    if len(combined.split()) > 120:
        raise ValueError("composed H4/H5 hint exceeds 120 words")
    return [combined]


class CompositeHarness:
    """A fresh, ordered composition of incremental candidate Harness layers."""

    h4_persistent = False
    h4_message_prefix = "Execution guidance: "
    h5_message_prefix = "\n\nProcedural guidance:\n"

    def __init__(self, layer_paths: Iterable[Path]):
        self.layers = [_load_layer(Path(path), index)
                       for index, path in enumerate(layer_paths)]

    def bind_task_context(self, context: Any) -> None:
        for layer in self.layers:
            bind = getattr(layer, "bind_task_context", None)
            if callable(bind):
                bind(copy.deepcopy(context))

    def h3(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        current = copy.deepcopy(tools)
        for layer in self.layers:
            hook = getattr(layer, "h3", None)
            if not callable(hook):
                continue
            before = copy.deepcopy(current)
            current = hook(copy.deepcopy(current))
            if not isinstance(current, list) or _without_descriptions(before) != _without_descriptions(current):
                raise ValueError("H3 layers may change tool descriptions only")
        return current

    def h5(self, messages: list[dict[str, Any]]) -> list[str]:
        groups = []
        for layer in self.layers:
            hook = getattr(layer, "h5", None)
            if callable(hook):
                groups.append(_validate_hints("H5", hook(copy.deepcopy(messages))))
        return _merge_hints("H5", groups)

    def h4(self, messages: list[dict[str, Any]], remaining: int) -> list[str]:
        groups = []
        for layer in self.layers:
            hook = getattr(layer, "h4", None)
            if callable(hook):
                groups.append(_validate_hints("H4", hook(copy.deepcopy(messages), remaining)))
        return _merge_hints("H4", groups)

    def h2(self, message: dict[str, Any], history: list[dict[str, Any]]) -> dict[str, Any]:
        current = copy.deepcopy(message)
        for layer in self.layers:
            hook = getattr(layer, "h2", None)
            if not callable(hook):
                continue
            current = hook(copy.deepcopy(current), copy.deepcopy(history))
            if not isinstance(current, dict) or current.get("role") != "assistant":
                raise ValueError("H2 layers must return an assistant dictionary")
        return current


def layer_paths(manifest_path: Path) -> list[Path]:
    manifest_path = Path(manifest_path).resolve()
    value = json.loads(manifest_path.read_text())
    paths = []
    for item in value.get("layers", []):
        path = Path(item)
        if not path.is_absolute():
            path = manifest_path.parent / path
        paths.append(path.resolve())
    return paths


def factory(manifest_path: Path):
    paths = layer_paths(manifest_path)

    def create(*_args: Any, **_kwargs: Any) -> CompositeHarness:
        return CompositeHarness(paths)

    return create


def inject_task_module(domain: str, task_module: Any, manifest_path: Path) -> None:
    """Replace only the task module's Harness constructor in the evaluator process."""
    create = factory(manifest_path)
    symbol = {"alfworld": "Harness", "dbbench": "DBBenchHarness",
              "webshop": "WebShopHarness"}[domain]
    setattr(task_module, symbol, create)
