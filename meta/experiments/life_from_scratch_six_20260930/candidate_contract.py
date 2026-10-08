"""Candidate validation and proposer prompt support."""
from __future__ import annotations

import ast
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

LAYERS = {"h2", "h3", "h4", "h5"}
FORBIDDEN_TEXT = (
    "tasks.json", "split_tasks", "test_indices", "test_task_ids", "ground_truth",
    "task-specific solution", "solution table",
)
SAFE_IMPORTS = {
    "__future__", "ast", "collections", "copy", "dataclasses", "datetime",
    "difflib", "functools", "itertools", "json", "math", "re", "statistics",
    "string", "typing",
}


def validate_source(path: Path, family: str) -> ast.Module:
    text = path.read_text()
    lowered = text.lower()
    hit = next((item for item in FORBIDDEN_TEXT if item.lower() in lowered), None)
    if hit:
        raise ValueError(f"forbidden candidate reference: {hit}")
    tree = ast.parse(text, filename=str(path))
    allowed = SAFE_IMPORTS | ({"tau2"} if family == "tau" else set())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots = [alias.name.split(".")[0] for alias in node.names]
            if any(root not in allowed for root in roots):
                raise ValueError(f"candidate import not allowed: {roots}")
        elif isinstance(node, ast.ImportFrom):
            roots = [(node.module or "").split(".")[0]]
            if any(root not in allowed for root in roots):
                raise ValueError(f"candidate import not allowed: {roots}")
        elif isinstance(node, ast.Name) and node.id in {
            "open", "eval", "exec", "compile", "__import__", "globals", "locals",
            "vars", "input", "breakpoint",
        }:
            raise ValueError(f"forbidden candidate operation: {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError("candidate dunder access is forbidden")
    if family == "agentbench":
        harnesses = [node for node in tree.body
                     if isinstance(node, ast.ClassDef) and node.name == "Harness"]
        if len(harnesses) != 1:
            raise ValueError("AgentBench candidate must define exactly one Harness class")
        declared = {node.name for node in harnesses[0].body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in LAYERS}
        targets = {node.value.value for node in tree.body
                   if isinstance(node, ast.Assign)
                   for target in node.targets if isinstance(target, ast.Name) and target.id == "TARGET_LAYER"
                   if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)}
        if len(targets) != 1 or declared != targets:
            raise ValueError("AgentBench candidate must implement only its TARGET_LAYER hook")
    return tree


def load_candidate(path: Path, family: str) -> Any:
    validate_source(path, family)
    name = "life_candidate_" + re.sub(r"\W", "_", path.stem)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError("candidate cannot be imported")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    layer = getattr(module, "TARGET_LAYER", None)
    if layer not in LAYERS:
        raise ValueError("candidate TARGET_LAYER must be h2, h3, h4, or h5")
    if family == "tau":
        if not callable(getattr(module, "register", None)):
            raise ValueError("Tau candidate must export register()")
    else:
        harness = getattr(module, "Harness", None)
        if not callable(harness):
            raise ValueError("AgentBench candidate must export Harness")
        instance = harness()
        method = getattr(instance, layer, None)
        if not callable(method):
            raise ValueError(f"AgentBench candidate Harness must implement {layer}")
    return module


def validate_proposal(value: dict[str, Any], domain: str) -> dict[str, Any]:
    required = {"name", "domain", "target_layer", "hypothesis"}
    if not required.issubset(value):
        raise ValueError("proposal is missing required fields")
    if value["domain"] != domain or value["target_layer"] not in LAYERS:
        raise ValueError("proposal domain or target layer does not match")
    if not re.fullmatch(r"[a-z][a-z0-9_]*", value["name"]):
        raise ValueError("proposal name must be lower_snake_case")
    return {key: value[key] for key in ("name", "domain", "target_layer", "hypothesis")}


def prompt(domain: str, family: str, round_number: int, workspace: Path) -> str:
    contract = (
        "export TARGET_LAYER and register(); use Tau H2-H5 registries through register()"
        if family == "tau" else
        "export TARGET_LAYER and an incremental Harness class implementing only that hook"
    )
    return f"""Improve the {domain} Life-Harness in round {round_number}/6.
Read only this workspace's accepted/, evidence/, source_view/, and history.json.
The fixed test split and all test results are hidden and must not be inferred.
Produce exactly one focused candidate at candidate.py and proposal.json with keys
name, domain={domain!r}, target_layer (h2|h3|h4|h5), and hypothesis.
Candidate contract: {contract}.
AgentBench candidates are structured H2/H3/H4/H5 layers, never free-form request/response transforms.
H3 may alter tool descriptions only. H2 must return an assistant dictionary.
H4/H5 may each return zero or one string; the composed total is at most one 120-word hint.
No file, network, subprocess, environment-variable, dynamic execution, hidden labels,
task IDs, dataset references, or instance-specific solution tables at runtime.
Use only public names from source_view; vars(), globals(), locals(), and dunder
attributes such as __dict__ are forbidden. Standard-library imports must come from
the allowlist in contract.txt.
Use one mechanism, preserve passing counterexamples, and do not run a benchmark.
Workspace: {workspace}
"""


def public_history(rounds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def scrub(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: scrub(item) for key, item in value.items()
                    if not key.startswith("test_")}
        if isinstance(value, list):
            return [scrub(item) for item in value]
        return value
    result = scrub(rounds)
    # A second serialized guard catches nested or accidentally unusual mappings.
    if "test_" in json.dumps(result, sort_keys=True):
        raise ValueError("test fields escaped proposer history isolation")
    return result
