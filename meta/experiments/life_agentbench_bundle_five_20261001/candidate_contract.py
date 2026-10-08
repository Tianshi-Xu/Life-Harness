"""Bundle-v2 validation, artifact safety, and proposer isolation."""
from __future__ import annotations

import ast
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

LAYERS = {"h2", "h3", "h4", "h5"}
OPS = {"add", "modify", "delete"}
IDENTIFIER = re.compile(r"[a-z][a-z0-9_]{0,63}")
SHA256 = re.compile(r"[0-9a-f]{64}")
FORBIDDEN_TEXT = (
    "tasks.json", "split_tasks", "test_indices", "test_task_ids", "ground_truth",
    "task-specific solution", "solution table", "results.md",
)
SAFE_IMPORT_MEMBERS = {
    "__future__": {"annotations"},
    "collections": {"Counter", "defaultdict", "deque"},
    "copy": {"copy", "deepcopy"},
    "dataclasses": {"dataclass", "field", "fields", "replace"},
    "datetime": {"date", "datetime", "time", "timedelta", "timezone"},
    "difflib": {"SequenceMatcher", "get_close_matches"},
    "functools": {"cache", "lru_cache", "partial", "reduce"},
    "itertools": {"accumulate", "chain", "combinations", "combinations_with_replacement",
                  "compress", "count", "cycle", "dropwhile", "filterfalse", "groupby",
                  "islice", "pairwise", "permutations", "product", "repeat", "starmap",
                  "takewhile", "tee", "zip_longest"},
    "json": {"dumps", "loads"},
    "math": {"ceil", "comb", "copysign", "e", "exp", "fabs", "factorial", "floor",
             "fmod", "fsum", "gcd", "inf", "isclose", "isfinite", "isinf", "isnan",
             "log", "log10", "nan", "pi", "pow", "prod", "remainder", "sqrt", "tau"},
    "re": {"A", "ASCII", "I", "IGNORECASE", "M", "MULTILINE", "Match", "Pattern",
           "S", "DOTALL", "X", "VERBOSE", "compile", "escape", "findall", "finditer",
           "fullmatch", "match", "search", "split", "sub", "subn"},
    "statistics": {"fmean", "mean", "median", "median_grouped", "median_high",
                   "median_low", "mode", "multimode", "pstdev", "pvariance", "stdev",
                   "variance"},
    "string": {"ascii_letters", "ascii_lowercase", "ascii_uppercase", "digits",
               "hexdigits", "octdigits", "printable", "punctuation", "whitespace"},
    "typing": {"Any", "Callable", "Iterable", "Mapping", "Sequence"},
}
SAFE_IMPORTS = set(SAFE_IMPORT_MEMBERS)


def validate_source(path: Path, target_layer: str) -> ast.Module:
    text = path.read_text()
    lowered = text.lower()
    hit = next((item for item in FORBIDDEN_TEXT if item in lowered), None)
    if hit:
        raise ValueError(f"forbidden artifact reference: {hit}")
    tree = ast.parse(text, filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            raise ValueError("artifact direct module imports are forbidden; use allowed from-imports")
        if isinstance(node, ast.ImportFrom):
            root = node.module or ""
            allowed = SAFE_IMPORT_MEMBERS.get(root)
            imported = {alias.name for alias in node.names}
            if node.level or allowed is None or not imported <= allowed:
                raise ValueError(f"artifact import not allowed: {root} {sorted(imported)}")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.Name) and (node.id.startswith("__") or node.id in {
            "open", "eval", "exec", "compile", "__import__", "globals", "locals",
            "vars", "input", "breakpoint", "getattr", "setattr", "delattr", "type",
            "exit", "quit", "BaseException", "SystemExit", "KeyboardInterrupt",
            "GeneratorExit",
        }):
            raise ValueError(f"forbidden artifact operation: {node.id}")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                raise ValueError("artifact dunder access is forbidden")
        elif isinstance(node, ast.ClassDef) and (node.bases or node.keywords or node.decorator_list):
            raise ValueError("artifact class inheritance and decorators are forbidden")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.decorator_list:
            raise ValueError("artifact function decorators are forbidden")
    harnesses = [node for node in tree.body
                 if isinstance(node, ast.ClassDef) and node.name == "Harness"]
    assignments = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "TARGET_LAYER"
                for target in node.targets):
            assignments.append(node)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) \
                and node.target.id == "TARGET_LAYER":
            assignments.append(node)
    declared = set()
    if len(harnesses) == 1:
        declared = {node.name for node in harnesses[0].body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name in LAYERS}
    if len(assignments) != 1 or not isinstance(assignments[0].value, ast.Constant):
        raise ValueError("artifact must assign TARGET_LAYER exactly once")
    if assignments[0].value.value != target_layer or len(harnesses) != 1 or declared != {target_layer}:
        raise ValueError("artifact must define exactly one Harness hook matching TARGET_LAYER")
    hook = next(node for node in harnesses[0].body if getattr(node, "name", None) == target_layer)
    arguments = hook.args
    expected_arguments = {"h2": 3, "h3": 2, "h4": 3, "h5": 2}[target_layer]
    positional = len(arguments.posonlyargs) + len(arguments.args)
    if (isinstance(hook, ast.AsyncFunctionDef) or positional != expected_arguments
            or arguments.vararg or arguments.kwarg or arguments.kwonlyargs
            or arguments.defaults or arguments.kw_defaults):
        raise ValueError(f"artifact {target_layer} hook has an invalid signature")
    return tree


def load_artifact(path: Path, target_layer: str) -> Any:
    validate_source(path, target_layer)
    name = "life_bundle_artifact_" + re.sub(r"\W", "_", path.stem) + "_" + str(id(path))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError("artifact cannot be imported")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        harness = getattr(module, "Harness", None)
        if not callable(harness):
            raise ValueError("artifact Harness is not callable")
        instance = harness()
        callable_layers = {layer for layer in LAYERS
                           if callable(getattr(instance, layer, None))}
        if callable_layers != {target_layer}:
            raise ValueError("artifact must expose exactly one callable target hook")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"artifact runtime validation failed: {type(exc).__name__}: {exc}") from exc
    finally:
        sys.modules.pop(name, None)
    return module


def _artifact_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not re.fullmatch(r"artifacts/[A-Za-z0-9_.-]+\.py", relative):
        raise ValueError("artifact must be a direct artifacts/*.py path")
    path = (root / relative).resolve()
    if path.parent != (root / "artifacts").resolve():
        raise ValueError("artifact path escapes artifacts directory")
    return path


def validate_bundle(value: dict[str, Any], domain: str, active: list[dict[str, Any]],
                    expected_snapshot_sha256: str, root: Path) -> dict[str, Any]:
    required = {"schema_version", "bundle_id", "domain", "base_snapshot_sha256",
                "hypothesis", "operations"}
    if set(value) != required or value.get("schema_version") != 2:
        raise ValueError("bundle must contain exactly the schema-v2 fields")
    if value["domain"] != domain or value["base_snapshot_sha256"] != expected_snapshot_sha256:
        raise ValueError("bundle domain or parent snapshot CAS mismatch")
    if not isinstance(value["bundle_id"], str) or not IDENTIFIER.fullmatch(value["bundle_id"]):
        raise ValueError("invalid bundle_id")
    if not isinstance(value["hypothesis"], str) or not value["hypothesis"].strip():
        raise ValueError("bundle hypothesis must be nonempty")
    operations = value["operations"]
    if not isinstance(operations, list) or not operations:
        raise ValueError("bundle operations must be a nonempty list")
    current = {item["mechanism_id"]: item for item in active}
    operation_ids: set[str] = set()
    mechanism_ids: set[str] = set()
    normalized = []
    for operation in operations:
        common = {"operation_id", "mechanism_id", "op", "target_layer", "hypothesis"}
        if not isinstance(operation, dict) or not common.issubset(operation):
            raise ValueError("operation is missing required fields")
        operation_id = operation["operation_id"]
        mechanism_id = operation["mechanism_id"]
        if not isinstance(operation_id, str) or not IDENTIFIER.fullmatch(operation_id):
            raise ValueError("invalid operation_id")
        if not isinstance(mechanism_id, str) or not IDENTIFIER.fullmatch(mechanism_id):
            raise ValueError("invalid mechanism_id")
        if operation_id in operation_ids or mechanism_id in mechanism_ids:
            raise ValueError("duplicate operation_id or mechanism_id in bundle")
        operation_ids.add(operation_id)
        mechanism_ids.add(mechanism_id)
        op = operation["op"]
        if op not in OPS or operation["target_layer"] not in LAYERS:
            raise ValueError("invalid operation type or target layer")
        if not isinstance(operation["hypothesis"], str) or not operation["hypothesis"].strip():
            raise ValueError("operation hypothesis must be nonempty")
        allowed = common | ({"artifact"} if op == "add" else
                            {"artifact", "expected_revision", "expected_sha256"} if op == "modify" else
                            {"expected_revision", "expected_sha256"})
        if set(operation) != allowed:
            raise ValueError("operation fields do not match its type")
        prior = current.get(mechanism_id)
        if op == "add":
            if prior is not None:
                raise ValueError("add mechanism already exists")
        else:
            if prior is None:
                raise ValueError("modify/delete mechanism does not exist")
            if operation["expected_revision"] != prior["revision"] or operation["expected_sha256"] != prior["sha256"]:
                raise ValueError("stale mechanism revision or hash")
            if op == "delete" and operation["target_layer"] != prior["target_layer"]:
                raise ValueError("delete target layer does not match mechanism")
        item = dict(operation)
        if op != "delete":
            artifact = _artifact_path(root, operation["artifact"])
            if not artifact.is_file():
                raise ValueError(f"missing artifact: {operation['artifact']}")
            load_artifact(artifact, operation["target_layer"])
            item["artifact_path"] = artifact
        normalized.append(item)
    return {**value, "operations": normalized}


def prompt(domain: str, round_number: int, workspace: Path) -> str:
    return f"""Improve the {domain} Life-Harness in round {round_number}/5.
You may read only active/, active_snapshot.json, inventory.json, evidence/, source_view/, and history.json.
The test split, test results, Meta results, and all private labels are hidden and forbidden.
Output bundle.json (schema_version 2) plus artifacts/*.py. A bundle may contain ANY NONZERO
NUMBER of operations spanning H2, H3, H4, and H5. Every operation is independently add,
modify, or delete and declares exactly ONE target_layer. Use the schema and current revisions
in source_view/contract.txt and inventory.json. Add/modify references one artifact; delete has none.
Each artifact assigns TARGET_LAYER and defines exactly one matching Harness hook.
H3 changes descriptions only; H2 returns an assistant dictionary; composed H4/H5 is <=120 words.
Use only Read, Write, Edit, Glob, and Grep; artifacts/ already exists. Do not call Bash,
Agent, Task, or any network tool. Do not modify frozen inputs and do not run a benchmark.
Workspace: {workspace}
"""


def public_history(rounds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = ("round", "train_batch", "incumbent_score", "candidate_score",
              "accepted", "next_batch", "batch_rotated", "bundle_id",
              "bundle_sha256", "bundle_hypothesis", "active_snapshot_sha256")
    operation_fields = ("operation_id", "mechanism_id", "op", "target_layer", "hypothesis")
    result = []
    for row in rounds:
        public = {key: row[key] for key in fields if key in row}
        public["operations"] = [
            {key: operation[key] for key in operation_fields if key in operation}
            for operation in row.get("operations", [])
        ]
        result.append(public)
    serialized = json.dumps(result, sort_keys=True).lower()
    if "test_" in serialized or "meta_result" in serialized:
        raise ValueError("hidden result fields escaped proposer history isolation")
    return result
