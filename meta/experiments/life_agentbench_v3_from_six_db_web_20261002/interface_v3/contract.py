"""Candidate validation and proposer-visible contract for interface v3.

This is a static restriction, not an OS security sandbox. Keep the proposer
workspace and evaluator isolated as in the historical experiment protocol.
"""
import ast
import importlib.util
from pathlib import Path

LAYERS = {"h2", "h3", "h4", "h5"}
OPTIONAL = {
    "bind_task_context": (LAYERS, 2),
    "bind_shared_state": (LAYERS, 2),
    "h3_message": ({"h3"}, 2),
    "h2_no_tool_message": ({"h2"}, 3),
    "drain_h2_messages": ({"h2"}, 1),
}
SAFE_IMPORTS = {
    "__future__": {"annotations"},
    "collections": {"Counter", "defaultdict", "deque"},
    "copy": {"copy", "deepcopy"},
    "json": {"dumps", "loads"},
    "re": {"A", "ASCII", "I", "IGNORECASE", "M", "MULTILINE", "S", "DOTALL",
           "X", "VERBOSE", "compile", "escape", "findall", "finditer", "fullmatch",
           "match", "search", "split", "sub", "subn", "Match", "Pattern"},
    "math": {"ceil", "floor", "fabs", "sqrt", "log", "log10", "exp", "isclose",
             "isfinite", "isinf", "isnan", "inf", "nan", "pi", "e", "prod", "fsum"},
    "difflib": {"SequenceMatcher", "get_close_matches"},
    "typing": {"Any", "Callable", "Iterable", "Mapping", "Sequence"},
    "string": {"ascii_letters", "ascii_lowercase", "ascii_uppercase", "digits",
               "hexdigits", "octdigits", "printable", "punctuation", "whitespace"},
    "itertools": {"chain", "combinations", "permutations", "product", "islice",
                  "repeat", "zip_longest", "groupby", "accumulate", "count"},
    "functools": {"partial", "reduce", "cache", "lru_cache"},
    "statistics": {"mean", "median", "fmean", "pstdev", "pvariance"},
    "datetime": {"date", "datetime", "time", "timedelta", "timezone"},
    "dataclasses": {"dataclass", "field", "fields", "replace"},
}
FORBIDDEN_NAMES = {
    "open", "eval", "exec", "compile", "__import__", "globals", "locals", "vars",
    "input", "breakpoint", "getattr", "setattr", "delattr", "type", "exit", "quit",
    "BaseException", "SystemExit", "KeyboardInterrupt", "GeneratorExit",
}
FORBIDDEN_TEXT = ("tasks.json", "split_tasks", "test_indices", "test_task_ids",
                  "ground_truth", "task-specific solution", "solution table", "results.md")


def _signature(node, count):
    args = node.args
    if (isinstance(node, ast.AsyncFunctionDef)
            or len(args.args) + len(args.posonlyargs) != count
            or args.vararg or args.kwarg or args.kwonlyargs or args.defaults):
        raise ValueError(f"invalid signature: {node.name}")


def validate_source(path, target):
    if target not in LAYERS:
        raise ValueError("unknown target layer")
    text = Path(path).read_text()
    if any(word in text.lower() for word in FORBIDDEN_TEXT):
        raise ValueError("forbidden private-data reference")
    tree = ast.parse(text, filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            raise ValueError("use allowed from-imports only")
        if isinstance(node, ast.ImportFrom):
            if (node.level or node.module not in SAFE_IMPORTS
                    or not {a.name for a in node.names} <= SAFE_IMPORTS[node.module]):
                raise ValueError("unsafe import")
            continue
        if isinstance(node, ast.Name) and (node.id.startswith("__") or node.id in FORBIDDEN_NAMES):
            raise ValueError(f"forbidden operation: {node.id}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError("dunder access forbidden")
        if isinstance(node, ast.ClassDef) and (node.bases or node.keywords or node.decorator_list):
            raise ValueError("inheritance and decorators forbidden")
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.decorator_list:
            raise ValueError("function decorators forbidden")
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Harness"]
    assignments = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == "TARGET_LAYER" for t in node.targets):
                assignments.append(node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "TARGET_LAYER":
            assignments.append(node.value)
    if (len(classes) != 1 or len(assignments) != 1
            or not isinstance(assignments[0], ast.Constant) or assignments[0].value != target):
        raise ValueError("declare one Harness and matching TARGET_LAYER")
    methods = [n for n in classes[0].body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    names = [n.name for n in methods]
    if len(names) != len(set(names)) or set(names).intersection(LAYERS) != {target}:
        raise ValueError("exactly one primary hook per artifact; no duplicate methods")
    for node in methods:
        if node.name in LAYERS:
            _signature(node, {"h2": 3, "h3": 2, "h4": 3, "h5": 2}[target])
        elif node.name in OPTIONAL:
            layers, count = OPTIONAL[node.name]
            if target not in layers:
                raise ValueError(f"{node.name} belongs to another layer")
            _signature(node, count)
    return tree


def load_artifact(path, target):
    path = Path(path)
    validate_source(path, target)
    spec = importlib.util.spec_from_file_location("life_interface_v3_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    instance = module.Harness()
    if {layer for layer in LAYERS if callable(getattr(instance, layer, None))} != {target}:
        raise ValueError("runtime primary hooks do not match TARGET_LAYER")
    for name, (layers, _) in OPTIONAL.items():
        if callable(getattr(instance, name, None)) and target not in layers:
            raise ValueError(f"runtime {name} belongs to another layer")
    return module


def proposer_instructions():
    """Include this entire text in the next run's source_view/contract.txt."""
    return Path(__file__).with_name("INTERFACE.md").read_text()
