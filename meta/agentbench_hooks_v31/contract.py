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
        raise ValueError(f"invalid signature: {node.name} at line {node.lineno}; expected exactly "
                         f"{count} positional arguments including self, no defaults/varargs/keyword-only/async; "
                         f"received ({ast.unparse(args)})")


def validate_source(path, target):
    if target not in LAYERS:
        raise ValueError(f"unknown target layer: received {target!r}; expected one of {sorted(LAYERS)}")
    text = Path(path).read_text()
    if any(word in text.lower() for word in FORBIDDEN_TEXT):
        matches = [word for word in FORBIDDEN_TEXT if word in text.lower()]
        raise ValueError(f"forbidden private-data reference in {path}: matched {matches!r}; "
                         "use only public runtime inputs, not hidden answers or task/test files")
    tree = ast.parse(text, filename=str(path))
    # Safe regex compilation must not be confused with builtin code compilation.
    regex_compile = any(isinstance(n, ast.ImportFrom) and n.module == 're'
                        and any(a.name == 'compile' and (a.asname or a.name) == 'compile'
                                for a in n.names) for n in ast.walk(tree))
    compile_shadowed = any(
        (isinstance(n, ast.Name) and n.id == 'compile' and isinstance(n.ctx, (ast.Store, ast.Del)))
        or (isinstance(n, ast.arg) and n.arg == 'compile')
        or (isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == 'compile')
        or (isinstance(n, ast.ImportFrom) and any((a.asname or a.name) == 'compile'
            and not (n.module == 're' and a.name == 'compile') for a in n.names))
        for n in ast.walk(tree))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            raise ValueError(f"use allowed from-imports only: line {node.lineno}: {ast.unparse(node)}; "
                             "use from module import name from SAFE_IMPORTS in source_view/contract.py")
        if isinstance(node, ast.ImportFrom):
            if (node.level or node.module not in SAFE_IMPORTS
                    or not {a.name for a in node.names} <= SAFE_IMPORTS[node.module]):
                raise ValueError(f"unsafe import at line {node.lineno}: {ast.unparse(node)}; "
                                 f"allowed names for {node.module!r}: {sorted(SAFE_IMPORTS.get(node.module, set()))}; "
                                 "relative imports are forbidden")
            continue
        if isinstance(node, ast.Name) and (node.id.startswith("__") or node.id in FORBIDDEN_NAMES):
            if not (node.id == 'compile' and regex_compile and not compile_shadowed
                    and isinstance(node.ctx, ast.Load)):
                raise ValueError(f"forbidden operation: {node.id} at line {node.lineno}; "
                                 "use public hook inputs and allowlisted pure functions")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError(f"dunder access forbidden: {node.attr} at line {node.lineno}")
        if isinstance(node, ast.ClassDef) and (node.bases or node.keywords or node.decorator_list):
            raise ValueError(f"inheritance and decorators forbidden: class {node.name} at line {node.lineno}; "
                             "define a plain Harness class without bases/decorators")
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.decorator_list:
            raise ValueError(f"function decorators forbidden: {node.name} at line {node.lineno}")
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
        raise ValueError(f"declare one Harness and matching TARGET_LAYER: expected TARGET_LAYER={target!r}, "
                         f"one class Harness; received Harness count={len(classes)}, "
                         f"TARGET_LAYER assignments={[ast.unparse(value) for value in assignments]!r}")
    methods = [n for n in classes[0].body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    names = [n.name for n in methods]
    if len(names) != len(set(names)) or set(names).intersection(LAYERS) != {target}:
        raise ValueError(f"exactly one primary hook per artifact; no duplicate methods: "
                         f"expected primary hook={target!r}; received primary hooks={sorted(set(names).intersection(LAYERS))}, "
                         f"duplicate methods={sorted({name for name in names if names.count(name) > 1})}")
    for node in methods:
        if node.name in LAYERS:
            _signature(node, {"h2": 3, "h3": 2, "h4": 3, "h5": 2}[target])
        elif node.name in OPTIONAL:
            layers, count = OPTIONAL[node.name]
            if target not in layers:
                raise ValueError(f"{node.name} belongs to another layer at line {node.lineno}: "
                                 f"allowed layers={sorted(layers)}, received {target!r}")
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
        raise ValueError(f"runtime primary hooks do not match TARGET_LAYER: expected {target!r}; "
                         f"received {[name for name in LAYERS if callable(getattr(instance, name, None))]}")
    for name, (layers, _) in OPTIONAL.items():
        if callable(getattr(instance, name, None)) and target not in layers:
            raise ValueError(f"runtime {name} belongs to another layer: allowed={sorted(layers)}, received {target!r}")
    return module


def proposer_instructions():
    """Include this entire text in the next run's source_view/contract.txt."""
    return Path(__file__).with_name("INTERFACE.md").read_text()
