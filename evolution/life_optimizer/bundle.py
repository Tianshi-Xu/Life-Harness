"""Bundle-v2 CAS and independent sibling filtering, without host-side execution."""
import re
from pathlib import Path

from evolution.agentbench_hooks.contract import LAYERS, validate_source

IDENTIFIER = re.compile(r"[a-z][a-z0-9_]{0,63}")


def filter_operations(document, domain, records, snapshot, workspace):
    required = {"schema_version", "bundle_id", "domain", "base_snapshot_sha256",
                "hypothesis", "operations"}
    if not isinstance(document, dict) or set(document) != required or document["schema_version"] != 2:
        raise ValueError("exact bundle-v2 fields required")
    if document["domain"] != domain or document["base_snapshot_sha256"] != snapshot:
        raise ValueError("domain or snapshot CAS mismatch")
    if not isinstance(document["bundle_id"], str) or not IDENTIFIER.fullmatch(document["bundle_id"]):
        raise ValueError("invalid bundle_id")
    if not isinstance(document["hypothesis"], str) or not document["hypothesis"].strip():
        raise ValueError("nonempty bundle hypothesis required")
    if not isinstance(document["operations"], list) or not document["operations"]:
        raise ValueError("nonempty operations required")
    active = {r["mechanism_id"]: r for r in records}
    seen_operations, seen_mechanisms, valid, rejected = set(), set(), [], []
    for index, op in enumerate(document["operations"]):
        try:
            common = {"operation_id", "mechanism_id", "op", "target_layer", "hypothesis"}
            if not isinstance(op, dict) or not common <= set(op):
                raise ValueError("missing operation fields")
            for key in ("operation_id", "mechanism_id"):
                if not isinstance(op[key], str) or not IDENTIFIER.fullmatch(op[key]):
                    raise ValueError("invalid operation/mechanism identifier")
            if op["operation_id"] in seen_operations or op["mechanism_id"] in seen_mechanisms:
                raise ValueError("duplicate operation/mechanism")
            kind = op["op"]
            if kind not in {"add", "modify", "delete"} or op["target_layer"] not in LAYERS:
                raise ValueError("invalid operation or layer")
            if not isinstance(op["hypothesis"], str) or not op["hypothesis"].strip():
                raise ValueError("nonempty operation hypothesis required")
            fields = common | ({"artifact"} if kind == "add" else
                               {"artifact", "expected_revision", "expected_sha256"} if kind == "modify" else
                               {"expected_revision", "expected_sha256"})
            if set(op) != fields:
                raise ValueError("operation fields do not match type")
            prior = active.get(op["mechanism_id"])
            if kind == "add" and prior is not None:
                raise ValueError("mechanism already exists")
            if kind != "add":
                if (prior is None or type(op["expected_revision"]) is not int or
                        op["expected_revision"] != prior["revision"] or op["expected_sha256"] != prior["sha256"]):
                    raise ValueError("stale mechanism revision/hash")
                if kind == "delete" and op["target_layer"] != prior["target_layer"]:
                    raise ValueError("delete layer mismatch")
            if kind != "delete":
                relative = op["artifact"]
                if not isinstance(relative, str) or not re.fullmatch(r"artifacts/[A-Za-z0-9_.-]+\.py", relative):
                    raise ValueError("direct artifacts/*.py path required")
                path = Path(workspace) / relative
                if path.is_symlink() or path.resolve().parent != (Path(workspace) / "artifacts").resolve():
                    raise ValueError("artifact path escape")
                validate_source(path, op["target_layer"])
            seen_operations.add(op["operation_id"])
            seen_mechanisms.add(op["mechanism_id"])
            valid.append(dict(op))
        except (ValueError, SyntaxError, OSError, TypeError) as exc:
            rejected.append({"index": index, "operation_id": op.get("operation_id") if isinstance(op, dict) else None,
                             "error": f"{type(exc).__name__}: {exc}"})
    if not valid:
        raise ValueError("all operations rejected")
    return {**document, "operations": valid}, rejected
