"""Bundle-v2 CAS and independent sibling filtering, without host-side execution."""
import re
from pathlib import Path

from evolution.agentbench_hooks_v31.contract import LAYERS, validate_source

IDENTIFIER = re.compile(r"[a-z][a-z0-9_]{0,63}")


class RejectedOperations(ValueError):
    """Retain every rejection even when there is no valid sibling operation."""
    def __init__(self, rejected):
        self.rejected_operations = rejected
        details = "; ".join(
            f"operations[{item['index']}] ({item['operation_id'] or 'unnamed'}): {item['error']}"
            for item in rejected)
        super().__init__("all operations rejected: " + details)


def filter_operations(document, domain, records, snapshot, workspace):
    required = {"schema_version", "bundle_id", "domain", "base_snapshot_sha256",
                "hypothesis", "operations"}
    if not isinstance(document, dict):
        raise ValueError("bundle-v2 document must be a JSON object")
    if set(document) != required or document["schema_version"] != 2:
        raise ValueError(f"exact bundle-v2 fields required: missing={sorted(required - set(document))}, "
                         f"unexpected={sorted(set(document) - required)}, "
                         f"schema_version must be 2 (received {document.get('schema_version')!r})")
    if document["domain"] != domain or document["base_snapshot_sha256"] != snapshot:
        raise ValueError(f"domain or snapshot CAS mismatch: expected domain={domain!r}, "
                         f"snapshot={snapshot!r}; received domain={document['domain']!r}, "
                         f"snapshot={document['base_snapshot_sha256']!r}")
    if not isinstance(document["bundle_id"], str) or not IDENTIFIER.fullmatch(document["bundle_id"]):
        raise ValueError(f"invalid bundle_id: expected {IDENTIFIER.pattern!r}; received {document['bundle_id']!r}")
    if not isinstance(document["hypothesis"], str) or not document["hypothesis"].strip():
        raise ValueError("nonempty bundle hypothesis required: hypothesis must be a nonblank string")
    if not isinstance(document["operations"], list) or not document["operations"]:
        raise ValueError("nonempty operations required: operations must be a nonempty JSON array")
    active = {r["mechanism_id"]: r for r in records}
    seen_operations, seen_mechanisms, valid, rejected = set(), set(), [], []
    for index, op in enumerate(document["operations"]):
        try:
            common = {"operation_id", "mechanism_id", "op", "target_layer", "hypothesis"}
            if not isinstance(op, dict):
                raise ValueError("operation must be a JSON object")
            if not common <= set(op):
                raise ValueError(f"missing operation fields: {sorted(common - set(op))}")
            for key in ("operation_id", "mechanism_id"):
                if not isinstance(op[key], str) or not IDENTIFIER.fullmatch(op[key]):
                    raise ValueError(f"invalid operation/mechanism identifier: {key} must match "
                                     f"{IDENTIFIER.pattern!r}; received {op[key]!r}")
            if op["operation_id"] in seen_operations or op["mechanism_id"] in seen_mechanisms:
                raise ValueError(f"duplicate operation/mechanism: operation_id={op['operation_id']!r}, "
                                 f"mechanism_id={op['mechanism_id']!r}; use one operation per mechanism and unique IDs")
            kind = op["op"]
            if (not isinstance(kind, str) or kind not in {"add", "modify", "delete"}
                    or not isinstance(op["target_layer"], str) or op["target_layer"] not in LAYERS):
                raise ValueError(f"invalid operation or layer: op={kind!r} must be add/modify/delete; "
                                 f"target_layer={op['target_layer']!r} must be one of {sorted(LAYERS)}")
            if not isinstance(op["hypothesis"], str) or not op["hypothesis"].strip():
                raise ValueError("nonempty operation hypothesis required: hypothesis must be a nonblank string")
            fields = common | ({"artifact"} if kind == "add" else
                               {"artifact", "expected_revision", "expected_sha256"} if kind == "modify" else
                               {"expected_revision", "expected_sha256"})
            if set(op) != fields:
                raise ValueError(f"operation fields do not match {kind}: "
                                 f"missing={sorted(fields - set(op))}, "
                                 f"unexpected={sorted(set(op) - fields)}")
            prior = active.get(op["mechanism_id"])
            if kind == "add" and prior is not None:
                raise ValueError(f"mechanism already exists: {op['mechanism_id']!r}; use modify with "
                                 "expected_revision and expected_sha256 from inventory.json, or a new mechanism ID")
            if kind != "add":
                if (prior is None or type(op["expected_revision"]) is not int or
                        op["expected_revision"] != prior["revision"] or op["expected_sha256"] != prior["sha256"]):
                    expected = ({"expected_revision": prior["revision"], "expected_sha256": prior["sha256"]}
                                if prior is not None else None)
                    raise ValueError(f"stale mechanism revision/hash: mechanism={op['mechanism_id']!r}, "
                                     f"current inventory record={expected!r}, "
                                     f"received revision={op['expected_revision']!r}, sha256={op['expected_sha256']!r}; "
                                     "copy the current inventory values; a missing mechanism requires add, not modify/delete")
                if kind == "delete" and op["target_layer"] != prior["target_layer"]:
                    raise ValueError(f"delete layer mismatch: expected {prior['target_layer']!r}, "
                                     f"received {op['target_layer']!r}")
            if kind != "delete":
                relative = op["artifact"]
                if not isinstance(relative, str) or not re.fullmatch(r"artifacts/[A-Za-z0-9_.-]+\.py", relative):
                    raise ValueError(f"direct artifacts/*.py path required: received {relative!r}; "
                                     "use artifacts/name.py, without subdirectories or parent traversal")
                path = Path(workspace) / relative
                if path.is_symlink() or path.resolve().parent != (Path(workspace) / "artifacts").resolve():
                    raise ValueError(f"artifact path escape: {relative!r} must be a real file directly "
                                     "inside this workspace's artifacts directory, not a symlink")
                validate_source(path, op["target_layer"])
            seen_operations.add(op["operation_id"])
            seen_mechanisms.add(op["mechanism_id"])
            valid.append(dict(op))
        except (ValueError, SyntaxError, OSError, TypeError) as exc:
            rejected.append({"index": index, "operation_id": op.get("operation_id") if isinstance(op, dict) else None,
                             "mechanism_id": op.get("mechanism_id") if isinstance(op, dict) else None,
                             "artifact": op.get("artifact") if isinstance(op, dict) else None,
                             "error": f"{type(exc).__name__}: {exc}"})
    if not valid:
        raise RejectedOperations(rejected)
    return {**document, "operations": valid}, rejected
