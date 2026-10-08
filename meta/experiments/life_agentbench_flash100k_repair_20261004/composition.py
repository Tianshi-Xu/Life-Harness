"""Proven bundle-v2 CAS composition, copied into this independent experiment."""
from pathlib import Path
import json
import shutil
from typing import Any, Callable
from agentbench_chain import write_manifest
from candidate_contract import validate_bundle
from protocol import digest, object_digest

def copy_verified(source, target):
    source, target = Path(source), Path(target)
    if target.exists():
        if digest(source) != digest(target):
            raise RuntimeError('Immutable staged artifact changed')
    else:
        shutil.copy2(source, target)
    if digest(source) != digest(target):
        raise RuntimeError('Artifact copy mismatch')

def commit_bundle(run_dir, bundle, state):
    verify_active(run_dir, state)
    final_dir = run_dir / 'versions' / bundle['bundle_id']
    final_dir.mkdir(parents=True, exist_ok=True)
    expected = {op['operation_id'] + '.py' for op in bundle['operations'] if op['op'] != 'delete'}
    if any(p.name not in expected for p in final_dir.iterdir()):
        raise RuntimeError('Unexpected accepted-version file')
    def accepted_path(operation, revision):
        source = Path(operation['artifact_path'])
        if digest(source) != operation['artifact_sha256']:
            raise RuntimeError('Evaluated artifact changed before commit')
        target = final_dir / (operation['operation_id'] + '.py')
        copy_verified(source, target)
        return str(target.relative_to(run_dir)), digest(target)
    return _apply(state['active_snapshot'], bundle, accepted_path)

def snapshot_sha256(records: list[dict[str, Any]]) -> str:
    return object_digest(records)


def verify_active(run_dir: Path, state: dict[str, Any]) -> None:
    records = state["active_snapshot"]
    if snapshot_sha256(records) != state["active_snapshot_sha256"]:
        raise RuntimeError("active snapshot fingerprint mismatch")
    seen = set()
    for record in records:
        if record["mechanism_id"] in seen:
            raise RuntimeError("duplicate active mechanism")
        seen.add(record["mechanism_id"])
        relative = Path(record["artifact_path"])
        if (len(relative.parts) != 3 or relative.parts[0] != "versions"
                or relative.suffix != ".py"):
            raise RuntimeError("accepted artifact path is invalid")
        path = (run_dir / relative).resolve()
        if run_dir.resolve() not in path.parents or digest(path) != record["sha256"]:
            raise RuntimeError(f"accepted artifact tampered: {record['mechanism_id']}")
    if state.get("rounds"):
        logs = state.get("operation_log", [])
        if state.get("accepted_bundle_ids") != [entry.get("bundle_id") for entry in logs]:
            raise RuntimeError("accepted bundle audit log mismatch")
        replayed: list[dict[str, Any]] = []
        for entry in logs:
            if entry.get("parent_snapshot_sha256") != snapshot_sha256(replayed):
                raise RuntimeError("operation log parent snapshot mismatch")
            for operation in entry.get("operations", []):
                positions = {item["mechanism_id"]: index
                             for index, item in enumerate(replayed)}
                if operation.get("op") not in {"add", "modify", "delete"}:
                    raise RuntimeError("operation log contains an invalid operation")
                mechanism_id = operation["mechanism_id"]
                prior = replayed[positions[mechanism_id]] if mechanism_id in positions else None
                if operation["op"] in {"modify", "delete"} and (
                        prior is None or operation.get("expected_revision") != prior["revision"]
                        or operation.get("expected_sha256") != prior["sha256"]):
                    raise RuntimeError("operation log has a stale mechanism reference")
                if operation["op"] == "delete":
                    if operation["target_layer"] != prior["target_layer"]:
                        raise RuntimeError("operation log delete layer mismatch")
                    replayed.pop(positions[mechanism_id])
                    continue
                previous = prior if operation["op"] == "modify" else None
                if operation["op"] == "modify" and previous is None:
                    raise RuntimeError("operation log modifies a missing mechanism")
                if operation["op"] == "add" and mechanism_id in positions:
                    raise RuntimeError("operation log adds a duplicate mechanism")
                revision = previous["revision"] + 1 if previous else 1
                relative = Path("versions") / entry["bundle_id"] / f"{operation['operation_id']}.py"
                artifact_path = run_dir / relative
                if (not artifact_path.is_file()
                        or digest(artifact_path) != operation.get("artifact_sha256")):
                    raise RuntimeError("operation log artifact mismatch")
                record = {"mechanism_id": mechanism_id, "revision": revision,
                          "target_layer": operation["target_layer"],
                          "artifact_path": str(relative), "sha256": digest(artifact_path)}
                if previous:
                    replayed[positions[mechanism_id]] = record
                else:
                    replayed.append(record)
            if entry.get("committed_snapshot_sha256") != snapshot_sha256(replayed):
                raise RuntimeError("operation log committed snapshot mismatch")
        if replayed != records:
            raise RuntimeError("active snapshot does not match operation log")


def _apply(active: list[dict[str, Any]], bundle: dict[str, Any],
           artifact_path: Callable[[dict[str, Any], int], tuple[str, str]]) -> list[dict[str, Any]]:
    result = [dict(item) for item in active]
    for operation in bundle["operations"]:
        positions = {item["mechanism_id"]: index for index, item in enumerate(result)}
        op = operation["op"]
        if op == "delete":
            result.pop(positions[operation["mechanism_id"]])
            continue
        previous = result[positions[operation["mechanism_id"]]] if op == "modify" else None
        revision = previous["revision"] + 1 if previous else 1
        relative, sha256 = artifact_path(operation, revision)
        record = {"mechanism_id": operation["mechanism_id"], "revision": revision,
                  "target_layer": operation["target_layer"],
                  "artifact_path": relative, "sha256": sha256}
        if previous:
            result[positions[operation["mechanism_id"]]] = record
        else:
            result.append(record)
    return result


def stage_bundle(run_dir: Path, round_number: int, bundle_path: Path,
                 state: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    verify_active(run_dir, state)
    root = bundle_path.parent
    bundle = validate_bundle(json.loads(bundle_path.read_text()), state["domain"],
                             state["active_snapshot"], state["active_snapshot_sha256"], root)
    if bundle["bundle_id"] in state.get("accepted_bundle_ids", []):
        raise ValueError("bundle_id was already accepted")
    stage_root = run_dir / "runtime" / f"staged_{round_number:02d}"
    artifacts = stage_root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    current_sources = {row["mechanism_id"]: run_dir / row["artifact_path"]
                       for row in state["active_snapshot"]}

    def staged_path(operation: dict[str, Any], revision: int) -> tuple[str, str]:
        target = artifacts / f"{operation['mechanism_id']}_r{revision}.py"
        copy_verified(operation["artifact_path"], target)
        return str(target.relative_to(stage_root)), digest(target)

    staged = _apply(state["active_snapshot"], bundle, staged_path)
    changed = {op["mechanism_id"] for op in bundle["operations"]}
    for row in staged:
        if row["mechanism_id"] not in changed:
            source = current_sources[row["mechanism_id"]]
            target = artifacts / f"{row['mechanism_id']}_r{row['revision']}.py"
            copy_verified(source, target)
            row["artifact_path"] = str(target.relative_to(stage_root))
    staged_sha = snapshot_sha256(staged)
    manifest = write_manifest(stage_root / "manifest.json", staged, stage_root, staged_sha)
    staged_by_mechanism = {row["mechanism_id"]: row for row in staged}
    for operation in bundle["operations"]:
        if operation["op"] != "delete":
            record = staged_by_mechanism[operation["mechanism_id"]]
            operation["artifact_path"] = (stage_root / record["artifact_path"]).resolve()
            operation["artifact_sha256"] = record["sha256"]
    return bundle, staged, manifest


def write_active_chain(run_dir: Path, state: dict[str, Any], name: str = "incumbent") -> Path:
    verify_active(run_dir, state)
    directory = run_dir / "runtime" / name
    directory.mkdir(parents=True, exist_ok=True)
    return write_manifest(directory / "manifest.json", state["active_snapshot"], run_dir,
                          state["active_snapshot_sha256"])
