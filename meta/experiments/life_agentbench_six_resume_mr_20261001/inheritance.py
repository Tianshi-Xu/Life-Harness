"""Read-only verification and explicit inheritance of the sealed five-round run."""
from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

from protocol import HERE, atomic_json, digest, object_digest, protocol, task_ids


BASE_ROUNDS = 5


def base_root() -> Path:
    return (HERE / protocol()["inherited_run"]).resolve()


def read(path: Path):
    return json.loads(path.read_text())


def verified_seed(domain: str) -> dict:
    """Validate training evidence only; never open the predecessor's test outputs."""
    root = base_root()
    cfg = protocol()
    seal_path = root / "training_seal.json"
    if digest(seal_path) != cfg["inherited_training_seal_sha256"]:
        raise RuntimeError("inherited training seal changed")
    seal = read(seal_path)
    if digest(root / "protocol.json") != seal["protocol_sha256"]:
        raise RuntimeError("inherited protocol changed")
    state_path = root / domain / "state.json"
    if digest(state_path) != seal["state_sha256"][domain]:
        raise RuntimeError("inherited state changed")
    run_config = root / domain / "run_config.json"
    if digest(run_config) != seal["run_config_sha256"][domain]:
        raise RuntimeError("inherited run configuration changed")
    for name, expected in read(run_config)["source_sha256"].items():
        if digest(root / name) != expected:
            raise RuntimeError(f"inherited source changed: {name}")
    state = read(state_path)
    if state["status"] != "complete" or state["round"] != BASE_ROUNDS or len(state["rounds"]) != BASE_ROUNDS:
        raise RuntimeError("predecessor did not finish five rounds")
    historical = read(root.parent / "meta_from_scratch_six_20260915/manifest.json")
    spec = historical["domains"][domain]
    active = []
    batch = 0
    incumbent = "batch_00_incumbent"

    def verified_summary(phase: str, expected_batch: int):
        folder = root / domain / "evals" / phase
        config = read(folder / "config.json")
        body = {k: v for k, v in config.items() if k != "config_sha256"}
        if object_digest(body) != config["config_sha256"]:
            raise RuntimeError("inherited evaluation configuration hash mismatch")
        if (config["split"] != "train" or config["domain"] != domain
                or config["batch"] != expected_batch
                or config["ids"] != task_ids(spec, expected_batch, "train")
                or config["protocol_hash"] != seal["protocol_sha256"]):
            raise RuntimeError("inherited training tasks or metadata changed")
        for name, expected in config["source_sha256"].items():
            if digest(root / name) != expected:
                raise RuntimeError("inherited evaluated source changed")
        outcomes = {}
        for index in config["ids"]:
            episode = read(folder / f"episode_{index}.json")
            if (episode["index"] != index or episode["config_sha256"] != config["config_sha256"]
                    or episode["success"] not in (0, 1) or episode["infrastructure_error"]):
                raise RuntimeError("inherited training episode is invalid")
            outcomes[str(index)] = episode["success"]
        summary = read(folder / "summary.json")
        if (any(summary.get(k) != v for k, v in config.items())
                or summary["per_task"] != outcomes or summary["n"] != len(outcomes)
                or summary["successes"] != sum(outcomes.values())
                or summary["accuracy"] != sum(outcomes.values()) / len(outcomes)
                or summary["infra_errors"] != 0):
            raise RuntimeError("inherited summary does not match episodes")
        return summary

    def identity(rows):
        return [{k: row[k] for k in ("mechanism_id", "revision", "target_layer", "sha256")}
                for row in rows]

    for number, row in enumerate(state["rounds"], 1):
        if row["round"] != number or row["train_batch"] != batch:
            raise RuntimeError("inherited round order or batch mismatch")
        baseline = verified_summary(incumbent, batch)
        candidate = verified_summary(f"candidate_{number:02d}", batch)
        if (baseline["accuracy"] != row["incumbent_score"]
                or candidate["accuracy"] != row["candidate_score"]
                or (candidate["accuracy"] > baseline["accuracy"]) != row["accepted"]):
            raise RuntimeError("inherited acceptance mismatch")
        cache = root / domain / "proposals" / f"round_{number:02d}"
        bundle_path = cache / "bundle.json"
        bundle = read(bundle_path)
        metadata = read(cache / "cache.json")
        if (digest(bundle_path) != row["bundle_sha256"]
                or bundle["base_snapshot_sha256"] != object_digest(active)
                or bundle["domain"] != domain or bundle["bundle_id"] != row["bundle_id"]
                or metadata["input_sha256"] != object_digest(read(root / domain / "proposals" / f"inputs_{number:02d}.json"))
                or metadata["outputs"] != {str(p.relative_to(cache)): digest(p)
                                           for p in sorted(cache.rglob("*"))
                                           if p.is_file() and p.name != "cache.json"}):
            raise RuntimeError("inherited proposal cache or parent changed")
        manifest_path = root / domain / "runtime" / f"staged_{number:02d}" / "manifest.json"
        manifest = read(manifest_path)
        if (manifest["manifest_sha256"] != object_digest({k: v for k, v in manifest.items() if k != "manifest_sha256"})
                or manifest["snapshot_sha256"] != object_digest(manifest["records"])
                or candidate["snapshot_sha256"] != manifest["snapshot_sha256"]
                or candidate["chain_hashes"]["manifest"] != digest(manifest_path)):
            raise RuntimeError("inherited staged manifest changed")
        staged = {r["mechanism_id"]: r for r in manifest["records"]}
        by_id = {r["mechanism_id"]: r for r in active}
        changed = {}
        for op in bundle["operations"]:
            mid = op["mechanism_id"]
            prior = by_id.get(mid)
            if op["op"] in {"modify", "delete"}:
                if not prior or op["expected_revision"] != prior["revision"] or op["expected_sha256"] != prior["sha256"]:
                    raise RuntimeError("inherited operation revision changed")
            elif op["op"] != "add" or prior:
                raise RuntimeError("inherited operation is invalid")
            if op["op"] == "delete":
                by_id.pop(mid)
                continue
            new = staged[mid]
            if (digest(cache / op["artifact"]) != new["sha256"]
                    or new["revision"] != (prior["revision"] + 1 if prior else 1)
                    or new["target_layer"] != op["target_layer"]):
                raise RuntimeError("inherited staged artifact changed")
            by_id[mid] = new
            changed[mid] = op
        if identity(list(by_id.values())) != identity(manifest["records"]):
            raise RuntimeError("inherited staged operations differ")
        if row["accepted"]:
            previous = {r["mechanism_id"]: r for r in active}
            active = []
            for record in manifest["records"]:
                mid = record["mechanism_id"]
                value = dict(record)
                if mid in changed:
                    value["artifact_path"] = str(Path("versions") / bundle["bundle_id"] / f"{changed[mid]['operation_id']}.py")
                else:
                    value = previous[mid]
                if digest(root / domain / value["artifact_path"]) != value["sha256"]:
                    raise RuntimeError("inherited committed artifact changed")
                active.append(value)
            incumbent = f"candidate_{number:02d}"
        else:
            batch = min(batch + 1, 5)
            incumbent = f"batch_{batch:02d}_incumbent"
        if row["next_batch"] != batch or row["active_snapshot_sha256"] != object_digest(active):
            raise RuntimeError("inherited state transition mismatch")
    if active != state["active_snapshot"] or state["batch"] != batch:
        raise RuntimeError("inherited final snapshot mismatch")
    return state


def initialize(domain: str, run_dir: Path) -> dict:
    state = copy.deepcopy(verified_seed(domain))
    origin = base_root() / domain
    # Copy immutable accepted artifacts, never symlink a writable versions directory.
    for entry in state["operation_log"]:
        for op in entry["operations"]:
            if op["op"] == "delete":
                continue
            relative = Path("versions") / entry["bundle_id"] / f"{op['operation_id']}.py"
            source, target = origin / relative, run_dir / relative
            if digest(source) != op["artifact_sha256"]:
                raise RuntimeError("inherited operation-log artifact changed")
            if target.exists():
                if digest(target) != digest(source):
                    raise RuntimeError("copied inherited artifact changed")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
    certificate = {"source_run": str(base_root()), "training_seal_sha256": protocol()["inherited_training_seal_sha256"],
                   "source_state_sha256": digest(origin / "state.json"), "inherited_rounds": BASE_ROUNDS,
                   "prior_test_exposure": True, "new_proposer": "ModelRouter deepseek-v4-flash"}
    path = run_dir / "inheritance.json"
    if path.exists() and read(path) != certificate:
        raise RuntimeError("inheritance certificate changed")
    if not path.exists():
        atomic_json(path, certificate)
    state["status"] = "running"
    return state


def phase_summary_path(domain: str, phase_name: str) -> Path:
    local = HERE / domain / "evals" / phase_name / "summary.json"
    if local.exists():
        return local
    seed = verified_seed(domain)
    allowed = {f"candidate_{number:02d}" for number in range(1, BASE_ROUNDS + 1)}
    allowed.update(f"batch_{row['train_batch']:02d}_incumbent" for row in seed["rounds"])
    if phase_name in allowed:
        inherited = base_root() / domain / "evals" / phase_name / "summary.json"
        if inherited.exists():
            return inherited
    return local
