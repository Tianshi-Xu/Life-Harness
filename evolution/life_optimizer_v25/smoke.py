"""Replay an unchanged incumbent on real TRAIN traces, without any model call."""
import argparse
import json
from pathlib import Path

from .profile import check_candidate, load_cases, prepare, write_json


def smoke(seed_run, eval_dir, output):
    output = Path(output)
    workspace = prepare(seed_run, eval_dir, output / "inputs")
    inventory = json.loads((workspace / "inventory.json").read_text())
    cases = load_cases(workspace)
    original = next(r for r in inventory["records"] if r["target_layer"] == "h2")
    candidate = output / "unchanged_incumbent_smoke"
    (candidate / "artifacts").mkdir(parents=True)
    (candidate / "artifacts/unchanged.py").write_text((workspace / original["visible_path"]).read_text())
    write_json(candidate / "bundle.json", {"schema_version": 2, "bundle_id": "offline_smoke_do_not_commit",
        "domain": inventory["domain"], "base_snapshot_sha256": inventory["snapshot_sha256"],
        "hypothesis": "Machinery smoke ONLY: unchanged incumbent, not an optimization proposal; never commit.",
        "operations": [{"operation_id": "unchanged", "mechanism_id": original["mechanism_id"], "op": "modify",
            "target_layer": "h2", "hypothesis": "Byte-identical content; no policy or score change.",
            "expected_revision": original["revision"], "expected_sha256": original["sha256"],
            "artifact": "artifacts/unchanged.py"}]})
    write_json(candidate / "diagnosis.json", {"schema_version": 1, "cases": [{
        "case_id": c["case_id"], "first_error_message_index": None,
        "observation": "Recorded public TRAIN trace used only for hook compatibility replay.",
        "root_cause": "Not diagnosed: this is a host machinery smoke, without an optimizer invocation.",
        "confidence": "low", "public_trigger": "No new behavioral trigger; incumbent code is unchanged.",
        "operation_ids": ["unchanged"], "regression_risk": "No claimed improvement or repair; replay is not an environment rollout."
        } for c in cases], "rules": [{"rule_id": "smoke_schema", "invariant": "Preserve assistant envelope on benign inputs.",
            "layer_contracts": {layer: "Unchanged incumbent content; machinery compatibility check only." for layer in ("h2", "h3", "h4", "h5")},
            "operation_ids": ["unchanged"], "test_ids": ["benign_a", "benign_b"]}], "shared_state_contract": []})
    write_json(candidate / "checks.json", {"schema_version": 1, "tests": [{
        "test_id": name, "rule_id": "smoke_schema", "kind": kind, "context": cases[0]["context"],
        "steps": [{"hook": "h2", "args": [{"role": "assistant", "content": "offline smoke " + name}, []],
                   "expected_subset": {"role": "assistant"}}]
        } for name, kind in (("benign_a", "trigger"), ("benign_b", "counterexample"))]})
    report = check_candidate(workspace, candidate)
    small = {k: v for k, v in report.items() if k not in {"bundle", "diagnosis", "checks"}}
    write_json(output / "verification.json", {**small, "optimizer_requests": 0, "solver_requests": 0,
        "training_episodes_run": 0, "scope_note": "UNCHANGED CONTENT smoke only; no training round, acceptance, environment rollout or performance claim."})
    return small


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-run", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = smoke(args.seed_run, args.eval_dir, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)
