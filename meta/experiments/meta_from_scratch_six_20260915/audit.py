"""Audit the compact record of the six-domain from-scratch experiment."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DOMAINS = ("airline", "retail", "telecom", "alfworld", "dbbench", "webshop")
TEST_SIZES = {
    "airline": 20,
    "retail": 40,
    "telecom": 40,
    "alfworld": 109,
    "dbbench": 300,
    "webshop": 200,
}
IDENTITY_SHA256 = "60203fe7ceb0d31c93729ad3797255470d1830bd75004394d9c1ac431763db88"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def phase_payload(domain: str, phase: str) -> tuple[Path, dict]:
    phase_dir = ROOT / domain / "evals" / phase
    for name in ("test.json", "summary.json", "val.json"):
        path = phase_dir / name
        if path.exists():
            return path, json.loads(path.read_text())
    raise AssertionError(f"missing summary for {domain}/{phase}")


def per_task_values(payload: dict) -> list[float]:
    if "per_task_trials" in payload:
        return [float(value) for trials in payload["per_task_trials"].values() for value in trials]
    per_task = payload.get("per_task", {})
    values = []
    for item in per_task.values():
        if isinstance(item, dict):
            value = item.get("reward", item.get("score", item.get("success")))
        else:
            value = item
        if isinstance(value, list):
            values.extend(float(x) for x in value)
        else:
            values.append(float(value))
    return values


def main() -> None:
    manifest = json.loads((ROOT / "manifest.json").read_text())
    assert manifest["zero_sha256"] == IDENTITY_SHA256
    assert manifest["iterations"] == 6
    assert manifest["acceptance"].startswith("strict same-batch score improvement")
    assert manifest["test_policy"].startswith("full fixed public test/std split after every round")

    frozen = json.loads((ROOT / "frozen.json").read_text())
    checked_hashes = {}
    for relative, expected in frozen.items():
        actual = sha256(ROOT / relative)
        assert actual == expected, f"frozen hash changed: {relative}"
        checked_hashes[relative] = actual

    report = {
        "ok": True,
        "identity_sha256": IDENTITY_SHA256,
        "frozen_hashes": checked_hashes,
        "domains": {},
    }
    for domain in DOMAINS:
        state = json.loads((ROOT / domain / "state.json").read_text())
        rounds = state["rounds"]
        assert state["status"] == "complete"
        assert state["from_scratch"] is True
        assert state["released_h_flags"] == []
        assert len(rounds) == 6
        assert [row["iteration"] for row in rounds] == list(range(1, 7))
        assert rounds[0]["baseline_candidate"] == "baseline_zero.py"

        tests = []
        for index, row in enumerate(rounds):
            improved = row["candidate_score"] > row["baseline_score"]
            assert row["accepted"] is improved
            assert row["released_h_flags"] == []
            assert row["test_n"] == TEST_SIZES[domain]
            assert row["test_score"] is not None
            if index:
                previous = rounds[index - 1]
                expected_batch = previous["train_batch"] + (0 if previous["accepted"] else 1)
                assert row["train_batch"] == expected_batch, (
                    f"{domain} round {row['iteration']} did not switch samples after "
                    f"round {previous['iteration']} acceptance={previous['accepted']}"
                )

            path, payload = phase_payload(domain, row["test_phase"])
            values = per_task_values(payload)
            assert len(values) == TEST_SIZES[domain], (
                f"{domain} round {row['iteration']} has {len(values)} test values"
            )
            mean = sum(values) / len(values)
            assert math.isclose(mean, row["test_score"], abs_tol=1e-12)
            tests.append({
                "round": row["iteration"],
                "path": str(path.relative_to(ROOT)),
                "n": len(values),
                "score": mean,
            })

        report["domains"][domain] = {
            "rounds": 6,
            "accepted": sum(bool(row["accepted"]) for row in rounds),
            "final_candidate": state["incumbent"],
            "final_test": rounds[-1]["test_score"],
            "best_test": max(row["test_score"] for row in rounds),
            "tests": tests,
        }

    (ROOT / "AUDIT.json").write_text(json.dumps(report, indent=2) + "\n")
    print("audit passed: 6 domains, 36 rounds, 36 complete test phases")


if __name__ == "__main__":
    main()
