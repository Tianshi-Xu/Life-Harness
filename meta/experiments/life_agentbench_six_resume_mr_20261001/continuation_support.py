"""Verify inherited evidence paths without changing frozen evaluation sources."""
from __future__ import annotations

import json
from pathlib import Path

import agentbench_evaluate
import driver
import protocol


def verified_phase_summary(run_dir: Path, domain: str, phase_name: str, batch: int,
                           expected_ids: list):
    path = driver._phase_summary_path(domain, phase_name)
    if not path.is_file():
        raise RuntimeError(f"training evidence is missing: {phase_name}")
    summary = json.loads(path.read_text())
    if (summary.get("domain") != domain or summary.get("split") != "train"
            or summary.get("phase") != phase_name or summary.get("batch") != batch):
        raise RuntimeError(f"training evidence metadata mismatch: {phase_name}")
    driver._assert_expected_keys(summary, expected_ids)
    driver.score(summary)
    agentbench_evaluate.validate_cached_summary(path.parent, summary)
    return summary


def install():
    """Limit the amendment to training-evidence verification and replay export."""
    original_export = driver._export_evidence

    def export(workspace, phase_name, summary, output_dir=None):
        if output_dir is not None and not Path(output_dir).exists():
            output_dir = driver._phase_summary_path(summary["domain"], phase_name).parent
        return original_export(workspace, phase_name, summary, output_dir)

    driver._verified_phase_summary = verified_phase_summary
    driver._export_evidence = export


def bind_amendment():
    value = {
        "reason": "sixth-round replay must resolve the imported baseline through its read-only source directory",
        "scope": ["training evidence lookup", "training evidence export during verification replay"],
        "evaluations_reexecuted": False,
        "frozen_driver_sha256": protocol.digest(protocol.HERE / "driver.py"),
        "amendment_source_sha256": {name: protocol.digest(protocol.HERE / name)
                                   for name in ("continuation_support.py", "resume_runner.py")},
    }
    protocol.bind_fingerprint(protocol.HERE / "verification_amendment.json", value)
    return value
