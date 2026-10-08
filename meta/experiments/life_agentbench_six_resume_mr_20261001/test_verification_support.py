"""Exercise the inherited baseline path after a real sixth-round evaluation."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import continuation_support
import driver
import inheritance
import protocol


class VerificationSupportTests(unittest.TestCase):
    def test_imported_incumbent_is_verified_in_its_original_folder(self):
        spec = protocol.validate_protocol(check_files=False)["domains"]["alfworld"]
        summary = continuation_support.verified_phase_summary(
            protocol.HERE / "alfworld", "alfworld", "candidate_05", 1,
            protocol.task_ids(spec, 1, "train"))
        self.assertEqual(summary["successes"], 49)
        self.assertEqual(summary["split"], "train")

    def test_missing_local_replay_export_resolves_imported_train_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            summary = {"domain": "alfworld", "split": "train"}
            original = driver._export_evidence
            verified = driver._verified_phase_summary
            with patch.object(driver, "_export_evidence") as capture:
                continuation_support.install()
                driver._export_evidence(root / "workspace", "candidate_05", summary,
                                        root / "missing-local-baseline")
                self.assertEqual(capture.call_args.args[-1],
                    inheritance.base_root() / "alfworld/evals/candidate_05")
            driver._export_evidence = original
            driver._verified_phase_summary = verified

    def test_real_saved_sixth_round_replays_without_rerunning_it(self):
        folder = protocol.HERE / "alfworld"
        state_path = folder / "state.json"
        if not state_path.exists():
            self.skipTest("real sixth-round artifact not present")
        state = json.loads(state_path.read_text())
        if state["round"] != 6:
            self.skipTest("real sixth round not finished yet")
        before = protocol.digest(state_path)
        original_export = driver._export_evidence
        original_verify = driver._verified_phase_summary
        try:
            continuation_support.install()
            driver.verify_training_history(folder, state)
        finally:
            driver._export_evidence = original_export
            driver._verified_phase_summary = original_verify
        self.assertEqual(protocol.digest(state_path), before)


if __name__ == "__main__":
    unittest.main()
