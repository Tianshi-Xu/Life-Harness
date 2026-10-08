"""Offline checks of inheritance and the single additional training round."""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import driver
import inheritance
import protocol
import status


class ContinuationTests(unittest.TestCase):
    def test_all_three_seeds_are_sealed_and_training_only(self):
        self.assertEqual(protocol.ROUNDS, 6)
        for domain in protocol.DOMAINS:
            seed = inheritance.verified_seed(domain)
            self.assertEqual(seed["round"], 5)
            self.assertEqual(len(seed["rounds"]), 5)
            if seed["baseline_summary"]:
                self.assertEqual(seed["baseline_summary"]["split"], "train")
        self.assertTrue(protocol.protocol()["prior_test_exposure"])

    def test_inheritance_mismatch_is_rejected(self):
        cfg = dict(protocol.protocol(), inherited_training_seal_sha256="0" * 64)
        with patch.object(inheritance, "protocol", return_value=cfg):
            with self.assertRaisesRegex(RuntimeError, "seal changed"):
                inheritance.verified_seed("alfworld")

    def test_training_summary_fallback_cannot_import_final_test(self):
        inherited = inheritance.phase_summary_path("alfworld", "candidate_05")
        self.assertEqual(inherited.parent.parent.parent, inheritance.base_root() / "alfworld")
        final = inheritance.phase_summary_path("alfworld", "final_test")
        self.assertEqual(final, protocol.HERE / "alfworld/evals/final_test/summary.json")

    def test_copied_artifact_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            seed = inheritance.initialize("alfworld", root)
            artifact = root / seed["active_snapshot"][0]["artifact_path"]
            artifact.write_text("changed")
            with self.assertRaisesRegex(RuntimeError, "copied inherited artifact changed"):
                inheritance.initialize("alfworld", root)

    def _run_sixth(self, domain: str, improved: bool):
        seed = inheritance.verified_seed(domain)
        original_state_hash = protocol.digest(inheritance.base_root() / domain / "state.json")
        historical = protocol.validate_protocol(check_files=False)
        ids = protocol.task_ids(historical["domains"][domain], seed["batch"], "train")
        evaluations = []
        proposals = []
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            shutil.copy2(protocol.HERE / "protocol.json", root / "protocol.json")

            def summary(phase, successes):
                values = {str(index): int(i < successes) for i, index in enumerate(ids)}
                return {"domain": domain, "split": "train", "batch": seed["batch"],
                        "phase": phase, "accuracy": successes / len(ids), "per_task": values}

            baseline_successes = seed["baseline_summary"]["successes"] if seed["baseline_summary"] else 10

            def evaluate(given_domain, split, phase, chain, batch):
                self.assertEqual((given_domain, split, batch), (domain, "train", seed["batch"]))
                evaluations.append(phase)
                count = baseline_successes + int(improved and phase == "candidate_06")
                return summary(phase, count)

            def propose(given_domain, number, workspace, run_dir, state):
                self.assertEqual(number, 6)
                self.assertEqual(state["rounds"], seed["rounds"])
                proposals.append(number)
                output = run_dir / "proposals/round_06"
                (output / "artifacts").mkdir(parents=True)
                (output / "artifacts/extra.py").write_text("TARGET_LAYER='h5'\nclass Harness:\n def h5(self, messages):\n  return []\n")
                path = output / "bundle.json"
                protocol.atomic_json(path, {"schema_version": 2, "domain": domain,
                    "bundle_id": "extra_sixth", "base_snapshot_sha256": state["active_snapshot_sha256"],
                    "hypothesis": "extra sixth round", "operations": [{"operation_id": "add_extra",
                    "mechanism_id": "extra_sixth", "op": "add", "target_layer": "h5",
                    "hypothesis": "additional mechanism", "artifact": "artifacts/extra.py"}]})
                return path

            with patch.object(driver, "HERE", root):
                result = driver.run_domain(domain, evaluate, propose, lambda chain: {"validation": "pass"})
            self.assertEqual(result["round"], 6)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["rounds"][:5], seed["rounds"])
            self.assertEqual(proposals, [6])
            self.assertEqual(evaluations, (["candidate_06"] if seed["baseline_summary"] else
                                          [f"batch_{seed['batch']:02d}_incumbent", "candidate_06"]))
            self.assertEqual(result["rounds"][-1]["accepted"], improved)
            if not improved:
                self.assertEqual(result["active_snapshot"], seed["active_snapshot"])
                self.assertEqual(result["batch"], seed["batch"] + 1)
            self.assertFalse((root / "training_seal.json").exists())
            self.assertFalse((root / domain / "evals/final_test").exists())
        self.assertEqual(original_state_hash, protocol.digest(inheritance.base_root() / domain / "state.json"))

    def test_only_sixth_round_runs_in_each_domain(self):
        for domain in protocol.DOMAINS:
            with self.subTest(domain=domain):
                self._run_sixth(domain, True)

    def test_sixth_round_tie_retains_five_round_snapshot(self):
        self._run_sixth("alfworld", False)

    def test_seal_refuses_incomplete_six_round_states(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            protocol.atomic_json(root / "alfworld/state.json", inheritance.verified_seed("alfworld"))
            with self.assertRaisesRegex(RuntimeError, "completed six rounds"):
                status.seal_training(root)


if __name__ == "__main__":
    unittest.main()
