"""Offline regression checks for exactly three additional rounds."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import driver
import inheritance
import protocol
import agentbench_evaluate


class ContinuationTests(unittest.TestCase):
    def test_scope_and_seed(self):
        self.assertEqual(protocol.DOMAINS, ("dbbench", "webshop"))
        self.assertEqual(protocol.ROUNDS, 9)
        for domain in protocol.DOMAINS:
            seed = inheritance.verified_seed(domain)
            self.assertEqual(seed["round"], 6)
            self.assertEqual(len(seed["rounds"]), 6)

    def test_seal_tampering_rejected(self):
        cfg = dict(protocol.protocol(), inherited_training_seal_sha256="0" * 64)
        with patch.object(inheritance, "protocol", return_value=cfg):
            with self.assertRaisesRegex(RuntimeError, "seal changed"):
                inheritance.verified_seed("dbbench")

    def test_test_evidence_not_inherited(self):
        self.assertEqual(inheritance.phase_summary_path("dbbench", "final_test"),
                         protocol.HERE / "dbbench/evals/final_test/summary.json")
        path = inheritance.phase_summary_path("dbbench", "candidate_06")
        self.assertEqual(path.parent.parent.parent, inheritance.base_root() / "dbbench")

    def test_explicit_missing_export_uses_inherited_train(self):
        source = inheritance.phase_summary_path("dbbench", "candidate_06")
        summary = json.loads(source.read_text())
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            driver._export_evidence(root / "workspace", "candidate_06", summary, root / "missing")
            rows = json.loads((root / "workspace/evidence/candidate_06/public_trajectories.json").read_text())
            self.assertEqual(len(rows), 50)

    def test_test_gate_closed_before_training(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(RuntimeError, "before training seal"):
                agentbench_evaluate.validate_test_gate(Path(folder))

    def test_exactly_three_new_rounds_and_no_test(self):
        spec = protocol.validate_protocol(check_files=False)["domains"]["dbbench"]
        evaluated, proposed = [], []
        seed = inheritance.verified_seed("dbbench")
        before = protocol.digest(inheritance.base_root() / "dbbench/state.json")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            shutil.copy2(protocol.HERE / "protocol.json", root / "protocol.json")

            def evaluate(domain, split, phase, chain, batch):
                self.assertEqual(split, "train")
                evaluated.append(phase)
                ids = protocol.task_ids(spec, batch, "train")
                values = {str(index): int(i < 10) for i, index in enumerate(ids)}
                return {"domain": domain, "split": split, "phase": phase, "batch": batch,
                        "accuracy": 0.2, "per_task": values}

            def propose(domain, number, workspace, run_dir, state):
                proposed.append(number)
                cache = run_dir / "proposals" / f"round_{number:02d}"
                (cache / "artifacts").mkdir(parents=True)
                artifact = cache / "artifacts/extra.py"
                artifact.write_text("TARGET_LAYER='h5'\nclass Harness:\n def h5(self, messages):\n  return None\n")
                path = cache / "bundle.json"
                protocol.atomic_json(path, {"schema_version": 2, "domain": domain,
                    "bundle_id": f"extra_{number}", "hypothesis": "additional mechanism",
                    "base_snapshot_sha256": state["active_snapshot_sha256"],
                    "operations": [{"operation_id": "extra", "mechanism_id": f"extra_{number}",
                        "op": "add", "target_layer": "h5", "artifact": "artifacts/extra.py",
                        "hypothesis": "additional mechanism"}]})
                return path

            with patch.object(driver, "HERE", root):
                result = driver.run_domain("dbbench", evaluate, propose, lambda path: {"ok": True})
            self.assertEqual(proposed, [7, 8, 9])
            self.assertEqual(result["round"], 9)
            self.assertEqual(result["rounds"][:6], seed["rounds"])
            self.assertEqual(result["active_snapshot"], seed["active_snapshot"])
            self.assertEqual([p for p in evaluated if p.startswith("candidate")],
                             ["candidate_07", "candidate_08", "candidate_09"])
        self.assertEqual(protocol.digest(inheritance.base_root() / "dbbench/state.json"), before)


if __name__ == "__main__":
    unittest.main()
