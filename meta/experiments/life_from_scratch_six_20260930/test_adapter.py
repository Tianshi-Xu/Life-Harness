from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import candidate_contract
import driver
import protocol
from agentbench_chain import CompositeHarness


class ProtocolTests(unittest.TestCase):
    def test_frozen_cardinalities(self):
        manifest = protocol.validate_protocol()
        expected = {
            "airline": (30, 20, 2),
            "retail": (50, 40, 1),
            "telecom": (50, 40, 1),
            "alfworld": (50, 109, 1),
            "dbbench": (50, 300, 1),
            "webshop": (50, 200, 1),
        }
        for domain, (train, test, trials) in expected.items():
            spec = manifest["domains"][domain]
            self.assertTrue(all(len(protocol.task_ids(spec, i, "train")) == train for i in range(6)))
            self.assertEqual(len(protocol.task_ids(spec, 0, "test")), test)
            self.assertEqual(spec.get("train_trials", 1), trials)

    def test_strict_acceptance_and_rotation(self):
        base = {"accuracy": 0.5, "per_task": {"1": 1, "2": 0}}
        tie = {"accuracy": 0.5, "per_task": {"1": 1, "2": 0}}
        win = {"accuracy": 1.0, "per_task": {"1": 1, "2": 1}}
        self.assertFalse(driver.accept(tie, base))
        self.assertTrue(driver.accept(win, base))
        self.assertEqual(driver.next_batch(2, True), 2)
        self.assertEqual(driver.next_batch(2, False), 3)

    def test_test_fields_are_removed_recursively(self):
        value = candidate_contract.public_history([
            {"round": 1, "test_score": 0.9, "nested": {"test_private": 1, "train": 2}}
        ])
        self.assertEqual(value, [{"round": 1, "nested": {"train": 2}}])


class CandidateTests(unittest.TestCase):
    def test_agentbench_candidate_must_implement_only_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "candidate.py"
            path.write_text(
                "TARGET_LAYER='h2'\n"
                "class Harness:\n"
                " def h2(self, message, history): return message\n"
            )
            module = candidate_contract.load_candidate(path, "agentbench")
            self.assertEqual(module.TARGET_LAYER, "h2")
            path.write_text(
                "TARGET_LAYER='h2'\n"
                "class Harness:\n"
                " def h2(self, message, history): return message\n"
                " def h4(self, history, remaining): return []\n"
            )
            with self.assertRaisesRegex(ValueError, "only its TARGET_LAYER"):
                candidate_contract.validate_source(path, "agentbench")

    def test_forbidden_operation_mentions_do_not_trigger_text_scan(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "candidate.py"
            path.write_text(
                '"""No file, network, or subprocess access."""\n'
                "from __future__ import annotations\n"
                "from datetime import datetime\n"
                "TARGET_LAYER='h2'\n"
                "class Harness:\n"
                " def h2(self, message, history): return message\n"
            )
            candidate_contract.validate_source(path, "agentbench")
            path.write_text(
                "import subprocess\n"
                "TARGET_LAYER='h2'\n"
                "class Harness:\n"
                " def h2(self, message, history): return message\n"
            )
            with self.assertRaisesRegex(ValueError, "import not allowed"):
                candidate_contract.validate_source(path, "agentbench")

    def test_ordered_composition_and_contracts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first.py"
            second = root / "second.py"
            first.write_text(
                "TARGET_LAYER='h3'\n"
                "class Harness:\n"
                " def h3(self, tools):\n"
                "  tools[0]['function']['description']='first'; return tools\n"
            )
            second.write_text(
                "TARGET_LAYER='h2'\n"
                "class Harness:\n"
                " def h2(self, message, history):\n"
                "  message['content']='fixed'; return message\n"
            )
            harness = CompositeHarness([first, second])
            tools = [{"type": "function", "function": {
                "name": "go", "description": "old", "parameters": {}}}]
            self.assertEqual(harness.h3(tools)[0]["function"]["description"], "first")
            self.assertEqual(harness.h2({"role": "assistant", "content": "raw"}, [])["content"], "fixed")


class DriverStateMachineTests(unittest.TestCase):
    def test_six_round_batch_policy(self):
        manifest = protocol.validate_protocol(check_files=False)
        spec = manifest["domains"]["alfworld"]
        candidate_scores = {1: 0.4, 2: 0.5, 3: 0.5, 4: 0.6, 5: 0.55, 6: 0.5}
        baseline_scores = {0: 0.4, 1: 0.4, 2: 0.45, 3: 0.5}

        def summary(ids, accuracy):
            return {"accuracy": accuracy, "per_task": {str(item): 0 for item in ids}}

        def evaluate(domain, split, phase, chain, batch):
            ids = protocol.task_ids(spec, batch, split)
            if split == "test":
                return summary(ids, 0.25)
            if phase.startswith("candidate_"):
                return summary(ids, candidate_scores[int(phase.rsplit("_", 1)[1])])
            return summary(ids, baseline_scores[batch])

        def propose(domain, family, round_number, workspace, run_dir):
            candidate = run_dir / "candidates" / f"candidate_{round_number:02d}.py"
            candidate.parent.mkdir(parents=True, exist_ok=True)
            candidate.write_text(
                "TARGET_LAYER='h2'\n"
                "class Harness:\n"
                " def h2(self, message, history): return message\n"
            )
            return candidate, {"name": f"round_{round_number}", "domain": domain,
                               "target_layer": "h2", "hypothesis": "test"}

        with tempfile.TemporaryDirectory() as temporary, patch.object(driver, "HERE", Path(temporary)):
            Path(temporary, "protocol.json").write_text((HERE / "protocol.json").read_text())
            state = driver.run_domain("alfworld", evaluate=evaluate,
                                      propose_candidate=propose,
                                      do_preflight=lambda *args: {"validation": "pass"})
            self.assertEqual([row["accepted"] for row in state["rounds"]],
                             [False, True, False, True, False, False])
            self.assertEqual([row["train_batch"] for row in state["rounds"]],
                             [0, 1, 1, 2, 2, 3])
            self.assertEqual(state["batch"], 3)
            self.assertEqual(len(state["accepted"]), 2)
            history = json.loads((Path(temporary) / "alfworld/proposer/round_06/history.json").read_text())
            self.assertNotIn("test_", json.dumps(history))


if __name__ == "__main__":
    unittest.main()
