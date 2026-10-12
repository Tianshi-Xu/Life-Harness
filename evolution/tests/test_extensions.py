"""End-to-end custom benchmark evolution, native scoring and cost boundaries."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from evolution import harness_hooks as hooks
from evolution.extensions import ExtensionRunner, evaluate
from evolution.extensions.examples import ToyAdapter, ToyEpisode, offline_proposer
from evolution.extensions.records import Ledger
from evolution.extensions.runner import strict_improvement
from evolution.extensions.runtime import case_id
from evolution.life_evolve import profile


class FailedTestAdapter(ToyAdapter):
    def open_episode(self, task_id, split):
        episode = super().open_episode(task_id, split)
        if split == "test":
            episode.grade = lambda: False
        return episode


class BrokenAdapter(ToyAdapter):
    def open_episode(self, task_id, split):
        episode = super().open_episode(task_id, split)
        def fail(*args):
            raise ConnectionError("fixture connection failure")
        episode.complete = fail
        return episode


class JudgeEpisode(ToyEpisode):
    def bind_request_recorder(self, record):
        self.record = record

    def grade(self):
        response = self.record("judge", lambda: {"private_verdict": self.accepted,
            "usage": {"prompt_tokens": 8, "completion_tokens": 1}})
        return response["private_verdict"]


class JudgeAdapter(ToyAdapter):
    def open_episode(self, task_id, split):
        return JudgeEpisode(self.tasks[split][task_id])


class DriftingAdapter(ToyAdapter):
    changed = False

    def fingerprint(self):
        return {**super().fingerprint(), "changed": self.changed}

    def open_episode(self, task_id, split):
        self.changed = True
        return super().open_episode(task_id, split)


class PrivateContextAdapter(ToyAdapter):
    def open_episode(self, task_id, split):
        episode = super().open_episode(task_id, split)
        episode.setup.public_context["hidden_answer"] = "private_marker_never_export"
        return episode


class ExtensionTests(unittest.TestCase):
    def test_configuration_drift_stops_before_optimization(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = ExtensionRunner(DriftingAdapter(), Path(tmp) / "run", rounds=1, offline_proposer=offline_proposer)
            with self.assertRaises(RuntimeError):
                runner.run()
            self.assertEqual(json.loads((runner.root / "status.json").read_text())["phase"], "stopped")
            self.assertFalse((runner.root / "optimizer_inputs").exists())

    def test_undeclared_context_does_not_reach_solver_or_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = ExtensionRunner(PrivateContextAdapter(), Path(tmp) / "run", rounds=1, offline_proposer=offline_proposer)
            with self.assertRaises(RuntimeError):
                runner.run()
            self.assertEqual(runner.ledger.summary()["calls"], 0)
            for path in runner.root.rglob("*.json"):
                self.assertNotIn("private_marker_never_export", path.read_text())

    def test_episode_state_is_fresh_for_each_native_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "counter.py"
            source.write_text('''TARGET_LAYER = "h5"
class Harness:
    def bind_shared_state(self, state):
        self.state = state
    def h5(self, messages):
        self.state["count"] = self.state.get("count", 0) + 1
        return [str(self.state["count"])]
''')
            records = [{"mechanism_id": "counter", "revision": 1, "target_layer": "h5",
                        "artifact_path": source.name, "sha256": hooks.digest(source)}]
            body = {"schema_version": 2, "artifact_root": str(root), "records": records,
                    "snapshot_sha256": hooks.object_digest(records)}
            (root / "manifest.json").write_text(json.dumps({**body, "manifest_sha256": hooks.object_digest(body)}))
            evaluate(ToyAdapter(), root / "manifest.json", root / "score")
            for path in (root / "score").glob("episode_*.json"):
                row = json.loads(path.read_text())
                self.assertEqual([r["hints"] for r in row["harness_trace"] if r["layer"] == "h5"], [["1"]])

    def test_unsafe_candidate_is_never_natively_evaluated(self):
        def unsafe(workspace, output, number):
            path = offline_proposer(workspace, output, number)
            artifact = output / "artifacts/integer_quantity.py"
            artifact.write_text("import os\n" + artifact.read_text())
            return path
        with tempfile.TemporaryDirectory() as tmp:
            runner = ExtensionRunner(ToyAdapter(), Path(tmp) / "run", rounds=1, offline_proposer=unsafe)
            with self.assertRaises(ValueError):
                runner.run()
            self.assertFalse((runner.work / "evals/candidate_01").exists())
            self.assertEqual(runner.state["round"], 0)

    def test_real_demo_accepts_improvement_rejects_tie_and_seals_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            runner = ExtensionRunner(ToyAdapter(), root, rounds=2, offline_proposer=offline_proposer)
            runner.run()
            state = json.loads((runner.work / "state.json").read_text())
            self.assertEqual([row["accepted"] for row in state["rounds"]], [True, False])
            final = json.loads((root / "FINAL_RESULTS.json").read_text())
            self.assertEqual((final["life"]["n"], final["life"]["successes"]), (2, 2))
            self.assertFalse(final["used_for_selection"])
            seal = json.loads((root / "training_seal.json").read_text())
            self.assertEqual(seal["state_sha256"], hooks.digest(runner.work / "state.json"))
            usage = json.loads((root / "TOKEN_USAGE.json").read_text())
            self.assertEqual(usage["optimizer_backend"], "offline_no_api")
            self.assertEqual(usage["optimizer"]["calls"], 0)
            self.assertEqual(usage["solver"]["local_no_api_calls"], 10)
            for path in (root / "optimizer_inputs").rglob("*"):
                if path.is_file():
                    self.assertNotIn("test_seven", path.read_text())
                    self.assertNotIn("test_nine", path.read_text())
            # Separate process restores the custom schema from the frozen workspace.
            subprocess.run([sys.executable, "-m", "evolution.life_evolve.profile", "check",
                "--workspace", str(root / "optimizer_inputs/round_01"),
                "--candidate", str(runner.work / "qualified/round_01")], check=True, capture_output=True)
            with self.assertRaises(FileExistsError):
                runner.run()
            score = evaluate(ToyAdapter(), final["manifest"], Path(tmp) / "standalone")
            self.assertEqual(score["successes"], 2)
            artifact = runner.work / state["active_snapshot"][0]["artifact_path"]
            artifact.write_text(artifact.read_text() + "\n# changed\n")
            with self.assertRaises(ValueError):
                evaluate(ToyAdapter(), final["manifest"], Path(tmp) / "tampered")

    def test_failed_test_never_changes_train_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = ExtensionRunner(FailedTestAdapter(), Path(tmp) / "run", rounds=1, offline_proposer=offline_proposer)
            runner.run()
            self.assertTrue(runner.state["rounds"][0]["accepted"])
            self.assertEqual(json.loads((runner.root / "FINAL_RESULTS.json").read_text())["life"]["successes"], 0)

    def test_infra_failure_keeps_denominator_and_unknown_usage_without_proposal(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = ExtensionRunner(BrokenAdapter(), Path(tmp) / "run", rounds=1, offline_proposer=offline_proposer)
            with self.assertRaises(RuntimeError):
                runner.run()
            summary = json.loads((runner.work / "evals/incumbent_01/summary.json").read_text())
            self.assertEqual((summary["n"], summary["successes"], summary["infra_errors"]), (2, 0, 2))
            self.assertEqual(summary["solver_usage"]["unknown_usage_calls"], 2)
            self.assertFalse(summary["solver_usage"]["accounting_complete"])
            self.assertFalse((runner.root / "optimizer_inputs").exists())

    def test_native_judge_usage_is_separate_and_private_response_is_not_logged(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = ExtensionRunner(JudgeAdapter(), Path(tmp) / "run", rounds=1, offline_proposer=offline_proposer)
            runner.run()
            usage = json.loads((runner.root / "TOKEN_USAGE.json").read_text())
            self.assertEqual(usage["optimizer"]["total_tokens"], 0)
            self.assertEqual(usage["other_model_roles"]["judge"]["total_tokens"], 54)
            self.assertNotIn("private_verdict", (runner.root / "requests.jsonl").read_text())

    def test_overlapping_splits_and_builtin_schema_overrides_are_rejected(self):
        adapter = ToyAdapter()
        adapter.tasks = {"train": {"same": 2}, "test": {"same": 3}}
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            ExtensionRunner(adapter, Path(tmp) / "run", offline_proposer=offline_proposer)
        with self.assertRaises(ValueError):
            hooks.register_domain("os", ["hidden_answer"])

    def test_incomplete_batch_cannot_select_a_candidate(self):
        row = {"n": 2, "complete": True, "successes": 1, "accuracy": 0.5,
               "per_task": {case_id("a"): 1}}
        with self.assertRaises(ValueError):
            strict_improvement(row, row, ["a", "b"])

    def test_pending_and_unknown_optimizer_calls_are_not_counted_as_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Ledger(Path(tmp) / "requests.jsonl")
            ledger.start("life_optimizer")
            unknown = ledger.start("life_optimizer")
            ledger.finish(unknown, None)
            known = ledger.start("life_optimizer")
            ledger.finish(known, {"prompt_tokens": 7, "completion_tokens": 3,
                                  "prompt_tokens_details": {"cached_tokens": 4}})
            usage = ledger.summary("life_optimizer")
            self.assertEqual((usage["pending_calls"], usage["unknown_usage_calls"], usage["total_tokens"]), (1, 1, 10))
            self.assertFalse(usage["accounting_complete"])
            self.assertIn('"cached_tokens": 4', ledger.path.read_text())

    def test_model_proposer_reuses_current_evolve_and_meters_requests(self):
        source = '''from json import loads, dumps
TARGET_LAYER = "h2"
class Harness:
    def h2(self, message, history):
        for call in message.get("tool_calls", []):
            arguments = loads(call["function"]["arguments"])
            arguments["quantity"] = int(arguments["quantity"])
            call["function"]["arguments"] = dumps(arguments)
        return message
'''
        requests = []
        def completion(messages, tools):
            requests.append(messages)
            if len(requests) == 1:
                bundle = {"schema_version": 2, "hypothesis": "Repair numeric representation.", "operations": [{
                    "operation_id": "repair", "mechanism_id": "repair", "op": "add", "target_layer": "h2",
                    "hypothesis": "Native tool expects an integer.", "artifact": "artifacts/repair.py"}]}
                calls = [{"id": str(i), "type": "function", "function": {"name": "write_file", "arguments": json.dumps(args)}}
                         for i, args in enumerate([{"path": "bundle.json", "content": json.dumps(bundle)},
                                                   {"path": "artifacts/repair.py", "content": source}])]
                message = {"role": "assistant", "tool_calls": calls}
            else:
                message = {"role": "assistant", "content": "Candidate ready."}
            return {"choices": [{"message": message, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}}
        counter = SimpleNamespace(count=lambda *args: 1, encode_messages=lambda *args, **kwargs: "prompt")
        with tempfile.TemporaryDirectory() as tmp, patch.object(profile, "flash_counter", return_value=counter):
            runner = ExtensionRunner(ToyAdapter(), Path(tmp) / "run", rounds=1, completion=completion)
            runner.run()
            self.assertTrue(runner.state["rounds"][0]["accepted"])
            self.assertEqual(runner.ledger.summary("life_optimizer")["total_tokens"], 12 * len(requests))


if __name__ == "__main__":
    unittest.main()
