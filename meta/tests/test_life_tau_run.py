"""Offline controller tests; native transport/optimizer remain separately covered."""
import hashlib
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from meta import harness_hooks as hooks
from meta.life_evolve import tau_run as runner
from meta.tau_comparison.common import read, save


class TauRunTests(unittest.TestCase):
    def test_stop_is_round_aware_and_requires_material_gap(self):
        row = lambda n, gap: {"round": n, "gap_pp": gap}
        self.assertIsNone(runner.stop_reason([]))
        self.assertIsNone(runner.stop_reason([row(1, 30)]))
        self.assertIsNone(runner.stop_reason([row(1, 19.9), row(2, 29.9)]))
        self.assertIsNotNone(runner.stop_reason([row(1, 20), row(2, 20)]))
        self.assertIsNotNone(runner.stop_reason([row(2, 0), row(3, 30)]))
        self.assertIsNone(runner.stop_reason([row(2, 20), row(3, 10)]))
        self.assertIsNone(runner.stop_reason([row(3, -10), row(4, -2)]))
        self.assertIn("unpaired", runner.stop_reason([row(3, 30)], same_batch=False))

    def test_usage_excludes_solver_simulator_other_experiments_and_rounds(self):
        def group(role="life_optimizer", experiment="run", number=1, **extra):
            return {"role": role, "experiment": experiment, "domain": "telecom", "round": number,
                    "input_tokens": 10, "output_tokens": 5, "total_tokens": 15, **extra}
        summary = {"groups": {"a": group(), "b": group(number=2), "c": group("user_simulator", pending_calls=1),
            "d": group(experiment="other"), "e": group("evolve")}}
        self.assertEqual(runner.optimizer_usage(summary, "run")["total_tokens"], 30)
        self.assertEqual(runner.optimizer_usage(summary, "run", 1)["total_tokens"], 15)
        self.assertTrue(runner.optimizer_usage(summary, "run")["inference_accounting_complete"])
        summary["groups"]["a"]["pending_calls"] = 1
        self.assertFalse(runner.optimizer_usage(summary, "run", 1)["inference_accounting_complete"])
        summary["groups"]["a"].update(pending_calls=0, unknown_usage_calls=1, rate_limit_rejections_without_usage=1)
        value=runner.optimizer_usage(summary, "run", 1)
        self.assertTrue(value["inference_accounting_complete"])
        self.assertFalse(value["accounting_complete"])

    def test_inherited_paid_cost_counts_once_only_in_first_round_and_total(self):
        prior = dict(attempts=52, known_usage_calls=52, input_tokens=8196321,
                     output_tokens=169255, total_tokens=8365576)
        summary = {"groups": {"new": dict(experiment="run", domain="telecom", round=2,
            role="life_optimizer", attempts=1, known_usage_calls=1,
            input_tokens=10, output_tokens=5, total_tokens=15)}}
        self.assertEqual(runner.optimizer_usage(summary, "run", 1, prior)["total_tokens"], 8365576)
        self.assertEqual(runner.optimizer_usage(summary, "run", 2, prior)["total_tokens"], 15)
        total = runner.optimizer_usage(summary, "run", inherited=prior)
        self.assertEqual((total["total_tokens"], total["attempts"]), (8365591, 53))
        self.assertTrue(total["accounting_complete"])

    def test_two_rounds_stage_commit_and_pair_with_exact_ahe_batch(self):
        self._two_rounds(accuracy_only=False)

    def test_accuracy_only_rounds_never_launch_extra_comparison(self):
        self._two_rounds(accuracy_only=True)

    def _two_rounds(self, accuracy_only):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ids = [f"task_{n}" for n in range(50)]
            state = {"domain": "telecom", "round": 0, "rounds": [], "prior_test_exposure": True,
                     "active_snapshot": [], "active_snapshot_sha256": hooks.object_digest([]),
                     "accepted_bundle_ids": [], "operation_log": []}
            save(root / "telecom/state.json", state)
            protocol = {"spec": {"batches": [{"task_ids": ids} for _ in range(6)]},
                "accuracy_only": accuracy_only,
                "ahe_reference": [{"round": n, "measured_batch": min(n, 5), "candidate_train_accuracy": .5}
                                  for n in range(1, 7)]}
            save(root / "protocol.json", protocol)
            save(root / "frozen_sources.json", {})
            run = runner.Run(root)
            run.broker = SimpleNamespace(ledger=SimpleNamespace(summary=lambda: {"groups": {
                "current": {"experiment": root.name, "domain": "telecom", "role": "life_optimizer",
                    "round": run.state["round"] + 1, "total_tokens": 15}}}))
            evaluated = []

            def evaluation(split, name, manifest, tasks, number):
                _, records = hooks.manifest_records(manifest)
                evaluated.append((split, name, tasks, len(records)))
                successes = 30 if name.startswith("candidate") else 35 if name.startswith("comparison") else 20
                return {"accuracy": successes / 50, "successes": successes, "infra_errors": 0,
                        "split": "train", "domain": "telecom", "n": 50,
                        "per_task": {"native_" + hashlib.sha256(t.encode()).hexdigest(): int(i < successes)
                                     for i, t in enumerate(tasks)}}, root / "public"

            def proposal(workspace, output, **kwargs):
                n = run.state["round"] + 1
                prior = run.state["active_snapshot"]
                op = {"operation_id": f"change_{n}", "mechanism_id": "mechanism", "op": "modify" if prior else "add",
                      "target_layer": "h2", "hypothesis": "synthetic", "artifact": "artifacts/h2.py"}
                if prior:
                    op.update(expected_revision=prior[0]["revision"], expected_sha256=prior[0]["sha256"])
                save(output / "bundle.json", {"schema_version": 2, "bundle_id": f"bundle_{n}", "domain": "telecom",
                    "base_snapshot_sha256": run.state["active_snapshot_sha256"], "hypothesis": "synthetic", "operations": [op]})
                (output / "artifacts").mkdir()
                (output / "artifacts/h2.py").write_text("TARGET_LAYER='h2'\nclass Harness:\n"
                    "    def bind_environment_state(self, state):\n        self.db=state\n"
                    "    def h2(self, message, history):\n        return message\n")
                save(output / "validation.json", {"usage": {"total_tokens": 15}, "proposer_requests": 1, "repairs_granted": 0})
                return output / "bundle.json"

            usage = {"total_tokens": 15, "inference_accounting_complete": True}
            with patch.object(run, "evaluate", side_effect=evaluation), \
                    patch.object(run, "usage", return_value=usage), patch.object(run, "completion"), \
                    patch.object(runner.profile, "prepare"), patch.object(runner.profile, "propose", side_effect=proposal):
                self.assertIsNone(run.round(1))
                self.assertIsNone(run.round(2))
            self.assertEqual(run.state["round"], 2)
            self.assertEqual(run.state["active_snapshot"][0]["revision"], 2 if accuracy_only else 1)
            self.assertEqual([r["accepted"] for r in run.state["rounds"]], [True, accuracy_only])
            self.assertEqual([r["comparison_batch"] for r in run.state["rounds"]], [0, 1] if accuracy_only else [1, 2])
            names = (["initial_batch_00", "candidate_01", "initial_batch_01", "candidate_02"] if accuracy_only else
                     ["initial_batch_00", "candidate_01", "comparison_01", "comparison_01", "candidate_02", "comparison_02"])
            self.assertEqual([e[1] for e in evaluated], names)
            if accuracy_only:
                self.assertIn("different batches", (root / "COMPARISON.md").read_text())
                self.assertEqual([r["same_batch_comparison"] for r in run.state["rounds"]], [False, False])
            self.assertEqual(read(root / "TRAINING_RESULTS.json")["rounds"], run.state["rounds"])
            run.verify()

    def test_accuracy_only_resume_refuses_live_or_wrong_phase(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save(root / "protocol.json", {})
            save(root / "frozen_sources.json", {})
            save(root / "telecom/state.json", {"round": 0, "rounds": []})
            for phase, evaluation, closed, pid in (("evaluating", "comparison_01", True, 1),
                    ("evaluating", "comparison_01", False, 0), ("evaluating", "candidate_01", True, 0)):
                save(root / "status.json", {"phase": phase, "evaluation": evaluation, "pid": pid})
                save(root / "flash_metrics.json", {"closed": closed})
                with self.assertRaises(ValueError):
                    runner.prepare_accuracy_only_resume(root)
                self.assertFalse((root / "ACCURACY_ONLY_AMENDMENT.json").exists())

    def test_initializer_refuses_existing_experiment(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileExistsError):
                runner.initialize(Path(tmp))

    def test_cached_proposal_has_zero_new_cost_but_keeps_paid_cost(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prior = dict(attempts=52, known_usage_calls=52, total_tokens=8365576,
                         input_tokens=8196321, output_tokens=169255)
            run = runner.Run.__new__(runner.Run)
            run.root = root
            run.protocol = {"continuation": {"optimizer_usage": prior}}
            run.broker = SimpleNamespace(ledger=SimpleNamespace(summary=lambda: {"groups": {}}))
            first = run.usage(1)
            self.assertEqual((first['total_tokens'], first['attempts']), (8365576, 52))
            self.assertTrue(first['inference_accounting_complete'])
            self.assertEqual(read(root / 'OPTIMIZER_TOKEN_USAGE.json')['total_tokens'], 8365576)
            current = runner.optimizer_usage(run.broker.ledger.summary(), root.name, 1)
            self.assertEqual(current['total_tokens'], 0)
            self.assertEqual(run.usage(2)['total_tokens'], 0)


if __name__ == "__main__":
    unittest.main()
