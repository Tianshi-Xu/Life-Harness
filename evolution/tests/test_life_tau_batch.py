"""Offline checks for fresh Airline/Retail metering and native trial identities."""
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import evolution
from evolution import harness_hooks as hooks
from evolution import life_tau_run as runner
from evolution.tau_comparison.common import read, save


class TauBatchTests(unittest.TestCase):
    def test_task_trial_identities_are_unique_and_single_trial_is_compatible(self):
        task = "task_1"
        self.assertEqual(runner.case_key(task), "native_" + runner.hashlib.sha256(task.encode()).hexdigest())
        keys = {runner.case_key(task, i, 2) for i in range(2)}
        self.assertEqual(len(keys), 2)
        self.assertNotIn(runner.case_key(task), keys)

    def test_trial_export_reuses_native_export_with_unique_ids_and_restores_binding(self):
        ids = []
        fake = SimpleNamespace()

        class Simulation:
            task_id = "task_1"
            def __init__(self, trial):
                self.trial = trial
            def model_copy(self, update):
                return SimpleNamespace(task_id=update["task_id"], trial=self.trial)

        def export(results, *args):
            ids.extend("native_" + runner.hashlib.sha256(s.task_id.encode()).hexdigest() for s in results.simulations)
        fake.export_training = export
        fake.evaluation_cli = lambda args: fake.export_training(
            SimpleNamespace(simulations=[Simulation(0), Simulation(1)]), "airline", "manifest", "out")
        with patch.dict(sys.modules, {"evolution.tau_hooks": fake}), patch.object(evolution, "tau_hooks", fake, create=True):
            runner.evaluation_cli("airline", ["--trials", "2"])
        self.assertIs(fake.export_training, export)
        self.assertEqual(set(ids), {runner.case_key("task_1", i, 2) for i in range(2)})

    def test_complete_sixty_episode_selection_is_not_the_old_fifty_task_gate(self):
        ids = [runner.case_key(str(t), i, 2) for t in range(30) for i in range(2)]
        def summary(wins):
            return {"domain": "airline", "split": "train", "n": 60, "successes": wins,
                    "accuracy": wins / 60, "per_task": {key: int(i < wins) for i, key in enumerate(ids)}}
        self.assertTrue(runner.strict_improvement(summary(32), summary(30), ids))
        self.assertFalse(runner.strict_improvement(summary(30), summary(30), ids))
        for invalid in ({**summary(32), "accuracy": float("nan")}, {**summary(32), "n": 30},
                        {**summary(32), "successes": 33}, {**summary(32), "split": "test"}):
            with self.assertRaises(ValueError):
                runner.strict_improvement(invalid, summary(30), ids)
        with self.assertRaises(ValueError):
            runner.strict_improvement(summary(32), summary(30), ids[:30])

    def test_native_failed_placeholder_is_zero_without_fabricated_or_private_trace(self):
        from evolution import tau_hooks
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save(root / "protocol.json", {"domains_spec": {"airline": {}}})
            save(root / "frozen_sources.json", {})
            save(root / "airline/state.json", {"domain": "airline", "round": 0, "rounds": [],
                "active_snapshot": [], "active_snapshot_sha256": hooks.object_digest([]),
                "accepted_bundle_ids": [], "operation_log": []})
            r = runner.Run(root, "airline")
            manifest = r.compose.write_active_chain(r.work, r.state, "empty")
            failed = SimpleNamespace(task_id="a", trial=1, reward_info=None,
                termination_reason="infrastructure_error", info={"error_type": "BadRequestError",
                    "failed_after_attempts": 4, "error_traceback": "PRIVATE_SENTINEL", "error": "PRIVATE_SENTINEL"})
            complete = SimpleNamespace(task_id="a", trial=0, reward_info=SimpleNamespace(reward=1),
                termination_reason="user_stop", info={"life_public_episode": {"index": "old",
                    "success": 0, "turns": 1, "history": [], "calls": [], "status": "completed"}})
            normalized = runner.export_results(SimpleNamespace(simulations=[complete, failed]), 2)
            summary = tau_hooks.export_training(normalized, "airline", manifest, root / "export")
            self.assertEqual((summary["n"], summary["successes"], summary["infra_errors"]), (2, 1, 1))
            row = read(root / "export" / f"episode_{runner.case_key('a', 1, 2)}.json")
            self.assertTrue(row["trace_unavailable"])
            self.assertEqual((row["success"], row["calls"], row["history"]), (0, [], []))
            self.assertNotIn("PRIVATE_SENTINEL", json.dumps(row))
            self.assertNotIn("life_public_episode", failed.info)
            self.assertEqual(complete.task_id, "a")
            for bad in (SimpleNamespace(**{**failed.__dict__, "termination_reason": "user_stop"}),
                        SimpleNamespace(**{**complete.__dict__, "info": {}})):
                with self.assertRaisesRegex(ValueError, "missing its Life public trace"):
                    tau_hooks.export_training(runner.export_results(SimpleNamespace(simulations=[bad]), 2),
                                              "airline", manifest, root / "invalid")

    def test_final_test_keeps_failed_tasks_in_denominator_and_rejects_incomplete_sets(self):
        good = {"task_id": "a", "trial": 0, "termination_reason": "user_stop", "reward_info": {"reward": 1}}
        failed = {"task_id": "b", "trial": 0, "termination_reason": "infrastructure_error", "reward_info": None}
        result = runner.fixed_test_summary({"per_task": {"a": 1.0}}, [good, failed], ["a", "b"])
        self.assertEqual((result["n"], result["successes"], result["infra_errors"], result["accuracy"]), (2, 1, 1, .5))
        self.assertEqual(result["per_task"], {"a": 1, "b": 0})
        for simulations in ([good], [good, good], [good, {**failed, "trial": 1}],
                            [good, {**failed, "reward_info": {"reward": .5}}]):
            with self.assertRaises(ValueError):
                runner.fixed_test_summary({}, simulations, ["a", "b"])

    def test_inherited_expense_is_counted_once_and_at_its_original_round(self):
        def group(experiment, domain, number, role="life_optimizer"):
            return {"experiment": experiment, "domain": domain, "round": number,
                    "role": role, "total_tokens": 15}
        parent = {"groups": {"1": group("parent", "airline", 1), "2": group("parent", "retail", 6),
                              "3": group("parent", "airline", 1, "user_simulator")}}
        current = {"groups": {"1": group("child", "airline", 3), "2": group("other", "airline", 3)}}
        self.assertEqual(runner.optimizer_usage(current, "child", inherited=parent)["total_tokens"], 45)
        self.assertEqual(runner.optimizer_usage(current, "child", "airline", 1, parent)["total_tokens"], 15)
        self.assertEqual(runner.optimizer_usage(current, "child", "airline", 3, parent)["total_tokens"], 15)
        self.assertEqual(runner.optimizer_usage(current, "child", "retail", 6, parent)["total_tokens"], 15)

    def test_train_continues_only_unfinished_rounds(self):
        for completed in (0, 2, 6):
            with self.subTest(completed=completed):
                r = object.__new__(runner.Run)
                r.state = {"round": completed}
                with patch.object(r, "round") as propose, patch.object(r, "progress"):
                    r.train()
                self.assertEqual([call.args[0] for call in propose.call_args_list], list(range(completed + 1, 7)))

    def test_sealed_final_evaluation_normalizes_native_failure_before_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save(root / "protocol.json", {"domains_spec": {"airline": {"train_trials": 2,
                "batches": [{"task_ids": ["train"]}], "test_task_ids": ["a", "b"]}}})
            save(root / "frozen_sources.json", {})
            save(root / "airline/state.json", {"domain": "airline", "round": 6, "rounds": [],
                "active_snapshot": [], "active_snapshot_sha256": hooks.object_digest([]),
                "accepted_bundle_ids": [], "operation_log": []})
            save(root / "training_seal.json", {"states": {"airline": runner.digest(root / "airline/state.json")}})
            r = runner.Run(root, "airline", ali=SimpleNamespace(url="http://127.0.0.1:1/v1"),
                           solver=SimpleNamespace(url="http://127.0.0.1:2/v1"))
            manifest = r.compose.write_active_chain(r.work, r.state, "final")
            directory = root / "airline/evals/final_test"
            def native(*args, **kwargs):
                save(directory / "harness_summary.json", {"domain": "airline", "per_task": {"a": 1.0}, "n": 2})
                save(directory / "results.json", {"simulations": [
                    {"task_id": "a", "trial": 0, "reward_info": {"reward": 1}, "termination_reason": "user_stop"},
                    {"task_id": "b", "trial": 0, "reward_info": None, "termination_reason": "infrastructure_error"}]})
                return SimpleNamespace(returncode=0)
            with patch.object(runner.subprocess, "run", side_effect=native) as api, patch.object(r, "progress"):
                summary, _ = r.evaluate("test", "final_test", manifest, ["a", "b"], 6)
                reused, _ = r.evaluate("test", "final_test", manifest, ["a", "b"], 6)
                self.assertEqual(api.call_count, 1)
            self.assertEqual(summary, reused)
            self.assertEqual((summary["n"], summary["successes"], summary["infra_errors"], summary["accuracy"]), (2, 1, 1, .5))
            self.assertEqual(read(directory / "evaluation_receipt.json")["summary_sha256"], runner.digest(directory / "harness_summary.json"))


    def test_usage_counts_only_own_optimizer_including_failed_attempts(self):
        def group(**extra):
            return {"experiment": "run", "domain": "airline", "round": 1, "role": "life_optimizer",
                    "total_tokens": 15, **extra}
        summary = {"groups": {str(i): value for i, value in enumerate([
            group(), group(domain="retail"), group(round=2), group(role="user_simulator"),
            group(role="judge"), group(experiment="other"), group(domain="webshop")])}}
        self.assertEqual(runner.optimizer_usage(summary, "run")["total_tokens"], 45)
        self.assertEqual(runner.optimizer_usage(summary, "run", "airline", 1)["total_tokens"], 15)
        summary["groups"]["0"].update(unknown_usage_calls=1, rate_limit_rejections_without_usage=1)
        value = runner.optimizer_usage(summary, "run", "airline", 1)
        self.assertFalse(value["accounting_complete"])
        self.assertTrue(value["inference_accounting_complete"])

    def test_native_command_uses_aliyun_full_concurrency_trials_and_retail_judge(self):
        sys.path.insert(0, str(runner.AHE_NATIVE))
        for domain, trials, n in (("airline", 2, 30), ("retail", 1, 50)):
            with self.subTest(domain=domain), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                ids = [str(i) for i in range(n)]
                save(root / "protocol.json", {"domains_spec": {domain: {"train_trials": trials,
                    "batches": [{"task_ids": ids}], "test_task_ids": ["test"]}}})
                save(root / "frozen_sources.json", {})
                save(root / domain / "state.json", {"domain": domain, "round": 0, "rounds": [],
                    "active_snapshot": [], "active_snapshot_sha256": hooks.object_digest([]),
                    "accepted_bundle_ids": [], "operation_log": []})
                r = runner.Run(root, domain, ali=SimpleNamespace(url="http://127.0.0.1:1/v1"),
                               solver=SimpleNamespace(url="http://127.0.0.1:2/v1"))
                manifest = r.compose.write_active_chain(r.work, r.state, "empty")
                captured = []

                def native(command, **kwargs):
                    captured.append(command)
                    expected = [runner.case_key(t, i, trials) for t in ids for i in range(trials)]
                    directory = r.work / "evals/incumbent_01/life_train"
                    save(directory / "summary.json", {"domain": domain, "split": "train", "n": len(expected),
                        "successes": 0, "infra_errors": 0, "per_task": {key: 0 for key in expected}})
                    return SimpleNamespace(returncode=0)

                with patch.object(runner.subprocess, "run", side_effect=native), patch.object(r, "progress"):
                    summary, _ = r.evaluate("train", "incumbent_01", manifest, ids, 1)
                command = captured[0]
                self.assertEqual(command[command.index("--user-llm") + 1], "openai/qwen3.8-flash")
                self.assertEqual(command[command.index("--concurrency") + 1], str(n * trials))
                self.assertEqual(command[command.index("--agent-max-tokens") + 1], "4096")
                self.assertIn("--user-enable-thinking", command)
                self.assertEqual("--nl" in command, domain == "retail")
                self.assertEqual(summary["n"], n * trials)
                self.assertTrue((r.work / "evals/incumbent_01/evaluation_receipt.json").exists())

    def test_real_prepare_two_rounds_for_both_domains_uses_workspace_outside_evidence(self):
        for domain, trials, n in (("airline", 2, 30), ("retail", 1, 50)):
            with self.subTest(domain=domain), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                ids = [str(i) for i in range(n)]
                save(root / "protocol.json", {"domains_spec": {domain: {"train_trials": trials,
                    "batches": [{"task_ids": ids}, {"task_ids": ids}]}}})
                save(root / "frozen_sources.json", {})
                save(root / domain / "state.json", {"domain": domain, "round": 0, "rounds": [],
                    "active_snapshot": [], "active_snapshot_sha256": hooks.object_digest([]),
                    "accepted_bundle_ids": [], "operation_log": [], "prior_test_exposure": True})
                r = runner.Run(root, domain)
                keys = [runner.case_key(t, i, trials) for t in ids for i in range(trials)]

                def evaluate(split, name, manifest, tasks, number):
                    _, records = hooks.manifest_records(manifest)
                    directory = r.work / "evals" / name / "life_train"
                    wins = 30 if name.startswith("incumbent") else 32 if number == 1 else 28
                    outcomes = {key: int(i < wins) for i, key in enumerate(keys)}
                    for key, success in outcomes.items():
                        save(directory / f"episode_{key}.json", {"index": key, "success": success,
                            "status": "completed", "infrastructure_error": False, "history": [], "calls": [], "turns": 0})
                    summary = {"domain": domain, "split": split, "n": len(keys), "successes": wins,
                        "accuracy": wins / len(keys), "infra_errors": 0, "per_task": outcomes,
                        "chain_hashes": {row["artifact_path"]: row["sha256"] for row in records}}
                    save(directory / "summary.json", summary)
                    return summary, directory

                def proposal(workspace, output, **kwargs):
                    self.assertNotIn(r.work, Path(workspace).parents)
                    self.assertEqual(read(Path(workspace) / "summary.json")["n"], len(keys))
                    number = r.state["round"] + 1
                    save(output / "bundle.json", {"schema_version": 2, "bundle_id": f"bundle_{number}",
                        "domain": domain, "base_snapshot_sha256": r.state["active_snapshot_sha256"], "hypothesis": "test",
                        "operations": [{"operation_id": f"add_{number}", "mechanism_id": f"monitor_{number}",
                            "op": "add", "target_layer": "h4", "hypothesis": "test", "artifact": "artifacts/h4.py"}]})
                    (output / "artifacts").mkdir()
                    (output / "artifacts/h4.py").write_text("TARGET_LAYER='h4'\nclass Harness:\n def h4(self, messages, remaining): return []\n")
                    save(output / "validation.json", {"usage": {"total_tokens": 15}, "proposer_requests": 1, "repairs_granted": 0})
                    return output / "bundle.json"

                with patch.object(r, "evaluate", side_effect=evaluate), patch.object(r, "completion"), \
                     patch.object(r, "progress"), patch.object(r, "usage", return_value={"total_tokens": 15, "inference_accounting_complete": True}), \
                     patch.object(runner.profile, "propose", side_effect=proposal):
                    r.round(1)
                    r.round(2)
                self.assertEqual([row["accepted"] for row in r.state["rounds"]], [True, False])
                self.assertEqual(len(r.state["active_snapshot"]), 1)
                self.assertTrue((r.work / "proposals/round_02/bundle.json").exists())
                r.compose.verify_active(r.work, r.state)


if __name__ == "__main__":
    unittest.main()
