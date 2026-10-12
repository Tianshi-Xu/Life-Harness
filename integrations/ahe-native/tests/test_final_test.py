import copy
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ahe_native import final_test
from ahe_native.backend import HERE, validate_request, workspace_hash
from ahe_native.capabilities import PROFILE
from ahe_native.records import validate_records, write_json
from ahe_native.training import AGENTBENCH


class FinalTestTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.batch = self.root / "batches/offline_six"
        self.protocol = {"name": "offline_six", "iterations_per_domain": 6,
            "capability_profile": PROFILE, "train_tasks_per_iteration": 50,
            "experiments": {d: d for d in AGENTBENCH}}
        write_json(self.batch / "training_schedule.json", {"offline": True})
        self.protocol["training_schedule_sha256"] = final_test.digest(self.batch / "training_schedule.json")
        write_json(self.batch / "protocol.json", self.protocol)
        write_json(self.batch / "source_receipt.json", {"offline_source": True})
        self.status = {"state": "complete", "domains": {}}
        for domain in AGENTBENCH:
            experiment = self.root / "experiments" / domain
            shutil.copytree(HERE / "baselines" / domain, experiment / "workspace")
            for i in range(1, 7):
                summary = experiment / f"runs/iteration_{i:03d}/evolve/evolve_summary.md"
                summary.parent.mkdir(parents=True)
                summary.write_text("completed offline fixture")
            write_json(self.batch / f"{domain}_final_candidate_train.json", {
                "split": "train", "optimization_step": 6, "tasks": 50,
                "workspace_sha256": workspace_hash(experiment / "workspace"),
                "selection_modified": False, "used_for_optimization": False})
            self.status["domains"][domain] = {"state": "complete", "current_iteration": 6,
                "evaluations": [{"iteration": i, "tasks": 50} for i in range(1, 7)]}
        write_json(self.batch / "status.json", self.status)
        context = patch.object(final_test, "HERE", self.root)
        context.start()
        self.addCleanup(context.stop)
        source = patch.object(final_test, "source_fingerprint", return_value={"offline_source": True})
        source.start()
        self.addCleanup(source.stop)

    def test_all_domains_are_sealed_before_any_test(self):
        seal = final_test.completed_training(self.batch)
        self.assertEqual(seal["optimization_step"], 6)
        self.assertEqual(set(seal["domains"]), set(AGENTBENCH))
        for domain in AGENTBENCH:
            self.assertEqual(len(seal["domains"][domain]["evolve_summary_sha256"]), 6)

    def test_running_or_failed_domain_cannot_start_test(self):
        for state in ("running", "failed"):
            value = copy.deepcopy(self.status)
            value["domains"]["webshop"]["state"] = state
            write_json(self.batch / "status.json", value)
            with self.subTest(state=state), self.assertRaises(ValueError):
                final_test.completed_training(self.batch)

    def test_last_evolve_receipt_and_measured_snapshot_are_required(self):
        summary = self.root / "experiments/dbbench/runs/iteration_006/evolve/evolve_summary.md"
        summary.unlink()
        with self.assertRaisesRegex(ValueError, "Evolve receipts"):
            final_test.completed_training(self.batch)
        summary.write_text("complete")
        prompt = self.root / "experiments/dbbench/workspace/systemprompt.md"
        prompt.write_text(prompt.read_text() + "\nDifferent candidate.\n")
        with self.assertRaisesRegex(ValueError, "final train measurement"):
            final_test.completed_training(self.batch)

    def test_optimizer_backend_and_record_export_still_reject_test(self):
        with self.assertRaisesRegex(ValueError, "never run test"):
            validate_request({"domain": "dbbench", "split": "test"})
        with self.assertRaisesRegex(ValueError, "only training"):
            validate_records({"schema_version": 1, "domain": "dbbench", "split": "test", "records": []})

    def test_fixed_full_test_spec_is_hash_pinned(self):
        manifest = json.loads(final_test.MANIFEST.read_text())
        if any(not (final_test.LIFE / "AgentBench" / relative).is_file()
               for relative in manifest["data_hashes"]):
            self.skipTest("Native game/WebShop runtime assets must be installed for this integration audit")
        spec = final_test.fixed_test_spec()
        self.assertEqual(spec["split"], "test")
        self.assertEqual({d: len(ids) for d, ids in spec["task_ids"].items()},
                         {"alfworld": 109, "dbbench": 300, "webshop": 200})

    def test_infrastructure_failure_remains_zero_in_full_denominator(self):
        def fixture(domain, task, workspace, folder):
            record = {"task_id": str(task), "trial": 0, "success": int(task == 0),
                "infrastructure_error": task == 2}
            write_json(folder / "record.json", record)
            return record
        output = self.root / "heldout/dbbench"
        with patch.object(final_test, "run_episode", side_effect=fixture):
            result = final_test.evaluate_domain("dbbench", [0, 1, 2],
                self.root / "experiments/dbbench/workspace", output, 3)
        self.assertEqual((result["successes"], result["n"], result["infra_errors"]), (1, 3, 1))
        self.assertEqual(result["accuracy"], 1 / 3)
        self.assertFalse(result["used_for_optimization"])
        self.assertEqual(result["modelrouter_test_requests"], 0)

    def test_test_worker_uses_existing_native_runner_and_local_solver_only(self):
        observed = []
        def worker(argv, **kwargs):
            observed.append((argv, kwargs))
            request = json.loads(Path(argv[argv.index("--request") + 1]).read_text())
            self.assertEqual(request["split"], "test")
            self.assertEqual(request["task_id"], "0")
            write_json(Path(argv[argv.index("--out") + 1]), {
                "task_id": "0", "trial": 0, "success": 1, "infrastructure_error": False})
            return subprocess.CompletedProcess(argv, 0)
        with patch.object(final_test.subprocess, "run", side_effect=worker):
            result = final_test.run_episode("dbbench", 0, self.root / "experiments/dbbench/workspace",
                                            self.root / "heldout/episode_0")
        self.assertEqual(len(observed), 1)
        argv, kwargs = observed[0]
        self.assertIn("ahe_native.native_worker", argv)
        self.assertEqual(kwargs["env"]["LLM_BASE_URL"], "http://127.0.0.1:8400/v1")
        self.assertEqual(kwargs["env"]["LLM_API_KEY"], "EMPTY")
        self.assertEqual(result["success"], 1)

    def test_native_worker_failure_is_not_retried_or_dropped(self):
        with patch.object(final_test.subprocess, "run", return_value=subprocess.CompletedProcess([], 1)) as run:
            result = final_test.run_episode("dbbench", 0, self.root / "experiments/dbbench/workspace",
                                            self.root / "heldout/episode_0")
        self.assertEqual(run.call_count, 1)
        self.assertTrue(result["infrastructure_error"])
        self.assertEqual(result["success"], 0)

    def test_comparison_reports_actual_scores_even_above_user_reference(self):
        results = {d: {"accuracy": 0.9, "successes": 9, "n": 10} for d in AGENTBENCH}
        reference = self.root / "reference/evolution/experiments/life_agentbench_six_resume_mr_20261001/final_results.json"
        write_json(reference, {"comparisons": {d: {"n": 10, "meta_test": 0.4, "life_six_test": 0.8}
                                                 for d in AGENTBENCH}})
        output = self.root / "report"
        output.mkdir()
        with patch.object(final_test, "LIFE", self.root / "reference"):
            report = final_test.comparison_report(results, output)
        self.assertFalse(report["rows"]["dbbench"]["ahe_below_release"])
        self.assertFalse(report["rows"]["webshop"]["ahe_below_life_six"])
        self.assertIsNone(report["rows"]["alfworld"]["ahe_below_release"])
        self.assertIn("90.00%", (output / "COMPARISON.md").read_text())


if __name__ == "__main__":
    unittest.main()
