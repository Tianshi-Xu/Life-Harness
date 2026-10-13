import concurrent.futures
import copy
import hashlib
import json
import tempfile
import threading
import unittest
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

import yaml

from ahe_native.api_broker import RequestGate
from ahe_native.backend import HERE, MANIFEST, validate_request
from ahe_native.backend import workspace_hash
from ahe_native.batch import reuse_baseline
from ahe_native.comparison import publish_comparison
from ahe_native.cli import config_for
from ahe_native.isolation import copy_unchanged_or_atomic
from ahe_native.records import write_json, export_job
from ahe_native.capabilities import restricted_policy
from ahe_native.training import AGENTBENCH, make_schedule, scheduled_batch
from ahe_native.upstream import bind_native_transport, load_upstream


class FormalBatchTests(unittest.TestCase):
    def setUp(self):
        self.schedule = make_schedule()
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / "schedule.json"
        write_json(self.path, self.schedule)

    def request(self, domain="dbbench", iteration=1):
        return {"domain": domain, "split": "train", "iteration": iteration,
            "training_schedule_file": str(self.path),
            "training_schedule_sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(),
            "task_ids": self.schedule["domains"][domain]["batches"][iteration - 1]["task_ids"],
            "k": 1, "concurrency": 50, "workspace": str(HERE / "baselines" / domain),
            "agent_config": "code_agent.yaml"}

    def test_nine_disjoint_fifty_task_batches_per_domain(self):
        for domain in AGENTBENCH:
            batches = self.schedule["domains"][domain]["batches"]
            self.assertEqual(len(batches), 9)
            self.assertTrue(all(len(batch["task_ids"]) == 50 for batch in batches))
            all_ids = [task for batch in batches for task in batch["task_ids"]]
            self.assertEqual(len(set(all_ids)), 450)
            if domain == "webshop":
                self.assertTrue(all(200 <= task < 1021 for task in all_ids))
        self.assertEqual(self.schedule, make_schedule())

    def test_new_formal_training_ids_are_validated_against_full_native_train_split(self):
        old = json.loads(MANIFEST.read_text())["domains"]["dbbench"]
        old_ids = {task for batch in old["batches"] for task in batch["indices"]}
        request = self.request()
        self.assertTrue(set(request["task_ids"]) - old_ids)
        self.assertEqual(validate_request(request), [str(task) for task in request["task_ids"]])

    def test_request_must_match_pinned_iteration(self):
        request = self.request()
        request["task_ids"] = list(reversed(request["task_ids"]))
        with self.assertRaisesRegex(ValueError, "pinned iteration"):
            validate_request(request)

    def test_schedule_tampering_fails_closed(self):
        request = self.request()
        self.path.write_text(self.path.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "schedule changed"):
            scheduled_batch(request)

    def test_rehashed_test_or_repeated_tasks_still_rejected(self):
        for kind in ("test-split", "webshop-test-id", "repeat"):
            schedule = copy.deepcopy(self.schedule)
            if kind == "test-split":
                schedule["split"] = "test"
            elif kind == "webshop-test-id":
                schedule["domains"]["webshop"]["batches"][0]["task_ids"][0] = 0
            else:
                schedule["domains"]["webshop"]["batches"][1]["task_ids"][0] = schedule["domains"]["webshop"]["batches"][0]["task_ids"][0]
            with self.subTest(kind=kind):
                write_json(self.path, schedule)
                with self.assertRaises(ValueError):
                    scheduled_batch(self.request("webshop"))

    def test_transport_uses_original_iteration_to_rotate_batches(self):
        module, _ = load_upstream()
        original_launch, original_wait = module.launch_harbor, module.wait_for_harbor
        original_loop, original_best = module.run_single_experiment, module.update_best_ever
        self.addCleanup(setattr, module, "launch_harbor", original_launch)
        self.addCleanup(setattr, module, "wait_for_harbor", original_wait)
        request = self.request(iteration=9)
        config = config_for("dbbench", self.schedule["domains"]["dbbench"]["batches"][0]["task_ids"], 9, "http://127.0.0.1:1/v1")
        config["native_backend"].update(training_schedule_file=str(self.path),
            training_schedule_sha256=request["training_schedule_sha256"])
        bind_native_transport(module)
        output = self.root / "runs/iteration_009/input/benchmark"
        output.mkdir(parents=True)
        with patch("ahe_native.upstream.subprocess.Popen", return_value=MagicMock()):
            module.launch_harbor(config, HERE / "baselines/dbbench", "code_agent.yaml", output)
        actual = json.loads((output / "native_request.json").read_text())
        self.assertEqual(actual["task_ids"], request["task_ids"])
        self.assertIs(module.run_single_experiment, original_loop)
        self.assertIs(module.update_best_ever, original_best)

    def test_smoke_defaults_are_preserved(self):
        config = config_for("dbbench", list(range(12)), 2, "http://127.0.0.1:1/v1")
        self.assertEqual(config["harbor"]["n_concurrent"], 2)
        self.assertEqual(config["agent_debugger"]["max_tasks"], 3)
        self.assertEqual(config["target_pass_rate"], 0.95)

    def test_flash_gate_is_global_and_releases_slots(self):
        gate = RequestGate(8)
        full = threading.Event()
        release = threading.Event()

        def call():
            with gate:
                if gate.snapshot()["active"] == 8:
                    full.set()
                if not release.wait(5):
                    raise TimeoutError("test gate was not released")

        with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
            futures = [pool.submit(call) for _ in range(24)]
            try:
                self.assertTrue(full.wait(5))
                self.assertEqual(gate.snapshot()["active"], 8)
            finally:
                release.set()
            for future in futures:
                future.result()
        self.assertEqual(gate.snapshot(), {"limit": 8, "active": 0, "peak": 8, "requests": 24})
        with self.assertRaises(RuntimeError):
            with gate:
                raise RuntimeError("synthetic request failure")
        self.assertEqual(gate.snapshot()["active"], 0)

    def test_shared_runtime_sources_are_not_rewritten_on_each_episode(self):
        source, target = self.root / "source.py", self.root / "target.py"
        source.write_text("first = True\n")
        copy_unchanged_or_atomic(source, target)
        first = target.stat()
        copy_unchanged_or_atomic(source, target)
        self.assertEqual(target.stat().st_ino, first.st_ino)
        self.assertEqual(target.stat().st_mtime_ns, first.st_mtime_ns)
        source.write_text("second = True\n")
        copy_unchanged_or_atomic(source, target)
        self.assertEqual(target.read_bytes(), source.read_bytes())
        self.assertNotEqual(target.stat().st_ino, first.st_ino)

    def test_api_pacing_retains_eight_slots_and_honors_shared_cooldown(self):
        clock = [0.0]
        def advance(seconds):
            clock[0] += seconds
        starts = []
        with patch("ahe_native.api_broker.time.monotonic", side_effect=lambda: clock[0]):
            with patch("ahe_native.api_broker.time.sleep", side_effect=advance):
                gate = RequestGate(8, min_interval=1.25)
                for _ in range(3):
                    with gate:
                        starts.append(clock[0])
                gate.defer(60)
                with gate:
                    starts.append(clock[0])
        self.assertEqual(starts, [0, 1.25, 2.5, 62.5])
        self.assertEqual(gate.snapshot()["limit"], 8)
        self.assertEqual(gate.snapshot()["paced_requests_per_minute"], 48)
        self.assertEqual(gate.snapshot()["active"], 0)

    def test_baseline_reuse_preserves_scores_and_rejects_changed_workspace(self):
        source, destination = self.root / "previous", self.root / "continued"
        snapshot = source / "runs/iteration_001/input/workspace"
        for path in (source / "workspace", snapshot):
            path.mkdir(parents=True)
            (path / "code_agent.yaml").write_text("public baseline\n")
        job = source / "runs/iteration_001/input/benchmark/2026-10-03__00-00-00"
        write_json(job / "result.json", {"complete": True})
        write_json(job / "workspace_receipt.json", {"workspace_sha256": workspace_hash(snapshot)})
        for task in range(3):
            write_json(job / f"native_{task}/native_record.json", {"task_id": str(task), "success": int(task == 0), "infrastructure_error": False})
        module = MagicMock()
        reuse_baseline(module, source, destination, list(range(3)))
        module.rollback_experiment_metadata.assert_called_once_with(destination, 1)
        receipt = json.loads((destination / "continuation_receipt.json").read_text())
        self.assertEqual(receipt["completed_native_episodes_reused"], 3)
        self.assertEqual((destination / job.relative_to(source) / "result.json").read_bytes(), (job / "result.json").read_bytes())
        (source / "workspace/code_agent.yaml").write_text("changed policy\n")
        with self.assertRaisesRegex(ValueError, "baseline changed"):
            reuse_baseline(module, source, self.root / "invalid", list(range(3)))

    def test_restricted_recovery_reuses_only_already_bounded_analysis_without_new_debugger_cost(self):
        source, destination = self.root / "restricted_source", self.root / "restricted_recovery"
        policy = restricted_policy("alfworld")
        shutil.copytree(HERE / "baselines/alfworld", source / "workspace")
        snapshot = source / "runs/iteration_001/input/workspace"
        shutil.copytree(source / "workspace", snapshot)
        config = config_for("alfworld", list(range(3)), 6, "unused", debugger_tasks=10, capability_policy=policy)
        (source / "config_snapshot.yaml").write_text(yaml.safe_dump(config))
        job = export_job({"schema_version": 1, "domain": "alfworld", "split": "train", "records": [
            {"task_id": str(i), "trial": 0, "success": int(i == 0), "messages": [{"role": "user", "content": "offline fixture"}]} for i in range(3)]},
            source / "runs/iteration_001/input/benchmark/2026-10-03__00-00-00")
        write_json(job / "workspace_receipt.json", {"workspace_sha256": workspace_hash(snapshot)})
        module, _ = load_upstream()
        stats = module.compute_stats(job)
        selected = module._build_adb_jobs(stats["task_results"], job, config["agent_debugger"])
        analysis = source / "runs/iteration_001/input/analysis"
        (analysis / "detail").mkdir(parents=True)
        (analysis / "overview.md").write_text("already completed selected-task analysis " + str(source))
        (analysis / "detail" / (selected[0].task_name + ".md")).write_text("genuine fixture result")
        (source / "runs/iteration_001/evolve").mkdir()
        (source / "runs/iteration_001/evolve/isolated_process.log").write_text("unfinished old process")
        reuse_baseline(module, source, destination, list(range(3)), policy)
        receipt = json.loads((destination / "continuation_receipt.json").read_text())
        self.assertTrue(receipt["restricted_selected_analysis_reused"])
        self.assertEqual(receipt["completed_native_episodes_reused"], 3)
        self.assertIn(str(destination), (destination / "runs/iteration_001/input/analysis/overview.md").read_text())
        self.assertFalse((destination / "runs/iteration_001/evolve").exists())
        self.assertIn(str(source), (analysis / "overview.md").read_text())
        (analysis / "detail/not_selected.md").write_text("outside the permitted sample")
        invalid = self.root / "restricted_recovery_archived"
        reuse_baseline(module, source, invalid, list(range(3)), policy)
        self.assertFalse(json.loads((invalid / "continuation_receipt.json").read_text())["restricted_selected_analysis_reused"])
        self.assertFalse((invalid / "runs/iteration_001/input/analysis").exists())

    def test_comparison_aligns_optimization_steps_and_marks_unmeasured_references(self):
        write_json(self.root / "protocol.json", {"name": "test-only", "iterations_per_domain": 9})
        write_json(self.root / "status.json", {"updated_at": "test-only", "domains": {"alfworld": {"evaluations": [
            {"iteration": 1, "tasks": 50, "successes": 14, "infrastructure_errors": 1}]}}})
        report = publish_comparison(self.root)
        rows = {(row["domain"], row["optimization_step"]): row for row in report["rows"]}
        self.assertEqual(rows["alfworld", 0]["ahe"]["accuracy"], 0.28)
        self.assertIsNone(rows["alfworld", 1]["ahe"])
        self.assertIsNone(rows["alfworld", 7]["meta"])
        self.assertIsNone(rows["alfworld", 7]["life"])
        self.assertIsNone(rows["webshop", 9]["life"])
        self.assertIsNone(rows["dbbench", 9]["life"])
        self.assertTrue((self.root / "COMPARISON.md").is_file())

    def test_six_round_restricted_comparison_keeps_variant_and_round_semantics(self):
        write_json(self.root / "protocol.json", {"name": "six-restricted-test", "iterations_per_domain": 6,
            "capability_profile": "h2345-budget-restricted-v1", "unrestricted_ahe_reproduction": False,
            "debugger_tasks_per_iteration": 10})
        write_json(self.root / "alfworld_final_candidate_train.json", {"domain": "alfworld", "optimization_step": 6,
            "tasks": 50, "successes": 25, "infrastructure_errors": 0, "batch": 5})
        report = publish_comparison(self.root)
        self.assertEqual(len(report["rows"]), 3 * 7)
        self.assertFalse(report["unrestricted_ahe_reproduction"])
        self.assertEqual(report["debugger_tasks_per_iteration"], 10)
        text = (self.root / "COMPARISON.md").read_text()
        self.assertIn("step 6 candidate", text)
        self.assertNotIn("step 9 candidate", text)
        self.assertIn("h2345-budget-restricted-v1", text)
        row = next(r for r in report["rows"] if r["domain"] == "alfworld" and r["optimization_step"] == 6)
        self.assertEqual(row["ahe"]["accuracy"], 0.5)
        self.assertIn("not used", report["notes"][2])


if __name__ == "__main__":
    unittest.main()
