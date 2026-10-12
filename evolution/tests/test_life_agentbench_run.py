"""Offline checks for the opt-in current-interface WebShop launcher."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import os
import unittest
from unittest.mock import patch

from evolution import harness_hooks as hooks
from evolution import life_agentbench_run as runner
from evolution.tau_comparison.common import digest, read, save


class WebShopRunTests(unittest.TestCase):
    def test_native_java_environment_and_credential_removal(self):
        with patch.dict(os.environ, {"MR_API_KEY": "not-a-real-key", "OMP_NUM_THREADS": "128"}):
            env = runner.native_environment()
        # Unit-test the constructed environment, not an untracked Java install.
        self.assertEqual(Path(env["JVM_PATH"]), Path(env["JAVA_HOME"]) / "lib/server/libjvm.so")
        self.assertEqual(Path(env["JAVA_HOME"]), runner.ROOT /
                         "AgentBench/.native/sysroot/usr/lib/jvm/java-11-openjdk-amd64")
        self.assertEqual(env["OMP_NUM_THREADS"], "1")
        self.assertNotIn("MR_API_KEY", env)
        self.assertEqual(env["_JAVA_OPTIONS"], "-Xmx4g")

    def test_worker_installs_current_interface_not_historical_injector(self):
        task = SimpleNamespace()
        native = SimpleNamespace(inject_task_module=lambda *args: self.fail("old interface"))
        native.initialize = lambda domain, split, path: native.inject_task_module(domain, task, path)
        native.episode = lambda index: {"index": index}
        receipt = {"interface_version": hooks.INTERFACE_VERSION}
        with patch.object(runner, "native_module", return_value=native), patch.object(
                hooks, "install_agentbench", return_value=receipt) as install:
            runner.initialize_worker("train", "manifest")
            install.assert_called_once_with(task, "webshop", "manifest")
            self.assertEqual(runner.episode_worker(3), {"index": 3, "interface": receipt})

    def test_usage_excludes_solver_simulator_other_round_and_domain(self):
        def group(**changes):
            return {"experiment": "run", "domain": "webshop", "role": "life_optimizer",
                    "round": 1, "total_tokens": 15, **changes}
        summary = {"groups": {str(i): g for i, g in enumerate([
            group(), group(round=2), group(role="solver"), group(role="user_simulator"),
            group(domain="telecom"), group(experiment="other")])}}
        self.assertEqual(runner.optimizer_usage(summary, "run")["total_tokens"], 30)
        self.assertEqual(runner.optimizer_usage(summary, "run", 1)["total_tokens"], 15)
        summary["groups"]["0"].update(unknown_usage_calls=1, rate_limit_rejections_without_usage=1)
        self.assertTrue(runner.optimizer_usage(summary, "run", 1)["inference_accounting_complete"])
        summary["groups"]["0"]["pending_calls"] = 1
        self.assertFalse(runner.optimizer_usage(summary, "run", 1)["inference_accounting_complete"])

    def test_test_requires_seal_full_ids_and_exact_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save(root / "protocol.json", {"test_ids": [10, 11], "batches": [[1, 2]], "solver_workers": 50})
            state = {"round": 6, "rounds": [{} for _ in range(6)]}
            save(root / "webshop/state.json", state)
            manifest = root / "manifest.json"
            save(manifest, {})
            with patch.object(hooks, "manifest_records", return_value=(root, [])):
                with self.assertRaises(FileNotFoundError):
                    runner.evaluation_config(root, "test", "final_test", manifest, [10, 11])
                save(root / "training_seal.json", {"state_sha256": digest(root / "webshop/state.json"), "snapshot": []})
                config = runner.evaluation_config(root, "test", "final_test", manifest, [10, 11])
                self.assertEqual(config["interface_version"], hooks.INTERFACE_VERSION)
                for split, name, ids in [("test", "early_test", [10, 11]), ("test", "final_test", [10]),
                                         ("train", "baseline", [10])]:
                    with self.assertRaises(ValueError):
                        runner.evaluation_config(root, split, name, manifest, ids)
                save(root / "webshop/state.json", {**state, "round": 5})
                with self.assertRaises(ValueError):
                    runner.evaluation_config(root, "test", "final_test", manifest, [10, 11])

    def test_two_rounds_rotate_pair_and_keep_rejected_candidate_without_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ids = [list(range(50)), list(range(50, 100))]
            save(root / "protocol.json", {"batches": ids, "meta_reference": [{}, {}], "ahe_reference": [{}, {}]})
            save(root / "frozen_sources.json", {})
            save(root / "webshop/state.json", {"domain": "webshop", "round": 0, "rounds": [],
                "active_snapshot": [], "active_snapshot_sha256": hooks.object_digest([]),
                "accepted_bundle_ids": [], "operation_log": [], "prior_test_exposure": True})
            run = runner.Run(root)
            evaluated = []

            def evaluate(split, name, manifest, tasks):
                evaluated.append((name, tasks))
                n = run.state["round"] + 1
                wins = 30 if name.startswith("candidate") and n == 1 else 20 if name.startswith("candidate") else 25
                return {"split": split, "domain": "webshop", "n": 50, "successes": wins,
                    "accuracy": wins / 50, "infra_errors": 0,
                    "per_task": {str(index): int(i < wins) for i, index in enumerate(tasks)}}, root / name

            def propose(workspace, output, **kwargs):
                n = run.state["round"] + 1
                output.mkdir(parents=True)
                bundle = {"schema_version": 2, "bundle_id": f"round_{n}", "hypothesis": "offline no-op test",
                    "base_snapshot_sha256": run.state["active_snapshot_sha256"], "domain": "webshop",
                    "operations": [{"operation_id": f"add_{n}", "mechanism_id": f"mechanism_{n}", "op": "add",
                        "target_layer": "h4", "artifact": "artifacts/h4.py", "hypothesis": "offline"}]}
                (output / "artifacts").mkdir()
                (output / "artifacts/h4.py").write_text("TARGET_LAYER='h4'\nclass Harness:\n def bind_context(self, context): pass\n def h4(self, history, remaining): return []\n")
                save(output / "bundle.json", bundle)
                save(output / "validation.json", {"usage": {"total_tokens": 15}, "proposer_requests": 1, "repairs_granted": 0})
                return output / "bundle.json"

            usage = {"inference_accounting_complete": True, "total_tokens": 15}
            with patch.object(run, "evaluate", side_effect=evaluate), patch.object(run, "completion", return_value=None), \
                 patch.object(run, "usage", return_value=usage), patch.object(run, "progress"), \
                 patch.object(runner.profile, "prepare"), patch.object(runner.profile, "propose", side_effect=propose):
                run.round(1)
                run.round(2)
            self.assertEqual(evaluated, [("incumbent_01", ids[0]), ("candidate_01", ids[0]),
                                         ("incumbent_02", ids[1]), ("candidate_02", ids[1])])
            state = read(root / "webshop/state.json")
            self.assertEqual([r["accepted"] for r in state["rounds"]], [True, False])
            self.assertEqual([r["retained_score"] for r in state["rounds"]], [.6, .5])
            self.assertEqual(len(state["active_snapshot"]), 1)
            self.assertTrue((root / "webshop/proposals/round_02/bundle.json").exists())
            run.compose.verify_active(run.work, state)


if __name__ == "__main__":
    unittest.main()
