"""Regression: nonempty incumbent evidence binding without model replay."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from meta import harness_hooks as hooks
from meta.life_evolve import profile
from meta import life_webshop_resume as runner
from meta.tau_comparison.common import digest, read, save


class WebShopResumeTests(unittest.TestCase):
    def test_binding_nonempty_snapshot_makes_original_prepare_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work, evidence = root / "work", root / "evidence"
            artifact = work / "versions/b/h4.py"
            artifact.parent.mkdir(parents=True)
            artifact.write_text("TARGET_LAYER='h4'\nclass Harness:\n def h4(self, messages, remaining): return []\n")
            records = [{"mechanism_id": "monitor", "revision": 1, "target_layer": "h4",
                        "artifact_path": "versions/b/h4.py", "sha256": digest(artifact)}]
            save(work / "state.json", {"domain": "webshop", "round": 1, "rounds": [],
                "active_snapshot": records, "active_snapshot_sha256": hooks.object_digest(records)})
            manifest = root / "manifest.json"
            body = {"schema_version": 2, "artifact_root": str(work), "records": records,
                    "snapshot_sha256": hooks.object_digest(records)}
            save(manifest, {**body, "manifest_sha256": hooks.object_digest(body)})
            config_body = {"domain": "webshop", "split": "train", "ids": [1],
                           "snapshot_sha256": hooks.object_digest(records)}
            config = {**config_body, "config_sha256": hooks.object_digest(config_body)}
            save(evidence / "config.json", config)
            save(evidence / "episode_1.json", {"episode_schema_version": 1, "config_sha256": config["config_sha256"],
                "index": 1, "success": 1, "infrastructure_error": False, "turns": 1, "tokens": 5,
                "status": "completed", "history": [], "calls": []})
            save(evidence / "summary.json", {**config, "n": 1, "successes": 1, "accuracy": 1.0,
                "infra_errors": 0, "tokens": 5, "turns": 1, "per_task": {"1": 1}})
            before = digest(evidence / "episode_1.json")
            with self.assertRaisesRegex(ValueError, "exact incumbent content"):
                profile.prepare(work, evidence, root / "before")
            summary = runner.bind_summary(evidence, manifest)
            self.assertEqual(summary["chain_hashes"]["versions/b/h4.py"], digest(artifact))
            self.assertEqual(digest(evidence / "episode_1.json"), before)
            profile.prepare(work, evidence, root / "after")
            self.assertEqual(read(root / "after/summary.json")["n"], 1)

    def test_inherited_cost_is_in_first_round_and_total_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = object.__new__(runner.Run)
            run.root = root
            run.protocol = {"continuation": {"optimizer_usage": {"input_tokens": 90,
                "output_tokens": 10, "total_tokens": 100}}}
            run.broker = SimpleNamespace(ledger=SimpleNamespace(summary=lambda: {"groups": {"current": {
                "experiment": root.name, "domain": "webshop", "round": 2, "role": "life_optimizer",
                "input_tokens": 10, "output_tokens": 5, "total_tokens": 15}}}))
            self.assertEqual(run.usage(1)["total_tokens"], 100)
            self.assertEqual(run.usage(2)["total_tokens"], 15)
            self.assertEqual(run.usage()["total_tokens"], 115)
            self.assertEqual(read(root / "OPTIMIZER_TOKEN_USAGE.json")["total_tokens"], 115)


if __name__ == "__main__":
    unittest.main()
