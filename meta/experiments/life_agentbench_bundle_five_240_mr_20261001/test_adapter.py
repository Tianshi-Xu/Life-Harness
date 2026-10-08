from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import agentbench_chain
import agentbench_evaluate
import candidate_contract
import driver
import protocol
import status


def artifact(path: Path, layer: str, body: str | None = None) -> Path:
    signatures = {
        "h2": "def h2(self, message, history):\n  " + (body or "return message"),
        "h3": "def h3(self, tools):\n  " + (body or "return tools"),
        "h4": "def h4(self, messages, remaining):\n  " + (body or "return []"),
        "h5": "def h5(self, messages):\n  " + (body or "return []"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"TARGET_LAYER='{layer}'\nclass Harness:\n {signatures[layer]}\n")
    return path


def operation(operation_id: str, mechanism_id: str, op: str, layer: str,
              artifact_name: str | None = None, prior: dict | None = None) -> dict:
    value = {"operation_id": operation_id, "mechanism_id": mechanism_id, "op": op,
             "target_layer": layer, "hypothesis": "focused mechanism"}
    if artifact_name is not None:
        value["artifact"] = f"artifacts/{artifact_name}"
    if prior is not None:
        value.update(expected_revision=prior["revision"], expected_sha256=prior["sha256"])
    return value


def bundle(path: Path, domain: str, base: str, bundle_id: str,
           operations: list[dict]) -> Path:
    value = {"schema_version": 2, "bundle_id": bundle_id, "domain": domain,
             "base_snapshot_sha256": base, "hypothesis": "atomic bundle",
             "operations": operations}
    path.write_text(json.dumps(value))
    return path


class ProtocolTests(unittest.TestCase):
    def test_protocol_is_three_domains_five_rounds_and_frozen_batches(self):
        historical = protocol.validate_protocol(check_files=False)
        self.assertEqual(protocol.DOMAINS, ("alfworld", "dbbench", "webshop"))
        self.assertEqual(protocol.ROUNDS, 5)
        for domain in protocol.DOMAINS:
            spec = historical["domains"][domain]
            self.assertEqual(len(spec["batches"]), 6)
            self.assertTrue(all(len(protocol.task_ids(spec, index, "train")) == 50
                                for index in range(6)))

    def test_strict_acceptance_and_rotation(self):
        base = {"accuracy": .5, "per_task": {"1": 1, "2": 0}}
        self.assertFalse(driver.accept(dict(base), base))
        improved = {"accuracy": 1.0, "per_task": {"1": 1, "2": 1}}
        self.assertTrue(driver.accept(improved, base))
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            driver.score({"accuracy": .9, "per_task": {"1": 1, "2": 0}})
        with self.assertRaisesRegex(RuntimeError, "must not contain per_task_trials"):
            driver.score({"accuracy": 1.0, "per_task": {"1": 0},
                          "per_task_trials": {"1": [1]}})
        self.assertEqual(driver.next_batch(2, True), 2)
        self.assertEqual(driver.next_batch(2, False), 3)


class BundleTests(unittest.TestCase):
    def test_multi_operation_add_modify_delete_order_and_retarget(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            state = driver.initial_state("alfworld")
            proposal = run / "proposal"
            artifact(proposal / "artifacts/a.py", "h2")
            artifact(proposal / "artifacts/b.py", "h3")
            first = bundle(proposal / "bundle.json", "alfworld", state["active_snapshot_sha256"],
                           "first_bundle", [operation("add_a", "alpha", "add", "h2", "a.py"),
                                            operation("add_b", "beta", "add", "h3", "b.py")])
            parsed, staged, manifest = driver.stage_bundle(run, 1, first, state)
            self.assertEqual([row["mechanism_id"] for row in staged], ["alpha", "beta"])
            self.assertEqual(agentbench_chain.layer_paths(manifest), [
                (manifest.parent / row["artifact_path"]).resolve() for row in staged])
            active = driver.commit_bundle(run, parsed, state)
            state.update(active_snapshot=active, active_snapshot_sha256=driver.snapshot_sha256(active))
            proposal2 = run / "proposal2"
            artifact(proposal2 / "artifacts/a2.py", "h4")
            artifact(proposal2 / "artifacts/c.py", "h5")
            second = bundle(proposal2 / "bundle.json", "alfworld", state["active_snapshot_sha256"],
                            "second_bundle", [
                                operation("mod_a", "alpha", "modify", "h4", "a2.py", active[0]),
                                operation("del_b", "beta", "delete", "h3", prior=active[1]),
                                operation("add_c", "gamma", "add", "h5", "c.py")])
            parsed2, _, _ = driver.stage_bundle(run, 2, second, state)
            updated = driver.commit_bundle(run, parsed2, state)
            self.assertEqual([(row["mechanism_id"], row["revision"], row["target_layer"])
                              for row in updated], [("alpha", 2, "h4"), ("gamma", 1, "h5")])
            self.assertTrue((run / active[0]["artifact_path"]).is_file())
            self.assertEqual((run / active[0]["artifact_path"]).read_text().splitlines()[0],
                             "TARGET_LAYER='h2'")

    def test_duplicate_stale_and_parent_cas_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact(root / "artifacts/a.py", "h2")
            empty = driver.snapshot_sha256([])
            duplicate = {"schema_version": 2, "bundle_id": "dup", "domain": "dbbench",
                         "base_snapshot_sha256": empty, "hypothesis": "x", "operations": [
                             operation("same", "a", "add", "h2", "a.py"),
                             operation("same", "b", "add", "h2", "a.py")]}
            with self.assertRaisesRegex(ValueError, "duplicate"):
                candidate_contract.validate_bundle(duplicate, "dbbench", [], empty, root)
            duplicate["operations"][1]["operation_id"] = "other"
            duplicate["operations"][1]["mechanism_id"] = "a"
            with self.assertRaisesRegex(ValueError, "duplicate"):
                candidate_contract.validate_bundle(duplicate, "dbbench", [], empty, root)
            duplicate["operations"] = [operation("new", "a", "add", "h2", "a.py")]
            duplicate["base_snapshot_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "CAS"):
                candidate_contract.validate_bundle(duplicate, "dbbench", [], empty, root)
            active = [{"mechanism_id": "a", "revision": 2, "target_layer": "h2",
                       "artifact_path": "versions/x.py", "sha256": "1" * 64}]
            stale = {**duplicate, "base_snapshot_sha256": driver.snapshot_sha256(active),
                     "operations": [operation("mod", "a", "modify", "h2", "a.py",
                                              {**active[0], "revision": 1})]}
            with self.assertRaisesRegex(ValueError, "stale"):
                candidate_contract.validate_bundle(stale, "dbbench", active,
                                                   driver.snapshot_sha256(active), root)

    def test_accepted_artifact_and_manifest_tampering_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            state = driver.initial_state("webshop")
            proposal = run / "p"
            artifact(proposal / "artifacts/a.py", "h2")
            path = bundle(proposal / "bundle.json", "webshop", state["active_snapshot_sha256"],
                          "immutable", [operation("add", "alpha", "add", "h2", "a.py")])
            parsed, _, _ = driver.stage_bundle(run, 1, path, state)
            active = driver.commit_bundle(run, parsed, state)
            state.update(active_snapshot=active, active_snapshot_sha256=driver.snapshot_sha256(active))
            manifest = driver.write_active_chain(run, state)
            value = json.loads(manifest.read_text())
            value["records"][0]["revision"] = 99
            manifest.write_text(json.dumps(value))
            with self.assertRaisesRegex(RuntimeError, "manifest hash"):
                agentbench_chain.layer_paths(manifest)
            (run / active[0]["artifact_path"]).write_text("tampered")
            with self.assertRaisesRegex(RuntimeError, "tampered"):
                driver.verify_active(run, state)

    def test_commit_uses_hash_pinned_evaluated_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            state = driver.initial_state("alfworld")
            proposal = run / "proposal"
            source = artifact(proposal / "artifacts/a.py", "h2")
            path = bundle(proposal / "bundle.json", "alfworld", state["active_snapshot_sha256"],
                          "evaluated_bytes", [operation("add_a", "alpha", "add", "h2", "a.py")])
            parsed, _, _ = driver.stage_bundle(run, 1, path, state)
            evaluated = Path(parsed["operations"][0]["artifact_path"])
            expected = evaluated.read_text()
            source.write_text("TARGET_LAYER='h2'\nclass Harness:\n def h2(self,m,h): return {'role':'assistant'}\n")
            active = driver.commit_bundle(run, parsed, state)
            self.assertEqual((run / active[0]["artifact_path"]).read_text(), expected)

        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            state = driver.initial_state("alfworld")
            proposal = run / "proposal"
            artifact(proposal / "artifacts/a.py", "h2")
            path = bundle(proposal / "bundle.json", "alfworld", state["active_snapshot_sha256"],
                          "staged_tamper", [operation("add_a", "alpha", "add", "h2", "a.py")])
            parsed, _, _ = driver.stage_bundle(run, 1, path, state)
            Path(parsed["operations"][0]["artifact_path"]).write_text("tampered")
            with self.assertRaisesRegex(RuntimeError, "staged artifact changed"):
                driver.commit_bundle(run, parsed, state)

    def test_cached_proposal_input_and_output_mismatch_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            workspace = run / "workspace"
            workspace.mkdir()
            (workspace / "history.json").write_text("[]")
            cache = run / "proposals/round_01"
            cache.mkdir(parents=True)
            (cache / "bundle.json").write_text("{}")
            protocol.atomic_json(cache / "cache.json", {"input_sha256": "wrong", "outputs": {}})
            with self.assertRaisesRegex(RuntimeError, "input mismatch"):
                driver.propose("alfworld", 1, workspace, run, driver.initial_state("alfworld"))

    def test_complete_interrupted_bundle_transaction_is_recovered(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            state = driver.initial_state("alfworld")
            proposal = run / "proposal"
            artifact(proposal / "artifacts/a.py", "h2")
            path = bundle(proposal / "bundle.json", "alfworld", state["active_snapshot_sha256"],
                          "recoverable", [operation("add_a", "alpha", "add", "h2", "a.py")])
            parsed, _, _ = driver.stage_bundle(run, 1, path, state)
            interrupted = run / "versions/recoverable.tmp"
            interrupted.mkdir(parents=True)
            shutil.copy2(parsed["operations"][0]["artifact_path"], interrupted / "add_a.py")
            active = driver.commit_bundle(run, parsed, state)
            self.assertFalse(interrupted.exists())
            self.assertTrue((run / active[0]["artifact_path"]).is_file())


class HookContractTests(unittest.TestCase):
    def test_exact_hook_h3_h2_hint_budget_and_fresh_instances(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            h2 = artifact(root / "h2.py", "h2", "message['content']='ok'; return message")
            h3 = artifact(root / "h3.py", "h3",
                          "tools[0]['function']['description']='new'; return tools")
            h5 = artifact(root / "h5.py", "h5", "return ['word ' * 120]")
            h4 = artifact(root / "h4.py", "h4", "return ['word ' * 120]")
            harness = agentbench_chain.CompositeHarness([h2, h3, h5, h4])
            tools = [{"type": "function", "function": {"name": "x", "description": "old",
                                                          "parameters": {}}}]
            self.assertEqual(harness.h3(tools)[0]["function"]["description"], "new")
            self.assertEqual(harness.h2({"role": "assistant", "content": "bad"}, [])["content"], "ok")
            self.assertEqual(len(harness.h5([])[0].split()), 120)
            self.assertEqual(len(harness.h4([], 1)[0].split()), 120)
            artifact(root / "bad.py", "h3", "tools[0]['function']['name']='bad'; return tools")
            with self.assertRaisesRegex(ValueError, "descriptions only"):
                agentbench_chain.CompositeHarness([root / "bad.py"]).h3(tools)
            artifact(root / "bad_h2.py", "h2", "return {'role':'user'}")
            artifact(root / "repair_h2.py", "h2", "return {'role':'assistant'}")
            with self.assertRaisesRegex(ValueError, "assistant dictionary"):
                agentbench_chain.CompositeHarness(
                    [root / "bad_h2.py", root / "repair_h2.py"]).h2(
                        {"role": "assistant", "content": "x"}, [])
            artifact(root / "long.py", "h4", "return ['word ' * 121]")
            over = agentbench_chain.CompositeHarness([h5, root / "long.py"])
            over.h5([])
            with self.assertRaisesRegex(ValueError, "H4 hint exceeds 120"):
                over.h4([], 1)
            records = [{"mechanism_id": "x", "revision": 1, "target_layer": "h2",
                        "artifact_path": "h2.py", "sha256": protocol.digest(h2)}]
            manifest = agentbench_chain.write_manifest(root / "manifest.json", records, root,
                                                       driver.snapshot_sha256(records))
            create = agentbench_chain.factory(manifest)
            self.assertIsNot(create().layers[0], create().layers[0])

    def test_webshop_request_history_is_bounded_without_mutation(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **_kwargs):
                words = sum(len(str(message.get("content") or "").split())
                            for message in messages)
                return list(range(words))

        history = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "old", "function": {"name": "click_action"}}]},
            {"role": "tool", "tool_call_id": "old", "content": "old old old"},
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "new", "function": {"name": "click_action"}}]},
            {"role": "tool", "tool_call_id": "new", "content": "new new new"},
            {"role": "user", "content": "new"},
        ]
        original = json.loads(json.dumps(history))
        old_domain = agentbench_evaluate.DOMAIN
        old_tokenizer = agentbench_evaluate.TOKENIZER
        try:
            agentbench_evaluate.DOMAIN = "webshop"
            agentbench_evaluate.TOKENIZER = Tokenizer()
            bounded = agentbench_evaluate._bounded_request_messages(
                history, [], 40960 - 256 - 6
            )
            self.assertEqual(history, original)
            self.assertEqual([item["role"] for item in bounded],
                             ["system", "user", "assistant", "tool", "user"])
            self.assertEqual(bounded[2]["tool_calls"][0]["id"], "new")
            self.assertEqual(bounded[3]["tool_call_id"], "new")
            agentbench_evaluate.DOMAIN = "dbbench"
            self.assertEqual(agentbench_evaluate._bounded_request_messages(
                history, [], 40960 - 256 - 6
            ), history)
        finally:
            agentbench_evaluate.DOMAIN = old_domain
            agentbench_evaluate.TOKENIZER = old_tokenizer

    def test_artifact_rejects_extra_hook_and_unsafe_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.py"
            path.write_text("import os\nTARGET_LAYER='h2'\nclass Harness:\n def h2(self,m,h): return m\n")
            with self.assertRaisesRegex(ValueError, "direct module imports"):
                candidate_contract.validate_source(path, "h2")
            path.write_text("import dataclasses\nTARGET_LAYER='h2'\nclass Harness:\n def h2(self,m,h): return dataclasses.sys.modules['builtins']\n")
            with self.assertRaisesRegex(ValueError, "direct module imports"):
                candidate_contract.validate_source(path, "h2")
            path.write_text("from dataclasses import sys\nTARGET_LAYER='h2'\nclass Harness:\n def h2(self,m,h): return m\n")
            with self.assertRaisesRegex(ValueError, "import not allowed"):
                candidate_contract.validate_source(path, "h2")
            path.write_text("from re import search\nTARGET_LAYER='h2'\nclass Harness:\n def h2(self,m,h):\n  search('x','x')\n  return m\n")
            candidate_contract.validate_source(path, "h2")
            path.write_text("from re import search as matcher\nTARGET_LAYER='h2'\nclass Harness:\n def h2(self,m,h): return matcher.__globals__\n")
            with self.assertRaisesRegex(ValueError, "dunder access"):
                candidate_contract.validate_source(path, "h2")
            path.write_text("TARGET_LAYER='h2'\nclass Harness:\n def h2(self,m,h): return m\n def h3(self,t): return t\n")
            with self.assertRaisesRegex(ValueError, "exactly one"):
                candidate_contract.validate_source(path, "h2")
            path.write_text("TARGET_LAYER='h2'\nclass Harness:\n def h2(self,m): return m\n")
            with self.assertRaisesRegex(ValueError, "invalid signature"):
                candidate_contract.validate_source(path, "h2")
            path.write_text("TARGET_LAYER='h2'\nclass Harness:\n async def h2(self,m,h): return m\n")
            with self.assertRaisesRegex(ValueError, "invalid signature"):
                candidate_contract.validate_source(path, "h2")
            path.write_text("raise SystemExit()\nTARGET_LAYER='h2'\nclass Harness:\n def h2(self,m,h): return m\n")
            with self.assertRaisesRegex(ValueError, "SystemExit"):
                candidate_contract.validate_source(path, "h2")
            path.write_text("exit()\nTARGET_LAYER='h2'\nclass Harness:\n def h2(self,m,h): return m\n")
            with self.assertRaisesRegex(ValueError, "exit"):
                candidate_contract.validate_source(path, "h2")

    def test_artifact_rejects_inheritance_decorators_and_dynamic_hooks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "artifacts/bad.py"
            path.parent.mkdir()
            path.write_text("TARGET_LAYER='h2'\nclass Harness(object):\n def h2(self,m,h): return m\n")
            with self.assertRaisesRegex(ValueError, "inheritance"):
                candidate_contract.validate_source(path, "h2")
            path.write_text("TARGET_LAYER='h2'\nclass Harness:\n @staticmethod\n def h2(m,h): return m\n")
            with self.assertRaisesRegex(ValueError, "decorators"):
                candidate_contract.validate_source(path, "h2")
            path.write_text("TARGET_LAYER='h2'\nclass Harness:\n def h2(self,m,h): return m\nHarness.h3=lambda self,t:t\n")
            with self.assertRaisesRegex(ValueError, "exactly one callable"):
                candidate_contract.load_artifact(path, "h2")
            state = driver.initial_state("alfworld")
            proposal = bundle(root / "bundle.json", "alfworld", state["active_snapshot_sha256"],
                              "dynamic_hook", [operation("add_bad", "bad", "add", "h2", "bad.py")])
            with self.assertRaisesRegex(ValueError, "exactly one callable"):
                candidate_contract.validate_bundle(json.loads(proposal.read_text()), "alfworld", [],
                                                     state["active_snapshot_sha256"], root)
            path.write_text("TARGET_LAYER='h2'\nmissing_name\nclass Harness:\n def h2(self,m,h): return m\n")
            with self.assertRaisesRegex(ValueError, "runtime validation failed: NameError"):
                candidate_contract.validate_bundle(json.loads(proposal.read_text()), "alfworld", [],
                                                     state["active_snapshot_sha256"], root)


class StateMachineTests(unittest.TestCase):
    def test_three_domains_five_train_only_rounds_atomic_acceptance(self):
        historical = protocol.validate_protocol(check_files=False)
        candidate_scores = {1: .5, 2: .6, 3: .6, 4: .7, 5: .6}
        baseline_scores = {0: .5, 1: .4, 2: .3}
        calls = []

        def evaluate(domain, split, phase, chain, batch):
            self.assertEqual(split, "train")
            calls.append((domain, split, phase, batch))
            ids = protocol.task_ids(historical["domains"][domain], batch, split)
            value = candidate_scores[int(phase[-2:])] if phase.startswith("candidate_") else baseline_scores[batch]
            successes = round(value * len(ids))
            outcomes = {str(item): int(index < successes) for index, item in enumerate(ids)}
            return {"domain": domain, "split": split, "phase": phase, "batch": batch,
                    "accuracy": sum(outcomes.values()) / len(outcomes), "per_task": outcomes}

        def propose(domain, round_number, workspace, run_dir, state):
            root = run_dir / "mock" / f"round_{round_number:02d}"
            artifact(root / "artifacts/a.py", "h2")
            return bundle(root / "bundle.json", domain, state["active_snapshot_sha256"],
                          f"{domain}_round_{round_number}",
                          [operation(f"add_{round_number}", f"mechanism_{round_number}",
                                     "add", "h2", "a.py")])

        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(driver, "HERE", Path(temporary)), \
                patch.object(driver, "validate_protocol", return_value=historical):
            shutil.copy2(HERE / "protocol.json", Path(temporary) / "protocol.json")
            for domain in protocol.DOMAINS:
                state = driver.run_domain(domain, evaluate=evaluate, propose_bundle=propose,
                                          do_preflight=lambda path: {"validation": "pass"})
                self.assertEqual([row["accepted"] for row in state["rounds"]],
                                 [False, True, False, True, False])
                self.assertEqual([row["train_batch"] for row in state["rounds"]], [0, 1, 1, 2, 2])
                self.assertEqual(state["batch"], 3)
                self.assertEqual(len(state["active_snapshot"]), 2)
                self.assertEqual([item["round"] for item in state["operation_log"]], [2, 4])
                self.assertTrue(all(item["operations"] for item in state["operation_log"]))
                versions = Path(temporary) / domain / "versions"
                self.assertEqual(sorted(path.name for path in versions.iterdir()),
                                 [f"{domain}_round_2", f"{domain}_round_4"])
                history = json.loads((Path(temporary) / domain /
                                      "proposer/round_05/history.json").read_text())
                self.assertNotIn("test", json.dumps(history).lower())
                self.assertNotIn("meta", json.dumps(history).lower())
            self.assertTrue(all(split == "train" for _, split, _, _ in calls))

    def test_rejected_bundle_commits_no_artifacts(self):
        # Covered through all three domains above: rounds 1, 3, and 5 are rejected.
        self.assertFalse(driver.accept({"accuracy": 0, "per_task": {"x": 0}},
                                       {"accuracy": 0, "per_task": {"x": 0}}))


class ReportingTests(unittest.TestCase):
    def test_evaluator_rejects_unknown_split_before_loading_tasks(self):
        with self.assertRaisesRegex(ValueError, "split must be train or test"):
            agentbench_evaluate.evaluate("alfworld", "testing", "probe", Path("missing"))

    def test_reporting_refuses_before_all_domains_complete_without_reading_meta(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(status, "load_meta_results") as load_meta:
                with self.assertRaisesRegex(RuntimeError, "refused"):
                    status.final_report(root, evaluate=lambda *args: {})
                load_meta.assert_not_called()

    def test_evaluator_test_gate_requires_matching_three_domain_seal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copy2(HERE / "protocol.json", root / "protocol.json")
            with self.assertRaisesRegex(RuntimeError, "before training seal"):
                agentbench_evaluate.validate_test_gate(root)
            for domain in protocol.DOMAINS:
                state = driver.initial_state(domain)
                state.update(status="complete", round=5,
                             rounds=[{"round": number} for number in range(1, 6)])
                protocol.atomic_json(root / domain / "state.json", state)
                protocol.atomic_json(root / domain / "run_config.json",
                                     driver.run_fingerprint(domain))
            with patch.object(driver, "verify_training_history"):
                status.seal_training(root)
            agentbench_evaluate.validate_test_gate(root)
            chain_root = root / "arbitrary_chain"
            candidate = artifact(chain_root / "a.py", "h2")
            records = [{"mechanism_id": "unsealed", "revision": 1, "target_layer": "h2",
                        "artifact_path": "a.py", "sha256": protocol.digest(candidate)}]
            manifest = agentbench_chain.write_manifest(chain_root / "manifest.json", records,
                                                       chain_root, driver.snapshot_sha256(records))
            with patch.object(agentbench_evaluate, "HERE", root), \
                    self.assertRaisesRegex(RuntimeError, "sealed active snapshot"):
                agentbench_evaluate.evaluate("alfworld", "test", "final_test", manifest)
            run_config_path = root / "alfworld/run_config.json"
            run_config = json.loads(run_config_path.read_text())
            run_config["rounds"] = 6
            protocol.atomic_json(run_config_path, run_config)
            with self.assertRaisesRegex(RuntimeError, "run fingerprint mismatch"):
                agentbench_evaluate.validate_test_gate(root)
            protocol.atomic_json(run_config_path, driver.run_fingerprint("alfworld"))
            state_path = root / "alfworld/state.json"
            state = json.loads(state_path.read_text())
            state["rounds"][0]["tampered"] = True
            protocol.atomic_json(state_path, state)
            with self.assertRaisesRegex(RuntimeError, "seal mismatch"):
                agentbench_evaluate.validate_test_gate(root)

    def test_episode_cache_requires_schema_index_and_config_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "episode_7.json"
            protocol.atomic_json(path, {"episode_schema_version": 1, "index": 7,
                                        "config_sha256": "abc", "success": 1,
                                        "infrastructure_error": False})
            self.assertTrue(agentbench_evaluate._episode_valid(path, 7, "abc"))
            self.assertFalse(agentbench_evaluate._episode_valid(path, 8, "abc"))
            self.assertFalse(agentbench_evaluate._episode_valid(path, 7, "changed"))
            converted = agentbench_evaluate._jsonable({"value": __import__("types").SimpleNamespace(x=1)})
            self.assertEqual(converted, {"value": {"x": 1}})
            body = {"ids": [7], "domain": "alfworld"}
            config = {**body, "config_sha256": protocol.object_digest(body)}
            protocol.atomic_json(path.parent / "config.json", config)
            row = json.loads(path.read_text())
            row.update(config_sha256=config["config_sha256"], tokens=2, turns=3)
            protocol.atomic_json(path, row)
            summary = {**config, "n": 1, "successes": 1, "accuracy": 1.0,
                       "infra_errors": 0, "tokens": 2, "turns": 3, "per_task": {"7": 1}}
            agentbench_evaluate.validate_cached_summary(path.parent, summary)
            config["domain"] = "tampered"
            protocol.atomic_json(path.parent / "config.json", config)
            with self.assertRaisesRegex(RuntimeError, "configuration mismatch"):
                agentbench_evaluate.validate_cached_summary(path.parent, summary)


class ProposerIsolationTests(unittest.TestCase):
    def test_model_router_tools_restrict_paths_and_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "input.txt").write_text("frozen evidence")
            (root / "artifacts").mkdir()
            listed = driver._execute_proposer_tool(root, "glob_files", {"pattern": "**/*"})
            self.assertEqual([item["path"] for item in listed], ["input.txt"])
            with self.assertRaisesRegex(ValueError, "relative"):
                driver._execute_proposer_tool(root, "read_file", {"path": "/etc/passwd"})
            with self.assertRaisesRegex(ValueError, "restricted"):
                driver._execute_proposer_tool(
                    root, "write_file", {"path": "input.txt", "content": "changed"})
            driver._execute_proposer_tool(
                root, "write_file", {"path": "artifacts/a.py", "content": "safe"})
            self.assertEqual((root / "artifacts/a.py").read_text(), "safe")

    def test_history_prompt_and_frozen_hash_exclude_test_meta_and_outputs(self):
        cleaned = candidate_contract.public_history([
            {"round": 1, "test_score": .9, "meta_result": .4,
             "nested": {"training": 1, "test_private": 2}}
        ])
        self.assertEqual(cleaned, [{"round": 1, "operations": []}])
        text = candidate_contract.prompt("alfworld", 2, Path("/tmp/work"))
        self.assertIn("ANY NONZERO", text)
        self.assertIn("add,\nmodify, or delete", text)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "input.json").write_text("one")
            (root / "bundle.json").write_text("output")
            (root / "artifacts").mkdir()
            (root / "artifacts/x.py").write_text("output")
            self.assertEqual(driver._workspace_hashes(root), {"input.json": protocol.digest(root / "input.json")})

    def test_validator_rejection_is_recorded_and_fed_to_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            workspace = run / "workspace"
            workspace.mkdir(parents=True)
            state = driver.initial_state("alfworld")
            prompts = []

            def fake_agent(root, instruction):
                prompts.append(instruction)
                artifact(root / "artifacts/a.py", "h2")
                proposal = bundle(root / "bundle.json", "alfworld",
                                  state["active_snapshot_sha256"], "retry_bundle",
                                  [operation("add_a", "alpha", "add", "h2", "a.py")])
                if len(prompts) == 1:
                    value = json.loads(proposal.read_text())
                    value["unexpected"] = True
                    proposal.write_text(json.dumps(value))
                return {"tool_calls": 2, "final_content": "done", "usage": {},
                        "message_count": 4}

            with patch.object(driver, "_run_model_router_agent", side_effect=fake_agent):
                result = driver.propose("alfworld", 1, workspace, run, state)

            self.assertTrue(result.is_file())
            first = json.loads(
                (run / "proposals/model_router_01_attempt_01.json").read_text())
            self.assertIn("exactly the schema-v2 fields", first["validation_error"])
            self.assertIn(first["validation_error"], prompts[1])

    def test_training_evidence_is_anonymized_and_excludes_private_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            domain = Path(temporary) / "alfworld"
            workspace = domain / "proposer/round_01"
            source = domain / "evals/batch_00_incumbent"
            source.mkdir(parents=True)
            protocol.atomic_json(source / "episode_42.json", {
                "index": 42, "success": 0, "status": "completed", "turns": 3,
                "history": [{"role": "user", "content": "public task"}],
                "calls": [], "rewards": [], "task_result_private": {"oracle": "secret"},
            })
            driver._export_evidence(workspace, "batch_00_incumbent", {
                "domain": "alfworld", "accuracy": 0.0, "ids": [42],
                "per_task": {"42": 0}, "test_private": "hidden",
            })
            evidence = workspace / "evidence/batch_00_incumbent"
            summary = json.loads((evidence / "summary.json").read_text())
            trajectories = json.loads((evidence / "public_trajectories.json").read_text())
            self.assertNotIn("ids", summary)
            self.assertNotIn("per_task", summary)
            serialized = json.dumps(trajectories)
            self.assertNotIn("42", serialized)
            self.assertNotIn("oracle", serialized)
            self.assertIn("public task", serialized)


if __name__ == "__main__":
    unittest.main()
