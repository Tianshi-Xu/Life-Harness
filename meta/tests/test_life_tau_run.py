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

    def test_round_two_branch_reuses_train_only_and_retains_prior_attempts(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent, child, reference = [Path(tmp) / name for name in ("parent", "child", "reference")]
            ids = [f"task_{n}" for n in range(50)]
            spec = {"batches": [{"task_ids": ids} for _ in range(6)]}
            save(reference / "protocol.json", {"domains_spec": {"telecom": spec}})
            rows = [{"round": n, "train_batch": n - 1, "incumbent_score": 0, "candidate_score": 0,
                     "comparison_score": 0, "ahe_score": 0, "accepted": n <= 2,
                     "active_snapshot_sha256": hooks.object_digest([]),
                     "optimizer_usage": {"total_tokens": 100}} for n in range(1, 7)]
            state = {"domain": "telecom", "round": 6, "rounds": rows, "active_snapshot": [],
                     "active_snapshot_sha256": hooks.object_digest([]), "accepted_bundle_ids": [], "operation_log": []}
            save(parent / "telecom/state.json", state)
            save(parent / "status.json", {"phase": "complete"})
            save(parent / "training_seal.json", {"snapshot": [],
                "optimizer_usage": {"total_tokens": 600},
                "state_sha256": runner.digest(parent / "telecom/state.json")})
            save(parent / "frozen_sources.json", {})
            save(parent / "protocol.json", {"accuracy_only": True, "spec": spec})
            (parent / "telecom/versions").mkdir()
            baseline = parent / "telecom/evals/initial_batch_02"
            manifest = parent / "telecom/runtime/incumbent_03/manifest.json"
            save(manifest, {})
            summary = {"n": 50, "infra_errors": 0, "successes": 0, "accuracy": 0,
                "per_task": {"native_" + hashlib.sha256(t.encode()).hexdigest(): 0 for t in ids},
                "chain_hashes": {"manifest": runner.digest(manifest)}}
            save(baseline / "life_train/summary.json", summary)
            save(baseline / "evaluation_receipt.json", {"split": "train", "ids": ids,
                "manifest_sha256": runner.digest(manifest),
                "summary_sha256": runner.digest(baseline / "life_train/summary.json")})
            original = runner.digest(parent / "telecom/state.json")

            def initialize(root):
                root.mkdir()
                protocol = {"spec": spec, "rounds": 6}
                save(root / "protocol.json", protocol)
                save(root / "frozen_sources.json", {})
                return protocol

            with patch.object(runner, "REFERENCE", reference), patch.object(runner, "initialize", side_effect=initialize), \
                    patch.object(runner.profile, "prepare") as prepare:
                protocol = runner.prepare_round_two_branch(child, parent)
                prepare.assert_called_once_with(parent / "telecom", baseline / "life_train",
                                                child / "optimizer_inputs/round_03", seed=3)
                self.assertTrue(protocol["train_only"])
                self.assertEqual(protocol["rounds"], 3)
                self.assertEqual(protocol["round_two_branch"]["seed_optimizer_tokens"], 200)
                self.assertEqual(protocol["ancestor_optimizer_tokens"], 600)
                self.assertIsNone(protocol["round_two_branch"]["ledger_sha256"])
                self.assertEqual(read(child / "telecom/state.json")["rounds"], rows[:2])
                self.assertEqual(runner.digest(parent / "telecom/state.json"), original)
                self.assertFalse((child / "telecom/evals/final_test").exists())
                self.assertEqual(read(child / "telecom/evals/initial_batch_02/life_train/summary.json"), summary)
                branches = [Path(tmp) / name for name in ('previous_a', 'previous_b')]
                for path, tokens in zip(branches, (120, 240)):
                    save(path / 'telecom/state.json', state)
                    usage = {'total_tokens': tokens, 'accounting_complete': True}
                    save(path / 'OPTIMIZER_TOKEN_USAGE.json', usage)
                    save(path / 'training_seal.json', {'snapshot': [], 'optimizer_usage': usage,
                        'state_sha256': runner.digest(path / 'telecom/state.json')})
                    save(path / 'status.json', {'phase': 'complete'})
                    save(path / 'protocol.json', {'round_two_branch': {'parent': str(parent)}})
                    save(path / 'flash_metrics.json', {'closed': True, 'gate': {'active': 0}})
                    save(path / 'frozen_sources.json', {})
                combined = runner.prepare_round_two_branch(Path(tmp) / 'combined', parent, history_from=branches)
                self.assertEqual(combined['ancestor_optimizer_tokens'], 960)
                self.assertEqual([row['root'] for row in combined['additional_train_history']], [str(p) for p in branches])
                self.assertEqual(prepare.call_args.kwargs['additional_history'], [p / 'telecom' for p in branches])
                with self.assertRaisesRegex(ValueError, 'distinct sealed branches'):
                    runner.prepare_round_two_branch(Path(tmp) / 'duplicate', parent, history_from=branches * 2)
                bad = read(baseline / "evaluation_receipt.json")
                save(baseline / "evaluation_receipt.json", {**bad, "split": "test"})
                with self.assertRaisesRegex(ValueError, "TRAIN baseline"):
                    runner.prepare_round_two_branch(Path(tmp) / "rejected", parent)
                self.assertEqual(prepare.call_count, 2)

    def test_source_upgrade_requires_exact_archived_before_and_current_after(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); parent=root/'parent'; archive=root/'archive'
            relative='meta/life_evolve/profile.py'; source=root/relative
            source.parent.mkdir(parents=True); source.write_text('before')
            before=runner.digest(source)
            original=archive/'source_before'/relative
            original.parent.mkdir(parents=True); original.write_text('before')
            save(parent/'frozen_sources.json', {str(source):before})
            source.write_text('after'); after=runner.digest(source)
            path=archive/'source_audit.json'
            save(path, {'source_files_before':{relative:before}, 'source_files_after':{relative:after}})
            with patch.object(runner,'ROOT',root):
                with self.assertRaisesRegex(ValueError,'matching before/after'):
                    runner.audit_branch_sources(parent)
                self.assertEqual(runner.audit_branch_sources(parent,path),{relative:{'before':before,'after':after}})
                source.write_text('unreviewed')
                with self.assertRaisesRegex(ValueError,'matching before/after'):
                    runner.audit_branch_sources(parent,path)
                source.write_text('after'); original.write_text('not original')
                with self.assertRaisesRegex(ValueError,'matching before/after'):
                    runner.audit_branch_sources(parent,path)
                earlier = root / 'earlier'
                saved = earlier / relative
                saved.parent.mkdir(parents=True); saved.write_text('before')
                save(path, {'source_files_before': {relative: runner.digest(original)},
                    'source_files_after': {relative: after},
                    'earlier_source_archives': [{'path': str(earlier), 'hashes': {relative: before}}]})
                self.assertEqual(runner.audit_branch_sources(parent, path), {relative: {'before': before, 'after': after}})
                saved.write_text('corrupted original')
                with self.assertRaisesRegex(ValueError, 'matching before/after'):
                    runner.audit_branch_sources(parent, path)
                native=root/'TauBench/native_scoring.py'; native.parent.mkdir(); native.write_text('old scoring')
                prior=runner.digest(native); native.write_text('new scoring')
                saved=archive/'source_before/TauBench/native_scoring.py'; saved.parent.mkdir(); saved.write_text('old scoring')
                save(parent/'frozen_sources.json',{str(native):prior})
                save(path,{'source_files_before':{'TauBench/native_scoring.py':prior},
                           'source_files_after':{'TauBench/native_scoring.py':runner.digest(native)}})
                with self.assertRaisesRegex(ValueError,'matching before/after'):
                    runner.audit_branch_sources(parent,path)

    def test_next_round_branch_keeps_accepted_snapshot_rotates_schedule_and_counts_lineage_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent, ancestor, child, reference = [Path(tmp) / name for name in ('parent', 'ancestor', 'child', 'reference')]
            spec = {'batches': [{'task_ids': [f'task_{batch}_{n}' for n in range(50)]} for batch in range(6)]}
            save(reference / 'protocol.json', {'domains_spec': {'telecom': spec}})
            source = parent / 'telecom/versions/r3/guard.py'
            source.parent.mkdir(parents=True)
            source.write_text("TARGET_LAYER='h2'\nclass Harness:\n    def h2(self,message,history):\n        return message\n")
            snapshot = [{'mechanism_id': 'guard', 'revision': 1, 'target_layer': 'h2',
                'artifact_path': 'versions/r3/guard.py', 'sha256': runner.digest(source)}]
            rows = [{'round': n, 'train_batch': n - 1, 'accepted': n == 3, 'incumbent_score': .7,
                'candidate_score': .9, 'comparison_score': .9, 'ahe_score': .86,
                'optimizer_usage': {'total_tokens': 100}} for n in range(1, 4)]
            state = {'domain': 'telecom', 'round': 3, 'rounds': rows, 'active_snapshot': snapshot,
                'active_snapshot_sha256': hooks.object_digest(snapshot), 'accepted_bundle_ids': ['r3'],
                'operation_log': [{'bundle_id': 'r3', 'parent_snapshot_sha256': hooks.object_digest([]),
                    'committed_snapshot_sha256': hooks.object_digest(snapshot), 'operations': [
                        {'op': 'add', 'operation_id': 'guard', 'mechanism_id': 'guard', 'target_layer': 'h2',
                         'artifact_sha256': runner.digest(source)}]}]}
            old = {**state, 'round': 2, 'rounds': rows[:2], 'active_snapshot': [],
                'active_snapshot_sha256': hooks.object_digest([]), 'accepted_bundle_ids': [], 'operation_log': []}
            for path, value, tokens in ((parent, state, 300), (ancestor, old, 400)):
                (path / 'telecom/versions').mkdir(parents=True, exist_ok=True)
                save(path / 'telecom/state.json', value)
                usage = {'total_tokens': tokens, 'accounting_complete': True}
                save(path / 'OPTIMIZER_TOKEN_USAGE.json', usage)
                save(path / 'training_seal.json', {'snapshot': value['active_snapshot'], 'optimizer_usage': usage,
                    'state_sha256': runner.digest(path / 'telecom/state.json')})
                save(path / 'status.json', {'phase': 'complete'})
                save(path / 'protocol.json', {'accuracy_only': True, 'spec': spec})
                save(path / 'flash_metrics.json', {'closed': True, 'gate': {'active': 0}})
                save(path / 'frozen_sources.json', {})
            prior = {'accuracy_only': True, 'spec': spec, 'ancestor_optimizer_tokens': 400,
                'round_two_branch': {'parent': str(ancestor)}}
            save(parent / 'protocol.json', prior)
            original = runner.digest(parent / 'telecom/state.json')
            def initialize(root):
                root.mkdir();save(root / 'frozen_sources.json', {})
                return {'spec': spec, 'rounds': 6}
            with patch.object(runner, 'REFERENCE', reference), patch.object(runner, 'initialize', side_effect=initialize), \
                    patch.object(runner.profile, 'prepare') as prepare:
                protocol = runner.prepare_next_round_branch(child, parent)
                self.assertEqual(read(child / 'telecom/state.json'), state)
                self.assertEqual(runner.digest(child / 'telecom/versions/r3/guard.py'), runner.digest(source))
                self.assertEqual(protocol['train_branch']['round'], 3)
                self.assertEqual(protocol['rounds'], 4)
                self.assertEqual(protocol['ancestor_optimizer_tokens'], 700)
                self.assertEqual(protocol['train_branch']['history_roots'], [str(parent), str(ancestor)])
                self.assertNotEqual(protocol['spec']['batches'][3]['task_ids'], protocol['spec']['batches'][2]['task_ids'])
                prepare.assert_not_called()
                self.assertFalse((child / 'telecom/evals').exists())
                self.assertTrue(protocol['train_only'])
                self.assertEqual(runner.digest(parent / 'telecom/state.json'), original)
                save(parent / 'protocol.json', {**prior, 'ancestor_optimizer_tokens': 401})
                with self.assertRaisesRegex(ValueError, 'lineage token total'):
                    runner.prepare_next_round_branch(Path(tmp) / 'bad_cost', parent)
                self.assertFalse((Path(tmp) / 'bad_cost').exists())
                save(parent / 'protocol.json', prior)
                save(ancestor / 'flash_metrics.json', {'closed': False, 'gate': {'active': 0}})
                with self.assertRaisesRegex(ValueError, 'closed, sealed TRAIN'):
                    runner.prepare_next_round_branch(Path(tmp) / 'live_history', parent)
                self.assertFalse((Path(tmp) / 'live_history').exists())

    def test_train_only_branch_runs_one_new_round_and_never_tests(self):
        self._train_only_branch(2)

    def test_next_round_branch_runs_only_round_four_and_no_test(self):
        self._train_only_branch(3)

    def _train_only_branch(self, inherited):
        from ahe_native import connection, metered_broker
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = {"domain": "telecom", "round": inherited, "rounds": [{"round": n} for n in range(1, inherited + 1)],
                     "active_snapshot": [], "active_snapshot_sha256": hooks.object_digest([]),
                     "accepted_bundle_ids": [], "operation_log": []}
            save(root / "telecom/state.json", state)
            save(root / "frozen_sources.json", {})
            save(root / "status.json", {"phase": "prepared"})
            dependency = root / "ongoing_tests"
            save(dependency / "protocol.json", {"global_mr_concurrency": 8})
            save(dependency / "status.json", {"mr_gate": {"active": 0}})
            save(root / "protocol.json", {"accuracy_only": True, "rounds": inherited + 1, "train_only": True,
                "wait_for_run": {"root": str(dependency), "pid": 987654,
                                 "protocol_sha256": runner.digest(dependency / "protocol.json")},
                'train_branch' if inherited == 3 else "round_two_branch": {"round": inherited, "ledger_sha256": None,
                                     "state_sha256": runner.digest(root / "telecom/state.json")}})
            r = runner.Run(root)
            broker = SimpleNamespace(close=lambda: None, ledger=SimpleNamespace(summary=lambda: {"groups": {}, "overall": {}}))
            solver = SimpleNamespace(close=lambda: None)

            def one_round(number):
                r.state["round"] = number
                r.state["rounds"].append({"round": number})
                save(root / "telecom/state.json", r.state)
                return "unpaired historical AHE gap must not stop this one-round probe"

            original_read, events = Path.read_text, []
            def read_with_dependency(path, *args, **kwargs):
                if str(path) == "/proc/987654/stat":
                    if not events:
                        events.append("live")
                        return "987654 (python) S 1"
                    raise FileNotFoundError()
                return original_read(path, *args, **kwargs)
            def credentials():
                self.assertEqual(events, ["live", "waited"])
                return "dummy", "unused"

            with patch.object(Path, "read_text", new=read_with_dependency), \
                    patch.object(runner.time, "sleep", side_effect=lambda _: events.append("waited")) as sleep, \
                    patch.object(connection, "credentials", side_effect=credentials), \
                    patch.object(metered_broker, "Broker", return_value=broker), \
                    patch.object(runner, "SolverProxy", return_value=solver), \
                    patch.object(r, "round", side_effect=one_round) as propose, patch.object(r, "evaluate") as evaluate:
                r.run()
                sleep.assert_called_once_with(30)
                propose.assert_called_once_with(inherited + 1)
                evaluate.assert_not_called()
            self.assertEqual(read(root / "status.json")["new_rounds"], 1)
            self.assertTrue(read(root / "status.json")["train_only"])
            self.assertFalse((root / "FINAL_RESULTS.json").exists())
            self.assertTrue((root / "training_seal.json").exists())
            # A completed paid branch must not silently replay.
            with self.assertRaisesRegex(RuntimeError, "explicit audited continuation"):
                r.run()


if __name__ == "__main__":
    unittest.main()
