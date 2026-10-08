"""Offline tests of fresh six-round integration; no model/API calls."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import composition as c
import driver as d
import protocol as p
from agentbench_chain import manifest_records, write_manifest
from agentbench_evaluate import _episode_valid, validate_test_gate

IDENTITY = "TARGET_LAYER='h2'\nclass Harness:\n    def h2(self,message,history):\n        return message\n"

class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / 'webshop'
        self.run.mkdir()
        self.state = d.initial_state('webshop')
        self.output = self.root / 'candidate'
        (self.output / 'artifacts').mkdir(parents=True)
        (self.output / 'artifacts/test.py').write_text(IDENTITY)
        self.bundle = {'schema_version': 2, 'bundle_id': 'synthetic_bundle', 'domain': 'webshop',
            'base_snapshot_sha256': p.object_digest([]), 'hypothesis': 'test machinery only',
            'operations': [{'operation_id': 'op_identity', 'mechanism_id': 'identity', 'op': 'add',
                'target_layer': 'h2', 'hypothesis': 'test', 'artifact': 'artifacts/test.py'}]}
        self.path = self.output / 'bundle.json'
        p.atomic_json(self.path, self.bundle)

    def test_empty_chain_is_policy_free_and_hash_bound(self):
        chain = c.write_active_chain(self.run, self.state)
        root, records = manifest_records(chain)
        self.assertEqual(root, self.run)
        self.assertEqual(records, [])
        c.verify_active(self.run, self.state)

    def test_stage_resume_preserves_bytes_and_commit_audit(self):
        before = p.object_digest(self.state)
        bundle, staged, chain = c.stage_bundle(self.run, 1, self.path, self.state)
        _, again, _ = c.stage_bundle(self.run, 1, self.path, self.state)
        self.assertEqual(staged, again)
        self.assertEqual(p.object_digest(self.state), before)
        records = c.commit_bundle(self.run, bundle, self.state)
        self.state.update(active_snapshot=records, active_snapshot_sha256=p.object_digest(records))
        self.state['accepted_bundle_ids'].append(bundle['bundle_id'])
        self.state['operation_log'].append({'bundle_id': bundle['bundle_id'],
            'parent_snapshot_sha256': p.object_digest([]),
            'committed_snapshot_sha256': p.object_digest(records), 'operations': d.json_operations(bundle)})
        self.state['rounds'].append({'round': 1})
        c.verify_active(self.run, self.state)
        self.assertEqual(records[0]['sha256'], staged[0]['sha256'])

    def test_staged_tampering_is_not_silently_overwritten(self):
        _, _, chain = c.stage_bundle(self.run, 1, self.path, self.state)
        root, records = manifest_records(chain)
        (root / records[0]['artifact_path']).write_text(IDENTITY + '# tampered\n')
        with self.assertRaises(RuntimeError):
            c.stage_bundle(self.run, 1, self.path, self.state)

    def test_stale_snapshot_cas_rejected(self):
        self.bundle['base_snapshot_sha256'] = 'stale'
        p.atomic_json(self.path, self.bundle)
        with self.assertRaises(ValueError):
            c.stage_bundle(self.run, 1, self.path, self.state)

    def test_fingerprint_mismatch_refuses_resume(self):
        path = self.root / 'config.json'
        p.bind_fingerprint(path, {'empty': []})
        p.bind_fingerprint(path, {'empty': []})
        with self.assertRaises(RuntimeError):
            p.bind_fingerprint(path, {'empty': ['inherited']})

    def test_test_gate_refuses_missing_or_incomplete_seal(self):
        with self.assertRaises(RuntimeError):
            validate_test_gate(self.root)
        p.atomic_json(self.root / 'training_seal.json', {})
        p.atomic_json(self.run / 'state.json', self.state)
        with self.assertRaises(RuntimeError):
            validate_test_gate(self.root)

    def test_credentials_stripped_from_native_workers(self):
        with patch.dict(os.environ, {'MR_API_KEY': 'synthetic', 'OPENAI_API_KEY': 'synthetic',
                                    'TASK_PASSWORD': 'synthetic'}):
            env = d.child_environment('webshop')
        self.assertNotIn('MR_API_KEY', env)
        self.assertNotIn('OPENAI_API_KEY', env)
        self.assertNotIn('TASK_PASSWORD', env)
        self.assertEqual(env['OMP_NUM_THREADS'], '1')

    def test_training_inputs_bind_the_exact_serialized_prior_state(self):
        state_path = self.run / 'state.json'
        prior = self.run / 'round_inputs/round_01.json'
        p.atomic_json(state_path, self.state)
        p.bind_fingerprint(prior, self.state)
        self.assertEqual(p.digest(prior), p.digest(state_path))

    def test_complete_fifty_task_strict_gate(self):
        def summary(count):
            return {'split': 'train', 'domain': 'webshop', 'n': 50, 'successes': count,
                    'accuracy': count/50, 'per_task': {str(i): int(i<count) for i in range(50)}}
        self.assertFalse(d.optimizer.strict_improvement(summary(25), summary(25), expected_task_ids=range(50)))
        self.assertTrue(d.optimizer.strict_improvement(summary(26), summary(25), expected_task_ids=range(50)))
        with self.assertRaises(ValueError):
            d.optimizer.strict_improvement(summary(26), summary(25), expected_task_ids=range(10))

    def test_six_normal_proposal_failures_do_not_abort_or_invent_candidate_scores(self):
        from unittest.mock import Mock
        manifest = p.validate_protocol(check_files=False)
        def phase(domain, split, name, chain, batch):
            self.assertEqual(split, 'train')
            self.assertTrue(name.endswith('incumbent'))
            ids = p.task_ids(manifest['domains'][domain], batch, 'train')
            directory = self.root/domain/'evals'/name
            config_body = {'ids': ids, 'domain': domain, 'split': split, 'phase': name,
                           'snapshot_sha256': p.object_digest([]),
                           'chain_hashes': {'manifest': p.digest(chain)}}
            config = {**config_body, 'config_sha256': p.object_digest(config_body)}
            p.atomic_json(directory/'config.json', config)
            for i in ids:
                p.atomic_json(directory/f'episode_{i}.json', {'episode_schema_version': 1,
                    'index': i, 'config_sha256': config['config_sha256'], 'success': 0,
                    'status': 'completed', 'infrastructure_error': False, 'turns': 0, 'tokens': 0,
                    'history': [{'role': 'user', 'content': 'Synthetic machinery-only task'}], 'calls': []})
            summary = {**config, 'n': 50, 'successes': 0, 'accuracy': 0.0, 'infra_errors': 0,
                       'tokens': 0, 'turns': 0, 'per_task': {str(i): 0 for i in ids}}
            p.atomic_json(directory/'summary.json', summary)
            return summary
        completion = Mock(return_value={'choices': [{'finish_reason': 'length',
            'message': {'role': 'assistant', 'content': '', 'reasoning_content': 'synthetic truncated reasoning'}}],
            'usage': {'completion_tokens': 4096}})
        with patch.object(d, 'HERE', self.root), patch.object(d, 'validate_protocol', return_value=manifest), \
             patch.object(d, 'run_fingerprint', return_value={'test': True}), \
             patch.object(d, 'phase', side_effect=phase), \
             patch.object(d, 'phase_directory', side_effect=lambda domain, name:self.root/domain/'evals'/name), \
             patch.object(d, 'render_training_report'), patch.object(d, 'progress'):
            state = d.run_domain('webshop', completion)
            d.verify_history('webshop', state)
        self.assertEqual(state['round'], 6)
        self.assertEqual(state['status'], 'complete')
        self.assertEqual(state['active_snapshot'], [])
        self.assertEqual(completion.call_count, 6)
        self.assertTrue(all(r['outcome']=='proposal_failed' and r['candidate_score'] is None
                            and r['accepted'] is False for r in state['rounds']))

    def test_solver_context_limit_is_an_episode_failure_not_infrastructure(self):
        import agentbench_evaluate as e
        from unittest.mock import Mock
        from types import SimpleNamespace
        task = SimpleNamespace(tools=[], sync_start_sample=Mock(side_effect=e.ContextLimit('normal context failure')))
        with patch.object(e, 'TASK', task), patch.object(e, 'DOMAIN', 'webshop'):
            row = e.episode(123)
        self.assertEqual(row['success'], 0)
        self.assertFalse(row['infrastructure_error'])
        self.assertEqual(row['status'], 'context_limit')

if __name__ == '__main__':
    unittest.main()
