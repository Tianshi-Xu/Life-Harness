"""Regression checks for model-only transfer without altering six-round policy."""
import copy
import json
from concurrent.futures import Future
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from meta import harness_hooks
from meta import life_model_transfer as transfer
from meta.tau_comparison.common import digest, save


class ModelTransferTests(unittest.TestCase):
    def tools(self):
        return [{'type': 'function', 'function': {'name': name, 'description': '',
            'parameters': {'type': 'object', 'properties': {key: {'type': kind}}, 'required': [key]}}}
            for name, key, kind in [('execute_sql', 'query', 'string'),
                                    ('commit_final_answer', 'answers', 'array'),
                                    ('take_action', 'action', 'string')]]

    def message(self, name, value):
        return {'role': 'assistant', 'content': 'original explanation', 'tool_calls': [
            {'id': 'unchanged-id', 'type': 'function',
             'function': {'name': name, 'arguments': json.dumps(value)}}]}

    def test_compatibility_only_wraps_single_parameter_values(self):
        for name, value, expected in [('execute_sql', 'SELECT `A` FROM `T`', {'query': 'SELECT `A` FROM `T`'}),
                ('execute_sql', ['SELECT 1'], {'query': 'SELECT 1'}),
                ('commit_final_answer', ['exact cell'], {'answers': ['exact cell']}),
                ('take_action', 'go to cabinet 1', {'action': 'go to cabinet 1'})]:
            message = self.message(name, value)
            before = copy.deepcopy(message)
            result = transfer.normalize_tool_arguments(message, self.tools())
            self.assertEqual(message, before)
            call = result['tool_calls'][0]
            self.assertEqual(json.loads(call['function']['arguments']), expected)
            self.assertEqual(call['id'], 'unchanged-id')
            self.assertEqual(result['content'], before['content'])
            self.assertNotIn('_harness_control', result)
        for name, value in [('execute_sql', {'query': 'SELECT 1'}),
                            ('commit_final_answer', {'answers': ['unchanged']})]:
            message = self.message(name, value)
            self.assertEqual(transfer.normalize_tool_arguments(message, self.tools()), message)

    def test_ambiguous_or_invalid_arguments_return_tool_error_without_guessing(self):
        for value in (None, 42, ['SELECT 1', 'DELETE FROM T']):
            result = transfer.normalize_tool_arguments(self.message('execute_sql', value), self.tools())
            error = result['_harness_control']['messages'][0]
            self.assertEqual(error['role'], 'tool')
            self.assertEqual(error['tool_call_id'], 'unchanged-id')
            self.assertIn('Nothing was executed', error['content'])
        message = self.message('execute_sql', 'unused')
        message['tool_calls'][0]['function']['arguments'] = '{broken json'
        self.assertIn('_harness_control', transfer.normalize_tool_arguments(message, self.tools()))
        tools = self.tools()
        tools[0]['function']['parameters']['properties']['another'] = {'type': 'string'}
        self.assertIn('_harness_control', transfer.normalize_tool_arguments(
            self.message('execute_sql', 'SELECT 1'), tools))

    @unittest.skipUnless((transfer.BASELINE / '7b/dbbench/final_test/episode_0.json').is_file(),
                         'optional archive replay requires local raw episodes, not stored in Git')
    def test_saved_db_schema_failures_repair_without_sql_or_answer_changes(self):
        repaired = 0
        for path in (transfer.BASELINE / '7b/dbbench/final_test').glob('episode_*.json'):
            row = json.loads(path.read_text())
            if row.get('failure_kind') != 'model_tool_argument_schema':
                continue
            call = row['calls'][-1]
            raw = call['response']['choices'][0]['message']
            fixed = transfer.normalize_tool_arguments(raw, call['request']['tools'])
            self.assertNotIn('_harness_control', fixed)
            for before, after in zip(raw['tool_calls'], fixed['tool_calls']):
                value = json.loads(before['function']['arguments'])
                envelope = json.loads(after['function']['arguments'])
                expected = value[0] if before['function']['name'] == 'execute_sql' and isinstance(value, list) else value
                self.assertEqual(list(envelope.values())[0], expected)
            repaired += 1
        self.assertEqual(repaired, 252)

    @unittest.skipUnless((transfer.BASELINE / '7b/alfworld/final_test/episode_10.json').is_file()
                         and (transfer.OLD / 'alfworld/runtime/final_test/manifest.json').is_file(),
                         'optional archive replay requires local raw episodes and runtime manifest')
    def test_compatible_session_records_recovered_actions_for_h4(self):
        from meta.tests.test_agentbench_hooks import Backend, SWITCHES
        base = transfer.load_module('transfer_test_frozen_session', transfer.ROOT /
            'meta/experiments/meta_from_scratch_six_20260915/task_snapshot/src/server/harness/session.py')
        task = SimpleNamespace(FourHookSession=base.FourHookSession)
        manifest = transfer.OLD / 'alfworld/runtime/final_test/manifest.json'
        transfer.install_compatibility(task, 'alfworld', manifest)
        row = json.loads((transfer.BASELINE / '7b/alfworld/final_test/episode_10.json').read_text())
        backend = Backend()
        session = task.FourHookSession(backend, task.Harness(), row['calls'][0]['request']['tools'], 50, SWITCHES)
        for message in row['calls'][0]['request']['messages']:
            session.inject(message)
        observations = [m for m in row['history'] if m['role'] == 'tool']
        for i in range(6):
            backend.response = row['calls'][i]['response']['choices'][0]['message']
            output = session.sync_action()
            self.assertTrue(output.messages[0]['tool_calls'])
            self.assertEqual(backend.history[-1]['tool_calls'], output.messages[0]['tool_calls'])
            session.inject(observations[i])
        backend.response = row['calls'][6]['response']['choices'][0]['message']
        session.sync_action()
        self.assertIn('go to countertop 3', backend.requests[-1][-1]['content'])
        self.assertTrue(session.trace['h2'])
        self.assertFalse(session.harness.h2_preserve_raw_history)

    @unittest.skipUnless((transfer.OLD / 'dbbench/runtime/final_test/manifest.json').is_file(),
                         'optional selected-policy replay requires local runtime manifest')
    def test_compatible_db_session_normalizes_before_policy_and_handles_bad_json(self):
        from meta.tests.test_agentbench_hooks import Backend, SWITCHES
        base = transfer.load_module('transfer_test_db_session', transfer.ROOT /
            'meta/experiments/meta_from_scratch_six_20260915/task_snapshot/src/server/harness/session.py')
        task = SimpleNamespace(FourHookSession=base.FourHookSession)
        transfer.install_compatibility(task, 'dbbench', transfer.OLD / 'dbbench/runtime/final_test/manifest.json')
        for value in ('SELECT 1', ['SELECT 1', 'SELECT 2']):
            backend = Backend(self.message('execute_sql', value))
            session = task.FourHookSession(backend, task.DBBenchHarness(), self.tools(), 50, SWITCHES)
            session.inject({'role': 'user', 'content': 'public task'})
            output = session.sync_action()
            if isinstance(value, str):
                self.assertEqual(json.loads(output.messages[0]['tool_calls'][0]['function']['arguments']), {'query': value})
                self.assertEqual(backend.history[-1]['tool_calls'], output.messages[0]['tool_calls'])
                self.assertEqual(session.trace['h2'][-1]['before']['tool_calls'][0]['function']['arguments'], json.dumps(value))
            else:
                self.assertEqual(output.messages, [])
                self.assertEqual(backend.history[-1]['role'], 'tool')

    def test_solver_params_do_not_inherit_qwen3_thinking_or_optimizer_limits(self):
        for key, port in [('7b', 8701), ('14b', 8702)]:
            body = transfer.model_body(key)
            self.assertEqual(body['url'], f'http://127.0.0.1:{port}/v1/chat/completions')
            self.assertEqual(body['max_tokens'], 4096)
            self.assertEqual(body['temperature'], 0)
            self.assertNotIn('chat_template_kwargs', body)
        self.assertNotEqual(transfer.model_body('7b')['model'], transfer.model_body('14b')['model'])

    def test_db_web_tokenizer_and_interface_override_are_worker_local(self):
        for domain in ('dbbench', 'webshop'):
            original = {'agent': {'model': 'qwen3-4b'}, 'domains': {'frozen': True}}
            before = copy.deepcopy(original)
            installed, task = Mock(), SimpleNamespace()
            loader = Mock(return_value='correct-tokenizer')
            tokenizer = SimpleNamespace(from_pretrained=loader)
            native = SimpleNamespace(HISTORICAL=original, inject_task_module=installed)
            def initialize(d, split, manifest):
                self.assertEqual(tokenizer.from_pretrained('old-qwen3-path'), 'correct-tokenizer')
                native.inject_task_module(d, task, manifest)
            native.initialize = initialize
            with patch.object(transfer, 'native_module', return_value=native), \
                 patch.dict(sys.modules, {'transformers': SimpleNamespace(AutoTokenizer=tokenizer)}), \
                 patch.object(harness_hooks, 'install_agentbench') as current:
                transfer.initialize_worker(domain, 'test', 'manifest', '14b')
            loader.assert_called_once_with(transfer.MODELS['14b']['path'], local_files_only=True)
            self.assertEqual(original, before)
            self.assertEqual(native.HISTORICAL['agent'], transfer.model_body('14b'))
            self.assertIs(tokenizer.from_pretrained, loader)
            self.assertEqual(current.call_count, int(domain == 'webshop'))
            self.assertEqual(installed.call_count, int(domain == 'dbbench'))

    def test_alfworld_uses_requested_tokenizer_subprocess(self):
        native = SimpleNamespace(HISTORICAL={'agent': {'model': 'qwen3'}},
                                 initialize=lambda *args: native.TokenizerProxy())
        with patch.object(transfer, 'native_module', return_value=native), \
             patch.object(transfer, 'TokenizerService') as service:
            transfer.initialize_worker('alfworld', 'test', 'manifest', '7b')
            service.assert_called_once_with(transfer.MODELS['7b']['path'])

    def test_seal_and_snapshot_mismatch_rejected_before_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = {'round': 6, 'rounds': [{}] * 6, 'active_snapshot': [],
                     'active_snapshot_sha256': harness_hooks.object_digest([])}
            manifest = root / 'webshop/runtime/final/manifest.json'
            save(manifest, {'artifact_root': str(root / 'webshop')})
            save(root / 'webshop/state.json', state)
            save(root / 'training_seal.json', {'state_sha256': digest(root / 'webshop/state.json'), 'snapshot': []})
            native = SimpleNamespace(object_digest=harness_hooks.object_digest)
            with patch.object(transfer, 'WEB', root), patch.object(harness_hooks, 'manifest_records', return_value=(root, [])):
                self.assertEqual(transfer.selected('webshop', native)['interface'], 'unified-v2')
                save(root / 'webshop/state.json', {**state, 'round': 5})
                with self.assertRaises(ValueError):
                    transfer.selected('webshop', native)

    def test_child_environment_strips_credentials_and_preserves_native_java(self):
        with patch.dict(transfer.os.environ, {'MR_API_KEY': 'fake', 'DASHSCOPE_API_KEY': 'fake'}):
            env = transfer.native_environment()
        self.assertNotIn('MR_API_KEY', env)
        self.assertNotIn('DASHSCOPE_API_KEY', env)
        self.assertEqual(env['OMP_NUM_THREADS'], '1')
        self.assertEqual(Path(env['JVM_PATH']), Path(env['JAVA_HOME']) / 'lib/server/libjvm.so')
        self.assertEqual(Path(env['JAVA_HOME']), transfer.ROOT /
                         'AgentBench/.native/sysroot/usr/lib/jvm/java-11-openjdk-amd64')

    def test_fixed_denominator_counts_infra_failure_and_resumes_without_requests(self):
        native = transfer.native_module()
        class Pool:
            calls = []
            def __init__(self, **kwargs): pass
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def submit(self, function, index):
                self.calls.append(index)
                future = Future()
                future.set_result({'index': index, 'success': int(index == 1),
                    'status': 'completed' if index == 1 else 'infrastructure_error',
                    'infrastructure_error': index == 2, 'tokens': 5, 'turns': 1})
                return future
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('config.json', 'tokenizer_config.json', 'tokenizer.json', 'model.safetensors.index.json'):
                save(root / 'model' / name, {})
            choice = {'manifest': 'readonly-manifest'}
            save(root / 'protocol.json', {'domains': {'dbbench': choice}, 'source_hashes': {},
                'test_ids': {'dbbench': [1, 2]}})
            models = {'7b': {**transfer.MODELS['7b'], 'path': str(root / 'model')}}
            with patch.object(transfer, 'EXPERIMENT', root), patch.object(transfer, 'MODELS', models), \
                 patch.object(transfer, 'selected', return_value=choice), \
                 patch.object(transfer, 'native_module', return_value=native), \
                 patch.object(transfer, 'ProcessPoolExecutor', Pool):
                result = transfer.evaluate('7b', 'dbbench')
                self.assertEqual((result['n'], result['accuracy'], result['infra_errors']), (2, .5, 1))
                self.assertEqual(transfer.evaluate('7b', 'dbbench'), result)
                self.assertEqual(Pool.calls, [1, 2])

    def test_native_scalar_argument_crash_is_model_failure_not_repaired(self):
        row = {'index': 1, 'success': 0, 'status': 'task error', 'infrastructure_error': True,
            'calls': [{'response': {'choices': [{'message': {'tool_calls': [
                {'function': {'name': 'execute_sql', 'arguments': '"SELECT 1"'}}]}}]}}]}
        original = copy.deepcopy(row)
        with patch.object(transfer, '_worker', SimpleNamespace(episode=lambda i: copy.deepcopy(row))):
            result = transfer.episode_worker(1)
        self.assertEqual(result['success'], 0)
        self.assertFalse(result['infrastructure_error'])
        self.assertTrue(result['native_infrastructure_error'])
        self.assertEqual(result['calls'], original['calls'])
        row['calls'][0]['response']['choices'][0]['message']['tool_calls'][0]['function']['arguments'] = '{"query":"SELECT 1"}'
        with patch.object(transfer, '_worker', SimpleNamespace(episode=lambda i: copy.deepcopy(row))):
            self.assertTrue(transfer.episode_worker(1)['infrastructure_error'])


if __name__ == '__main__':
    unittest.main()
