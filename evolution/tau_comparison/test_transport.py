"""Offline mixed-provider/accounting/native split regression; no paid calls."""
import ast
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .common import AHE_NATIVE, child_env, public_message, public_simulation, summarize_rows, user_model
sys.path.insert(0, str(AHE_NATIVE))
from ahe_native.backend import native_env
from ahe_native.metered_broker import Broker
from ahe_native.token_accounting import TokenLedger, decode_scope, scoped_url
from ahe_native.tau_training import make_tau_schedule, scheduled_tau_batch
from .meta_port import validate_candidate
from .run import build_protocol, select_optimizer_groups


class TransportTests(unittest.TestCase):
    def test_native_tool_calls_export_actual_name_and_arguments(self):
        message = public_message({'role': 'assistant', 'content': None, 'tool_calls': [
            {'id': 'call', 'name': 'native_tool', 'arguments': {'public': 'value'},
             'requestor': 'assistant', 'hidden': 'PRIVATE_SENTINEL'}]})
        self.assertEqual(message['tool_calls'][0]['function']['name'], 'native_tool')
        self.assertEqual(json.loads(message['tool_calls'][0]['function']['arguments']), {'public': 'value'})
        self.assertNotIn('PRIVATE_SENTINEL', json.dumps(message))
        self.assertEqual(public_message({'role': 'tool', 'id': 'call', 'content': 'ok'})['tool_call_id'], 'call')

    def test_solver_budget_is_shared_and_not_a_deepseek_limit(self):
        from .solver_proxy import SolverProxy
        with patch('evolution.tau_comparison.solver_proxy.subprocess.Popen') as process, \
                patch('evolution.tau_comparison.solver_proxy.httpx.Client') as client, tempfile.TemporaryDirectory():
            counter = process.return_value
            counter.poll.return_value = None
            counter.stdout.readline.side_effect = [json.dumps({'length': 28672}), json.dumps({'length': 28673})]
            response = client.return_value.post.return_value
            response.status_code, response.content = 200, b'{"choices":[]}'
            proxy = SolverProxy()
            try:
                request = Request(proxy.url + '/chat/completions', headers={'Content-Type': 'application/json'},
                    data=json.dumps({'model': 'qwen3-4b', 'max_tokens': 2048,
                                     'messages': [{'role': 'user', 'content': 'public'}]}).encode())
                with urlopen(request, timeout=5) as result: self.assertEqual(result.status, 200)
                self.assertEqual(client.return_value.post.call_args.kwargs['json']['max_tokens'], 4096)
                with self.assertRaises(HTTPError) as caught: urlopen(request, timeout=5)
                self.assertIn(b'solver_context_limit_32768', caught.exception.read())
                self.assertEqual(client.return_value.post.call_count, 1)
                request.data = json.dumps({'model': 'deepseek-v4-flash', 'messages': []}).encode()
                with self.assertRaises(HTTPError): urlopen(request, timeout=5)
            finally:
                proxy.close()

    def test_routing_and_frozen_native_capacity(self):
        protocol = build_protocol('offline_test')
        self.assertEqual(protocol['global_mr_concurrency'], 8)
        self.assertFalse(protocol['life_agentbench_started'])
        self.assertEqual(protocol['domains_spec']['airline']['train_trials'], 2)
        self.assertEqual(len(protocol['domains_spec']['airline']['batches'][0]['task_ids']), 30)
        self.assertEqual(user_model('retail'), 'openai/qwen3.8-flash')
        self.assertEqual(user_model('airline'), 'openai/qwen3.8-flash')
        self.assertEqual(user_model('telecom'), 'openai/deepseek-v4-flash')
        with self.assertRaises(ValueError): user_model('unknown')

    def test_both_native_ports_keep_simulator_output_separate_from_solver(self):
        ahe = ast.parse((AHE_NATIVE / 'ahe_native/tau_bridge.py').read_text())
        user_args = next(node.value for node in ast.walk(ahe) if isinstance(node, ast.Assign)
                         and any(isinstance(t, ast.Name) and t.id == 'user_args' for t in node.targets))
        ahe_output = next(v for k, v in zip(user_args.keys, user_args.values)
                          if isinstance(k, ast.Constant) and k.value == 'max_tokens')
        meta = ast.parse(Path(__file__).with_name('meta_evaluate.py').read_text())
        argv = next(node.value.elts for node in ast.walk(meta) if isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Attribute) and t.attr == 'argv' for t in node.targets))
        meta_output = argv[next(i for i, n in enumerate(argv) if isinstance(n, ast.Constant)
                                and n.value == '--user-max-tokens') + 1]
        for domain, expected in (('airline', 4096), ('retail', 4096), ('telecom', 100000)):
            for expression in (ahe_output, meta_output):
                with self.subTest(domain=domain, expression=ast.unparse(expression)):
                    value = eval(compile(ast.Expression(expression), '<offline-budget>', 'eval'),
                                 {'__builtins__': {}},
                                 {'user_model': user_model(domain), 'request': {'domain': domain}})
                    self.assertEqual(int(value), expected)

    def test_simulator_and_judge_scopes_are_distinct(self):
        urls = [scoped_url('http://localhost/v1', experiment='offline', domain='retail', round=2, role=role)
                for role in ('user_simulator', 'judge')]
        self.assertNotEqual(*urls)
        self.assertEqual(decode_scope(urls[0].removeprefix('http://localhost') + '/chat/completions')['role'],
                         'user_simulator')
        with self.assertRaises(ValueError):
            scoped_url('http://localhost/v1', experiment='offline', domain='retail', round=2, role='solver')

    def test_optimizer_usage_excludes_large_simulator_and_judge_costs(self):
        with tempfile.TemporaryDirectory() as root:
            ledger = TokenLedger(Path(root) / 'requests.jsonl')
            for role, tokens in [('evolve', 30), ('debugger', 50), ('user_simulator', 500000), ('judge', 200000)]:
                ident = ledger.start(dict(experiment='same', domain='retail', round=1, role=role),
                                     model='deepseek-v4-flash', attempt=1, stream=False)
                ledger.finish(ident, status=200, usage={'prompt_tokens': tokens, 'completion_tokens': 5})
            result = select_optimizer_groups(ledger.summary(), 'same', 'retail', 1)
            self.assertEqual(result['total_tokens'], 90)
            self.assertEqual(result['attempts'], 2)
            self.assertTrue(result['accounting_complete'])
            self.assertEqual(result['by_role']['evolve']['total_tokens'], 35)
            self.assertEqual(select_optimizer_groups(ledger.summary(), 'other')['total_tokens'], 0)

    def test_missing_optimizer_usage_remains_unknown(self):
        with tempfile.TemporaryDirectory() as root:
            ledger = TokenLedger(Path(root) / 'requests.jsonl')
            ident = ledger.start(dict(experiment='same', domain='airline', round=1, role='evolve'),
                                 model='deepseek-v4-flash', attempt=1, stream=False)
            ledger.finish(ident, status=502)
            self.assertFalse(select_optimizer_groups(ledger.summary(), 'same')['accounting_complete'])

    def test_native_user_route_has_no_mr_pacing_or_real_credentials(self):
        with patch.dict('os.environ', {'AHE_TAU_USER_BASE_RETAIL': 'http://localhost/v1',
            'AHE_TAU_USER_MODEL_RETAIL': 'openai/qwen3.8-flash', 'DASHSCOPE_API_KEY': 'fake-private',
            'MR_API_KEY': 'also-fake'}, clear=True):
            env = native_env('retail', scope={'experiment': 'offline', 'domain': 'retail', 'round': 1})
        self.assertEqual(env['TAU_USER_LLM_MIN_INTERVAL_SECONDS'], '0')
        self.assertEqual(env['AHE_USER_API_KEY'], 'EMPTY')
        self.assertNotIn('DASHSCOPE_API_KEY', env)
        self.assertNotIn('MR_API_KEY', env)
        self.assertNotEqual(env['AHE_TAU_USER_API_BASE'], env['AHE_TAU_JUDGE_API_BASE'])

    def test_historical_native_env_defaults_unchanged(self):
        with patch.dict('os.environ', {}, clear=True):
            env = native_env('retail')
        self.assertEqual(env['TAU_USER_LLM_MIN_INTERVAL_SECONDS'], '3.75')
        self.assertNotIn('AHE_TAU_USER_MODEL', env)

    def test_child_env_strips_host_keys(self):
        with patch.dict('os.environ', {'DASHSCOPE_API_KEY': 'fake-private', 'MR_API_KEY': 'also-fake'}, clear=True):
            env = child_env()
        self.assertNotIn('DASHSCOPE_API_KEY', env)
        self.assertNotIn('MR_API_KEY', env)
        self.assertEqual(env['OPENAI_API_KEY'], 'EMPTY')

    def test_tau_schedule_uses_frozen_overlapping_train_batches(self):
        protocol = build_protocol('offline')
        historical = {'domains': protocol['domains_spec'], 'sha256': protocol['historical_manifest_sha256']}
        schedule = make_tau_schedule(historical)
        ids = scheduled_tau_batch({'domain': 'airline', 'iteration': 6, 'k': 2}, schedule, historical)
        self.assertEqual(ids, historical['domains']['airline']['batches'][5]['task_ids'])
        modified = copy.deepcopy(schedule)
        modified['domains']['airline']['batches'][0]['task_ids'][0] = 'hidden_test'
        with self.assertRaises(ValueError):
            scheduled_tau_batch({'domain': 'airline', 'iteration': 1, 'k': 2}, modified, historical)
        with self.assertRaises(ValueError):
            scheduled_tau_batch({'domain': 'airline', 'iteration': 1, 'k': 1}, schedule, historical)

    def test_fixed_denominators_include_missing_and_infra(self):
        result = summarize_rows(['A', 'B'], 2, [{'task_id': 'A', 'trial': 0, 'reward': 1},
            {'task_id': 'A', 'trial': 1, 'reward': 0, 'infrastructure_error': True},
            {'task_id': 'B', 'trial': 0, 'reward': 1}])
        self.assertEqual((result['successes'], result['accuracy'], result['episodes']), (2, .5, 4))
        self.assertEqual(result['infra_errors'], 2)
        self.assertEqual(result['per_task']['B'], [1, 0])

    def test_invalid_trial_or_duplicate_native_result_rejected(self):
        row = {'task_id': 'A', 'trial': 0, 'reward': 1}
        for rows in ([row, row], [{**row, 'trial': 2}], [{**row, 'reward': .5}],
                     [{**row, 'infrastructure_error': True}]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                summarize_rows(['A'], 1, rows)

    def test_public_export_has_no_judge_or_user_prompt_private_fields(self):
        result = public_simulation({'task_id': 'A', 'trial': 0, 'reward_info':
            {'reward': 0, 'hidden_answer': 'PRIVATE_SENTINEL'},
            'user_system_prompt': 'PRIVATE_SENTINEL', 'messages': [
                {'role': 'user', 'content': 'public conversation', 'user_goal': 'PRIVATE_SENTINEL'},
                {'role': 'tool', 'content': 'public result', 'private': 'PRIVATE_SENTINEL'}]})
        self.assertNotIn('PRIVATE_SENTINEL', json.dumps(result))
        self.assertEqual(result['reward'], 0)

    def test_meta_candidate_rejects_private_or_released_imports_before_execution(self):
        with tempfile.TemporaryDirectory() as root, patch('subprocess.run') as execute:
            path = Path(root) / 'candidate.py'
            for source in ('import os', 'from tau2.evaluator.evaluator import EvaluationType',
                           'from tau2.utils.utils import DATA_DIR', 'import tau2\nx=tau2.__dict__'):
                path.write_text(source)
                with self.subTest(source=source), self.assertRaises(ValueError):
                    validate_candidate(path, 'retail')
            execute.assert_not_called()

    def test_ali_broker_unlimited_and_real_usage_with_no_secret_saved(self):
        with tempfile.TemporaryDirectory() as root:
            client = MagicMock()
            response = MagicMock()
            response.status_code = 200
            response.headers = {'Content-Type': 'application/json'}
            response.iter_bytes.return_value = [json.dumps({'choices': [{'message': {'role': 'assistant', 'content': 'ok'}}],
                'usage': {'prompt_tokens': 5, 'completion_tokens': 3, 'total_tokens': 8}}).encode()]
            context = MagicMock()
            context.__enter__.return_value = response
            client.stream.return_value = context
            ledger_path = Path(root) / 'requests.jsonl'
            with patch('ahe_native.metered_broker.httpx.Client', return_value=client):
                broker = Broker('synthetic-private', 'http://unused/v1', ledger_path=ledger_path,
                                allowed_model='qwen3.8-flash')
            try:
                self.assertIsNone(broker.request_gate)
                url = scoped_url(broker.url, experiment='offline', domain='retail', round=1, role='user_simulator')
                request = Request(url + '/chat/completions', headers={'Content-Type': 'application/json'},
                    data=json.dumps({'model': 'qwen3.8-flash', 'messages': [{'role': 'user', 'content': 'hello'}],
                                     'enable_thinking': True}).encode())
                with urlopen(request, timeout=5) as result: result.read()
                self.assertEqual(broker.ledger.summary()['overall']['total_tokens'], 8)
                self.assertNotIn('synthetic-private', ledger_path.read_text())
                self.assertTrue(json.loads(client.stream.call_args.kwargs['content'])['enable_thinking'])
                request.data = json.dumps({'model': 'deepseek-v4-flash', 'messages': []}).encode()
                with self.assertRaises(HTTPError): urlopen(request, timeout=5)
            finally:
                broker.close()


if __name__ == '__main__':
    unittest.main()
