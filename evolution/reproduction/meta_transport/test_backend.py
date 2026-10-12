import json
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path

from api_agent import AgentRunner
from usage import Ledger, normalize


def response(content='', calls=None):
    return {'choices': [{'finish_reason': 'tool_calls' if calls else 'stop',
        'message': {'role': 'assistant', 'content': content, **({'tool_calls': calls} if calls else {})}}]}


def call(name, **args):
    return {'id': name + '_test', 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}


class Tests(unittest.TestCase):
    def test_native_reward_namespaces_are_serialized_without_changing_values(self):
        from common import save
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'episode.json'
            save(path, {'success': 1, 'rewards': [SimpleNamespace(reward=0.5,
                details=SimpleNamespace(completed=True))]})
            self.assertEqual(json.loads(path.read_text()), {'success': 1, 'rewards': [
                {'reward': 0.5, 'details': {'completed': True}}]})
    def test_cache_and_reasoning_not_double_counted(self):
        row = normalize({'prompt_tokens': 100, 'completion_tokens': 40, 'total_tokens': 140,
            'prompt_cache_hit_tokens': 60, 'prompt_tokens_details': {'cached_tokens': 60},
            'completion_tokens_details': {'reasoning_tokens': 30}})
        self.assertEqual((row['total_tokens'], row['uncached_input_tokens'], row['reasoning_tokens']), (140, 40, 30))

    def test_missing_zero_or_inconsistent_usage_rejected(self):
        for usage in (None, {}, {'prompt_tokens': 0, 'completion_tokens': 0},
                      {'prompt_tokens': 10, 'completion_tokens': 2, 'total_tokens': 99}):
            with self.assertRaises(ValueError):
                normalize(usage)

    def test_unknown_call_is_not_faked_as_zero(self):
        with tempfile.TemporaryDirectory() as root:
            ledger = Ledger(root)
            ledger.record('webshop', {'round': 1, 'role': 'analysis'}, 200, {'usage':
                {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}})
            ledger.record('webshop', {'round': 1, 'role': 'parent'}, 502)
            summary = ledger.summarize('webshop', 1)
            self.assertEqual(summary['total_tokens'], 15)
            self.assertFalse(summary['accounting_complete'])
            self.assertEqual(summary['calls_with_unknown_usage'], 1)

    def test_429_and_role_accounting(self):
        with tempfile.TemporaryDirectory() as root:
            ledger = Ledger(root)
            ledger.record('dbbench', {'round': 2, 'role': 'implementation'}, 429)
            ledger.record('dbbench', {'round': 2, 'role': 'implementation'}, 200,
                          {'usage': {'prompt_tokens': 7, 'completion_tokens': 2}})
            summary = ledger.summarize('dbbench', 2)
            self.assertEqual((summary['successful_calls'], summary['recorded_upstream_attempts']), (1, 2))
            self.assertTrue(summary['accounting_complete'])
            self.assertEqual(summary['by_role']['implementation']['total_tokens'], 9)

    def test_two_independent_agents_and_durable_resume(self):
        with tempfile.TemporaryDirectory() as root:
            work, out = Path(root) / 'workspace', Path(root) / 'output'
            (work / 'agents').mkdir(parents=True)
            (work / 'logs/round_01').mkdir(parents=True)
            (work / 'agents/baseline_zero.py').write_text('identity')
            queues = {
                'parent': [response(calls=[call('Agent', prompt='analyze')]),
                    response(calls=[call('Agent', prompt='implement')]), response('done')],
                'analysis': [response(calls=[call('Read', file_path='agents/baseline_zero.py')]), response('hypothesis')],
                'implementation': [response(calls=[call('Write', file_path='agents/new_candidate.py', content='candidate')]), response('implemented')]}
            prompts = {}

            def fake(messages, tools, role, request_id):
                prompts.setdefault(role, []).append(messages)
                self.assertTrue(request_id)
                return queues[role].pop(0)
            runner = AgentRunner(work, out, {'port_contract': 'FULL_NATIVE_CONTRACT'}, completion=fake)
            self.assertEqual(runner.run('parent', 'original skill prompt'), 'done')
            self.assertEqual((work / 'agents/new_candidate.py').read_text(), 'candidate')
            self.assertEqual(json.loads((out / 'parent/session.json').read_text())['delegations'], ['analysis', 'implementation'])
            self.assertIn('FULL_NATIVE_CONTRACT', prompts['analysis'][0][1]['content'])
            self.assertEqual(len(prompts['analysis'][0]), 2)
            self.assertEqual(runner.run('parent', 'original skill prompt'), 'done')
            self.assertTrue(all(not queue for queue in queues.values()))

    def test_frozen_paths_and_no_host_bash(self):
        with tempfile.TemporaryDirectory() as root:
            work = Path(root) / 'workspace'
            (work / 'agents').mkdir(parents=True)
            (work / 'agents/baseline_zero.py').write_text('identity')
            runner = AgentRunner(work, Path(root) / 'out', {})
            for path in ('../private', '/etc/passwd'):
                with self.assertRaises(ValueError):
                    runner.path(path)
            with self.assertRaises(ValueError):
                runner.path('agents/baseline_zero.py', write=True)
            with self.assertRaises(ValueError):
                runner.dispatch('Bash', {'command': 'true'}, {'delegations': []}, 'parent')

    def test_nested_and_excess_agents_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            runner = AgentRunner(root, root, {})
            for role, count in (('analysis', 0), ('parent', 2)):
                with self.assertRaises(ValueError):
                    runner.dispatch('Agent', {'prompt': 'x'}, {'delegations': ['x'] * count}, role)

    def test_glob_is_not_regex_and_read_has_no_case_or_character_quota(self):
        with tempfile.TemporaryDirectory() as root:
            work = Path(root) / 'workspace'
            (work / 'agents').mkdir(parents=True)
            (work / 'agents/baseline_zero.py').write_text('identity')
            (work / 'evidence').mkdir()
            (work / 'evidence/long.json').write_text('\n'.join(['public'] * 5000))
            runner = AgentRunner(work, Path(root) / 'out', {})
            state = {'delegations': []}
            self.assertEqual(runner.dispatch('Glob', {'pattern': '**/*.py'}, state, 'analysis'), 'agents/baseline_zero.py')
            self.assertIn('baseline_zero.py:1: identity',
                          runner.dispatch('Grep', {'pattern': 'identity', 'path': 'agents'}, state, 'analysis'))
            self.assertIn('5000: public', runner.dispatch('Read', {'file_path': 'evidence/long.json'}, state, 'analysis'))

    def test_truncated_model_output_cannot_promote(self):
        with tempfile.TemporaryDirectory() as root:
            def fake(*args):
                value = response('partial')
                value['choices'][0]['finish_reason'] = 'length'
                return value
            runner = AgentRunner(root, Path(root) / 'out', {}, completion=fake)
            with self.assertRaisesRegex(RuntimeError, 'truncated'):
                runner.run('parent', 'x')

    def test_all_three_first_rounds_gate_before_later_rounds(self):
        from runner import first_round_valid
        from common import DOMAINS
        with tempfile.TemporaryDirectory() as root:
            ledger = Ledger(root)
            states = {d: {'rounds': [{'candidate_successes': 0, 'candidate_infra_errors': 0}]} for d in DOMAINS}
            for d in DOMAINS:
                for role in ('parent', 'analysis', 'implementation'):
                    ledger.record(d, {'round': 1, 'role': role}, 200,
                                  {'usage': {'prompt_tokens': 10, 'completion_tokens': 1}})
            self.assertTrue(first_round_valid(states, ledger))
            states['webshop']['rounds'][0]['candidate_successes'] = None
            self.assertFalse(first_round_valid(states, ledger))
            self.assertFalse(first_round_valid({'dbbench': states['dbbench']}, ledger))


class FreshReportTests(unittest.TestCase):
    def test_new_run_reports_only_actual_calls_and_retains_unknown_usage(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Ledger(Path(tmp))
            ledger.record('webshop', {'round': 1, 'role': 'parent'}, 200,
                          {'usage': {'prompt_tokens': 20, 'completion_tokens': 3}})
            ledger.record('webshop', {'round': 1, 'role': 'implementation'}, 500)
            report = ledger.report(['webshop'])
            self.assertEqual(report['overall']['total_tokens'], 23)
            self.assertFalse(report['overall']['accounting_complete'])
            self.assertEqual(report['overall']['calls_with_unknown_usage'], 1)
            self.assertNotIn('startup_excluded_usage', report)
            self.assertTrue((Path(tmp) / 'TOKEN_USAGE.json').is_file())


if __name__ == '__main__':
    unittest.main()
