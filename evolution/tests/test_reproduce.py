"""Fresh-run orchestration and lossless accuracy/token reports; no model calls."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evolution import reproduce


class ReproductionTests(unittest.TestCase):
    def test_agentbench_all_runs_each_method_sequentially_and_measures_last_ahe_candidate(self):
        plan = reproduce.make_plan('fresh_test', 'agentbench', 'all')
        methods = [step['method'] for step in plan['steps']]
        self.assertEqual(methods, ['life'] * 6 + ['meta'] * 2 + ['ahe'] * 2)
        self.assertTrue(all('--launch' not in step['command'] for step in plan['steps']))
        self.assertIn('--measure-final-candidate', plan['steps'][-2]['command'])
        self.assertIn('ahe_native.final_test', plan['steps'][-1]['command'])
        self.assertIn('identity', plan['initialization'])

    def test_report_preserves_unknown_cost_rejected_round_and_missing_measurements(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = {'rounds': [{'round': 1, 'candidate_score': .4, 'accepted': False,
                'optimizer_usage': {'input_tokens': 17, 'output_tokens': 4,
                    'accounting_complete': False, 'unknown_usage_calls': 1}}]}
            reproduce.save(root / 'state.json', state)
            final = {'split': 'test', 'evaluations': {'alfworld': {'successes': 3, 'total': 5}}}
            reproduce.save(root / 'final_results.json', final)
            (root / 'flash_requests.jsonl').write_text('{"accounting":"unknown"}\n')
            plan = {'name': 'test', 'comparison': 'six rounds', 'steps': [
                {'method': 'life', 'record_roots': [str(root)]},
                {'method': 'life', 'record_roots': [str(root)]},
                {'method': 'meta', 'record_roots': [str(root / 'missing')]}]}
            report = reproduce.collect(plan)
            self.assertEqual(len(report['runs']), 2)
            measured = report['runs']['life:' + str(root)]
            self.assertEqual(measured['artifacts']['state.json']['record'], state)
            self.assertEqual(measured['artifacts']['final_results.json']['record'], final)
            self.assertEqual(measured['raw_request_ledgers'], ['flash_requests.jsonl'])
            self.assertFalse(report['runs']['meta:' + str(root / 'missing')]['exists'])
            self.assertEqual(report['runs']['meta:' + str(root / 'missing')]['artifacts'], {})

    def test_stopped_paid_run_cannot_be_automatically_reissued(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reproduce.save(root / 'status.json', {'phase': 'stopped'})
            with patch.object(reproduce.subprocess, 'run') as request:
                with self.assertRaisesRegex(RuntimeError, 'no automatic replay'):
                    reproduce.run(root, {'steps': []})
                request.assert_not_called()


if __name__ == '__main__':
    unittest.main()
