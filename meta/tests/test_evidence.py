import json
import pathlib
import sys
import tempfile
import unittest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import evidence as e


def row(i, success=0):
    return {'index': i, 'success': success, 'status': 'completed', 'turns': 2, 'tokens': 100,
            'infrastructure_error': False, 'task_result_private': {'correct_sql': 'PRIVATE_ORACLE'},
            'calls': [{'turn': 1}, {'turn': 2}], 'rewards': ['PRIVATE_ORACLE'],
            'history': [{'role': 'system', 'content': 'public system', 'private': 'PRIVATE_ORACLE'},
                        {'role': 'user', 'content': 'Find total population'},
                        {'role': 'assistant', 'content': 'query'},
                        {'role': 'tool', 'content': '1064 SQL syntax error' if not success else '[(1,)]'},
                        {'role': 'assistant', 'content': 'answer'},
                        {'role': 'tool', 'content': 'ok'},
                        {'role': 'system', 'content': '[HARNESS_TRACE_V1] PRIVATE_ORACLE'}],
            'harness_trace': [{'layer': 'h4', 'turn': 2, 'hints': ['check syntax'], 'private': 'PRIVATE_ORACLE'}]}


def dataset(path, count=20):
    path.mkdir()
    rows = [row(i, i % 3 == 0) for i in range(count)]
    for r in rows:
        e.save(path / f"episode_{r['index']}.json", r)
    e.save(path / 'summary.json', {'n': count, 'successes': sum(r['success'] for r in rows),
           'infra_errors': 0, 'per_task': {str(r['index']): r['success'] for r in rows},
           'plugin_hash': 'p', 'runner_hash': 'r', 'manifest_hash': 'm'})


class EvidenceTests(unittest.TestCase):
    def test_public_allowlist_and_turn_alignment(self):
        ep = e.public_episode(row(0))
        self.assertNotIn('PRIVATE_ORACLE', e.encoded(ep))
        self.assertEqual([m['model_turn'] for m in ep['messages']], [None, None, 1, 1, 2, 2])
        bad = row(0); bad['turns'] = 3
        self.assertFalse(e.public_episode(bad)['turn_alignment_verified'])
        self.assertTrue(all(m['model_turn'] is None for m in e.public_episode(bad)['messages']))

    def test_bounded_large_pool_retrieval_resume_and_no_path_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp); data = root / 'data'; host = root / 'host'; visible = root / 'visible'
            dataset(data, 1000)
            index = e.build_pack(data, host, visible, max_cases=8, initial_chars=9000, total_chars=14000)
            self.assertEqual(index['n'], 1000)
            self.assertLessEqual(len(index['cases']), 8)
            self.assertLessEqual(len((visible / 'index.json').read_text()), 9000)
            req = [{'task_id': '1', 'start_turn': 2, 'end_turn': 2}]
            result = e.retrieve(host, visible, req, 0)
            self.assertEqual(result['results'][0]['episode']['excerpt_turn_range'], [2, 2])
            self.assertNotIn('PRIVATE_ORACLE', e.encoded(result))
            before = (host / 'ledger.json').read_text()
            self.assertEqual(e.retrieve(host, visible, req, 0), result)
            self.assertEqual(before, (host / 'ledger.json').read_text())
            e.retrieve(host, visible, [{'task_id': '../../secret'}], 1)
            self.assertIn('unknown task', (visible / 'details_1.json').read_text())
            e.retrieve(host, visible, req, 0)  # Older batch must not roll back newer accounting.
            ledger = json.loads((host / 'ledger.json').read_text())
            self.assertEqual(len(ledger['requests']), 2)
            self.assertEqual(ledger['exposed_chars'], sum(len(p.read_text()) for p in visible.glob('*.json')))
            self.assertLessEqual(ledger['exposed_chars'], 14000)
            invalid = e.retrieve(host, visible, [{'task_id': '1', 'start_turn': -1}], 2)
            self.assertEqual(invalid['results'][0]['available_turns'], [1, 2])
            mixed = e.retrieve(host, visible, [{'group': 'sql_syntax', 'end_turn': 99}, {'task_id': '2', 'end_turn': 1}], 3)
            self.assertIn('denied', mixed['results'][0])
            self.assertIn('episode', mixed['results'][1])

    def test_oversized_episode_denied_not_truncated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp); data = root / 'data'; dataset(data)
            r = row(1); r['history'][1]['content'] = 'x' * 20000
            e.save(data / 'episode_1.json', r)
            e.build_pack(data, root/'host', root/'visible', initial_chars=3000, total_chars=5000)
            result = e.retrieve(root/'host', root/'visible', [{'task_id': '1'}], 0)
            self.assertIn('denied', result['results'][0])
            self.assertNotIn('episode', result['results'][0])

    def test_overfull_batch_keeps_valid_subset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp); data = root / 'data'; dataset(data)
            e.build_pack(data, root/'host', root/'visible', initial_chars=3000, total_chars=10000)
            result = e.retrieve(root/'host', root/'visible', [{'task_id': str(i)} for i in range(12)], 0)
            self.assertTrue(any('episode' in r for r in result['results']))
            self.assertTrue(any('denied' in r for r in result['results']))
            ledger = json.loads((root/'host/ledger.json').read_text())
            self.assertEqual(ledger['exposed_chars'], sum(len(p.read_text()) for p in (root/'visible').glob('*.json')))
            self.assertLessEqual(ledger['exposed_chars'], 10000)

    def test_screen_pairs_and_exploration(self):
        self.assertFalse(e.screen_decision({'a': 0}, {'a': 1}, 1)['run_full'])
        self.assertTrue(e.screen_decision({'a': 0}, {'a': 1}, 3)['run_full'])
        self.assertTrue(e.screen_decision({'a': 0}, {'a': 0}, 1)['run_full'])
        with self.assertRaises(ValueError):
            e.screen_decision({'a': 0}, {'b': 0}, 1)
        eps = {str(i): e.public_episode(row(i, i % 3 == 0)) for i in range(50)}
        ids = e.screen_ids(eps, 16, 1)
        self.assertEqual(ids, e.screen_ids(eps, 16, 1))
        self.assertEqual(len(set(ids)), 16)
        self.assertTrue(any(eps[k]['success'] for k in ids))
        self.assertTrue(any(not eps[k]['success'] for k in ids))
