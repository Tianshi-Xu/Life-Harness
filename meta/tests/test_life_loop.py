import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import life_loop as loop


def args():
    return argparse.Namespace(run_name='test', fresh=False, iterations=1,
        domains=['retail'], trials=1, num_tasks=None, concurrency=10,
        agent_llm='agent', user_llm='user', user_api_base=None,
        user_disable_thinking=False, max_steps=None, nl_domains=['retail'],
        h5_top_k=1, from_scratch=False, mock_proposer=True,
        proposer_model='test', propose_timeout=10, eval_timeout=10,
        accept_on='full', full_every=3, screen_sentry=1)


class LoopTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_changed_config_cannot_resume(self):
        p = self.root / 'config.json'
        a = args()
        loop.bind_record(p, loop.eval_config(a))
        a.trials = 3
        with self.assertRaisesRegex(RuntimeError, 'mismatch'):
            loop.bind_record(p, loop.eval_config(a))

    def test_cache_binds_chain_contents(self):
        layer = self.root / 'layer.py'
        layer.write_text('x = 1')
        loader = loop.write_loader(self.root / 'chain', [layer])
        run = self.root / 'run'
        a = args()
        with patch.object(loop.subprocess, 'run') as process:
            out = run / 'evals/retail/tag/val.json'
            def evaluated(*unused, **kw):
                loop.save_json(out, {'accuracy': .5, 'per_task': {'1': .5}})
                return argparse.Namespace(returncode=0)
            process.side_effect = evaluated
            loop.evaluate(self.root, run, 'tag', 'search', a, plugin=loader)
            loop.evaluate(self.root, run, 'tag', 'search', a, plugin=loader)
            self.assertEqual(process.call_count, 1)
            layer.write_text('x = 2')
            with self.assertRaisesRegex(RuntimeError, 'mismatch'):
                loop.evaluate(self.root, run, 'tag', 'search', a, plugin=loader)

    def test_resume_preserves_proposal_and_failed_confirmation_cache(self):
        initial = ({'retail': .5}, {}, {'retail': {'1': 0., '2': 1.}})
        screen = ({'retail': 1.}, {}, {'retail': {'1': 1., '2': 1.}})
        a = args()
        run = self.root / 'test'
        with patch.object(loop, 'RUNS_DIR', self.root), \
             patch.object(loop, 'setup_worktree', return_value=self.root), \
             patch.object(loop, 'clean_worktree', return_value=[]), \
             patch.object(loop, 'validate_candidate', return_value=''), \
             patch.object(loop, 'evaluate', side_effect=[initial, KeyboardInterrupt]):
            with self.assertRaises(KeyboardInterrupt):
                loop.evolve(a)
        candidate = run / 'candidates/iter_01.py'
        saved = candidate.read_bytes()
        with patch.object(loop, 'RUNS_DIR', self.root), \
             patch.object(loop, 'setup_worktree', return_value=self.root), \
             patch.object(loop, 'clean_worktree', return_value=[]), \
             patch.object(loop, 'validate_candidate', return_value=''), \
             patch.object(loop, 'evaluate', side_effect=[screen, None]), \
             patch.object(loop, 'render_prompt', side_effect=AssertionError('reproposed')):
            loop.evolve(a)
        state = loop.load_json(run / 'state.json')
        self.assertEqual(candidate.read_bytes(), saved)
        self.assertEqual(state['task_scores'], initial[2])
        self.assertEqual(state['accepted'], [])
        self.assertEqual(state['history'][-1]['decision'], 'eval_error_reject')

    def test_frozen_run_cannot_evolve(self):
        run = self.root / 'test'
        loop.save_json(run / 'finalized.json', {'status': 'in_progress'})
        with patch.object(loop, 'RUNS_DIR', self.root):
            with self.assertRaisesRegex(RuntimeError, 'frozen'):
                loop.evolve(args())

    def test_test_failure_does_not_mark_complete(self):
        run = self.root / 'test'
        loop.save_json(run / 'state.json', {'accepted': [], 'from_scratch': False})
        with patch.object(loop, 'RUNS_DIR', self.root), \
             patch.object(loop, 'evaluate', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'incomplete'):
                loop.finalize(args())
        self.assertEqual(loop.load_json(run / 'finalized.json')['status'], 'in_progress')


if __name__ == '__main__':
    unittest.main()
