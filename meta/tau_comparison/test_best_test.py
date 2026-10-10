"""Selection and new-snapshot trial transport, with no real model calls."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from .best_test import select_final
from .common import read, save
from .pass_three import Tests


class BestSelectionTests(unittest.TestCase):
    def test_final_train_can_improve_original_before_evolve_best(self):
        self.assertTrue(select_final(0.88, 50, 50))

    def test_ties_keep_historical_best_and_worse_final_does_not_replace_it(self):
        self.assertFalse(select_final(0.98, 49, 50))
        self.assertFalse(select_final(0.5, 10, 50))

    def test_invalid_train_counts_fail_before_model_calls(self):
        for values in [(0.5, 0, 0), (0.5, 51, 50), (2, 40, 50)]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                select_final(*values)

    def test_new_snapshot_runs_all_three_trials_without_old_trial_reuse(self):
        from ahe_native.backend import workspace_hash
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            source.mkdir()
            save(source / 'fixture.json', {})
            runner = Tests.__new__(Tests)
            runner.root = root / 'results'
            runner.plan = {'ahe:retail': {'ids': ['one', 'two'], 'snapshot': str(source),
                'snapshot_sha256': workspace_hash(source), 'trial_zero': [], 'new_trials': [0, 1, 2]}}
            runner.solver = SimpleNamespace(url='local-solver')
            runner.route = Mock(return_value='local-simulator')
            runner.finish = Mock()

            def worker(command, **kwargs):
                request = read(Path(command[command.index('--request') + 1]))
                save(Path(command[command.index('--out') + 1]),
                    {k: request[k] for k in ('task_id', 'trial')} | {'success': 1})
                return SimpleNamespace(returncode=0)

            with patch('meta.tau_comparison.pass_three.subprocess.run', side_effect=worker):
                runner.ahe('retail')
            rows = runner.finish.call_args.args[2]
            self.assertEqual({(row['task_id'], row['trial']) for row in rows},
                             {(task, trial) for task in ('one', 'two') for trial in range(3)})


if __name__ == '__main__':
    unittest.main()
