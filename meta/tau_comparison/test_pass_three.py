"""No API calls: strict pass^3 and optional native checkpoint transport."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from .common import child_env, save
from .pass_three import pass_three
from . import meta_evaluate


class PassThreeTests(unittest.TestCase):
    def rows(self):
        return [{'task_id': task, 'trial': trial, 'reward': reward}
                for task, values in {'all': [1, 1, 1], 'some': [1, 0, 1], 'none': [0, 0, 0]}.items()
                for trial, reward in enumerate(values)]

    def test_all_three_not_any_three_or_accuracy_cubed(self):
        score = pass_three(['all', 'some', 'none'], self.rows())
        self.assertEqual(score['pass_caret_3'], 1 / 3)
        self.assertEqual(score['pass_caret_3_successes'], 1)
        self.assertEqual(score['successes'], 5)
        self.assertEqual(score['accuracy'], 5 / 9)
        self.assertEqual(score['trial_successes'], [2, 1, 2])
        self.assertNotEqual(score['pass_caret_3'], score['accuracy'] ** 3)

    def test_trial_order_is_explicit_not_completion_order(self):
        rows = list(reversed(self.rows()))
        self.assertEqual(pass_three(['all', 'some', 'none'], rows)['per_task']['some'], [1, 0, 1])

    def test_missing_trial_cannot_pass_and_has_fixed_denominator(self):
        rows = [row for row in self.rows() if (row['task_id'], row['trial']) != ('all', 2)]
        score = pass_three(['all', 'some', 'none'], rows)
        self.assertEqual(score['pass_caret_3'], 0)
        self.assertEqual(score['missing_episodes'], 1)
        self.assertEqual(score['episodes'], 9)
        self.assertIsNone(score['per_task']['all'][2])

    def test_infrastructure_zero_remains_a_failure(self):
        rows = self.rows()
        rows[2].update(reward=0, infrastructure_error=True)
        score = pass_three(['all', 'some', 'none'], rows)
        self.assertEqual(score['pass_caret_3'], 0)
        self.assertEqual(score['infra_errors'], 1)

    def test_invalid_duplicates_or_nonbinary_reward_rejected(self):
        for rows in (self.rows() + [self.rows()[0]],
                     [{'task_id': 'all', 'trial': 3, 'reward': 1}],
                     [{'task_id': 'unknown', 'trial': 0, 'reward': 1}],
                     [{'task_id': 'all', 'trial': 0, 'reward': 0.5}],
                     [{'task_id': 'all', 'trial': 0, 'reward': 1, 'infrastructure_error': True}]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                pass_three(['all', 'some', 'none'], rows)

    def test_empty_or_duplicate_ids_rejected(self):
        for ids in ([], ['all', 'all']):
            with self.assertRaises(ValueError):
                pass_three(ids, [])

    def test_worker_env_does_not_inherit_real_credentials(self):
        with patch.dict('os.environ', {'MR_API_KEY': 'test-secret', 'DASHSCOPE_API_KEY': 'test-secret'}):
            env = child_env()
        self.assertNotIn('MR_API_KEY', env)
        self.assertNotIn('DASHSCOPE_API_KEY', env)
        self.assertEqual(env['OPENAI_API_KEY'], 'EMPTY')

    def test_resume_option_keeps_default_native_behavior_unchanged(self):
        for resume in (False, True):
            with self.subTest(resume=resume), tempfile.TemporaryDirectory() as directory:
                request = {'domain': 'airline', 'split': 'test', 'task_id_file': 'ids',
                           'trials': 3, 'concurrency': 50, 'candidate': 'candidate',
                           'solver_base': 'solver', 'user_model': 'user', 'user_base': 'simulator',
                           'judge_base': 'judge', 'save_to': 'native', 'resume_trials': resume}
                path = Path(directory) / 'request.json'
                save(path, request)
                config = SimpleNamespace(auto_resume=False)
                original = Mock()
                module = SimpleNamespace(_set_nl_assertions_llm=Mock(), run_tasks=original)
                module.main = lambda: module.run_tasks(config, 'tasks', save_path='path')
                with patch.object(meta_evaluate.importlib.util, 'spec_from_file_location') as spec, \
                     patch.object(meta_evaluate.importlib.util, 'module_from_spec', return_value=module), \
                     patch.object(meta_evaluate.sys, 'argv', []), patch.dict('os.environ'):
                    meta_evaluate.main(path)
                self.assertEqual(config.auto_resume, resume)
                original.assert_called_once_with(config, 'tasks', save_path='path')
                spec.return_value.loader.exec_module.assert_called_once_with(module)


if __name__ == '__main__':
    unittest.main()
