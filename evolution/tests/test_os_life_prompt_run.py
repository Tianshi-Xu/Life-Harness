import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch
from evolution.os_comparison.life_prompt_run import LifePromptRun
from evolution.os_comparison.protocol import save,digest


class TestSealedTestGate(unittest.TestCase):
    def test_training_seal_is_consumable_by_shared_evaluator(self):
        from evolution.os_comparison.evaluate import evaluate
        with tempfile.TemporaryDirectory() as root:
            run=self.fixture(root)
            run.run_life=Mock();run.report=Mock(return_value=(6,{'accounting_usable_with_estimates':True}))
            run.verify=Mock();run.mr=Mock();run.mr.ledger.rows=[]
            run.compose=Mock();run.compose.write_active_chain.return_value=run.life_final
            run.training()
            save(run.root/'protocol.json',{'native_source_hashes':{},'methods':['life'],
                 'native_spec':{'test_ids':[0]}})
            # Reach the execution pool only after the real shared TEST gate passes.
            # No model or environment call is made in this regression check.
            with patch('evolution.os_comparison.evaluate.verify_sources'), patch('evolution.os_comparison.evaluate.ProcessPoolExecutor',
                       side_effect=RuntimeError('shared TEST gate passed')):
                with self.assertRaisesRegex(RuntimeError,'shared TEST gate passed'):
                    evaluate(run.root,'life','test','final_test',[0],
                             'http://unused/v1',manifest=run.life_final)

    def fixture(self,root):
        run=LifePromptRun.__new__(LifePromptRun)
        run.root=Path(root);run.work=run.root/'life';run.work.mkdir()
        run.life_final=run.work/'final.json';run.life_final.write_text('{}')
        save(run.work/'state.json',{'round':6})
        run.state={'round':6};run.evaluate_life=Mock()
        save(run.root/'training_seal.json',{'completed_rounds':{'life':6},
            'candidate_sha256':digest(run.life_final),'state_sha256':digest(run.work/'state.json')})
        return run

    def test_unfinished_training_cannot_make_any_test_call(self):
        with tempfile.TemporaryDirectory() as root:
            run=self.fixture(root);run.state['round']=5
            with self.assertRaisesRegex(RuntimeError,'before TEST'):run.tests()
            run.evaluate_life.assert_not_called()

    def test_changed_selection_cannot_make_any_test_call(self):
        with tempfile.TemporaryDirectory() as root:
            run=self.fixture(root);run.life_final.write_text('{"changed":true}')
            with self.assertRaisesRegex(RuntimeError,'before TEST'):run.tests()
            run.evaluate_life.assert_not_called()

    def test_missing_seal_cannot_make_any_test_call(self):
        with tempfile.TemporaryDirectory() as root:
            run=self.fixture(root);(run.root/'training_seal.json').unlink()
            with self.assertRaises(FileNotFoundError):run.tests()
            run.evaluate_life.assert_not_called()

if __name__=='__main__':unittest.main()
