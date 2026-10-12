"""Result records distinguish next-batch AHE scores from retained same-batch scores."""
import json
from pathlib import Path
import tempfile
import unittest
from evolution.os_comparison.results import build


class OSResultsTests(unittest.TestCase):
    def test_next_batch_mapping_rejection_and_unknown_usage_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            usage={'total_tokens':100,'estimated_total_tokens':20,'budget_tokens_including_estimates':120,
                   'accounting_complete':False,'accounting_usable_with_estimates':True}
            training={'meta':[{'iteration':1,'train_batch':0,'baseline_score':.54,'candidate_score':.44,
                'accepted':False,'candidate_infra_errors':0,'usage':usage}],
                'life':[{'round':1,'train_batch':0,'incumbent_score':.54,'candidate_score':.48,
                'accepted':False,'candidate_infra_errors':1,'optimizer_usage':usage}],
                'ahe':{'evaluations':[{'iteration':1,'tasks':50,'successes':29,'infrastructure_errors':0},
                                     {'iteration':2,'tasks':50,'successes':18,'infrastructure_errors':0}]},
                'ahe_round_usage':{'1':usage},'ahe_final_train':None}
            (root/'TRAINING_RESULTS.json').write_text(json.dumps(training))
            (root/'OPTIMIZER_TOKEN_USAGE.json').write_text(json.dumps({'meta':usage,'life':usage,'ahe':usage}))
            (root/'status.json').write_text(json.dumps({'phase':'training'}))
            summary=root/'ahe/experiments'/(root.name+'_ahe_os')/'runs/iteration_001/evolve/evolve_summary.md'
            summary.parent.mkdir(parents=True);summary.write_text('completed paid first evolution')
            result=build(root);rows={r['method']:r for r in result['rounds']}
            self.assertEqual(rows['ahe']['candidate_accuracy'],.36)
            self.assertEqual(rows['ahe']['measurement_batch'],1)
            self.assertIsNone(rows['ahe']['retained_accuracy'])
            for method in ('meta','life'):
                self.assertEqual(rows[method]['retained_accuracy'],.54)
                self.assertEqual(rows[method]['measured_optimizer_tokens'],100)
                self.assertEqual(rows[method]['estimated_optimizer_tokens'],20)
                self.assertFalse(rows[method]['accounting_complete'])
            self.assertFalse(result['six_rounds_complete'])
            self.assertIsNone(result['final_test'])
