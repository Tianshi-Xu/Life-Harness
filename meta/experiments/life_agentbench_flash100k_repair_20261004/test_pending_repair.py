"""Offline tests for bounded same-round repair accounting and fixture preservation."""
import tempfile
import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3].parent/'ahe-native'))
from repair_pending import aggregate_attempts,fixture_contract,merge_ledgers
from ahe_native.token_accounting import TokenLedger

class PendingRepairTests(unittest.TestCase):
    def test_failed_spend_is_added_not_replaced(self):
        previous={'requests':32,'usage':{'prompt_tokens':100,'completion_tokens':20,'total_tokens':120}}
        successful={'proposer_requests':4,'usage':{'prompt_tokens':50,'completion_tokens':10,'total_tokens':60},'passed':True}
        merged=aggregate_attempts(previous,successful)
        self.assertEqual(merged['proposer_requests'],36)
        self.assertEqual(merged['usage']['total_tokens'],180)
        self.assertEqual(successful['usage']['total_tokens'],60)
        self.assertEqual(merged['successful_attempt']['requests'],4)

    def test_fixture_comparison_ignores_only_runtime_metadata(self):
        original={'tests':[{'test_id':'x','rule_id':'rule','kind':'counterexample','context':None,
                           'steps':[{'hook':'h2','args':[{},[]],'expected':{}}]}]}
        changed={'tests':[{**original['tests'][0],'tools':[]}]}
        self.assertEqual(fixture_contract(original),fixture_contract(changed))
        changed['tests'][0]['steps']=[{'hook':'h2','args':[{},[]],'expected':{'tool_calls':[]}}]
        self.assertNotEqual(fixture_contract(original),fixture_contract(changed))

    def test_ledger_merge_idempotent_and_preserves_spend(self):
        with tempfile.TemporaryDirectory() as temporary:
            a=Path(temporary)/'original/flash_requests.jsonl';b=Path(temporary)/'repair/flash_requests.jsonl'
            for path,total in ((a,100),(b,50)):
                ledger=TokenLedger(path)
                ident=ledger.start({'experiment':'synthetic','domain':'webshop','round':6,'role':'life_optimizer'},
                    model='deepseek-v4-flash',attempt=1,stream=False)
                ledger.finish(ident,status=200,usage={'prompt_tokens':total,'completion_tokens':0,'total_tokens':total})
            first=merge_ledgers(b,a);second=merge_ledgers(b,a)
            self.assertEqual(first,second)
            self.assertEqual(first['overall']['total_tokens'],150)
            self.assertEqual(first['overall']['attempts'],2)

    def test_pending_or_unknown_spend_never_merged_as_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            a=Path(temporary)/'original/flash_requests.jsonl';b=Path(temporary)/'repair/flash_requests.jsonl'
            ledger=TokenLedger(b)
            ledger.start({'experiment':'synthetic','domain':'webshop','round':6,'role':'life_optimizer'},
                model='deepseek-v4-flash',attempt=1,stream=False)
            with self.assertRaises(RuntimeError):merge_ledgers(b,a)

if __name__=='__main__':unittest.main()
