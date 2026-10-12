"""Timeout continuation preserves paid history and retries only an undelivered request."""
import http.client
import io
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import MagicMock,patch

from evolution.tau_comparison.common import AHE_NATIVE
sys.path.insert(0,str(AHE_NATIVE))
from ahe_native.token_accounting import TokenLedger
from evolution.os_comparison.run import Run,life_usage
from evolution.os_comparison.recovery import estimate_timeout


class TimeoutRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.ledger=TokenLedger(self.root/'requests.jsonl')
        self.scope={'experiment':self.root.name,'domain':'os','round':1,'role':'life_optimizer'}
        self.previous=self.ledger.start(self.scope,model='deepseek-v4.1-flash',attempt=1,stream=False)
        self.ledger.finish(self.previous,status=200,usage={'prompt_tokens':100,'completion_tokens':20,'total_tokens':120})
        self.failed=self.ledger.start(self.scope,model='deepseek-v4.1-flash',attempt=1,stream=False)
        self.ledger.finish(self.failed,status=None,usage=None,error_type='ReadTimeout',delivered=False)
        self.runner=Run.__new__(Run);self.runner.root=self.root
        self.runner.mr=SimpleNamespace(url='http://synthetic/v1',ledger=self.ledger)

    def test_connection_closed_before_response_retries_identical_payload_once(self):
        opener=MagicMock()
        result={'choices':[{'message':{'role':'assistant','content':'ok'}}],
                'usage':{'prompt_tokens':100,'completion_tokens':20,'total_tokens':120}}
        opener.open.side_effect=[http.client.RemoteDisconnected('synthetic timeout'),io.StringIO(json.dumps(result))]
        before=list(self.ledger.rows)
        with patch('urllib.request.build_opener',return_value=opener):
            value=self.runner.completion(1)([{'role':'user','content':'public prompt'}],[])
        self.assertEqual(value,result);self.assertEqual(opener.open.call_count,2)
        calls=opener.open.call_args_list
        self.assertEqual(calls[0].args[0].data,calls[1].args[0].data)
        self.assertEqual(self.ledger.rows,before)
        usage=life_usage(self.ledger.summary(),self.root.name,1)
        self.assertEqual(usage['total_tokens'],120)
        self.assertEqual(usage['estimated_usage_calls'],1)
        self.assertFalse(usage['accounting_complete'])
        self.assertTrue(usage['accounting_usable_with_estimates'])

    def test_second_timeout_stops_instead_of_looping(self):
        opener=MagicMock();opener.open.side_effect=http.client.RemoteDisconnected('synthetic timeout')
        with patch('urllib.request.build_opener',return_value=opener):
            with self.assertRaises(http.client.RemoteDisconnected):self.runner.completion(1)([],[])
        self.assertEqual(opener.open.call_count,2)

    def test_pending_request_blocks_estimation(self):
        self.ledger.start(self.scope,model='deepseek-v4.1-flash',attempt=1,stream=False)
        with self.assertRaises(RuntimeError):estimate_timeout(self.ledger,self.root.name,1)

    def test_http_response_received_is_not_blindly_retried(self):
        self.ledger.rows[-1]['http_status']=200
        with self.assertRaises(RuntimeError):estimate_timeout(self.ledger,self.root.name,1)
