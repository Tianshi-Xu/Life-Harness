"""Local fake-provider end-to-end test; NEVER sends an external model request."""
import json
import shutil
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import httpx

from api_broker import TokenBroker
from common import ROOT
from isolation import isolated_proposal
from test_backend import call, response
from usage import Ledger

import sys
sys.path.insert(0, str(ROOT / 'integrations/ahe-native'))
from ahe_native.api_broker import RequestGate


class TransportTests(unittest.TestCase):
    def test_single_gate_never_exceeds_eight(self):
        gate = RequestGate(8)

        def invoke(_):
            with gate:
                time.sleep(0.02)
        with ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(invoke, range(40)))
        self.assertEqual(gate.snapshot()['requests'], 40)
        self.assertLessEqual(gate.snapshot()['peak'], 8)
        self.assertEqual(gate.snapshot()['active'], 0)

    def test_actual_chroot_and_parent_children_receipts(self):
        from ahe_native.isolation import BASE
        if not (BASE / ".ready").is_file():
            self.skipTest("Initialize the dedicated AHE isolation rootfs for this chroot integration test")
        # This diagnostic needs loopback/chroot permissions, but the provider is
        # entirely fake and the credential is an intentionally fake test string.
        with tempfile.TemporaryDirectory(prefix='meta-api-offline-') as name:
            root = Path(name)
            shutil.copy2(Path(__file__).with_name('api_agent.py'), root / 'api_agent.py')
            work = root / 'webshop/workspace'
            (work / 'agents').mkdir(parents=True)
            (work / 'logs/round_01').mkdir(parents=True)
            (work / 'agents/baseline_zero.py').write_text('identity\n')
            pending = work / 'logs/round_01/pending_eval.json'
            doc = {'iteration': 1, 'candidates': [{'name': 'new_candidate',
                   'import_path': 'agents.new_candidate:AgentHarness', 'hypothesis': 'test',
                   'changes': 'test', 'expected_efficiency': 'test'}]}
            queues = {
                'PARENT': [response(calls=[call('Agent', prompt='ANALYSIS')]),
                    response(calls=[call('Agent', prompt='IMPLEMENTATION')]),
                    response(calls=[call('Write', file_path='logs/round_01/pending_eval.json', content=json.dumps(doc))]),
                    response('done')],
                'ANALYSIS': [response(calls=[call('Read', file_path='agents/baseline_zero.py')]), response('hypothesis')],
                'IMPLEMENTATION': [response(calls=[call('Write', file_path='agents/new_candidate.py', content='candidate\n')]), response('implemented')]}
            ledger, gate = Ledger(root), RequestGate(8)
            broker = TokenBroker('FAKE_DIAGNOSTIC_CREDENTIAL', 'https://never-called.invalid', gate, ledger, ('webshop',))

            def fake_post(url, *, content, headers):
                self.assertEqual(url, 'https://never-called.invalid/chat/completions')
                payload = json.loads(content)
                first = payload['messages'][1]['content'].splitlines()[0]
                value = queues[first].pop(0)
                value.update(model='deepseek-v4-flash', usage={
                    'prompt_tokens': 100, 'completion_tokens': 10, 'total_tokens': 110})
                return httpx.Response(200, json=value)
            try:
                with patch.object(broker.client, 'post', side_effect=fake_post), patch('isolation.HERE', root):
                    self.assertTrue(isolated_proposal('webshop', 1, work, 'PARENT',
                                    'NATIVE_PORT_CONTRACT', broker.url, pending))
                self.assertEqual(json.loads(pending.read_text()), doc)
                summary = ledger.summarize('webshop', 1)
                self.assertTrue(summary['accounting_complete'])
                self.assertEqual(summary['successful_calls'], 8)
                self.assertEqual(summary['total_tokens'], 880)
                self.assertEqual(set(summary['by_role']), {'parent', 'analysis', 'implementation'})
                self.assertTrue(all(not q for q in queues.values()))
                traces = root / 'webshop/proposer_01/traces'
                self.assertTrue((traces / 'parent/session.json').is_file())
                for p in root.rglob('*'):
                    if p.is_file():
                        self.assertNotIn(b'FAKE_DIAGNOSTIC_CREDENTIAL', p.read_bytes())
            finally:
                broker.close()


if __name__ == '__main__':
    unittest.main()
