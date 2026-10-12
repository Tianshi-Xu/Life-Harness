"""Offline native OS split/evidence/budget guards; never call a model or Docker."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from evolution.life_evolve import profile
from evolution.os_comparison import protocol
from evolution.os_comparison.evaluate import validate_meta
from evolution.os_comparison.evaluate import evaluate, trace_events


class OSProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "task.py"
        self.source.write_text("immutable native task\n")
        self.schedule = {"schema_version": 1, "split": "train", "iterations": 6,
            "tasks_per_iteration": 50, "batch_policy": "frozen_os_native_batches",
            "native_source_hashes": {str(self.source): protocol.digest(self.source)},
            "domains": {"os": {"train_pool_size": 1000, "batches": [
                {"iteration": i+1, "task_ids": list(range(i*50, (i+1)*50))} for i in range(6)]}}}
        self.request = {"domain": "os", "split": "train", "k": 1, "iteration": 1}

    def test_schedule_keeps_disjoint_native_train_batches(self):
        for i in range(1, 7):
            self.assertEqual(protocol.scheduled_batch({**self.request,"iteration":i}, self.schedule),
                             list(range((i-1)*50,i*50)))

    def test_test_requests_wrong_trials_rounds_and_domain_are_rejected(self):
        for update in ({"split":"test"},{"k":2},{"iteration":True},{"iteration":7},{"domain":"webshop"}):
            with self.assertRaises(ValueError):
                protocol.scheduled_batch({**self.request,**update},self.schedule)

    def test_native_source_and_schedule_tampering_fail_before_execution(self):
        for bad in (True,1000,-1):
            changed=copy.deepcopy(self.schedule)
            changed['domains']['os']['batches'][0]['task_ids'][0]=bad
            with self.assertRaises(ValueError):protocol.scheduled_batch(self.request,changed)
        changed=copy.deepcopy(self.schedule)
        changed['domains']['os']['batches'][1]['task_ids'][0]=0
        with self.assertRaises(ValueError):protocol.scheduled_batch(self.request,changed)
        self.source.write_text('changed native task')
        with self.assertRaises(ValueError):protocol.scheduled_batch(self.request,self.schedule)

    def test_life_os_context_is_exact_public_description_only(self):
        row={'public_task_context':{'description':'public OS task'},
             'task_result_private':{'match':'SECRET_ANSWER','check':'SECRET_SCRIPT'}}
        context=profile.public_context('os',row,None)
        self.assertEqual(context,{'description':'public OS task'})
        context['description']='local change'
        self.assertEqual(row['public_task_context']['description'],'public OS task')
        for invalid in (None,{}, {'description':'public','match':'SECRET_ANSWER'}, {'description':42}):
            with self.assertRaises(ValueError):
                profile.public_context('os',{'public_task_context':invalid},None)

    def test_native_layer_trace_is_public_evidence_events(self):
        native = {'h2': [{'turn':1,'before':{},'after':{}}], 'h3': [], 'h4': [], 'h5': []}
        events = trace_events(native)
        self.assertEqual(events,[{'layer':'h2','turn':1,'before':{},'after':{}}])
        events[0]['before']['local']='change'
        self.assertEqual(native['h2'][0]['before'],{})
        self.assertEqual(trace_events({'h2':[],'h3':[],'h4':[],'h5':[]}),[])

    def test_meta_identity_and_candidate_io_restrictions(self):
        source=self.root/'candidate.py'
        identity='class AgentHarness:\n    def __init__(self,public_context=None):\n        pass\n    def transform_request(self,request):\n        return request\n    def transform_response(self,response,history):\n        return response\n'
        source.write_text(identity)
        self.assertEqual(validate_meta(source).AgentHarness().transform_request({'model':'fixed'}),{'model':'fixed'})
        source.write_text('import random\n'+identity)
        self.assertEqual(validate_meta(source).AgentHarness().transform_response({'role':'assistant'},[]),{'role':'assistant'})
        for extra in ('import os\n','open("/private")\n','x=object.__subclasses__()\n','import random\nx=random._os\n'):
            source.write_text(identity+extra)
            with self.assertRaises(ValueError):validate_meta(source)

    def test_held_out_execution_requires_all_eighteen_rounds_and_unchanged_states(self):
        candidate = self.root / 'candidate.py'
        candidate.write_text('sealed candidate')
        state = self.root / 'state.json'
        state.write_text('{}')
        spec = {'methods': ['meta','ahe','life'], 'native_spec': {'test_ids': [0,1]},
                'native_source_hashes': {str(self.source): protocol.digest(self.source)}}
        protocol.save(self.root / 'protocol.json', spec)
        seal = {'completed_rounds': {'meta':6,'ahe':5,'life':6},
                'candidates': {'meta': protocol.digest(candidate)},
                'state_hashes': {str(state): protocol.digest(state)}}
        protocol.save(self.root / 'training_seal.json', seal)
        with self.assertRaisesRegex(ValueError,'eighteen'):
            evaluate(self.root,'meta','test','final_test',[0,1],'http://invalid',plugin=candidate)
        seal['completed_rounds']['ahe'] = 6
        protocol.save(self.root / 'training_seal.json', seal)
        state.write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError,'state changed'):
            evaluate(self.root,'meta','test','final_test',[0,1],'http://invalid',plugin=candidate)
        self.assertFalse((self.root / 'evolution/evals').exists())


class ShellTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_merged_and_split_stdout_frames_are_both_preserved(self):
        from evolution.os_comparison.transport import install_shell_transport
        prompt = b'\x1b[32mroot@container:~# '
        for chunks in ([b'ACTUAL_RESULT\r\n' + prompt], [b'ACTUAL_', b'RESULT\r\n', prompt]):
            class Stream:
                def __init__(self):
                    self.queue, self.writes = [], []
                async def write_in(self, data):
                    self.writes.append(data)
                    self.queue = [prompt] if data == b'stty -echo\n' else list(chunks)
                async def read_out(self):
                    return SimpleNamespace(data=self.queue.pop(0)) if self.queue else None
            stream = Stream()
            controller = SimpleNamespace(_shells={})
            async def original_create(environment_id, shell):
                controller._shells[environment_id] = stream
            controller.create_shell = original_create
            install_shell_transport(controller)
            output = await controller.execute_shell('native','printf ACTUAL_RESULT')
            self.assertIn(b'ACTUAL_RESULT',output)
            self.assertEqual(stream.writes,[b'stty -echo\n',b'printf ACTUAL_RESULT\n'])
            await controller.execute_shell('native','printf ACTUAL_RESULT')
            self.assertEqual(stream.writes.count(b'stty -echo\n'),1)


if __name__=='__main__':
    unittest.main()
