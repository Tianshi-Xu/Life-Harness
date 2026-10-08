"""Offline lifecycle tests: unqualified attempts never become completed rounds."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch
import driver as d
import runner as r
import protocol as p

class RepairContinuationTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name);self.run=self.root/'webshop';self.run.mkdir()
        self.state={'domain':'webshop','round':1,'rounds':[{'round':1}],
            'status':'running','active_snapshot':[],'active_snapshot_sha256':p.object_digest([]),
            'accepted_bundle_ids':[],'operation_log':[]}
        p.atomic_json(self.run/'state.json',self.state)
        self.directory=self.run/'synthetic_eval'
        self.incumbent={'split':'train','domain':'webshop','n':50,'successes':25,'accuracy':.5,
            'infra_errors':0,'snapshot_sha256':p.object_digest([]),
            'per_task':{str(i):int(i<25) for i in range(50)}}
        p.atomic_json(self.directory/'summary.json',self.incumbent)
        self.proposal=self.run/'proposals/round_02'
        p.atomic_json(self.proposal/'validation.json',{'usage':{'prompt_tokens':8,'completion_tokens':2,'total_tokens':10},
            'proposer_requests':1,'fallback_reason':None,'recovered_qualified':False})
        p.atomic_json(self.proposal/'bundle.json',{})
        self.manifest={'domains':{'webshop':{'batches':[{'task_ids':list(range(50))}]*6}}}
        for name,value in (('HERE',self.root),('validate_protocol',Mock(return_value=self.manifest)),
                ('run_fingerprint',Mock(return_value={'test':True})),('verify_history',Mock()),
                ('phase',Mock(return_value=(self.incumbent,self.directory))),
                ('round_recovery',Mock(return_value=None)),('progress',Mock()),('render_training_report',Mock()),
                ('write_active_chain',Mock(return_value=self.root/'chain.json'))):
            manager=patch.object(d,name,value);manager.start();self.addCleanup(manager.stop)
        manager=patch.object(d.optimizer,'prepare');manager.start();self.addCleanup(manager.stop)

    def test_failed_proposal_keeps_exact_round_and_state(self):
        with patch.object(d.optimizer,'propose',side_effect=d.ProposalFailure('no_valid_candidate','synthetic')):
            with self.assertRaises(d.ProposalFailure):d.run_domain('webshop',Mock(),stop_after=2)
        self.assertEqual(json.loads((self.run/'state.json').read_text()),self.state)
        failure=json.loads((self.run/'pending_failure.json').read_text())
        self.assertFalse(failure['round_advanced']);self.assertFalse(failure['candidate_measured'])
        self.assertEqual(d.phase.call_count,1)

    def test_complete_rejected_candidate_still_counts_as_measured_round(self):
        bundle={'bundle_id':'synthetic','hypothesis':'synthetic','operations':[]}
        with patch.object(d.optimizer,'propose',return_value=self.proposal/'bundle.json'), \
             patch.object(d,'stage_bundle',return_value=(bundle,[],self.root/'candidate_manifest.json')):
            result=d.run_domain('webshop',Mock(),stop_after=2)
        self.assertEqual(result['round'],2)
        self.assertEqual(result['rounds'][1]['candidate_successes'],25)
        self.assertFalse(result['rounds'][1]['accepted'])
        self.assertEqual(d.phase.call_count,2)

    def test_all_infra_candidate_does_not_advance_round(self):
        bad={**self.incumbent,'infra_errors':50}
        d.phase.side_effect=[(self.incumbent,self.directory),(bad,self.directory)]
        bundle={'bundle_id':'synthetic','hypothesis':'synthetic','operations':[]}
        with patch.object(d.optimizer,'propose',return_value=self.proposal/'bundle.json'), \
             patch.object(d,'stage_bundle',return_value=(bundle,[],self.root/'candidate_manifest.json')):
            with self.assertRaises(RuntimeError):d.run_domain('webshop',Mock(),stop_after=2)
        self.assertEqual(json.loads((self.run/'state.json').read_text()),self.state)

    def test_stop_at_completed_prefix_does_not_rerun_it(self):
        result=d.run_domain('webshop',Mock(),stop_after=1)
        self.assertEqual(result,self.state);d.phase.assert_not_called();d.optimizer.prepare.assert_not_called()

    def test_no_premature_seal_written(self):
        with patch.object(d,'DOMAINS',('webshop',)):
            with self.assertRaises(RuntimeError):d.seal_training()
        self.assertFalse((self.root/'training_seal.json').exists())

class GateTests(unittest.TestCase):
    def gate_states(self):
        row={'round':2,'outcome':'evaluated_candidate','candidate_successes':25,
            'incumbent_infra_errors':0,'candidate_infra_errors':0,'proposer_requests':1,
            'usage':{'total_tokens':100},'recovered_qualified':False}
        return {domain:{'round':2,'rounds':[{},copy.deepcopy(row)]} for domain in r.RETRY_DOMAINS}

    def test_both_failed_rounds_must_have_qualified_measured_candidates(self):
        states=self.gate_states();usage={'overall':{'accounting_complete':True}}
        self.assertTrue(r.retry_gate_valid(states,usage))
        states['webshop']['rounds'][1]['candidate_successes']=None
        self.assertFalse(r.retry_gate_valid(states,usage))

    def test_zero_cost_is_only_allowed_for_rechecked_existing_qualified_candidate(self):
        states=self.gate_states();usage={'overall':{'accounting_complete':True}}
        for state in states.values():
            state['rounds'][1].update(proposer_requests=0,usage={'total_tokens':0},recovered_qualified=True)
        self.assertTrue(r.retry_gate_valid(states,usage))
        states['webshop']['rounds'][1]['recovered_qualified']=False
        self.assertFalse(r.retry_gate_valid(states,usage))

    def test_flash_output_not_solver_context_or_solver_output(self):
        cfg={'proposer_model':'deepseek-v4-flash','flash_max_tokens':100000,'flash_context_limit':None}
        self.assertEqual(r.build_payload(cfg,[],[])['max_tokens'],100000)
        for key,bad in (('flash_max_tokens',4096),('flash_max_tokens',60000),('flash_context_limit',32768)):
            with self.assertRaises(RuntimeError):r.build_payload({**cfg,key:bad},[],[])

if __name__=='__main__':unittest.main()
