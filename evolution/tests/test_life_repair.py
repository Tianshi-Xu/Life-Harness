"""Offline regression coverage for retention, regex and cross-layer contracts."""
import json
from unittest.mock import Mock
from pathlib import Path
from types import ModuleType
import re


def _revision_fixtures(name, filename, optimizer=False, profile_module='evolution.life_optimizer_v25'):
    # Reuse the canonical fixture suites; do not keep a copy for each revision.
    path = Path(__file__).with_name(filename)
    source = path.read_text().replace('evolution.agentbench_hooks', 'evolution.agentbench_hooks_v31')
    source = source.replace('evolution/agentbench_hooks', 'evolution/agentbench_hooks_v31')
    source = source.replace('evolution.life_optimizer', profile_module)
    source = source.replace('life-agentbench-h2345-v3', 'life-agentbench-h2345-v3.1')
    if optimizer:
        source = re.sub(r'\b60000\b', '100000', source)
    module = ModuleType(name)
    module.__file__ = str(path)
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module


fixtures = _revision_fixtures('_life_repair_optimizer_fixtures', 'test_life_optimizer.py', True)
interface_fixtures = _revision_fixtures('_life_repair_interface_fixtures', 'test_agentbench_hooks.py')
OptimizerRegressionTests = fixtures.OptimizerTests
InterfaceRegressionTests = interface_fixtures.InterfaceTests
from evolution.agentbench_hooks_v31.contract import validate_source, load_artifact
from evolution.life_optimizer_v25 import profile as p


class RepairTests(fixtures.OptimizerTests):
    def test_qualified_recovery_is_rechecked_without_paid_api(self):
        self.prepare(); self.write_candidate()
        completion=Mock(side_effect=AssertionError('Already qualified recovery must not call API'))
        output=self.root/'recovered'
        p.propose(self.workspace,output,completion=completion,max_turns=2,
                  checkpoint_dir=self.root/'recovery_checkpoint',recovery_outputs=self.candidate)
        completion.assert_not_called()
        receipt=json.loads((output/'validation.json').read_text())
        self.assertTrue(receipt['recovered_qualified'])
        self.assertEqual(receipt['proposer_requests'],0)
        self.assertEqual(receipt['usage'],{})
        self.assertTrue(p.check_candidate(self.workspace,output)['passed'])

    def response(self, calls=None, truncated=False):
        message={'role':'assistant','content':''}
        if calls is not None:message['tool_calls']=calls
        return {'choices':[{'message':message,'finish_reason':'length' if truncated else 'stop'}],
                'usage':{'prompt_tokens':80,'completion_tokens':20,'total_tokens':100}}

    def qualified_calls(self):
        files=self.files()
        calls=[{'id':name,'type':'function','function':{'name':'write_file',
                'arguments':json.dumps({'path':name,'content':files[name]})}}
               for name in ('diagnosis.json','bundle.json','artifacts/new.py','checks.json')]
        calls.append({'id':'check','type':'function','function':{'name':'check_candidate','arguments':'{}'}})
        return calls

    def test_qualified_candidate_kept_after_invalid_edit(self):
        self.prepare()
        broken=[{'id':'break','type':'function','function':{'name':'write_file',
                  'arguments':json.dumps({'path':'artifacts/new.py','content':fixtures.IDENTITY+'\nimport os\n'})}}]
        completion=Mock(side_effect=[self.response(self.qualified_calls()),self.response(broken)])
        output=self.root/'kept'
        p.propose(self.workspace,output,completion=completion,max_turns=2,checkpoint_dir=self.root/'checkpoint')
        self.assertTrue(p.check_candidate(self.workspace,output)['passed'])
        self.assertEqual((output/'artifacts/new.py').read_text(),fixtures.IDENTITY)
        receipt=json.loads((output/'validation.json').read_text())
        self.assertEqual(receipt['fallback_reason'],'later_edits_invalid_after_valid_candidate')
        self.assertEqual((receipt['proposer_requests'],receipt['usage']['total_tokens']),(2,200))

    def test_qualified_candidate_kept_after_output_truncation(self):
        self.prepare()
        completion=Mock(side_effect=[self.response(self.qualified_calls()),self.response(truncated=True)])
        output=self.root/'kept'
        p.propose(self.workspace,output,completion=completion,max_turns=2,checkpoint_dir=self.root/'checkpoint')
        self.assertTrue(p.check_candidate(self.workspace,output)['passed'])
        receipt=json.loads((output/'validation.json').read_text())
        self.assertEqual(receipt['fallback_reason'],'output_truncated_after_valid_candidate')
        self.assertEqual((receipt['proposer_requests'],receipt['usage']['total_tokens']),(2,200))

    def test_safe_regex_compile_not_builtin_or_shadowed(self):
        safe="from re import compile\nTARGET_LAYER='h2'\nPATTERN=compile('a')\nclass Harness:\n    def h2(self,message,history):\n        return message\n"
        path=self.root/'regex.py';path.write_text(safe)
        validate_source(path,'h2');load_artifact(path,'h2')
        for bad in (safe.replace('from re import compile\n',''),
                    safe.replace("PATTERN=compile('a')","compile=3\nPATTERN=compile('a')"),
                    safe.replace('def h2(self,message,history):','def h2(self,message,compile):')):
            path.write_text(bad)
            with self.assertRaises(ValueError):validate_source(path,'h2')

    def test_filtered_shared_state_owner_removed_not_mistaken_for_unknown(self):
        self.prepare();self.write_candidate()
        self.mutate('bundle.json',lambda b:b['operations'].append({**b['operations'][0],
            'operation_id':'bad_op','mechanism_id':'bad_mechanism','artifact':'artifacts/bad.py'}))
        (self.candidate/'artifacts/bad.py').write_text(fixtures.IDENTITY+'\nimport os\n')
        self.mutate('diagnosis.json',lambda r:r['shared_state_contract'].append({
            'key':'k','writers':['bad_mechanism'],'readers':['new_mechanism'],
            'value_contract':'optional string','reset':'per_episode'}))
        report=self.report()
        self.assertTrue(report['passed'],report)
        self.assertEqual(report['diagnosis']['shared_state_contract'][0]['writers'],[])

    def test_related_rules_share_cross_layer_fixture_not_unrelated_tests(self):
        self.prepare();files=self.files()
        bundle=json.loads(files['bundle.json'])
        bundle['operations'].append({'operation_id':'plan_op','mechanism_id':'planner','op':'add',
            'target_layer':'h4','hypothesis':'synthetic','artifact':'artifacts/plan.py'})
        files['bundle.json']=json.dumps(bundle)
        files['artifacts/plan.py']="TARGET_LAYER='h4'\nclass Harness:\n    def h4(self,messages,remaining):\n        return []\n"
        review=json.loads(files['diagnosis.json'])
        for case in review['cases']:case['operation_ids'].append('plan_op')
        review['rules'][0]['test_ids'].append('shared_cross')
        review['rules'].append({**review['rules'][0],'rule_id':'plan_rule','operation_ids':['plan_op'],
            'test_ids':['plan_trigger','plan_control','shared_cross']})
        files['diagnosis.json']=json.dumps(review)
        checks=json.loads(files['checks.json'])
        checks['tests'] += [{'test_id':name,'rule_id':'plan_rule','kind':kind,
            'steps':[{'hook':'h4','args':[[],2],'expected':[]}]} for name,kind in
            [('plan_trigger','trigger'),('plan_control','counterexample')]]
        checks['tests'].append({'test_id':'shared_cross','rule_id':'plan_rule','kind':'cross_layer',
            'steps':[{'hook':'h4','args':[[],2],'expected':[]},
                     {'hook':'h2','args':[fixtures.RAW,[]],'expected':fixtures.RAW}]})
        files['checks.json']=json.dumps(checks);self.write_candidate(files)
        self.assertTrue(self.report()['passed'],self.report())
        self.mutate('checks.json',lambda c:c['tests'][-1].update(kind='trigger'))
        self.assertFalse(self.report()['passed'])
