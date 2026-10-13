"""Offline coverage: unlimited agent turns, only three preflight repair cycles."""
import json
from unittest.mock import Mock, patch

from evolution.tests.test_life_repair import _revision_fixtures
from evolution.life_evolve import profile as p


fixtures = _revision_fixtures('_life_evolve_fixtures', 'test_life_optimizer.py',
                              optimizer=True, profile_module='evolution.life_evolve')


class EvolveTests(fixtures.OptimizerTests):
    def archived_candidate(self):
        before = self.seed / 'evals/initial_batch_00/life_train'
        after = self.seed / 'evals/candidate_03/life_train'
        p.shutil.copytree(self.eval, before)
        p.shutil.copytree(self.eval, after)
        summary = json.loads(json.dumps(self.summary))
        summary['per_task'].update({'2': 1, '3': 0})
        summary['successes'] = sum(summary['per_task'].values())
        p.write_json(after / 'summary.json', summary)
        for case_id in ('2', '3'):
            path = after / f'episode_{case_id}.json'
            row = json.loads(path.read_text())
            row['success'] = summary['per_task'][case_id]
            row['history'][2]['content'] = 'historical candidate behavior'
            row['calls'][0]['response']['choices'][0]['message']['content'] = 'historical candidate behavior'
            p.write_json(path, row)
        self.state['rounds'].append({'round': 3, 'train_batch': 0, 'accepted': False,
            'incumbent_successes': self.summary['successes'], 'candidate_successes': summary['successes']})
        p.write_json(self.seed / 'state.json', self.state)
        return after

    def stateful_evidence(self):
        from evolution.environment_state import StateLog
        for path in self.eval.glob('episode_*.json'):
            row=json.loads(path.read_text())
            log=StateLog()
            log.record({'db':{'bills':{'B1':{'paid':False}},'unqueried':'BUSINESS_ONLY_SENTINEL'}})
            row['initial_environment_state_index']=log.index
            row['calls'][0]['environment_state_index']=log.index
            log.record({'db':{'bills':{'B1':{'paid':True}},'unqueried':'BUSINESS_ONLY_SENTINEL'}})
            row['calls'][0]['action_environment_state_index']=log.index
            row['environment_states']=log.records
            p.write_json(path,row)

    def test_business_state_is_explicit_on_demand_and_boundary_specific(self):
        self.stateful_evidence();self.prepare()
        plain=p.tool_call(self.workspace,'read_case',{'case_id':'2','call_index':0})
        self.assertNotIn('BUSINESS_ONLY_SENTINEL',json.dumps(plain))
        for phase,paid in [('model',False),('action',True)]:
            view=p.tool_call(self.workspace,'read_case',{'case_id':'2','call_index':0,
                'environment_state':phase,'state_path':['db','bills','B1','paid']})
            self.assertIs(view['environment_state'],paid)
            self.assertNotIn('BUSINESS_ONLY_SENTINEL',json.dumps(view))
        ledger=json.loads((self.workspace/p.READ_LOG).read_text())
        self.assertEqual(ledger['requests'][-1]['environment_state'],'action')
        for args in ({'case_id':'2','environment_state':'action'},
                     {'case_id':'2','call_index':0,'state_path':['db']},
                     {'case_id':'2','call_index':0,'environment_state':'action','state_path':[True]}):
            with self.assertRaises(ValueError):p.tool_call(self.workspace,'read_case',args)

    def test_business_state_replay_uses_action_boundary_without_fixtures(self):
        self.stateful_evidence();self.prepare()
        source="""TARGET_LAYER='h2'
class Harness:
    def bind_environment_state(self,state):
        self.environment=state
    def h2(self,message,history):
        if not self.environment['db']['bills']['B1']['paid']:
            raise ValueError('wrong business-state boundary')
        return message
"""
        files=self.files(source)
        files = {name: text for name, text in files.items()
                 if name == 'bundle.json' or name.endswith('.py')}
        self.write_candidate(files)
        report=self.report()
        self.assertTrue(report['passed'],report)
        self.assertEqual((report['calls_replayed'],len(report['cases'])),(20,20))
        self.assertNotIn('tests',report)
        self.assertIn('evolution/environment_state.py',p.source_hashes())
        self.assertTrue((self.workspace/'source_view/environment_state.py').is_file())

    def test_business_state_tampering_is_rejected(self):
        self.stateful_evidence()
        path=self.eval/'episode_2.json'
        row=json.loads(path.read_text())
        row['environment_states'][0]['state']['db']['unqueried']='tampered'
        p.write_json(path,row)
        self.prepare();self.write_candidate()
        report=self.report()
        self.assertFalse(report['passed'])
        self.assertTrue(any('hash mismatch' in c.get('error','') for c in report['cases']))

    # Replace inherited historical assertions for the removed paperwork gates.
    # The sealed optimizer's original tests still run in test_life_repair.py.
    def test_diagnosis_must_precede_code_and_inputs_are_read_only(self):
        self.prepare()
        files = self.files()
        for name in ('artifacts/new.py', 'bundle.json'):
            p.tool_call(self.workspace, 'write_file', {'path': name, 'content': files[name]})
        self.assertTrue(p.check_candidate(self.workspace, self.workspace)['passed'])
        for name in ('inventory.json', '../escape.py', p.READ_LOG):
            with self.assertRaises(ValueError):
                p.tool_call(self.workspace, 'write_file', {'path': name, 'content': 'x'})
        with self.assertRaises(ValueError):
            p.tool_call(self.workspace, 'Bash', {'path': 'checks.json', 'content': 'x'})
        with self.assertRaises(ValueError):
            p.tool_call(self.workspace, 'read_file', {'path': '../../etc/passwd'})

    def test_diagnosis_requires_valid_citations_indices_and_case_support(self):
        self.prepare(); self.write_candidate()
        for content in ('not JSON', '{}', json.dumps({'cases': [{'case_id': 'unknown',
                'first_error_message_index': 999}], 'rules': [], 'extra_note': 'optional'})):
            (self.candidate / 'diagnosis.json').write_text(content)
            self.assertTrue(self.report()['passed'], self.report())
        (self.candidate / 'diagnosis.json').unlink()
        (self.candidate / 'checks.json').unlink()
        self.assertTrue(self.report()['passed'], self.report())

    def test_only_cited_subset_needs_analysis_and_replay(self):
        self.prepare(); self.write_candidate()
        self.mutate('diagnosis.json', lambda review: review.update(cases=review['cases'][:2]))
        report = self.report()
        self.assertTrue(report['passed'], report)
        self.assertEqual((len(report['cases']), report['calls_replayed']), (20, 20))

    def test_rule_requires_counterexample_and_assertion(self):
        self.prepare(); self.write_candidate()
        self.mutate('checks.json', lambda checks: checks['tests'][1].update(kind='boundary'))
        self.mutate('checks.json', lambda checks: checks['tests'][0]['steps'][0].pop('expected'))
        # Reproduces the paid candidate's top-level state assertion rejection.
        self.mutate('checks.json', lambda checks: checks['tests'][0].update(
            expected_state={'any_state': 'archival only'}))
        self.assertTrue(self.report()['passed'], self.report())

    def test_invalid_sibling_filtered_and_review_normalized(self):
        self.prepare(); self.write_candidate()
        self.mutate('bundle.json', lambda bundle: bundle['operations'].append({**bundle['operations'][0],
            'operation_id': 'bad_op', 'mechanism_id': 'bad_mechanism', 'artifact': 'artifacts/bad.py'}))
        (self.candidate / 'artifacts/bad.py').write_text(fixtures.IDENTITY + '\nimport os\n')
        self.mutate('diagnosis.json', lambda review: review['cases'][0]['operation_ids'].append('bad_op'))
        before = (self.candidate / 'diagnosis.json').read_bytes()
        report = self.report()
        self.assertTrue(report['passed'], report)
        self.assertEqual(len(report['rejected_operations']), 1)
        self.assertEqual(len(report['bundle']['operations']), 1)
        self.assertEqual(before, (self.candidate / 'diagnosis.json').read_bytes())

    def test_fixture_private_context_and_unbounded_steps_rejected(self):
        self.prepare(); self.write_candidate()
        self.mutate('checks.json', lambda checks: checks['tests'][0].update(
            context={'hidden_goal': 'fabricated, not a runtime input'},
            steps=[{'hook': 'missing_hook', 'args': []}] * 41))
        self.assertTrue(self.report()['passed'], self.report())

    def test_cross_layer_plan_action_fixture_and_state_reset(self):
        # Test actual coupled hooks on recorded inputs, not a declared test kind.
        for path in self.eval.glob('episode_*.json'):
            row = json.loads(path.read_text())
            second = json.loads(json.dumps(row['calls'][0]))
            second['turn'] = 2
            second['request']['messages'] = row['history']
            row['calls'].append(second)
            p.write_json(path, row)
        self.prepare()
        files = self.files("""TARGET_LAYER='h2'
class Harness:
    def bind_shared_state(self,state):
        self.state=state
    def h2(self,message,history):
        if len(history)==2 and self.state:
            raise ValueError('state leaked between episodes')
        if len(history)>2 and self.state.pop('next',None)!='planned':
            raise ValueError('H4 state not available to H2')
        return message
""")
        bundle = json.loads(files['bundle.json'])
        bundle['operations'].append({'operation_id': 'plan_op', 'mechanism_id': 'planner',
            'op': 'add', 'target_layer': 'h4', 'hypothesis': 'synthetic', 'artifact': 'artifacts/plan.py'})
        files = {'bundle.json': json.dumps(bundle), 'artifacts/new.py': files['artifacts/new.py'],
            'artifacts/plan.py': """TARGET_LAYER='h4'
class Harness:
    def bind_shared_state(self,state):
        self.state=state
    def h4(self,messages,remaining):
        self.state['next']='planned'
        return []
"""}
        self.write_candidate(files)
        report = self.report()
        self.assertTrue(report['passed'], report)
        self.assertEqual(report['calls_replayed'], 40)

    def test_unknown_shared_state_owner_rejected(self):
        self.prepare(); self.write_candidate()
        self.mutate('diagnosis.json', lambda review: review['shared_state_contract'].append({
            'key': 'k', 'writers': ['nonexistent'], 'readers': [], 'value_contract': 'string',
            'reset': 'per_episode'}))
        self.assertTrue(self.report()['passed'], self.report())

    def test_new_native_tools_bad_schema_and_false_assertions_rejected(self):
        self.prepare()
        for update in ("message['tool_calls'][0]['function']['name']='invented'",
                       "message['tool_calls'][0]['function']['arguments']='{\"value\":3}'",
                       "return []"):
            source = "TARGET_LAYER='h2'\nclass Harness:\n    def h2(self,message,history):\n        " + update + "\n        return message\n"
            self.write_candidate(self.files(source))
            self.assertFalse(self.report()['passed'])
        self.write_candidate()
        self.mutate('checks.json', lambda checks: checks['tests'][0]['steps'][0].update(
            expected={'role': 'assistant', 'content': 'false archival assertion'}))
        self.assertTrue(self.report()['passed'], self.report())

    def test_host_fills_bundle_binding_and_id_without_mutating_model_document(self):
        self.prepare(); self.write_candidate()
        document = json.loads((self.candidate / 'bundle.json').read_text())
        for values in ({}, {'domain': 'typo', 'base_snapshot_sha256': 'mistyped',
                           'bundle_id': 'Telecom-Version-1'}):
            model = {key: value for key, value in document.items()
                     if key not in {'domain', 'base_snapshot_sha256', 'bundle_id'}}
            model.update(values)
            original = json.dumps(model, sort_keys=True)
            inventory = json.loads((self.workspace / 'inventory.json').read_text())
            bundle, rejected = p.filter_operations(model, inventory['domain'], inventory['records'],
                inventory['snapshot_sha256'], self.candidate)
            self.assertFalse(rejected)
            self.assertEqual(bundle['domain'], inventory['domain'])
            self.assertEqual(bundle['base_snapshot_sha256'], inventory['snapshot_sha256'])
            self.assertIsNotNone(p.IDENTIFIER.fullmatch(bundle['bundle_id']))
            self.assertEqual(json.dumps(model, sort_keys=True), original)

    def test_all_rejections_include_missing_fields_and_artifact_errors(self):
        self.prepare(); self.write_candidate()
        document = json.loads((self.candidate / 'bundle.json').read_text())
        modify = {key: value for key, value in document['operations'][0].items() if key != 'artifact'}
        modify.update(operation_id='modify_op', mechanism_id='identity', op='modify')
        unsafe = {**document['operations'][0], 'artifact': 'artifacts/unsafe.py'}
        (self.candidate / unsafe['artifact']).write_text(fixtures.IDENTITY + '\nimport os\n')
        document['operations'] = [modify, unsafe]
        p.write_json(self.candidate / 'bundle.json', document)
        before = (self.candidate / 'bundle.json').read_bytes()
        report = self.report()
        self.assertFalse(report['passed'])
        rejected = report['rejected_operations']
        self.assertEqual([item['index'] for item in rejected], [0, 1])
        self.assertEqual(rejected[0]['operation_id'], 'modify_op')
        self.assertEqual(rejected[0]['mechanism_id'], 'identity')
        self.assertIn('artifact', rejected[0]['error'])
        self.assertNotIn('expected_revision', rejected[0]['error'])
        self.assertNotIn('expected_sha256', rejected[0]['error'])
        self.assertEqual(rejected[1]['artifact'], 'artifacts/unsafe.py')
        self.assertIn('allowed from-imports', rejected[1]['error'])
        self.assertIn('operations[0]', report['error'])
        self.assertEqual(before, (self.candidate / 'bundle.json').read_bytes())

    def test_all_rejections_keep_syntax_missing_file_cas_and_path_errors(self):
        self.prepare(); self.write_candidate()
        document = json.loads((self.candidate / 'bundle.json').read_text())
        original = document['operations'][0]
        record = json.loads((self.workspace / 'inventory.json').read_text())['records'][0]
        (self.candidate / 'artifacts/syntax.py').write_text('class Harness(:\n')
        document['operations'] = [
            {**original, 'operation_id': 'syntax_op', 'artifact': 'artifacts/syntax.py'},
            {**original, 'operation_id': 'missing_op', 'artifact': 'artifacts/missing.py'},
            {**original, 'operation_id': 'stale_op', 'mechanism_id': record['mechanism_id'],
             'op': 'modify', 'expected_revision': record['revision'] + 1, 'expected_sha256': record['sha256']},
            {**original, 'operation_id': 'escape_op', 'artifact': '../escape.py'},
        ]
        p.write_json(self.candidate / 'bundle.json', document)
        report = self.report()
        self.assertFalse(report['passed'])
        rejected = report['rejected_operations']
        self.assertEqual(len(rejected), 4)
        for item, message in zip(rejected, ('SyntaxError', 'FileNotFoundError', 'stale mechanism', 'artifacts/*.py')):
            self.assertIn(message, item['error'])

    def test_omitted_version_fields_are_bound_from_frozen_inventory(self):
        self.prepare()
        record = json.loads((self.workspace / 'inventory.json').read_text())['records'][0]
        for operation in ('modify', 'delete'):
            with self.subTest(operation=operation):
                files = self.files()
                document = json.loads(files['bundle.json'])
                op = document['operations'][0]
                op.update(op=operation, mechanism_id=record['mechanism_id'], target_layer=record['target_layer'])
                if operation == 'delete':
                    op.pop('artifact')
                files['bundle.json'] = json.dumps(document)
                self.write_candidate(files)
                before = (self.candidate / 'bundle.json').read_bytes()
                report = self.report()
                self.assertTrue(report['passed'], report)
                canonical = report['bundle']['operations'][0]
                self.assertEqual(canonical['expected_revision'], record['revision'])
                self.assertEqual(canonical['expected_sha256'], record['sha256'])
                self.assertEqual(before, (self.candidate / 'bundle.json').read_bytes())
                p.verify_inputs(self.workspace)

    def test_version_binding_preserves_supplied_errors_and_unknown_mechanisms(self):
        self.prepare()
        record = json.loads((self.workspace / 'inventory.json').read_text())['records'][0]
        for override in ({'expected_revision': 99}, {'expected_sha256': 'wrong'},
                         {'expected_revision': None}, {'expected_sha256': None},
                         {'mechanism_id': 'unknown_mechanism'}):
            with self.subTest(override=override):
                files = self.files(); document = json.loads(files['bundle.json'])
                document['operations'][0].update({'op': 'modify', 'mechanism_id': record['mechanism_id'], **override})
                files['bundle.json'] = json.dumps(document)
                self.write_candidate(files)
                self.assertFalse(self.report()['passed'])

    def test_current_interface_all_rejections_are_also_returned(self):
        self.prepare(); self.write_candidate()
        with patch.object(p, 'validate_source', side_effect=ValueError('current interface failure')):
            report = self.report()
        self.assertFalse(report['passed'])
        self.assertEqual(report['rejected_operations'][0]['artifact'], 'artifacts/new.py')
        self.assertIn('current interface failure', report['rejected_operations'][0]['error'])

    def test_root_and_common_field_errors_name_missing_and_unexpected_fields(self):
        self.prepare(); self.write_candidate()
        self.mutate('bundle.json', lambda document: document.pop('hypothesis'))
        self.assertIn("missing=['hypothesis']", self.report()['error'])
        self.write_candidate()
        self.mutate('bundle.json', lambda document: document['operations'][0].update(extra='unexpected'))
        self.assertIn("unexpected=['extra']", self.report()['rejected_operations'][0]['error'])
        self.write_candidate()
        self.mutate('bundle.json', lambda document: document['operations'][0].pop('target_layer'))
        self.assertIn('target_layer', self.report()['rejected_operations'][0]['error'])

    def test_proposer_receives_all_rejection_details_then_repairs_once(self):
        self.prepare()
        files = self.files()
        document = json.loads(files['bundle.json'])
        document['operations'][0].pop('artifact')
        calls = [{'id': 'bad_bundle', 'type': 'function', 'function': {'name': 'write_file',
            'arguments': json.dumps({'path': 'bundle.json', 'content': json.dumps(document)})}}] + self.check_calls()
        responses = [self.response(calls), self.response(self.qualified_calls()), self.response()]
        def completion(messages, tools):
            if len(responses) == 2:
                feedback = [json.loads(message['content']) for message in messages if message['role'] == 'tool'][-1]
                self.assertFalse(feedback['passed'])
                self.assertEqual(feedback['rejected_operations'][0]['operation_id'], 'new_op')
                self.assertIn("missing=['artifact']", feedback['rejected_operations'][0]['error'])
                self.assertIn('rejected_operations', messages[-1]['content'])
            return responses.pop(0)
        output = self.root / 'detailed_repair'
        p.propose(self.workspace, output, completion=completion, checkpoint_dir=self.root / 'checkpoint')
        receipt = self.validation(output)
        self.assertEqual((receipt['repairs_granted'], receipt['failed_preflights']), (1, 1))
        self.assertEqual(receipt['proposer_requests'], 3)
        self.assertTrue(p.check_candidate(self.workspace, output)['passed'])

    def test_bundle_constraint_diagnostics_matrix(self):
        self.prepare(); self.write_candidate()
        base = json.loads((self.candidate / 'bundle.json').read_text())
        op = base['operations'][0]
        record = json.loads((self.workspace / 'inventory.json').read_text())['records'][0]
        cases = [
            ([{**op, 'operation_id': 'BAD'}], ('operation_id', 'received')),
            ([{**op, 'op': []}], ('add/modify/delete', 'op=[]')),
            ([{**op, 'target_layer': 'wrong'}], ('target_layer', 'h2')),
            ([op, op], ('duplicate', 'new_op')),
            ([{**op, 'mechanism_id': record['mechanism_id']}], ('already exists', 'modify')),
            ([{**op, 'op': 'modify', 'mechanism_id': record['mechanism_id'],
               'expected_revision': 99, 'expected_sha256': 'wrong'}], ('current inventory record', record['sha256'])),
            ([{key: value for key, value in {**op, 'op': 'delete', 'mechanism_id': record['mechanism_id'],
                'target_layer': 'h4', 'expected_revision': record['revision'], 'expected_sha256': record['sha256']}.items()
              if key != 'artifact'}], ('delete layer mismatch', 'h2', 'h4')),
        ]
        for operations, fragments in cases:
            with self.subTest(operations=operations):
                p.write_json(self.candidate / 'bundle.json', {**base, 'operations': operations})
                report = self.report()
                if operations == [op, op]:
                    self.assertTrue(report['passed'])
                else:
                    self.assertFalse(report['passed'])
                error = report['rejected_operations'][0]['error']
                for fragment in fragments: self.assertIn(fragment, error)

    def test_static_constraint_diagnostics_matrix(self):
        self.prepare()
        cases = [
            (fixtures.IDENTITY + '\nfrom os import getenv\n', ('unsafe import', 'line', 'os')),
            (fixtures.IDENTITY.replace('self,message,history', 'self,message,history=None'), ('invalid signature', 'positional', 'line')),
            (fixtures.IDENTITY.replace("TARGET_LAYER='h2'", "TARGET_LAYER='h3'"), ('TARGET_LAYER', 'h2', 'h3')),
            (fixtures.IDENTITY + '\n# ground_truth\n', ('private-data reference', 'ground_truth')),
            (fixtures.IDENTITY.replace('return message', 'return message.__class__'), ('dunder', '__class__', 'line')),
            (fixtures.IDENTITY.replace('class Harness:', 'class Harness(dict):'), ('inheritance', 'line')),
            (fixtures.IDENTITY + '\n    def h3_messages(self,messages):\n        return messages\n', ('optional hook', 'h3')),
        ]
        for source, fragments in cases:
            with self.subTest(source=source):
                self.write_candidate(self.files(source)); report = self.report()
                self.assertFalse(report['passed'])
                detail = report['rejected_operations'][0]
                self.assertEqual(detail['artifact'], 'artifacts/new.py')
                for fragment in fragments: self.assertIn(fragment, detail['error'])

    def test_runtime_constraint_diagnostics_matrix(self):
        for path in self.eval.glob('episode_*.json'):
            row = json.loads(path.read_text()); row['calls'].append(json.loads(json.dumps(row['calls'][0])))
            p.write_json(path, row)
        self.prepare()
        cases = [
            ('h2', "def h2(self,message,history):\n        return {'role':'user','content':'wrong'}", 'h2', 'assistant'),
            ('h2', 'def h2(self,message,history):\n        return message\n    def h2_no_tool_message(self,message,remaining):\n        return 7', 'h2_no_tool_message', 'string or None'),
            ('h2', 'def h2(self,message,history):\n        return message\n    def drain_h2_messages(self):\n        return {}', 'drain_h2_messages', 'list'),
            ('h4', "def h4(self,messages,remaining):\n        return 'wrong'", 'h4', "[] or ['text']"),
            ('h5', "def h5(self,messages):\n        return ['one','two']", 'h5', 'count=2'),
            ('h3', "def h3(self,tools):\n        tools[0]['function']['name']='invented'\n        return tools", 'h3', 'parameter schemas'),
            ('h3', "def h3(self,tools):\n        return tools\n    def h3_messages(self,messages):\n        return []", 'h3_messages', 'count/order'),
        ]
        for layer, method, hook, fragment in cases:
            with self.subTest(hook=hook):
                self.write_candidate(self.files(f"TARGET_LAYER='{layer}'\nclass Harness:\n    {method}\n", layer))
                report = self.report(); self.assertFalse(report['passed'])
                detail = report['cases'][0]
                self.assertEqual(detail['hook'], hook)
                self.assertEqual(detail['artifact'], 'artifacts/new.py')
                self.assertEqual(detail['mechanism_id'], 'new_mechanism')
                self.assertIsInstance(detail['line'], int)
                self.assertIn(fragment, detail['error'])
                self.assertIn('call_index', detail)

    def test_nested_action_schema_errors_include_field_paths(self):
        from evolution.life_optimizer_v25.worker import check_schema
        schema = {'type': 'object', 'required': ['items'], 'additionalProperties': False, 'properties': {
            'items': {'type': 'array', 'items': {'type': 'object', 'required': ['count'],
                'additionalProperties': False, 'properties': {'count': {'type': 'integer', 'enum': [1]}}}}}}
        for value, fragment in [({}, "missing=['items']"), ({'extra': 1, 'items': []}, "unexpected=['extra']"),
                ({'items': [{}]}, "arguments.items[0]"), ({'items': [{'count': 'wrong'}]}, "arguments.items[0].count"),
                ({'items': [{'count': 2}]}, 'enum=[1]')]:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, fragment.replace('[', r'\[').replace(']', r'\]')):
                check_schema(value, schema)
        check_schema({'items': [{'count': 1}]}, schema)

    def test_search_pages_are_utf8_offsets_and_evidence_reads_are_recorded(self):
        self.prepare()
        source = fixtures.IDENTITY + '#marker-é\n' * 25
        p.tool_call(self.workspace, 'write_file', {'path': 'artifacts/new.py', 'content': source})
        first = p.tool_call(self.workspace, 'search_file', {'path': 'artifacts/new.py', 'text': 'marker-é'})
        second = p.tool_call(self.workspace, 'search_file', {'path': 'artifacts/new.py', 'text': 'marker-é', 'start_byte': first['next_byte']})
        self.assertEqual((len(first['matches']), len(second['matches'])), (20, 5))
        self.assertFalse(first['eof']); self.assertTrue(second['eof'])
        for match in first['matches'] + second['matches']:
            self.assertEqual(source.encode()[match['start_byte']:match['end_byte']].decode(), 'marker-é')
        p.tool_call(self.workspace, 'search_file', {'path': 'evidence/case_2.json', 'text': 'case_id'})
        ledger = json.loads((self.workspace / p.READ_LOG).read_text())
        self.assertEqual((ledger['requests'][-1]['tool'], ledger['case_ids']), ('search_file', ['2']))
        with self.assertRaisesRegex(ValueError, 'allowlist'):
            p.tool_call(self.workspace, 'search_file', {'path': 'unlisted.json', 'text': 'anything'})

    def test_replace_is_unique_atomic_and_cannot_edit_inputs(self):
        self.prepare()
        args = {'path': 'artifacts/new.py', 'content': fixtures.IDENTITY}
        p.tool_call(self.workspace, 'write_file', args)
        path = self.workspace / args['path']
        p.tool_call(self.workspace, 'replace_file', {'path': args['path'], 'old_text': 'return message', 'new_text': 'return message # retained'})
        expected = path.read_bytes()
        for old in ('absent', ' '):
            with self.assertRaisesRegex(ValueError, 'exact match'):
                p.tool_call(self.workspace, 'replace_file', {'path': args['path'], 'old_text': old, 'new_text': 'bad'})
            self.assertEqual(expected, path.read_bytes())
        with self.assertRaisesRegex(ValueError, 'frozen input'):
            p.tool_call(self.workspace, 'replace_file', {'path': 'inventory.json', 'old_text': 'domain', 'new_text': 'bad'})
        p.verify_inputs(self.workspace)

    def test_preview_is_isolated_prefix_diagnostics_not_qualification(self):
        self.prepare()
        for name, content in self.files().items():
            if name in ('bundle.json', 'artifacts/new.py'):
                p.tool_call(self.workspace, 'write_file', {'path': name, 'content': content})
        with patch.object(p, 'model_router_completion', side_effect=AssertionError('no paid preview')):
            report = p.tool_call(self.workspace, 'preview_candidate', {'case_id': '2', 'call_index': 0})
        self.assertTrue(report['passed'], report); self.assertFalse(report['qualified'])
        self.assertEqual(report['calls_replayed'], 1)
        self.assertEqual(report['previews'][0]['action_before'], report['previews'][0]['action_after'])
        self.assertEqual(report['previews'][0]['guidance_hook'], 'h5')
        self.assertTrue(p.tool_call(self.workspace, 'check_candidate', {})['passed'])
        with self.assertRaisesRegex(ValueError, 'expected integer'):
            p.preview_candidate(self.workspace, '2', 100)

    def test_preview_before_coding_uses_incumbent_without_qualifying(self):
        self.prepare()
        before = (self.workspace / 'receipt.json').read_bytes()
        with patch.object(p, 'model_router_completion', side_effect=AssertionError('no paid preview')):
            report = p.tool_call(self.workspace, 'preview_candidate', {'case_id': '2', 'call_index': 0})
        self.assertTrue(report['passed'], report)
        self.assertEqual(report['chain'], 'incumbent')
        self.assertFalse(report['qualified'])
        self.assertEqual(report['calls_replayed'], 1)
        self.assertEqual(before, (self.workspace / 'receipt.json').read_bytes())
        self.assertFalse((self.workspace / 'bundle.json').exists())
        self.assertFalse(p.check_candidate(self.workspace, self.workspace)['passed'])
        p.verify_inputs(self.workspace)

    def test_explicit_incumbent_preview_ignores_broken_working_candidate(self):
        self.prepare()
        p.tool_call(self.workspace, 'write_file', {'path': 'bundle.json', 'content': '{'})
        report = p.tool_call(self.workspace, 'preview_candidate', {
            'case_id': '2', 'call_index': 0, 'chain': 'incumbent'})
        self.assertTrue(report['passed'], report)
        self.assertEqual(report['chain'], 'incumbent')
        for chain in (None, 'candidate'):
            with self.subTest(chain=chain), self.assertRaises(json.JSONDecodeError):
                p.preview_candidate(self.workspace, '2', 0, chain=chain)
        self.assertFalse(p.check_candidate(self.workspace, self.workspace)['passed'])

    def test_preview_chain_errors_and_schema_are_actionable(self):
        self.prepare()
        with self.assertRaisesRegex(ValueError, "write them first or.*chain='incumbent'"):
            p.preview_candidate(self.workspace, '2', 0, chain='candidate')
        for chain in ('unknown', [], True):
            with self.subTest(chain=chain), self.assertRaisesRegex(ValueError, 'preview chain'):
                p.preview_candidate(self.workspace, '2', 0, chain=chain)
        schema = next(row['function']['parameters'] for row in p.TOOLS
                      if row['function']['name'] == 'preview_candidate')
        self.assertEqual(schema['properties']['chain']['enum'], ['incumbent', 'candidate'])
        self.assertNotIn('chain', schema['required'])

    def test_old_batch_file_errors_route_to_paired_historical_evidence(self):
        self.prepare()
        for name, args in (
                ('read_case', {'case_id': 'not_in_this_batch'}),
                ('read_file', {'path': 'evidence/case_old.json'}),
                ('search_file', {'path': 'evidence/case_old.json', 'text': 'case_id'})):
            with self.subTest(tool=name), self.assertRaises(ValueError) as failure:
                p.tool_call(self.workspace, name, args)
            for fragment in ('history_index', 'history_variant', 'candidate', 'incumbent'):
                self.assertIn(fragment, str(failure.exception))

    def test_grouped_feedback_preserves_every_failure_and_full_report(self):
        cases = [{'case_id': str(index), 'call_index': index, 'passed': False, 'hook': 'h2', 'error': 'same error'}
                 for index in range(50)]
        report = {'passed': False, 'cases': cases, 'bundle': {}, 'calls_replayed': 0}
        feedback = p.candidate_feedback(report)
        self.assertEqual(len(feedback['replay_failures']), 1)
        self.assertEqual(len(feedback['replay_failures'][0]['occurrences']), 50)
        self.assertEqual(feedback['replayed_cases'], 50)
        self.assertNotIn('bundle', feedback)
        self.assertEqual(len(report['cases']), 50)

    def test_preview_shows_coupled_hooks_and_state_without_shortening_full_check(self):
        for path in self.eval.glob('episode_*.json'):
            row = json.loads(path.read_text()); row['calls'].append(json.loads(json.dumps(row['calls'][0])))
            p.write_json(path, row)
        self.prepare()
        files = self.files(); bundle = json.loads(files['bundle.json'])
        sources = {
            'h3': "def h3(self,tools):\n        tools[0]['function']['description']='preview desc'\n        return tools\n    def h3_messages(self,messages):\n        messages[0]['content']+=' preview'\n        return messages",
            'h4': "def h4(self,messages,remaining):\n        self.state['pending']=True\n        return ['progress']",
            'h5': "def h5(self,messages):\n        self.state['started']=True\n        return ['startup']",
        }
        for layer, methods in sources.items():
            relative = f'artifacts/{layer}.py'
            bundle['operations'].append({'operation_id': layer+'_op', 'mechanism_id': layer+'_mechanism',
                'op': 'add', 'target_layer': layer, 'hypothesis': 'synthetic preview', 'artifact': relative})
            files[relative] = f"TARGET_LAYER='{layer}'\nclass Harness:\n    def bind_shared_state(self,state):\n        self.state=state\n    {methods}\n"
        files['bundle.json'] = json.dumps(bundle)
        for name, content in files.items():
            if name == 'bundle.json' or name.endswith('.py'):
                p.tool_call(self.workspace, 'write_file', {'path': name, 'content': content})
        report = p.preview_candidate(self.workspace, '2', 1); self.assertTrue(report['passed'], report)
        preview = report['previews'][0]
        self.assertEqual((report['calls_replayed'], report['cases'][0]['calls']), (2, 2))
        self.assertEqual((preview['guidance_hook'], preview['guidance']), ('h4', ['progress']))
        self.assertEqual(preview['tool_description_changes'][0]['after'], 'preview desc')
        self.assertTrue(preview['model_view_changes'])
        self.assertTrue(preview['shared_state_summary']['started']['value'])
        self.assertTrue(preview['shared_state_summary']['pending']['value'])
        self.assertEqual(p.check_candidate(self.workspace, self.workspace)['calls_replayed'], 40)

    def test_preview_failure_does_not_consume_full_preflight_repairs(self):
        self.prepare()
        calls = self.qualified_calls()[:-1]
        bad = fixtures.IDENTITY.replace('return message', "return {'role':'user','content':'wrong'}")
        calls[0]['function']['arguments'] = json.dumps({'path': 'artifacts/new.py', 'content': bad})
        calls.append({'id': 'preview', 'type': 'function', 'function': {'name': 'preview_candidate',
            'arguments': json.dumps({'case_id': '2', 'call_index': 0})}})
        completion = Mock(side_effect=[self.response(calls), self.response(self.qualified_calls()), self.response()])
        output = self.root / 'preview_repair'
        p.propose(self.workspace, output, completion=completion, checkpoint_dir=self.root / 'checkpoint')
        receipt = self.validation(output)
        self.assertEqual((receipt['repairs_granted'], receipt['failed_preflights']), (0, 0))
        saved = json.loads((self.root / 'checkpoint/agent_checkpoint.json').read_text())
        preview = [json.loads(message['content']) for message in saved['messages']
                   if message['role'] == 'tool' and message['tool_call_id'] == 'preview'][0]
        self.assertFalse(preview['passed']); self.assertFalse(preview['qualified'])
        self.assertIn('hook', preview['cases'][0])

    def test_worker_and_usage_failures_have_actionable_diagnostics(self):
        self.prepare(); self.write_candidate()
        with patch.object(p.subprocess, 'run', return_value=Mock(returncode=-24, stderr='cpu limit', stdout='')):
            report = self.report()
        self.assertFalse(report['passed'])
        for fragment in ('SIGXCPU', 'CPU=3s', 'case_ids', 'cpu limit'): self.assertIn(fragment, report['error'])
        self.assertEqual(report['stage'], 'hook_replay')
        for result in ([], {'usage': None}, {'usage': {'prompt_tokens': True, 'completion_tokens': 1}}):
            with self.subTest(result=result), self.assertRaisesRegex(RuntimeError, 'expected'):
                p.verify_usage(result)

    def test_history_code_is_hash_verified_read_only_and_not_active(self):
        source = self.seed / 'proposals/round_03/artifacts/old.py'
        source.parent.mkdir(parents=True); source.write_text(fixtures.IDENTITY + '# prior rejected code\n')
        row = {'round': 3, 'accepted': False, 'operations': [{'operation_id': 'prior_op',
            'mechanism_id': 'prior_mechanism', 'op': 'add', 'target_layer': 'h2',
            'artifact': 'artifacts/old.py', 'artifact_sha256': p.digest(source)}]}
        self.state['rounds'].append(row); p.write_json(self.seed / 'state.json', self.state)
        before = p.digest(self.seed / 'state.json'); self.prepare()
        op = json.loads((self.workspace / 'history.json').read_text())[-1]['operations'][0]
        self.assertEqual(p.digest(self.workspace / op['historical_code_path']), p.digest(source))
        self.assertIn('prior rejected code', p.tool_call(self.workspace, 'read_file', {'path': op['historical_code_path']})['content'])
        self.assertEqual(len(json.loads((self.workspace / 'inventory.json').read_text())['records']), 1)
        with self.assertRaises(ValueError):
            p.tool_call(self.workspace, 'replace_file', {'path': op['historical_code_path'], 'old_text': 'prior', 'new_text': 'bad'})
        self.assertEqual(before, p.digest(self.seed / 'state.json'))
        row['operations'][0]['artifact_sha256'] = 'mismatch'
        unavailable = p.archive_history_code(self.seed, self.workspace, row, row['operations'][0])
        self.assertIn('historical_code_unavailable', unavailable)

    def test_paired_history_feedback_requires_train_matching_keys_and_counts(self):
        before = {**self.summary}; after = {**before, 'per_task': {**before['per_task'], '0': 0, '2': 1}}
        after['successes'] = sum(after['per_task'].values())
        row = {'round': 3, 'train_batch': 2, 'incumbent_successes': before['successes'], 'candidate_successes': after['successes']}
        a = self.seed / 'evals/initial_batch_02/life_train/summary.json'
        b = self.seed / 'evals/candidate_03/life_train/summary.json'
        p.write_json(a, before); p.write_json(b, after)
        result = p.paired_history_outcomes(self.seed, row)
        self.assertEqual(result['paired_gain_case_ids'], ['2']); self.assertEqual(result['paired_regression_case_ids'], ['0'])
        p.write_json(b, {**after, 'split': 'test'})
        self.assertEqual(p.paired_history_outcomes(self.seed, row), {})

    def test_public_pattern_groups_separate_infra_and_include_controls(self):
        self.prepare()
        overview = json.loads((self.workspace / 'pattern_overview.json').read_text())
        self.assertEqual(overview['infra_case_ids'], ['1'])
        self.assertEqual(set(overview['successful_case_ids']), {str(i) for i in range(0, 20, 3)})
        self.assertNotIn('1', overview['behavioral_failure_case_ids'])
        self.assertIn('not root causes', overview['note'])
        for group in overview['groups']:
            self.assertFalse(set(group['infra_case_ids']) & set(group['failed_case_ids']))

    def test_strategy_history_keeps_bundle_scores_not_private_test_values(self):
        self.state['rounds'].extend([
            {'round': 2, 'train_batch': 0, 'accepted': False,
             'incumbent_score': .4, 'candidate_score': .3,
             'candidate_successes': 15, 'test_accuracy': 'PRIVATE_SENTINEL',
             'operations': [{'operation_id': 'change', 'mechanism_id': 'identity',
                             'op': 'modify', 'target_layer': 'h2', 'hypothesis': 'try change'}]},
            {'round': 3, 'train_batch': 1, 'accepted': True,
             'operations': [{'mechanism_id': 'removed', 'op': 'delete', 'target_layer': 'h4'}]}])
        p.write_json(self.seed / 'state.json', self.state)
        self.prepare()
        history = json.loads((self.workspace / 'strategy_history.json').read_text())
        identity = next(m for m in history['mechanisms'] if m['mechanism_id'] == 'identity')
        self.assertTrue(identity['active'])
        self.assertEqual(identity['attempts'][0]['candidate_score'], .3)
        self.assertFalse(identity['attempts'][0]['accepted'])
        removed = next(m for m in history['mechanisms'] if m['mechanism_id'] == 'removed')
        self.assertFalse(removed['active'])
        self.assertNotIn('candidate_score', removed['attempts'][0])
        self.assertNotIn('PRIVATE_SENTINEL', json.dumps(history))
        self.assertIn('not individual', history['note'])

    def test_additional_train_history_deduplicates_inherited_rows_and_preserves_distinct_code(self):
        import copy
        source=self.seed/'proposals/round_03/artifacts/old.py'
        source.parent.mkdir(parents=True); source.write_text(fixtures.IDENTITY+'# rejected original\n')
        row={'round':3,'train_batch':2,'accepted':False,'candidate_score':.3,
             'bundle_id':'past_bundle','operations':[{'operation_id':'same_op','mechanism_id':'past_mechanism',
             'op':'add','target_layer':'h2','hypothesis':'same hypothesis','artifact':'artifacts/old.py',
             'artifact_sha256':p.digest(source)}]}
        self.state['rounds'].append(row); p.write_json(self.seed/'state.json',self.state)
        extra=self.root/'another_branch/telecom'
        other=extra/'proposals/round_03/artifacts/old.py'
        other.parent.mkdir(parents=True); other.write_text(fixtures.IDENTITY+'# rejected alternate\n')
        past=copy.deepcopy(self.state)
        past['rounds'][-1]['operations'][0]['artifact_sha256']=p.digest(other)
        past['rounds'][-1]['test_accuracy']='PRIVATE_SENTINEL'
        p.write_json(extra/'state.json',past)
        p.prepare(self.seed,self.eval,self.workspace,additional_history=[extra])
        history=json.loads((self.workspace/'history.json').read_text())
        self.assertEqual(len(history),3)
        a,b=[item['operations'][0] for item in history if item.get('round')==3]
        self.assertNotEqual(a['historical_code_path'],b['historical_code_path'])
        self.assertEqual(p.digest(self.workspace/a['historical_code_path']),p.digest(source))
        self.assertEqual(p.digest(self.workspace/b['historical_code_path']),p.digest(other))
        self.assertEqual(history[-1]['history_source'],'another_branch')
        self.assertNotIn('PRIVATE_SENTINEL',json.dumps(history))
        self.assertEqual(len(json.loads((self.workspace/'inventory.json').read_text())['records']),1)
        p.verify_inputs(self.workspace)
        past['domain']='airline';p.write_json(extra/'state.json',past)
        with self.assertRaisesRegex(ValueError,'history domain mismatch'):
            p.prepare(self.seed,self.eval,self.root/'bad_history',additional_history=[extra])

    def test_schema_reference_is_copied_fingerprinted_and_immutable(self):
        self.prepare()
        self.assertEqual((self.workspace / 'SCHEMAS.md').read_bytes(),
                         p.Path(p.__file__).with_name('SCHEMAS.md').read_bytes())
        self.assertIn('evolution/life_evolve/SCHEMAS.md', p.source_hashes())
        (self.workspace / 'SCHEMAS.md').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'frozen input changed'):
            p.verify_inputs(self.workspace)

    def test_read_case_restores_history_indices_and_audits(self):
        row_path = self.eval / 'episode_2.json'
        row = json.loads(row_path.read_text())
        row['history'].append(row['history'][1])  # Dedup means pool index != history index.
        p.write_json(row_path, row)
        self.prepare()
        view = p.tool_call(self.workspace, 'read_case', {'case_id': '2', 'message_count': 1000})
        self.assertEqual([r['message_index'] for r in view['history']], list(range(5)))
        self.assertEqual(view['history'][4]['message'], view['history'][1]['message'])
        self.assertEqual(view['history'][3]['message']['content'], 'invalid action')
        self.assertTrue(view['eof'])
        self.assertNotIn('PRIVATE_SENTINEL', json.dumps(view))
        ledger = json.loads((self.workspace / p.READ_LOG).read_text())
        self.assertEqual(ledger['case_ids'], ['2'])
        self.assertEqual(ledger['requests'][0]['message_count'], 5)

    def test_read_case_recorded_request_response_and_pagination(self):
        self.prepare()
        call = p.tool_call(self.workspace, 'read_case', {'case_id': '2', 'call_index': 0})['call']
        self.assertEqual(call['assistant'], fixtures.RAW)
        self.assertEqual(len(call['messages']), 2)
        page = p.tool_call(self.workspace, 'read_case',
                           {'case_id': '2', 'start_message': 2, 'message_count': 1})
        self.assertEqual((page['next_message'], page['eof']), (3, False))
        self.assertEqual(page['history'][0]['message_index'], 2)
        self.assertEqual(page['history'][0]['message'], fixtures.RAW)

    def test_read_case_setup_is_optional_and_recorded_call_messages_page_independently(self):
        self.prepare()
        plain = p.tool_call(self.workspace, 'read_case', {'case_id': '2', 'call_index': 0})
        self.assertNotIn('context', plain)
        self.assertNotIn('tools', plain)
        full = p.tool_call(self.workspace, 'read_case', {'case_id': '2', 'include_setup': True})
        self.assertEqual(full['context'], {'instruction': 'Find a synthetic object'})
        self.assertEqual(full['tools'], fixtures.TOOLS)
        page = p.tool_call(self.workspace, 'read_case', {'case_id': '2', 'call_index': 0,
            'start_message': 1, 'message_count': 1})
        self.assertEqual(page['call']['messages'], plain['call']['messages'][1:2])
        self.assertEqual((page['call']['full_message_count'], page['call']['next_message'], page['call']['eof']), (2, 2, True))
        self.assertEqual(page['call']['assistant'], fixtures.RAW)

    def test_valid_dynamic_h3_descriptions_remain_readable_at_each_call(self):
        path = self.eval / 'episode_2.json'
        row = json.loads(path.read_text())
        call = json.loads(json.dumps(row['calls'][0]))
        call['turn'] = 2
        call['request']['tools'][0]['function']['description'] = 'public H3-updated description'
        row['calls'].append(call)
        p.write_json(path, row)
        self.prepare()
        first = p.tool_call(self.workspace, 'read_case', {'case_id': '2', 'call_index': 0, 'include_setup': True})
        later = p.tool_call(self.workspace, 'read_case', {'case_id': '2', 'call_index': 1, 'include_setup': True})
        self.assertEqual(first['tools'][0]['function']['description'], 'native')
        self.assertEqual(later['tools'][0]['function']['description'], 'public H3-updated description')
        self.assertEqual(later['tools'][0]['function']['parameters'], fixtures.TOOLS[0]['function']['parameters'])
        self.assertNotIn('tools', p.tool_call(self.workspace, 'read_case', {'case_id': '2', 'call_index': 1}))
        self.assertNotIn('PRIVATE_SENTINEL', json.dumps(later))
        self.write_candidate()
        report = self.report()
        self.assertTrue(report['passed'], report)
        self.assertEqual(report['calls_replayed'], 21)

    def test_native_name_parameter_or_type_changes_still_fail_with_specific_feedback(self):
        row = json.loads((self.eval / 'episode_2.json').read_text())
        for field in ('name', 'parameters', 'type'):
            changed = json.loads(json.dumps(row))
            call = json.loads(json.dumps(row['calls'][0]));call['turn'] = 2
            tool = call['request']['tools'][0]
            if field == 'type':tool['type'] = 'different_native_type'
            elif field == 'name':tool['function']['name'] = 'different_native_name'
            else:tool['function']['parameters']['properties']['value']['type'] = 'integer'
            changed['calls'].append(call)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'native tool names/parameters.*case=2.*turn=2'):
                p.compact_case(changed, None)

    def test_read_case_selects_actual_historical_candidate_without_private_fields(self):
        after = self.archived_candidate()
        self.prepare()
        history = json.loads((self.workspace / 'history.json').read_text())
        self.assertEqual(history[1]['paired_regression_case_ids'], ['3'])
        self.assertEqual(history[1]['candidate_evidence_cases'], 20)
        self.assertFalse(any((self.workspace / 'history_evidence').glob('episode_*')))
        current = p.tool_call(self.workspace, 'read_case', {'case_id': '3', 'call_index': 0})
        prior = p.tool_call(self.workspace, 'read_case', {'case_id': '3', 'history_index': 1,
            'call_index': 0, 'start_message': 1, 'message_count': 1, 'include_setup': True})
        self.assertEqual((current['success'], prior['success']), (1, 0))
        self.assertEqual(current['call']['assistant']['content'], 'unchanged')
        self.assertEqual(prior['call']['assistant']['content'], 'historical candidate behavior')
        self.assertEqual(prior['origin'], {'kind': 'historical_candidate', 'history_index': 1,
            'source_sha256': p.digest(after / 'episode_3.json')})
        self.assertNotIn('PRIVATE_SENTINEL', json.dumps(prior))
        request = json.loads((self.workspace / p.READ_LOG).read_text())['requests'][-1]
        self.assertEqual(request['history_index'], 1)
        self.assertEqual(request['path'], history[1]['candidate_evidence'])
        with self.assertRaisesRegex(ValueError, 'no archived public candidate TRAIN trace'):
            p.tool_call(self.workspace, 'read_case', {'case_id': '3', 'history_index': 0})
        (after / 'episode_3.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'historical TRAIN episode changed'):
            p.tool_call(self.workspace, 'read_case', {'case_id': '3', 'history_index': 1})

    def test_paired_history_views_work_after_the_current_train_batch_rotates(self):
        self.archived_candidate()
        rotated = self.root / 'rotated'
        rotated.mkdir()
        summary = json.loads(json.dumps(self.summary))
        summary['per_task'] = {str(int(key) + 100): value for key, value in summary['per_task'].items()}
        p.write_json(rotated / 'summary.json', summary)
        for path in self.eval.glob('episode_*.json'):
            row = json.loads(path.read_text());row['index'] += 100
            p.write_json(rotated / f"episode_{row['index']}.json", row)
        p.prepare(self.seed, rotated, self.workspace)
        with self.assertRaises(ValueError):
            p.tool_call(self.workspace, 'read_case', {'case_id': '3'})
        args = {'case_id': '3', 'history_index': 1, 'call_index': 0}
        candidate = p.tool_call(self.workspace, 'read_case', args)
        incumbent = p.tool_call(self.workspace, 'read_case', {**args, 'history_variant': 'incumbent'})
        self.assertEqual((candidate['success'], incumbent['success']), (0, 1))
        self.assertEqual(incumbent['origin']['kind'], 'historical_incumbent')
        self.assertNotIn('PRIVATE_SENTINEL', json.dumps(incumbent))
        feedback = p.startup_feedback(self.workspace)
        self.assertEqual(feedback['latest_primary_attempt_index'], 1)
        self.assertEqual(feedback['latest_measured_TRAIN_attempt']['paired_regression_case_ids'], ['3'])
        self.assertEqual(feedback['current_TRAIN']['n'], 20)
        self.assertNotIn('PRIVATE_SENTINEL', json.dumps(feedback))

    def test_copied_ancestor_rows_gain_original_code_and_paired_traces_without_duplication(self):
        self.archived_candidate()
        source = self.seed / 'proposals/round_03/artifacts/old.py'
        source.parent.mkdir(parents=True);source.write_text(fixtures.IDENTITY)
        self.state['rounds'][-1]['operations'] = [{'operation_id': 'same_op', 'mechanism_id': 'past_mechanism',
            'op': 'add', 'target_layer': 'h2', 'hypothesis': 'synthetic', 'artifact': 'artifacts/old.py',
            'artifact_sha256': p.digest(source)}]
        p.write_json(self.seed / 'state.json', self.state)
        copied = self.root / 'copied'
        (copied / 'versions').mkdir(parents=True)
        p.shutil.copytree(self.seed / 'versions', copied / 'versions', dirs_exist_ok=True)
        p.write_json(copied / 'state.json', self.state)
        p.prepare(copied, self.eval, self.workspace, additional_history=[self.seed])
        history = json.loads((self.workspace / 'history.json').read_text())
        self.assertEqual(len(history), 2)
        for key in ('candidate_evidence', 'incumbent_evidence'):
            self.assertIn(key, history[1]);self.assertNotIn(key + '_unavailable', history[1])
        operation = history[1]['operations'][0]
        self.assertEqual(p.digest(self.workspace / operation['historical_code_path']), p.digest(source))
        self.assertNotIn('historical_code_unavailable', operation)
        self.assertEqual(p.startup_feedback(self.workspace)['latest_primary_attempt_index'], 1)
        self.assertEqual(p.tool_call(self.workspace, 'read_case',
            {'case_id': '3', 'history_index': 1, 'history_variant': 'incumbent'})['success'], 1)

    def test_startup_uses_primary_latest_attempt_not_the_last_appended_history_source(self):
        self.archived_candidate()
        extra = self.root / 'older_branch/telecom'
        old = json.loads(json.dumps(self.state))
        old['rounds'][-1]['bundle_hypothesis'] = 'different old attempt'
        p.write_json(extra / 'state.json', old)
        p.prepare(self.seed, self.eval, self.workspace, additional_history=[extra])
        self.assertEqual(p.startup_feedback(self.workspace)['latest_primary_attempt_index'], 1)
        history = json.loads((self.workspace / 'history.json').read_text())
        self.assertEqual(len(history), 3)
        self.assertEqual(history[-1]['history_source'], 'older_branch')

    def test_startup_contrasts_are_paired_public_observations_without_read_ledger_side_effects(self):
        self.stateful_evidence(); self.archived_candidate(); self.prepare()
        feedback = p.startup_feedback(self.workspace)
        contrast = feedback['latest_regression_contrasts'][0]
        self.assertEqual((contrast['case_id'], contrast['history_index']), ('3', 1))
        self.assertEqual((contrast['incumbent']['success'], contrast['candidate']['success']), (1, 0))
        self.assertIn('historical candidate behavior', json.dumps(contrast['candidate']))
        for variant in ('incumbent', 'candidate'):
            self.assertLessEqual(len(contrast[variant]['tail_excerpts']), 4)
            for message in contrast[variant]['tail_excerpts']:
                self.assertLessEqual(len(message['content_excerpt']), 500)
        for secret in ('PRIVATE_SENTINEL', 'BUSINESS_ONLY_SENTINEL'):
            self.assertNotIn(secret, json.dumps(feedback))
        self.assertFalse((self.workspace / p.READ_LOG).exists())
        self.assertIn('full reads remain unlimited', feedback['note'])

    def test_startup_contrasts_report_unavailable_history_without_current_or_test_fallback(self):
        self.archived_candidate(); self.prepare()
        history = json.loads((self.workspace / 'history.json').read_text())
        path = self.workspace / history[1]['candidate_evidence']
        evidence = json.loads(path.read_text()); evidence['split'] = 'test'
        p.write_json(path, evidence)
        contrast = p.startup_feedback(self.workspace)['latest_regression_contrasts'][0]
        self.assertIn('unavailable', contrast['candidate'])
        self.assertNotIn('tail_excerpts', contrast['candidate'])
        self.assertEqual(contrast['incumbent']['success'], 1)
        self.assertFalse((self.workspace / p.READ_LOG).exists())

    def test_qualified_feedback_requests_real_evaluation_without_a_hard_cap(self):
        good = p.candidate_feedback({'passed': True, 'cases': [{'case_id': 'x', 'passed': True}]})
        self.assertIn('not a measured success score', good['next_step'])
        self.assertIn('no interaction-count cap', good['next_step'])
        self.assertIn('token budget', good['next_step'])
        self.assertNotIn('next_step', p.candidate_feedback({'passed': False}))

    def test_historical_business_state_is_only_returned_on_explicit_boundary_read(self):
        self.stateful_evidence()
        after = self.archived_candidate()
        self.prepare()
        args = {'case_id': '3', 'history_index': 1, 'call_index': 0}
        self.assertNotIn('BUSINESS_ONLY_SENTINEL', json.dumps(p.tool_call(self.workspace, 'read_case', args)))
        value = p.tool_call(self.workspace, 'read_case', {**args, 'environment_state': 'action',
            'state_path': ['db', 'bills', 'B1', 'paid']})
        self.assertIs(value['environment_state'], True)
        request = json.loads((self.workspace / p.READ_LOG).read_text())['requests'][-1]
        self.assertIsInstance(request['path'], str)
        self.assertEqual(request['state_path'], ['db', 'bills', 'B1', 'paid'])
        summary = json.loads((after / 'summary.json').read_text())
        summary['split'] = 'test'
        p.write_json(after / 'summary.json', summary)
        with self.assertRaisesRegex(ValueError, 'historical TRAIN summary changed'):
            p.tool_call(self.workspace, 'read_case', args)

    def test_test_or_incomplete_historical_evidence_is_never_substituted(self):
        after = self.archived_candidate()
        summary = json.loads((after / 'summary.json').read_text())
        summary['split'] = 'test'
        p.write_json(after / 'summary.json', summary)
        self.prepare()
        with self.assertRaisesRegex(ValueError, 'no archived public candidate TRAIN trace'):
            p.tool_call(self.workspace, 'read_case', {'case_id': '3', 'history_index': 1})
        summary['split'] = 'train'
        p.write_json(after / 'summary.json', summary)
        (after / 'episode_3.json').unlink()
        other = self.root / 'missing_episode'
        p.prepare(self.seed, self.eval, other)
        self.assertIn('candidate_evidence_unavailable', json.loads((other / 'history.json').read_text())[1])
        with self.assertRaisesRegex(ValueError, 'no archived public candidate TRAIN trace'):
            p.tool_call(other, 'read_case', {'case_id': '3', 'history_index': 1})

    def test_read_case_has_no_ten_case_or_aggregate_evidence_quota(self):
        self.prepare()
        for index in range(20):
            p.tool_call(self.workspace, 'read_case', {'case_id': str(index), 'message_count': 1000000})
        ledger = json.loads((self.workspace / p.READ_LOG).read_text())
        self.assertEqual(len(ledger['case_ids']), 20)
        self.assertEqual(len(ledger['requests']), 20)

    def test_read_case_refuses_unknown_private_and_invalid_selections(self):
        self.prepare()
        for args in ({'case_id': '../secret'}, {'case_id': '999'},
                     {'case_id': '2', 'call_index': True}, {'case_id': '2', 'call_index': 1},
                     {'case_id': '2', 'start_message': -1}, {'case_id': '2', 'message_count': 0},
                     {'case_id': '2', 'call_index': 0, 'start_message': 3},
                     {'case_id': '2', 'call_index': 0, 'message_count': True},
                     {'case_id': '2', 'include_setup': 1},
                     {'case_id': '2', 'history_index': True}, {'case_id': '2', 'history_index': 10},
                     {'case_id': '2', 'history_variant': 'incumbent'},
                     {'case_id': '2', 'history_index': 0, 'history_variant': ['candidate']},
                     {'case_id': '2', 'history_index': 0, 'history_variant': 'test'},
                     {'case_id': '2', 'path': 'env.sh'}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                p.tool_call(self.workspace, 'read_case', args)
        self.assertFalse((self.workspace / p.READ_LOG).exists())

    def test_read_case_verifies_frozen_case_before_access(self):
        self.prepare()
        index = json.loads((self.workspace / 'evidence.json').read_text())
        (self.workspace / index[0]['path']).write_text('{}')
        with self.assertRaisesRegex(ValueError, 'frozen input changed'):
            p.tool_call(self.workspace, 'read_case', {'case_id': index[0]['case_id']})

    def test_prompt_prioritizes_mechanisms_without_hand_filling_policies(self):
        self.prepare()
        prompt = (self.workspace / 'OPTIMIZER.md').read_text()
        for phrase in ('matched successful controls', 'Replace or delete', 'Pivot the mechanism',
                       'not four content quotas', 'SCHEMAS.md', 'optional notes', 'SAME complete TRAIN'):
            self.assertIn(phrase, prompt)
        self.assertNotIn('"schema_version":1', prompt)
        self.assertLess(len(prompt), 5000)

    def response(self, calls=None, truncated=False):
        message = {'role': 'assistant', 'content': ''}
        if calls is not None:
            message['tool_calls'] = calls
        return {'choices': [{'message': message, 'finish_reason': 'length' if truncated else 'stop'}],
                'usage': {'prompt_tokens': 80, 'completion_tokens': 20, 'total_tokens': 100}}

    def check_calls(self, count=1):
        return [{'id': f'check_{i}', 'type': 'function', 'function': {
            'name': 'check_candidate', 'arguments': '{}'}} for i in range(count)]

    def qualified_calls(self):
        files = self.files()
        calls = [{'id': name, 'type': 'function', 'function': {'name': 'write_file',
            'arguments': json.dumps({'path': name, 'content': files[name]})}}
            for name in ('artifacts/new.py', 'bundle.json')]
        return calls + self.check_calls()

    def test_optional_malformed_notes_are_retained_without_repairs_or_extra_api(self):
        self.prepare()
        calls = self.qualified_calls()
        notes = {'diagnosis.json': 'not required JSON\n',
                 'checks.json': '{"tests":[{"expected_state":{},"kind":"unclassified"}]}\n'}
        calls = [{'id': name, 'type': 'function', 'function': {'name': 'write_file',
                  'arguments': json.dumps({'path': name, 'content': content})}}
                 for name, content in notes.items()] + calls
        completion = Mock(side_effect=[self.response(calls), self.response(truncated=True)])
        output = self.root / 'optional_notes'
        p.propose(self.workspace, output, completion=completion, checkpoint_dir=self.root / 'checkpoint')
        receipt = self.validation(output)
        self.assertEqual(receipt['repairs_granted'], 0)
        self.assertEqual(receipt['fallback_reason'], 'output_truncated_after_valid_candidate')
        for name, text in notes.items():
            self.assertEqual((output / name).read_text(), text)
        cached = Mock(side_effect=AssertionError('cache must not make a paid call'))
        p.propose(self.workspace, output, completion=cached)
        cached.assert_not_called()

    def read_response(self, path='summary.json'):
        return self.response([{'id': 'read', 'type': 'function', 'function': {
            'name': 'read_file', 'arguments': json.dumps({'path': path})}}])

    def validation(self, output):
        return json.loads((output / 'validation.json').read_text())

    def test_proposer_budget_and_frozen_input_tampering(self):
        # Replace the historical suite's total-turn-limit assertion.
        self.prepare()
        completion = Mock(return_value=self.response())
        with self.assertRaises(p.ProposalFailure) as caught:
            p.propose(self.workspace, self.root / 'bad', completion=completion)
        self.assertEqual(caught.exception.code, 'repair_limit')
        self.assertEqual(completion.call_count, 4)  # Initial failure + three repairs.
        (self.workspace / 'summary.json').write_text('{}')
        with self.assertRaises(ValueError):
            p.verify_inputs(self.workspace)

    def test_optimizer_prompt_does_not_inherit_qwen_token_settings(self):
        self.prepare()
        prompt = (self.workspace / 'OPTIMIZER.md').read_text()
        self.assertNotIn('32768', prompt)
        self.assertNotIn('4096', prompt)
        self.assertNotIn('32 API requests', prompt)
        self.assertIn('no total API-request or interaction-count cap', prompt)
        self.assertIn('at most three repair opportunities', prompt)
        budget = json.loads((self.workspace / 'receipt.json').read_text())['budget']
        self.assertEqual(budget, {'max_proposer_turns': None, 'max_repairs': 3,
                                  'max_total_tokens': 10_000_000})

    def test_default_transport_rejects_missing_or_invalid_usage(self):
        for usage in (None, {}, {'prompt_tokens': -1, 'completion_tokens': 1}):
            response = {'choices': [{'message': {'role': 'assistant', 'content': 'synthetic'}}]}
            if usage is not None:
                response['usage'] = usage
            opener = Mock()
            opener.open.return_value = fixtures.StringIO(json.dumps(response))
            with self.subTest(usage=usage), \
                    patch.dict(p.os.environ, {'MR_API_KEY': 'synthetic-test-key', 'MR_HOST': 'https://router.invalid'}), \
                    patch.object(p.urllib.request, 'build_opener', return_value=opener), \
                    self.assertRaisesRegex(RuntimeError, 'API token usage missing or invalid'):
                p.model_router_completion([], p.TOOLS)

    def test_total_budget_stops_before_any_paid_request(self):
        self.prepare()
        completion = Mock(side_effect=AssertionError('no paid request fits'))
        checkpoint = self.root / 'budget_checkpoint'
        with self.assertRaises(p.ProposalFailure) as caught:
            p.propose(self.workspace, self.root / 'budget_empty', completion=completion,
                      checkpoint_dir=checkpoint, token_budget=100_000)
        self.assertEqual(caught.exception.code, 'token_budget')
        completion.assert_not_called()
        saved = json.loads((checkpoint / 'agent_checkpoint.json').read_text())
        self.assertEqual((saved['requests'], saved['inflight'], saved['max_total_tokens']),
                         (0, False, 100_000))
        self.assertFalse((self.root / 'budget_empty').exists())
        with self.assertRaises(p.ProposalFailure) as resumed:
            p.propose(self.workspace, self.root / 'budget_empty', completion=completion,
                      checkpoint_dir=checkpoint, token_budget=100_000)
        self.assertEqual(resumed.exception.code, 'token_budget')
        with self.assertRaisesRegex(ValueError, 'total token budget'):
            p.propose(self.workspace, self.root / 'budget_empty', completion=completion,
                      checkpoint_dir=checkpoint, token_budget=200_000)
        completion.assert_not_called()

    def test_total_budget_restores_qualified_snapshot_after_invalid_edits(self):
        self.prepare()
        invalid = [{'id': 'invalid', 'type': 'function', 'function': {'name': 'write_file',
                   'arguments': json.dumps({'path': 'bundle.json', 'content': '{}'})}}]
        completion = Mock(side_effect=[self.response(self.qualified_calls()), self.response(invalid)])
        output = self.root / 'budget_qualified'
        with patch.object(p, 'request_token_reserve', side_effect=[100, 100, 101]):
            p.propose(self.workspace, output, completion=completion,
                      checkpoint_dir=self.root / 'budget_checkpoint', token_budget=300)
        result = self.validation(output)
        self.assertEqual(completion.call_count, 2)
        self.assertEqual(result['fallback_reason'], 'token_budget_after_valid_candidate')
        self.assertEqual(result['usage']['prompt_tokens'] + result['usage']['completion_tokens'], 200)
        self.assertEqual(result['max_total_tokens'], 300)
        self.assertTrue(p.check_candidate(self.workspace, output)['passed'])
        cached = Mock(side_effect=AssertionError('qualified cache must not pay again'))
        p.propose(self.workspace, output, completion=cached, token_budget=300)
        cached.assert_not_called()
        with self.assertRaises(ValueError):
            p.propose(self.workspace, output, completion=cached, token_budget=400)

    def test_repairs_share_total_budget_without_request_count_quota(self):
        self.prepare()
        completion = Mock(side_effect=[self.response(self.check_calls()), self.read_response()])
        checkpoint = self.root / 'repair_budget_checkpoint'
        with patch.object(p, 'request_token_reserve', side_effect=[100, 100, 101]), \
                self.assertRaises(p.ProposalFailure) as caught:
            p.propose(self.workspace, self.root / 'repair_budget', completion=completion,
                      checkpoint_dir=checkpoint, token_budget=300)
        self.assertEqual(caught.exception.code, 'token_budget')
        saved = json.loads((checkpoint / 'agent_checkpoint.json').read_text())
        self.assertEqual((saved['requests'], saved['repairs_granted'], saved['failed_preflights']), (2, 1, 1))
        self.assertEqual(saved['usage']['prompt_tokens'] + saved['usage']['completion_tokens'], 200)

    def test_total_budget_validates_numeric_ceiling_before_requests(self):
        self.prepare()
        completion = Mock(side_effect=AssertionError('invalid budget must not pay'))
        for value in (True, False, 0, -1, 1.5, '1000000', 10_000_001):
            with self.subTest(budget=value), self.assertRaisesRegex(ValueError, 'token_budget'):
                p.propose(self.workspace, self.root / 'invalid_budget', completion=completion,
                          token_budget=value)
        completion.assert_not_called()

    def test_token_reservation_includes_tools_reasoning_and_full_output(self):
        messages = [{'role': 'user', 'content': 'mixed Unicode context 🚀'},
                    {'role': 'assistant', 'content': '', 'reasoning_content': 'preserved reasoning'}]
        reservation = p.request_token_reserve(messages, p.TOOLS)
        self.assertGreaterEqual(reservation, p.flash_counter().count(messages, p.TOOLS) + 100_000)
        self.assertNotIn('tools', messages[0])
        self.assertEqual(messages[0]['role'], 'user')

    def test_custom_transport_cannot_hide_actual_usage(self):
        self.prepare()
        response = self.response()
        response['usage'] = {'total_tokens': 100}
        completion = Mock(return_value=response)
        with self.assertRaisesRegex(RuntimeError, 'API token usage missing or invalid'):
            p.propose(self.workspace, self.root / 'unknown_usage', completion=completion)
        self.assertEqual(completion.call_count, 1)

    def test_context_archive_is_exact_readable_and_not_model_writable(self):
        self.prepare()
        old = [{'role': 'assistant', 'content': 'full public context\ncomplete original text 🚀',
                'reasoning_content': 'original reasoning', 'tool_calls': self.check_calls()},
               {'role': 'tool', 'tool_call_id': 'check_0', 'content': '{"passed":true}'}]
        ref = p.archive_context(self.workspace, old)
        self.assertEqual(json.loads((self.workspace / ref).read_text()), old)
        result = p.tool_call(self.workspace, 'read_file', {'path': ref, 'max_bytes': 16000})
        self.assertEqual(json.loads(result['content']), old)
        self.assertTrue(p.tool_call(self.workspace, 'search_file', {'path': ref, 'text': 'original reasoning'})['matches'])
        with self.assertRaises(ValueError):
            p.tool_call(self.workspace, 'write_file', {'path': ref, 'content': '{}'})
        (self.workspace / ref).write_text('{}')
        with self.assertRaisesRegex(ValueError, 'archive hash mismatch'):
            p.tool_call(self.workspace, 'read_file', {'path': ref})

    def test_same_agent_checkpoint_preserves_candidate_usage_repairs_and_error_feedback(self):
        self.prepare()
        memory = 'Hypothesis and risk from evidence/case_2.json call 0; candidate pending; inspect successful control.'
        step = 0

        def completion(messages, tools):
            nonlocal step
            step += 1
            if step == 1:
                return self.response(self.check_calls())
            if step == 2:
                self.assertEqual([t['function']['name'] for t in tools], ['checkpoint_context'])
                return self.response([{'id': 'checkpoint', 'type': 'function', 'function': {
                    'name': 'checkpoint_context', 'arguments': json.dumps({'working_memory': memory})}}])
            if step == 3:
                self.assertEqual(len(messages), 3)
                self.assertIn(memory, messages[-1]['content'])
                self.assertIn('FileNotFoundError', messages[-1]['content'])
                self.assertIn('"repairs_granted": 1', messages[-1]['content'])
                self.assertIn('context_history/', messages[-1]['content'])
                self.assertIn('input + output token budget', messages[1]['content'])
                return self.response(self.qualified_calls())
            return self.response()

        output, checkpoint = self.root / 'memory_output', self.root / 'memory_checkpoint'
        with patch.object(p, 'needs_context_checkpoint', side_effect=[False, True, False, False]):
            p.propose(self.workspace, output, completion=completion, checkpoint_dir=checkpoint)
        receipt = self.validation(output)
        self.assertEqual((receipt['proposer_requests'], receipt['usage']['total_tokens']), (4, 400))
        self.assertEqual((receipt['repairs_granted'], receipt['failed_preflights']), (1, 1))
        self.assertTrue(p.check_candidate(self.workspace, output)['passed'])
        saved = json.loads((checkpoint / 'agent_checkpoint.json').read_text())
        row = saved['context_checkpoints'][0]
        self.assertEqual((row['memory'], row['request'], row['usage_so_far']['total_tokens']), (memory, 2, 200))
        self.assertEqual(row['usage'], {'prompt_tokens': 80, 'completion_tokens': 20, 'total_tokens': 100})
        archive = json.loads((checkpoint / 'workspace' / row['archive']).read_text())
        self.assertTrue(any(m.get('tool_call_id') == 'checkpoint' for m in archive))
        self.assertTrue(any(m.get('tool_call_id') == 'check_0' for m in archive))

    def test_checkpoint_plain_text_is_allowed_and_empty_summary_defers_without_discarding_context(self):
        self.prepare()
        self.assertEqual(p.checkpoint_memory({'choices': [{'message': {'content': 'source references and pending work'}}]}),
                         'source references and pending work')
        for message in ({'content': ''}, {'tool_calls': self.qualified_calls()}):
            with self.assertRaises(ValueError):
                p.checkpoint_memory({'choices': [{'message': message}]})
        checkpoint = self.root / 'bad_memory_checkpoint'
        step = 0
        def completion(messages, tools):
            nonlocal step
            step += 1
            if step == 1:
                return self.response()
            self.assertIn('Memory refresh was deferred', messages[-1]['content']) if step == 2 else None
            return self.response(self.qualified_calls()) if step == 2 else self.response()
        with patch.object(p, 'needs_context_checkpoint', side_effect=[True, True, False]):
            p.propose(self.workspace, self.root / 'bad_memory', completion=completion, checkpoint_dir=checkpoint)
        saved = json.loads((checkpoint / 'agent_checkpoint.json').read_text())
        self.assertEqual((saved['requests'], saved['usage']['total_tokens'], saved['inflight']), (3, 300, False))
        self.assertGreater(len(saved['messages']), 2)
        self.assertEqual(len(saved['context_checkpoint_failures']), 1)
        self.assertEqual(saved['repairs_granted'], 0)
        self.assertEqual(step, 3)

    def test_bad_context_checkpoint_preserves_qualified_candidate_and_continues(self):
        self.prepare()
        completion = Mock(side_effect=[self.response(self.qualified_calls()), self.response(), self.response()])
        output = self.root / 'bad_memory_qualified'
        with patch.object(p, 'needs_context_checkpoint', side_effect=[False, True, True]):
            p.propose(self.workspace, output, completion=completion, checkpoint_dir=self.root / 'memory_checkpoint')
        receipt = self.validation(output)
        self.assertIsNone(receipt['fallback_reason'])
        self.assertEqual((receipt['proposer_requests'], receipt['usage']['total_tokens']), (3, 300))
        self.assertEqual(len(receipt['context_checkpoint_failures']), 1)
        self.assertTrue(p.check_candidate(self.workspace, output)['passed'])

    def test_unadvertised_memory_action_is_not_executed_and_normal_turn_can_follow(self):
        self.prepare()
        calls = self.qualified_calls()
        step = 0
        def completion(messages, tools):
            nonlocal step
            step += 1
            if step == 1:
                return self.response(calls)
            if step == 2:
                self.assertFalse((self.root / 'deferred_checkpoint/workspace/bundle.json').exists())
                self.assertIn('copy_file', [t['function']['name'] for t in tools])
                return self.response(calls)
            return self.response()
        output = self.root / 'deferred_action'
        with patch.object(p, 'needs_context_checkpoint', side_effect=[True, True, False]):
            p.propose(self.workspace, output, completion=completion,
                      checkpoint_dir=self.root / 'deferred_checkpoint')
        receipt = self.validation(output)
        self.assertEqual((receipt['proposer_requests'], receipt['usage']['total_tokens']), (3, 300))
        self.assertEqual((receipt['repairs_granted'], receipt['failed_preflights']), (0, 0))
        self.assertEqual(receipt['context_checkpoint_failures'][0]['usage']['total_tokens'], 100)

    def test_copy_frozen_python_source_then_edit_without_reprinting(self):
        self.prepare()
        source = next((self.workspace / 'active').glob('*.py'))
        target = 'artifacts/copied.py'
        result = p.tool_call(self.workspace, 'copy_file',
                             {'source': str(source.relative_to(self.workspace)), 'destination': target})
        self.assertEqual(source.read_bytes(), (self.workspace / target).read_bytes())
        self.assertEqual(result['sha256'], p.digest(source))
        p.tool_call(self.workspace, 'replace_file', {'path': target, 'old_text': source.read_text(),
                                                    'new_text': source.read_text() + '\n# local candidate edit\n'})
        self.assertNotEqual(source.read_bytes(), (self.workspace / target).read_bytes())
        for origin, destination in [('summary.json', target),
                                    (str(source.relative_to(self.workspace)), 'active/copied.py'),
                                    (target, 'artifacts/second.py')]:
            with self.subTest(source=origin, destination=destination), self.assertRaises(ValueError):
                p.tool_call(self.workspace, 'copy_file', {'source': origin, 'destination': destination})

    def test_refresh_threshold_is_not_a_context_or_interaction_rejection(self):
        with patch.object(p, 'flash_counter') as counter:
            counter.return_value.count.return_value = p.CONTEXT_CHECKPOINT_TOKENS
            self.assertTrue(p.needs_context_checkpoint([]))
            counter.return_value.count.return_value = p.CONTEXT_CHECKPOINT_TOKENS - 1
            self.assertFalse(p.needs_context_checkpoint([]))

    def test_resume_after_context_checkpoint_keeps_paid_usage_and_repairs(self):
        self.prepare()
        checkpoint, output = self.root / 'memory_resume_checkpoint', self.root / 'memory_resume'
        original = p.save_agent

        def pause(path, sandbox, **values):
            original(path, sandbox, **values)
            if not values['inflight'] and values['requests'] == 2 and values['context_checkpoints']:
                raise RuntimeError('synthetic safe pause after memory checkpoint')

        summary = self.response()
        summary['choices'][0]['message']['content'] = 'Retain failed bundle.json validation; one repair granted; next write valid code.'
        paid = Mock(side_effect=[self.response(self.check_calls()), summary])
        with patch.object(p, 'needs_context_checkpoint', side_effect=[False, True]), \
                patch.object(p, 'save_agent', side_effect=pause), \
                self.assertRaisesRegex(RuntimeError, 'safe pause'):
            p.propose(self.workspace, output, completion=paid, checkpoint_dir=checkpoint)
        resumed = Mock(side_effect=[self.response(self.qualified_calls()), self.response()])
        with patch.object(p, 'needs_context_checkpoint', side_effect=[True, False]):
            p.propose(self.workspace, output, completion=resumed, checkpoint_dir=checkpoint)
        receipt = self.validation(output)
        self.assertEqual((paid.call_count, resumed.call_count), (2, 2))
        self.assertEqual((receipt['proposer_requests'], receipt['usage']['total_tokens']), (4, 400))
        self.assertEqual((receipt['repairs_granted'], receipt['failed_preflights']), (1, 1))
        self.assertEqual(len(receipt['context_checkpoints']), 1)

    def test_context_checkpoint_expense_is_inside_total_budget(self):
        self.prepare()
        summary = self.response()
        summary['choices'][0]['message']['content'] = 'Read summary.json; now implement the observed fix and check all TRAIN cases.'
        completion = Mock(side_effect=[self.read_response(), summary, self.response(self.qualified_calls())])
        output = self.root / 'memory_budget_output'
        with patch.object(p, 'needs_context_checkpoint', side_effect=[False, True, False, False]), \
                patch.object(p, 'request_token_reserve', side_effect=[100, 100, 100, 101]):
            p.propose(self.workspace, output, completion=completion,
                      checkpoint_dir=self.root / 'memory_budget_checkpoint', token_budget=400)
        receipt = self.validation(output)
        self.assertEqual((receipt['proposer_requests'], receipt['usage']['total_tokens']), (3, 300))
        self.assertEqual(receipt['fallback_reason'], 'token_budget_after_valid_candidate')
        self.assertEqual(receipt['context_checkpoints'][0]['usage_so_far']['total_tokens'], 200)
        self.assertTrue(p.check_candidate(self.workspace, output)['passed'])

    def test_resumed_validation_feedback_survives_repeated_memory_checkpoints(self):
        self.prepare()
        checkpoint, output = self.root / 'validation_resume_checkpoint', self.root / 'validation_resume'
        original = p.save_agent

        def pause(path, sandbox, **values):
            original(path, sandbox, **values)
            if not values['inflight'] and values['requests'] == 1:
                raise RuntimeError('synthetic safe pause after failed preflight')

        with patch.object(p, 'needs_context_checkpoint', return_value=False), \
                patch.object(p, 'save_agent', side_effect=pause), \
                self.assertRaisesRegex(RuntimeError, 'safe pause'):
            p.propose(self.workspace, output, completion=Mock(return_value=self.response(self.check_calls())),
                      checkpoint_dir=checkpoint)
        saved = json.loads((checkpoint / 'agent_checkpoint.json').read_text())
        self.assertFalse(saved['validation']['passed'])
        expected = p.candidate_feedback(saved['validation'])
        self.assertIn('FileNotFoundError', json.dumps(expected))
        step = 0

        def completion(messages, tools):
            nonlocal step
            step += 1
            if step in (1, 3):
                self.assertEqual([t['function']['name'] for t in tools], ['checkpoint_context'])
                response = self.response()
                response['choices'][0]['message']['content'] = 'Continue the pending proposal; consult original evidence.'
                return response
            if step in (2, 4):
                payload = messages[-1]['content'].split('Verified host bookkeeping (not a diagnosis): ', 1)[1]
                state = json.loads(payload.split('\nAll source files', 1)[0])
                self.assertEqual(state['last_validation'], expected)
                self.assertEqual((state['repairs_granted'], state['failed_preflights']), (1, 1))
                return self.read_response() if step == 2 else self.response(self.qualified_calls())
            return self.response()

        with patch.object(p, 'needs_context_checkpoint', side_effect=[True, False, True, False, False]):
            p.propose(self.workspace, output, completion=completion, checkpoint_dir=checkpoint)
        receipt = self.validation(output)
        self.assertEqual((receipt['proposer_requests'], receipt['usage']['total_tokens']), (6, 600))
        self.assertEqual(len(receipt['context_checkpoints']), 2)
        self.assertEqual((receipt['repairs_granted'], receipt['failed_preflights']), (1, 1))
        self.assertTrue(p.check_candidate(self.workspace, output)['passed'])

    def test_initial_analysis_can_exceed_32_requests(self):
        self.prepare()
        completion = Mock(side_effect=[self.read_response() for _ in range(40)] +
                          [self.response(self.qualified_calls()), self.response()])
        output = self.root / 'long_analysis'
        p.propose(self.workspace, output, completion=completion)
        receipt = self.validation(output)
        self.assertEqual(completion.call_count, 42)
        self.assertEqual(receipt['proposer_requests'], 42)
        self.assertIsNone(receipt['max_proposer_turns'])
        self.assertEqual(receipt['repairs_granted'], 0)
        self.assertEqual(receipt['usage']['total_tokens'], 4200)

    def test_one_repair_can_exceed_32_requests(self):
        self.prepare()
        completion = Mock(side_effect=[self.response(self.check_calls())] +
                          [self.read_response() for _ in range(40)] +
                          [self.response(self.qualified_calls()), self.response()])
        output = self.root / 'long_repair'
        p.propose(self.workspace, output, completion=completion)
        receipt = self.validation(output)
        self.assertEqual(completion.call_count, 43)
        self.assertEqual((receipt['repairs_granted'], receipt['failed_preflights']), (1, 1))

    def test_fourth_failed_preflight_stops_without_fourth_repair(self):
        self.prepare()
        checkpoint, output = self.root / 'checkpoint', self.root / 'failed'
        completion = Mock(return_value=self.response(self.check_calls()))
        with self.assertRaises(p.ProposalFailure) as caught:
            p.propose(self.workspace, output, completion=completion, checkpoint_dir=checkpoint)
        self.assertEqual(caught.exception.code, 'repair_limit')
        self.assertEqual(completion.call_count, 4)
        self.assertFalse(output.exists())
        saved = json.loads((checkpoint / 'agent_checkpoint.json').read_text())
        self.assertEqual((saved['repairs_granted'], saved['failed_preflights']), (3, 4))
        self.assertEqual(saved['usage']['total_tokens'], 400)
        self.assertFalse(saved['inflight'])
        retry = Mock(side_effect=AssertionError('must not reset exhausted repairs'))
        with self.assertRaises(p.ProposalFailure):
            p.propose(self.workspace, output, completion=retry, checkpoint_dir=checkpoint)
        retry.assert_not_called()

    def test_third_repair_can_succeed(self):
        self.prepare()
        completion = Mock(side_effect=[self.response(self.check_calls()) for _ in range(3)] +
                          [self.response(self.qualified_calls()), self.response()])
        output = self.root / 'third_repair'
        p.propose(self.workspace, output, completion=completion)
        receipt = self.validation(output)
        self.assertEqual((receipt['repairs_granted'], receipt['failed_preflights']), (3, 3))
        self.assertTrue(p.check_candidate(self.workspace, output)['passed'])

    def test_multiple_failed_checks_in_one_response_are_one_repair_cycle(self):
        self.prepare()
        completion = Mock(side_effect=[self.response(self.check_calls(count=5)),
                          self.response(self.qualified_calls()), self.response()])
        output = self.root / 'batch_checks'
        p.propose(self.workspace, output, completion=completion)
        self.assertEqual(self.validation(output)['repairs_granted'], 1)

    def test_read_error_does_not_consume_repair(self):
        self.prepare()
        completion = Mock(side_effect=[self.read_response('missing.json'),
                          self.response(self.qualified_calls()), self.response()])
        output = self.root / 'read_error'
        p.propose(self.workspace, output, completion=completion)
        self.assertEqual(self.validation(output)['repairs_granted'], 0)

    def test_repair_counter_survives_safe_checkpoint_resume(self):
        self.prepare()
        checkpoint, output = self.root / 'checkpoint', self.root / 'resumed'
        original = p.save_agent

        def pause(path, sandbox, **values):
            original(path, sandbox, **values)
            if not values['inflight'] and values.get('repairs_granted') == 1:
                raise RuntimeError('safe repair pause')

        with patch.object(p, 'save_agent', side_effect=pause):
            with self.assertRaisesRegex(RuntimeError, 'safe repair pause'):
                p.propose(self.workspace, output, completion=Mock(return_value=self.response(self.check_calls())),
                          checkpoint_dir=checkpoint)
        completion = Mock(return_value=self.response(self.check_calls()))
        with self.assertRaises(p.ProposalFailure):
            p.propose(self.workspace, output, completion=completion, checkpoint_dir=checkpoint)
        self.assertEqual(completion.call_count, 3)
        saved = json.loads((checkpoint / 'agent_checkpoint.json').read_text())
        self.assertEqual((saved['repairs_granted'], saved['requests']), (3, 4))

    def test_qualified_candidate_kept_after_invalid_edit(self):
        self.prepare()
        broken = [{'id': 'break', 'type': 'function', 'function': {'name': 'write_file',
            'arguments': json.dumps({'path': 'artifacts/new.py', 'content': fixtures.IDENTITY + '\nimport os\n'})}}]
        completion = Mock(side_effect=[self.response(self.qualified_calls()), self.response(broken), self.response()])
        output = self.root / 'kept'
        p.propose(self.workspace, output, completion=completion, checkpoint_dir=self.root / 'checkpoint')
        self.assertEqual((output / 'artifacts/new.py').read_text(), fixtures.IDENTITY)
        self.assertEqual(self.validation(output)['fallback_reason'], 'later_edits_invalid_after_valid_candidate')
        self.assertTrue(p.check_candidate(self.workspace, output)['passed'])

    def test_qualified_candidate_kept_after_output_truncation(self):
        self.prepare()
        completion = Mock(side_effect=[self.response(self.qualified_calls()), self.response(truncated=True)])
        output = self.root / 'kept'
        p.propose(self.workspace, output, completion=completion, checkpoint_dir=self.root / 'checkpoint')
        self.assertEqual(self.validation(output)['fallback_reason'], 'output_truncated_after_valid_candidate')
        self.assertEqual(self.validation(output)['usage']['total_tokens'], 200)
        self.assertTrue(p.check_candidate(self.workspace, output)['passed'])

    def test_qualified_candidate_kept_after_context_limit(self):
        self.prepare()
        completion = Mock(side_effect=[self.response(self.qualified_calls()), p.ContextLimit('provider context limit')])
        output, checkpoint = self.root / 'kept', self.root / 'checkpoint'
        p.propose(self.workspace, output, completion=completion, checkpoint_dir=checkpoint)
        receipt = self.validation(output)
        self.assertEqual(receipt['fallback_reason'], 'context_limit_after_valid_candidate')
        self.assertEqual((receipt['usage']['total_tokens'], receipt['repairs_granted']), (100, 0))
        self.assertEqual(completion.call_count, 2)
        self.assertTrue(p.check_candidate(self.workspace, output)['passed'])
        saved = json.loads((checkpoint / 'agent_checkpoint.json').read_text())
        self.assertEqual(saved['error'], 'context_limit')
        self.assertFalse(saved['inflight'])

    def test_context_limit_without_qualified_candidate_still_fails(self):
        self.prepare()
        completion = Mock(side_effect=p.ContextLimit('provider context limit'))
        output, checkpoint = self.root / 'failed', self.root / 'checkpoint'
        with self.assertRaises(p.ProposalFailure) as caught:
            p.propose(self.workspace, output, completion=completion, checkpoint_dir=checkpoint)
        self.assertEqual(caught.exception.code, 'context_limit')
        self.assertEqual(completion.call_count, 1)
        self.assertFalse(output.exists())
        saved = json.loads((checkpoint / 'agent_checkpoint.json').read_text())
        self.assertEqual(saved['repairs_granted'], 0)
        self.assertFalse(saved['inflight'])

    def test_retention_keeps_latest_qualified_candidate_not_most_operations(self):
        self.prepare()
        larger = self.files()
        bundle = json.loads(larger['bundle.json'])
        bundle['operations'].append({'operation_id': 'remove_identity', 'mechanism_id': 'identity',
            'op': 'delete', 'target_layer': 'h2', 'hypothesis': 'synthetic redundancy',
            'expected_revision': 1, 'expected_sha256': p.digest(self.source)})
        larger['bundle.json'] = json.dumps(bundle)
        diagnosis = json.loads(larger['diagnosis.json'])
        for entry in diagnosis['cases'] + diagnosis['rules']:
            entry['operation_ids'].append('remove_identity')
        larger['diagnosis.json'] = json.dumps(diagnosis)
        calls = [{'id': name, 'type': 'function', 'function': {'name': 'write_file',
            'arguments': json.dumps({'path': name, 'content': larger[name]})}}
            for name in ('diagnosis.json', 'bundle.json', 'artifacts/new.py', 'checks.json')] + self.check_calls()
        completion = Mock(side_effect=[self.response(calls),
            self.response(self.qualified_calls()), self.response(truncated=True)])
        output, checkpoint = self.root / 'kept', self.root / 'checkpoint'
        with patch.object(p, 'valid_snapshot', wraps=p.valid_snapshot) as snapshotter:
            p.propose(self.workspace, output, completion=completion, checkpoint_dir=checkpoint)
        self.assertEqual([len(call.args[0]['bundle']['operations'])
                          for call in snapshotter.call_args_list], [2, 1])
        self.assertEqual(len(json.loads((output / 'bundle.json').read_text())['operations']), 1)
        snapshot = json.loads((checkpoint / 'validated_candidate.json').read_text())
        self.assertEqual(snapshot['validated_at_request'], 2)
        self.assertTrue(p.check_candidate(self.workspace, output)['passed'])

    def test_qualified_recovery_requires_no_paid_api(self):
        self.prepare()
        self.write_candidate()
        completion = Mock(side_effect=AssertionError('already qualified'))
        output = self.root / 'recovered'
        p.propose(self.workspace, output, completion=completion, recovery_outputs=self.candidate)
        completion.assert_not_called()
        self.assertTrue(self.validation(output)['recovered_qualified'])
        self.assertEqual(self.validation(output)['repairs_granted'], 0)

    def test_repair_limit_preserves_prior_qualified_candidate(self):
        self.prepare()
        broken = [{'id': 'break', 'type': 'function', 'function': {'name': 'write_file',
            'arguments': json.dumps({'path': 'artifacts/new.py', 'content': fixtures.IDENTITY + '\nimport os\n'})}}]
        completion = Mock(side_effect=[self.response(self.qualified_calls()),
                          self.response(broken + self.check_calls())] +
                          [self.response(self.check_calls()) for _ in range(3)])
        output = self.root / 'fallback'
        p.propose(self.workspace, output, completion=completion, checkpoint_dir=self.root / 'checkpoint')
        receipt = self.validation(output)
        self.assertEqual(receipt['fallback_reason'], 'repair_limit_after_valid_candidate')
        self.assertEqual((receipt['repairs_granted'], receipt['failed_preflights']), (3, 4))
        self.assertTrue(p.check_candidate(self.workspace, output)['passed'])

    def test_paid_cli_defaults_to_persistent_public_session(self):
        output = self.root / 'candidate_output'
        with patch.object(p.sys, 'argv', ['life-evolve', 'propose', '--workspace', str(self.workspace),
                                         '--output', str(output)]), \
                patch.object(p, 'propose', return_value=output / 'bundle.json') as proposer, \
                patch('builtins.print'):
            p.main()
        self.assertEqual(proposer.call_args.kwargs['checkpoint_dir'],
                         self.root / 'candidate_output_session')

    def test_invalid_repair_policy_is_rejected_before_api(self):
        self.prepare()
        completion = Mock(side_effect=AssertionError('invalid policy must not call API'))
        for limit in (-1, 4, True, '3'):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                p.propose(self.workspace, self.root / 'invalid', completion=completion, max_repairs=limit)
        completion.assert_not_called()

    def test_zero_repairs_means_initial_attempt_only(self):
        self.prepare()
        completion = Mock(return_value=self.response(self.check_calls()))
        with self.assertRaises(p.ProposalFailure):
            p.propose(self.workspace, self.root / 'zero', completion=completion, max_repairs=0)
        self.assertEqual(completion.call_count, 1)
