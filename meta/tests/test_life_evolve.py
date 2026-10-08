"""Offline coverage: unlimited agent turns, only three preflight repair cycles."""
import json
from unittest.mock import Mock, patch

from meta.tests.test_life_repair import _revision_fixtures
from meta.life_evolve import profile as p


fixtures = _revision_fixtures('_life_evolve_fixtures', 'test_life_optimizer.py',
                              optimizer=True, profile_module='meta.life_evolve')


class EvolveTests(fixtures.OptimizerTests):
    def stateful_evidence(self):
        from meta.environment_state import StateLog
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
        self.assertIn('meta/environment_state.py',p.source_hashes())
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

    def test_schema_reference_is_copied_fingerprinted_and_immutable(self):
        self.prepare()
        self.assertEqual((self.workspace / 'SCHEMAS.md').read_bytes(),
                         p.Path(p.__file__).with_name('SCHEMAS.md').read_bytes())
        self.assertIn('meta/life_evolve/SCHEMAS.md', p.source_hashes())
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
                     {'case_id': '2', 'call_index': 0, 'start_message': 0},
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
        self.assertEqual(budget, {'max_proposer_turns': None, 'max_repairs': 3})

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
