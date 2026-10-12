"""Offline shared-contract and native Tau transport tests; no policy priors/API calls."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from evolution import harness_hooks as hooks
from evolution.tests.test_agentbench_hooks import Backend, TOOLS, RAW, SWITCHES, native


class SharedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def artifact(self, layer, body, name=None):
        path = self.root / ((name or layer) + ".py")
        path.write_text(f"TARGET_LAYER='{layer}'\nclass Harness:\n" + body)
        return path, layer

    def harness(self, artifacts=(), domain="telecom", options=None):
        return hooks.CompositeHarness(artifacts, domain, options)

    def manifest(self, artifacts=()):
        records = [{"mechanism_id": path.stem, "revision": 1, "target_layer": layer,
                    "artifact_path": path.name, "sha256": hooks.digest(path)} for path, layer in artifacts]
        value = {"schema_version": 2, "artifact_root": str(self.root), "records": records,
                 "snapshot_sha256": hooks.object_digest(records)}
        value["manifest_sha256"] = hooks.object_digest(value)
        path = self.root / "manifest.json"
        path.write_text(json.dumps(value))
        return path

    def test_same_empty_hooks_all_six_domains(self):
        artifacts = [(hooks.TEMPLATES / f"h{i}.py", f"h{i}") for i in range(2, 6)]
        for domain in hooks.CONTEXT_FIELDS:
            h = self.harness(artifacts, domain)
            self.assertEqual(h.h3(TOOLS), TOOLS)
            self.assertEqual(h.h2(RAW, []), RAW)
            self.assertEqual(h.h4([], 5), [])
            self.assertEqual(h.h5([]), [])
            self.assertIsNone(h.h2_no_tool_message(RAW, 5))
            self.assertEqual(h.drain_h2_messages(), [])
            self.assertEqual(h.shared_state, {})

    def test_state_is_shared_and_reset_between_episodes(self):
        writer = self.artifact("h4", """    def bind_shared_state(self,state):
        self.state=state
    def h4(self,messages,remaining):
        self.state['pending']='synthetic'
        return []
""")
        reader = self.artifact("h2", """    def bind_shared_state(self,state):
        self.state=state
    def h2(self,message,history):
        message['content']=self.state.pop('pending','raw')
        return message
""")
        h = self.harness([writer, reader])
        h.h4([], 2)
        self.assertEqual(h.h2(RAW, [])['content'], 'synthetic')
        self.assertIs(h.layers[0][1].state, h.layers[1][1].state)
        self.assertEqual(self.harness([writer, reader]).shared_state, {})

    def test_private_context_and_schema_mutation_rejected(self):
        for key in ('task_id', 'reward', 'answer', 'evaluation_criteria', 'db', 'user_scenario'):
            with self.assertRaises(ValueError):
                self.harness().bind_task_context({key: 'private'})
        bad = self.artifact('h3', """    def h3(self,tools):
        tools[0]['function']['parameters']={}
        return tools
""")
        with self.assertRaises(ValueError): self.harness([bad]).h3(TOOLS)

    def test_veto_closes_only_actual_proposed_call_ids(self):
        h2 = self.artifact('h2', """    def h2(self,message,history):
        message['_harness_control']={'action':'suppress_tool','messages':[{'role':'user','content':'synthetic retry'}]}
        return message
""")
        value = self.harness([h2]).h2(RAW, [])
        feedback = value['_harness_control']['messages']
        self.assertEqual(feedback[0]['tool_call_id'], 'c1')
        self.assertIn('no environment action', feedback[0]['content'])
        self.assertEqual(feedback[1]['content'], 'synthetic retry')

    def test_forged_feedback_and_empty_veto_rejected(self):
        for messages in ([], [{'role':'tool','tool_call_id':'forged','content':'bad'}]):
            artifact = self.artifact('h2', f"    def h2(self,message,history):\n        message['_harness_control']={{'action':'suppress_tool','messages':{messages!r}}}\n        return message\n")
            with self.assertRaises(ValueError): self.harness([artifact]).h2(RAW, [])

    def test_agentbench_new_install_and_fingerprint(self):
        task = SimpleNamespace(FourHookSession=native.FourHookSession)
        receipt = hooks.install_agentbench(task, 'webshop', self.manifest())
        self.assertEqual(receipt['interface_version'], hooks.INTERFACE_VERSION)
        self.assertIn('common_sources', receipt)
        backend = Backend()
        session = task.FourHookSession(backend, task.WebShopHarness(), TOOLS, 3, SWITCHES)
        session.inject({'role':'system','content':'native'})
        session.inject({'role':'user','content':'public'})
        self.assertEqual(session.sync_action().messages, [RAW])
        self.assertEqual(backend.requests[0], [{'role':'system','content':'native'},{'role':'user','content':'public'}])

    def test_os_binding_uses_only_public_description_and_resets_state(self):
        h3 = self.artifact('h3', """    def bind_task_context(self,context):
        self.context=context
    def h3(self,tools):
        tools[0]['function']['description']=self.context['description']
        return tools
""")
        task=SimpleNamespace(FourHookSession=native.FourHookSession)
        receipt=hooks.install_agentbench(task,'os',self.manifest([h3]))
        self.assertEqual(receipt['public_context_fields']['os'],['description'])
        backend=Backend()
        harness=task.OSInteractionHarness()
        session=task.FourHookSession(backend,harness,TOOLS,16,SWITCHES,
                                     task_context={'description':'public OS request'})
        self.assertEqual(backend.tools[0]['function']['description'],'public OS request')
        self.assertEqual(session.sync_action().messages,[RAW])
        for key in ('check','match','init_script','example_script','task_id','container_id'):
            with self.assertRaises(ValueError):
                harness.bind_task_context({'description':'public',key:'private'})
        harness.shared_state['progress']=1
        self.assertEqual(task.OSInteractionHarness().shared_state,{})

    def test_h3_model_view_updates_shared_state_each_turn_without_history_rewrite(self):
        h3=self.artifact('h3', """    def bind_shared_state(self,state):
        self.state=state
    def h3(self,tools):
        return tools
    def h3_messages(self,messages):
        messages[0]['content']+=' state='+str(self.state.get('progress',0))
        return messages
""")
        h4=self.artifact('h4', """    def bind_shared_state(self,state):
        self.state=state
    def h4(self,messages,remaining):
        self.state['progress']=1
        return []
""")
        for domain in hooks.CONTEXT_FIELDS:
            backend=Backend()
            session=hooks.session_class(native.FourHookSession)(backend,self.harness([h3,h4],domain),TOOLS,3,SWITCHES)
            session.inject({'role':'system','content':'public system'})
            session.sync_action();session.sync_action()
            self.assertEqual(backend.requests[0][0]['content'],'public system state=0')
            self.assertEqual(backend.requests[1][0]['content'],'public system state=1')
            self.assertEqual(backend.history[0]['content'],'public system')

    def test_new_h3_hook_signature_role_and_layer_guards(self):
        for layer,signature in [('h2','self,messages'),('h3','self,messages,remaining')]:
            primary='self,message,history' if layer=='h2' else 'self,tools'
            artifact=self.artifact(layer,f"    def {layer}({primary}):\n        return {'message' if layer=='h2' else 'tools'}\n    def h3_messages({signature}):\n        return messages\n")
            with self.assertRaises(ValueError):hooks.validate_source(*artifact)
        artifact=self.artifact('h3', """    def h3(self,tools):
        return tools
    def h3_messages(self,messages):
        messages[0]['role']='user'
        return messages
""")
        backend=Backend();session=hooks.session_class(native.FourHookSession)(backend,self.harness([artifact]),TOOLS,3,SWITCHES)
        session.inject({'role':'system','content':'public'})
        with self.assertRaises(ValueError):session.sync_action()
        self.assertEqual(backend.history,[{'role':'system','content':'public'}])
        self.assertEqual(backend.requests,[])

    def test_environment_state_is_detached_per_layer_and_resets_per_episode(self):
        artifacts=[]
        for layer,args,result in [('h2','message,history','message'),('h3','tools','tools'),
                                  ('h4','messages,remaining','[]'),('h5','messages','[]')]:
            artifacts.append(self.artifact(layer,f"    def bind_environment_state(self,state):\n        self.environment=state\n    def {layer}(self,{args}):\n        return {result}\n"))
        state={'db':{'items':[{'value':1}]}}
        for domain in hooks.CONTEXT_FIELDS:
            harness=self.harness(artifacts,domain)
            self.assertTrue(all(instance.environment is None for _,instance in harness.layers))
            harness.bind_environment_state(state)
            with self.assertRaises(TypeError):
                harness.layers[0][1].environment['db']['items'][0]['value']=99
            self.assertEqual(state['db']['items'][0]['value'],1)
            self.assertEqual(harness.layers[1][1].environment['db']['items'][0]['value'],1)
            self.assertTrue(all(instance.environment is None for _,instance in self.harness(artifacts,domain).layers))
        for invalid in ({'db':{},'evaluation_criteria':{}},{'db':object()}, {'db':{'live':lambda:None}}):
            with self.assertRaises(ValueError):harness.bind_environment_state(invalid)

    def test_environment_hook_signature_and_agentbench_provider(self):
        invalid=self.artifact('h2', """    def bind_environment_state(self,state,extra):
        pass
    def h2(self,message,history):
        return message
""")
        with self.assertRaises(ValueError):hooks.validate_source(*invalid)
        h2=self.artifact('h2', """    def bind_environment_state(self,state):
        self.environment=state
    def h2(self,message,history):
        message['content']=str(self.environment['db']['value'])
        return message
""")
        task=SimpleNamespace(FourHookSession=native.FourHookSession)
        state={'db':{'value':1}}
        receipt=hooks.install_agentbench(task,'webshop',self.manifest([h2]),environment_state_provider=lambda:state)
        self.assertTrue(receipt['environment_state_binding'])
        backend=Backend()
        session=task.FourHookSession(backend,task.WebShopHarness(),TOOLS,3,SWITCHES)
        session.inject({'role':'system','content':'public'})
        state['db']['value']=2
        self.assertEqual(session.sync_action().messages[0]['content'],'2')
        self.assertNotIn('db',json.dumps(backend.requests))


class EnvironmentStateTests(unittest.TestCase):
    def test_read_only_views_are_json_compatible_and_share_only_unchanged_branches(self):
        from evolution.environment_state import StateLog,StateReader,read_only
        log=StateLog();log.record({'db':{'stable':[{'value':1}],'changing':0}})
        initial=log.view
        log.record({'db':{'stable':[{'value':1}],'changing':1}})
        self.assertIs(initial['db']['stable'],log.view['db']['stable'])
        reader=StateReader(log.records)
        a,b=reader.get(0),reader.get(1)
        self.assertIs(a['db']['stable'],b['db']['stable'])
        self.assertEqual(json.loads(json.dumps(b)),{'db':{'stable':[{'value':1}],'changing':1}})
        independent=read_only(b)
        self.assertIsNot(independent['db']['stable'],b['db']['stable'])
        for mutate in (lambda:independent['db'].update({'changing':3}),
                       lambda:independent['db']['stable'].append({}),
                       lambda:independent['db']['stable'].pop()):
            with self.assertRaises(TypeError):mutate()
        with self.assertRaises(ValueError):reader.get(True)
        with self.assertRaises(ValueError):read_only({'db':{'invalid':float('nan')}})

    def test_state_log_deduplicates_and_replays_nested_changes_and_returns_to_old_state(self):
        from evolution.environment_state import StateLog,resolve
        log=StateLog()
        initial={'db':{'users':{'U1':{'paid':False,'items':[1,2]}},'obsolete':True}}
        log.record(initial);first=log.index
        log.record(initial)
        self.assertEqual(len(log.records),1)
        updated={'db':{'users':{'U1':{'paid':True,'items':[1,3,4]},'U2':{}},'new':None}}
        log.record(updated);second=log.index
        view=resolve(log.records,second)
        self.assertNotIn('state',log.records[second])
        self.assertEqual(resolve(log.records,first),initial)
        self.assertEqual(resolve(log.records,second),updated)
        view['db'].clear()
        self.assertEqual(resolve(log.records,second),updated)
        log.record(initial)
        self.assertEqual((log.index,len(log.records)),(first,2))
        log.record(None)
        self.assertIsNone(resolve(log.records,log.index))
        log.record({'db':{'new':1}})
        self.assertEqual(resolve(log.records,log.index),{'db':{'new':1}})

    def test_state_log_rejects_hash_tampering_invalid_indices_and_forward_references(self):
        from evolution.environment_state import StateLog,resolve
        log=StateLog();log.record({'db':{'value':1}});log.record({'db':{'value':2}})
        for bad in (-1,True,99):
            with self.assertRaises(ValueError):resolve(log.records,bad)
        changed=copy.deepcopy(log.records);changed[0]['state']['db']['value']=3
        with self.assertRaises(ValueError):resolve(changed,0)
        changed=copy.deepcopy(log.records);changed[1]['base']=1
        with self.assertRaises(ValueError):resolve(changed,1)


class TauTests(SharedTests):
    @classmethod
    def setUpClass(cls):
        from evolution import tau_hooks
        cls.tau = tau_hooks
        from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage
        cls.AssistantMessage, cls.ToolCall = AssistantMessage, ToolCall
        cls.ToolMessage, cls.UserMessage = ToolMessage, UserMessage

    def agent(self, artifacts=(), domain='telecom'):
        agent = self.tau.LifeAgent(tools=[self.tau.SchemaTool(TOOLS[0])], domain_policy='native public policy',
            llm='offline', llm_args={}, harness_factory=lambda: self.harness(artifacts, domain))
        agent._life_remaining = agent._life_turn_limit = 10
        return agent, agent.get_init_state()

    def raw(self, calls=True, name='act', arguments=None):
        if calls:
            return self.AssistantMessage(role='assistant', tool_calls=[self.ToolCall(id='c1', name=name, arguments=arguments or {})], cost=.1, usage={'prompt_tokens':10,'completion_tokens':2})
        return self.AssistantMessage(role='assistant', content='synthetic response', cost=.1, usage={'prompt_tokens':10,'completion_tokens':2})

    def test_empty_tau_requests_equal_native_and_no_private_context(self):
        from tau2.agent.llm_agent import LLMAgent
        from tau2.utils.llm_utils import to_litellm_messages
        response = self.raw(False)
        for domain in ('airline','retail','telecom'):
            agent, state = self.agent(domain=domain)
            message = self.UserMessage(role='user', content='public request')
            with patch('evolution.tau_hooks.generate', return_value=response) as wrapped:
                actual, _ = agent.generate_next_message(message, state)
            plain = LLMAgent(tools=agent.tools, domain_policy=agent.domain_policy, llm=agent.llm, llm_args={})
            with patch('tau2.agent.llm_agent.generate', return_value=response) as native_generate:
                expected, _ = plain.generate_next_message(message, plain.get_init_state())
            self.assertEqual(actual, expected)
            a, b = wrapped.call_args.kwargs, native_generate.call_args.kwargs
            self.assertEqual(to_litellm_messages(a['messages']), to_litellm_messages(b['messages']))
            self.assertEqual([t.openai_schema for t in a['tools']], [t.openai_schema for t in b['tools']])
            self.assertEqual(set(a), set(b))
            self.assertEqual(state._life_session.harness.shared_state, {})

    def test_native_business_db_access_is_fresh_at_model_and_action_without_private_state(self):
        from evolution.environment_state import resolve
        h2=self.artifact('h2', """    def bind_environment_state(self,state):
        self.environment=state
    def h2(self,message,history):
        if not self.environment['db']['permitted']:
            message['_harness_control']={'action':'suppress_tool','messages':[{'role':'user','content':'synthetic business check'}]}
        return message
""")
        h4=self.artifact('h4', """    def bind_environment_state(self,state):
        self.environment=state
    def bind_shared_state(self,state):
        self.state=state
    def h4(self,messages,remaining):
        self.state['business_value']=self.environment['db']['permitted']
        return []
""")
        for domain in ('airline','retail','telecom'):
            data={'permitted':True,'unqueried':'BUSINESS_ONLY_SENTINEL'}
            db=SimpleNamespace(model_dump=lambda **kwargs:copy.deepcopy(data))
            env=SimpleNamespace(tools=SimpleNamespace(db=db),sync_tools=Mock(),
                user_tools=SimpleNamespace(db='PRIVATE_USER_DB'),evaluation_criteria='PRIVATE_EVAL')
            agent=self.tau.LifeAgent(tools=[self.tau.SchemaTool(TOOLS[0])],domain_policy='public',
                llm='offline',llm_args={},harness_factory=lambda:self.harness([h2,h4],domain))
            orch=self.tau.LifeOrchestrator(domain=domain,agent=agent,user=SimpleNamespace(generate_next_message=Mock()),
                environment=env,task=None,max_steps=4)
            state=agent.get_init_state()
            def response(**kwargs):
                data['permitted']=False
                return self.raw()
            with patch('evolution.tau_hooks.generate',side_effect=response) as generate:
                with self.assertRaises(self.tau.SuppressedAction):
                    agent.generate_next_message(self.UserMessage(role='user',content='public'),state)
            row=agent.public_episode(state,index='opaque',success=0,status='completed')
            call=row['calls'][0]
            self.assertTrue(resolve(row['environment_states'],call['environment_state_index'])['db']['permitted'])
            self.assertFalse(resolve(row['environment_states'],call['action_environment_state_index'])['db']['permitted'])
            request=generate.call_args.kwargs
            self.assertNotIn('BUSINESS_ONLY_SENTINEL',json.dumps([self.tau.public_message(m) for m in request['messages']]))
            self.assertNotIn('PRIVATE_',json.dumps(row))
            self.assertEqual(len(row['environment_states']),2)
            data['permitted']=True
            with patch('evolution.tau_hooks.generate',return_value=self.raw()):
                agent.generate_next_message(self.UserMessage(role='user',content='public'),state)
            self.assertTrue(state._life_session.harness.shared_state['business_value'])
            self.assertTrue(data['permitted'])

    def test_failed_solver_request_does_not_rewrite_previous_action_state(self):
        agent,state=self.agent()
        data={'db':{'value':1}}
        agent._life_environment_state_provider=lambda:data
        with patch('evolution.tau_hooks.generate',return_value=self.raw(False)):
            agent.generate_next_message(self.UserMessage(role='user',content='public'),state)
        before=copy.deepcopy(state._life_session.session.calls)
        data['db']['value']=2
        with patch('evolution.tau_hooks.generate',side_effect=RuntimeError('synthetic transport failure')):
            with self.assertRaises(RuntimeError):
                agent.generate_next_message(self.UserMessage(role='user',content='public'),state)
        self.assertEqual(state._life_session.session.calls,before)

    def test_h3_h5_and_long_h4_reach_solver_without_environment_changes(self):
        h3 = self.artifact('h3', """    def h3(self,tools):
        tools[0]['function']['description']='synthetic description'
        return tools
    def h3_message(self,message):
        if message['role']=='system': message['content']+=' conditioned'
        return message
""")
        h5 = self.artifact('h5', "    def h5(self,messages):\n        return ['synthetic skill']\n")
        h4 = self.artifact('h4', "    def h4(self,messages,remaining):\n        return ['synthetic '*300]\n")
        agent, state = self.agent([h3,h5,h4])
        with patch('evolution.tau_hooks.generate', return_value=self.raw()) as generate:
            agent.generate_next_message(self.UserMessage(role='user',content='public'),state)
            agent.generate_next_message(self.ToolMessage(role='tool',id='c1',content='actual result'),state)
        first = generate.call_args_list[0].kwargs
        self.assertIn('synthetic skill', first['messages'][0].content)
        self.assertIn('conditioned', first['messages'][0].content)
        self.assertEqual(first['tools'][0].openai_schema['function']['description'], 'synthetic description')
        self.assertEqual(agent._life_native_tools[0].openai_schema, TOOLS[0])
        self.assertEqual(agent.tools[0].openai_schema['function']['description'], 'synthetic description')
        self.assertIn('[H4] '+'synthetic '*300, [m.content for m in generate.call_args_list[1].kwargs['messages']])
        self.assertNotIn('synthetic skill', state.system_messages[0].content)

    def test_h2_replacement_has_matching_executed_history(self):
        h2 = self.artifact('h2', """    def h2(self,message,history):
        message['tool_calls'][0]['function']['arguments']='{"repaired":true}'
        return message
""")
        agent, state = self.agent([h2])
        with patch('evolution.tau_hooks.generate', return_value=self.raw()) as generate:
            value, _ = agent.generate_next_message(self.UserMessage(role='user',content='public'),state)
            self.assertEqual(value.tool_calls[0].arguments, {'repaired':True})
            agent.generate_next_message(self.ToolMessage(role='tool',id='c1',content='actual result'),state)
        prior = next(m for m in generate.call_args.kwargs['messages'] if m.role=='assistant')
        self.assertEqual(prior.tool_calls[0].arguments, {'repaired':True})
        self.assertEqual(state._life_session.trace['h2'][0]['before']['tool_calls'][0]['function']['arguments'], '{}')

    def test_veto_then_recovery_no_duplicate_user_or_extra_dispatch(self):
        h2 = self.artifact('h2', """    def bind_shared_state(self,state):
        self.state=state
    def h2(self,message,history):
        if not self.state.get('blocked'):
            self.state['blocked']=True
            message['_harness_control']={'action':'suppress_tool','messages':[{'role':'user','content':'synthetic correction'}]}
        return message
""")
        agent, state = self.agent([h2])
        message = self.UserMessage(role='user',content='one public request')
        with patch('evolution.tau_hooks.generate', return_value=self.raw()) as generate:
            with self.assertRaises(self.tau.SuppressedAction): agent.generate_next_message(message,state)
            value, _ = agent.generate_next_message(message,state)
        self.assertEqual(value.tool_calls[0].name,'act')
        history = generate.call_args.kwargs['messages']
        self.assertEqual(sum(m.content=='one public request' for m in history),1)
        self.assertTrue(any(m.role=='tool' and 'suppressed before execution' in m.content for m in history))
        self.assertAlmostEqual(value.cost,.2)
        self.assertEqual(value.usage['prompt_tokens'],20)

    def test_orchestrator_suppression_consumes_original_steps_and_stops(self):
        h2 = self.artifact('h2', """    def h2(self,message,history):
        message['_harness_control']={'action':'suppress_tool','messages':[{'role':'user','content':'synthetic correction'}]}
        return message
""")
        agent, state = self.agent([h2])
        env = SimpleNamespace(sync_tools=Mock())
        user = SimpleNamespace(generate_next_message=Mock(side_effect=AssertionError('must not simulate user')))
        orch = self.tau.LifeOrchestrator(domain='telecom',agent=agent,user=user,environment=env,task=None,max_steps=2)
        orch.agent_state=state
        orch.from_role, orch.to_role = self.tau.Role.USER, self.tau.Role.AGENT
        orch.message=self.UserMessage(role='user',content='public')
        with patch('evolution.tau_hooks.generate',return_value=self.raw()) as generate:
            orch.step(); orch._check_termination()
            self.assertFalse(orch.done)
            orch.step(); orch._check_termination()
            self.assertTrue(orch.done)
            self.assertEqual(generate.call_count,2)
        self.assertEqual(orch.step_count,2)
        self.assertEqual(orch.termination_reason.value,'max_steps')
        self.assertEqual(orch.trajectory,[])
        user.generate_next_message.assert_not_called()
        self.assertEqual(env.sync_tools.call_count,2)

    def test_no_tool_nudge_and_state_do_not_cross_episodes(self):
        h2 = self.artifact('h2', """    def bind_shared_state(self,state):
        self.state=state
    def h2(self,message,history):
        return message
    def h2_no_tool_message(self,message,remaining):
        if not self.state.get('nudge'):
            self.state['nudge']=True
            return 'synthetic execute-next reminder'
        return None
""")
        agent, state = self.agent([h2])
        with patch('evolution.tau_hooks.generate',return_value=self.raw(False)):
            with self.assertRaises(self.tau.SuppressedAction): agent.generate_next_message(self.UserMessage(role='user',content='public'),state)
        other = agent.get_init_state()
        self.assertEqual(other._life_session.harness.shared_state,{})
        self.assertFalse(other._life_retry)
        self.assertTrue(state._life_session.harness.shared_state['nudge'])

    def test_public_export_prepares_tau_evolve_without_private_fields(self):
        from evolution.life_evolve import profile
        agent,state=self.agent()
        agent.llm_args={'api_key':'PRIVATE_SENTINEL'}
        with patch('evolution.tau_hooks.generate',return_value=self.raw()):
            agent.generate_next_message(self.UserMessage(role='user',content='public'),state)
        row=agent.public_episode(state,index='public_hash',success=0,status='completed')
        case=profile.compact_case(row,profile.public_context('telecom',row,None))
        self.assertNotIn('PRIVATE_SENTINEL',json.dumps(case))
        self.assertEqual(case['context']['policy'],state.system_messages[0].content)
        self.assertEqual(case['calls'][0]['remaining'],10)
        self.assertEqual(case['turn_limit'],10)

    def test_install_is_opt_in_and_preserves_native_factory(self):
        from tau2.registry import registry
        from tau2.runner import build
        baseline=registry.get_agent_factory('llm_agent')
        with patch.object(build,'Orchestrator',self.tau.Orchestrator):
            name, receipt=self.tau.install('telecom',self.manifest())
            self.assertIs(registry.get_agent_factory('llm_agent'),baseline)
            self.assertIs(build.Orchestrator,self.tau.LifeOrchestrator)
            self.assertEqual(receipt['interface_version'],hooks.INTERFACE_VERSION)
            self.assertEqual(receipt['native_session_sha256'],hooks.digest(Path(__file__).resolve().parents[2]/'AgentBench/src/server/harness/session.py'))
            self.assertIsInstance(registry.get_agent_factory(name)(tools=[],domain_policy='public',llm='offline'),self.tau.LifeAgent)
            self.assertEqual(name,self.tau.install('telecom',self.root/'manifest.json')[0])

    def test_native_cli_train_export_connects_to_evolve_without_test_leakage(self):
        from evolution.life_evolve import profile
        agent,state=self.agent()
        with patch('evolution.tau_hooks.generate',return_value=self.raw(False)):
            agent.generate_next_message(self.UserMessage(role='user',content='public'),state)
        native_id='[service_issue]private_fault|another_fault[PERSONA:Hard]'
        row=agent.public_episode(state,index=native_id,success=0,status='user_stop')
        result=SimpleNamespace(info={'life_public_episode':row,'private':'PRIVATE_SENTINEL'},
                               reward_info=SimpleNamespace(reward=1),termination_reason='user_stop')
        manifest=self.manifest()
        destination=self.root/'train_export'
        summary=self.tau.export_training(SimpleNamespace(simulations=[result]),'telecom',manifest,destination)
        self.assertEqual((summary['n'],summary['successes'],summary['infra_errors']),(1,1,0))
        case_id=self.tau.training_case_id(native_id)
        self.assertEqual(set(summary['per_task']),{case_id})
        self.assertTrue((destination/f'episode_{case_id}.json').is_file())
        self.assertNotIn(native_id,json.dumps(summary))
        seed=self.root/'seed';seed.mkdir()
        (seed/'state.json').write_text(json.dumps({'domain':'telecom','active_snapshot':[],
            'active_snapshot_sha256':hooks.object_digest([]),'rounds':[]}))
        workspace=self.root/'workspace'
        profile.prepare(seed,destination,workspace)
        self.assertEqual(len(profile.load_cases(workspace)),1)
        self.assertNotIn('PRIVATE_SENTINEL',''.join(p.read_text() for p in destination.glob('*.json')))
        self.assertNotIn('private_fault',''.join(p.read_text() for p in workspace.rglob('*.json')))
        with self.assertRaises(ValueError):
            self.tau.export_training(SimpleNamespace(simulations=[result]),'telecom',manifest,destination)

    def test_native_orchestrator_dispatches_only_repaired_action(self):
        h2=self.artifact('h2', """    def h2(self,message,history):
        message['tool_calls'][0]['function']['arguments']='{"repaired":true}'
        return message
""")
        agent,state=self.agent([h2])
        env=SimpleNamespace(sync_tools=Mock())
        user=SimpleNamespace(generate_next_message=Mock())
        orch=self.tau.LifeOrchestrator(domain='telecom',agent=agent,user=user,environment=env,task=None,max_steps=4)
        orch.agent_state=state
        orch.from_role,orch.to_role=self.tau.Role.USER,self.tau.Role.AGENT
        orch.message=self.UserMessage(role='user',content='public')
        with patch('evolution.tau_hooks.generate',return_value=self.raw()):orch.step()
        self.assertEqual(orch.message.tool_calls[0].arguments,{'repaired':True})
        self.assertEqual(orch.to_role,self.tau.Role.ENV)
        result=self.ToolMessage(role='tool',id='c1',content='actual environment result')
        with patch.object(orch,'_execute_tool_calls',return_value=[result]) as execute:
            orch.step()
        execute.assert_called_once()
        self.assertEqual(execute.call_args.args[0][0].arguments,{'repaired':True})
        user.generate_next_message.assert_not_called()

    def test_cli_opt_in_does_not_change_legacy_defaults_and_rejects_mixing(self):
        import sys
        path=Path(__file__).resolve().parents[2]/'TauBench/scripts/eval_harness.py'
        spec=importlib.util.spec_from_file_location('life_tau_cli_fixture',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with patch.object(sys,'argv',['eval_harness']):
            args=module.parse_args()
            self.assertFalse(args.enabled)
            self.assertFalse(hasattr(args,'life_manifest'))
        with self.assertRaisesRegex(ValueError,'do not mix legacy'):
            self.tau.evaluation_cli(['--life-manifest','unused','--h2'])

    def test_opt_in_cli_preserves_native_budgets_exports_train_only_and_counts_infra(self):
        import io
        from contextlib import redirect_stdout
        agent,state=self.agent(domain='retail')
        with patch('evolution.tau_hooks.generate',return_value=self.raw(False)):
            agent.generate_next_message(self.UserMessage(role='user',content='public'),state)
        manifest=self.manifest()
        for split in ('train','test'):
            simulations=[]
            for number in range(2):
                row=agent.public_episode(state,index=f'opaque_{number}',success=0,status='user_stop')
                simulations.append(SimpleNamespace(task_id=str(number),trial=0,messages=[],
                    info={'life_public_episode':row},termination_reason='user_stop',
                    reward_info=SimpleNamespace(reward=1,reward_breakdown={}) if number==0 else None))
            results=SimpleNamespace(simulations=simulations)
            output=self.root/split
            output.mkdir()
            receipt={'interface_version':hooks.INTERFACE_VERSION}
            with patch('evolution.tau_hooks.install',return_value=('synthetic_life',receipt)), \
                 patch('tau2.runner.batch.run_tasks',return_value=results) as run, \
                 patch('tau2.domains.retail.environment.get_tasks',return_value=[SimpleNamespace(id='0'),SimpleNamespace(id='1')]), \
                 redirect_stdout(io.StringIO()):
                self.tau.evaluation_cli(['--life-manifest',str(manifest),'--domain','retail',
                    '--split',split,'--save-to',str(output),'--user-api-base','',
                    '--agent-max-tokens','4096','--max-steps','123','--concurrency','17'])
            config=run.call_args.args[0]
            self.assertEqual((config.agent,config.max_steps,config.max_concurrency),('synthetic_life',123,17))
            self.assertEqual(config.llm_args_agent['max_tokens'],4096)
            self.assertFalse(any((config.harness_enabled,config.harness_h3,config.harness_h4,config.harness_h5)))
            summary=json.loads((output/'harness_summary.json').read_text())
            self.assertEqual((summary['n'],summary['successes'],summary['infra_errors'],summary['accuracy']),(2,1,1,.5))
            self.assertIn('native_cli_sha256',summary['life_interface'])
            self.assertEqual((output/'life_train').exists(),split=='train')
            if split=='train':
                train=json.loads((output/'life_train/summary.json').read_text())
                self.assertEqual((train['n'],train['successes'],train['infra_errors']),(2,1,1))
                self.assertEqual(train['life_interface'],summary['life_interface'])


class TauEvolveTests(unittest.TestCase):
    def test_all_three_tau_domains_prepare_and_check_via_current_evolve(self):
        from evolution.tests.test_life_repair import _revision_fixtures
        fixtures=_revision_fixtures('_shared_evolve_fixture','test_life_optimizer.py',True,profile_module='evolution.life_evolve')
        for domain in ('airline','retail','telecom'):
            case=fixtures.OptimizerTests()
            case.setUp()
            try:
                case.state['domain']=case.summary['domain']=domain
                fixtures.p.write_json(case.seed/'state.json',case.state)
                fixtures.p.write_json(case.eval/'summary.json',case.summary)
                case.prepare(); case.write_candidate()
                report=case.report()
                self.assertTrue(report['passed'],report)
                self.assertIn('Shared Life H2', (case.workspace/'source_view/INTERFACE.md').read_text())
                self.assertIn('evolution/tau_hooks.py',fixtures.p.source_hashes())
                self.assertEqual(len(fixtures.p.load_cases(case.workspace)),20)
            finally:
                case.doCleanups()


if __name__=='__main__':
    unittest.main()
