"""Offline interface tests; synthetic policies here are NOT proposer priors."""
import asyncio
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from evolution.agentbench_hooks import CompositeHarness, install
from evolution.agentbench_hooks.contract import load_artifact, proposer_instructions, validate_source
from evolution.agentbench_hooks.runtime import digest, object_digest, session_class

spec = importlib.util.spec_from_file_location(
    "interface_test_native_session", ROOT / "AgentBench/src/server/harness/session.py")
native = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = native
spec.loader.exec_module(native)
CompleteSession = session_class(native.FourHookSession)
TEMPLATES = ROOT / "evolution/agentbench_hooks/templates"
TOOLS = [{"type": "function", "function": {
    "name": "act", "description": "unchanged", "parameters": {"type": "object"}}}]
RAW = {"role": "assistant", "content": "raw", "tool_calls": [{
    "id": "c1", "type": "function", "function": {"name": "act", "arguments": "{}"}}]}
SWITCHES = SimpleNamespace(**{f"h{i}_enabled": True for i in range(2, 6)})


class Backend:
    def __init__(self, response=None):
        self.history = []
        self.requests = []
        self.response = response or RAW
        self.controller = SimpleNamespace(env_output=SimpleNamespace(history=[]))

    def set_tools(self, tools):
        self.tools = tools

    def set_full_history(self, enabled):
        self.full_history = enabled

    def inject(self, item):
        self.history.append(copy.deepcopy(item))

    def cover(self, history):
        self.history = copy.deepcopy(history)

    def sync_action(self):
        self.requests.append(copy.deepcopy(self.history))
        return SimpleNamespace(messages=[copy.deepcopy(self.response)])

    async def action(self):
        return self.sync_action()


class InterfaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def artifact(self, layer, body, suffix=""):
        path = self.root / (layer + suffix + ".py")
        path.write_text(f"TARGET_LAYER = '{layer}'\nclass Harness:\n" + body)
        return path, layer

    def empty(self, domain="dbbench", options=None):
        return CompositeHarness([(TEMPLATES / f"h{i}.py", f"h{i}") for i in range(2, 6)], domain, options)

    def session(self, harness, response=None, switches=SWITCHES):
        backend = Backend(response)
        session = CompleteSession(backend, harness, TOOLS, 4, switches)
        session.inject({"role": "system", "content": "system"})
        session.inject({"role": "user", "content": "task"})
        return session, backend

    def test_all_empty_templates_insert_no_policy(self):
        for domain in ("dbbench", "webshop", "alfworld"):
            harness = self.empty(domain)
            self.assertEqual(harness.h3(TOOLS), TOOLS)
            self.assertEqual(harness.h3_message({"role": "user", "content": "task"}),
                             {"role": "user", "content": "task"})
            self.assertEqual(harness.h4([], 2), [])
            self.assertEqual(harness.h5([]), [])
            self.assertEqual(harness.h2(RAW, []), RAW)
            self.assertIsNone(harness.h2_no_tool_message(RAW, 2))
            self.assertEqual(harness.drain_h2_messages(), [])
            self.assertEqual(harness.shared_state, {})
            session, backend = self.session(harness)
            self.assertEqual(session.sync_action().messages, [RAW])
            self.assertEqual(backend.requests[0], [
                {"role": "system", "content": "system"}, {"role": "user", "content": "task"}])

    def test_context_is_bound_before_h3_and_is_copied(self):
        artifact = self.artifact("h3", """    def bind_task_context(self, context):
        self.context = context
    def h3(self, tools):
        tools[0]['function']['description'] = self.context['type'][0]
        self.context['table']['columns'].append('local')
        return tools
""")
        context = {"description": "public", "type": ["INSERT"], "table": {"columns": ["a,b", "Model 01+"]}}
        harness = CompositeHarness([artifact], "dbbench")
        session = CompleteSession(Backend(), harness, TOOLS, 4, SWITCHES, context)
        self.assertEqual(session.session.tools[0]['function']['description'], "INSERT")
        self.assertEqual(context['table']['columns'], ["a,b", "Model 01+"])
        for _, layer in self.empty().layers:
            self.assertFalse(hasattr(layer, "context"))

    def test_private_context_rejected(self):
        for key in ("sql", "label", "answer_md5", "hidden_goal", "reward"):
            with self.assertRaises(ValueError):
                self.empty().bind_task_context({key: "private"})
        self.empty("webshop").bind_task_context({"instruction": "public"})
        with self.assertRaises(ValueError):
            self.empty("webshop").bind_task_context({"instruction": "public", "type": "hidden"})

    def test_shared_state_cross_layer_and_episode_isolation(self):
        h4 = self.artifact("h4", """    def bind_shared_state(self, state):
        self.state = state
    def h4(self, messages, remaining):
        self.state['next'] = 'synthetic'
        return []
""")
        h2 = self.artifact("h2", """    def bind_shared_state(self, state):
        self.state = state
    def h2(self, message, history):
        if 'next' in self.state:
            message['content'] = self.state.pop('next')
        return message
""")
        harness = CompositeHarness([h4, h2], "webshop")
        session, backend = self.session(harness)
        self.assertEqual(session.sync_action().messages[0]['content'], "raw")
        self.assertEqual(session.sync_action().messages[0]['content'], "synthetic")
        self.assertIs(harness.layers[0][1].state, harness.layers[1][1].state)
        other = CompositeHarness([h4, h2], "webshop")
        self.assertIsNot(other.shared_state, harness.shared_state)
        self.assertEqual(other.shared_state, {})

    def test_h3_message_conditioning_and_metadata_guard(self):
        h3 = self.artifact("h3", """    def h3(self, tools):
        return tools
    def h3_message(self, message):
        message['content'] += ' conditioned'
        return message
""")
        session, backend = self.session(CompositeHarness([h3], "dbbench"))
        self.assertEqual(backend.history[1]['content'], "task conditioned")
        self.assertEqual(len(session.trace['h3']), 2)
        bad = self.artifact("h3", """    def h3(self, tools):
        return tools
    def h3_message(self, message):
        message['role'] = 'tool'
        return message
""", "bad")
        with self.assertRaises(ValueError):
            CompositeHarness([bad], "dbbench").h3_message({"role": "user", "content": "x"})

    def test_h3_cannot_change_schema(self):
        bad = self.artifact("h3", """    def h3(self, tools):
        tools[0]['function']['parameters'] = {}
        return tools
""")
        with self.assertRaises(ValueError):
            CompositeHarness([bad], "dbbench").h3(TOOLS)

    def test_long_hints_reach_real_bridge_and_h4_persists(self):
        artifacts = [self.artifact(layer, f"    def {layer}({args}):\n        return ['word ' * 200]\n")
                     for layer, args in (("h4", "self, messages, remaining"), ("h5", "self, messages"))]
        session, backend = self.session(CompositeHarness(artifacts, "dbbench"))
        session.sync_action()
        session.sync_action()
        self.assertIn("word " * 200, backend.requests[0][0]['content'])
        self.assertTrue(any(m.get('content') == 'word ' * 200 for m in backend.history))
        self.assertEqual(len(session.trace['h4'][0]['hints'][0].split()), 200)
        self.assertEqual(backend.history[0]['content'], 'system')  # H5 is payload-only.

    def test_ephemeral_hint_option(self):
        h4 = self.artifact("h4", "    def h4(self, messages, remaining):\n        return ['ephemeral']\n")
        session, backend = self.session(CompositeHarness([h4], "webshop", {"h4_persistent": False}))
        session.sync_action()
        session.sync_action()
        self.assertTrue(any(m.get('content') == 'ephemeral' for m in backend.requests[-1]))
        self.assertFalse(any(m.get('content') == 'ephemeral' for m in backend.history))

    def test_raw_history_and_side_messages(self):
        h2 = self.artifact("h2", """    def h2(self, message, history):
        message['content'] = 'repaired'
        return message
    def drain_h2_messages(self):
        return ['public side message']
""")
        session, backend = self.session(CompositeHarness([h2], "dbbench"))
        self.assertEqual(session.sync_action().messages[0]['content'], 'repaired')
        self.assertEqual(backend.history[-2], RAW)
        self.assertEqual(backend.history[-1], {'role': 'user', 'content': 'public side message'})

    def test_h2_veto_stops_later_h2_and_task_execution(self):
        first = self.artifact("h2", """    def h2(self, message, history):
        message['_harness_control'] = {'action': 'suppress_tool', 'messages': [
            {'role': 'tool', 'tool_call_id': 'c1', 'content': 'blocked'}]}
        return message
""")
        second = self.artifact("h2", "    def h2(self, message, history):\n        raise ValueError('must not run')\n", "second")
        session, backend = self.session(CompositeHarness([first, second], "dbbench"))
        self.assertEqual(session.sync_action().messages, [])
        self.assertTrue(session.consume_h2_suppressed())
        self.assertFalse(session.consume_h2_suppressed())
        self.assertEqual(backend.history[-1]['content'], 'blocked')

    def test_no_tool_recovery(self):
        h2 = self.artifact("h2", """    def h2(self, message, history):
        return message
    def h2_no_tool_message(self, message, remaining):
        return 'public recovery'
""")
        session, backend = self.session(CompositeHarness([h2], "alfworld"), {'role': 'assistant', 'content': 'no action'})
        self.assertEqual(session.sync_action().messages, [])
        self.assertTrue(session.consume_h2_suppressed())
        self.assertEqual(backend.history[-1]['content'], 'public recovery')

    def test_async_bridge_and_switches(self):
        h4 = self.artifact("h4", "    def h4(self, messages, remaining):\n        raise ValueError('disabled')\n")
        off = SimpleNamespace(**{f"h{i}_enabled": False for i in range(2, 6)})
        session, _ = self.session(CompositeHarness([h4], "dbbench"), switches=off)
        self.assertEqual(asyncio.run(session.action()).messages, [RAW])
        self.assertEqual(asyncio.run(session.action()).messages, [RAW])

    def test_validator_extensions_and_unsafe_access(self):
        for layer in ('h2', 'h3', 'h4', 'h5'):
            load_artifact(TEMPLATES / f'{layer}.py', layer)
        bad = self.artifact('h4', """    def h4(self, messages, remaining):
        return []
    def h3_message(self, message):
        return message
""")
        with self.assertRaises(ValueError):
            validate_source(*bad)
        for expression in ("open('private')", "getattr(self, 'secret')", "self.__dict__"):
            bad = self.artifact('h5', f"    def h5(self, messages):\n        {expression}\n        return []\n")
            with self.assertRaises(ValueError):
                validate_source(*bad)

    def test_install_is_local_hash_bound_and_tamper_rejected(self):
        artifact_root = self.root / 'artifacts'
        artifact_root.mkdir()
        path = artifact_root / 'h2.py'
        path.write_text((TEMPLATES / 'h2.py').read_text())
        records = [{"mechanism_id": "empty", "revision": 1, "target_layer": "h2",
                    "artifact_path": 'artifacts/h2.py', 'sha256': digest(path)}]
        body = {'schema_version': 2, 'snapshot_sha256': object_digest(records),
                'artifact_root': str(self.root), 'records': records}
        manifest = self.root / 'manifest.json'
        manifest.write_text(json.dumps({**body, 'manifest_sha256': object_digest(body)}))
        module = SimpleNamespace(FourHookSession=native.FourHookSession, DBBenchHarness=None)
        receipt = install(module, 'dbbench', manifest)
        self.assertEqual(receipt['interface_version'], 'life-agentbench-h2345-v3')
        self.assertIn('templates/h2.py', receipt['sources'])
        self.assertIsNot(module.FourHookSession, native.FourHookSession)
        self.assertIs(native.FourHookSession, CompleteSession.__bases__[0])
        self.assertEqual(module.DBBenchHarness().h2(RAW, []), RAW)
        path.write_text(path.read_text() + '\n# tampered')
        with self.assertRaises(ValueError):
            module.DBBenchHarness()
        with self.assertRaises(ValueError):
            install(module, 'dbbench', manifest)

    def test_contract_explicitly_documents_empty_scaffold(self):
        text = proposer_instructions()
        for phrase in ('policy-free', 'bind_task_context', 'bind_shared_state', 'h3_message',
                       'h2_no_tool_message', 'drain_h2_messages', 'no word-count limit'):
            self.assertIn(phrase, text)


if __name__ == '__main__':
    unittest.main()
