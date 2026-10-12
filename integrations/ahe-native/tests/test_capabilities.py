import copy
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import yaml

from ahe_native.backend import HERE, MANIFEST, validate_request, workspace_hash
from ahe_native.capabilities import (PROFILE, CapabilityViolation, DispatchBudget,
    constraint_text, install_runtime_guard, restricted_policy, sha, validate_workspace)
from ahe_native.capability_evidence import project_public_evidence
from ahe_native.cli import config_for
from ahe_native.environment_rpc import Channel
from ahe_native.evolve_isolation import bind_isolated_evolve
from ahe_native.debugger_isolation import evidence_config
from ahe_native.records import export_job, write_json
from ahe_native.upstream import bind_native_transport, load_upstream

PURE_MIDDLEWARE = '''from nexau.archs.main_sub.execution.hooks import Middleware, HookResult
class Planner(Middleware):
    def __init__(self):
        self.visited = set()
    def before_tool(self, hook_input):
        action = dict(hook_input.tool_input)
        if action.get("action") == "bad":
            action["action"] = "look"
        return HookResult.with_modifications(tool_input=action)
'''


class CapabilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.workspace = self.root / "workspace"
        shutil.copytree(HERE / "baselines/alfworld", self.workspace)
        (self.workspace / "middleware").mkdir(exist_ok=True)
        (self.workspace / "skills").mkdir(exist_ok=True)
        self.policy = restricted_policy("alfworld")

    def register(self, source=PURE_MIDDLEWARE):
        (self.workspace / "middleware/planner.py").write_text(source)
        config = copy.deepcopy(self.policy["reference_config"])
        config["middlewares"] = [{"import": "middleware.planner:Planner"}]
        (self.workspace / "code_agent.yaml").write_text(yaml.safe_dump(config))

    def test_all_six_native_baselines_pass_without_domain_policy(self):
        for domain in ("alfworld", "dbbench", "webshop", "airline", "retail", "telecom"):
            with self.subTest(domain=domain):
                result = validate_workspace(HERE / "baselines" / domain, restricted_policy(domain))
                self.assertEqual(result["profile"], PROFILE)

    def test_planning_action_repair_and_frozen_skill_text_are_allowed(self):
        self.register()
        (self.workspace / "systemprompt.md").write_text("{{ native_instructions }}\n{{ frozen_skills }}\n{{ frozen_memory }}")
        (self.workspace / "LongTermMEMORY.md").write_text("Candidate-produced frozen knowledge.")
        folder = self.workspace / "skills/learned"
        folder.mkdir()
        (folder / "SKILL.md").write_text("Candidate-produced frozen procedural guidance.")
        tool = self.workspace / "tool_descriptions/take_action.tool.yaml"
        value = yaml.safe_load(tool.read_text())
        value["description"] = "Candidate-produced tool description."
        tool.write_text(yaml.safe_dump(value))
        self.assertTrue(validate_workspace(self.workspace, self.policy)["candidate_valid"])

    def test_extra_tools_subagents_registered_skill_tools_and_config_changes_rejected(self):
        changes = {"sub_agents": [], "skills": ["./skills/learned"], "sandbox_config": {},
            "max_iterations": 999, "before_tool_hooks": [], "tools": [], "llm_config": {"model": "different"}}
        path = self.workspace / "code_agent.yaml"
        for key, value in changes.items():
            with self.subTest(key=key):
                config = copy.deepcopy(self.policy["reference_config"])
                config[key] = value
                path.write_text(yaml.safe_dump(config))
                with self.assertRaisesRegex(ValueError, "registry.*config"):
                    validate_workspace(self.workspace, self.policy)

    def test_edited_native_tool_implementation_rejected(self):
        (self.workspace / "tools/native.py").write_text("def take_action(**arguments):\n    return 'changed'\n")
        with self.assertRaisesRegex(ValueError, "immutable native file"):
            validate_workspace(self.workspace, self.policy)

    def test_extra_component_scripts_and_runtime_memory_seed_rejected(self):
        for relative in ("tools/new.py", "sub_agents/agent.yaml", "skills/script.py", "extra.py"):
            with self.subTest(relative=relative):
                path = self.workspace / relative
                path.parent.mkdir(exist_ok=True)
                path.write_text("not permitted")
                with self.assertRaisesRegex(ValueError, "new component"):
                    validate_workspace(self.workspace, self.policy)
                path.unlink()
        (self.workspace / "ShortTermMEMORY.md").write_text("cross-task seed")
        with self.assertRaisesRegex(ValueError, "immutable native file"):
            validate_workspace(self.workspace, self.policy)

    def test_network_llm_rpc_filesystem_executor_and_wrapper_paths_rejected(self):
        forbidden = [
            "import httpx\n", "from openai import OpenAI\n",
            "from ahe_native.environment_rpc import invoke\n",
            "from nexau import Agent\n", "from json import __builtins__ as b\n",
            "from copy import sys as s\n", "alias = getattr\n",
            "getattr(object(), 'executor')\n", "open('memory','w')\n",
            "hook_input.agent_state.executor.run()\n", "hook_input.sandbox.execute()\n",
            "def wrap_model_call(self, params, call_next):\n    return call_next(params)\n",
            "def wrap_tool_call(self, params, call_next):\n    return call_next(params)\n"]
        for source in forbidden:
            with self.subTest(source=source):
                self.register(source)
                with self.assertRaises(ValueError):
                    validate_workspace(self.workspace, self.policy)

    def test_public_getattr_and_episode_local_state_remain_available(self):
        self.register(PURE_MIDDLEWARE.replace("dict(hook_input.tool_input)",
                                             "dict(getattr(hook_input, 'tool_input', {}))"))
        self.assertTrue(validate_workspace(self.workspace, self.policy)["candidate_valid"])

    def test_native_episode_state_and_parsed_tool_getattr_are_public(self):
        fields = ('get_context_value', 'set_context_value', 'tool_calls', 'parameters', 'tool_call_id')
        self.register(PURE_MIDDLEWARE + '\n' + '\n'.join(
            'field_' + str(i) + ' = getattr(object(), ' + repr(field) + ', None)'
            for i, field in enumerate(fields)))
        self.assertTrue(validate_workspace(self.workspace, self.policy)['candidate_valid'])
        for field in ('executor', '_executor', 'sandbox', 'openai_client', '__dict__',
                      'get_global_value', 'set_global_value', 'get_variable'):
            with self.subTest(field=field):
                self.register(PURE_MIDDLEWARE + '\ngetattr(object(), ' + repr(field) + ')')
                with self.assertRaisesRegex(ValueError, 'public hook fields'):
                    validate_workspace(self.workspace, self.policy)

    def test_public_message_response_and_hook_data_fields_are_complete(self):
        from nexau.core.messages import Message, TextBlock, ToolUseBlock, ToolResultBlock, Role
        from nexau.archs.main_sub.execution.parse_structures import ToolCall, ParsedResponse
        from nexau.archs.main_sub.execution import hooks
        models = (Message, TextBlock, ToolUseBlock, ToolResultBlock, ToolCall, ParsedResponse,
                  *(getattr(hooks, name) for name in ("BeforeAgentHookInput", "AfterAgentHookInput",
                     "BeforeModelHookInput", "AfterModelHookInput", "BeforeToolHookInput", "AfterToolHookInput")))
        names = {"value"}
        for model in models:
            for base in model.__mro__:
                names.update(n for n in getattr(base, "__annotations__", {})
                             if not n.startswith('_') and n != 'sandbox')
        self.register(PURE_MIDDLEWARE + '\n' + '\n'.join(
            'field_' + str(i) + ' = getattr(object(), ' + repr(name) + ', None)'
            for i, name in enumerate(sorted(names))))
        self.assertTrue(validate_workspace(self.workspace, self.policy)['candidate_valid'])
        self.assertEqual(getattr(TextBlock(text='public message'), 'text'), 'public message')
        self.assertEqual(getattr(Role.USER, 'value'), 'user')
        for name in ('sandbox', 'llm_config', 'executor', '__dict__', 'get_global_value', 'set_global_value'):
            with self.subTest(name=name):
                self.register(PURE_MIDDLEWARE + '\ngetattr(object(), ' + repr(name) + ')')
                with self.assertRaisesRegex(ValueError, 'public hook fields'):
                    validate_workspace(self.workspace, self.policy)

    def test_completed_evolve_cache_avoids_api_and_rejects_mutation_or_unsafe_candidate(self):
        for tamper in (None, 'input', 'candidate', 'result', 'unsafe'):
            with self.subTest(tamper=tamper), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                host, private = root / 'host', root / 'case/experiment'
                shutil.copytree(HERE / 'baselines/alfworld', host / 'workspace')
                shutil.copytree(host / 'workspace', private / 'workspace')
                (private / 'workspace/systemprompt.md').write_text('{{ native_instructions }}\nCandidate fixture.')
                if tamper == 'unsafe':
                    (private / 'workspace/tools/native.py').write_text('forbidden tool change')
                evolve = private / 'runs/iteration_002/evolve'
                evolve.mkdir(parents=True)
                write_json(evolve / 'evolve_trace.json', {'original_paid': True})
                result = private.parent / 'evolve_result.json'
                write_json(result, {'result': 'original paid result'})
                cache = {'iteration': 2, 'private_experiment': str(private),
                    'input_workspace_sha256': workspace_hash(host / 'workspace'),
                    'candidate_workspace_sha256': workspace_hash(private / 'workspace'),
                    'files': {'evolve_result.json': sha(result),
                              str((evolve / 'evolve_trace.json').relative_to(private.parent)): sha(evolve / 'evolve_trace.json')}}
                if tamper in ('input', 'candidate'):
                    where = host if tamper == 'input' else private
                    (where / 'workspace/systemprompt.md').write_text('{{ native_instructions }}\nChanged later.')
                elif tamper == 'result':
                    write_json(result, {'result': 'changed paid result'})
                module = SimpleNamespace(run_evolve_agent=MagicMock())
                bind_isolated_evolve(module)
                config = config_for('alfworld', [0], 6, 'unused', debugger_tasks=10,
                                    capability_policy=restricted_policy('alfworld'))
                config['native_backend']['reuse_completed_evolve'] = cache
                iteration = host / 'runs/iteration_002'
                with patch('ahe_native.evolve_isolation.subprocess.run') as api, \
                     patch('ahe_native.evolve_isolation.case_root') as isolation:
                    if tamper:
                        with self.assertRaises((RuntimeError, ValueError)):
                            module.run_evolve_agent(config, host, 2, 'not resent', root / 'job', iteration)
                    else:
                        self.assertEqual(module.run_evolve_agent(config, host, 2, 'not resent', root / 'job', iteration),
                                         'original paid result')
                        self.assertEqual(workspace_hash(host / 'workspace'), cache['candidate_workspace_sha256'])
                        self.assertFalse(json.loads((iteration / 'evolve/completed_cache_receipt.json').read_text())['optimizer_replayed'])
                    api.assert_not_called()
                    isolation.assert_not_called()

    def test_pure_date_helpers_allowed_but_private_imports_remain_forbidden(self):
        for helper in ('import datetime\nday = datetime.datetime.strptime("2026-10-05", "%Y-%m-%d")\n',
                       'from datetime import datetime, timedelta\nday = datetime(2026, 10, 5) + timedelta(days=1)\n'):
            self.register(helper + PURE_MIDDLEWARE)
            self.assertTrue(validate_workspace(self.workspace, self.policy)["candidate_valid"])
        for helper in ('from datetime import sys\n', 'from datetime import __builtins__\n',
                       'import datetime\nx = datetime.__dict__\n'):
            with self.subTest(helper=helper):
                self.register(helper + PURE_MIDDLEWARE)
                with self.assertRaises(ValueError):
                    validate_workspace(self.workspace, self.policy)

    def test_literal_data_parser_allowed_without_code_execution(self):
        import ast
        for helper in ('import ast\nvalue = ast.literal_eval("{\'ok\': [1, None]}")\n',
                       'from ast import literal_eval\nvalue = literal_eval("[1, 2]")\n'):
            self.register(helper + PURE_MIDDLEWARE)
            self.assertTrue(validate_workspace(self.workspace, self.policy)['candidate_valid'])
        self.assertEqual(ast.literal_eval("{'ok': [1, None]}"), {'ok': [1, None]})
        with self.assertRaises(ValueError):
            ast.literal_eval("__import__('os').system('true')")
        for helper in ('from ast import __builtins__\n', 'from ast import parse\n',
                       'import ast\nast.parse("source")\n', 'import ast\nast.__dict__\n',
                       'import ast\nast.main()\n', 'import ast as parser\nparser.main()\n'):
            with self.subTest(helper=helper):
                self.register(helper + PURE_MIDDLEWARE)
                with self.assertRaises(ValueError):
                    validate_workspace(self.workspace, self.policy)

    def test_pure_helpers_and_public_message_action_interfaces_remain_available(self):
        (self.workspace / "middleware/helper.py").write_text("def repair(action):\n    return dict(action)\n")
        source = PURE_MIDDLEWARE.replace("from nexau.archs", "from middleware.helper import repair\n"
            "from nexau.core.messages import Message, Role, TextBlock\n"
            "from nexau.archs.main_sub.execution.parse_structures import ToolCall, ParsedResponse\n"
            "from nexau.archs").replace("dict(hook_input.tool_input)", "repair(hook_input.tool_input)")
        self.register(source)
        self.assertTrue(validate_workspace(self.workspace, self.policy)["candidate_valid"])
        (self.workspace / "middleware/helper.py").write_text("from openai import OpenAI\n")
        with self.assertRaisesRegex(ValueError, "import"):
            validate_workspace(self.workspace, self.policy)

    def test_tool_schema_and_native_instructions_are_frozen(self):
        path = self.workspace / "tool_descriptions/take_action.tool.yaml"
        original = path.read_text()
        data = yaml.safe_load(original)
        data["name"] = "new_tool"
        path.write_text(yaml.safe_dump(data))
        with self.assertRaisesRegex(ValueError, "descriptions"):
            validate_workspace(self.workspace, self.policy)
        path.write_text(original)
        (self.workspace / "systemprompt.md").write_text("Instructions replaced.")
        with self.assertRaisesRegex(ValueError, "native instructions"):
            validate_workspace(self.workspace, self.policy)

    def test_parameter_descriptions_are_editable_without_changing_schema(self):
        for domain in ("alfworld", "dbbench", "webshop", "airline", "retail", "telecom"):
            with self.subTest(domain=domain):
                root = self.root / domain
                shutil.copytree(HERE / "baselines" / domain, root)
                policy = restricted_policy(domain)
                for name, reference in policy["reference_descriptions"].items():
                    data = copy.deepcopy(reference)
                    for field in data.get("input_schema", {}).get("properties", {}).values():
                        field["description"] = "Candidate-generated parameter guidance."
                        if isinstance(field.get("items"), dict):
                            field["items"]["description"] = "Candidate-generated item guidance."
                    (root / name).write_text(yaml.safe_dump(data))
                self.assertTrue(validate_workspace(root, policy)["candidate_valid"])

    def test_parameter_descriptions_do_not_hide_contract_edits(self):
        path = self.workspace / "tool_descriptions/take_action.tool.yaml"
        reference = copy.deepcopy(self.policy["reference_descriptions"][str(path.relative_to(self.workspace))])
        reference["input_schema"] = {"type": "object", "properties": {
            "description": {"type": "string", "description": "An actual parameter named description."},
            "actions": {"type": "array", "items": {"type": "string", "enum": ["look", "walk"]}},
            "payload": {"type": "object", "default": {"description": "default value, not documentation"}}},
            "required": ["description"], "additionalProperties": False}
        self.policy["reference_descriptions"][str(path.relative_to(self.workspace))] = reference
        cases = ("parameter_removed", "parameter_type", "enum", "required", "default", "dispatch", "nontext")
        for case in cases:
            with self.subTest(case=case):
                data = copy.deepcopy(reference)
                data["input_schema"]["description"] = "Editable schema documentation."
                if case == "parameter_removed":
                    del data["input_schema"]["properties"]["description"]
                elif case == "parameter_type":
                    data["input_schema"]["properties"]["description"]["type"] = "integer"
                elif case == "enum":
                    data["input_schema"]["properties"]["actions"]["items"]["enum"].append("new_action")
                elif case == "required":
                    data["input_schema"]["required"] = []
                elif case == "default":
                    data["input_schema"]["properties"]["payload"]["default"]["description"] = "changed value"
                elif case == "dispatch":
                    data["implementation"] = "new.dispatch"
                else:
                    data["input_schema"]["properties"]["description"]["description"] = {"malformed": True}
                path.write_text(yaml.safe_dump(data))
                with self.assertRaisesRegex(ValueError, "descriptions"):
                    validate_workspace(self.workspace, self.policy)

    def test_workspace_symlink_rejected(self):
        (self.workspace / "middleware/link.py").symlink_to(self.workspace / "tools/native.py")
        with self.assertRaisesRegex(ValueError, "symlink"):
            validate_workspace(self.workspace, self.policy)

    def test_restricted_config_separates_ten_analyses_from_fifty_evaluations(self):
        policy = restricted_policy("dbbench")
        config = config_for("dbbench", list(range(50)), 9, "http://127.0.0.1:1/v1",
                            debugger_tasks=10, concurrency=50, capability_policy=policy)
        self.assertEqual(config["agent_debugger"]["max_tasks"], 10)
        self.assertEqual(config["agent_debugger"]["retry_attempts"], 1)
        self.assertEqual(len(config["native_backend"]["task_ids"]), 50)
        self.assertEqual(config["native_backend"]["capability_policy"], policy)
        with self.assertRaisesRegex(ValueError, "ten"):
            config_for("dbbench", list(range(50)), 9, "unused", debugger_tasks=50, capability_policy=policy)
        self.assertIn("overrides wider original permissions", constraint_text(policy))

    def test_debugger_projection_uses_actual_upstream_subconfig_boundary(self):
        config = config_for("alfworld", list(range(50)), 6, "unused", debugger_tasks=10,
                            capability_policy=self.policy)
        (self.root / "config_snapshot.yaml").write_text(yaml.safe_dump(config))
        adb_only = config["agent_debugger"]
        actual = evidence_config(adb_only, self.root, 1)
        self.assertEqual(actual["native_capability_policy"], self.policy)
        self.assertIs(actual["agent_debugger"], adb_only)
        self.assertEqual(actual["harbor"]["k"], 1)
        self.assertEqual(evidence_config(config, self.root, 1)["native_capability_policy"], self.policy)
        changed = {**adb_only, "max_tasks": 50}
        self.assertEqual(evidence_config(changed, self.root, 1)["agent_debugger"]["max_tasks"], 50)
        # Projection itself rejects this rather than silently substituting a
        # cheap snapshot config while executing a more expensive original one.
        (self.root / "config_snapshot.yaml").unlink()
        self.assertIsNone(evidence_config(adb_only, self.root, 1))

    def test_host_request_rejects_tampered_policy_before_any_model_call(self):
        ids = json.loads(MANIFEST.read_text())["domains"]["alfworld"]["batches"][0]["indices"][:1]
        request = {"domain": "alfworld", "split": "train", "task_ids": ids, "k": 1,
            "concurrency": 1, "workspace": str(self.workspace), "agent_config": "code_agent.yaml",
            "capability_policy": copy.deepcopy(self.policy)}
        self.assertEqual(validate_request(request), [str(ids[0])])
        request["capability_policy"]["frozen_files"] = {}
        with self.assertRaisesRegex(ValueError, "host-pinned"):
            validate_request(request)

    def test_transport_carries_policy_without_changing_original_loop(self):
        module, _ = load_upstream()
        loop, debugger, evolve = module.run_single_experiment, module.run_parallel_adb_ask, module.run_evolve_agent
        config = config_for("alfworld", [0], 9, "unused", debugger_tasks=10, capability_policy=self.policy)
        output = self.root / "runs/iteration_001/input/benchmark"
        output.mkdir(parents=True)
        bind_native_transport(module)
        with patch("ahe_native.upstream.subprocess.Popen", return_value=MagicMock()):
            module.launch_harbor(config, self.workspace, "code_agent.yaml", output)
        request = json.loads((output / "native_request.json").read_text())
        self.assertEqual(request["capability_policy"], self.policy)
        self.assertIs(module.run_single_experiment, loop)
        self.assertIs(module.run_parallel_adb_ask, debugger)
        self.assertIs(module.run_evolve_agent, evolve)

    def test_old_tool_augmented_candidates_are_rejected_not_silently_reused(self):
        path = HERE / "experiments/ahe_agentbench_nine_rotating_flash8_rpm48_20261003_alfworld/workspace"
        if not path.exists():
            self.skipTest("historical candidate is not present")
        before = (path / "tools/native.py").read_bytes()
        with self.assertRaisesRegex(ValueError, "immutable native file"):
            validate_workspace(path, self.policy)
        self.assertEqual(before, (path / "tools/native.py").read_bytes())

    def test_runtime_guard_allows_one_ordinary_call_and_blocks_auxiliary_calls(self):
        create = MagicMock(return_value="model result")
        manager = SimpleNamespace(middlewares=[])
        agent = SimpleNamespace(openai_client=SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=create))), executor=SimpleNamespace(middleware_manager=manager),
            tool_registry={"take_action": object()}, config=SimpleNamespace(sub_agents={}))
        channel = SimpleNamespace()
        budget = install_runtime_guard(agent, channel, self.policy)
        guard = manager.middlewares[0]
        with self.assertRaises(CapabilityViolation):
            agent.openai_client.chat.completions.create()
        self.assertEqual(create.call_count, 0)
        result = guard.wrap_model_call(None, lambda params: agent.openai_client.chat.completions.create())
        self.assertEqual(result, "model result")
        self.assertEqual(create.call_count, 1)
        def two_calls(params):
            agent.openai_client.chat.completions.create()
            return agent.openai_client.chat.completions.create()
        with self.assertRaises(CapabilityViolation):
            guard.wrap_model_call(None, two_calls)
        self.assertFalse(budget.model_active)

    def test_runtime_guard_rejects_unknown_registry_before_execution(self):
        agent = SimpleNamespace(tool_registry={"take_action": None, "new_tool": None},
                                config=SimpleNamespace(sub_agents={}))
        with self.assertRaises(CapabilityViolation):
            install_runtime_guard(agent, SimpleNamespace(), self.policy)

    def test_runtime_guard_preserves_framework_error_retry_but_not_extra_generations(self):
        create = MagicMock(side_effect=[RuntimeError("fixture transport failure"), "retry result"])
        manager = SimpleNamespace(middlewares=[])
        agent = SimpleNamespace(openai_client=SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=create))), executor=SimpleNamespace(middleware_manager=manager),
            tool_registry={"take_action": object()}, config=SimpleNamespace(sub_agents={}))
        install_runtime_guard(agent, SimpleNamespace(), self.policy)
        def ordinary_retry(params):
            try:
                agent.openai_client.chat.completions.create()
            except RuntimeError:
                return agent.openai_client.chat.completions.create()
        self.assertEqual(manager.middlewares[0].wrap_model_call(None, ordinary_retry), "retry result")
        self.assertEqual(create.call_count, 2)
        with self.assertRaises(CapabilityViolation):
            agent.openai_client.chat.completions.create()

    def test_rpc_guard_blocks_unapproved_and_multiple_invokes_before_dispatch(self):
        import threading
        channel = Channel.__new__(Channel)
        channel.reader = io.StringIO('{"kind":"feedback","messages":[{"role":"tool","content":"ok"}]}\n')
        channel.writer = io.StringIO()
        channel.lock = threading.RLock()
        channel.counter = 0
        budget = DispatchBudget(["take_action"])
        channel.dispatch_budget = budget
        with self.assertRaises(CapabilityViolation):
            channel.invoke("take_action", {})
        self.assertEqual(channel.writer.getvalue(), "")
        budget.tool_active, budget.tool_credit = "take_action", 1
        self.assertEqual(channel.invoke("take_action", {"action": "look"}), "ok")
        with self.assertRaises(CapabilityViolation):
            channel.invoke("take_action", {})
        self.assertEqual(channel.counter, 1)

    def test_real_nexau_baseline_initializes_guard_without_api_call(self):
        from nexau import Agent
        for domain in ("alfworld", "dbbench", "webshop", "airline", "retail", "telecom"):
            workspace = HERE / "baselines" / domain
            with self.subTest(domain=domain), patch.object(sys, "path", [str(workspace), *sys.path]), \
                    patch.dict(sys.modules), patch.dict("os.environ", LLM_MODEL="qwen3-4b",
                        LLM_BASE_URL="http://127.0.0.1:1/v1", LLM_API_KEY="EMPTY"):
                for name in list(sys.modules):
                    if name == "tools" or name.startswith("tools."):
                        del sys.modules[name]
                agent = Agent.from_yaml(workspace / "code_agent.yaml")
                budget = install_runtime_guard(agent, SimpleNamespace(), restricted_policy(domain))
                self.assertFalse(budget.model_active)
                with self.assertRaises(CapabilityViolation):
                    agent.openai_client.chat.completions.create(model="qwen3-4b", messages=[])

    def test_real_nexau_planner_and_guard_complete_ordinary_model_tool_loop_offline(self):
        import threading
        from nexau import Agent
        from openai.types.chat import ChatCompletion
        from ahe_native import environment_rpc
        self.register()
        channel = Channel.__new__(Channel)
        channel.reader = io.StringIO('{"kind":"feedback","messages":[{"role":"tool","content":"fixture observation"}]}\n')
        channel.writer, channel.lock, channel.counter = io.StringIO(), threading.RLock(), 0
        response = lambda message, reason: ChatCompletion.model_validate({"id": "offline", "object": "chat.completion",
            "created": 0, "model": "qwen3-4b", "choices": [{"index": 0, "message": message, "finish_reason": reason}]})
        fixture = MagicMock(side_effect=[response({"role": "assistant", "content": None,
            "tool_calls": [{"id": "fixture_action", "type": "function", "function": {
                "name": "take_action", "arguments": '{"action":"bad"}'}}]}, "tool_calls"),
            response({"role": "assistant", "content": "fixture complete"}, "stop")])
        # Candidate modules are process-local in production; isolate their
        # import cache here too, so one fixture cannot select another workspace.
        with patch.object(sys, "path", [str(self.workspace), *sys.path]), patch.dict(sys.modules), patch.dict(
                "os.environ", LLM_MODEL="qwen3-4b", LLM_BASE_URL="http://127.0.0.1:1/v1", LLM_API_KEY="EMPTY"):
            for name in list(sys.modules):
                if name == "middleware" or name.startswith("middleware.") or name == "tools" or name.startswith("tools."):
                    del sys.modules[name]
            agent = Agent.from_yaml(self.workspace / "code_agent.yaml")
            with patch.object(agent.openai_client.chat.completions, "create", fixture), patch.object(environment_rpc, "CHANNEL", channel):
                budget = install_runtime_guard(agent, channel, self.policy)
                result = agent.run(message="Offline fixture, not a benchmark task.",
                    context={"native_instructions": "Fixture instruction.", "working_directory": str(self.workspace)})
        self.assertIn("fixture complete", str(result))
        self.assertEqual(fixture.call_count, 2)
        self.assertEqual(channel.counter, 1)
        self.assertEqual(json.loads(channel.writer.getvalue())["arguments"]["action"], "look")
        self.assertFalse(budget.model_active)
        self.assertIsNone(budget.tool_active)


class PromotionBoundaryTests(unittest.TestCase):
    def test_invalid_candidate_is_kept_private_and_never_promoted(self):
        self.exercise_candidate(allowed=False)

    def test_allowed_candidate_promotes_only_after_host_validation(self):
        self.exercise_candidate(allowed=True)

    def exercise_candidate(self, allowed):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            host = root / "host"
            shutil.copytree(HERE / "baselines/alfworld", host / "workspace")
            (host / "evolve_agent").mkdir()
            original_connection = {"llm_config": {"api_key": "EMPTY", "base_url": "http://localhost:999/v1",
                                   "model": "deepseek-v4-flash", "max_tokens": 32000},
                                   "system_prompt": "unchanged", "max_iterations": 500}
            (host / "evolve_agent/evolve_agent.yaml").write_text(yaml.safe_dump(original_connection))
            iteration = host / "runs/iteration_001"
            job = iteration / "input/benchmark"
            job.mkdir(parents=True)
            runtime = root / "runtime"
            case = runtime / "episodes" / ("b" * 32)
            case.mkdir(parents=True)
            module = SimpleNamespace(run_evolve_agent=MagicMock())
            original = module.run_evolve_agent
            bind_isolated_evolve(module)
            self.assertIs(module.run_evolve_agent.__wrapped__, original)
            config = config_for("alfworld", [0], 1, "unused", debugger_tasks=10,
                                capability_policy=restricted_policy("alfworld"))
            before = workspace_hash(host / "workspace")
            def fake_original_process(*args, **kwargs):
                from ahe_native.token_accounting import decode_scope
                agent = yaml.safe_load((case / "experiment/evolve_agent/evolve_agent.yaml").read_text())
                scope = decode_scope(agent["llm_config"]["base_url"].replace("http://127.0.0.1:1", "") + "/chat/completions")
                self.assertEqual(scope, {"experiment": "host", "domain": "alfworld", "round": 1, "role": "evolve"})
                self.assertEqual(agent["llm_config"]["api_key"], "EMPTY")
                agent["llm_config"]["base_url"] = original_connection["llm_config"]["base_url"]
                self.assertEqual(agent, original_connection)
                self.assertEqual(yaml.safe_load((host / "evolve_agent/evolve_agent.yaml").read_text()), original_connection)
                workspace = case / "experiment/workspace"
                (case / "experiment/runs/iteration_001/evolve").mkdir(parents=True, exist_ok=True)
                if allowed:
                    with (workspace / "systemprompt.md").open("a") as stream:
                        stream.write("\nFixture candidate prompt improvement.")
                else:
                    (workspace / "tools/native.py").write_text("# prohibited candidate tool edit")
                write_json(case / "evolve_result.json", {"result": "fixture result"})
                return SimpleNamespace(returncode=0)
            with patch("ahe_native.evolve_isolation.case_root", return_value=(runtime, case, "/episodes/" + "b" * 32, 999)), \
                    patch("ahe_native.evolve_isolation.copy_unchanged_or_atomic"), \
                    patch("ahe_native.evolve_isolation.own_tree"), \
                    patch("ahe_native.evolve_isolation.subprocess.run", side_effect=fake_original_process), \
                    patch.dict("os.environ", AHE_LOCAL_BROKER_URL="http://127.0.0.1:1/v1"):
                if allowed:
                    self.assertEqual(module.run_evolve_agent(config, host, 1, "fixture query", job, iteration), "fixture result")
                else:
                    with self.assertRaisesRegex(ValueError, "immutable native file"):
                        module.run_evolve_agent(config, host, 1, "fixture query", job, iteration)
            if allowed:
                self.assertNotEqual(workspace_hash(host / "workspace"), before)
                self.assertTrue((iteration / "evolve/capability_validation.json").exists())
            else:
                self.assertEqual(workspace_hash(host / "workspace"), before)
                self.assertFalse(json.loads((iteration / "evolve/capability_rejection.json").read_text())["candidate_promoted"])
                self.assertEqual((case / "experiment/workspace/tools/native.py").read_text(), "# prohibited candidate tool edit")


class EvidenceProjectionTests(unittest.TestCase):
    def test_projection_preserves_host_fifty_scores_and_only_exposes_current_ten_traces(self):
        module, _ = load_upstream()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            host = root / "host"
            (host / "workspace").mkdir(parents=True)
            for iteration in (1, 2):
                records = [{"task_id": str(i), "trial": 0, "success": int(i >= 30),
                    "messages": [{"role": "user", "content": f"public task {i}"}]} for i in range(50)]
                export_job({"schema_version": 1, "domain": "alfworld", "split": "train", "records": records},
                    host / "runs" / f"iteration_{iteration:03d}" / "input/benchmark/2026-10-03__00-00-00")
                if iteration == 1:
                    export_job({"schema_version": 1, "domain": "alfworld", "split": "train", "records": records},
                        host / "runs/iteration_001/input/benchmark/2026-10-03__00-00-01")
                archive = host / "runs" / f"iteration_{iteration:03d}" / "input/analysis_before_rpm_guard_archive"
                archive.mkdir()
                (archive / "old.md").write_text("unrestricted old details")
                diagnostics = host / "runs" / f"iteration_{iteration:03d}" / "debugger/subprocess_diagnostics"
                diagnostics.mkdir(parents=True)
                (diagnostics / "old.json").write_text("old debugger output")
            experiment = root / "episodes" / ("a" * 32) / "experiment"
            shutil.copytree(host, experiment)
            config = config_for("alfworld", list(range(50)), 9, "unused", debugger_tasks=10,
                                capability_policy=restricted_policy("alfworld"))
            receipt = project_public_evidence(experiment, config, module, 2)
            self.assertEqual([row["detailed_tasks"] for row in receipt["iterations"]], [0, 10])
            current = experiment / "runs/iteration_002/input/benchmark/2026-10-03__00-00-00"
            self.assertEqual(len(list(current.glob("native_*/agent/*.json"))), 10)
            self.assertEqual(module.compute_stats(current)["n_total"], 50)
            self.assertEqual(module.compute_stats(current)["pass_rate"], 0.4)
            self.assertTrue(all(module.compute_stats(current)["task_results"][name] == "fail"
                                for name in receipt["iterations"][1]["selected_tasks"]))
            self.assertFalse(list(experiment.glob("runs/*/debugger")))
            self.assertFalse(list(experiment.glob("runs/*/input/analysis_*")))
            self.assertFalse(list(experiment.glob("runs/iteration_001/input/benchmark/*/native_*/agent/*.json")))
            for job in (experiment / "runs/iteration_001/input/benchmark").iterdir():
                self.assertEqual(module.compute_stats(job)["n_total"], 50)
                self.assertEqual(module.compute_stats(job)["pass_rate"], 0.4)
            self.assertEqual(len(list(host.glob("runs/*/input/benchmark/*/native_*/agent/*.json"))), 150)
            with self.assertRaisesRegex(ValueError, "disposable"):
                project_public_evidence(host, config, module, 2)
            shutil.copytree(current, current.with_name("2026-10-03__00-00-01"))
            with self.assertRaisesRegex(ValueError, "one native job"):
                project_public_evidence(experiment, config, module, 2)


if __name__ == "__main__":
    unittest.main()
