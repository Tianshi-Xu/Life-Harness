"""Offline-only machinery tests; synthetic rules are never optimizer priors."""
import copy
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from evolution.agentbench_hooks.runtime import object_digest
from evolution.life_optimizer import profile as p

TOOLS = [{"type": "function", "function": {"name": "act", "description": "native",
    "parameters": {"type": "object", "properties": {"value": {"type": "string"}},
                   "required": ["value"], "additionalProperties": False}}}]
RAW = {"role": "assistant", "content": "unchanged", "tool_calls": [
    {"id": "c1", "type": "function", "function": {"name": "act", "arguments": '{"value":"native"}'}}]}
IDENTITY = "TARGET_LAYER='h2'\nclass Harness:\n    def h2(self,message,history):\n        return message\n"


class OptimizerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.seed = self.root / "seed"
        self.eval = self.root / "eval"
        self.workspace = self.root / "workspace"
        self.candidate = self.root / "candidate"
        self.seed.mkdir(); self.eval.mkdir(); self.candidate.mkdir()
        self.source = self.seed / "versions/initial/identity.py"
        self.source.parent.mkdir(parents=True)
        self.source.write_text(IDENTITY)
        records = [{"mechanism_id": "identity", "revision": 1, "target_layer": "h2",
                    "artifact_path": "versions/initial/identity.py", "sha256": p.digest(self.source)}]
        self.state = {"domain": "webshop", "active_snapshot": records,
            "active_snapshot_sha256": object_digest(records), "round": 6,
            "rounds": [{"round": 1, "accepted": True, "bundle_hypothesis": "historical hypothesis",
                        "test_accuracy": "PRIVATE_SENTINEL", "raw_trajectories": "PRIVATE_SENTINEL"}]}
        p.write_json(self.seed / "state.json", self.state)
        outcomes = {}
        for index in range(20):
            success = int(index % 3 == 0)
            outcomes[str(index)] = success
            history = [{"role": "system", "content": "native system", "private": "PRIVATE_SENTINEL"},
                {"role": "user", "content": "Find a synthetic object"}, RAW,
                {"role": "tool", "tool_call_id": "c1", "content": "ok" if success else "invalid action"}]
            row = {"index": index, "success": success, "status": "completed", "turns": 1,
                   "infrastructure_error": index == 1, "history": history, "tokens": 10,
                   "task_result_private": "PRIVATE_SENTINEL", "rewards": "PRIVATE_SENTINEL",
                   "calls": [{"turn": 1, "request": {"messages": history[:2], "tools": TOOLS,
                              "api_key": "PRIVATE_SENTINEL"}, "response": {"choices": [{"message": RAW}],
                              "private": "PRIVATE_SENTINEL"}}]}
            p.write_json(self.eval / f"episode_{index}.json", row)
        self.summary = {"split": "train", "domain": "webshop", "n": 20, "per_task": outcomes,
            "successes": sum(outcomes.values()), "infra_errors": 1,
            "chain_hashes": {"artifacts/identity.py": records[0]["sha256"], "manifest": "ignored"}}
        p.write_json(self.eval / "summary.json", self.summary)

    def prepare(self, **kwargs):
        return p.prepare(self.seed, self.eval, self.workspace, **kwargs)

    def files(self, source=IDENTITY, layer="h2"):
        cases = p.load_cases(self.workspace)
        inventory = json.loads((self.workspace / "inventory.json").read_text())
        bundle = {"schema_version": 2, "bundle_id": "new_bundle", "domain": inventory["domain"],
            "base_snapshot_sha256": inventory["snapshot_sha256"], "hypothesis": "synthetic machinery test",
            "operations": [{"operation_id": "new_op", "mechanism_id": "new_mechanism", "op": "add",
                "target_layer": layer, "hypothesis": "synthetic", "artifact": "artifacts/new.py"}]}
        review = {"schema_version": 1, "cases": [{"case_id": c["case_id"],
            "first_error_message_index": None if c["success"] else 3,
            "observation": "synthetic public symptom", "root_cause": "uncertain hypothesis",
            "confidence": "low", "public_trigger": "synthetic signal", "operation_ids": ["new_op"],
            "regression_risk": "preserve successful behavior"} for c in cases],
            "rules": [{"rule_id": "identity_rule", "invariant": "preserve identity",
                "layer_contracts": {l: "unchanged identity" for l in ("h2", "h3", "h4", "h5")},
                "operation_ids": ["new_op"], "test_ids": ["trigger_test", "control_test"]}], "shared_state_contract": []}
        checks = {"schema_version": 1, "tests": [{"test_id": name, "rule_id": "identity_rule", "kind": kind,
            "steps": [{"hook": "h2", "args": [RAW, []], "expected": RAW}]}
            for name, kind in (("trigger_test", "trigger"), ("control_test", "counterexample"))]}
        return {"bundle.json": json.dumps(bundle), "diagnosis.json": json.dumps(review),
                "checks.json": json.dumps(checks), "artifacts/new.py": source}

    def write_candidate(self, files=None):
        for name, content in (files or self.files()).items():
            path = self.candidate / name
            path.parent.mkdir(exist_ok=True)
            path.write_text(content)

    def mutate(self, file, change):
        path = self.candidate / file
        value = json.loads(path.read_text()); change(value); p.write_json(path, value)

    def report(self):
        return p.check_candidate(self.workspace, self.candidate)

    def test_prepare_exposes_all_public_evidence_and_preserves_history(self):
        before = p.digest(self.seed / "state.json")
        self.prepare()
        cases = p.load_cases(self.workspace)
        self.assertEqual(len(cases), 20)
        self.assertTrue(any(c["success"] for c in cases))
        self.assertTrue(next(c for c in cases if c["case_id"] == "1")["infrastructure_error"])
        self.assertNotIn("PRIVATE_SENTINEL", "".join(f.read_text() for f in self.workspace.rglob("*") if f.is_file()))
        summary = json.loads((self.workspace / "summary.json").read_text())
        self.assertEqual((summary["n"], summary["infra_errors"]), (20, 1))
        self.assertEqual(summary["evidence_chars"], sum(len(f.read_text()) for f in (self.workspace / "evidence").glob("*.json")))
        self.assertIsNone(summary["case_limit"])
        self.assertIsNone(summary["evidence_char_limit"])
        self.assertIn("historical hypothesis", (self.workspace / "history.json").read_text())
        self.assertEqual(before, p.digest(self.seed / "state.json"))

    def test_prepare_is_deterministic_and_never_overwrites(self):
        self.prepare(seed=42)
        second = self.root / "second"
        p.prepare(self.seed, self.eval, second, seed=42)
        self.assertEqual((self.workspace / "evidence.json").read_bytes(), (second / "evidence.json").read_bytes())
        with self.assertRaises(ValueError): self.prepare()
        with self.assertRaises(ValueError): p.prepare(self.seed, self.eval, self.seed / "child")

    def test_test_split_wrong_incumbent_and_incomplete_summary_rejected(self):
        for field, value in (("split", "test"), ("chain_hashes", {}), ("n", 1)):
            with self.subTest(field=field):
                summary = {**self.summary, field: value}
                p.write_json(self.eval / "summary.json", summary)
                with self.assertRaises(ValueError): self.prepare()
        p.write_json(self.eval / "summary.json", self.summary)
        p.write_json(self.eval / "episode_2.json", {"index": 2, "success": 1})
        with self.assertRaises(ValueError): self.prepare()

    def test_large_cases_and_more_than_ten_reads_are_not_limited(self):
        large = "public long observation " * 10000
        path = self.eval / "episode_2.json"
        row = json.loads(path.read_text())
        row["history"][1]["content"] = large
        row["calls"][0]["request"]["messages"][1]["content"] = large
        p.write_json(path, row)
        self.prepare()
        index = json.loads((self.workspace / "evidence.json").read_text())
        self.assertEqual(len(index), 20)
        case = p.load_cases(self.workspace, ["2"])[0]
        self.assertIn(large, [m.get("content") for m in case["message_pool"]])
        self.assertGreater(len((self.workspace / "evidence/case_2.json").read_text()), 160000)
        for entry in index[:15]:
            result = p.tool_call(self.workspace, "read_file", {"path": entry["path"]})
            self.assertIn("content", result)
        ledger = json.loads((self.workspace / p.READ_LOG).read_text())
        self.assertEqual(len(ledger["case_ids"]), 15)
        chunks, offset = [], 0
        while True:
            result = p.tool_call(self.workspace, "read_file", {
                "path": "evidence/case_2.json", "start_byte": offset, "max_bytes": 16000})
            chunks.append(result["content"])
            offset = result["next_byte"]
            if result["eof"]:
                break
        self.assertEqual("".join(chunks), (self.workspace / "evidence/case_2.json").read_text())
        self.assertGreater(offset, 160000)
        with self.assertRaises(ValueError): p.tool_call(self.workspace, "write_file", {"path": p.READ_LOG, "content": "{}"})

    def test_diagnosis_must_precede_code_and_inputs_are_read_only(self):
        self.prepare()
        for path in ("bundle.json", "artifacts/new.py", "inventory.json", "../escape.py"):
            with self.assertRaises(ValueError): p.tool_call(self.workspace, "write_file", {"path": path, "content": "x"})
        with self.assertRaises(ValueError): p.tool_call(self.workspace, "Bash", {"path": "checks.json", "content": "x"})
        with self.assertRaises(ValueError): p.tool_call(self.workspace, "read_file", {"path": "../../etc/passwd"})
        p.tool_call(self.workspace, "write_file", {"path": "diagnosis.json", "content": "{}"})
        with self.assertRaises(ValueError): p.tool_call(self.workspace, "write_file", {"path": "artifacts/new.py", "content": IDENTITY})

    def test_real_replay_and_fixtures_pass_without_network(self):
        self.prepare(); self.write_candidate()
        with patch.object(p, "model_router_completion", side_effect=AssertionError("no paid calls")):
            report = self.report()
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["calls_replayed"], len(p.load_cases(self.workspace)))
        self.assertEqual(report["acceptance"], "not evaluated; requires same-batch strict improvement")

    def test_diagnosis_requires_valid_citations_indices_and_case_support(self):
        self.prepare()
        for mutation in (lambda r: r["cases"][0].update(case_id="unknown"),
                         lambda r: r["cases"][0].update(first_error_message_index=999),
                         lambda r: [c.update(operation_ids=[]) for c in r["cases"]]):
            self.write_candidate(); self.mutate("diagnosis.json", mutation)
            self.assertFalse(self.report()["passed"])

    def test_only_cited_subset_needs_analysis_and_replay(self):
        self.prepare(); self.write_candidate()
        self.mutate("diagnosis.json", lambda r: r.update(cases=r["cases"][:2]))
        report = self.report()
        self.assertTrue(report["passed"], report)
        self.assertEqual(len(report["cases"]), 2)
        self.assertEqual(report["calls_replayed"], 2)
        self.assertEqual(len(json.loads((self.workspace / "evidence.json").read_text())), 20)

    def test_rule_requires_counterexample_and_assertion(self):
        self.prepare(); self.write_candidate()
        self.mutate("checks.json", lambda v: v["tests"][1].update(kind="boundary"))
        self.assertFalse(self.report()["passed"])
        self.write_candidate()
        self.mutate("checks.json", lambda v: v["tests"][0]["steps"][0].pop("expected"))
        self.assertFalse(self.report()["passed"])

    def test_invalid_sibling_filtered_and_review_normalized(self):
        self.prepare(); self.write_candidate()
        self.mutate("bundle.json", lambda b: b["operations"].append({**b["operations"][0],
            "operation_id": "bad_op", "mechanism_id": "bad_mechanism", "artifact": "artifacts/bad.py"}))
        (self.candidate / "artifacts/bad.py").write_text(IDENTITY + "\nimport os\n")
        self.mutate("diagnosis.json", lambda r: [c["operation_ids"].append("bad_op") for c in r["cases"]])
        report = self.report()
        self.assertTrue(report["passed"], report)
        self.assertEqual(len(report["rejected_operations"]), 1)
        self.assertNotIn("bad_op", json.dumps(report["diagnosis"]))

    def test_modify_delete_and_stale_cas(self):
        self.prepare()
        for kind in ("modify", "delete"):
            self.write_candidate()
            def update(bundle):
                op = bundle["operations"][0]
                op.update(op=kind, mechanism_id="identity", expected_revision=1,
                          expected_sha256=self.state["active_snapshot"][0]["sha256"])
                if kind == "delete": op.pop("artifact")
            self.mutate("bundle.json", update)
            self.assertTrue(self.report()["passed"], self.report())
            self.mutate("bundle.json", lambda b: b["operations"][0].update(expected_revision=2))
            self.assertFalse(self.report()["passed"])

    def test_fixture_private_context_and_unbounded_steps_rejected(self):
        self.prepare(); self.write_candidate()
        self.mutate("checks.json", lambda c: c["tests"][0].update(context={"hidden_goal": "private"}))
        self.assertFalse(self.report()["passed"])
        self.write_candidate()
        self.mutate("checks.json", lambda c: c["tests"][0].update(steps=c["tests"][0]["steps"] * 41))
        self.assertFalse(self.report()["passed"])

    def test_cross_layer_plan_action_fixture_and_state_reset(self):
        self.prepare()
        files = self.files("""TARGET_LAYER='h2'
class Harness:
    def bind_shared_state(self,state):
        self.state=state
    def h2(self,message,history):
        if 'next' in self.state:
            message['content']=self.state.pop('next')
        return message
""")
        bundle = json.loads(files["bundle.json"])
        bundle["operations"].append({"operation_id": "plan_op", "mechanism_id": "planner", "op": "add",
            "target_layer": "h4", "hypothesis": "synthetic plan", "artifact": "artifacts/plan.py"})
        files["bundle.json"] = json.dumps(bundle)
        files["artifacts/plan.py"] = """TARGET_LAYER='h4'
class Harness:
    def bind_shared_state(self,state):
        self.state=state
    def h4(self,messages,remaining):
        if messages and messages[-1].get('content') == 'trigger':
            self.state['next']='planned'
        return []
"""
        review = json.loads(files["diagnosis.json"])
        for row in review["cases"]: row["operation_ids"].append("plan_op")
        review["rules"][0]["operation_ids"].append("plan_op")
        review["rules"][0]["test_ids"].append("cross_test")
        review["shared_state_contract"] = [{"key": "next", "writers": ["planner"], "readers": ["new_mechanism"],
            "value_contract": "string action consumed once", "reset": "per_episode"}]
        files["diagnosis.json"] = json.dumps(review)
        checks = json.loads(files["checks.json"])
        checks["tests"].append({"test_id": "cross_test", "rule_id": "identity_rule", "kind": "cross_layer", "steps": [
            {"hook": "h4", "args": [[{"role": "user", "content": "trigger"}], 4], "expected_state": {"next": "planned"}},
            {"hook": "h2", "args": [RAW, []], "expected_subset": {"content": "planned"}}]})
        files["checks.json"] = json.dumps(checks)
        self.write_candidate(files)
        self.assertTrue(self.report()["passed"], self.report())
        self.mutate("checks.json", lambda v: v["tests"][-1].update(kind="boundary"))
        self.assertFalse(self.report()["passed"])

    def test_unknown_shared_state_owner_rejected(self):
        self.prepare(); self.write_candidate()
        self.mutate("diagnosis.json", lambda r: r["shared_state_contract"].append({"key": "k", "writers": ["nonexistent"],
            "readers": [], "value_contract": "string", "reset": "per_episode"}))
        self.assertFalse(self.report()["passed"])

    def test_new_native_tools_bad_schema_and_false_assertions_rejected(self):
        self.prepare()
        for update in ("message['tool_calls'][0]['function']['name']='invented'",
                       "message['tool_calls'][0]['function']['arguments']='{\"value\":3}'"):
            source = "TARGET_LAYER='h2'\nclass Harness:\n    def h2(self,message,history):\n        " + update + "\n        return message\n"
            self.write_candidate(self.files(source))
            self.assertFalse(self.report()["passed"])
        self.write_candidate()
        self.mutate("checks.json", lambda c: c["tests"][0]["steps"][0].update(expected={"role": "assistant", "content": "false"}))
        self.assertFalse(self.report()["passed"])

    def test_h3_schema_violation_rejected_in_worker(self):
        self.prepare()
        source = "TARGET_LAYER='h3'\nclass Harness:\n    def h3(self,tools):\n        tools[0]['function']['name']='invented'\n        return tools\n"
        self.write_candidate(self.files(source, layer="h3"))
        self.assertFalse(self.report()["passed"])

    def test_worker_time_limit_blocks_hanging_hook(self):
        self.prepare()
        source = "TARGET_LAYER='h2'\nclass Harness:\n    def h2(self,message,history):\n        while True:\n            pass\n"
        self.write_candidate(self.files(source))
        report = self.report()
        self.assertFalse(report["passed"])
        self.assertIn("worker", report["error"])

    def test_worker_audit_and_environment_defense_independent_of_ast(self):
        self.prepare()
        cases = p.load_cases(self.workspace)[:1]
        for source, passed in (("import os\nos.open('/etc/passwd',os.O_RDONLY)\n" + IDENTITY, False),
                               ("import os\nos.system('true')\n" + IDENTITY, False),
                               ("import os\nassert os.environ.get('MR_API_KEY') is None\n" + IDENTITY, True)):
            payload = {"domain": "webshop", "cases": cases, "tests": [],
                "artifacts": [{"name": "guard", "layer": "h2", "source": source}]}
            worker = getattr(p, "WORKER_PATH", p.Path(p.__file__).with_name("worker.py"))
            result = subprocess.run([sys.executable, "-I", "-S", "-B", str(worker)],
                input=json.dumps(payload), text=True, capture_output=True, timeout=10, env={})
            self.assertEqual(json.loads(result.stdout)["passed"], passed, result.stdout)

    def test_all_three_domains_identity_replay_and_public_db_context(self):
        for domain in ("alfworld", "dbbench", "webshop"):
            with self.subTest(domain=domain):
                self.state["domain"] = self.summary["domain"] = domain
                p.write_json(self.seed / "state.json", self.state)
                p.write_json(self.eval / "summary.json", self.summary)
                db = self.root / "db.jsonl"
                db.write_text("\n".join(json.dumps({"description": "public", "type": ["SELECT"], "table": {"a": []},
                                                "sql": "PRIVATE_SENTINEL", "answer_md5": "PRIVATE_SENTINEL"}) for _ in range(20)))
                self.workspace = self.root / ("workspace_" + domain)
                self.prepare(db_source=db)
                self.write_candidate()
                self.assertTrue(self.report()["passed"], self.report())
                cases = p.load_cases(self.workspace)
                self.assertNotIn("PRIVATE_SENTINEL", json.dumps(cases))
                if domain == "dbbench": self.assertEqual(set(cases[0]["context"]), {"description", "type", "table"})

    def test_one_mocked_proposer_session_cache_and_usage(self):
        self.prepare()
        files = self.files()
        first = json.loads((self.workspace / "evidence.json").read_text())[0]
        calls = [{"id": "read_case", "type": "function", "function": {"name": "read_file",
            "arguments": json.dumps({"path": first["path"]})}}]
        for name in ("diagnosis.json", "bundle.json", "artifacts/new.py", "checks.json"):
            calls.append({"id": name, "type": "function", "function": {"name": "write_file",
                "arguments": json.dumps({"path": name, "content": files[name]})}})
        calls.append({"id": "check", "type": "function", "function": {"name": "check_candidate", "arguments": "{}"}})
        response = lambda message: {"choices": [{"message": message}],
                                    "usage": {"prompt_tokens": 80, "completion_tokens": 20, "total_tokens": 100}}
        completion = Mock(side_effect=[response({"role": "assistant", "tool_calls": calls}),
                                      response({"role": "assistant", "content": "done"})])
        output = self.root / "optimized"
        path = p.propose(self.workspace, output, completion=completion)
        self.assertEqual(completion.call_count, 2)
        self.assertTrue(p.check_candidate(self.workspace, output)["passed"])
        receipt = json.loads((output / "validation.json").read_text())
        self.assertEqual((receipt["proposer_requests"], receipt["usage"]["total_tokens"]), (2, 200))
        self.assertEqual(receipt["evidence_access"]["case_ids"], [first["case_id"]])
        self.assertEqual(p.propose(self.workspace, output, completion=Mock(side_effect=AssertionError("cached"))), path)
        (output / "artifacts/new.py").write_text(IDENTITY + "\n# changed")
        with self.assertRaises(ValueError): p.propose(self.workspace, output, completion=completion)

    def test_proposer_budget_and_frozen_input_tampering(self):
        self.prepare()
        completion = Mock(return_value={"choices": [{"message": {"role": "assistant", "content": "no files"}}]})
        with self.assertRaises(ValueError): p.propose(self.workspace, self.root / "bad", completion=completion, max_turns=1)
        self.assertEqual(completion.call_count, 1)
        (self.workspace / "summary.json").write_text("{}")
        with self.assertRaises(ValueError): p.verify_inputs(self.workspace)

    def test_persistent_session_resumes_completed_requests_and_keeps_reasoning(self):
        self.prepare()
        files = self.files()
        first = json.loads((self.workspace / 'evidence.json').read_text())[0]
        calls = [{'id': 'read', 'type': 'function', 'function': {'name': 'read_file',
                 'arguments': json.dumps({'path': first['path']})}}]
        for name in ('diagnosis.json', 'bundle.json', 'artifacts/new.py', 'checks.json'):
            calls.append({'id': name, 'type': 'function', 'function': {'name': 'write_file',
                'arguments': json.dumps({'path': name, 'content': files[name]})}})
        first_response = {'choices': [{'message': {'role': 'assistant', 'tool_calls': calls,
                          'reasoning_content': 'actual provider reasoning'}}],
                          'usage': {'prompt_tokens': 80, 'completion_tokens': 20, 'total_tokens': 100}}
        output, checkpoint = self.root/'resumed', self.root/'checkpoint'
        original = p.save_agent
        def pause(path, sandbox, **values):
            original(path, sandbox, **values)
            if not values['inflight'] and values['requests'] == 1:
                raise RuntimeError('synthetic safe pause after request')
        with patch.object(p, 'save_agent', side_effect=pause):
            with self.assertRaisesRegex(RuntimeError, 'safe pause'):
                p.propose(self.workspace, output, completion=Mock(return_value=first_response), checkpoint_dir=checkpoint)
        def finish(messages, tools):
            assistant = next(m for m in messages if m['role'] == 'assistant')
            self.assertEqual(assistant['reasoning_content'], 'actual provider reasoning')
            return {'choices': [{'message': {'role': 'assistant', 'content': 'done'}}],
                    'usage': {'prompt_tokens': 180, 'completion_tokens': 20, 'total_tokens': 200}}
        completion = Mock(side_effect=finish)
        p.propose(self.workspace, output, completion=completion, checkpoint_dir=checkpoint)
        self.assertEqual(completion.call_count, 1)
        report = json.loads((output/'validation.json').read_text())
        self.assertEqual(report['proposer_requests'], 2)
        self.assertEqual(report['usage']['total_tokens'], 300)

    def test_output_truncation_stops_once_and_keeps_checkpoint(self):
        self.prepare()
        checkpoint = self.root/'truncated_checkpoint'
        completion = Mock(return_value={'choices': [{'finish_reason': 'length', 'message': {
            'role': 'assistant', 'content': '', 'reasoning_content': 'actual partial reasoning'}}],
            'usage': {'prompt_tokens': 80, 'completion_tokens': 60000, 'total_tokens': 60080}})
        with self.assertRaisesRegex(ValueError, 'output truncated'):
            p.propose(self.workspace, self.root/'truncated', completion=completion, checkpoint_dir=checkpoint)
        self.assertEqual(completion.call_count, 1)
        saved = json.loads((checkpoint/'agent_checkpoint.json').read_text())
        self.assertEqual(saved['error'], 'output_truncated')
        self.assertEqual(saved['requests'], 1)
        self.assertFalse(saved['inflight'])

    def test_context_exhaustion_is_normal_and_checkpoint_not_inflight(self):
        self.prepare()
        checkpoint = self.root/'context_checkpoint'
        completion = Mock(side_effect=p.ContextLimit('provider context window exhausted'))
        with self.assertRaises(p.ProposalFailure) as caught:
            p.propose(self.workspace, self.root/'too_long', completion=completion, checkpoint_dir=checkpoint)
        self.assertEqual(caught.exception.code, 'context_limit')
        saved = json.loads((checkpoint/'agent_checkpoint.json').read_text())
        self.assertFalse(saved['inflight'])
        self.assertEqual(saved['error'], 'context_limit')
        self.assertEqual(completion.call_count, 1)

    def test_flash_accounting_does_not_impose_solver_context_limit(self):
        from evolution.life_optimizer.context import FlashCounter, CONTEXT_LIMIT, OUTPUT_LIMIT, TOKEN_MARGIN
        self.assertIsNone(CONTEXT_LIMIT)
        self.assertEqual(OUTPUT_LIMIT, 60000)
        self.assertEqual(TOKEN_MARGIN, 0)
        counter = object.__new__(FlashCounter)
        counter.count = Mock(return_value=50000)
        self.assertEqual(counter.guard([], []), 50000)
        counter.count.return_value = 100000
        self.assertEqual(counter.guard([], []), 100000)

    def test_api_usage_accounting_does_not_apply_removed_flash_limits(self):
        from evolution.life_optimizer.context import verify_usage
        for usage in ({'prompt_tokens': 50000, 'completion_tokens': 6000},
                      {'prompt_tokens': 100000, 'completion_tokens': 60000}):
            verify_usage({'usage': usage})
        for usage in ({'prompt_tokens': -1, 'completion_tokens': 1},
                      {'prompt_tokens': 1, 'completion_tokens': -1}, {}):
            with self.assertRaises(RuntimeError): verify_usage({'usage': usage})

    def test_default_flash_transport_uses_requested_60000_without_solver_guard(self):
        messages = [{'role': 'system', 'content': 'public training evidence ' * 15000}]
        result = {'choices': [{'message': {'role': 'assistant', 'content': 'synthetic'}}],
                  'usage': {'prompt_tokens': 50000, 'completion_tokens': 50000}}
        opener = Mock()
        opener.open.return_value = StringIO(json.dumps(result))
        with patch.dict(p.os.environ, {'MR_API_KEY': 'synthetic-test-key',
                                     'MR_HOST': 'https://router.invalid'}), \
                patch.object(p.urllib.request, 'build_opener', return_value=opener), \
                patch('evolution.life_optimizer.context.FlashCounter',
                      side_effect=AssertionError('Flash transport must not invoke the solver-budget guard')):
            self.assertEqual(p.model_router_completion(messages, p.TOOLS), result)
        request = opener.open.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(payload['model'], 'deepseek-v4-flash')
        self.assertEqual(payload['max_tokens'], 60000)
        self.assertEqual(payload['messages'], messages)
        self.assertEqual(payload['tools'], p.TOOLS)
        self.assertNotIn('max_context_tokens', payload)

    def test_optimizer_prompt_does_not_inherit_qwen_token_settings(self):
        self.prepare()
        prompt = (self.workspace/'OPTIMIZER.md').read_text()
        self.assertNotIn('32768', prompt)
        self.assertNotIn('4096', prompt)
        self.assertIn('32 API requests', prompt)

    def test_unknown_inflight_request_is_not_blindly_reissued(self):
        self.prepare()
        checkpoint = self.root/'inflight_checkpoint'
        with self.assertRaises(RuntimeError):
            p.propose(self.workspace, self.root/'inflight', completion=Mock(side_effect=RuntimeError('transport')),
                      checkpoint_dir=checkpoint)
        completion = Mock(side_effect=AssertionError('must not call'))
        with self.assertRaisesRegex(ValueError, 'unknown outcome'):
            p.propose(self.workspace, self.root/'inflight', completion=completion, checkpoint_dir=checkpoint)
        completion.assert_not_called()

    def test_strict_acceptance_requires_full_predeclared_50_keys(self):
        ids = list(range(50))
        incumbent = {"split": "train", "domain": "webshop", "n": 50, "successes": 25,
                     "per_task": {str(i): int(i < 25) for i in ids}}
        candidate = copy.deepcopy(incumbent)
        self.assertFalse(p.strict_improvement(candidate, incumbent, expected_task_ids=ids))
        candidate["per_task"]["25"] = 1; candidate["successes"] = 26
        self.assertTrue(p.strict_improvement(candidate, incumbent, expected_task_ids=ids))
        for field, value in (("split", "test"), ("n", 8), ("domain", "dbbench")):
            with self.assertRaises(ValueError): p.strict_improvement({**candidate, field: value}, incumbent, expected_task_ids=ids)
        with self.assertRaises(ValueError): p.strict_improvement(candidate, incumbent, expected_task_ids=ids[:8])
        with self.assertRaises(ValueError): p.strict_improvement({**candidate, "accuracy": 1.0}, incumbent, expected_task_ids=ids)


if __name__ == "__main__":
    unittest.main()
