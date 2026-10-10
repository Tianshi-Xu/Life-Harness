import copy
import json
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from urllib.request import Request, urlopen
from pathlib import Path

from ahe_native.records import export_job, validate_records
from ahe_native.upstream import bind_native_transport, load_upstream, source_fingerprint
from ahe_native.connection import bind_connections
from ahe_native.backend import validate_request, HERE, MANIFEST
from ahe_native.api_broker import Broker, isolated_connection, ReasoningCache, capture_tool_ids, RequestGate
from ahe_native.debugger_isolation import bind_isolated_debugger
from ahe_native.evolve_isolation import bind_isolated_evolve
from ahe_native.isolation import make_source_readable, require_regular_workspace


def payload():
    return {"schema_version": 1, "domain": "dbbench", "split": "train", "records": [
        {"task_id": "../../task-A", "trial": 0, "success": 1,
         "messages": [{"role": "user", "content": "public task"},
                      {"role": "assistant", "content": "answer"}]},
        {"task_id": "task-B", "trial": 0, "success": 0,
         "infrastructure_error": True,
         "messages": [{"role": "user", "content": "public task"}]}]}


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ahe, cls.receipt = load_upstream()

    def test_upstream_pin_and_no_tracked_edits(self):
        self.assertEqual(source_fingerprint(), self.receipt)

    def test_only_transport_binding_changes(self):
        names = ("run_single_experiment", "run_evolve_agent", "compute_stats",
                 "build_evolution_query", "evaluate_changes", "update_best_ever",
                 "select_variant_winner", "run_parallel_adb_ask")
        before = {name: getattr(self.ahe, name) for name in names}
        original = self.ahe.launch_harbor
        original_wait = self.ahe.wait_for_harbor
        try:
            receipt = bind_native_transport(self.ahe)
            self.assertEqual(receipt["replaced_functions"], ["launch_harbor", "wait_for_harbor"])
            self.assertIsNot(original, self.ahe.launch_harbor)
            for name in names:
                self.assertIs(before[name], getattr(self.ahe, name))
        finally:
            self.ahe.launch_harbor = original
            self.ahe.wait_for_harbor = original_wait

    def test_original_stats_and_debugger_reader_accept_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = export_job(payload(), Path(tmp) / "2026-10-02__00-00-01")
            stats = self.ahe.compute_stats(job, k=1)
            self.assertEqual(stats["n_total"], 2)
            self.assertEqual(stats["pass_rate"], 0.5)
            jobs = self.ahe._build_adb_jobs(stats["task_results"], job, {})
            self.assertEqual(len(jobs), 2)
            for item in jobs:
                self.assertIsNone(item.trace_type)  # original default: OpenAI messages
                trace = json.loads(item.trace_paths[0].read_text())
                self.assertTrue(trace["messages"])
            self.assertEqual(self.ahe.wait_for_job(Path(tmp), None, poll_interval=0.01,
                                                 timeout_minutes=0.01), job)

    def test_pass_at_one_multiple_trials(self):
        value = payload()
        extra = copy.deepcopy(value["records"])
        for row in extra:
            row["trial"] = 1
            row["infrastructure_error"] = False
            row["success"] = 0
        value["records"].extend(extra)
        with tempfile.TemporaryDirectory() as tmp:
            job = export_job(value, Path(tmp) / "2026-10-02__00-00-01")
            stats = self.ahe.compute_stats(job, k=2)
            self.assertEqual(stats["n_total"], 2)
            self.assertEqual(stats["pass_rate"], 0.25)

    def test_test_split_never_exported(self):
        value = payload()
        value["split"] = "test"
        with self.assertRaises(ValueError):
            validate_records(value)

    def test_native_os_train_records_reach_original_stats(self):
        value = payload()
        value['domain'] = 'os'
        with tempfile.TemporaryDirectory() as tmp:
            job = export_job(value, Path(tmp) / '2026-10-08__00-00-01')
            stats = self.ahe.compute_stats(job, k=1)
            self.assertEqual((stats['n_total'],stats['n_pass'],stats['n_fail']),(2,1,1))
            records = [json.loads(p.read_text()) for p in job.glob('native_*/native_record.json')]
            self.assertEqual(sum(r.get('infrastructure_error',False) for r in records),1)
        value['split'] = 'test'
        with self.assertRaises(ValueError):
            validate_records(value)

    def test_private_fields_rejected(self):
        for key in ("task_result_private", "expected_answer", "oracle_sql", "reward_info"):
            value = payload()
            value["records"][0][key] = "private"
            with self.assertRaises(ValueError):
                validate_records(value)

    def test_private_message_fields_rejected(self):
        value = payload()
        value["records"][0]["messages"][0]["hidden_goal"] = "private"
        with self.assertRaises(ValueError):
            validate_records(value)

    def test_duplicate_trials_rejected(self):
        value = payload()
        value["records"].append(copy.deepcopy(value["records"][0]))
        with self.assertRaises(ValueError):
            validate_records(value)

    def test_previous_results_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = export_job(payload(), Path(tmp) / "2026-10-02__00-00-01")
            before = (job / "result.json").read_bytes()
            with self.assertRaises(FileExistsError):
                export_job(payload(), job)
            self.assertEqual(before, (job / "result.json").read_bytes())

    def test_connection_resolution_does_not_mutate_saved_config(self):
        original = self.ahe.get_llm_config
        config = {"llm": {"api_key": "${env.AHE_TEST_KEY}",
                          "base_url": "http://localhost/v1", "model": "test"}}
        before = copy.deepcopy(config)
        try:
            bind_connections(self.ahe)
            with patch.dict("os.environ", {"AHE_TEST_KEY": "dummy-test-only"}):
                self.assertEqual(self.ahe.get_llm_config(config)["api_key"], "dummy-test-only")
            self.assertEqual(config, before)
            self.assertIs(self.ahe.get_llm_config.__wrapped__, original)
        finally:
            self.ahe.get_llm_config = original

    def test_failed_native_backend_cannot_reach_original_wait(self):
        original_launch, original_wait = self.ahe.launch_harbor, self.ahe.wait_for_harbor
        class FailedProcess:
            ahe_native_backend = True
            def wait(self, timeout=None):
                return 7
        try:
            bind_native_transport(self.ahe)
            with self.assertRaisesRegex(RuntimeError, "incomplete results"):
                self.ahe.wait_for_harbor(FailedProcess(), Path("unused"), "")
        finally:
            self.ahe.launch_harbor, self.ahe.wait_for_harbor = original_launch, original_wait

    def test_native_backend_train_pool_guard(self):
        domain = json.loads(MANIFEST.read_text())["domains"]["dbbench"]
        ids = domain["batches"][0]["indices"][:1]
        request = {"domain": "dbbench", "split": "train", "task_ids": ids,
                   "k": 1, "concurrency": 1, "workspace": str(HERE / "baselines/dbbench"),
                   "agent_config": "code_agent.yaml"}
        self.assertEqual(validate_request(request), [str(x) for x in ids])
        for override in ({"split": "test"}, {"task_ids": ["outside-train-pool"]},
                         {"k": True}, {"concurrency": 0}):
            with self.assertRaises(ValueError):
                validate_request({**request, **override})

    def test_isolation_wrappers_retain_original_bodies(self):
        original_evolve = self.ahe.run_evolve_agent
        original_debugger = self.ahe.run_parallel_adb_ask
        original_loop = self.ahe.run_single_experiment
        try:
            bind_isolated_evolve(self.ahe)
            bind_isolated_debugger(self.ahe)
            self.assertIs(self.ahe.run_evolve_agent.__wrapped__, original_evolve)
            self.assertIs(self.ahe.run_parallel_adb_ask.__wrapped__, original_debugger)
            self.assertIs(self.ahe.run_single_experiment, original_loop)
        finally:
            self.ahe.run_evolve_agent = original_evolve
            self.ahe.run_parallel_adb_ask = original_debugger

    def test_broker_changes_only_connections_and_holds_no_key_in_payload(self):
        config = {"evolve_agent": {"llm_config": {"model": "deepseek-v4-flash",
                  "api_key": "${env.AHE_EVOLVE_API_KEY}", "base_url": "${env.AHE_EVOLVE_BASE_URL}"}},
                  "agent_debugger": {"llm": {"model": "deepseek-v4-flash"}}, "target_pass_rate": 0.95}
        before = copy.deepcopy(config)
        isolated = isolated_connection(config, "http://127.0.0.1:1/v1")
        self.assertEqual(config, before)
        self.assertEqual(isolated["target_pass_rate"], config["target_pass_rate"])
        self.assertEqual(isolated["evolve_agent"], config["evolve_agent"])
        self.assertEqual(isolated["agent_debugger"]["llm"]["api_key"], "EMPTY")

    def test_reasoning_transport_restores_only_matching_actual_response(self):
        cache = ReasoningCache()
        prefix = [{"role": "user", "content": "public task"}]
        cache.save(prefix, ["tool-id"], "actual model reasoning")
        call = {"role": "assistant", "content": "", "tool_calls": [{"id": "tool-id"}]}
        payload = {"messages": [*copy.deepcopy(prefix), copy.deepcopy(call)]}
        cache.restore(payload)
        self.assertEqual(payload["messages"][1]["reasoning_content"], "actual model reasoning")
        other = {"messages": [{"role": "user", "content": "different task"},
            {"role": "assistant", "tool_calls": [{"id": "never-observed-response-id"}]}]}
        cache.restore(other)
        self.assertNotIn("reasoning_content", other["messages"][1])
        rewritten = {"messages": [{"role": "system", "content": "new middleware context"},
            *copy.deepcopy(prefix), copy.deepcopy(call)]}
        cache.restore(rewritten)
        self.assertEqual(rewritten["messages"][2]["reasoning_content"], "actual model reasoning")

    def test_stream_repeated_tool_id_is_not_concatenated(self):
        ids = {}
        capture_tool_ids(ids, [{"index": 0, "id": "actual-call-id"}])
        capture_tool_ids(ids, [{"index": 0, "id": "actual-call-id"}])
        capture_tool_ids(ids, [{"index": 0, "function": {"arguments": "more"}}])
        self.assertEqual(ids, {0: "actual-call-id"})

    def test_unique_actual_response_ids_survive_prefix_rewrite_and_call_split(self):
        cache = ReasoningCache()
        prefix = [{"role": "user", "content": "original public context"}]
        cache.save(prefix, ["parallel-one", "parallel-two"], "actual complete reasoning")
        payload = {"messages": [{"role": "user", "content": "rewritten public context"},
            {"role": "assistant", "tool_calls": [{"id": "parallel-two"}]}]}
        cache.restore(payload)
        self.assertEqual(payload["messages"][-1]["reasoning_content"], "actual complete reasoning")
        cache.save([{ "role": "user", "content": "unrelated response"}], ["parallel-two"], "conflicting reasoning")
        ambiguous = {"messages": [{"role": "user", "content": "third context"},
            {"role": "assistant", "tool_calls": [{"id": "parallel-two"}]}]}
        cache.restore(ambiguous)
        self.assertNotIn("reasoning_content", ambiguous["messages"][-1])

    def test_actual_zero_length_reasoning_is_replayed_without_fabrication(self):
        cache = ReasoningCache()
        prefix = [{"role": "user", "content": "public task"}]
        cache.save(prefix, ["actual-empty-response"], "")
        payload = {"messages": [*prefix, {"role": "assistant", "tool_calls": [{"id": "actual-empty-response"}]}]}
        cache.restore(payload)
        self.assertIn("reasoning_content", payload["messages"][-1])
        self.assertEqual(payload["messages"][-1]["reasoning_content"], "")

    def test_private_generated_source_becomes_readable_but_not_solver_writable(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "workspace"
            modules = root / "middleware"
            modules.mkdir(parents=True, mode=0o700)
            source = modules / "generated.py"
            source.write_text("actual_generated_source = True\n")
            source.chmod(0o600)
            before = source.read_bytes()
            make_source_readable(root)
            self.assertEqual(source.stat().st_mode & 0o777, 0o444)
            self.assertEqual(modules.stat().st_mode & 0o777, 0o555)
            self.assertEqual(source.read_bytes(), before)

    def test_agent_symlink_is_rejected_before_host_resolution(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "not_a_regular_module.py").symlink_to("/nonexistent/private-host-target")
            with self.assertRaises(ValueError):
                require_regular_workspace(root)

    def test_stream_caches_complete_reasoning_not_ambiguous_partial_chunks(self):
        response = MagicMock()
        response.status_code = 200
        response.headers = {"Content-Type": "text/event-stream"}
        chunks = [{"choices": [{"delta": {"reasoning_content": "first "}}]},
                  {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "real-id"}],
                                            "reasoning_content": "second"}}]},
                  {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "real-id"}]}}]}]
        response.iter_bytes.return_value = [*(b"data: " + json.dumps(c).encode() + b"\n\n" for c in chunks),
                                             b"data: [DONE]\n\n"]
        client = MagicMock()
        client.stream.return_value.__enter__.return_value = response
        gate = RequestGate(8)
        with patch("ahe_native.api_broker.httpx.Client", return_value=client):
            broker = Broker("unit-secret", "http://unused/v1", request_gate=gate)
        try:
            prefix = [{"role": "user", "content": "public task"}]
            request = Request(broker.url + "/chat/completions", data=json.dumps({
                "model": "deepseek-v4-flash", "messages": prefix, "stream": True}).encode(),
                headers={"Content-Type": "application/json"})
            with urlopen(request) as result:
                result.read()
            payload = {"messages": [{"role": "system", "content": "rewritten context"}, *prefix,
                {"role": "assistant", "tool_calls": [{"id": "real-id"}]}]}
            broker.reasoning.restore(payload)
            self.assertEqual(payload["messages"][-1]["reasoning_content"], "first second")
            self.assertEqual(broker.reasoning.with_reasoning, 1)
            self.assertEqual(broker.reasoning.hits, 1)
            self.assertEqual(gate.snapshot(), {"limit": 8, "active": 0, "peak": 1, "requests": 1})
        finally:
            broker.close()


if __name__ == "__main__":
    unittest.main()
