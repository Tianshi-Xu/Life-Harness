import json
import hashlib
import tempfile
import threading
import unittest
import httpx
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.request import Request, urlopen

from ahe_native.api_broker import RequestGate
from ahe_native.metered_broker import Broker
from ahe_native.token_accounting import TokenLedger, canonical_usage, decode_scope, scoped_url


class AccountingTests(unittest.TestCase):
    def test_extended_optimizer_timeout_keeps_historical_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            for timeout in (600,1800):
                broker = Broker('synthetic', 'http://127.0.0.1:1/v1',
                    ledger_path=Path(tmp)/str(timeout), upstream_timeout=timeout)
                try:
                    self.assertEqual(broker.client.timeout.read,timeout)
                finally:
                    broker.close()
            with self.assertRaises(ValueError):
                Broker('synthetic','http://127.0.0.1:1/v1',ledger_path=Path(tmp)/'invalid',upstream_timeout=0)

    def test_authorized_os_timeout_remains_unknown_and_estimate_is_receipt_bound(self):
        ledger = TokenLedger(self.path)
        ident = ledger.start(dict(experiment='synthetic', domain='os', round=1, role='life_optimizer'),
                             model='deepseek-v4-flash', attempt=1, stream=False)
        ledger.finish(ident, status=None, usage=None, error_type='ReadTimeout', delivered=False)
        start, end = ledger.rows
        receipt = hashlib.sha256(json.dumps([start,end],sort_keys=True,separators=(',',':')).encode()).hexdigest()
        estimate = {'request_id':ident,'user_authorized':True,'authorization_quote':'explicit test authorization',
                    'exception_kind':'authorized_os_read_timeout','basis':'synthetic prior measured reference',
                    'output_token_range':[0,100000],'receipt_sha256':receipt,
                    'usage':{'prompt_tokens':100,'completion_tokens':20,'total_tokens':120}}
        path = self.path.with_suffix('.estimates.json')
        path.write_text(json.dumps([estimate]))
        summary = ledger.summary()['overall']
        self.assertEqual((summary['unknown_usage_calls'],summary['total_tokens'],summary['estimated_total_tokens']),(1,0,120))
        self.assertFalse(summary['accounting_complete'])
        self.assertTrue(summary['accounting_usable_with_estimates'])
        for patch in ({'authorization_quote':''},{'user_authorized':False},{'receipt_sha256':'changed'},
                      {'exception_kind':'generic_retry'},{'output_token_range':[0,1]}):
            path.write_text(json.dumps([{**estimate,**patch}]))
            with self.assertRaises(ValueError): ledger.summary()

    def test_timeout_before_response_does_not_claim_delivery(self):
        client=MagicMock()
        client.stream.side_effect=httpx.ReadTimeout('synthetic timeout')
        with patch('ahe_native.metered_broker.httpx.Client',return_value=client):
            broker=Broker('synthetic-secret','http://unused/v1',ledger_path=self.path,upstream_timeout=1800)
        self.addCleanup(broker.close)
        with self.assertRaises(Exception): self.call(broker)
        finished=[r for r in broker.ledger.rows if r['event']=='finished']
        self.assertEqual(len(finished),1)
        self.assertEqual(finished[0]['error_type'],'ReadTimeout')
        self.assertIsNone(finished[0]['http_status'])
        self.assertFalse(finished[0]['delivered_to_client'])
        self.assertIsNone(finished[0]['usage'])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "flash_requests.jsonl"

    def test_scope_round_trip(self):
        url = scoped_url("http://localhost/v1", experiment="synthetic", domain="dbbench", round=3, role="debugger")
        self.assertEqual(decode_scope(url.removeprefix("http://localhost") + "/chat/completions"),
                         dict(experiment="synthetic", domain="dbbench", round=3, role="debugger"))
        with self.assertRaises(ValueError):
            decode_scope("/private/chat/completions")

    def test_missing_is_unknown_not_zero_and_restart_keeps_pending(self):
        ledger = TokenLedger(self.path)
        scope = dict(experiment="synthetic", domain="dbbench", round=1, role="evolve")
        first = ledger.start(scope, model="deepseek-v4-flash", attempt=1, stream=False)
        ledger.finish(first, status=200, usage={})
        ledger.start(scope, model="deepseek-v4-flash", attempt=1, stream=False)
        result = TokenLedger(self.path).summary()["overall"]
        self.assertEqual((result["unknown_usage_calls"], result["pending_calls"]), (1, 1))
        self.assertFalse(result["inference_accounting_complete"])
        self.assertFalse(result["accounting_complete"])
        self.assertEqual(result["rate_limit_rejections_without_usage"], 0)

    def test_cache_reasoning_not_double_counted_and_secret_not_saved(self):
        ledger = TokenLedger(self.path)
        ident = ledger.start(dict(experiment="synthetic", domain="dbbench", round=1, role="evolve"),
                             model="deepseek-v4-flash", attempt=1, stream=False)
        ledger.finish(ident, status=200, usage={"prompt_tokens": 100, "completion_tokens": 20,
            "total_tokens": 120, "prompt_cache_hit_tokens": 80,
            "completion_tokens_details": {"reasoning_tokens": 15}, "secret": "never-save-me"})
        result = ledger.publish()["overall"]
        self.assertEqual(result["total_tokens"], 120)
        self.assertTrue(result["accounting_complete"])
        self.assertNotIn("never-save-me", self.path.read_text())
        self.assertIsNone(canonical_usage({"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 999}))

    def broker(self, responses):
        client = MagicMock()
        client.stream.side_effect = [self.context(r) for r in responses]
        with patch("ahe_native.metered_broker.httpx.Client", return_value=client):
            broker = Broker("synthetic-secret", "http://unused/v1", RequestGate(8),
                            rate_limit_retries=1, ledger_path=self.path)
        self.addCleanup(broker.close)
        return broker, client

    @staticmethod
    def context(response):
        value = MagicMock()
        value.__enter__.return_value = response
        return value

    @staticmethod
    def response(result, status=200):
        value = MagicMock()
        value.status_code = status
        value.headers = {"Content-Type": "application/json"}
        value.iter_bytes.return_value = [json.dumps(result).encode()]
        return value

    def call(self, broker, stream=False):
        url = scoped_url(broker.url, experiment="synthetic", domain="webshop", round=2, role="debugger")
        request = Request(url + "/chat/completions", data=json.dumps({"model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "public synthetic prompt"}], "stream": stream}).encode(),
            headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=5) as response:
            data = response.read()
        return data

    def test_nonstream_full_usage(self):
        broker, client = self.broker([self.response({"choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 90, "completion_tokens": 10, "total_tokens": 100}})])
        self.call(broker)
        result = broker.ledger.summary()
        self.assertEqual(result["overall"]["total_tokens"], 100)
        self.assertEqual(next(iter(result["groups"].values()))["role"], "debugger")
        self.assertNotIn("public synthetic prompt", self.path.read_text())
        self.assertNotIn("synthetic-secret", self.path.read_text())

    def test_stream_fragmented_usage_without_choices(self):
        response = MagicMock()
        response.status_code = 200
        response.headers = {"Content-Type": "text/event-stream"}
        wire = b'data: {"choices":[{"delta":{"reasoning_content":"real","tool_calls":[{"index":0,"id":"actual-id"}]}}]}\r\n\r\n'
        wire += b'data: {"choices":[],"usage":{"prompt_tokens":100,"completion_tokens":7,"total_tokens":107}}\n\n'
        wire += b'data: [DONE]\n\n'
        response.iter_bytes.return_value = [wire[i:i+7] for i in range(0, len(wire), 7)]
        broker, client = self.broker([response])
        self.assertEqual(self.call(broker, True), wire)
        self.assertEqual(broker.ledger.summary()["overall"]["total_tokens"], 107)
        forwarded = json.loads(client.stream.call_args.kwargs["content"])
        self.assertTrue(forwarded["stream_options"]["include_usage"])
        self.assertEqual(broker.reasoning.with_reasoning, 1)

    def test_retry_attempt_is_not_lost(self):
        bad = self.response({"error": "rate limit"}, status=429)
        good = self.response({"choices": [{"message": {"content": "ok"}}],
                              "usage": {"prompt_tokens": 100, "completion_tokens": 1}})
        broker, _ = self.broker([bad, good])
        with patch.object(broker.request_gate, "defer"):
            self.call(broker)
        result = broker.ledger.summary()["overall"]
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(result["unknown_usage_calls"], 1)
        self.assertEqual(result["total_tokens"], 101)
        self.assertFalse(result["accounting_complete"])
        self.assertTrue(result["inference_accounting_complete"])
        self.assertEqual(result["rate_limit_rejections_without_usage"], 1)

    @staticmethod
    def stream_response(wire):
        response = MagicMock()
        response.status_code = 200
        response.headers = {"Content-Type": "text/event-stream"}
        response.iter_bytes.return_value = [wire[i:i+7] for i in range(0, len(wire), 7)]
        return response

    def test_valid_stream_usage_survives_empty_partial_and_invalid_tail(self):
        for i, tail in enumerate((None, {}, {'total_tokens': 107}, 'invalid',
                {'prompt_tokens': -1, 'completion_tokens': 7},
                {'prompt_tokens': 100, 'completion_tokens': 7, 'total_tokens': 999},
                {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
                {'prompt_tokens': 99, 'completion_tokens': 0, 'total_tokens': 99})):
            with self.subTest(tail=tail):
                self.path = self.path.with_name(f'tail_{i}.jsonl')
                wire = b'data: {"choices":[],"usage":{"prompt_tokens":100,"completion_tokens":7}}\n\n'
                wire += ('data: ' + json.dumps({'choices': [], 'usage': tail}) + '\n\n').encode()
                wire += b'data: [DONE]\n\n'
                broker, _ = self.broker([self.stream_response(wire)])
                self.assertEqual(self.call(broker, True), wire)
                result = broker.ledger.summary()['overall']
                self.assertEqual((result['total_tokens'], result['unknown_usage_calls']), (107, 0))
                transport = broker.ledger.rows[-1]['transport']
                self.assertTrue(transport['done_seen'])
                self.assertTrue(transport['upstream_eof'])
                self.assertEqual(transport['invalid_usage_chunks'], int(tail is not None))

    def test_latest_valid_usage_is_cumulative_not_added(self):
        wire = b'data: {"usage":{"prompt_tokens":100,"completion_tokens":7}}\n\n'
        wire += b'data: {"usage":{"prompt_tokens":200,"completion_tokens":20}}\n\n'
        wire += b'data: {"usage":{}}\n\ndata: [DONE]\n\n'
        broker, _ = self.broker([self.stream_response(wire)])
        self.call(broker, True)
        self.assertEqual(broker.ledger.summary()['overall']['total_tokens'], 220)

    def test_stream_eof_without_done_is_distinguished_from_missing_usage(self):
        for i, wire in enumerate((b'data: {"usage":{"prompt_tokens":100,"completion_tokens":7}}\n',
                                 b'data: {"choices":[]}\n\ndata: [DONE]\n\n')):
            with self.subTest(i=i):
                self.path = self.path.with_name(f'eof_{i}.jsonl')
                broker, _ = self.broker([self.stream_response(wire)])
                self.assertEqual(self.call(broker, True), wire)
                row = broker.ledger.rows[-1]
                self.assertEqual(row['transport']['done_seen'], bool(i))
                self.assertTrue(row['transport']['upstream_eof'])
                self.assertEqual(row['usage'] is None, bool(i))

    def test_stream_read_error_keeps_received_usage_and_records_incomplete_eof(self):
        wire = b'data: {"usage":{"prompt_tokens":100,"completion_tokens":7}}\n\n'
        response = self.stream_response(wire)
        def interrupted():
            yield wire
            raise httpx.ReadError('synthetic-secret-must-not-be-recorded')
        response.iter_bytes.side_effect = interrupted
        broker, _ = self.broker([response])
        self.assertEqual(self.call(broker, True), wire)
        row = broker.ledger.rows[-1]
        self.assertEqual(row['usage']['total_tokens'], 107)
        self.assertEqual(row['error_type'], 'ReadError')
        self.assertFalse(row['transport']['done_seen'])
        self.assertFalse(row['transport']['upstream_eof'])
        self.assertNotIn('synthetic-secret-must-not-be-recorded', self.path.read_text())

    def test_stream_error_event_is_recorded_without_error_message(self):
        wire = b'data: {"error":{"message":"synthetic-secret-must-not-be-recorded"}}\n\n'
        broker, _ = self.broker([self.stream_response(wire)])
        self.call(broker, True)
        self.assertTrue(broker.ledger.rows[-1]['transport']['error_event_seen'])
        self.assertEqual(broker.ledger.summary()['overall']['unknown_usage_calls'], 1)
        self.assertNotIn('synthetic-secret-must-not-be-recorded', self.path.read_text())

    def test_parallel_streams_preserve_usage_and_shared_gate(self):
        wire = b'data: {"usage":{"prompt_tokens":100,"completion_tokens":7}}\n\n'
        wire += b'data: {"usage":{}}\n\ndata: [DONE]\n\n'
        broker, _ = self.broker([self.stream_response(wire) for _ in range(8)])
        errors = []
        def call():
            try:
                self.call(broker, True)
            except Exception as exc:
                errors.append(type(exc).__name__)
        threads = [threading.Thread(target=call) for _ in range(8)]
        for thread in threads: thread.start()
        for thread in threads: thread.join()
        self.assertEqual(errors, [])
        result = broker.ledger.summary()['overall']
        self.assertEqual((result['attempts'], result['total_tokens'], result['unknown_usage_calls']), (8, 856, 0))
        self.assertLessEqual(broker.request_gate.snapshot()['peak'], 8)

    def test_authorized_estimate_is_separate_and_bound_to_exact_unknown_receipt(self):
        ledger = TokenLedger(self.path)
        ident = ledger.start(dict(experiment='synthetic', domain='telecom', round=2, role='evolve'),
                             model='deepseek-v4-flash', attempt=1, stream=True)
        ledger.finish(ident, status=200)
        before = self.path.read_bytes()
        estimate = {'request_id': ident, 'user_authorized': True, 'basis': 'synthetic neighboring calls',
            'receipt_sha256': hashlib.sha256(json.dumps(ledger.rows, sort_keys=True,
                separators=(',', ':')).encode()).hexdigest(),
            'usage': {'input_tokens': 100, 'output_tokens': 7, 'total_tokens': 107}}
        path = self.path.with_suffix('.estimates.json')
        path.write_text(json.dumps([estimate]))
        result = TokenLedger(self.path).publish()['overall']
        self.assertEqual((result['total_tokens'], result['estimated_total_tokens'],
                          result['budget_tokens_including_estimates']), (0, 107, 107))
        self.assertFalse(result['inference_accounting_complete'])
        self.assertTrue(result['accounting_usable_with_estimates'])
        self.assertEqual(self.path.read_bytes(), before)
        for changed in ({'user_authorized': False}, {'receipt_sha256': 'changed'},
                        {'request_id': 'unrelated'}, {'usage': {'total_tokens': 107}}):
            path.write_text(json.dumps([{**estimate, **changed}]))
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                ledger.summary()
        path.write_text(json.dumps([estimate, estimate]))
        with self.assertRaises(ValueError): ledger.summary()

    def test_estimate_does_not_allow_future_unknown_requests_or_pending_calls(self):
        ledger = TokenLedger(self.path)
        scope = dict(experiment='synthetic', domain='telecom', round=2, role='evolve')
        ident = ledger.start(scope, model='deepseek-v4-flash', attempt=1, stream=True)
        ledger.finish(ident, status=200)
        estimate = {'request_id': ident, 'user_authorized': True, 'basis': 'synthetic',
            'receipt_sha256': hashlib.sha256(json.dumps(ledger.rows, sort_keys=True,
                separators=(',', ':')).encode()).hexdigest(), 'usage': {'input_tokens': 100, 'output_tokens': 7}}
        self.path.with_suffix('.estimates.json').write_text(json.dumps([estimate]))
        extra = ledger.start(scope, model='deepseek-v4-flash', attempt=1, stream=True)
        self.assertFalse(ledger.summary()['overall']['accounting_usable_with_estimates'])
        ledger.finish(extra, status=200)
        self.assertFalse(ledger.summary()['overall']['accounting_usable_with_estimates'])

    def test_parallel_append_and_publish_are_consistent(self):
        ledger = TokenLedger(self.path)
        def record(index):
            ident = ledger.start(dict(experiment="synthetic", domain="alfworld", round=1, role="evolve"),
                                 model="deepseek-v4-flash", attempt=1, stream=False)
            ledger.finish(ident, status=200, usage={"prompt_tokens": index, "completion_tokens": 1})
            ledger.publish()
        threads = [threading.Thread(target=record, args=(i,)) for i in range(16)]
        for thread in threads: thread.start()
        for thread in threads: thread.join()
        self.assertEqual(ledger.summary()["overall"]["total_tokens"], sum(range(16)) + 16)
        self.assertEqual(json.loads(self.path.with_name("TOKEN_USAGE.json").read_text()), ledger.summary())


if __name__ == "__main__":
    unittest.main()
