"""API transport with a complete attempt ledger; the optimizer is unchanged.

The old api_broker.py remains byte-identical for sealed experiment provenance.
"""
import json
import threading
import time
from contextlib import contextmanager, nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx

from .api_broker import ReasoningCache, RequestGate, capture_tool_ids
from .token_accounting import TokenLedger, canonical_usage, decode_scope, numeric_usage


class Broker:
    def __init__(self, key, base_url, request_gate=None, rate_limit_retries=0, *, ledger_path,
                 allowed_model="deepseek-v4-flash", ledger_summary_path=None, upstream_model=None,
                 upstream_timeout=600):
        if allowed_model not in {"deepseek-v4-flash", "qwen3.8-flash"}:
            raise ValueError("unsupported metered provider model")
        if upstream_model not in {None, allowed_model, "deepseek-v4.1-flash"}:
            raise ValueError("unsupported upstream provider model")
        if not isinstance(upstream_timeout, (int, float)) or upstream_timeout <= 0:
            raise ValueError("positive upstream timeout required")
        self.client = httpx.Client(timeout=upstream_timeout, trust_env=False)
        self.reasoning = ReasoningCache()
        self.request_gate = request_gate
        self.rate_limit_responses = 0
        self.blocked_roles = set()  # Lifecycle gate; empty preserves historical transport.
        self.ledger = TokenLedger(ledger_path, summary_path=ledger_summary_path)
        secret = key.encode()
        broker = self

        @contextmanager
        def upstream(content, scope, payload):
            for attempt in range(rate_limit_retries + 1):
                ident = None
                status, usage, error, delivered, transport = None, None, None, False, None
                try:
                    with broker.request_gate or nullcontext():
                        ident = broker.ledger.start(scope, model=payload["model"], attempt=attempt + 1,
                                                    stream=payload.get("stream", False))
                        with broker.client.stream("POST", base_url + "/chat/completions",
                                headers={"Authorization": "Bearer " + key,
                                         "Content-Type": "application/json", "Accept-Encoding": "identity"},
                                content=content) as response:
                            status = response.status_code
                            if status == 429:
                                broker.rate_limit_responses += 1
                                if attempt < rate_limit_retries:
                                    response.read()
                                    if broker.request_gate:
                                        broker.request_gate.defer(60)
                                    else:
                                        time.sleep(60)
                                    continue
                            receipt = {"usage": None, "delivered": True, "transport": None}
                            try:
                                yield response, receipt
                            finally:
                                usage, delivered = receipt["usage"], receipt["delivered"]
                                transport = receipt["transport"]
                            return
                except Exception as exc:
                    error = type(exc).__name__  # no exception text/URL/header secrets
                    raise
                finally:
                    if ident is not None:
                        broker.ledger.finish(ident, status=status, usage=usage,
                                             error_type=error, delivered=delivered, transport=transport)
                        broker.ledger.publish()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                try:
                    scope = decode_scope(self.path)
                except (ValueError, TypeError, KeyError):
                    self.send_error(404)
                    return
                if scope["role"] in broker.blocked_roles:
                    self.send_error(403, "optimization is sealed")
                    return
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                try:
                    payload = json.loads(body)
                    if payload.get("model") != allowed_model or secret in body:
                        self.send_error(400, "unsupported model or forbidden credential content")
                        return
                    broker.reasoning.restore(payload)
                    if upstream_model is not None:
                        payload["model"] = upstream_model
                    if payload.get("stream"):
                        # Usage is transport metadata, not a model/prompt/search change.
                        payload.setdefault("stream_options", {})["include_usage"] = True
                    body = json.dumps(payload, ensure_ascii=False).encode()
                    with upstream(body, scope, payload) as (response, receipt):
                        self.send_response(response.status_code)
                        self.send_header("Content-Type", response.headers.get("Content-Type", "application/json"))
                        self.send_header("Connection", "close")
                        self.end_headers()
                        buffer, reasoning, call_ids = b"", "", {}
                        streamed = "text/event-stream" in response.headers.get("Content-Type", "").lower()
                        transport = receipt["transport"] = {"streamed": streamed, "done_seen": False,
                            "upstream_eof": False, "usage_chunks": 0, "invalid_usage_chunks": 0,
                            "error_event_seen": False, "finish_reason": None}

                        def capture_usage(value):
                            if value is None:
                                return
                            transport["usage_chunks"] += 1
                            measured = canonical_usage(numeric_usage(value))
                            previous = canonical_usage(numeric_usage(receipt["usage"]))
                            if measured is not None and (previous is None or
                                    measured['total_tokens'] >= previous['total_tokens']):
                                # Usage is cumulative for this request, not additive.
                                receipt["usage"] = value
                            else:
                                transport["invalid_usage_chunks"] += 1
                                if receipt["usage"] is None:
                                    receipt["usage"] = value

                        def parse_line(line):
                            nonlocal reasoning
                            if not line.startswith(b"data:"):
                                return
                            data = line[5:].strip()
                            if data == b"[DONE]":
                                transport["done_seen"] = True
                                broker.reasoning.save(payload.get("messages", []),
                                    [call_ids[i] for i in sorted(call_ids)], reasoning)
                                return
                            chunk = json.loads(data)
                            capture_usage(chunk.get("usage"))
                            if chunk.get("error") is not None:
                                transport["error_event_seen"] = True
                            for choice in chunk.get("choices", []):
                                if choice.get("index", 0) == 0:
                                    reason = choice.get("finish_reason")
                                    if reason is not None:
                                        transport["finish_reason"] = reason if reason in {
                                            "stop", "length", "tool_calls", "function_call", "content_filter"} else "other"
                                    delta = choice.get("delta", {})
                                    reasoning += delta.get("reasoning_content") or ""
                                    capture_tool_ids(call_ids, delta.get("tool_calls"))

                        for data in response.iter_bytes():
                            buffer += data
                            if streamed:
                                while b"\n" in buffer:
                                    line, buffer = buffer.split(b"\n", 1)
                                    parse_line(line)
                            if receipt["delivered"]:
                                try:
                                    self.wfile.write(data)
                                    self.wfile.flush()
                                except (BrokenPipeError, ConnectionResetError):
                                    # Still drain upstream, so a disconnected client
                                    # cannot erase already-incurred token spend.
                                    receipt["delivered"] = False
                        if streamed and buffer.strip():
                            parse_line(buffer)
                        elif not streamed:
                            result = json.loads(buffer)
                            capture_usage(result.get("usage"))
                            transport["error_event_seen"] = result.get("error") is not None
                            if response.status_code == 200:
                                for choice in result.get("choices", [])[:1]:
                                    message = choice.get("message", {})
                                    broker.reasoning.save(payload.get("messages", []),
                                        [c["id"] for c in message.get("tool_calls") or []],
                                        message.get("reasoning_content"))
                        transport["upstream_eof"] = True
                except Exception:
                    self.close_connection = True

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.ledger.publish()
        self.client.close()
