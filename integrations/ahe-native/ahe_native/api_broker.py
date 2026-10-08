"""Loopback connection transport; keep real API credentials out of agent shells."""
import json
import hashlib
import threading
import time
from contextlib import contextmanager, nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx


def capture_tool_ids(ids, calls):
    # Same semantics as pinned NexAU's _ToolCallAggregator: IDs are complete
    # values, even when providers repeat them on later argument chunks.
    for call in calls or []:
        if call.get("id"):
            ids[call.get("index", 0)] = call["id"]


class ReasoningCache:
    """Restore only actual API reasoning omitted by old NexAU serialization."""
    def __init__(self):
        self.values = {}
        self.by_ids = {}
        self.unique_calls = {}
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.response_ids = []
        self.miss_ids = []
        self.with_reasoning = 0
        self.without_reasoning = 0
        self.observed_ids = {}
        self.miss_details = []

    @staticmethod
    def key(messages, ids):
        public = [{k: v for k, v in m.items() if k != "reasoning_content"}
                  for m in messages]
        return hashlib.sha256(json.dumps([public, ids], sort_keys=True,
            ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()

    def save(self, messages, ids, reasoning):
        if ids:
            self.response_ids = ids
            if reasoning:
                self.with_reasoning += 1
            else:
                self.without_reasoning += 1
            with self.lock:
                self.observed_ids[tuple(ids)] = {"anchor": self.anchor(messages),
                    "reasoning_length": len(reasoning or "")}
        if ids:
            # Flash legitimately returns tool continuations with zero reasoning
            # text. NexAU drops that empty field, but the API still requires the
            # field on replay. Preserve the actual zero-length value, not a
            # fabricated thought and not a change of thinking mode.
            reasoning = reasoning or ""
            with self.lock:
                self.values[self.key(messages, ids)] = reasoning
                identity = (self.anchor(messages), tuple(ids))
                if identity not in self.by_ids:
                    self.by_ids[identity] = reasoning
                elif self.by_ids[identity] != reasoning:
                    self.by_ids[identity] = None  # ambiguous response IDs: never guess
                # Some runtime history transformations change the user prefix
                # or split parallel calls into separate assistant messages.
                # An actual globally unique response call ID still identifies
                # its original reasoning; conflicting IDs are never restored.
                for call_id in ids:
                    if call_id not in self.unique_calls:
                        self.unique_calls[call_id] = reasoning
                    elif self.unique_calls[call_id] != reasoning:
                        self.unique_calls[call_id] = None

    @staticmethod
    def anchor(messages):
        first_user = next((m.get("content") for m in messages if m.get("role") == "user"), None)
        return hashlib.sha256(json.dumps(first_user, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def restore(self, payload):
        messages = payload.get("messages", [])
        for index, message in enumerate(messages):
            calls = message.get("tool_calls")
            if message.get("role") == "assistant" and calls and not message.get("reasoning_content"):
                with self.lock:
                    ids = [c["id"] for c in calls]
                    reasoning = self.values.get(self.key(messages[:index], ids))
                    if reasoning is None:
                        # Original middleware can rewrite prior tool results or
                        # system context. Stable first-user identity + unique
                        # actual response IDs still identify the API response.
                        reasoning = self.by_ids.get((self.anchor(messages[:index]), tuple(ids)))
                    if reasoning is None:
                        candidates = [self.unique_calls.get(call_id) for call_id in ids]
                        if candidates and all(c is not None for c in candidates) and len(set(candidates)) == 1:
                            reasoning = candidates[0]
                if reasoning is not None:
                    message["reasoning_content"] = reasoning
                    self.hits += 1
                else:
                    self.misses += 1
                    self.miss_ids = ids
                    with self.lock:
                        observed = self.observed_ids.get(tuple(ids))
                        self.miss_details = (self.miss_details + [{"ids": ids,
                            "request_anchor": self.anchor(messages[:index]),
                            "observed_response": observed}])[-12:]


class RequestGate:
    """One host-side API request limit shared by every Flash caller."""
    def __init__(self, limit, min_interval=0, initial_delay=0):
        if type(limit) is not int or limit < 1:
            raise ValueError("positive API concurrency required")
        self.limit = limit
        self.slots = threading.BoundedSemaphore(limit)
        self.lock = threading.Lock()
        self.active = self.peak = self.requests = 0
        self.min_interval = float(min_interval)
        if self.min_interval < 0 or initial_delay < 0:
            raise ValueError("request pacing cannot be negative")
        self.next_start = time.monotonic() + initial_delay

    def __enter__(self):
        self.slots.acquire()
        if self.min_interval:
            while True:
                with self.lock:
                    delay = self.next_start - time.monotonic()
                    if delay <= 0:
                        self.next_start = time.monotonic() + self.min_interval
                        break
                time.sleep(min(delay, 1.0))
        with self.lock:
            self.active += 1
            self.requests += 1
            self.peak = max(self.peak, self.active)
        return self

    def __exit__(self, *args):
        with self.lock:
            self.active -= 1
        self.slots.release()

    def snapshot(self):
        with self.lock:
            result = {"limit": self.limit, "active": self.active,
                    "peak": self.peak, "requests": self.requests}
            if self.min_interval:
                result.update(min_interval_seconds=self.min_interval,
                    paced_requests_per_minute=60 / self.min_interval)
            return result

    def defer(self, seconds):
        with self.lock:
            self.next_start = max(self.next_start, time.monotonic() + seconds)


class Broker:
    def __init__(self, key, base_url, request_gate=None, rate_limit_retries=0):
        self.client = httpx.Client(timeout=600, trust_env=False)
        self.reasoning = ReasoningCache()
        self.request_gate = request_gate
        self.rate_limit_responses = 0
        secret = key.encode()
        broker = self

        @contextmanager
        def upstream(content):
            for attempt in range(rate_limit_retries + 1):
                with (broker.request_gate or nullcontext()), broker.client.stream(
                        "POST", base_url + "/chat/completions",
                        headers={"Authorization": "Bearer " + key,
                                 "Content-Type": "application/json", "Accept-Encoding": "identity"},
                        content=content) as response:
                    if response.status_code == 429:
                        broker.rate_limit_responses += 1
                        if attempt < rate_limit_retries:
                            response.read()
                            if broker.request_gate:
                                broker.request_gate.defer(60)
                            else:
                                time.sleep(60)
                            continue
                    yield response
                    return

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass  # never log request contents, credentials or headers

            def do_POST(self):
                if self.path != "/v1/chat/completions":
                    self.send_error(404)
                    return
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                try:
                    payload = json.loads(body)
                    if payload.get("model") != "deepseek-v4-flash" or secret in body:
                        self.send_error(400, "unsupported model or forbidden credential content")
                        return
                    broker.reasoning.restore(payload)
                    body = json.dumps(payload, ensure_ascii=False).encode()
                    with upstream(body) as response:
                        self.send_response(response.status_code)
                        self.send_header("Content-Type", response.headers.get("Content-Type", "application/json"))
                        self.send_header("Connection", "close")
                        self.end_headers()
                        buffer = b""
                        reasoning = ""
                        call_ids = {}
                        streamed = "text/event-stream" in response.headers.get("Content-Type", "")
                        # httpx decodes any upstream compression. Do not forward
                        # compressed bytes without their encoding header.
                        for data in response.iter_bytes():
                            buffer += data
                            if streamed:
                                while b"\n" in buffer:
                                    line, buffer = buffer.split(b"\n", 1)
                                    if line.startswith(b"data:") and line[5:].strip() != b"[DONE]":
                                        chunk = json.loads(line[5:].strip())
                                        for choice in chunk.get("choices", []):
                                            if choice.get("index", 0) != 0:
                                                continue
                                            delta = choice.get("delta", {})
                                            reasoning += delta.get("reasoning_content") or ""
                                            capture_tool_ids(call_ids, delta.get("tool_calls"))
                                    elif line.startswith(b"data:") and line[5:].strip() == b"[DONE]":
                                        # Cache the complete response once, before
                                        # forwarding DONE. Partial reasoning chunks
                                        # are not different responses and must not
                                        # mark this response ID as ambiguous.
                                        broker.reasoning.save(payload.get("messages", []),
                                            [call_ids[i] for i in sorted(call_ids)], reasoning)
                            self.wfile.write(data)
                            self.wfile.flush()
                        if not streamed and response.status_code == 200:
                            result = json.loads(buffer)
                            for choice in result.get("choices", [])[:1]:
                                message = choice.get("message", {})
                                broker.reasoning.save(payload.get("messages", []),
                                    [c["id"] for c in message.get("tool_calls") or []],
                                    message.get("reasoning_content"))
                except (BrokenPipeError, ConnectionResetError):
                    pass
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
        self.client.close()


def isolated_connection(config, broker_url, debugger=False):
    """Change only connection settings; no prompts/search/metrics are altered."""
    import copy
    value = copy.deepcopy(config)
    if debugger:
        value.setdefault("llm", {}).update(api_key="EMPTY", base_url=broker_url)
    else:
        value["evolve_agent"]["llm_config"].update(api_key="${env.AHE_EVOLVE_API_KEY}",
            base_url="${env.AHE_EVOLVE_BASE_URL}")
        value.setdefault("agent_debugger", {}).setdefault("llm", {}).update(
            api_key="EMPTY", base_url=broker_url)
    return value
