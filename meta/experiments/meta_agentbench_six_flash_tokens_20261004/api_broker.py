"""Only this host process has a credential; all optimizer agents share one gate."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx


class TokenBroker:
    def __init__(self, key, url, gate, ledger, domains):
        self.gate, self.ledger = gate, ledger
        self.client = httpx.Client(timeout=1800, trust_env=False)
        self.rate_limit_responses = 0
        self.lock = threading.Lock()
        self.cache = {}
        self.accounting_failed = threading.Event()
        broker = self
        secret = key.encode()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, status, value):
                body = json.dumps(value, ensure_ascii=False).encode()
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                if self.path != '/v1/chat/completions':
                    self.reply(404, {'error': 'unsupported endpoint'})
                    return
                raw = self.rfile.read(int(self.headers.get('Content-Length', '0')))
                try:
                    if broker.accounting_failed.is_set():
                        self.reply(503, {'error': 'optimization paused after unknown API usage'})
                        return
                    payload = json.loads(raw)
                    domain = self.headers.get('X-Meta-Domain')
                    role = self.headers.get('X-Meta-Role')
                    iteration = int(self.headers.get('X-Meta-Round', '0'))
                    request_id = self.headers.get('X-Meta-Request-ID', '')
                    if (domain not in domains or role not in ('parent', 'analysis', 'implementation')
                            or not 1 <= iteration <= 6 or not request_id
                            or payload.get('model') != 'deepseek-v4-flash'
                            or payload.get('max_tokens') != 60000 or payload.get('stream') is not False
                            or secret in raw):
                        self.reply(400, {'error': 'frozen API contract or credential boundary violation'})
                        return
                    identity = (domain, iteration, role, request_id)
                    with broker.lock:
                        prior = broker.cache.get(identity)
                    if prior is not None:
                        self.reply(*prior)
                        return
                    metadata = {'round': iteration, 'role': role, 'request_id': request_id,
                                'requested_model': payload['model'], 'max_output_tokens': 60000,
                                'local_context_limit': None,
                                'input_characters': len(json.dumps(payload['messages'], ensure_ascii=False))}
                    for attempt in range(3):
                        started = time.monotonic()
                        try:
                            with gate:
                                response = broker.client.post(url + '/chat/completions', content=raw,
                                    headers={'Authorization': 'Bearer ' + key,
                                             'Content-Type': 'application/json'})
                            status = response.status_code
                            try:
                                value = response.json()
                            except ValueError:
                                value = {'error': 'provider returned non-JSON data'}
                            # Never store or return upstream credentials, even on an error echo.
                            if secret in response.content:
                                status, value = 502, {'error': 'credential echo rejected'}
                        except Exception as exc:
                            status, value = 502, {'error': 'transport failure: ' + type(exc).__name__}
                        row = ledger.record(domain, {**metadata, 'attempt': attempt + 1,
                            'wall_seconds': time.monotonic() - started,
                            'gate': gate.snapshot()}, status, value)
                        if row['accounting'] != 'complete' and status != 429:
                            broker.accounting_failed.set()
                        if status == 429:
                            with broker.lock:
                                broker.rate_limit_responses += 1
                            if attempt < 2:
                                gate.defer(60)
                                continue
                        if status == 200 and row['accounting'] != 'complete':
                            status, value = 502, {'error': 'API usage missing; experiment must stop'}
                        with broker.lock:
                            broker.cache[identity] = (status, value)
                        self.reply(status, value)
                        return
                except (BrokenPipeError, ConnectionResetError):
                    pass
                except Exception:
                    self.close_connection = True

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.daemon_threads = True
        self.url = f'http://127.0.0.1:{self.server.server_port}/v1'
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.client.close()
