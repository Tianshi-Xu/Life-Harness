"""One shared local Qwen request budget, independent of optimizer/simulator APIs."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import subprocess
import threading

import httpx

from .common import ROOT, META_TRANSPORT, child_env


class SolverProxy:
    def __init__(self):
        self.counter = subprocess.Popen([str(ROOT / 'AgentBench/.native/core/bin/python'), '-u',
            str(META_TRANSPORT / 'tokenizer_worker.py')], env=child_env(),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        self.lock = threading.Lock()
        self.client = httpx.Client(timeout=600, trust_env=False)
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, status, body):
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                try:
                    if self.path != '/v1/chat/completions':
                        self.send_error(404)
                        return
                    payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                    if payload.get('model') != 'qwen3-4b' or payload.get('stream'):
                        self.send_error(400, 'frozen local solver model/non-streaming contract')
                        return
                    # These are benchmark settings, not optimized policy content.
                    payload.update(max_tokens=4096, temperature=0.0)
                    payload.setdefault('chat_template_kwargs', {})['enable_thinking'] = False
                    with proxy.lock:
                        if proxy.counter.poll() is not None:
                            raise RuntimeError('tokenizer process exited')
                        proxy.counter.stdin.write(json.dumps({'messages': payload['messages'],
                            'tools': payload.get('tools') or []}) + '\n')
                        proxy.counter.stdin.flush()
                        count = json.loads(proxy.counter.stdout.readline())
                    if 'length' not in count:
                        raise RuntimeError('local token count failed')
                    if count['length'] + 4096 > 32768:
                        self.reply(400, json.dumps({'error': {'type': 'context_length_exceeded',
                            'message': 'solver_context_limit_32768', 'code': 'context_length_exceeded'}}).encode())
                        return
                    response = proxy.client.post('http://127.0.0.1:8400/v1/chat/completions',
                        json=payload, headers={'Authorization': 'Bearer EMPTY'})
                    self.reply(response.status_code, response.content)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                except Exception:
                    self.reply(502, b'{"error":"local solver transport failed"}')

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.daemon_threads = True
        self.url = f'http://127.0.0.1:{self.server.server_port}/v1'
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.client.close()
        self.counter.terminate()
        self.counter.wait(timeout=10)
