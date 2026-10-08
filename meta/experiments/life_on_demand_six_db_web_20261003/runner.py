"""One credential-only broker and two independently locked six-round domains."""
import copy
import fcntl
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys
import urllib.request
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from driver import (DOMAINS, HERE, ROOT, final_test, preservation_check, progress,
                    render_training_report, run_domain, seal_training, validate_protocol)
from protocol import atomic_json, digest, protocol

def broker_modules():
    adapter = ROOT.parent / 'ahe-native'
    sys.path.insert(0, str(adapter))
    from ahe_native.api_broker import Broker, RequestGate
    from ahe_native.connection import credentials
    return Broker, RequestGate, credentials

def main():
    with (HERE / '.runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        validate_protocol()
        preservation_check()
        cfg = protocol()
        Broker, RequestGate, credentials = broker_modules()
        key, url = credentials()
        if url != cfg['proposer_host'] + '/protocol/openai/v1':
            raise RuntimeError('Unexpected ModelRouter host')
        gate = RequestGate(cfg['flash_concurrency'], min_interval=cfg['flash_min_interval_seconds'])
        broker = Broker(key, url, request_gate=gate, rate_limit_retries=2)
        del key
        atomic_json(HERE / 'process.json', {'pid': os.getpid(), 'pgid': os.getpgid(0),
                    'experiment': HERE.name, 'flash_limit': gate.limit,
                    'solver_workers_per_domain': cfg['solver_workers_per_domain'],
                    'credential_policy': 'host broker memory only; native workers scrub secret env variables',
                    'transport_sha256': {name: digest(ROOT.parent / 'ahe-native/ahe_native' / name)
                                         for name in ('api_broker.py', 'connection.py')}})
        def completion(domain):
            counter = 0
            def call(messages, tools):
                nonlocal counter
                counter += 1
                progress(domain, phase='flash_request', domain_request_number=counter, gate=gate.snapshot())
                payload = {'model': cfg['proposer_model'], 'messages': copy.deepcopy(messages),
                           'tools': tools, 'tool_choice': 'auto', 'max_tokens': 8192, 'stream': False}
                request = urllib.request.Request(broker.url + '/chat/completions', method='POST',
                    headers={'Authorization': 'Bearer EMPTY', 'Content-Type': 'application/json'},
                    data=json.dumps(payload, ensure_ascii=False).encode())
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with opener.open(request, timeout=2100) as response:
                    result = json.load(response)
                if not result.get('choices') or result.get('error'):
                    raise RuntimeError('Flash response missing valid assistant choice')
                # Metadata only: never persist messages, auth headers, or actual credentials.
                with (HERE / domain / 'flash_requests.jsonl').open('a') as log:
                    log.write(json.dumps({'request_number': counter, 'usage': result.get('usage', {}),
                               'finish_reason': result['choices'][0].get('finish_reason'), 'gate': gate.snapshot()}) + '\n')
                return result
            return call
        errors = {}
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = {pool.submit(run_domain, d, completion(d)): d for d in DOMAINS}
                for future in as_completed(futures):
                    domain = futures[future]
                    try:
                        future.result()
                    except Exception as exc:
                        errors[domain] = f'{type(exc).__name__}: {exc}'[:12000]
                        progress(domain, phase='stopped', error=errors[domain])
            if errors:
                atomic_json(HERE / 'errors.json', errors)
                raise RuntimeError('Training stopped in one or more domains; inspect errors.json. Completed evals/proposals preserved.')
            seal_training()
        finally:
            atomic_json(HERE / 'flash_metrics.json', {'gate': gate.snapshot(),
                       'rate_limit_responses': broker.rate_limit_responses,
                       'reasoning_restores': broker.reasoning.hits, 'reasoning_misses': broker.reasoning.misses})
            broker.close()
            preservation_check()
            render_training_report()
        # Optimizer transport is closed before any test traces are created.
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = dict(zip(DOMAINS, pool.map(final_test, DOMAINS)))
        atomic_json(HERE / 'final_results.json', {d: {k: r[k] for k in
                    ('n', 'successes', 'accuracy', 'infra_errors', 'snapshot_sha256')} for d, r in results.items()})
        preservation_check()
        print(json.dumps({'status': 'complete', 'final_results': str(HERE / 'final_results.json')}), flush=True)

if __name__ == '__main__':
    if sys.argv[1:] == ['--launch']:
        with (HERE / 'runner.log').open('a') as log:
            child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve())],
                cwd=HERE, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
        print(json.dumps({'launched_pid': child.pid, 'log': str(HERE / 'runner.log')}))
    elif len(sys.argv) == 1:
        main()
    else:
        raise SystemExit('usage: runner.py [--launch]')
