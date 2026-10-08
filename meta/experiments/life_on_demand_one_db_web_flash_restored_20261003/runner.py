"""One credential-only broker, exactly one TRAIN iteration in each domain."""
import copy
import fcntl
import json
import os
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from driver import (DOMAINS, HERE, ROOT, preservation_check, progress,
                    render_training_report, run_domain, seal_training, verify_seal,
                    validate_protocol)
from protocol import atomic_json, digest, protocol


def broker_modules():
    sys.path.insert(0, str(ROOT.parent / 'ahe-native'))
    from ahe_native.api_broker import Broker, RequestGate
    from ahe_native.connection import credentials
    return Broker, RequestGate, credentials


def build_payload(cfg, messages, tools):
    if (cfg['proposer_model'] != 'deepseek-v4-flash' or cfg['flash_max_tokens'] != 8192
            or cfg['flash_context_limit'] is not None):
        raise RuntimeError('Refuse accidental solver token limits on Flash')
    return {'model': cfg['proposer_model'], 'messages': copy.deepcopy(messages),
            'tools': tools, 'tool_choice': 'auto', 'max_tokens': 8192, 'stream': False}


def main():
    with (HERE / '.runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        validate_protocol()
        preservation_check()
        cfg = protocol()
        if (cfg['rounds_per_domain'] != 1 or cfg['stop_after_round'] != 1
                or cfg['run_test'] is not False):
            raise RuntimeError('Only one TRAIN round per domain is authorized')
        if (HERE / 'round_results.json').is_file():
            verify_seal()
            print(json.dumps({'status': 'already_complete', 'round_results': str(HERE/'round_results.json')}), flush=True)
            return
        Broker, RequestGate, credentials = broker_modules()
        key, url = credentials()
        if url != cfg['proposer_host'] + '/protocol/openai/v1':
            raise RuntimeError('Unexpected ModelRouter host')
        gate = RequestGate(cfg['flash_concurrency'], min_interval=cfg['flash_min_interval_seconds'])
        broker = Broker(key, url, request_gate=gate, rate_limit_retries=2)
        del key
        atomic_json(HERE / 'process.json', {'pid': os.getpid(), 'pgid': os.getpgid(0),
            'experiment': HERE.name, 'rounds_per_domain': 1, 'run_test': False,
            'flash_limit': gate.limit, 'flash_max_tokens': 8192, 'flash_local_context_limit': None,
            'solver_workers_per_domain': cfg['solver_workers_per_domain'],
            'solver_max_context_tokens': 32768, 'solver_max_tokens': 4096,
            'credential_policy': 'host broker memory only; native workers scrub secret env variables',
            'transport_sha256': {name: digest(ROOT.parent/'ahe-native/ahe_native'/name)
                                for name in ('api_broker.py','connection.py')}})

        def completion(domain):
            log_path = HERE/domain/'flash_requests.jsonl'
            request_number = max((json.loads(line)['request_number']
                                  for line in log_path.read_text().splitlines()), default=0) if log_path.exists() else 0

            def call(messages, tools):
                nonlocal request_number
                state = json.loads((HERE/domain/'state.json').read_text())
                if state['round'] != 0:
                    raise RuntimeError('Refuse a Flash request after the only round')
                payload = build_payload(cfg, messages, tools)
                request_number += 1
                metadata = {'round': 1, 'request_number': request_number,
                    'model': payload['model'], 'max_tokens': payload['max_tokens'],
                    'local_context_limit': None, 'message_count': len(messages),
                    'input_characters': len(json.dumps(messages, ensure_ascii=False))}
                progress(domain, phase='flash_request', **metadata, gate=gate.snapshot())
                with (HERE/domain/'flash_started.jsonl').open('a') as log:
                    log.write(json.dumps(metadata)+'\n')
                request = urllib.request.Request(broker.url+'/chat/completions', method='POST',
                    headers={'Authorization':'Bearer EMPTY', 'Content-Type':'application/json'},
                    data=json.dumps(payload, ensure_ascii=False).encode())
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with opener.open(request, timeout=2100) as response:
                    result = json.load(response)
                if not result.get('choices') or result.get('error'):
                    raise RuntimeError('Flash response missing valid assistant choice')
                choice = result['choices'][0]
                with log_path.open('a') as log:
                    log.write(json.dumps({**metadata, 'usage':result.get('usage',{}),
                        'finish_reason':choice.get('finish_reason'),
                        'tool_calls':len(choice['message'].get('tool_calls') or []),
                        'content_characters':len(choice['message'].get('content') or ''),
                        'reasoning_characters':len(choice['message'].get('reasoning_content') or ''),
                        'gate':gate.snapshot()})+'\n')
                return result
            return call

        errors = {}
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = {pool.submit(run_domain, d, completion(d)):d for d in DOMAINS}
                for future in as_completed(futures):
                    domain = futures[future]
                    try:
                        state = future.result()
                        if state['round'] != 1 or len(state['rounds']) != 1:
                            raise RuntimeError('Expected exactly one recorded TRAIN decision')
                    except Exception as exc:
                        errors[domain] = f'{type(exc).__name__}: {exc}'[:12000]
                        progress(domain, phase='stopped', error=errors[domain])
            if errors:
                atomic_json(HERE/'errors.json', errors)
                raise RuntimeError('One-round validation stopped; no further round or TEST will run')
            seal_training()
            states = {d:json.loads((HERE/d/'state.json').read_text()) for d in DOMAINS}
            rows = {d:state['rounds'][0] for d,state in states.items()}
            atomic_json(HERE/'round_results.json', {'split':'train', 'rounds_per_domain':1,
                'test_run':False, 'domains':{d:{key:row.get(key) for key in
                    ('round','train_batch','incumbent_successes','candidate_successes','accepted',
                     'outcome','failure_code','incumbent_infra_errors','candidate_infra_errors',
                     'proposer_requests','cases_opened','active_snapshot_sha256')}
                    for d,row in rows.items()}})
            print(json.dumps({'status':'one_round_finished', 'round_results':str(HERE/'round_results.json'),
                'candidate_measured':{d:rows[d]['candidate_successes'] is not None for d in DOMAINS},
                'will_run_another_round':False, 'test_run':False}), flush=True)
        finally:
            atomic_json(HERE/'flash_metrics.json', {'gate':gate.snapshot(),
                'rate_limit_responses':broker.rate_limit_responses,
                'reasoning_restores':broker.reasoning.hits, 'reasoning_misses':broker.reasoning.misses})
            broker.close()
            preservation_check()
            render_training_report()


if __name__ == '__main__':
    if sys.argv[1:] == ['--launch']:
        with (HERE/'runner.log').open('a') as log:
            child = subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve())],
                cwd=HERE, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
        print(json.dumps({'launched_pid':child.pid, 'log':str(HERE/'runner.log')}))
    elif len(sys.argv) == 1:
        main()
    else:
        raise SystemExit('usage: runner.py [--launch]')
