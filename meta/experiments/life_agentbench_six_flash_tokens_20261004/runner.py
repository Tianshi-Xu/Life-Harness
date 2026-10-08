"""Six rounds, one-round integration gate, durable API usage, then sealed TEST."""
import copy
import fcntl
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from driver import (DOMAINS, HERE, ROOT, ROUNDS, initial_state, phase, preservation_check,
                    progress, render_training_report, run_domain, seal_training, verify_seal)
from composition import write_active_chain
from protocol import atomic_json, digest, protocol, validate_protocol
from meta.life_optimizer.context import verify_usage


def build_payload(cfg, messages, tools):
    if (cfg['proposer_model'] != 'deepseek-v4-flash' or cfg['flash_max_tokens'] != 60000
            or cfg['flash_context_limit'] is not None):
        raise RuntimeError('Refuse accidental solver token limits on Flash')
    return {'model': cfg['proposer_model'], 'messages': copy.deepcopy(messages),
            'tools': tools, 'tool_choice': 'auto', 'max_tokens': 60000, 'stream': False}


def parallel(callback):
    result, errors = {}, {}
    with ThreadPoolExecutor(max_workers=len(DOMAINS)) as pool:
        futures = {pool.submit(callback, domain):domain for domain in DOMAINS}
        for future in as_completed(futures):
            domain = futures[future]
            try:
                result[domain] = future.result()
            except Exception as exc:
                errors[domain] = {'type': type(exc).__name__, 'detail': str(exc)[:6000]}
    if errors:
        atomic_json(HERE/'errors.json', errors)
        raise RuntimeError('Domain stage failed; inspect errors.json; no automatic paid restart')
    return result


def state_for(domain):
    return json.loads((HERE/domain/'state.json').read_text())


def audit_usage(ledger):
    result = ledger.summary()
    if not result['overall']['accounting_complete']:
        raise RuntimeError('Unknown or pending API token usage; stop before further optimization')
    for domain in DOMAINS:
        for row in state_for(domain)['rounds']:
            scope = f"{HERE.name}|{domain}|{row['round']}|life_optimizer"
            measured = result['groups'].get(scope)
            if measured is None or not measured['accounting_complete']:
                raise RuntimeError('Whole-round API receipts missing')
            if measured['total_tokens'] != row['usage']['total_tokens']:
                raise RuntimeError('API ledger versus proposer checkpoint token mismatch')
    return result


def first_round_valid(states, usage):
    return (usage['overall']['accounting_complete'] and len(states) == len(DOMAINS)
            and all(len(s['rounds']) >= 1 and s['rounds'][0]['candidate_successes'] is not None
                    and s['rounds'][0]['outcome'] == 'evaluated_candidate'
                    and s['rounds'][0]['proposer_requests'] > 0
                    and s['rounds'][0]['usage']['total_tokens'] > 0
                    and s['rounds'][0]['incumbent_infra_errors'] < 50
                    and s['rounds'][0]['candidate_infra_errors'] < 50
                    for s in states.values()))


def main():
    with (HERE/'.runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        validate_protocol()
        preservation_check()
        cfg = protocol()
        if (HERE/'FINAL_RESULTS.json').exists():
            verify_seal()
            return
        sys.path.insert(0, str(ROOT.parent/'ahe-native'))
        from ahe_native.metered_broker import Broker, RequestGate
        from ahe_native.token_accounting import TokenLedger, scoped_url
        from ahe_native.connection import credentials
        prior_usage = TokenLedger(HERE/'flash_requests.jsonl').summary()
        if not prior_usage['overall']['accounting_complete']:
            raise RuntimeError('Prior API request has unknown usage; no blind paid retry')
        atomic_json(HERE/'status.json', {'phase': 'setup_smoke', 'pid': os.getpid()})

        def smoke(domain):
            directory = HERE/domain
            directory.mkdir(exist_ok=True)
            reuse_path = HERE/'smoke_reuse_receipt.json'
            if reuse_path.exists():
                reused = json.loads(reuse_path.read_text()).get(domain)
                if reused is not None:
                    summary_path = directory/'evals/setup_smoke/summary.json'
                    if digest(summary_path) != reused['summary_sha256']:
                        raise RuntimeError('Previously passed native smoke changed')
                    from agentbench_evaluate import validate_cached_summary
                    summary = json.loads(summary_path.read_text())
                    validate_cached_summary(summary_path.parent,summary)
                    if summary['n'] != 2 or summary['infra_errors']:
                        raise RuntimeError('Smoke reuse must have two non-infra episodes')
                    progress(domain,phase='reused_setup_smoke',source_sha256=reused['summary_sha256'])
                    return {'n':2,'infra_errors':0,'reused':True}
            chain = write_active_chain(directory, initial_state(domain), 'setup_empty')
            python = ROOT/{'alfworld': 'AgentBench/.native/alfworld-venv/bin/python',
                           'dbbench': 'AgentBench/.native/core/bin/python',
                           'webshop': 'AgentBench/.native/webshop-venv/bin/python'}[domain]
            with (directory/'setup_smoke.log').open('a') as log:
                from driver import child_environment
                result = subprocess.run([str(python), '-u', str(HERE/'agentbench_evaluate.py'),
                    domain, 'train', 'setup_smoke', str(chain), '--batch', '0', '--task-limit', '2'],
                    cwd=ROOT, env=child_environment(domain), stdout=log, stderr=subprocess.STDOUT, timeout=1800)
            if result.returncode:
                raise RuntimeError('Native setup smoke failed')
            summary = json.loads((directory/'evals/setup_smoke/summary.json').read_text())
            if summary['n'] != 2 or summary['infra_errors']:
                raise RuntimeError('Native setup smoke contains infrastructure errors')
            return {'n': summary['n'], 'infra_errors': summary['infra_errors']}

        atomic_json(HERE/'setup_smoke.json', parallel(smoke))
        key, url = credentials()
        if url != cfg['proposer_host']+'/protocol/openai/v1':
            raise RuntimeError('Unexpected ModelRouter host')
        gate = RequestGate(8, min_interval=cfg['flash_min_interval_seconds'])
        broker = Broker(key, url, gate, rate_limit_retries=2, ledger_path=HERE/'flash_requests.jsonl')
        key_bytes = key.encode()
        if any(key_bytes in p.read_bytes() for p in HERE.rglob('*')
               if p.is_file() and '__pycache__' not in p.parts):
            broker.close()
            raise RuntimeError('Credential found on disk; refuse startup')
        del key, key_bytes
        atomic_json(HERE/'process.json', {'pid': os.getpid(), 'pgid': os.getpgid(0),
            'experiment': HERE.name, 'domains': list(DOMAINS), 'rounds_per_domain': 6,
            'flash_limit': 8, 'flash_max_tokens': 60000, 'flash_local_context_limit': None,
            'cumulative_optimizer_token_limit': None, 'solver_workers_per_domain': 50,
            'solver_max_context_tokens': 32768, 'solver_max_tokens': 4096,
            'credential_policy': 'host broker memory only; scrubbed native workers; no key in receipts',
            'transport_sha256': {name:digest(ROOT.parent/'ahe-native/ahe_native'/name)
                                for name in ('api_broker.py','metered_broker.py','token_accounting.py','connection.py')}})

        def completion(domain):
            log_path = HERE/domain/'flash_requests.jsonl'
            request_number = len(log_path.read_text().splitlines()) if log_path.exists() else 0
            def call(messages, tools):
                nonlocal request_number
                number = state_for(domain)['round']+1
                if not 1 <= number <= ROUNDS or (HERE/'training_seal.json').exists():
                    raise RuntimeError('Refuse a Flash call outside an unsealed TRAIN round')
                request_number += 1
                payload = build_payload(cfg, messages, tools)
                metadata = {'round': number, 'request_number': request_number,
                    'model': payload['model'], 'max_tokens': 60000,
                    'local_context_limit': None, 'message_count': len(messages)}
                endpoint = scoped_url(broker.url, experiment=HERE.name, domain=domain,
                                      round=number, role='life_optimizer')
                progress(domain, phase='flash_request', **metadata, gate=gate.snapshot())
                request = urllib.request.Request(endpoint+'/chat/completions', method='POST',
                    headers={'Authorization':'Bearer EMPTY', 'Content-Type':'application/json'},
                    data=json.dumps(payload, ensure_ascii=False).encode())
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with opener.open(request, timeout=2100) as response:
                    result = json.load(response)
                # Save every paid response BEFORE checking candidate/output validity.
                with log_path.open('a') as log:
                    log.write(json.dumps({**metadata, 'usage':result.get('usage',{})})+'\n')
                    log.flush()
                    os.fsync(log.fileno())
                verify_usage(result)
                if not result.get('choices') or result.get('error'):
                    raise RuntimeError('Flash response missing assistant choice')
                return result
            return call

        calls = {d:completion(d) for d in DOMAINS}
        stopped = threading.Event()
        def monitor():
            while not stopped.wait(30):
                broker.ledger.publish()
                render_training_report()
                from reporting import compare
                compare()
                atomic_json(HERE/'flash_metrics.json', {'gate':gate.snapshot(),
                    'rate_limit_responses':broker.rate_limit_responses, 'usage':broker.ledger.summary()['overall']})
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        try:
            atomic_json(HERE/'status.json', {'phase':'first_round_training','pid':os.getpid()})
            states = parallel(lambda d:run_domain(d,calls[d],stop_after=1))
            usage = audit_usage(broker.ledger)
            passed = first_round_valid(states, usage)
            atomic_json(HERE/'first_round_gate.json', {'passed':passed,
                'rule':cfg['first_round_gate'], 'usage':usage['overall'],
                'domains':{d:{k:states[d]['rounds'][0].get(k) for k in
                    ('round','incumbent_successes','candidate_successes','accepted','usage')}
                    for d in DOMAINS}})
            if not passed:
                raise RuntimeError('First-round integration gate failed; rounds 2-6 will NOT start')
            atomic_json(HERE/'status.json', {'phase':'remaining_training','pid':os.getpid()})
            parallel(lambda d:run_domain(d,calls[d]))
            audit_usage(broker.ledger)
        finally:
            stopped.set()
            thread.join(timeout=40)
            broker.close()
            atomic_json(HERE/'flash_metrics.json', {'gate':gate.snapshot(),
                'rate_limit_responses':broker.rate_limit_responses, 'usage':broker.ledger.summary()['overall']})
            preservation_check()
            render_training_report()
        seal_training()
        atomic_json(HERE/'status.json', {'phase':'final_test','pid':os.getpid()})
        ledger_sha = digest(HERE/'flash_requests.jsonl')
        results = parallel(lambda d:phase(d,'test','final_test',
            write_active_chain(HERE/d,state_for(d),'final_test'),0))
        if digest(HERE/'flash_requests.jsonl') != ledger_sha:
            raise RuntimeError('External optimization call after seal')
        atomic_json(HERE/'FINAL_RESULTS.json', {'split':'test','domains':results,
            'optimizer_usage':broker.ledger.summary(), 'checkpoint_policy':'final retained strict-TRAIN incumbent',
            'prior_test_exposure':True, 'test_used_for_optimization':False})
        from reporting import compare
        compare()
        preservation_check()
        atomic_json(HERE/'status.json', {'phase':'complete','pid':os.getpid()})


if __name__ == '__main__':
    if sys.argv[1:] == ['--launch']:
        with (HERE/'runner.log').open('a') as log:
            proc = subprocess.Popen([sys.executable,'-u',str(HERE/'runner.py')], cwd=HERE,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        atomic_json(HERE/'launch.json', {'pid':proc.pid,'started_at':time.time(),'log':str(HERE/'runner.log')})
        print(json.dumps({'launched_pid':proc.pid,'log':str(HERE/'runner.log')}))
    elif sys.argv[1:] == ['--preflight']:
        validate_protocol()
        preservation_check()
        print(json.dumps({'status':'offline_preflight_passed','domains':DOMAINS,'rounds':ROUNDS}))
    elif not sys.argv[1:]:
        try:
            main()
        except Exception as exc:
            atomic_json(HERE/'status.json', {'phase':'stopped','pid':os.getpid(),
                'error':f'{type(exc).__name__}: {exc}'})
            raise
    else:
        raise SystemExit('usage: runner.py [--launch|--preflight]')
