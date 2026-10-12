"""Launch, gate the first round, then complete six TRAIN rounds and final TEST."""
import fcntl
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from api_broker import TokenBroker
from common import AB, DOMAINS, HERE, ROOT, digest, manifest, preservation_check, progress, save, verify_frozen
from driver import phase, render_report, run_round, seal_training, state_for, upstream
from usage import Ledger


def parallel(function):
    errors, results = {}, {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(function, d): d for d in DOMAINS}
        for future in as_completed(futures):
            d = futures[future]
            try:
                results[d] = future.result()
            except Exception as exc:
                errors[d] = f'{type(exc).__name__}: {exc}'
                progress(d, phase='stopped', error=errors[d])
    if errors:
        save(HERE / 'errors.json', errors)
        raise RuntimeError('Domain stopped; check errors.json. No automatic rerun.')
    return results


def first_round_valid(states, ledger):
    if set(states) != set(DOMAINS):
        return False
    return all(len(s['rounds']) >= 1 and s['rounds'][0]['candidate_successes'] is not None
        and s['rounds'][0]['candidate_infra_errors'] <= 2
        and ledger.summarize(d, 1)['accounting_complete']
        and set(ledger.summarize(d, 1)['by_role']) == {'parent', 'analysis', 'implementation'}
        and all(v['total_tokens'] > 0 for v in ledger.summarize(d, 1)['by_role'].values())
        for d, s in states.items())


def main():
    with (HERE / '.runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg = verify_frozen(data=True)
        preservation_check()
        sys.path.insert(0, str(ROOT / "integrations/ahe-native"))
        from ahe_native.isolation import prepare_base, ensure_shell_utilities
        prepare_base()
        ensure_shell_utilities()
        if (HERE / 'FINAL_RESULTS.json').exists():
            print(json.dumps({'status': 'already_complete'}), flush=True)
            return
        ledger = Ledger(HERE)
        if not ledger.summarize()['accounting_complete']:
            raise RuntimeError('A prior API request has unknown usage; do not automatically spend more tokens')
        # Native setup must succeed before any paid optimization.
        def smoke(d):
            summary = phase(d, 'train', 0, 'setup_smoke', HERE / d / 'workspace/agents/baseline_zero.py', smoke_count=2)
            if summary['infra_errors']:
                raise RuntimeError('Native setup smoke failed')
            return summary
        save(HERE / 'status.json', {'phase': 'setup_smoke', 'pid': os.getpid()})
        parallel(smoke)
        sys.path.insert(0, str(ROOT / 'integrations/ahe-native'))
        from ahe_native.api_broker import RequestGate
        from ahe_native.connection import credentials
        modules = {d: upstream(d) for d in DOMAINS}
        key, url = credentials()
        if url != cfg['teacher']['host'] + '/protocol/openai/v1':
            raise RuntimeError('Unexpected ModelRouter host')
        gate = RequestGate(8, min_interval=cfg['flash_min_interval_seconds'])
        broker = TokenBroker(key, url, gate, ledger, DOMAINS)
        key_bytes = key.encode()
        if any(key_bytes in p.read_bytes() for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts):
            broker.close()
            raise RuntimeError('A credential was found on disk; refuse experiment startup')
        del key, key_bytes
        save(HERE / 'process.json', {'pid': os.getpid(), 'pgid': os.getpgid(0), 'experiment': HERE.name,
            'domains': DOMAINS, 'rounds_per_domain': 6, 'flash_model': 'deepseek-v4-flash',
            'flash_concurrency_limit': 8, 'flash_max_output_tokens': 60000,
            'flash_local_context_limit': None, 'cumulative_optimizer_token_limit': None,
            'solver_workers_per_domain': 50, 'solver_total_context': 32768, 'solver_max_output': 4096,
            'credential_policy': 'host memory only; stripped child env; isolated optimizer filesystem',
            'backend': cfg['optimizer_backend']})
        stopped = threading.Event()

        def monitor():
            while not stopped.wait(15):
                render_report(ledger)
                save(HERE / 'flash_metrics.json', {'gate': gate.snapshot(),
                    'rate_limit_responses': broker.rate_limit_responses, 'usage': ledger.summarize()})
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        try:
            save(HERE / 'status.json', {'phase': 'first_round_training', 'pid': os.getpid()})
            states = parallel(lambda d: run_round(d, modules[d], broker.url, ledger)
                              if not state_for(d)['rounds'] else state_for(d))
            gate_ok = first_round_valid(states, ledger)
            save(HERE / 'first_round_gate.json', {'passed': gate_ok, 'rule': cfg['first_round_gate'],
                'domains': {d: s['rounds'][0] if s['rounds'] else None for d, s in states.items()}})
            render_report(ledger)
            if not gate_ok:
                raise RuntimeError('First-round integration gate failed; rounds 2-6 will NOT start')
            save(HERE / 'status.json', {'phase': 'remaining_training', 'pid': os.getpid()})

            def finish(d):
                while len(state_for(d)['rounds']) < 6:
                    run_round(d, modules[d], broker.url, ledger)
                return state_for(d)
            parallel(finish)
        finally:
            stopped.set()
            thread.join(timeout=20)
            broker.close()
            save(HERE / 'flash_metrics.json', {'gate': gate.snapshot(),
                'rate_limit_responses': broker.rate_limit_responses, 'usage': ledger.summarize()})
            render_report(ledger)
            preservation_check()
        # Closing the optimizer transport before sealing prevents post-seal model calls.
        seal_training(ledger)
        save(HERE / 'status.json', {'phase': 'final_test', 'pid': os.getpid()})
        results = parallel(lambda d: phase(d, 'test', 0, 'final_test',
            HERE / d / 'workspace/agents' / state_for(d)['incumbent']))
        save(HERE / 'FINAL_RESULTS.json', {'split': 'test', 'domains': results,
            'optimizer_usage': ledger.summarize(), 'budget_label': cfg['comparison_label']})
        preservation_check()
        save(HERE / 'status.json', {'phase': 'complete', 'pid': os.getpid()})
        print(json.dumps({'status': 'complete', 'results': str(HERE / 'FINAL_RESULTS.json')}), flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['--launch']:
        with (HERE / 'runner.log').open('a') as log:
            proc = subprocess.Popen([sys.executable, '-u', str(HERE / 'runner.py')], cwd=HERE,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        save(HERE / 'launch.json', {'pid': proc.pid, 'started_at': time.time(), 'log': str(HERE / 'runner.log')})
        print(json.dumps({'launched_pid': proc.pid, 'log': str(HERE / 'runner.log')}))
    elif not sys.argv[1:]:
        try:
            main()
        except Exception as exc:
            save(HERE / 'status.json', {'phase': 'stopped', 'pid': os.getpid(),
                                      'error': f'{type(exc).__name__}: {exc}'})
            raise
    else:
        raise SystemExit('usage: runner.py [--launch]')
