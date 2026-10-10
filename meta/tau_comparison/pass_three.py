"""Test-only completion of sealed Meta/AHE Tau snapshots to three trials."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait
import hashlib
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

from .common import (AHE_NATIVE, ALI_URL, DOMAINS, ROOT, TAU, aliyun_key,
                     child_env, digest, read, save, summarize_rows, user_model)


def pass_three(ids, rows):
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('nonempty unique fixed TEST task IDs required')
    score = summarize_rows(ids, 3, rows)
    outcomes = {str(task): [None] * 3 for task in ids}
    for row in rows:
        reward = float((row.get('reward_info') or {}).get('reward', row.get('reward', 0)))
        outcomes[str(row['task_id'])][row.get('trial', 0)] = reward
    counts = [sum(values[t] == 1 for values in outcomes.values()) for t in range(3)]
    complete = sum(all(value == 1 for value in values) for values in outcomes.values())
    score.update(trials=3, per_task=outcomes, trial_successes=counts,
                 trial_accuracy=[n / len(ids) for n in counts],
                 pass_caret_3_successes=complete, pass_caret_3=complete / len(ids),
                 metric='fraction of fixed TEST tasks correct in all three trials; not pass@3')
    return score


def audit(parent):
    """Read-only audit before any simulator API request."""
    sys.path.insert(0, str(AHE_NATIVE))
    from ahe_native.backend import workspace_hash
    protocol, seal = read(parent / 'protocol.json'), read(parent / 'training_seal.json')
    final = read(parent / 'FINAL_RESULTS.json')['methods']
    frozen = read(parent / 'frozen_sources.json')
    for name, sha in frozen.items():
        if '/TauBench/' in name and digest(name) != sha:
            raise RuntimeError('frozen Tau environment/scoring/data changed: ' + name)
    plan = {}
    for domain in DOMAINS:
        ids = protocol['domains_spec'][domain]['test_task_ids']
        if set(ids) != set(read(TAU / 'data/tau2/domains' / domain / 'split_tasks.json')['test']):
            raise RuntimeError('native TEST identities changed')
        for method in ('meta', 'ahe'):
            key = method + ':' + domain
            if method == 'meta':
                phase = parent / 'meta' / domain / 'evals/final_test'
                state = read(parent / 'meta' / domain / 'state.json')
                snapshot = parent / 'meta' / domain / 'workspace/agents' / state['incumbent']
                rows = read(phase / 'native/results.json')['simulations']
                if digest(snapshot) != seal['snapshots'][key]:
                    raise RuntimeError('Meta sealed candidate changed')
                if digest(phase / 'native/results.json') != read(phase / 'summary.json')['native_results_sha256']:
                    raise RuntimeError('historical Meta TEST receipt changed')
                seeds = random.Random(300)
                if {row['seed'] for row in rows} != {seeds.randint(0, 1000000)}:
                    raise RuntimeError('Meta historical seed differs from native trial zero')
            else:
                phase = parent / 'final_test/ahe' / domain
                snapshot = parent / 'test_snapshots' / domain
                if workspace_hash(snapshot) != seal['snapshots'][key]:
                    raise RuntimeError('AHE sealed candidate changed')
                rows = [{**read(p), 'reward': read(p)['success']} for p in phase.glob('*/record.json')]
            score = summarize_rows(ids, 1, rows)
            if score['missing_episodes'] or score['per_task'] != final[key]['per_task']:
                raise RuntimeError('historical TEST is incomplete or disagrees with final results')
            if score['infra_errors']:
                raise RuntimeError('historical infrastructure failures need explicit reuse handling')
            plan[key] = {'ids': ids, 'snapshot': str(snapshot), 'snapshot_sha256': seal['snapshots'][key],
                         'phase': str(phase), 'trial_zero': rows,
                         'records_sha256': {str(p): digest(p) for p in
                             ([phase / 'native/results.json'] if method == 'meta' else phase.glob('*/record.json'))}}
    return plan


class Tests:
    def __init__(self, root, plan):
        sys.path.insert(0, str(AHE_NATIVE))
        from ahe_native.api_broker import RequestGate
        from ahe_native.connection import credentials
        from ahe_native.metered_broker import Broker
        from .solver_proxy import SolverProxy
        self.root, self.plan = root, plan
        self.gate = RequestGate(8, min_interval=1.25)
        key, url = credentials()
        self.mr = Broker(key, url, self.gate, rate_limit_retries=3,
                         ledger_path=root / 'ledgers/mr_requests.jsonl',
                         ledger_summary_path=root / 'ledgers/MR_TEST_USAGE.json')
        self.ali = Broker(aliyun_key(), ALI_URL, allowed_model='qwen3.8-flash',
                          ledger_path=root / 'ledgers/aliyun_requests.jsonl',
                          ledger_summary_path=root / 'ledgers/ALIYUN_TEST_USAGE.json')
        self.mr.blocked_roles.update({'debugger', 'evolve', 'life_optimizer', 'unknown'})
        self.ali.blocked_roles.update(self.mr.blocked_roles)
        self.solver = SolverProxy()

    def route(self, method, domain, role):
        from ahe_native.token_accounting import scoped_url
        broker = self.mr if domain == 'telecom' else self.ali
        return scoped_url(broker.url, experiment=self.root.name + '_' + method,
                          domain=domain, round=6, role=role)

    def meta(self, domain):
        spec = self.plan['meta:' + domain]
        folder = self.root / 'meta' / domain
        folder.mkdir(parents=True)
        candidate = folder / 'candidate.py'
        shutil.copy2(spec['snapshot'], candidate)
        ids = folder / 'task_ids.txt'
        ids.write_text('\n'.join(spec['ids']) + '\n')
        native = folder / 'native'
        native.mkdir()
        # Native checkpoint resume preserves trial 0, and uses its original
        # seed stream (seed=300) for trial 1/2. Never touch the ancestor file.
        shutil.copy2(Path(spec['phase']) / 'native/results.json', native / 'results.json')
        request = {'domain': domain, 'split': 'test', 'task_id_file': str(ids),
                   'trials': 3, 'concurrency': 50, 'candidate': str(candidate),
                   'solver_base': self.solver.url, 'user_model': user_model(domain),
                   'user_base': self.route('meta', domain, 'user_simulator'),
                   'judge_base': self.route('meta', domain, 'judge'),
                   'save_to': str(native), 'resume_trials': True}
        save(folder / 'request.json', request)
        with (folder / 'driver.log').open('w') as log:
            process = subprocess.run([str(TAU / '.venv/bin/python'), '-m', 'meta.tau_comparison.meta_evaluate',
                '--request', str(folder / 'request.json')], cwd=ROOT,
                env=child_env(PYTHONPATH=str(ROOT)), stdout=log, stderr=subprocess.STDOUT, timeout=21600)
        if process.returncode:
            raise RuntimeError('Meta TEST driver failed: ' + domain)
        rows = read(native / 'results.json')['simulations']
        if digest(candidate) != spec['snapshot_sha256']:
            raise RuntimeError('Meta test snapshot mutated')
        self.finish('meta', domain, rows)

    def ahe(self, domain):
        from ahe_native.backend import episode_timeout, interpreter, workspace_hash
        from ahe_native.capabilities import restricted_policy
        spec = self.plan['ahe:' + domain]
        folder = self.root / 'ahe' / domain
        snapshot = folder / 'workspace'
        shutil.copytree(spec['snapshot'], snapshot)

        def episode(task, trial):
            directory = folder / hashlib.sha256(f'{task}:{trial}'.encode()).hexdigest()[:16]
            directory.mkdir()
            request = {'domain': domain, 'split': 'test', 'task_id': task, 'trial': trial,
                       'workspace': str(snapshot), 'agent_config': 'code_agent.yaml',
                       'trace_dir': str(directory / 'agent'), 'capability_policy': restricted_policy(domain)}
            save(directory / 'request.json', request)
            env = child_env()
            # Only scoped loopback connections and EMPTY enter native workers.
            env.update(PYTHONPATH=str(AHE_NATIVE), LLM_API_KEY='EMPTY',
                LLM_BASE_URL=self.solver.url, LLM_MODEL='qwen3-4b', AHE_USER_API_KEY='EMPTY',
                AHE_TAU_USER_API_BASE=self.route('ahe', domain, 'user_simulator'),
                AHE_TAU_JUDGE_API_BASE=self.route('ahe', domain, 'judge'),
                AHE_TAU_USER_MODEL=user_model(domain), TAU_USER_LLM_MIN_INTERVAL_SECONDS='0')
            with (directory / 'environment.log').open('w') as log:
                process = subprocess.run([str(interpreter(domain)), '-m', 'ahe_native.native_worker',
                    '--request', str(directory / 'request.json'), '--out', str(directory / 'record.json')],
                    cwd=AHE_NATIVE, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=episode_timeout(domain))
            if process.returncode or not (directory / 'record.json').is_file():
                raise RuntimeError('AHE TEST worker failed: ' + str(directory))
            row = read(directory / 'record.json')
            return {**row, 'reward': row['success']}

        with ThreadPoolExecutor(max_workers=50) as pool:
            futures = [pool.submit(episode, task, trial) for task in spec['ids'] for trial in (1, 2)]
            rows = spec['trial_zero'] + [future.result() for future in futures]
        if workspace_hash(snapshot) != spec['snapshot_sha256']:
            raise RuntimeError('AHE test snapshot mutated')
        self.finish('ahe', domain, rows)

    def finish(self, method, domain, rows):
        score = pass_three(self.plan[method + ':' + domain]['ids'], rows)
        if score['missing_episodes']:
            raise RuntimeError('cannot publish pass^3 with missing trials')
        # Reused first-trial outcomes must remain exactly unchanged.
        old = summarize_rows(self.plan[method + ':' + domain]['ids'], 1,
                             self.plan[method + ':' + domain]['trial_zero'])['per_task']
        if any(values[0] != old[task][0] for task, values in score['per_task'].items()):
            raise RuntimeError('historical trial zero was not preserved')
        save(self.root / method / domain / 'summary.json', score)

    def close(self):
        for name in ('solver', 'mr', 'ali'):
            if getattr(self, name, None):
                getattr(self, name).close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    parent = args.parent.resolve()
    plan = audit(parent)
    if args.audit_only:
        print({key: {'tasks': len(value['ids']), 'trial_zero_reused': True} for key, value in plan.items()})
        return
    if not args.name.replace('_', '').replace('-', '').isalnum() or len(args.name) > 120:
        raise ValueError('invalid experiment name')
    root = ROOT / 'meta/experiments' / args.name
    root.mkdir()  # Existing paid runs are never overwritten or blindly replayed.
    save(root / 'reuse_receipt.json', {'parent': str(parent),
        'parent_seal_sha256': digest(parent / 'training_seal.json'), 'plan': plan})
    save(root / 'protocol.json', {'methods': ['meta', 'ahe'], 'domains': list(DOMAINS),
        'split': 'test', 'trials': 3, 'reused_episodes': 200, 'new_episodes': 400,
        'selection': 'same original sealed snapshots; no optimization or test-based selection',
        'metric': 'pass^3 = all three trials correct on the same task',
        'global_mr_concurrency': 8, 'optimizer_tokens': 0,
        'accounting': 'simulator/judge TEST costs only, excluded from evolve costs',
        'runner_sha256': digest(__file__), 'meta_evaluate_sha256': digest(Path(__file__).with_name('meta_evaluate.py'))})
    tests = Tests.__new__(Tests)
    try:
        tests.__init__(root, plan)
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = {method + ':' + domain: pool.submit(getattr(tests, method), domain)
                       for method in ('meta', 'ahe') for domain in DOMAINS}
            while True:
                pending = [f for f in futures.values() if not f.done()]
                save(root / 'status.json', {'phase': 'running' if pending else 'finished', 'pid': os.getpid(),
                    'completed': [k for k, f in futures.items() if f.done() and f.exception() is None],
                    'failed': [k for k, f in futures.items() if f.done() and f.exception() is not None],
                    'mr_gate': tests.gate.snapshot(), 'updated_at': time.time()})
                if not pending:
                    break
                wait(pending, timeout=30)
            for future in futures.values():
                future.result()
        for spec in plan.values():
            for path, sha in spec['records_sha256'].items():
                if digest(path) != sha:
                    raise RuntimeError('ancestor TEST record mutated')
        save(root / 'FINAL_RESULTS.json', {'split': 'test', 'trials': 3, 'parent': str(parent),
             'optimizer_tokens': 0, 'methods': {method + ':' + domain: read(root / method / domain / 'summary.json')
                 for method in ('meta', 'ahe') for domain in DOMAINS}})
        print('Completed: ' + str(root / 'FINAL_RESULTS.json'), flush=True)
    except Exception as exc:
        save(root / 'error.json', {'type': type(exc).__name__, 'message': str(exc)})
        raise
    finally:
        tests.close()


if __name__ == '__main__':
    main()
