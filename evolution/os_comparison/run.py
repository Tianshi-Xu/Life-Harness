"""Three concurrent native OS six-round runs with one optimizer request gate."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

from evolution import harness_hooks as hooks
from evolution.life_evolve import profile
from evolution.life_evolve.context import ContextLimit, OUTPUT_LIMIT, verify_usage
from evolution.life_evolve.tau_run import composition_module
from evolution.tau_comparison.common import AHE_NATIVE, ROOT, child_env, read, transport_modules
from evolution.tau_comparison.meta_port import upstream
from evolution.tau_comparison.run import Comparison as TauBoundary, select_optimizer_groups
from evolution.tau_comparison.solver_proxy import SolverProxy
from .meta_port import MetaPort, prepare_workspace
from .protocol import digest, save, verify_sources


def life_usage(summary, experiment, number=None):
    groups = [g for g in summary['groups'].values() if g['experiment'] == experiment
              and g['domain'] == 'os' and g['role'] == 'life_optimizer'
              and (number is None or g['round'] == number)]
    fields = ('attempts', 'known_usage_calls', 'unknown_usage_calls', 'pending_calls',
              'input_tokens', 'output_tokens', 'total_tokens', 'rate_limit_rejections_without_usage', 'estimated_usage_calls',
              'estimated_input_tokens', 'estimated_output_tokens', 'estimated_total_tokens')
    result = {key: sum(g.get(key, 0) for g in groups) for key in fields}
    result['accounting_complete'] = result['unknown_usage_calls'] == result['pending_calls'] == 0
    result['accounting_usable_with_estimates'] = (result['pending_calls'] == 0 and
        result['unknown_usage_calls'] == result['rate_limit_rejections_without_usage'] + result['estimated_usage_calls'])
    result['budget_tokens_including_estimates'] = result['total_tokens'] + result['estimated_total_tokens']
    return result


class Run(TauBoundary):
    def __init__(self, root):
        sys.path.insert(0, str(AHE_NATIVE))
        from ahe_native.api_broker import RequestGate
        from ahe_native.capabilities import restricted_policy
        from ahe_native.cli import config_for
        from ahe_native.connection import bind_connections
        from ahe_native.debugger_isolation import bind_isolated_debugger
        from ahe_native.evolve_isolation import bind_isolated_evolve
        from ahe_native.isolation import prepare_base, ensure_shell_utilities
        from ahe_native.metered_broker import Broker
        from ahe_native.prepare import prepare
        from ahe_native.upstream import AHE, bind_native_transport, load_upstream
        self.root, self.protocol = root, read(root / 'protocol.json')
        receipt_path = root / 'recovery/continuation_02.json'
        if not receipt_path.exists():
            receipt_path = root / 'recovery/receipt.json'
        self.recovery = read(receipt_path) if receipt_path.exists() else None
        if self.recovery:
            if read(root / 'status.json')['phase'] != 'stopped_on_error':
                raise RuntimeError('audited continuation requires a fully stopped ancestor')
            if digest(root / 'frozen_runtime_sources.json') != self.recovery['original_frozen_receipt_sha256']:
                raise RuntimeError('original source receipt changed')
            for path, sha in self.recovery['starting_artifact_hashes'].items():
                if digest(path) != sha:
                    raise RuntimeError('audited recovery artifact changed')
        else:
            if read(root / 'status.json')['phase'] != 'prepared':
                raise RuntimeError('existing run requires an explicit artifact audit; no blind replay')
            frozen = {str(ROOT / name): sha for name, sha in profile.source_hashes().items()}
            for folder in (Path(__file__).parent, AHE_NATIVE / 'ahe_native', ROOT / 'evolution/tau_comparison'):
                frozen.update({str(p): digest(p) for p in folder.glob('*.py')})
            save(root / 'frozen_runtime_sources.json', frozen)
        self.verify()
        save(root / 'status.json', {'phase': 'initializing', 'pid': os.getpid(), 'optimizer_calls': 0})
        if os.geteuid() != 0:
            raise RuntimeError('run in the dedicated experiment container for chroot isolation')
        self.ready, self.abort, self.lock = threading.Event(), threading.Event(), threading.Lock()
        self.first_round, self.continuation = {}, {}
        if self.recovery and self.recovery.get('completed_first_rounds'):
            self.first_round = {'meta:os':True, 'life:os':True, 'ahe:os':True}
            self.continuation = {'ahe_resume':{'os':{'start_iteration':2,'skip_eval':True}}}
            self.ready.set()
        self.names = {'os': root.name + '_ahe_os'}
        self.spec = {'batches': self.protocol['batches'], 'train_trials': 1,
                     'test_task_ids': self.protocol['native_spec']['test_ids']}
        self.protocol['domains_spec'] = {'os': self.spec}
        baseline = AHE_NATIVE / 'baselines/os'
        if not baseline.exists():
            prepare('os', baseline)
        base = prepare_base()
        (base / 'episodes').mkdir(mode=0o711, exist_ok=True)
        ensure_shell_utilities()
        if not self.recovery:
            prepare_workspace(root, baseline / 'public_tool_schema.json')
        self.work = root / 'life'
        self.compose = composition_module()
        self.state = {'domain': 'os', 'round': 0, 'rounds': [], 'active_snapshot': [],
                      'active_snapshot_sha256': hooks.object_digest([]), 'accepted_bundle_ids': [],
                      'operation_log': [], 'prior_test_exposure': False}
        if self.recovery:
            self.state = read(self.work / 'state.json')
        else:
            save(self.work / 'state.json', self.state)
        result = subprocess.run(['bash', '-c',
            'source "$1"; printf "%s" "${DASHSCOPE_API_KEY:-${API_KEY:-}}"', 'bash', str(ROOT / 'env.sh')],
            capture_output=True, check=True, timeout=30)
        key = result.stdout.decode().strip()
        if not key:
            raise RuntimeError('gateway key unavailable in local env.sh')
        self.gate = RequestGate(8, min_interval=1.25)
        self.mr = Broker(key, self.protocol['optimizer']['base_url'], request_gate=self.gate,
            upstream_model='deepseek-v4.1-flash', upstream_timeout=1800, rate_limit_retries=3,
            ledger_path=root / 'ledgers/optimizer_requests.jsonl',
            ledger_summary_path=root / 'ledgers/OPTIMIZER_PROVIDER_USAGE.json')
        Ledger, TokenBroker = transport_modules()
        self.meta_ledger = Ledger(root / 'meta')
        self.meta_broker = TokenBroker(key, self.protocol['optimizer']['base_url'], self.gate,
                                      self.meta_ledger, ('os',))
        del key, result
        self.solver = SolverProxy()
        os.environ.update(AHE_EVOLVE_API_KEY='EMPTY', AHE_EVOLVE_BASE_URL=self.mr.url,
            AHE_LOCAL_BROKER_URL=self.mr.url, QA_API_KEY='EMPTY', QA_BASE_URL=self.mr.url,
            QA_MODEL_NAME='deepseek-v4-flash', AHE_HOME=str(AHE),
            GIT_AUTHOR_NAME='OS native comparison', GIT_COMMITTER_NAME='OS native comparison',
            GIT_AUTHOR_EMAIL='os-native@localhost', GIT_COMMITTER_EMAIL='os-native@localhost')
        os.environ['PATH'] = str(AHE_NATIVE / '.venv/bin') + ':' + os.environ.get('PATH', '')
        self.ahe_module, self.ahe_receipt = load_upstream()
        for binder in (bind_connections, bind_native_transport, bind_isolated_evolve, bind_isolated_debugger):
            binder(self.ahe_module)
        self.ahe_module.EXPERIMENTS_DIR = root / 'ahe/experiments'
        save(root / 'ahe_source_receipt.json', self.ahe_receipt)
        def guarded(function, position):
            def call(*args, **kwargs):
                number = kwargs.get('iteration', args[position] if len(args) > position else 1)
                if number >= 2:
                    self.wait_first()
                return function(*args, **kwargs)
            return call
        self.ahe_module.run_parallel_adb_ask = guarded(self.ahe_module.run_parallel_adb_ask, 4)
        self.ahe_module.run_evolve_agent = guarded(self.ahe_module.run_evolve_agent, 2)
        module = upstream('os')
        self.meta_ports = {'os': MetaPort(root, 'os', self.spec, module, self.meta_broker.url,
                                         self.meta_ledger, None, self.solver.url)}
        config = config_for('os', self.spec['batches'][0]['task_ids'], 6, self.mr.url,
            debugger_tasks=10, concurrency=50, debugger_concurrency=8, target_pass_rate=1.01,
            capability_policy=restricted_policy('os'))
        config['llm']['base_url'] = self.solver.url
        config['native_backend'].update(training_schedule_file=str(root / 'training_schedule.json'),
            training_schedule_sha256=digest(root / 'training_schedule.json'))
        if self.recovery and 'ahe_baseline_reuse' in self.recovery:
            config['native_backend']['reuse_completed_episodes'] = self.recovery['ahe_baseline_reuse']
        self.ahe_configs = {'os': config}
        save(root / 'ahe/configs/os.json', config)

    def verify(self):
        protocol = read(self.root / 'protocol.json')
        verify_sources(protocol['native_source_hashes'])
        if digest(self.root / 'training_schedule.json') != protocol['training_schedule_sha256']:
            raise RuntimeError('training schedule changed')
        frozen = self.root / 'frozen_runtime_sources.json'
        if frozen.exists():
            if getattr(self, 'recovery', None):
                verify_sources(self.recovery['effective_runtime_sources'])
                if digest(frozen) != self.recovery['original_frozen_receipt_sha256']:
                    raise RuntimeError('original frozen sources changed')
            else:
                verify_sources(read(frozen))
        if hasattr(self, 'state'):
            self.compose.verify_active(self.work, self.state)

    def wait_first(self):
        self.ready.wait()
        if self.abort.is_set():
            raise RuntimeError('first-round validation failed; stop later optimization')

    def inspect_first_ahe(self, domain):
        if self.recovery and self.recovery.get('completed_first_rounds'):
            return  # Exact completed paid round and next-batch measurement were audited.
        return super().inspect_first_ahe(domain)

    def ahe_usage(self, domain='os', round=None):
        return select_optimizer_groups(self.mr.ledger.summary(), self.names[domain], domain, round)

    def evaluate_life(self, split, phase, manifest, ids):
        self.verify()
        cached = self.work / 'evals' / phase / 'summary.json'
        if self.recovery and cached.exists():
            config = read(cached.parent / 'config.json')
            if config['candidate_sha256'] != digest(manifest) or config['ids'] != ids:
                raise RuntimeError('cached Life phase fingerprint changed')
            return read(cached), cached.parent
        log = self.work / 'logs' / (phase + '.log')
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('w') as output:
            subprocess.run([str(ROOT / 'AgentBench/.native/core/bin/python'), '-u', '-m',
                'evolution.os_comparison.evaluate', '--root', str(self.root), '--method', 'life',
                '--split', split, '--phase', phase, '--ids', json.dumps(ids), '--manifest', str(manifest),
                '--solver-base', self.solver.url], cwd=ROOT, env=child_env(PYTHONPATH=str(ROOT)),
                stdout=output, stderr=subprocess.STDOUT, timeout=21600, check=True)
        out = self.work / 'evals' / phase
        return read(out / 'summary.json'), out

    def completion(self, number):
        from ahe_native.token_accounting import scoped_url
        endpoint = scoped_url(self.mr.url, experiment=self.root.name, domain='os',
                              round=number, role='life_optimizer')
        def once(messages, tools):
            request = urllib.request.Request(endpoint + '/chat/completions', method='POST',
                data=json.dumps({'model': 'deepseek-v4-flash', 'messages': messages, 'tools': tools,
                    'tool_choice': 'auto', 'max_tokens': OUTPUT_LIMIT, 'stream': False}).encode(),
                headers={'Authorization': 'Bearer EMPTY', 'Content-Type': 'application/json'})
            try:
                with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=2100) as response:
                    value = json.load(response)
            except urllib.error.HTTPError as exc:
                body = exc.read().decode(errors='replace')
                if exc.code == 400 and any(t in body.lower() for t in ('context', 'maximum sequence', 'too long')):
                    raise ContextLimit('provider context exhausted') from exc
                raise RuntimeError('optimizer HTTP failure ' + str(exc.code)) from exc
            verify_usage(value)
            if not value.get('choices') or value.get('error'):
                raise RuntimeError('optimizer response missing assistant choice')
            return value
        def call(messages, tools):
            try:
                return once(messages, tools)
            except (RuntimeError, ConnectionError, urllib.error.URLError):
                # Retry only a finished, undelivered ReadTimeout, once. Other failures stay closed.
                from .recovery import estimate_timeout
                matching = [r for r in self.mr.ledger.rows if r['event'] == 'started'
                    and r['experiment'] == self.root.name and r['round'] == number
                    and r['role'] == 'life_optimizer']
                last = matching[-1]['request_id'] if matching else None
                end = next((r for r in reversed(self.mr.ledger.rows)
                    if r['event'] == 'finished' and r['request_id'] == last), {})
                if end.get('error_type') != 'ReadTimeout' or end.get('http_status') is not None or end.get('transport') is not None:
                    raise
                estimate_timeout(self.mr.ledger, self.root.name, number)
                return once(messages, tools)
        return call

    def run_life(self):
        for number in range(self.state['round'] + 1, 7):
            if number > 1:
                self.wait_first()
            ids = self.spec['batches'][number - 1]['task_ids']
            manifest = self.compose.write_active_chain(self.work, self.state, f'incumbent_{number:02d}')
            incumbent, evidence = self.evaluate_life('train', f'incumbent_{number:02d}', manifest, ids)
            if incumbent['infra_errors'] > 2:
                raise RuntimeError('Life infrastructure failures; stop before optimization')
            workspace = self.root / 'optimizer_inputs' / f'round_{number:02d}'
            output = self.work / 'proposals' / f'round_{number:02d}'
            if not (self.recovery and number == 1):
                profile.prepare(self.work, evidence, workspace, seed=number)
            else:
                profile.verify_inputs(workspace)
            bundle_path = profile.propose(workspace, output, completion=self.completion(number),
                checkpoint_dir=self.work / 'optimizer_sessions' / f'round_{number:02d}')
            validation = read(output / 'validation.json')
            usage = life_usage(self.mr.ledger.summary(), self.root.name, number)
            if (not usage['accounting_usable_with_estimates'] or not usage['total_tokens'] or
                    usage['total_tokens'] != validation['usage'].get('total_tokens', 0)):
                raise RuntimeError('Life optimizer ledger/checkpoint mismatch')
            bundle, _, staged = self.compose.stage_bundle(self.work, number, bundle_path, self.state)
            candidate, _ = self.evaluate_life('train', f'candidate_{number:02d}', staged, ids)
            accepted = profile.strict_improvement(candidate, incumbent, expected_task_ids=[str(i) for i in ids])
            parent = self.state['active_snapshot_sha256']
            operations = [{k: v for k, v in op.items() if k != 'artifact_path'} for op in bundle['operations']]
            if accepted:
                active = self.compose.commit_bundle(self.work, bundle, self.state)
                self.state.update(active_snapshot=active, active_snapshot_sha256=hooks.object_digest(active))
                self.state['accepted_bundle_ids'].append(bundle['bundle_id'])
                self.state['operation_log'].append({'bundle_id': bundle['bundle_id'], 'operations': operations,
                    'parent_snapshot_sha256': parent, 'committed_snapshot_sha256': hooks.object_digest(active)})
            row = {'round': number, 'train_batch': number - 1, 'incumbent_score': incumbent['accuracy'],
                'candidate_score': candidate['accuracy'], 'retained_score': (candidate if accepted else incumbent)['accuracy'],
                'incumbent_successes': incumbent['successes'], 'candidate_successes': candidate['successes'],
                'incumbent_infra_errors': incumbent['infra_errors'], 'candidate_infra_errors': candidate['infra_errors'],
                'accepted': accepted, 'bundle_id': bundle['bundle_id'], 'operations': operations,
                'active_snapshot_sha256': self.state['active_snapshot_sha256'], 'optimizer_usage': usage,
                'proposer_requests': validation['proposer_requests'], 'repairs_granted': validation['repairs_granted']}
            self.state['rounds'].append(row)
            self.state['round'] = number
            save(self.work / 'state.json', self.state)
            if number == 1:
                with self.lock:
                    self.first_round['life:os'] = candidate['infra_errors'] <= 2 and usage['accounting_usable_with_estimates']

    def smoke(self):
        save(self.root / 'status.json', {'phase': 'native_smoke', 'pid': os.getpid(), 'optimizer_calls': 0})
        subprocess.run([str(ROOT / 'AgentBench/.native/core/bin/python'), '-m',
            'evolution.os_comparison.native', '--smoke', '--out', str(self.root / 'docker_smoke.json')],
            cwd=ROOT, env=child_env(PYTHONPATH=str(ROOT)), timeout=600, check=True)
        empty = self.compose.write_active_chain(self.work, self.state, 'setup_empty')
        port = self.meta_ports['os']
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(port.phase, 'train', 0, 'setup_smoke', port.work / 'agents/baseline_zero.py', 1, True),
                pool.submit(self.evaluate_life, 'train', 'setup_smoke', empty, self.spec['batches'][0]['task_ids'][:1]),
                pool.submit(self.ahe_native, 'os', 'train', self.spec['batches'][0]['task_ids'][:1],
                            AHE_NATIVE / 'baselines/os', self.root / 'smoke/ahe/os', round=1)]
            results = [f.result() for f in futures]
        scores = [results[0], results[1][0], results[2]]
        if any(score['infra_errors'] for score in scores):
            raise RuntimeError('native solver smoke failed; no optimizer calls issued')
        save(self.root / 'setup_smoke.json', {'passed': True, 'optimizer_calls': 0,
             'methods': dict(zip(('meta', 'life', 'ahe'), scores))})
        deadline = time.monotonic() + 900
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        while True:
            with opener.open('http://127.0.0.1:5020/api/list_workers', timeout=15) as response:
                workers = json.load(response)
            expected = ('os-env_train', 'os-std')
            live = {name: sum(worker.get('status') in (0, 'ALIVE') for worker in
                             workers.get(name, {}).get('workers', {}).values()) for name in expected}
            if all(live.values()):
                save(self.root / 'worker_registration.json', {'passed': True, 'live_workers': live})
                break
            save(self.root / 'status.json', {'phase': 'waiting_for_os_workers', 'pid': os.getpid(),
                 'optimizer_calls': 0, 'live_workers': live})
            if time.monotonic() > deadline:
                raise RuntimeError('OS TRAIN/TEST workers are not both registered; no optimization')
            time.sleep(5)

    def report(self):
        from ahe_native.batch import progress
        ahe = progress(self.ahe_experiment('os'))
        meta = read(self.root / 'meta/os/state.json')
        life = read(self.work / 'state.json')
        counts = {'meta': len(meta['rounds']), 'life': life['round'],
                  'ahe': sum((self.ahe_experiment('os') / f'runs/iteration_{i:03d}/evolve/evolve_summary.md').is_file()
                             for i in range(1, 7))}
        save(self.root / 'status.json', {'phase': 'training', 'pid': os.getpid(), 'utc_unix': time.time(),
             'completed_rounds': counts, 'first_round_gate': dict(self.first_round), 'gate': self.gate.snapshot()})
        usage = {'meta': self.meta_ledger.summarize('os'), 'ahe': self.ahe_usage(),
                 'life': life_usage(self.mr.ledger.summary(), self.root.name)}
        save(self.root / 'OPTIMIZER_TOKEN_USAGE.json', usage)
        save(self.root / 'TRAINING_RESULTS.json', {'meta': meta['rounds'], 'life': life['rounds'], 'ahe': ahe,
             'ahe_round_usage': {str(i): self.ahe_usage(round=i) for i in range(1, 7)},
             'ahe_final_train': read(self.root / 'ahe/os/final_train.json')
                 if (self.root / 'ahe/os/final_train.json').exists() else None})
        return counts

    def training(self):
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {'meta': pool.submit(self.run_meta, 'os'), 'ahe': pool.submit(self.run_ahe, 'os'),
                       'life': pool.submit(self.run_life)}
            while not all(f.done() for f in futures.values()):
                self.inspect_first_ahe('os')
                errors = {m: str(f.exception()) for m, f in futures.items() if f.done() and f.exception()}
                if errors:
                    save(self.root / 'errors.json', errors)
                    self.abort.set()
                    self.ready.set()
                if len(self.first_round) == 3 and not self.ready.is_set():
                    passed = all(self.first_round.values())
                    save(self.root / 'first_round_gate.json', {'passed': passed, 'methods': self.first_round})
                    if not passed:
                        self.abort.set()
                    self.ready.set()
                self.report()
                time.sleep(15)
            for future in futures.values():
                future.result()
        if self.report() != {'meta': 6, 'ahe': 6, 'life': 6}:
            raise RuntimeError('all eighteen rounds must complete')
        self.verify()
        from ahe_native.upstream import source_fingerprint
        from ahe_native.backend import workspace_hash
        if source_fingerprint() != self.ahe_receipt:
            raise RuntimeError('official AHE sources changed')
        usage = read(self.root / 'OPTIMIZER_TOKEN_USAGE.json')
        if any(not value.get('accounting_usable_with_estimates', value['accounting_complete']) for value in usage.values()):
            raise RuntimeError('optimizer token accounting incomplete')
        best = read(self.ahe_experiment('os') / 'best_ever.json')
        self.ahe_selected = self.ahe_experiment('os') / f'runs/iteration_{best["iteration"]:03d}/input/workspace'
        self.life_final = self.compose.write_active_chain(self.work, self.state, 'final')
        port = self.meta_ports['os']
        self.meta_final = port.work / 'agents' / port.state['incumbent']
        state_paths = [self.work / 'state.json', self.root / 'meta/os/state.json',
                       self.ahe_experiment('os') / 'best_ever.json']
        save(self.root / 'training_seal.json', {'completed_rounds': {'meta': 6, 'ahe': 6, 'life': 6},
            'candidates': {'meta': digest(self.meta_final), 'life': digest(self.life_final),
                           'ahe': workspace_hash(self.ahe_selected)},
            'state_hashes': {str(p): digest(p) for p in state_paths}, 'optimizer_usage': usage,
            'ahe_selection': 'official best_ever TRAIN snapshot', 'ahe_best': best})

    def tests(self):
        from ahe_native.backend import workspace_hash
        seal = read(self.root / 'training_seal.json')
        self.meta_broker.close()
        self.meta_broker = None
        self.mr.blocked_roles.update({'debugger', 'evolve', 'life_optimizer'})
        if workspace_hash(self.ahe_selected) != seal['candidates']['ahe']:
            raise RuntimeError('AHE TRAIN-selected snapshot changed')
        ids = self.spec['test_task_ids']
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {'meta': pool.submit(self.meta_ports['os'].phase, 'test', 0, 'final_test', self.meta_final, 6),
                'life': pool.submit(self.evaluate_life, 'test', 'final_test', self.life_final, ids),
                'ahe': pool.submit(self.ahe_native, 'os', 'test', ids, self.ahe_selected,
                                   self.root / 'ahe/final_test/os', round=6)}
            scores = {m: f.result() for m, f in futures.items()}
        scores['life'] = scores['life'][0]
        save(self.root / 'FINAL_RESULTS.json', {'split': 'test', 'methods': scores,
             'selection': 'sealed TRAIN-retained Meta/Life and official AHE best-ever; all TEST144 outcomes retained'})
        save(self.root / 'status.json', {'phase': 'complete', 'completed_rounds': {'meta': 6, 'ahe': 6, 'life': 6},
             'test_accuracy': {m: s['accuracy'] for m, s in scores.items()}})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    runner = None
    with (root / '.runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            runner = Run(root)
            if not runner.recovery:
                runner.smoke()
            elif not read(root / 'setup_smoke.json')['passed'] or not read(root / 'worker_registration.json')['passed']:
                raise RuntimeError('audited native setup receipts missing')
            if args.smoke_only:
                save(root / 'status.json', {'phase': 'smoke_complete', 'optimizer_calls': 0})
                return
            runner.training()
            runner.tests()
        except Exception as exc:
            save(root / 'failure.json', {'type': type(exc).__name__, 'detail': str(exc)[:6000]})
            save(root / 'status.json', {'phase': 'stopped_on_error', 'error_type': type(exc).__name__})
            raise
        finally:
            if runner is not None:
                runner.close()


if __name__ == '__main__':
    main()
