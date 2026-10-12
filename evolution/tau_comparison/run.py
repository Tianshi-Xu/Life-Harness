"""Six native Tau evolutions for Meta/AHE, then sealed final tests, never Life."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import threading
import time

from .common import (AHE_NATIVE, ALI_URL, DOMAINS, HISTORICAL, ROOT, TAU, aliyun_key,
    child_env, digest, read, save, source_fingerprint, summarize_rows, transport_modules, user_model)
from .meta_port import MetaPort, prepare_workspace, upstream
from .solver_proxy import SolverProxy


def build_protocol(name):
    historical = read(HISTORICAL / 'manifest.json')
    for domain in DOMAINS:
        native = read(TAU / 'data/tau2/domains' / domain / 'split_tasks.json')
        spec = historical['domains'][domain]
        train, test = set(native['train']), set(native['test'])
        if train & test or set(spec['test_task_ids']) != test or any(
                not set(row['task_ids']) <= train for row in spec['batches']):
            raise RuntimeError('historical/native Tau TRAIN/TEST identities diverged')
    return {'name': name, 'methods': ['meta', 'ahe'], 'domains': list(DOMAINS), 'rounds': 6,
        'historical_manifest_sha256': digest(HISTORICAL / 'manifest.json'),
        'domains_spec': {d: historical['domains'][d] for d in DOMAINS},
        'solver': {'model': 'qwen3-4b', 'total_context': 32768, 'max_output': 4096,
                   'temperature': 0, 'workers_per_domain': 50, 'budget_gate': 'shared local proxy'},
        'user_routes': {d: {'model': user_model(d), 'provider': 'MR' if d == 'telecom' else 'Aliyun',
                           'thinking': d != 'telecom'} for d in DOMAINS},
        'user_max_output': {d: 100000 if d == 'telecom' else 4096 for d in DOMAINS},
        'thinking_metadata_note': 'user_routes.thinking denotes explicitly enabling Aliyun thinking. '
            'MR simulator reasoning is left at the provider default, not disabled.',
        'optimizer': 'deepseek-v4-flash', 'global_mr_concurrency': 8, 'global_mr_min_interval': 1.25,
        'budget': 'six optimization steps; actual measured API tokens, no cumulative token ceiling',
        'optimizer_accounting': 'Meta parent/analysis/implementation; AHE Debugger/Evolve including compaction/retries. '
            'Qwen solver, user simulator and NL judge excluded; separate raw ledgers retained.',
        'ahe': {'profile': 'h2345-budget-restricted-v1', 'debugger_tasks': 10, 'debugger_retries': 1,
                'loop': 'original run_single_experiment', 'selection': 'original best-ever',
                'train_curve': 'candidate after step N is measured by evaluation N+1 on the next frozen batch; '
                    'step 6 receives a separate evaluation on batch 5, not fed back into selection'},
        'meta': {'loop': 'original run_evolve and official skill', 'agents': ['analysis', 'implementation'],
                 'selection': 'strict same-complete-batch improvement; batch changes after non-improvement'},
        'comparison_caveat': 'Not identical selection or successive train batches; label each batch/denominator. '
            'No test-selected best model; final retained snapshots are tested once.',
        'initialization': 'identity Meta / blank native AHE; no released Life strategies or prior optimized code',
        'test_policy': 'all six evolutions for both methods in all domains must be complete and sealed first',
        'first_round_gate': 'all methods/domains must produce a measured qualified candidate with complete '
                            'role-separated optimizer usage before steps 2-6',
        'life_agentbench_started': False}


def select_optimizer_groups(summary, experiment, domain=None, round=None, correction=None):
    values = list(summary['groups'].values())
    if correction:
        raw = summary['groups'].get(correction['group_key'])
        if raw != correction['raw_group']:
            raise RuntimeError('audited unscoped usage changed; refuse ambiguous attribution')
        values = [{**v, **correction['scope']} if v == raw else v for v in values]
    groups = [v for v in values if v['experiment'] == experiment and
        v['role'] in {'debugger', 'evolve'} and (domain is None or v['domain'] == domain) and
        (round is None or v['round'] == round)]
    keys = ('attempts', 'known_usage_calls', 'unknown_usage_calls', 'pending_calls',
            'input_tokens', 'output_tokens', 'total_tokens', 'rate_limit_rejections_without_usage',
            'estimated_usage_calls', 'estimated_input_tokens', 'estimated_output_tokens', 'estimated_total_tokens')
    result = {k: sum(g.get(k, 0) for g in groups) for k in keys}
    result['accounting_complete'] = result['unknown_usage_calls'] == result['pending_calls'] == 0
    result['inference_accounting_complete'] = (result['pending_calls'] == 0 and
        result['unknown_usage_calls'] == result['rate_limit_rejections_without_usage'])
    result['accounting_usable_with_estimates'] = (result['pending_calls'] == 0 and
        result['unknown_usage_calls'] == result['rate_limit_rejections_without_usage'] + result['estimated_usage_calls'])
    result['budget_tokens_including_estimates'] = result['total_tokens'] + result['estimated_total_tokens']
    result['by_role'] = {role: {k: sum(g.get(k, 0) for g in groups if g['role'] == role) for k in keys}
                         for role in ('debugger', 'evolve')}
    return result


class Comparison:
    def __init__(self, root, protocol, *, smoke_only=False, continuation=None):
        sys.path.insert(0, str(AHE_NATIVE))
        from ahe_native.api_broker import RequestGate
        from ahe_native.backend import MANIFEST_SHA256
        from ahe_native.connection import credentials
        from ahe_native.isolation import BASE, ensure_shell_utilities
        from ahe_native.metered_broker import Broker
        from ahe_native.upstream import AHE, bind_native_transport, load_upstream
        from ahe_native.connection import bind_connections
        from ahe_native.debugger_isolation import bind_isolated_debugger
        from ahe_native.evolve_isolation import bind_isolated_evolve
        from ahe_native.tau_training import make_tau_schedule
        if protocol['historical_manifest_sha256'] != MANIFEST_SHA256:
            raise RuntimeError('native manifest differs from audited historical split')
        if not (BASE / '.ready').is_file():
            raise RuntimeError('AHE isolated runtime is not ready')
        ensure_shell_utilities()
        self.root, self.protocol, self.smoke_only = root, protocol, smoke_only
        self.continuation = continuation or {}
        self.names = {d: self.continuation.get('ahe_resume', {}).get(d, {}).get(
            'name', root.name + '_ahe_' + d) for d in DOMAINS}
        self.ready = threading.Event()
        self.abort = threading.Event()
        self.first_round = {}
        self.lock = threading.Lock()
        self.gate = RequestGate(8, min_interval=1.25)
        key, router = credentials()
        ali = aliyun_key()
        # Real credentials are held only in these host-side connections.
        self.mr = Broker(key, router, request_gate=self.gate, rate_limit_retries=3,
                         ledger_path=root / 'ledgers/mr_requests.jsonl',
                         ledger_summary_path=root / 'ledgers/MR_TOKEN_USAGE.json')
        self.ali = Broker(ali, ALI_URL, allowed_model='qwen3.8-flash',
                          ledger_path=root / 'ledgers/aliyun_requests.jsonl',
                          ledger_summary_path=root / 'ledgers/ALIYUN_TOKEN_USAGE.json')
        Ledger, TokenBroker = transport_modules()
        self.meta_ledger = Ledger(root / 'meta')
        self.meta_broker = TokenBroker(key, router, self.gate, self.meta_ledger, DOMAINS)
        self.solver = SolverProxy()
        for domain in DOMAINS:
            os.environ['AHE_TAU_USER_BASE_' + domain.upper()] = self.route(domain)
            os.environ['AHE_TAU_USER_MODEL_' + domain.upper()] = user_model(domain)
        os.environ.update(AHE_EVOLVE_API_KEY='EMPTY', AHE_EVOLVE_BASE_URL=self.mr.url,
            AHE_LOCAL_BROKER_URL=self.mr.url, AHE_USER_API_KEY='EMPTY', QA_API_KEY='EMPTY',
            QA_BASE_URL=self.mr.url, QA_MODEL_NAME='deepseek-v4-flash', AHE_HOME=str(AHE),
            GIT_AUTHOR_NAME='Tau native comparison', GIT_COMMITTER_NAME='Tau native comparison',
            GIT_AUTHOR_EMAIL='tau-native@localhost', GIT_COMMITTER_EMAIL='tau-native@localhost')
        os.environ['PATH'] = str(AHE_NATIVE / '.venv/bin') + ':' + os.environ.get('PATH', '')
        self.ahe_module, self.ahe_receipt = load_upstream()
        bind_connections(self.ahe_module)
        bind_native_transport(self.ahe_module)
        bind_isolated_evolve(self.ahe_module)
        bind_isolated_debugger(self.ahe_module)
        self.ahe_module.EXPERIMENTS_DIR = root / 'ahe/experiments'
        def guarded(function, position):
            def call(*args, **kwargs):
                iteration = kwargs.get('iteration', args[position] if len(args) > position else 1)
                if iteration >= 2:
                    # AHE evaluates candidate 1 at iteration 2 before Debugger/Evolve 2.
                    # Gate only the paid optimization boundaries, not that measurement.
                    self.ready.wait()
                    if self.abort.is_set():
                        raise RuntimeError('first-round integration failed; no later optimization')
                return function(*args, **kwargs)
            return call
        self.ahe_module.run_parallel_adb_ask = guarded(self.ahe_module.run_parallel_adb_ask, 4)
        self.ahe_module.run_evolve_agent = guarded(self.ahe_module.run_evolve_agent, 2)
        self.meta_modules = {d: upstream(d) for d in DOMAINS}
        historical = read(HISTORICAL / 'manifest.json')
        historical['sha256'] = MANIFEST_SHA256
        schedule = make_tau_schedule(historical)
        save(root / 'training_schedule.json', schedule)
        save(root / 'ahe_source_receipt.json', self.ahe_receipt)
        self.meta_ports = {d: MetaPort(root, d, protocol['domains_spec'][d], self.meta_modules[d],
            self.meta_broker.url, self.meta_ledger, self.route(d), self.solver.url) for d in DOMAINS}
        self.ahe_configs = {d: self.ahe_config(d) for d in DOMAINS}
        # Metadata-only safety assertion; never print secret values or matching text.
        for secret in (key.encode(), ali.encode()):
            if any(secret in p.read_bytes() for p in root.rglob('*') if p.is_file()):
                self.close()
                raise RuntimeError('credential appeared in an experiment artifact; refuse startup')

    def route(self, domain):
        return self.mr.url if domain == 'telecom' else self.ali.url

    def ahe_config(self, domain):
        from ahe_native.cli import config_for
        from ahe_native.capabilities import restricted_policy
        config = config_for(domain, self.protocol['domains_spec'][domain]['batches'][0]['task_ids'],
            6, self.mr.url, debugger_tasks=10, concurrency=50, debugger_concurrency=8,
            target_pass_rate=1.01, capability_policy=restricted_policy(domain))
        config['llm']['base_url'] = self.solver.url
        config['harbor']['k'] = self.protocol['domains_spec'][domain]['train_trials']
        config['native_backend'].update(training_schedule_file=str(self.root / 'training_schedule.json'),
            training_schedule_sha256=digest(self.root / 'training_schedule.json'))
        reuse = self.continuation.get('ahe_resume', {}).get(domain, {}).get('reuse_completed_episodes')
        if reuse:
            config['native_backend']['reuse_completed_episodes'] = reuse
        completed_evolve = self.continuation.get('ahe_resume', {}).get(domain, {}).get('reuse_completed_evolve')
        if completed_evolve:
            config['native_backend']['reuse_completed_evolve'] = completed_evolve
        save(self.root / 'ahe/configs' / (domain + '.json'), config)
        return config

    def ahe_experiment(self, domain):
        return self.root / 'ahe/experiments' / self.names[domain]

    def ahe_usage(self, domain, round=None):
        return select_optimizer_groups(self.mr.ledger.summary(), self.names[domain], domain, round,
            self.continuation.get('optimizer_scope_correction'))

    def ahe_native(self, domain, split, ids, workspace, out, trials=1, round=6):
        from ahe_native.backend import episode_timeout, interpreter, native_env
        from ahe_native.capabilities import restricted_policy
        from ahe_native.token_accounting import scoped_url
        out.mkdir(parents=True, exist_ok=True)
        def episode(task_id, trial):
            import hashlib
            folder = out / hashlib.sha256(f'{task_id}:{trial}'.encode()).hexdigest()[:16]
            folder.mkdir(exist_ok=True)
            request = {'domain': domain, 'split': split, 'task_id': str(task_id), 'trial': trial,
                'workspace': str(workspace), 'agent_config': 'code_agent.yaml',
                'trace_dir': str(folder / 'agent'), 'capability_policy': restricted_policy(domain)}
            save(folder / 'request.json', request)
            env = native_env(domain, scope={'experiment': self.names[domain],
                                          'domain': domain, 'round': round})
            env.update(LLM_API_KEY='EMPTY', LLM_BASE_URL=self.solver.url, LLM_MODEL='qwen3-4b')
            # No host credential is needed by a native worker.
            for key in list(env):
                if any(word in key.upper() for word in ('KEY', 'TOKEN', 'SECRET', 'PASSWORD', 'CREDENTIAL', 'AUTH')):
                    env.pop(key)
            env.update(LLM_API_KEY='EMPTY', AHE_USER_API_KEY='EMPTY', OPENAI_API_KEY='EMPTY')
            with (folder / 'environment.log').open('a') as log:
                process = subprocess.run([str(interpreter(domain)), '-m', 'ahe_native.native_worker',
                    '--request', str(folder / 'request.json'), '--out', str(folder / 'record.json')],
                    cwd=AHE_NATIVE, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=episode_timeout(domain))
            if process.returncode or not (folder / 'record.json').is_file():
                raise RuntimeError('native AHE episode process failed; inspect ' + str(folder))
            return read(folder / 'record.json')
        with ThreadPoolExecutor(max_workers=50) as pool:
            futures = [pool.submit(episode, task, trial) for task in ids for trial in range(trials)]
            rows = [future.result() for future in futures]
        score = summarize_rows(ids, trials, [{**r, 'reward': r['success']} for r in rows])
        score.update(domain=domain, split=split, round=round)
        save(out / 'summary.json', score)
        return score

    def run_meta(self, domain):
        port = self.meta_ports[domain]
        row = port.state['rounds'][0] if port.state['rounds'] else port.round()
        with self.lock:
            self.first_round['meta:' + domain] = bool(row['candidate_score'] is not None and
                row['candidate_infra_errors'] <= 2 and row['usage']['accounting_complete'] and
                set(row['usage']['by_role']) == {'parent', 'analysis', 'implementation'})
        self.ready.wait()
        if self.abort.is_set():
            raise RuntimeError('first-round integration gate failed')
        candidates = self.continuation.get('meta_date_candidates', {}).get(domain)
        if candidates:
            port.repair_date_rejections({int(i): sha for i, sha in candidates.items()})
        while len(port.state['rounds']) < 6:
            port.round()

    def run_ahe(self, domain):
        config = self.ahe_configs[domain]
        resume = self.continuation.get('ahe_resume', {}).get(domain, {})
        if resume.get('repair_baseline'):
            self.repair_ahe_baseline(domain, config, resume['repair_baseline'])
        self.ahe_module.run_single_experiment(config, str(self.root / 'ahe/configs' / (domain + '.json')),
            experiment_name=self.names[domain], start_iteration=resume.get('start_iteration', 1),
            skip_eval=resume.get('skip_eval', False))
        from ahe_native.batch import progress
        from ahe_native.capabilities import validate_workspace, restricted_policy
        experiment = self.ahe_experiment(domain)
        check = progress(experiment)
        if len(check['evaluations']) != 6 or any(not (experiment / f'runs/iteration_{i:03d}/evolve/evolve_summary.md').is_file()
                                               for i in range(1, 7)):
            raise RuntimeError('original AHE did not complete six evolutions')
        validate_workspace(experiment / 'workspace', restricted_policy(domain))
        if not self.ahe_usage(domain)['accounting_usable_with_estimates']:
            raise RuntimeError('AHE optimizer token accounting incomplete')
        if resume.get('final_train_reused'):
            from .recovery import audit_final_train
            audit_final_train(self.root, domain, experiment, self.protocol['domains_spec'][domain])
            return
        config_final = dict(config)
        final_dir = self.root / 'ahe/final_train' / domain / 'iteration_006'
        final_dir.mkdir(parents=True, exist_ok=True)
        job = self.ahe_module.run_harbor(config_final, experiment / 'workspace', 'code_agent.yaml', final_dir)
        rows = [read(p) for p in job.glob('native_*/native_record.json')]
        ids = self.protocol['domains_spec'][domain]['batches'][5]['task_ids']
        score = summarize_rows(ids, config['harbor']['k'], [{**r, 'reward': r['success']} for r in rows])
        save(self.root / 'ahe' / domain / 'final_train.json', {**score, 'job': str(job), 'used_for_selection': False})

    def repair_ahe_baseline(self, domain, config, reuse):
        """Replace eight invalid simulator episodes, using the original score/history functions."""
        from ahe_native.capabilities import validate_workspace, restricted_policy
        import copy
        u, experiment = self.ahe_module, self.ahe_experiment(domain)
        first = experiment / 'runs/iteration_001'
        snapshot = first / 'input/workspace'
        validate_workspace(snapshot, restricted_policy(domain))
        repaired = copy.deepcopy(config)
        repaired['native_backend']['reuse_completed_episodes'] = reuse
        job = u.run_harbor(repaired, snapshot, 'code_agent.yaml', first / 'input/benchmark')
        rows = [read(p) for p in job.glob('native_*/native_record.json')]
        spec = self.protocol['domains_spec'][domain]
        score = summarize_rows(spec['batches'][0]['task_ids'], spec['train_trials'],
                               [{**r, 'reward': r['success']} for r in rows])
        if score['infra_errors'] or score.get('missing_episodes'):
            raise RuntimeError('simulator baseline repair did not produce a complete valid measurement')
        stats = u.compute_stats(job, k=spec['train_trials'])
        u.rollback_experiment_metadata(experiment, 1)
        u.update_task_history(experiment, 1, stats['task_results'],
                              per_task_rollouts=stats.get('per_task_rollouts'))
        u.update_history_before(experiment, 1, stats, job)
        u.update_history_after(experiment, 1, (first / 'evolve/evolve_summary.md').read_text())
        u.update_iteration_scores(experiment, config, 1, stats['pass_rate'], stats['n_pass'],
                                  stats['n_total'], job, n_exception=stats['n_exception'], stats=stats)
        u.update_best_ever(experiment, 1, stats)
        save(experiment / 'baseline_repair_receipt.json', {'score': score, 'job': str(job),
            'completed_episodes_reused': len(reuse['episodes']),
            'optimizer_replayed': False, 'selection_core_changed': False,
            'first_optimizer_evidence': 'historical pre-repair TRAIN; paid result retained'})

    def inspect_first_ahe(self, domain):
        path = self.ahe_experiment(domain) / 'runs/iteration_001/evolve/evolve_summary.md'
        jobs = list((self.ahe_experiment(domain) / 'runs/iteration_002/input/benchmark').glob('20*/result.json'))
        if path.is_file() and jobs:
            usage = self.ahe_usage(domain, 1)
            rows = [read(p) for p in jobs[-1].parent.glob('native_*/native_record.json')]
            spec = self.protocol['domains_spec'][domain]
            expected = len(spec['batches'][1]['task_ids']) * spec['train_trials']
            self.first_round['ahe:' + domain] = bool(usage['inference_accounting_complete'] and
                len(rows) == expected and sum(r['infrastructure_error'] for r in rows) <= 2 and
                all(usage['by_role'][role]['total_tokens'] > 0 for role in ('debugger', 'evolve')))

    def report(self):
        from ahe_native.batch import progress
        result = {'pid': os.getpid(), 'gate': self.gate.snapshot(), 'life_agentbench_started': False,
            'first_round_gate': dict(self.first_round), 'meta': {}, 'ahe': {},
            'provider_usage': {'MR': self.mr.ledger.summary(), 'Aliyun': self.ali.ledger.summary()}}
        for domain in DOMAINS:
            path = self.root / 'meta' / domain / 'state.json'
            result['meta'][domain] = read(path) if path.is_file() else {}
            result['ahe'][domain] = {'progress': progress(self.ahe_experiment(domain)),
                'optimizer_usage': self.ahe_usage(domain)}
        save(self.root / 'progress.json', result)
        save(self.root / 'OPTIMIZER_TOKEN_USAGE.json', {'meta': {d: self.meta_ledger.summarize(d) for d in DOMAINS},
            'ahe': {d: {'total': self.ahe_usage(d), 'rounds': {str(i): self.ahe_usage(d, i) for i in range(1, 7)}}
                    for d in DOMAINS}, 'excludes': ['solver', 'user_simulator', 'judge']})
        rows = {'meta': {}, 'ahe': {}}
        for domain in DOMAINS:
            rows['meta'][domain] = result['meta'][domain].get('rounds', [])
            evaluations = {row['iteration']: row for row in result['ahe'][domain]['progress']['evaluations']}
            final_path = self.root / 'ahe' / domain / 'final_train.json'
            final = read(final_path) if final_path.is_file() else None
            rows['ahe'][domain] = []
            for number in range(1, 7):
                value = evaluations.get(number + 1) if number < 6 else final
                episodes = (value.get('tasks', value.get('episodes')) if value else None)
                score = value['successes'] / episodes if value and episodes else None
                usage = self.ahe_usage(domain, number)
                if usage['attempts'] or value:
                    rows['ahe'][domain].append({'round': number, 'evolve_batch': number - 1,
                        'measured_batch': min(number, 5), 'train_episodes': episodes,
                        'candidate_train_accuracy': score, 'optimizer_usage': usage,
                        'selection': 'original best-ever, no extra measurement fed back'})
        save(self.root / 'TRAINING_RESULTS.json', {'methods': rows,
            'note': self.protocol['comparison_caveat'], 'test_results_in_optimizer': False})
        return result

    def smoke(self):
        def check(method, domain):
            ids = self.protocol['domains_spec'][domain]['batches'][0]['task_ids'][:1]
            if method == 'meta':
                score = self.meta_ports[domain].phase('train', 0, 'setup_smoke',
                    self.meta_ports[domain].work / 'agents/baseline_zero.py', 1, smoke=True)
            else:
                score = self.ahe_native(domain, 'train', ids, AHE_NATIVE / 'baselines' / domain,
                                        self.root / 'smoke' / method / domain, round=1)
            if score['infra_errors']:
                raise RuntimeError('setup smoke infrastructure failed: ' + method + ':' + domain)
            return method + ':' + domain, score
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = [pool.submit(check, method, domain) for method in ('meta', 'ahe') for domain in DOMAINS]
            scores = dict(f.result() for f in futures)
        save(self.root / 'setup_smoke.json', {'passed': True, 'domains': scores,
            'optimizer_calls': 0, 'MR_gate': self.gate.snapshot()})

    def training(self):
        from ahe_native.upstream import source_fingerprint as upstream_hashes
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = {f'{m}:{d}': pool.submit(self.run_meta if m == 'meta' else self.run_ahe, d)
                       for m in ('meta', 'ahe') for d in DOMAINS}
            while not all(f.done() for f in futures.values()):
                for domain in DOMAINS:
                    self.inspect_first_ahe(domain)
                failures = [name for name, future in futures.items() if future.done() and future.exception()]
                if failures:
                    save(self.root / 'errors.json', {name: {
                        'type': type(futures[name].exception()).__name__,
                        'error': str(futures[name].exception())} for name in failures})
                    self.abort.set()
                    self.ready.set()
                if len(self.first_round) == 6 and not self.ready.is_set():
                    passed = all(self.first_round.values())
                    save(self.root / 'first_round_gate.json', {'passed': passed, 'methods': self.first_round})
                    if not passed:
                        self.abort.set()
                    self.ready.set()
                self.report()
                time.sleep(15)
            errors = {name: {'type': type(future.exception()).__name__, 'error': str(future.exception())}
                      for name, future in futures.items() if future.exception()}
            if errors:
                save(self.root / 'errors.json', errors)
                raise RuntimeError('training failed; later work stopped, no automatic new experiment')
        if upstream_hashes() != self.ahe_receipt:
            raise RuntimeError('original AHE source changed')
        if source_fingerprint() != read(self.root / 'frozen_sources.json'):
            raise RuntimeError('protocol sources/data changed during training')
        snapshots = {f'meta:{d}': digest(self.meta_ports[d].work / 'agents' / self.meta_ports[d].state['incumbent'])
                     for d in DOMAINS}
        from ahe_native.backend import workspace_hash
        snapshots.update({f'ahe:{d}': workspace_hash(self.ahe_experiment(d) / 'workspace') for d in DOMAINS})
        save(self.root / 'training_seal.json', {'snapshots': snapshots,
            'optimizer_usage': read(self.root / 'OPTIMIZER_TOKEN_USAGE.json')})

    def tests(self):
        # Optimization is fully sealed. Only simulator/judge transports remain open.
        self.meta_broker.close()
        self.meta_broker = None
        self.mr.blocked_roles.update({'debugger', 'evolve', 'life_optimizer'})
        seal = read(self.root / 'training_seal.json')
        def test(method, domain):
            if method == 'meta':
                port = self.meta_ports[domain]
                if digest(port.work / 'agents' / port.state['incumbent']) != seal['snapshots']['meta:' + domain]:
                    raise RuntimeError('Meta test candidate changed after training seal')
                return method + ':' + domain, port.phase('test', 0, 'final_test',
                    port.work / 'agents' / port.state['incumbent'], 6)
            from ahe_native.backend import workspace_hash
            if workspace_hash(self.ahe_experiment(domain) / 'workspace') != seal['snapshots']['ahe:' + domain]:
                raise RuntimeError('AHE test candidate changed after training seal')
            workspace = self.root / 'test_snapshots' / domain
            shutil.copytree(self.ahe_experiment(domain) / 'workspace', workspace)
            return method + ':' + domain, self.ahe_native(domain, 'test',
                self.protocol['domains_spec'][domain]['test_task_ids'], workspace,
                self.root / 'final_test/ahe' / domain)
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = [pool.submit(test, m, d) for m in ('meta', 'ahe') for d in DOMAINS]
            scores = dict(f.result() for f in futures)
        save(self.root / 'FINAL_RESULTS.json', {'split': 'test', 'methods': scores,
            'selection': 'sealed training-retained snapshots, not best-of-test'})

    def close(self):
        if getattr(self, 'meta_broker', None) is not None:
            self.meta_broker.close()
        for name in ('solver', 'mr', 'ali'):
            if getattr(self, name, None) is not None:
                getattr(self, name).close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--smoke-only', action='store_true')
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--resume-from', help='audit/reuse the interrupted first run; wait for its process to exit')
    parser.add_argument('--repair-date-imports', action='store_true')
    args = parser.parse_args()
    for name in filter(None, (args.name, args.resume_from)):
        if not name.replace('_', '').replace('-', '').isalnum():
            raise ValueError('isolated simple experiment name required')
    if args.resume_from and args.smoke_only:
        raise ValueError('continuation reuses verified setup; smoke-only must be a new run')
    if args.repair_date_imports and not args.resume_from:
        raise ValueError('date repair requires an audited paid ancestor')
    root = ROOT / 'evolution/experiments' / args.name
    if args.launch:
        if root.exists():
            from .recovery import live_process
            status = read(root / 'status.json') if (root / 'status.json').is_file() else {}
            phases = {'waiting_for_ancestor_quiescence'}
            if args.repair_date_imports:
                phases.add('stopped_pending_import_contract_choice')
            if (not args.resume_from or status.get('phase') not in phases
                    or live_process(status['pid']) or (root / 'protocol.json').exists()
                    or (root / 'ledgers').exists() or status.get('new_api_calls') != 0):
                raise FileExistsError('refuse to overwrite historical experiment or live waiting process')
            archive_name = ('launch_before_date_import_resume.json' if args.repair_date_imports
                            else 'waiting_launch_before_accounting_fix.json')
            save(root / archive_name, read(root / 'launch.json'))
            save(root / 'status_before_authorized_resume.json', status)
        else:
            root.mkdir(parents=True)
        with (root / 'runner.log').open('a') as log:
            process = subprocess.Popen([sys.executable, '-u', '-m', 'evolution.tau_comparison.run', '--name', args.name,
                *(['--smoke-only'] if args.smoke_only else []),
                *(['--resume-from', args.resume_from] if args.resume_from else []),
                *(['--repair-date-imports'] if args.repair_date_imports else [])], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True)
        save(root / 'launch.json', {'pid': process.pid, 'started': time.time(), 'life_agentbench_started': False})
        print(json.dumps({'launched': process.pid, 'path': str(root)}), flush=True)
        return
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / 'protocol.json').exists():
            raise RuntimeError('audit existing state before continuation; no blind API replay')
        protocol = build_protocol(args.name)
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(max(soft, 8192), hard), hard))
        comparison = None
        try:
            continuation = None
            if args.resume_from:
                from .recovery import audit_parent, copy_parent, live_process
                parent = ROOT / 'evolution/experiments' / args.resume_from
                pid = read(parent / 'launch.json')['pid']
                while live_process(pid):
                    save(root / 'status.json', {'phase': 'waiting_for_ancestor_quiescence',
                        'ancestor_pid': pid, 'pid': os.getpid(), 'new_api_calls': 0,
                        'life_agentbench_started': False})
                    time.sleep(45)
                continuation = audit_parent(parent, protocol, repair_date_imports=args.repair_date_imports)
                copy_parent(parent, root, continuation)
                protocol.update(ancestor=str(parent), initialization='audited transport-repair continuation')
            else:
                for domain in DOMAINS:
                    prepare_workspace(root, domain)
            save(root / 'protocol.json', protocol)
            save(root / 'status.json', {'phase': 'setup', 'pid': os.getpid(), 'life_agentbench_started': False})
            comparison = Comparison(root, protocol, smoke_only=args.smoke_only, continuation=continuation)
            save(root / 'frozen_sources.json', source_fingerprint())
            if not continuation:
                comparison.smoke()
            if not args.smoke_only:
                save(root / 'status.json', {'phase': 'training', 'pid': os.getpid()})
                comparison.training()
                save(root / 'status.json', {'phase': 'final_test', 'pid': os.getpid()})
                comparison.tests()
            comparison.report()
            save(root / 'status.json', {'phase': 'smoke_complete' if args.smoke_only else 'complete', 'pid': os.getpid(),
                                      'life_agentbench_started': False})
        except Exception as exc:
            save(root / 'status.json', {'phase': 'stopped', 'pid': os.getpid(), 'error_type': type(exc).__name__,
                'error': str(exc), 'life_agentbench_started': False})
            raise
        finally:
            if comparison is not None:
                comparison.close()


if __name__ == '__main__':
    main()
