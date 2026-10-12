"""Audit/reuse interrupted Tau artifacts; execution stays in run.py."""
import ast
import json
from pathlib import Path
import shutil
import sys

from .common import AHE_NATIVE, DOMAINS, ROOT, digest, read, save, summarize_rows, transport_modules


def live_process(pid):
    try:
        value = Path(f'/proc/{int(pid)}/stat').read_text().rsplit(')', 1)[1].split()
    except (OSError, ValueError, IndexError):
        return False
    return value[0] not in {'Z', 'X'}


def ledger_complete(path):
    from ahe_native.token_accounting import TokenLedger
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.is_file() else []
    starts = {r['request_id']: r for r in rows if r['event'] == 'started'}
    ends = {r['request_id']: r for r in rows if r['event'] == 'finished'}
    missing = starts.keys() - ends.keys()
    unknown = [k for k in starts if k in ends and ends[k].get('usage') is None]
    estimates = TokenLedger(path).estimates(starts, ends)
    if missing or any(ends[k].get('http_status') != 429 and
            starts[k].get('role') not in {'user_simulator', 'judge'} and k not in estimates for k in unknown):
        raise RuntimeError('ancestor API usage incomplete; do not automatically restart or estimate it')
    result = {'attempts': len(starts), 'tokens': sum(ends[k]['usage']['total_tokens']
              for k in starts if ends[k].get('usage') is not None)}
    if unknown:
        result['rate_limit_rejections_without_usage'] = sum(ends[k].get('http_status') == 429 for k in unknown)
        result['excluded_simulator_or_judge_unknown_usage_calls'] = sum(
            ends[k].get('http_status') != 429 and starts[k].get('role') in {'user_simulator', 'judge'} for k in unknown)
    if estimates:
        result.update(estimated_usage_calls=len(estimates),
            estimated_tokens=sum(v['total_tokens'] for v in estimates.values()))
        result['tokens_including_estimates'] = result['tokens'] + result['estimated_tokens']
    return result


def audit_meta_zero_api(parent, domain):
    state = read(parent / 'meta' / domain / 'state.json')
    requests = parent / 'meta' / domain / 'flash_requests.jsonl'
    if requests.is_file() and requests.read_text().strip():
        raise RuntimeError('Meta has paid requests; this zero-API repair path cannot replay them')
    if (len(state['rounds']) != 1 or state['rounds'][0]['candidate'] is not None or
            state['rounds'][0]['usage']['recorded_upstream_attempts'] != 0):
        raise RuntimeError('expected exactly one pre-API infrastructure attempt, not a paid Meta round')
    if "KeyError: 'isolated_uid'" not in (parent / 'meta' / domain / 'proposer_01/process.log').read_text():
        raise RuntimeError('only the verified missing-UID transport failure may be resumed here')
    baseline = read(parent / 'meta' / domain / 'evals/batch_00_baseline/summary.json')
    if baseline['infra_errors'] > 2 or baseline.get('missing_episodes'):
        raise RuntimeError('do not reuse an incomplete native training baseline')
    if digest(parent / 'meta' / domain / 'workspace/agents' / state['incumbent']) != baseline['plugin_hash']:
        raise RuntimeError('retained Meta baseline changed after its measurement')
    return state


def audit_meta_cached(parent, domain, spec):
    """Keep completed paid rounds, including recorded failures, without replay."""
    run = parent / 'meta' / domain
    state = read(run / 'state.json')
    Ledger, _ = transport_modules()
    ledger = Ledger(parent / 'meta')
    rounds = state['rounds']
    count = max(1, len(rounds))
    roles = {'parent', 'analysis', 'implementation'}
    if count > 6 or any(r['round'] not in range(1, count + 1) for r in ledger.rows(domain)):
        raise RuntimeError('uncommitted paid Meta round; do not automatically replay it')
    if not rounds and (domain != 'telecom' or (run / 'evals/candidate_01/summary.json').exists()):
        raise RuntimeError('only the interrupted Telecom candidate evaluation may resume')
    batch, incumbent, phase = 0, 'baseline_zero.py', 'batch_00_baseline'
    for number in range(1, count + 1):
        row = rounds[number - 1] if rounds else None
        usage = ledger.summarize(domain, number)
        if (not usage['accounting_complete'] or not set(usage['by_role']) <= roles
                or number == 1 and (set(usage['by_role']) != roles or
                    any(v['total_tokens'] <= 0 for v in usage['by_role'].values()))):
            raise RuntimeError('paid Meta recovery requires complete first-round and later usage receipts')
        baseline = read(run / 'evals' / phase / 'summary.json')
        if (baseline['infra_errors'] > 2 or baseline.get('missing_episodes')
                or set(baseline['per_task']) != set(spec['batches'][batch]['task_ids'])
                or baseline['plugin_hash'] != digest(run / 'workspace/agents' / incumbent)
                or baseline['config'] != {'domain': domain, 'split': 'train',
                    'task_ids': spec['batches'][batch]['task_ids'], 'trials': spec['train_trials'],
                    'candidate_sha256': baseline['plugin_hash'],
                    'solver_total_context': 32768, 'solver_output': 4096}
                or baseline['episodes'] != len(spec['batches'][batch]['task_ids']) * spec['train_trials']):
            raise RuntimeError('paid Meta baseline is incomplete or from another batch')
        if row and (row['iteration'] != number or row['train_batch'] != batch
                or row['parent'] != incumbent or row['usage'] != usage
                or row['baseline_successes'] != baseline['successes']
                or row['baseline_score'] != baseline['accuracy']):
            raise RuntimeError('paid Meta history differs from its baseline or usage receipts')
        pending = run / f'pending_{number:02d}.json'
        score_path = run / f'evals/candidate_{number:02d}/summary.json'
        if row and row['candidate'] is None:
            rejection = read(run / f'proposer_{number:02d}/rejection.json')
            result = read(run / f'proposer_{number:02d}/traces/result.json')
            if (number == 1 or pending.exists() or score_path.exists() or result.get('success') is not False
                    or rejection != {'reason': 'API proposer failed', 'result': result}):
                raise RuntimeError('unmeasured Meta proposal lacks its original failure receipt')
        else:
            receipt = read(run / f'proposer_{number:02d}/receipt.json')
            document = read(pending)
            if (len(document['candidates']) != 1 or not receipt['success']
                    or receipt['delegations'] != ['analysis', 'implementation']):
                raise RuntimeError('original paid Meta workflow was not completed')
            candidate = document['candidates'][0]
            source = run / 'workspace/agents' / (candidate['name'] + '.py')
            preflight = read(run / f'preflight_{number:02d}.json')
            if (receipt['pending_sha256'] != digest(pending) or receipt['candidate_sha256'] != digest(source)
                    or preflight['passed'] and preflight != {'passed': True, 'sha256': digest(source)}
                    or row and row['candidate'] != candidate):
                raise RuntimeError('paid Meta candidate or preflight receipt changed')
            if not preflight['passed'] and (number == 1 or score_path.exists()):
                raise RuntimeError('invalid Meta preflight/measurement combination')
            if row and preflight['passed']:
                score = read(score_path)
                if (set(usage['by_role']) != roles or row['candidate_successes'] != score['successes']
                        or row['candidate_score'] != score['accuracy'] or score['infra_errors'] > 2
                        or score.get('missing_episodes') or score['plugin_hash'] != digest(source)
                        or score['config'] != {**baseline['config'], 'candidate_sha256': digest(source)}
                        or score['episodes'] != baseline['episodes']
                        or row['accepted'] != (score['successes'] > baseline['successes'])):
                    raise RuntimeError('paid Meta round does not match its complete measurement')
        if row:
            if not score_path.exists() and (row['candidate_score'] is not None or row['accepted']):
                raise RuntimeError('unmeasured Meta round cannot have a score or be accepted')
            if row['accepted']:
                incumbent, phase = row['candidate']['name'] + '.py', f'candidate_{number:02d}'
            elif number < 6:
                batch, phase = min(batch + 1, 5), f'batch_{min(batch + 1, 5):02d}_baseline'
            if row['retained_candidate'] != incumbent:
                raise RuntimeError('retained Meta candidate differs from the original decision')
    if rounds and (state['incumbent'] != incumbent or state['batch'] != batch):
        raise RuntimeError('Meta continuation state differs from completed round history')
    return state


def audit_debugger_cache(second, paid):
    """Validate the completed analysis reused by the original AHE loop."""
    number = int(second.name.rsplit('_', 1)[1])
    analysis = second / 'input/analysis'
    if not paid and not (analysis / 'overview.md').exists():
        return None
    projection = read(second / 'debugger/evidence_projection.json')
    selected = next(r['selected_tasks'] for r in projection['iterations']
                    if r['iteration'] == second.name)
    details = list((analysis / 'detail').glob('*.md'))
    if (len(paid) != 1 or paid[0]['pending_calls']
            or paid[0]['unknown_usage_calls'] != paid[0].get('rate_limit_rejections_without_usage', 0)
            or paid[0]['total_tokens'] <= 0 or projection['current_iteration'] != number
            or projection['detailed_tasks_max'] != 10 or not 1 <= len(details) == len(selected) <= 10
            or {p.stem for p in details} != set(selected)
            or not (analysis / 'overview.md').read_text().startswith(f'# Debugger Analysis Overview — Iteration {number}\n')
            or any(not p.read_text().strip() for p in details)):
        raise RuntimeError('paid AHE Debugger cache incomplete; refuse replay')
    return {str(p.relative_to(second)): digest(p) for p in [analysis / 'overview.md', *details]}


def audit_completed_evolve(experiment, domain, paid, iteration=2):
    """Reuse original paid output only behind a diagnosed, now-passing compatibility rejection."""
    from ahe_native.backend import workspace_hash
    from ahe_native.capabilities import restricted_policy, validate_workspace
    from ahe_native.evolve_isolation import validate_completed_cache
    from ahe_native.isolation import BASE, require_regular_workspace
    second = experiment / f'runs/iteration_{iteration:03d}'
    rejection = read(second / 'evolve/capability_rejection.json')
    private = Path(rejection['private_candidate']).parent
    case = private.parent
    if (rejection['candidate_promoted'] or rejection['error'] not in {
            'restricted getattr accepts public hook fields only',
            'restricted middleware import is not allowed',
            'restricted tool descriptions must preserve names, schemas and dispatch settings'}
            or private.name != 'experiment' or case.parent != BASE / 'episodes'
            or len(paid) != 1 or paid[0]['pending_calls']
            or paid[0]['unknown_usage_calls'] != paid[0].get('rate_limit_rejections_without_usage', 0) +
               paid[0].get('estimated_usage_calls', 0)
            or paid[0]['total_tokens'] <= 0):
        raise RuntimeError('completed Evolve cache lacks its exact rejection and complete paid receipts')
    request = read(case / 'evolve_request.json')
    if (not 1 <= iteration <= 6 or request['iteration'] != iteration or request['config']['native_backend']['domain'] != domain
            or '[evolve] Evolve agent completed' not in (second / 'evolve/isolated_process.log').read_text()
            or workspace_hash(private / f'runs/iteration_{iteration:03d}/input/workspace') !=
               workspace_hash(second / 'input/workspace')):
        raise RuntimeError('completed Evolve belongs to another iteration/domain/input')
    require_regular_workspace(private / 'workspace')
    validate_workspace(private / 'workspace', restricted_policy(domain))
    paths = [case / 'evolve_request.json', case / 'evolve_result.json',
             *(private / 'workspace').rglob('*'), *(private / f'runs/iteration_{iteration:03d}/evolve').rglob('*')]
    cache = {'iteration': iteration, 'private_experiment': str(private),
        'input_workspace_sha256': workspace_hash(second / 'input/workspace'),
        'candidate_workspace_sha256': workspace_hash(private / 'workspace'),
        'files': {str(p.relative_to(case)): digest(p) for p in paths
                  if p.is_file() and '__pycache__' not in p.parts}}
    validate_completed_cache(cache, second / 'input/workspace')
    return cache


def audit_ahe_rounds(experiment, domain, spec, paid):
    """Find the first unfinished original step; never replay completed paid optimization."""
    from ahe_native.backend import workspace_hash
    from ahe_native.capabilities import restricted_policy, validate_workspace
    if any(g['round'] not in range(1, 7) for g in paid):
        raise RuntimeError('unexpected paid AHE step outside six-round protocol')
    plan = {'name': experiment.name, 'start_iteration': 2, 'skip_eval': False,
            'completed_episodes_reused': 0}
    for number in range(2, 7):
        folder = experiment / f'runs/iteration_{number:03d}'
        snapshot = folder / 'input/workspace'
        jobs = list((folder / 'input/benchmark').glob('20*/result.json'))
        current_paid = [g for g in paid if g['round'] == number]
        if not jobs:
            if current_paid or (folder / 'input/analysis/overview.md').exists():
                raise RuntimeError('paid AHE step lacks its complete native measurement')
            if list((folder / 'input/benchmark/.native_pending').glob('*/*/record.json')):
                raise RuntimeError('partial AHE evaluation requires an episode-level reuse audit')
            break
        rows = [read(p) for p in jobs[0].parent.glob('native_*/native_record.json')]
        expected = {(t, i) for t in spec['batches'][number - 1]['task_ids'] for i in range(spec['train_trials'])}
        if (len(jobs) != 1 or len(rows) != len(expected)
                or {(r['task_id'], r['trial']) for r in rows} != expected
                or read(jobs[0].parent / 'workspace_receipt.json')['workspace_sha256'] != workspace_hash(snapshot)):
            raise RuntimeError('cached AHE evaluation incomplete, changed or from another batch')
        validate_workspace(snapshot, restricted_policy(domain))
        plan.update(start_iteration=number, skip_eval=True,
                    completed_episodes_reused=plan['completed_episodes_reused'] + len(rows))
        evolve_paid = [g for g in current_paid if g['role'] == 'evolve']
        summary = folder / 'evolve/evolve_summary.md'
        if summary.is_file():
            if (len(evolve_paid) != 1 or evolve_paid[0]['pending_calls']
                    or evolve_paid[0]['unknown_usage_calls'] !=
                       evolve_paid[0].get('rate_limit_rejections_without_usage', 0) +
                       evolve_paid[0].get('estimated_usage_calls', 0)
                    or evolve_paid[0]['total_tokens'] <= 0 or not summary.read_text().strip()
                    or not read(folder / 'evolve/capability_validation.json')['candidate_valid']):
                raise RuntimeError('promoted Evolve lacks complete validation/usage')
            plan.update(start_iteration=number + 1, skip_eval=False)
            continue
        cache = audit_debugger_cache(folder, [g for g in current_paid if g['role'] == 'debugger'])
        if cache:
            plan['debugger_analysis_reused'] = cache
        if evolve_paid:
            plan['reuse_completed_evolve'] = audit_completed_evolve(experiment, domain, evolve_paid, number)
        break
    if any(g['round'] > plan['start_iteration'] for g in paid):
        raise RuntimeError('paid AHE suffix lies beyond its audited resume boundary')
    return plan


def cached_episodes(benchmark, snapshot, inherited=(), *, exclude_infra=False):
    from ahe_native.backend import workspace_hash
    episodes, pairs = [], set()
    folders = [Path(item['folder']) for item in inherited]
    for item in inherited:
        if any(digest(Path(item['folder']) / name) != sha for name, sha in item['files'].items()):
            raise RuntimeError('inherited episode changed')
    folders += [p.parent for p in (benchmark / '.native_pending').glob('*/*/record.json')]
    for folder in folders:
        record = read(folder / 'record.json')
        if exclude_infra and record['infrastructure_error']:
            continue
        pair = (record['task_id'], record['trial'])
        if pair in pairs:
            raise RuntimeError('duplicate cached episode')
        pairs.add(pair)
        files = [folder / 'request.json', folder / 'record.json', *(folder / 'agent').rglob('*')]
        episodes.append({'folder': str(folder), 'files': {
            str(p.relative_to(folder)): digest(p) for p in files if p.is_file()}})
    return {'workspace_sha256': workspace_hash(snapshot), 'episodes': episodes}


def audit_final_train(parent, domain, experiment, spec):
    """Reuse a complete last-candidate measurement, never select on it or replay it."""
    from ahe_native.backend import workspace_hash
    receipt = parent / 'ahe' / domain / 'final_train.json'
    folder = parent / 'ahe/final_train' / domain
    if not receipt.is_file():
        if folder.exists():
            raise RuntimeError('partial final TRAIN requires an episode-level reuse audit')
        return None
    value = read(receipt)
    job = Path(value['job'])
    if not job.is_relative_to(folder) or not (job / 'result.json').is_file():
        raise RuntimeError('final TRAIN job is missing or outside its domain')
    rows = [read(p) for p in job.glob('native_*/native_record.json')]
    score = summarize_rows(spec['batches'][5]['task_ids'], spec['train_trials'],
                           [{**r, 'reward': r['success']} for r in rows])
    if (value != {**score, 'job': str(job), 'used_for_selection': False}
            or score.get('missing_episodes') or read(job / 'workspace_receipt.json')['workspace_sha256'] !=
               workspace_hash(experiment / 'workspace')):
        raise RuntimeError('final TRAIN score, task identities or workspace changed')
    return {'completed_episodes_reused': score['episodes'],
            'files': {str(p.relative_to(parent)): digest(p) for p in
                      [receipt, *folder.rglob('*')] if p.is_file() and '__pycache__' not in p.parts}}


def audit_unscoped_evolve(parent, ledger, name):
    """Attribute the known resume-address incident by exact original trace receipts."""
    key = 'unscoped|unknown|None|unknown'
    group = ledger['groups'].get(key)
    if not group:
        return None
    trace = parent / 'ahe/experiments' / name / 'runs/iteration_001/evolve/nexau_in_memory_tracer.cleaned.json'
    evidence = read(trace)
    rows = [json.loads(line) for line in (parent / 'ledgers/mr_requests.jsonl').read_text().splitlines()]
    starts = [r for r in rows if r['event'] == 'started' and r['role'] == 'unknown']
    ends = {r['request_id']: r for r in rows if r['event'] == 'finished'}
    begin, end = evidence['timestamp'], evidence['timestamp'] + evidence['latency'] / 1000
    if (not group['accounting_complete'] or evidence['subagents']
            or group['attempts'] != evidence['generation_count']
            or group['total_tokens'] != evidence['total_tokens']
            or any(r['experiment'] != 'unscoped' or r['model'] != evidence['model']
                   or not begin <= r['utc_unix'] <= ends[r['request_id']]['utc_unix'] <= end + 1
                   for r in starts)):
        raise RuntimeError('unscoped optimizer usage cannot be uniquely attributed to its original trace')
    return {'group_key': key, 'raw_group': group,
        'scope': {'experiment': name, 'domain': 'telecom', 'round': 1, 'role': 'evolve'},
        'trace': str(trace), 'trace_sha256': digest(trace),
        'request_ids': [r['request_id'] for r in starts], 'raw_ledger_modified': False}


def audit_parent(parent, protocol, *, repair_date_imports=False):
    sys.path.insert(0, str(AHE_NATIVE))
    from ahe_native.backend import workspace_hash
    from ahe_native.capabilities import restricted_policy, validate_workspace
    from ahe_native.token_accounting import TokenLedger
    if (live_process(read(parent / 'launch.json')['pid']) or
            read(parent / 'status.json')['phase'] != 'stopped'):
        raise RuntimeError('ancestor must be stopped and its process gone; never overlap MR gates')
    if (parent / 'training_seal.json').exists() or (parent / 'FINAL_RESULTS.json').exists():
        raise RuntimeError('do not reopen a sealed training/test experiment')
    previous = read(parent / 'protocol.json')
    for key in ('domains_spec', 'solver', 'user_routes', 'global_mr_concurrency', 'ahe', 'meta'):
        if previous[key] != protocol[key]:
            raise RuntimeError('native splits/budgets/models/optimizer protocol changed')
    inherited = read(parent / 'continuation_receipt.json') if (parent / 'continuation_receipt.json').is_file() else {}
    incident_root = parent
    incident_path = parent / 'USER_SIMULATOR_OUTPUT_INCIDENT.json'
    seen = {parent}
    while not incident_path.is_file() and (incident_root / 'continuation_receipt.json').is_file():
        incident_root = Path(read(incident_root / 'continuation_receipt.json')['ancestor'])
        if incident_root in seen:
            raise RuntimeError('cyclic continuation provenance')
        seen.add(incident_root)
        incident_path = incident_root / 'USER_SIMULATOR_OUTPUT_INCIDENT.json'
    incident = read(incident_path) if incident_path.is_file() else None
    repairs = {str(ROOT / 'evolution/tau_comparison/meta_port.py'): ['meta_port.before_uid.py', 'meta_port.before_date_import.py'],
        str(ROOT / 'evolution/tau_comparison/run.py'): ['run.before_recovery.py', 'run.before_paid_resume.py', 'run.before_scope_fix.py', 'run.before_date_import.py', 'run.before_completed_evolve_cache.py', 'run.before_usage_estimates.py', 'run.before_failure_reporting.py', 'run.before_final_train_reuse.py'],
        str(ROOT / 'evolution/tau_comparison/recovery.py'): ['recovery.before_paid_resume.py', 'recovery.before_scope_fix.py',
                                                       'recovery.before_multiround_resume.py', 'recovery.before_date_import.py', 'recovery.before_completed_evolve_cache.py', 'recovery.before_usage_estimates.py', 'recovery.before_general_round_resume.py', 'recovery.before_final_train_reuse.py'],
        str(AHE_NATIVE / 'ahe_native/capabilities.py'): ['capabilities.before_date_import.py', 'capabilities.before_public_accessors.py', 'capabilities.before_parameter_descriptions.py', 'capabilities.before_literal_parser.py'],
        str(ROOT / 'evolution/tau_comparison/meta_evaluate.py'): ['meta_evaluate.before_user_budget.py'],
        str(ROOT / 'evolution/tau_comparison/test_transport.py'): ['test_transport.before_user_budget.py'],
        str(AHE_NATIVE / 'ahe_native/tau_bridge.py'): ['tau_bridge.before_user_budget.py'],
        str(AHE_NATIVE / 'ahe_native/evolve_isolation.py'): ['evolve_isolation.before_scope_fix.py', 'evolve_isolation.before_completed_cache.py'],
        str(AHE_NATIVE / 'ahe_native/capability_evidence.py'): ['capability_evidence.before_history_projection.py'],
        str(AHE_NATIVE / 'ahe_native/backend.py'): ['backend.before_recovery.py'],
        str(AHE_NATIVE / 'ahe_native/upstream.py'): ['upstream.before_recovery.py'],
        str(AHE_NATIVE / 'ahe_native/metered_broker.py'): ['metered_broker.before_recovery.py', 'metered_broker.before_usage_preservation.py'],
        str(AHE_NATIVE / 'ahe_native/token_accounting.py'): ['token_accounting.before_recovery.py', 'token_accounting.before_usage_estimates.py']}
    for path, sha in read(parent / 'frozen_sources.json').items():
        sources = [Path(path)] + [parent / 'source_archive' / name for name in repairs.get(path, [])]
        if not any(p.is_file() and digest(p) == sha for p in sources):
            raise RuntimeError('unexpected historical source/data change: ' + path)
    if incident:
        if (incident['repair_applied']['telecom_user_max_output'] != 100000 or
                incident['evaluation_stop']['ambiguous_api_receipts'] != 0 or
                len(incident['evidence']) != 8):
            raise RuntimeError('unexpected simulator repair incident')
        for row in incident['evidence']:
            if digest(incident_root / row['record']) != row['sha256']:
                raise RuntimeError('simulator incident episode changed')
        states = {d: audit_meta_cached(parent, d, protocol['domains_spec'][d]) for d in DOMAINS}
    else:
        states = {d: audit_meta_zero_api(parent, d) for d in DOMAINS}
    ledger = TokenLedger(parent / 'ledgers/mr_requests.jsonl').summary()
    plans = {}
    for domain in DOMAINS:
        name = inherited.get('ahe_resume', {}).get(domain, {}).get('name', parent.name + '_ahe_' + domain)
        experiment = parent / 'ahe/experiments' / name
        first = experiment / 'runs/iteration_001'
        second = experiment / 'runs/iteration_002'
        paid = [g for g in ledger['groups'].values() if g['experiment'] == name and
                g['role'] in {'debugger', 'evolve'}]
        completed = (first / 'evolve/evolve_summary.md').is_file()
        if completed:
            plans[domain] = audit_ahe_rounds(experiment, domain, protocol['domains_spec'][domain], paid)
            snapshot = second / 'input/workspace'
            if incident and domain == 'telecom':
                jobs = list((second / 'input/benchmark').glob('20*/result.json'))
                rows = [read(p) for p in jobs[0].parent.glob('native_*/native_record.json')]
                if sum(r['infrastructure_error'] for r in rows) > 2:
                    raise RuntimeError('repaired Telecom candidate evaluation still has widespread errors')
            repaired_baseline = experiment / 'baseline_repair_receipt.json'
            if repaired_baseline.is_file():
                repair = read(repaired_baseline)
                repaired_rows = [read(p) for p in Path(repair['job']).glob('native_*/native_record.json')]
                spec = protocol['domains_spec'][domain]
                expected = {(task, trial) for task in spec['batches'][0]['task_ids'] for trial in range(spec['train_trials'])}
                if (repair['optimizer_replayed'] or repair['selection_core_changed']
                        or len(repaired_rows) != len(expected)
                        or {(r['task_id'], r['trial']) for r in repaired_rows} != expected
                        or any(r['infrastructure_error'] for r in repaired_rows)):
                    raise RuntimeError('corrected baseline receipt is incomplete or changed')
            elif incident and domain == 'telecom':
                baseline_snapshot = first / 'input/workspace'
                old = inherited.get('ahe_resume', {}).get(domain, {}).get('reuse_completed_episodes', {})
                reuse = cached_episodes(first / 'input/benchmark', baseline_snapshot,
                                        old.get('episodes', []), exclude_infra=True)
                if len(reuse['episodes']) != len(protocol['domains_spec'][domain]['batches'][0]['task_ids']) - 8:
                    raise RuntimeError('expected exactly eight diagnosed baseline episodes to repair')
                plans[domain]['repair_baseline'] = reuse
        else:
            if paid or domain != 'telecom' or 'TimeoutExpired' not in (parent / 'runner.log').read_text():
                raise RuntimeError('cannot replay paid/incomplete optimization or an unverified failure')
            snapshot = first / 'input/workspace'
            if workspace_hash(experiment / 'workspace') != workspace_hash(snapshot):
                raise RuntimeError('interrupted baseline workspace changed')
            request = read(first / 'input/benchmark/native_request.json')
            if request['iteration'] != 1 or request['domain'] != domain:
                raise RuntimeError('partial baseline belongs to another iteration/domain')
            reuse = cached_episodes(first / 'input/benchmark', snapshot)
            plans[domain] = {'name': name, 'start_iteration': 1, 'skip_eval': False,
                'completed_episodes_reused': len(reuse['episodes']), 'reuse_completed_episodes': reuse}
        validate_workspace(snapshot, restricted_policy(domain))
        if plans[domain]['start_iteration'] == 7:
            final = audit_final_train(parent, domain, experiment, protocol['domains_spec'][domain])
            if final:
                plans[domain]['final_train_reused'] = final
    costs = {p: ledger_complete(parent / 'ledgers' / (p + '_requests.jsonl')) for p in ('mr', 'aliyun')}
    date_candidates = {}
    if repair_date_imports:
        from .meta_port import validate_candidate
        for domain, state in states.items():
            values = {}
            for row in state['rounds']:
                if row['candidate'] is None or row['candidate_score'] is not None:
                    continue
                path = parent / 'meta' / domain / 'workspace/agents' / (row['candidate']['name'] + '.py')
                tree = ast.parse(path.read_text())
                if not any(isinstance(n, ast.Import) and any(a.name == 'datetime' for a in n.names)
                           or isinstance(n, ast.ImportFrom) and n.module == 'datetime' for n in ast.walk(tree)):
                    continue
                validate_candidate(path, domain)  # All other checks remain mandatory.
                values[row['iteration']] = digest(path)
            if values:
                if set(values) != set(range(min(values), 7)):
                    raise RuntimeError('date repair requires an uninterrupted rejected suffix')
                date_candidates[domain] = values
        if not date_candidates:
            raise RuntimeError('no date-only paid rejections available to repair')
    return {'meta_states': states, 'ahe_resume': plans, 'inherited_provider_costs': costs,
        'meta_date_candidates': date_candidates,
        'optimizer_scope_correction': audit_unscoped_evolve(parent, ledger, plans['telecom']['name']),
        'meta_pre_api_attempts_discarded_from_round_count': 0 if incident else 3,
        'meta_paid_cache_reused': bool(incident), 'paid_meta_requests_replayed': 0,
        'raw_ledgers_preserved_without_relabeling': True}


def copy_parent(parent, destination, audit):
    for domain in DOMAINS:
        src = parent / 'meta' / domain
        dst = destination / 'meta' / domain
        shutil.copytree(src, dst)
        state = dict(audit['meta_states'][domain])
        if audit.get('meta_paid_cache_reused'):
            # Airline/retail completed the unchanged Aliyun condition. Telecom
            # remeasures both sides under the fixed simulator, without new LLM optimization.
            if domain == 'telecom' and not state['rounds']:
                archive = dst / 'simulator_output_repair'
                archive.mkdir()
                for name in ('batch_00_baseline', 'candidate_01'):
                    path = dst / 'evals' / name
                    if path.exists():
                        path.rename(archive / name)
                logs = dst / 'workspace/logs/round_01'
                if logs.exists():
                    logs.rename(archive / 'workspace_round_01')
                state.update(baseline_phase=None, status='running')
        else:
            archive = dst / 'pre_api_uid_failure'
            archive.mkdir()
            for name in ('proposer_01', 'prompt_01.md', 'preflight_01.json'):
                path = dst / name
                if path.exists():
                    path.rename(archive / name)
            save(archive / 'state_before_repair.json', state)
            state.update(rounds=[], status='running', batch=0, baseline_phase='batch_00_baseline')
            logs = dst / 'workspace/logs/round_01'
            if logs.exists():
                logs.rename(archive / 'workspace_round_01')
        save(dst / 'state.json', state)
        save(dst / 'workspace/history.json', state['rounds'])
        name = audit['ahe_resume'][domain]['name']
        src_ahe = parent / 'ahe/experiments' / name
        dst_ahe = destination / 'ahe/experiments' / name
        shutil.copytree(src_ahe, dst_ahe, ignore=shutil.ignore_patterns('.native_pending', '__pycache__'))
        for path in (dst_ahe / 'runs').glob('*/input/analysis/**/*.md'):
            path.write_text(path.read_text().replace(str(src_ahe), str(dst_ahe)))
        final = audit['ahe_resume'][domain].get('final_train_reused')
        if final:
            if any(digest(parent / name) != sha for name, sha in final['files'].items()):
                raise RuntimeError('final TRAIN changed after audit; refuse reuse')
            relative = Path('ahe/final_train') / domain
            shutil.copytree(parent / relative, destination / relative,
                            ignore=shutil.ignore_patterns('.native_pending', '__pycache__'))
            value = read(parent / 'ahe' / domain / 'final_train.json')
            value['job'] = str(destination / Path(value['job']).relative_to(parent))
            save(destination / 'ahe' / domain / 'final_train.json', value)
    shutil.copytree(parent / 'ledgers', destination / 'ledgers')
    shutil.copy2(parent / 'setup_smoke.json', destination / 'setup_smoke.json')
    save(destination / 'continuation_receipt.json', {'ancestor': str(parent), **audit,
        'ancestor_protocol_sha256': digest(parent / 'protocol.json'),
        'ancestor_ledger_sha256': {p.name: digest(p) for p in (parent / 'ledgers').glob('*requests.jsonl')},
        'transport_fixes': (['Telecom simulator output 100000; paid proposals reused; corrected baseline scores']
            if audit.get('meta_paid_cache_reused') else
            ['supply isolated_uid', 'Tau host episode deadline matches native evaluator']),
        'baseline_policy_manually_modified': False, 'initialization': 'audited continuation, not a fresh run'})
