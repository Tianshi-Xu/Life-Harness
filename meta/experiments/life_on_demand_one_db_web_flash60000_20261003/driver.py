"""Exactly ONE Life decision per domain; TRAIN only, never automatic TEST."""
from __future__ import annotations
import copy
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from meta.life_optimizer import profile as optimizer
from meta.life_optimizer.context import ProposalFailure
from composition import commit_bundle, stage_bundle, verify_active, write_active_chain
from protocol import (DOMAINS, HERE, ROOT, ROUNDS, atomic_json, bind_fingerprint,
                      digest, object_digest, protocol, task_ids, validate_protocol)

SOURCES = ('protocol.py', 'candidate_contract.py', 'driver.py', 'composition.py',
           'agentbench_chain.py', 'agentbench_evaluate.py', 'runner.py')

def run_fingerprint(domain):
    return {'experiment': HERE.name, 'domain': domain, 'rounds': ROUNDS,
            'protocol_sha256': digest(HERE / 'protocol.json'),
            'initial_active_snapshot': [], 'prior_test_exposure': True,
            'source_sha256': {name: digest(HERE / name) for name in SOURCES},
            'transport_sha256': {name: digest(ROOT.parent / 'ahe-native/ahe_native' / name)
                                 for name in ('api_broker.py', 'connection.py')},
            'optimizer_source_sha256': optimizer.source_hashes()}

def request_rows(domain, number):
    path = HERE / domain / 'flash_requests.jsonl'
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()
            if json.loads(line).get('round') == number]

def failure_receipt(domain, number, workspace, code, detail):
    """Persist a normal failed attempt; never promote partial files or score them."""
    run_dir = HERE / domain
    checkpoint = run_dir / 'optimizer_sessions' / f'round_{number:02d}' / 'agent_checkpoint.json'
    saved = json.loads(checkpoint.read_text())
    if saved.get('inflight'):
        raise RuntimeError('Cannot classify an unknown in-flight request as a normal failure')
    sandbox = checkpoint.parent / 'workspace'
    actual = {str(p.relative_to(sandbox)): digest(p) for p in sandbox.rglob('*') if p.is_file()}
    if actual != saved['workspace_hashes']:
        raise RuntimeError('Failed proposal checkpoint changed')
    receipt = optimizer.verify_inputs(workspace)
    rows = request_rows(domain, number)
    reads = sandbox / optimizer.READ_LOG
    access = json.loads(reads.read_text()) if reads.is_file() else {'case_ids': [], 'requests': []}
    value = {'schema_version': 1, 'domain': domain, 'round': number, 'code': code,
             'detail': detail, 'input_sha256': object_digest(receipt),
             'checkpoint_path': str(checkpoint.relative_to(run_dir)), 'checkpoint_sha256': digest(checkpoint),
             'proposer_requests': len(rows), 'agent_request_slots': saved['requests'],
             'usage': saved['usage'], 'cases_opened': len(access['case_ids']),
             'request_receipts': rows, 'candidate_measured': False, 'incumbent_retained': True}
    path = run_dir / 'failures' / f'round_{number:02d}.json'
    bind_fingerprint(path, value)
    return path, value

def verify_failure(domain, number, row, workspace):
    run_dir = HERE / domain
    path = run_dir / row['failure_path']
    if digest(path) != row['failure_sha256']:
        raise RuntimeError('Failed attempt receipt changed')
    failure = json.loads(path.read_text())
    if (failure['code'] not in {'output_truncated', 'context_limit', 'no_valid_candidate'}
            or failure['domain'] != domain or failure['round'] != number
            or failure['input_sha256'] != object_digest(optimizer.verify_inputs(workspace))
            or failure['candidate_measured'] is not False or failure['incumbent_retained'] is not True):
        raise RuntimeError('Failed attempt provenance mismatch')
    checkpoint = run_dir / failure['checkpoint_path']
    if digest(checkpoint) != failure['checkpoint_sha256']:
        raise RuntimeError('Failed attempt checkpoint changed')
    saved = json.loads(checkpoint.read_text())
    sandbox = checkpoint.parent / 'workspace'
    if saved.get('inflight') or saved['workspace_hashes'] != {
            str(p.relative_to(sandbox)): digest(p) for p in sandbox.rglob('*') if p.is_file()}:
        raise RuntimeError('Failed attempt public workspace changed')
    if failure['request_receipts'] != request_rows(domain, number):
        raise RuntimeError('Failed attempt request receipts changed')

def initial_state(domain):
    return {'domain': domain, 'round': 0, 'status': 'running', 'rounds': [],
            'prior_test_exposure': True, 'active_snapshot': [],
            'active_snapshot_sha256': object_digest([]),
            'accepted_bundle_ids': [], 'operation_log': []}

def progress(domain, **values):
    row = {'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'domain': domain, **values}
    atomic_json(HERE / domain / 'progress.json', row)
    print(json.dumps(row, ensure_ascii=False), flush=True)

def child_environment(domain):
    # Host credentials stay in the broker's memory, never in native child envs.
    env = {k: v for k, v in os.environ.items() if not any(
        word in k.upper() for word in ('API_KEY', 'API_TOKEN', 'SECRET', 'PASSWORD'))}
    env.update(TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    if domain == 'webshop':
        java = ROOT / 'AgentBench/.native/sysroot/usr/lib/jvm/java-11-openjdk-amd64'
        env.update(JAVA_HOME=str(java), JVM_PATH=str(java / 'lib/server/libjvm.so'),
                   PATH=str(java / 'bin') + ':' + env.get('PATH', ''), _JAVA_OPTIONS='-Xmx4g',
                   PYTHONPATH=str(ROOT / 'AgentBench/.native/webshop-src') + ':' + str(ROOT / 'AgentBench'))
    return env

def phase_directory(domain, name):
    if name == 'round_01_incumbent':
        return HERE.parent / protocol()['predecessor_run'] / domain / 'evals' / name
    return HERE / domain / 'evals' / name

def phase(domain, split, name, chain, batch):
    if split != 'train':
        raise RuntimeError('This one-round run is TRAIN-only; TEST is not authorized')
    if name == 'round_01_incumbent' and split == 'train':
        directory = phase_directory(domain, name)
        expected = protocol()['reused_baselines'][domain]
        path = directory / 'summary.json'
        if digest(path) != expected['summary_sha256']:
            raise RuntimeError('Reused baseline summary changed')
        from agentbench_evaluate import validate_cached_summary
        summary = json.loads(path.read_text())
        validate_cached_summary(directory, summary)
        from agentbench_chain import manifest_records
        _, records = manifest_records(chain)
        if (records != [] or batch != 0 or summary['snapshot_sha256'] != object_digest([])
                or summary['domain'] != domain or summary['split'] != 'train' or summary['n'] != 50):
            raise RuntimeError('Baseline reuse conditions differ')
        progress(domain, phase=name, status='verified_reused_baseline', successes=summary['successes'], n=50)
        return summary
    python = ROOT / ('AgentBench/.native/core/bin/python' if domain == 'dbbench'
                     else 'AgentBench/.native/webshop-venv/bin/python')
    command = [str(python), '-u', str(HERE / 'agentbench_evaluate.py'), domain,
               split, name, str(chain), '--batch', str(batch)]
    path = HERE / domain / 'evals' / name / 'summary.json'
    if path.exists():
        # Invoke evaluator even for completed cache: it checks current config and episodes.
        progress(domain, phase=name, status='verifying_completed_eval_cache')
    else:
        progress(domain, phase=name, status='evaluating', batch=batch)
    with (HERE / domain / (name + '.log')).open('a') as log:
        result = subprocess.run(command, cwd=ROOT, env=child_environment(domain),
                                stdout=log, stderr=subprocess.STDOUT, timeout=21600)
    if result.returncode:
        raise RuntimeError(f'Evaluation subprocess failed: {domain}/{name}.log')
    return json.loads(path.read_text())

def json_operations(bundle):
    return [{k: v for k, v in op.items() if k != 'artifact_path'} for op in bundle['operations']]

def verify_history(domain, state):
    """Verify all decision receipts, source hashes, complete paired scores, and CAS."""
    run_dir = HERE / domain
    if len(state['rounds']) != state['round'] or state['round'] > ROUNDS:
        raise RuntimeError('Round history length mismatch')
    verify_active(run_dir, state)
    manifest = validate_protocol(check_files=False)
    parent, logs = object_digest([]), []
    from agentbench_evaluate import validate_cached_summary
    for number, row in enumerate(state['rounds'], 1):
        if row['round'] != number or row['train_batch'] != number - 1 or row['parent_snapshot_sha256'] != parent:
            raise RuntimeError('Decision history ordering mismatch')
        failed = row.get('outcome') == 'proposal_failed'
        for kind in (('incumbent',) if failed else ('incumbent', 'candidate')):
            directory = phase_directory(domain, row[kind + '_phase'])
            path = directory / 'summary.json'
            if digest(path) != row[kind + '_summary_sha256']:
                raise RuntimeError('Decision summary changed')
            summary = json.loads(path.read_text())
            validate_cached_summary(directory, summary)
            if summary['snapshot_sha256'] != row[kind + '_snapshot_sha256']:
                raise RuntimeError('Evaluated snapshot mismatch')
            if kind == 'incumbent':
                incumbent = summary
            else:
                candidate = summary
        expected = task_ids(manifest['domains'][domain], number - 1, 'train')
        optimizer.strict_improvement(incumbent, incumbent, expected_task_ids=expected)
        if incumbent['accuracy'] != row['incumbent_score']:
            raise RuntimeError('Decision score mismatch')
        proposal = run_dir / 'proposals' / f'round_{number:02d}'
        workspace = HERE / 'optimizer_inputs' / domain / f'round_{number:02d}'
        receipt = optimizer.verify_inputs(workspace)
        prior_state = json.loads((run_dir / 'round_inputs' / f'round_{number:02d}.json').read_text())
        if (receipt['seed_state_sha256'] != digest(run_dir / 'round_inputs' / f'round_{number:02d}.json')
                or prior_state['active_snapshot_sha256'] != parent
                or receipt['evaluation_sha256'] != row['incumbent_summary_sha256']):
            raise RuntimeError('Optimizer incumbent provenance changed')
        if failed:
            if (row['accepted'] is not False or row['candidate_score'] is not None
                    or row['candidate_phase'] is not None or row['candidate_successes'] is not None
                    or row['operations'] or row['active_snapshot_sha256'] != parent):
                raise RuntimeError('Failed proposal must remain unmeasured and cannot change incumbent')
            verify_failure(domain, number, row, workspace)
        else:
            if row.get('outcome') != 'evaluated_candidate':
                raise RuntimeError('Unknown optimization outcome')
            if optimizer.strict_improvement(candidate, incumbent, expected_task_ids=expected) != row['accepted']:
                raise RuntimeError('Strict paired decision mismatch')
            if candidate['accuracy'] != row['candidate_score']:
                raise RuntimeError('Candidate score mismatch')
            # Exact validated cached output is checked without a network call.
            optimizer.propose(workspace, proposal, completion=lambda *_: (_ for _ in ()).throw(
                RuntimeError('Audit must never call optimizer API')))
            if digest(proposal / 'validation.json') != row['validation_sha256']:
                raise RuntimeError('Proposal validation changed')
        if row['accepted']:
            logs.append(row['bundle_id'])
        parent = row['active_snapshot_sha256']
    if parent != state['active_snapshot_sha256'] or logs != state['accepted_bundle_ids']:
        raise RuntimeError('Final incumbent differs from decision log')

def run_domain(domain, completion):
    if ROUNDS != 1:
        raise RuntimeError('Refuse more than one iteration in this run')
    validate_protocol()
    run_dir = HERE / domain
    run_dir.mkdir(exist_ok=True)
    with (run_dir / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bind_fingerprint(run_dir / 'run_config.json', run_fingerprint(domain))
        state_path = run_dir / 'state.json'
        state = json.loads(state_path.read_text()) if state_path.exists() else initial_state(domain)
        if not state_path.exists():
            atomic_json(state_path, state)
        verify_history(domain, state)
        manifest = validate_protocol(check_files=False)
        for number in range(state['round'] + 1, ROUNDS + 1):
            batch = number - 1
            prior_path = run_dir / 'round_inputs' / f'round_{number:02d}.json'
            bind_fingerprint(prior_path, state)
            chain = write_active_chain(run_dir, state, f'incumbent_{number:02d}')
            incumbent_phase = f'round_{number:02d}_incumbent'
            candidate_phase = f'round_{number:02d}_candidate'
            incumbent = phase(domain, 'train', incumbent_phase, chain, batch)
            workspace = HERE / 'optimizer_inputs' / domain / f'round_{number:02d}'
            proposal = run_dir / 'proposals' / f'round_{number:02d}'
            if not workspace.exists():
                optimizer.prepare(run_dir, phase_directory(domain, incumbent_phase), workspace, seed=number)
            receipt = optimizer.verify_inputs(workspace)
            if (receipt['seed_state_sha256'] != digest(prior_path)
                    or receipt['evaluation_sha256'] != digest(phase_directory(domain, incumbent_phase) / 'summary.json')):
                raise RuntimeError('Optimizer input provenance mismatch')
            progress(domain, round=number, batch=batch, phase='proposing', incumbent=incumbent['successes'], n=50)
            parent = state['active_snapshot_sha256']
            row = {'round': number, 'train_batch': batch, 'incumbent_phase': incumbent_phase,
                   'incumbent_score': incumbent['accuracy'], 'incumbent_successes': incumbent['successes'],
                   'parent_snapshot_sha256': parent,
                   'incumbent_snapshot_sha256': incumbent['snapshot_sha256'],
                   'incumbent_summary_sha256': digest(phase_directory(domain, incumbent_phase) / 'summary.json'),
                   'cases_available': 50, 'incumbent_infra_errors': incumbent['infra_errors']}
            try:
                bundle_path = optimizer.propose(workspace, proposal, completion=completion,
                    max_turns=32, checkpoint_dir=run_dir/'optimizer_sessions'/f'round_{number:02d}')
            except ProposalFailure as exc:
                path, failure = failure_receipt(domain, number, workspace, exc.code, str(exc))
                row.update(outcome='proposal_failed', candidate_phase=None, candidate_score=None,
                    candidate_successes=None, accepted=False, bundle_id=None,
                    bundle_hypothesis='No candidate produced within fixed token/request limits: ' + exc.code,
                    operations=[], candidate_snapshot_sha256=None, candidate_summary_sha256=None,
                    validation_sha256=None, candidate_infra_errors=None,
                    proposer_requests=failure['proposer_requests'], usage=failure['usage'],
                    cases_opened=failure['cases_opened'], failure_path=str(path.relative_to(run_dir)),
                    failure_sha256=digest(path), failure_code=exc.code)
            else:
                bundle, staged, staged_chain = stage_bundle(run_dir, number, bundle_path, state)
                candidate = phase(domain, 'train', candidate_phase, staged_chain, batch)
                accepted = optimizer.strict_improvement(candidate, incumbent,
                    expected_task_ids=task_ids(manifest['domains'][domain], batch, 'train'))
                operations = json_operations(bundle)
                if accepted:
                    records = commit_bundle(run_dir, bundle, state)
                    state['active_snapshot'] = records
                    state['active_snapshot_sha256'] = object_digest(records)
                    state['accepted_bundle_ids'].append(bundle['bundle_id'])
                    state['operation_log'].append({'bundle_id': bundle['bundle_id'], 'round': number,
                        'parent_snapshot_sha256': parent, 'committed_snapshot_sha256': object_digest(records),
                        'operations': operations})
                validation = json.loads((proposal / 'validation.json').read_text())
                row.update(outcome='evaluated_candidate', candidate_phase=candidate_phase,
                    candidate_score=candidate['accuracy'], candidate_successes=candidate['successes'],
                    accepted=accepted, bundle_id=bundle['bundle_id'], bundle_hypothesis=bundle['hypothesis'],
                    operations=operations, candidate_snapshot_sha256=candidate['snapshot_sha256'],
                    candidate_summary_sha256=digest(run_dir/'evals'/candidate_phase/'summary.json'),
                    validation_sha256=digest(proposal/'validation.json'),
                    proposer_requests=validation['proposer_requests'], usage=validation['usage'],
                    cases_opened=len(validation['evidence_access']['case_ids']),
                    candidate_infra_errors=candidate['infra_errors'])
            row['active_snapshot_sha256'] = state['active_snapshot_sha256']
            state['rounds'].append(row)
            state['round'] = number
            state['status'] = 'complete' if number == ROUNDS else 'running'
            verify_active(run_dir, state)
            atomic_json(state_path, state)
            verify_history(domain, state)
            progress(domain, round=number, phase='decision', accepted=row['accepted'], outcome=row['outcome'],
                     incumbent=incumbent['successes'], candidate=row['candidate_successes'], n=50,
                     proposer_requests=row['proposer_requests'], cases_opened=row['cases_opened'])
            render_training_report()
        return state

def seal_value(root):
    return {'experiment': HERE.name, 'protocol_sha256': digest(root / 'protocol.json'),
            'prior_test_exposure': True,
            'state_sha256': {d: digest(root / d / 'state.json') for d in DOMAINS},
            'run_config_sha256': {d: digest(root / d / 'run_config.json') for d in DOMAINS}}

def verify_seal(root=HERE):
    root = Path(root)
    path = root / 'training_seal.json'
    if not path.is_file():
        raise RuntimeError('Test refused before both training domains are sealed')
    validate_protocol(check_files=False)
    for domain in DOMAINS:
        state_path = root / domain / 'state.json'
        if not state_path.is_file():
            raise RuntimeError('Test refused: training state missing')
        state = json.loads(state_path.read_text())
        if state['status'] != 'complete' or state['round'] != ROUNDS or len(state['rounds']) != ROUNDS:
            raise RuntimeError('Seal refused: exactly one training decision required')
        if json.loads((root / domain / 'run_config.json').read_text()) != run_fingerprint(domain):
            raise RuntimeError('Test refused: source fingerprint changed')
        verify_history(domain, state)
    expected = seal_value(root)
    if json.loads(path.read_text()) != expected:
        raise RuntimeError('Test refused: training seal changed')
    return expected

def seal_training():
    for domain in DOMAINS:
        state = json.loads((HERE / domain / 'state.json').read_text())
        if state['status'] != 'complete' or state['round'] != ROUNDS:
            raise RuntimeError('Cannot seal incomplete training')
        verify_history(domain, state)
    bind_fingerprint(HERE / 'training_seal.json', seal_value(HERE))
    return verify_seal()

def preservation_check():
    before = json.loads((HERE / 'historical_preservation_before.json').read_text())
    changed = [name for name, sha in before.items() if digest(ROOT / name) != sha]
    atomic_json(HERE / 'historical_preservation_check.json', {'files_checked': len(before), 'changed': changed})
    if changed:
        raise RuntimeError('Historical artifacts changed')

def render_training_report():
    lines = ['# New Life: ONE-round TRAIN-only validation', '',
        'Empty H2–H5 contents; all 50 current public training cases available on demand, no case-count quota.',
        'Exactly one round, frozen b0. Candidate and incumbent are paired on the same 50 IDs; equality rejected. No TEST evaluation.',
        'Historical training rows may use different batches and batch progression; not a continuous test curve.', '',
        '| Domain | Round | Batch | Incumbent | Candidate | Accepted | Flash requests | Cases opened | Infra incumbent/candidate |',
        '|---|---:|---:|---:|---:|---|---:|---:|---|']
    for domain in DOMAINS:
        path = HERE / domain / 'state.json'
        if not path.exists():
            continue
        for row in json.loads(path.read_text())['rounds']:
            candidate = '未测：' + row['failure_code'] if row['outcome'] == 'proposal_failed' else str(row['candidate_successes']) + '/50'
            lines.append(f"| {domain} | {row['round']} | {row['train_batch']} | {row['incumbent_successes']}/50 | {candidate} | {row['accepted']} | {row['proposer_requests']} | {row['cases_opened']}/50 | {row['incumbent_infra_errors']}/{row['candidate_infra_errors']} |")
    # A single host report writer is serialized across the two domain threads.
    with (HERE / '.report.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        target = HERE / 'TRAINING.md'
        temp = target.with_suffix('.tmp')
        temp.write_text('\n'.join(lines) + '\n')
        temp.replace(target)
    render_comparison()

def render_comparison():
    """Host-only historical TRAIN comparison; never copied to optimizer inputs."""
    sources = {'Meta': HERE.parent / 'meta_from_scratch_six_20260915',
               'old Life': HERE.parent / 'life_agentbench_six_resume_mr_20261001',
               'new Life': HERE}
    states = {}
    for name, root in sources.items():
        for domain in DOMAINS:
            path = root / domain / 'state.json'
            states[name, domain] = json.loads(path.read_text()).get('rounds', []) if path.exists() else []
    lines = ['# 第一轮训练比较（本次只跑一轮）', '',
        '各格：批次，基线→候选成功数/50，接收与否。不同实验可能批次不同，只能描述记录，不能当作同题性能胜负。',
        '本次 Life 从空 H2～H5 开始，只跑第一轮 b0。历史 Meta/Life 的接口和预算不同，不是严格同条件比较。', '',
        '| 场景 | 轮次 | Meta | 旧 Life | 新 Life |', '|---|---:|---|---|---|']
    for domain in DOMAINS:
        for number in range(1, ROUNDS + 1):
            cells = []
            for name in sources:
                row = next((r for r in states[name, domain] if r.get('round', r.get('iteration')) == number), None)
                if row is None:
                    cells.append('待完成')
                else:
                    baseline = row.get('incumbent_score', row.get('baseline_score'))
                    candidate = row.get('candidate_score')
                    result = '接收' if row['accepted'] else '拒绝'
                    if row.get('outcome') == 'proposal_failed':
                        cells.append(f"b{row['train_batch']}，基线 {round(baseline*50)}/50；候选未测（{row['failure_code']}），未接收")
                    else:
                        cells.append(f"b{row['train_batch']}，{round(baseline*50)}→{round(candidate*50)}/50，{result}")
            lines.append(f"| {domain} | {number} | " + ' | '.join(cells) + ' |')
    lines += ['', '这些是训练分数，不是测试分数。本次没有测试集评估，也不会自动推进第二轮。']
    with (HERE / '.comparison.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = HERE / 'COMPARISON.md'
        temp = path.with_suffix('.tmp')
        temp.write_text('\n'.join(lines) + '\n')
        temp.replace(path)
