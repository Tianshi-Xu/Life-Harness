"""Isolated, policy-free ONE-round TRAIN-only Life experiment."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
HISTORICAL_ROOT = HERE.parent / 'meta_from_scratch_six_20260915'
DOMAINS = ('dbbench', 'webshop')
ROUNDS = 1

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + '\n')
    temp.replace(path)

def protocol():
    return json.loads((HERE / 'protocol.json').read_text())

def task_ids(spec, batch, split):
    if split == 'train':
        row = spec['batches'][batch]
        return list(row.get('task_ids', row.get('indices', [])))
    return list(spec.get('test_task_ids', spec.get('test_indices', [])))

def bind_fingerprint(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise RuntimeError(f'Resume fingerprint mismatch: {path}')
    else:
        atomic_json(path, value)

def validate_protocol(check_files=True):
    cfg = protocol()
    if (ROUNDS != 1 or cfg['experiment'] != HERE.name or tuple(cfg['domains']) != DOMAINS
            or cfg['rounds_per_domain'] != ROUNDS or cfg['initial_active_snapshot'] != []
            or cfg['stop_after_round'] != 1 or cfg['run_test'] is not False
            or cfg['prior_test_exposure'] is not True or cfg['flash_concurrency'] != 8
            or cfg['solver_workers_per_domain'] != 50 or cfg['max_proposer_turns'] != 32
            or cfg['max_context_tokens'] != 32768 or cfg['flash_max_tokens'] != 8192
            or cfg['flash_context_limit'] is not None
            or cfg['optimizer_profile'] != 'life-h2345-diagnose-replay-v2.3'):
        raise RuntimeError('Experiment protocol changed')
    from meta.life_optimizer import PROFILE
    from meta.life_optimizer.context import CONTEXT_LIMIT, OUTPUT_LIMIT
    if PROFILE != cfg['optimizer_profile'] or CONTEXT_LIMIT is not None or OUTPUT_LIMIT != 8192:
        raise RuntimeError('Flash must use restored settings, not the Qwen token limits')
    manifest = HISTORICAL_ROOT / 'manifest.json'
    if digest(manifest) != cfg['historical_manifest_sha256']:
        raise RuntimeError('Historical manifest changed')
    source = json.loads(manifest.read_text())
    if (object_digest(source['snapshot_hashes']) != cfg['snapshot_hashes_sha256']
            or object_digest(source['data_hashes']) != cfg['data_hashes_sha256']
            or source['agent'] != cfg['solver']):
        raise RuntimeError('Frozen benchmark configuration changed')
    for name, sha in cfg['interface_source_sha256'].items():
        if digest(ROOT / 'meta/agentbench_hooks' / name) != sha:
            raise RuntimeError(f'Interface source changed: {name}')
    for domain in DOMAINS:
        spec = source['domains'][domain]
        if len(spec['batches']) != 6:
            raise RuntimeError('Six frozen batches required')
        for batch in range(6):
            ids = task_ids(spec, batch, 'train')
            if len(ids) != 50 or len(set(ids)) != 50:
                raise RuntimeError('Complete unique 50-task batches required')
    if check_files:
        predecessor = HERE.parent / cfg['predecessor_run']
        for domain, row in cfg['reused_baselines'].items():
            base = predecessor / domain / 'evals/round_01_incumbent'
            if (digest(base / 'summary.json') != row['summary_sha256']
                    or digest(predecessor / domain / 'run_config.json') != row['run_config_sha256']):
                raise RuntimeError('Baseline provenance changed')
            for name, sha in row['episode_hashes'].items():
                if digest(base / name) != sha:
                    raise RuntimeError('Reused baseline episode changed')
        for name, sha in source['snapshot_hashes'].items():
            if digest(HISTORICAL_ROOT / 'task_snapshot' / name) != sha:
                raise RuntimeError(f'Frozen task source changed: {name}')
        for name, sha in source['data_hashes'].items():
            if name.startswith('data/alfworld'):
                continue  # Neither ALFWorld evaluation nor optimization is in scope.
            if digest(ROOT / 'AgentBench' / name) != sha:
                raise RuntimeError(f'Frozen benchmark data changed: {name}')
    return source
