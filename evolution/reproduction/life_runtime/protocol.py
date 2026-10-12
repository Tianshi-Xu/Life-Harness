"""Content-addressed helpers for immutable candidate composition."""
import hashlib
import json
from pathlib import Path

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')
    temporary.replace(path)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
HISTORICAL_ROOT = HERE.parent / 'reference'
DOMAINS = ('alfworld', 'dbbench', 'webshop')
ROUNDS = 6
MANIFEST_SHA256 = '1d60aa4e2d905a393d7dea0f81da02d9d8ec58a48fc7b8070532b574ceccd96a'

def task_ids(spec, batch, split):
    if split == 'train':
        row = spec['batches'][batch]
        return list(row.get('task_ids', row.get('indices', [])))
    return list(spec.get('test_task_ids', spec.get('test_indices', [])))

def validate_protocol(check_files=True):
    path = HISTORICAL_ROOT / 'manifest.json'
    if digest(path) != MANIFEST_SHA256:
        raise RuntimeError('Frozen benchmark manifest changed')
    source = json.loads(path.read_text())
    if check_files:
        for name, expected in source['snapshot_hashes'].items():
            if digest(HISTORICAL_ROOT / 'task_snapshot' / name) != expected:
                raise RuntimeError('Frozen task source changed: ' + name)
        for name, expected in source['data_hashes'].items():
            if digest(ROOT / 'AgentBench' / name) != expected:
                raise RuntimeError('Frozen benchmark data missing/changed: ' + name)
    return source
