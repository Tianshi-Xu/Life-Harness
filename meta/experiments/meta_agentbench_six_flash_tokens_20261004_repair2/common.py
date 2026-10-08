"""Infrastructure for the separately recorded, identity-starting Meta rerun."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AB = ROOT / 'AgentBench'
OLD = HERE.parent / 'meta_from_scratch_six_20260915'
DOMAINS = ('alfworld', 'dbbench', 'webshop')
ROUNDS = 6


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def json_default(value):
    dump = getattr(value, 'model_dump', None)
    if callable(dump):
        return dump(mode='json')
    if isinstance(value, SimpleNamespace):
        return vars(value)
    # Native rewards can contain scalar numpy values; preserve their value.
    if type(value).__module__.startswith('numpy') and callable(getattr(value, 'item', None)):
        return value.item()
    return str(value)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, default=json_default) + '\n')
    tmp.replace(path)


def manifest():
    return json.loads((HERE / 'manifest.json').read_text())


def verify_frozen(data=False):
    cfg = manifest()
    if (tuple(cfg['domains']) != DOMAINS or cfg['iterations'] != 6
            or cfg['teacher']['model'] != 'deepseek-v4-flash'
            or cfg['flash_concurrency'] != 8 or cfg['flash_max_tokens'] != 60000
            or cfg['flash_context_limit'] is not None
            or cfg['cumulative_optimizer_token_limit'] is not None
            or cfg['agent']['max_tokens'] != 4096 or cfg['max_context_tokens'] != 32768):
        raise RuntimeError('Frozen experiment configuration changed')
    for name, sha in json.loads((HERE / 'frozen.json').read_text()).items():
        if digest(HERE / name) != sha:
            raise RuntimeError('Experiment source changed: ' + name)
    for name, sha in cfg['upstream']['files'].items():
        if digest(HERE / 'upstream_reference' / name) != sha:
            raise RuntimeError('Original Meta source changed: ' + name)
    for name, sha in cfg['snapshot_hashes'].items():
        if digest(HERE / 'task_snapshot' / name) != sha:
            raise RuntimeError('Frozen task source changed: ' + name)
    if data:
        for name, sha in cfg['data_hashes'].items():
            if digest(AB / name) != sha:
                raise RuntimeError('Frozen benchmark data changed: ' + name)
    return cfg


def preservation_check():
    expected = json.loads((HERE / 'historical_preservation.json').read_text())
    changed = [p for p, sha in expected.items() if not Path(p).is_file() or digest(p) != sha]
    save(HERE / 'historical_preservation_check.json', {
        'checked_files': len(expected), 'changed': changed, 'unchanged': not changed})
    if changed:
        raise RuntimeError('Historical record changed; stop without overwriting it')


def progress(domain, **values):
    save(HERE / domain / 'progress.json', values)
