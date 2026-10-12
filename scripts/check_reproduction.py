"""Offline audit of all frozen benchmark and original Meta source inputs."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evolution.life_evolve.profile import source_hashes


def check(directory, hashes):
    missing_runtime = []
    for relative, expected in hashes.items():
        path = directory / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Missing or changed frozen input: {path}')


def main():
    reference = ROOT / 'evolution/reproduction/reference'
    manifest = json.loads((reference / 'manifest.json').read_text())
    check(reference / 'task_snapshot', manifest['snapshot_hashes'])
    missing_runtime = []
    for relative, expected in manifest['data_hashes'].items():
        path = ROOT / 'AgentBench' / relative
        if path.is_file():
            check(ROOT / 'AgentBench', {relative: expected})
            continue
        prefix = 'data/alfworld/'
        if relative.startswith('.native/webshop-src/data/'):
            missing_runtime.append(relative)
            continue
        if not relative.startswith(prefix + 'json_2.1.1/'):
            raise RuntimeError(f'Missing benchmark data: {path}')
        member = relative[len(prefix):]
        found = False
        for archive in (ROOT / 'AgentBench/data/alfworld').glob('json_2.1.1_*.zip'):
            with zipfile.ZipFile(archive) as source:
                if member in source.namelist():
                    if hashlib.sha256(source.read(member)).hexdigest() != expected:
                        raise RuntimeError(f'Changed archived game: {member}')
                    found = True
                    break
        if not found:
            raise RuntimeError(f'Missing archived game: {member}')
    check(reference / 'upstream_reference', manifest['upstream']['files'])
    check(ROOT, source_hashes())
    print('Frozen task/archived-data/Meta hashes and current Life source inputs verified.')
    if missing_runtime:
        print('WebShop runtime data not installed (run its native setup): ' + ', '.join(missing_runtime))


if __name__ == '__main__':
    main()
