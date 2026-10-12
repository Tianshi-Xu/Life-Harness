import os
"""Launch the trusted host controller in a dedicated Docker experiment container."""
import argparse
import json
from pathlib import Path
import subprocess
import urllib.request

from evolution.tau_comparison.common import AHE_NATIVE, ROOT, read
from .protocol import save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    if root.parent != ROOT / 'evolution/experiments' or read(root / 'status.json')['phase'] != 'prepared':
        raise ValueError('fresh prepared local experiment required')
    if (root / 'launch.json').exists():
        raise FileExistsError('audit an earlier container before another launch')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open('http://127.0.0.1:8400/health', timeout=30) as response:
        health = json.load(response)
    if len(health) != 10 or any(status != 200 for status in health.values()):
        raise RuntimeError('all ten Qwen backends must be healthy')
    name = 'life-harness-' + root.name.replace('_', '-')
    command = ['docker', 'run', '-d', '--name', name, '--network', 'host',
               '--cap-add', 'SYS_CHROOT', '--cap-add', 'MKNOD',
               '--mount', f'type=bind,source={ROOT.parent},target={ROOT.parent}',
               '--mount', f'type=bind,source={Path(os.environ["LIFE_SOLVER_TOKENIZER"]).resolve()},target={Path(os.environ["LIFE_SOLVER_TOKENIZER"]).resolve()},readonly',
               '--mount', 'type=bind,source=/var/run/docker.sock,target=/var/run/docker.sock',
               '-w', str(ROOT), '-e', f'PYTHONPATH={ROOT}:{AHE_NATIVE}',
               '-e', f'LIFE_SOLVER_TOKENIZER={Path(os.environ["LIFE_SOLVER_TOKENIZER"]).resolve()}',
               '-e', 'PYTHONDONTWRITEBYTECODE=1', '-e', 'NO_PROXY=127.0.0.1,localhost',
               '-e', 'GIT_CONFIG_COUNT=1', '-e', 'GIT_CONFIG_KEY_0=safe.directory',
               '-e', f'GIT_CONFIG_VALUE_0={ROOT / "integrations/agentic-harness-engineering"}',
               'local-os/default', str(AHE_NATIVE / '.venv/bin/python'), '-u', '-m',
               'evolution.os_comparison.run', '--root', str(root)]
    if args.smoke_only:
        command.append('--smoke-only')
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    save(root / 'launch.json', {'container': name, 'container_id': result.stdout.strip(),
         'smoke_only': args.smoke_only, 'optimizer_credentials': 'read locally by trusted broker only'})
    print(json.dumps({'container': name, 'root': str(root)}))


if __name__ == '__main__':
    main()
