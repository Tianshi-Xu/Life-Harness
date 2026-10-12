"""Credential-free chroot for the API coding shell and its pure Bash checks."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from common import HERE, ROOT, digest, save


def isolated_proposal(domain, iteration, workspace, prompt, contract, broker_url, pending):
    sys.path.insert(0, str(ROOT / 'integrations/ahe-native'))
    from ahe_native.isolation import case_root, copy_unchanged_or_atomic, enter, require_regular_workspace
    trace = HERE / domain / f'proposer_{iteration:02d}'
    trace.mkdir(parents=True, exist_ok=True)
    root, case, logical, uid = case_root(workspace, trace)
    worker = root / 'adapter/meta_flash_tokens/api_agent.py'
    worker.parent.mkdir(parents=True, exist_ok=True)
    copy_unchanged_or_atomic(HERE / 'api_agent.py', worker)
    logical_work = logical + '/workspace'
    public = case / 'workspace'
    frozen = {str(p.relative_to(workspace)): digest(p) for p in workspace.rglob('*')
              if p.is_file() and '__pycache__' not in p.parts}
    save(trace / 'frozen_inputs.json', frozen)
    for path in [public, *public.rglob('*')]:
        if path.is_dir():
            rel = path.relative_to(public)
            if not rel.parts or rel.parts[0] == 'agents' or (
                    rel.parts[0] == 'logs' and len(rel.parts) > 1 and rel.parts[1] == f'round_{iteration:02d}'):
                os.chown(path, uid, uid)
                path.chmod(0o700)
    request = {'domain': domain, 'round': iteration, 'isolated_uid': uid,
        'workspace': logical_work, 'broker_url': broker_url, 'timeout_seconds': 5400,
        'prompt': prompt.replace(str(workspace), logical_work),
        'port_contract': contract.replace(str(workspace), logical_work)}
    save(case / 'request.json', request)
    (case / 'request.json').chmod(0o444)
    save(trace / 'transport.json', {'isolated_uid': uid, 'case_directory': str(case),
        'workspace': logical_work, 'credential_policy': 'no real key in chroot/env/config',
        'worker_sha256': digest(HERE / 'api_agent.py')})
    env = {'PATH': '/runtime/bin:/usr/bin:/bin', 'PYTHONPATH': '/adapter/meta_flash_tokens',
           'PYTHONDONTWRITEBYTECODE': '1', 'TMPDIR': '/tmp', 'NO_PROXY': '127.0.0.1,localhost'}
    with (trace / 'process.log').open('a') as log:
        proc = subprocess.run(['/runtime/bin/python', '-u', str('/adapter/meta_flash_tokens/api_agent.py'),
            '--request', logical + '/request.json', '--output', logical + '/output'],
            preexec_fn=lambda: enter(str(root), logical_work, uid), env=env,
            stdout=log, stderr=subprocess.STDOUT, timeout=5520)
    require_regular_workspace(public)
    shutil.copytree(case / 'output', trace / 'traces', dirs_exist_ok=True)
    changed = [name for name, sha in frozen.items()
               if not (public / name).is_file() or digest(public / name) != sha]
    if changed:
        save(trace / 'rejection.json', {'reason': 'frozen input mutation', 'paths': changed})
        return False
    result_path = case / 'output/result.json'
    result = json.loads(result_path.read_text()) if result_path.exists() else {'success': False, 'error': 'no result'}
    if proc.returncode or not result.get('success'):
        save(trace / 'rejection.json', {'reason': 'API proposer failed', 'result': result})
        return False
    parent = json.loads((case / 'output/parent/session.json').read_text())
    if parent['delegations'] != ['analysis', 'implementation']:
        raise RuntimeError('Original Meta two-Agent workflow missing')
    staged_pending = public / pending.relative_to(workspace)
    if not staged_pending.exists():
        save(trace / 'rejection.json', {'reason': 'missing pending_eval.json'})
        return False
    document = json.loads(staged_pending.read_text())
    if len(document.get('candidates', [])) != 1:
        raise ValueError('The original Meta skill requires exactly one candidate')
    candidate = document['candidates'][0]
    import re
    name = candidate['name']
    if (not re.fullmatch('[a-z][a-z0-9_]*', name)
            or candidate['import_path'] != f'agents.{name}:AgentHarness'
            or f'agents/{name}.py' in frozen):
        raise ValueError('Invalid or existing candidate path')
    source = public / 'agents' / (name + '.py')
    if not source.is_file():
        raise ValueError('Candidate file is missing')
    shutil.copy2(source, workspace / 'agents' / source.name)
    shutil.copy2(staged_pending, pending)
    save(trace / 'receipt.json', {'success': True, 'delegations': parent['delegations'],
        'candidate_sha256': digest(source), 'pending_sha256': digest(staged_pending),
        'frozen_input_count': len(frozen)})
    return True
