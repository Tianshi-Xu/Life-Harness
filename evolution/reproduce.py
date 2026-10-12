"""Prepare/run identity-starting Life, Meta and AHE; collect accuracy/token records."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
AHE = ROOT / 'integrations/ahe-native'
TAU_PYTHON = ROOT / 'TauBench/.venv/bin/python'
AHE_PYTHON = AHE / '.venv/bin/python'
DOMAINS = ('alfworld', 'dbbench', 'webshop')
RECORD_NAMES = {'state.json', 'summary.json', 'harness_summary.json', 'status.json',
    'TRAINING_RESULTS.json', 'FINAL_RESULTS.json', 'TOKEN_USAGE.json', 'OPTIMIZER_TOKEN_USAGE.json',
    'MR_TOKEN_USAGE.json', 'ALIYUN_TOKEN_USAGE.json', 'training_seal.json',
    'iteration_scores.yaml', 'final_candidate_train.json', 'record.json',
    'final_results.json', 'final_train.json', 'result.json', 'protocol.json',
    'training_schedule.json', 'test_spec.json', 'progress.json', 'errors.json',
    'flash_metrics.json'}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def make_plan(name, suite, method):
    """Commands execute sequentially; each method keeps its original selection."""
    if not re.fullmatch(r'[a-z][a-z0-9_]+', name):
        raise ValueError('safe fresh run name required')
    if suite == 'tau' and method not in {'all', 'life', 'meta-ahe'}:
        raise ValueError('Tau supports life, meta-ahe (the paired controller), or all')
    if suite == 'os':
        raise ValueError('OS requires its dedicated Docker controller; use evolution.os_comparison.protocol/launch')
    steps = []
    def add(label, command, sources):
        steps.append({'method': label, 'command': command, 'record_roots': [str(p) for p in sources]})
    if suite == 'tau':
        if method in {'all', 'life'}:
            child = name + '_life_tau'
            run = ROOT / 'evolution/experiments' / child
            add('life', [str(TAU_PYTHON), '-m', 'evolution.life_tau_run', '--name', child], [run])
            add('life', [str(TAU_PYTHON), '-m', 'evolution.life_tau_run', '--name', child, '--run'], [run])
        if method in {'all', 'meta-ahe'}:
            child = name + '_meta_ahe_tau'
            add('meta+ahe', [str(AHE_PYTHON), '-m', 'evolution.tau_comparison.run', '--name', child],
                [ROOT / 'evolution/experiments' / child])
    else:
        if method in {'all', 'life'}:
            for domain in DOMAINS:
                child = name + '_life_' + domain
                run = ROOT / 'evolution/experiments' / child
                command = [str(TAU_PYTHON), '-m', 'evolution.life_agentbench_run', '--domain', domain, '--name', child]
                add('life', command, [run])
                add('life', command + ['--run'], [run])
        if method in {'all', 'meta'}:
            child = name + '_meta_agentbench'
            run = ROOT / 'evolution/experiments' / child
            add('meta', [str(TAU_PYTHON), '-m', 'evolution.reproduction.meta_transport.launch', '--name', child], [run])
            add('meta', [str(AHE_PYTHON), str(run / 'runner.py')], [run])
        if method in {'all', 'ahe'}:
            child = name + '_ahe_agentbench'
            sources = [AHE / 'batches' / child] + [AHE / 'experiments' / (child + '_' + d) for d in DOMAINS]
            add('ahe', [str(AHE_PYTHON), '-m', 'ahe_native.batch', '--name', child,
                '--iterations', '6', '--training-tasks', '50', '--debugger-tasks', '10',
                '--measure-final-candidate'], sources)
            add('ahe', [str(AHE_PYTHON), '-m', 'ahe_native.final_test', '--batch', child,
                '--name', child + '_test'], [AHE / 'heldout_tests' / (child + '_test')])
    if not steps:
        raise ValueError('unsupported method/suite combination')
    return {'name': name, 'suite': suite, 'method': method, 'initialization': 'identity; no imported evolved strategies',
        'execution': 'sequential methods; each controller uses its own global Flash gate',
        'comparison': 'six rounds; native splits and original method selection; not equal-token-budget',
        'steps': steps}


def collect(plan):
    """Copy genuine summaries losslessly; missing metrics stay missing."""
    runs = {}
    for step in plan['steps']:
        for directory in step['record_roots']:
            path = Path(directory)
            key = step['method'] + ':' + directory
            if key in runs:
                continue
            artifacts, ledgers = {}, []
            for base, folders, files in os.walk(path):
                folders[:] = [d for d in folders if d not in
                    {'workspace', 'task_snapshot', 'upstream_reference', 'optimizer_inputs',
                     'optimizer_sessions', 'runtime_rootfs', '__pycache__', 'traces', 'input'}]
                for filename in files:
                    source = Path(base) / filename
                    relative = str(source.relative_to(path))
                    if filename.endswith('.jsonl') and ('requests' in filename or 'ledger' in filename):
                        ledgers.append(relative)
                    if filename not in RECORD_NAMES and not filename.endswith('_final_candidate_train.json'):
                        continue
                    try:
                        body = json.loads(source.read_text()) if source.suffix == '.json' else source.read_text()
                    except (OSError, ValueError) as exc:
                        artifacts[relative] = {'read_error': str(exc)}
                    else:
                        artifacts[relative] = {'sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'record': body}
            runs[key] = {'method': step['method'], 'root': directory, 'exists': path.is_dir(),
                'artifacts': artifacts, 'raw_request_ledgers': ledgers}
    return {'schema_version': 1, 'name': plan['name'], 'updated_at': time.time(),
        'scope': 'native accuracy summaries, complete round states and measured API usage; raw ledgers retained in runs',
        'cost_roles': 'optimizer input/output/cache/unknown usage; solver/simulator/judge remain separately labeled',
        'missing_usage': 'unknown; never filled with zero or estimated as provider usage',
        'comparison': plan['comparison'], 'runs': runs}


def run(root, plan):
    with (root / '.runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = json.loads((root / 'status.json').read_text())
        if status['phase'] != 'prepared':
            raise RuntimeError('started/failed paid runs require audit; no automatic replay')
        env = os.environ.copy()
        env['PYTHONPATH'] = os.pathsep.join((str(ROOT), str(AHE), str(ROOT / 'TauBench/src')))
        try:
            for number, step in enumerate(plan['steps'], 1):
                save(root / 'status.json', {'phase': 'running', 'step': number, 'method': step['method'], 'pid': os.getpid()})
                with (root / f'step_{number:02d}.log').open('a') as log:
                    subprocess.run(step['command'], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                save(root / 'REPRODUCTION_RECORDS.json', collect(plan))
            save(root / 'status.json', {'phase': 'complete', 'steps': len(plan['steps']), 'pid': os.getpid()})
        except Exception:
            save(root / 'status.json', {'phase': 'stopped', 'step': number, 'method': step['method'], 'pid': os.getpid()})
            save(root / 'REPRODUCTION_RECORDS.json', collect(plan))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--suite', choices=('tau', 'agentbench'), default='tau')
    parser.add_argument('--method', choices=('all', 'life', 'meta', 'ahe', 'meta-ahe'), default='all')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--launch', action='store_true', help='starts paid native runs sequentially')
    modes.add_argument('--run', action='store_true', help=argparse.SUPPRESS)
    modes.add_argument('--report', action='store_true', help='offline; collect existing accuracy/token summaries')
    args = parser.parse_args()
    proposed = make_plan(args.name, args.suite, args.method)
    root = ROOT / 'evolution/experiments' / args.name
    file = root / 'reproduction.json'
    if file.exists():
        plan = json.loads(file.read_text())
        if plan != proposed:
            raise ValueError('run protocol changed; use a new name')
    else:
        if args.run or args.report or root.exists():
            raise ValueError('prepare a fresh reproduction plan first')
        root.mkdir(parents=True)
        plan = proposed
        save(file, plan)
        save(root / 'status.json', {'phase': 'prepared'})
    if args.report:
        save(root / 'REPRODUCTION_RECORDS.json', collect(plan))
    elif args.run:
        run(root, plan)
    elif args.launch:
        if (root / 'launch.json').exists() or json.loads((root / 'status.json').read_text())['phase'] != 'prepared':
            raise RuntimeError('existing launcher/paid run cannot be replayed')
        with (root / 'runner.log').open('a') as log:
            process = subprocess.Popen([sys.executable, '-m', 'evolution.reproduce', '--name', args.name,
                '--suite', args.suite, '--method', args.method, '--run'], cwd=ROOT,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        save(root / 'launch.json', {'pid': process.pid, 'started_at': time.time()})
    print(root)


if __name__ == '__main__':
    main()
