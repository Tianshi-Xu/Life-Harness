"""Prepare or launch fresh original-Meta AgentBench runs with full API ledgers."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__).resolve().parent
REFERENCE = ROOT / 'evolution/reproduction/reference'
DOMAINS = ('alfworld', 'dbbench', 'webshop')
FILES = ('api_agent.py', 'api_broker.py', 'common.py', 'driver.py', 'evaluator.py',
         'isolation.py', 'runner.py', 'tokenizer_worker.py', 'usage.py')


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')


def prepare(name):
    if not re.fullmatch(r'[a-z][a-z0-9_]+', name):
        raise ValueError('safe fresh run name required')
    destination = ROOT / 'evolution/experiments' / name
    if destination.exists():
        raise FileExistsError('new run required; never overwrite or replay paid work')
    cfg = json.loads((REFERENCE / 'manifest.json').read_text())
    cfg = copy.deepcopy(cfg)
    cfg.update(name=name, domains={d: cfg['domains'][d] for d in DOMAINS},
        initialization='identity; no imported evolved policy', iterations=6,
        teacher={'model': 'deepseek-v4-flash',
                 'host': os.environ.get('MR_HOST', 'https://api.example.com').rstrip('/')},
        flash_concurrency=8, flash_max_tokens=60000, flash_context_limit=None,
        cumulative_optimizer_token_limit=None, flash_min_interval_seconds=1.25,
        max_context_tokens=32768, propose_timeout_seconds=5400,
        solver_request_retries=3, max_batch_infra_errors_before_pause=2,
        test_policy='full fixed final TEST only after all six-round TRAIN states are sealed',
        first_round_gate='every domain has a measured valid candidate and complete optimizer usage',
        episode_error_policy='preserve infrastructure errors once as zero in the fixed denominator',
        comparison_label='six rounds with measured tokens; different methods may have different token budgets',
        optimizer_backend='original Meta run_evolve; metered OpenAI-compatible API coding shell')
    destination.mkdir(parents=True)
    for filename in FILES:
        shutil.copy2(SOURCE / filename, destination / filename)
    for folder in ('upstream_reference', 'task_snapshot'):
        shutil.copytree(REFERENCE / folder, destination / folder,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    skill = REFERENCE / 'upstream_reference/.claude/skills/meta-harness-terminal-bench-2/SKILL.md'
    for domain in DOMAINS:
        work = destination / domain / 'workspace'
        (work / 'agents').mkdir(parents=True)
        (work / 'source_view').mkdir()
        shutil.copy2(REFERENCE / domain / 'workspace/agents/baseline_zero.py', work / 'agents/baseline_zero.py')
        shutil.copy2(skill, work / 'upstream_SKILL.md')
        (work / 'source_view/CONTRACT.md').write_text(
            'Export AgentHarness(public_context=None) with transform_request and transform_response. '
            'Each episode gets a new instance. Only public native tasks, tools and trajectories are available. '
            'No new runtime model/tool/network/filesystem call, hidden task answer or shared mutable state. '
            'All released H2-H5 flags are disabled. Solver and native scoring budgets stay fixed.\n')
    save(destination / 'manifest.json', cfg)
    inputs = [destination / n for n in FILES] + [destination / 'manifest.json']
    save(destination / 'frozen.json', {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--launch', action='store_true', help='starts paid optimization after native smoke checks')
    args = parser.parse_args()
    directory = prepare(args.name)
    if args.launch:
        subprocess.run([sys.executable, str(directory / 'runner.py'), '--launch'], cwd=ROOT, check=True)
    else:
        print(directory)
        print('Run after native setup: ' + sys.executable + ' ' + str(directory / 'runner.py'))


if __name__ == '__main__':
    main()
