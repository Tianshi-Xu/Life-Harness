"""Generate a new sealed-input experiment; never modify the historical run."""
import copy
import json
import shutil
import sys
from pathlib import Path

from common import AB, DOMAINS, HERE, OLD, ROOT, digest, save, verify_frozen


def main():
    if (HERE / 'manifest.json').exists():
        if sys.argv[1:] in (['--refresh-prelaunch'], ['--repair-prepaid']):
            if sys.argv[1:] == ['--repair-prepaid']:
                if ((HERE / 'process.json').exists() or any(HERE.glob('*/flash_requests.jsonl'))
                        or any(HERE.glob('*/state.json'))
                        or any(p.name != 'setup_smoke' for p in HERE.glob('*/evals/*'))):
                    raise RuntimeError('Only a pre-optimizer setup repair is allowed')
                if not (HERE / 'setup_serialization_repair_backup/frozen.json').exists():
                    raise RuntimeError('Preserve the failed setup source and records before repair')
                launch = json.loads((HERE / 'launch.json').read_text())
                if Path('/proc/' + str(launch['pid']) + '/stat').exists():
                    raise RuntimeError('Cannot change a running experiment')
            elif ((HERE / 'launch.json').exists() or (HERE / 'status.json').exists()
                  or any(HERE.glob('*/flash_requests.jsonl')) or any(HERE.glob('*/evals/*'))):
                raise RuntimeError('Cannot refresh a started experiment')
            previous = HERE / 'prelaunch_original_frozen.json'
            if not previous.exists():
                shutil.copy2(HERE / 'frozen.json', previous)
            files = [*HERE.glob('*.py'), HERE / 'README.md', HERE / 'manifest.json']
            save(HERE / 'frozen.json', {str(p.relative_to(HERE)): digest(p) for p in files})
        verify_frozen(data=True)
        print('Already prepared; immutable input hashes verified')
        return
    original = json.loads((OLD / 'manifest.json').read_text())
    for rel, sha in original['upstream']['files'].items():
        if digest(OLD / 'upstream_reference' / rel) != sha:
            raise RuntimeError('Original Meta source mismatch')
    cfg = copy.deepcopy(original)
    cfg.update(name=HERE.name, domains={d: original['domains'][d] for d in DOMAINS},
        method='original frozen Meta run_evolve and original skill; OpenAI-compatible API tool-shell port',
        teacher={'model': 'deepseek-v4-flash', 'host': 'https://routify-pub.alibaba-inc.com',
                 'credentials': 'resolved in host broker memory only'},
        concurrency=50, flash_concurrency=8, flash_max_tokens=60000, flash_context_limit=None,
        cumulative_optimizer_token_limit=None, flash_min_interval_seconds=1.25,
        max_context_tokens=32768, propose_timeout_seconds=5400,
        test_policy='full fixed final TEST only after all three six-round TRAIN states are sealed',
        first_round_gate='all domains must generate and score a candidate with complete optimizer usage before round 2',
        episode_error_policy='cache individual errors once as zero in fixed denominator; no silent retry',
        comparison_label='six-round iteration-budget experiment with measured tokens; NOT equal-token-budget',
        prior_test_exposure=True, task_batch_provenance='reuse frozen public training partitions and batch IDs',
        optimizer_backend='API execution-shell replacement, not a byte-for-byte official CLI reproduction')
    source = HERE.parent / 'meta_agentbench_six_flash_tokens_20261004'
    startup = json.loads((source / 'flash_metrics.json').read_text())
    if startup['gate']['active'] or not startup['usage']['accounting_complete']:
        raise RuntimeError('Startup usage must be completely recorded before repair')
    if Path('/proc/' + str(json.loads((source / 'launch.json').read_text())['pid']) + '/stat').exists():
        raise RuntimeError('Startup experiment must be stopped')
    cfg.update(reused_baseline_source=str(source), startup_excluded_source=str(source),
        startup_excluded_usage=startup['usage'],
        startup_exclusion_reason='batch-wide transient local Qwen HTTP 500; API port omitted original three request retries; no scored/accepted candidate',
        solver_request_retries=3, max_batch_infra_errors_before_pause=2,
        reused_baseline_episode_hashes={})
    for d in DOMAINS:
        cfg['reused_baseline_episode_hashes'][d] = {
            p.name: digest(p) for p in (source / d / 'evals/batch_00_baseline').glob('episode_*.json')
            if not json.loads(p.read_text())['infrastructure_error']}
    for d in DOMAINS:
        assert all(len(b['indices']) == len(set(b['indices'])) == 50 for b in cfg['domains'][d]['batches'])
    shutil.copytree(OLD / 'upstream_reference', HERE / 'upstream_reference',
                    ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(OLD / 'task_snapshot', HERE / 'task_snapshot',
                    ignore=shutil.ignore_patterns('__pycache__'))
    preserved = {}
    for path in source.rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts and not path.name.startswith('.'):
            preserved[str(path)] = digest(path)
    # Raw histories and all old candidate bytes are protected, not just a final score.
    for path in OLD.rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts and not path.name.startswith('.'):
            preserved[str(path)] = digest(path)
    life = HERE.parent / 'life_on_demand_one_db_web_flash60000_20261003'
    for path in [life / 'training_seal.json', *life.glob('*/state.json')]:
        if path.is_file():
            preserved[str(path)] = digest(path)
    save(HERE / 'historical_preservation.json', preserved)
    skill = HERE / 'upstream_reference/.claude/skills/meta-harness-terminal-bench-2/SKILL.md'
    for d in DOMAINS:
        work = HERE / d / 'workspace'
        (work / 'agents').mkdir(parents=True)
        (work / 'source_view').mkdir()
        shutil.copy2(OLD / d / 'workspace/agents/baseline_zero.py', work / 'agents/baseline_zero.py')
        shutil.copy2(skill, work / 'upstream_SKILL.md')
        (work / 'source_view/CONTRACT.md').write_text(
            'AgentHarness(public_context=None) exports synchronous transform_request(request_dict) '
            'and transform_response(assistant_dict, public_history). Both return ordinary dictionaries. '
            'The host constructs one harness per episode. Only public task descriptions, table metadata, '
            'tool schemas and public interaction history are available. No new runtime LLM/tool, '
            'network/filesystem/subprocess operation or cross-episode state. Released H2-H5 are disabled. '
            'Solver settings are frozen and cannot be modified. The optimizer may use python for pure '
            'import/prototype checks but must not execute benchmarks.\n')
    save(HERE / 'manifest.json', cfg)
    files = [*HERE.glob('*.py'), HERE / 'README.md', HERE / 'manifest.json']
    save(HERE / 'frozen.json', {str(p.relative_to(HERE)): digest(p) for p in files})
    verify_frozen(data=True)
    print(json.dumps({'prepared': HERE.name, 'domains': DOMAINS, 'rounds': 6,
                      'historical_files_protected': len(preserved)}))


if __name__ == '__main__':
    main()
