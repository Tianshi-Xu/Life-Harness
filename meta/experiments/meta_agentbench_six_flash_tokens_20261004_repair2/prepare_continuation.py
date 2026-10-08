"""Freeze a copied continuation without modifying or rerunning its predecessor."""
import copy
import json
import shutil
from pathlib import Path

from common import DOMAINS, HERE, digest, save, verify_frozen


def main():
    receipt_path = HERE / 'continuation_receipt.json'
    if receipt_path.exists():
        verify_frozen(data=True)
        print('Continuation already prepared and verified')
        return
    source = HERE.with_name('meta_agentbench_six_flash_tokens_20261004_repair1')
    pid = json.loads((source / 'launch.json').read_text())['pid']
    if Path(f'/proc/{pid}/stat').exists():
        raise RuntimeError('Predecessor must be stopped before making a continuation')
    metrics = json.loads((source / 'flash_metrics.json').read_text())
    if metrics['gate']['active'] or not metrics['usage']['accounting_complete']:
        raise RuntimeError('All predecessor API receipts must be complete')
    backup = HERE / 'continuation_backup'
    backup.mkdir()
    for name in ('frozen.json', 'manifest.json', 'historical_preservation.json',
                 'status.json', 'first_round_gate.json', 'launch.json', 'process.json'):
        shutil.copy2(HERE / name, backup / name)
    before = json.loads((HERE / 'webshop/state.json').read_text())
    if (len(before['rounds']) != 1 or before['rounds'][0]['candidate_successes'] is not None
            or before['rounds'][0]['candidate']['name'] != 'option_first_buy'):
        raise RuntimeError('Unexpected WebShop recovery point')
    shutil.copy2(HERE / 'webshop/state.json', backup / 'webshop_state.json')
    shutil.copy2(HERE / 'webshop/preflight_01.json', backup / 'webshop_preflight_01.json')
    # The failed import was an integration error, not a second optimization
    # attempt. Archive its frontier/summary in this NEW copy; the immutable
    # predecessor continues to hold the original unmeasured round verbatim.
    shutil.move(str(HERE / 'webshop/workspace/logs/round_01'),
                str(backup / 'webshop_failed_round_01'))
    after = copy.deepcopy(before)
    after.update(batch=0, baseline_phase='batch_00_baseline', rounds=[])
    save(HERE / 'webshop/state.json', after)
    cfg = json.loads((source / 'manifest.json').read_text())
    cfg['name'] = HERE.name
    # Only the run identity changes. Model, prompt, budgets, task IDs, sources,
    # acceptance, retries and the already-spent token ledger are unchanged.
    save(HERE / 'manifest.json', cfg)
    protected = json.loads((source / 'historical_preservation.json').read_text())
    for path in source.rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts and not path.name.startswith('.'):
            protected[str(path)] = digest(path)
    save(HERE / 'historical_preservation.json', protected)
    imported = {}
    for domain in DOMAINS:
        for path in (source / domain).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:
                imported[str(path.relative_to(source))] = digest(path)
    save(receipt_path, {
        'source': str(source), 'source_manifest_sha256': digest(source / 'manifest.json'),
        'source_frozen_sha256': digest(source / 'frozen.json'),
        'reason': 'comment/docstring substring requests falsely treated as a forbidden import',
        'protocol_changes': ['run identity only; inert prose is excluded from source substring precheck'],
        'retained': 'all 150 baselines, two scored candidates, three pending proposals, all 238 Flash calls',
        'additional_paid_calls_for_webshop_round_1_recovery': 0,
        'copied_receipt_policy': 'count imported receipts once in cumulative optimization usage',
        'source_optimizer_usage': metrics['usage'], 'imported_file_hashes': imported,
        'incomplete_round_source_sha256': digest(source / 'webshop/state.json'),
        'candidate_sha256': digest(source / 'webshop/workspace/agents/option_first_buy.py'),
    })
    files = [*HERE.glob('*.py'), HERE / 'README.md', HERE / 'manifest.json', receipt_path]
    save(HERE / 'frozen.json', {str(p.relative_to(HERE)): digest(p) for p in files})
    verify_frozen(data=True)
    print(json.dumps({'prepared': HERE.name, 'predecessor_calls_reused': metrics['usage']['successful_calls'],
                      'predecessor_tokens_reused': metrics['usage']['total_tokens']}))


if __name__ == '__main__':
    main()
