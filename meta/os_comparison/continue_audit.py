"""Audit a second continuation preserving all three first-round paid outcomes."""
import argparse
from pathlib import Path
import sys
from meta.tau_comparison.common import AHE_NATIVE,ROOT,read,transport_modules
from .protocol import digest,save,verify_sources


def audit(root):
    sys.path.insert(0,str(AHE_NATIVE))
    from ahe_native.backend import workspace_hash
    from ahe_native.token_accounting import TokenLedger
    from ahe_native.upstream import source_fingerprint
    if read(root/'status.json')['phase']!='stopped_on_error' or (root/'training_seal.json').exists():
        raise RuntimeError('fully stopped TRAIN-only ancestor required')
    target=root/'recovery/continuation_02.json'
    if target.exists():raise FileExistsError('continuation already audited')
    previous=root/'recovery/receipt.json';old=read(previous)
    changed={p:{'before_sha256':sha,'after_sha256':digest(p)} for p,sha in old['effective_runtime_sources'].items() if digest(p)!=sha}
    allowed={str(ROOT/'meta/os_comparison/run.py'),str(AHE_NATIVE/'ahe_native/isolation.py')}
    if not set(changed)<=allowed:raise RuntimeError('unexpected new source changes: '+str(set(changed)-allowed))
    verify_sources(read(root/'protocol.json')['native_source_hashes'])
    experiment=root/'ahe/experiments'/(root.name+'_ahe_os')
    meta,life=read(root/'meta/os/state.json'),read(root/'life/state.json')
    if len(meta['rounds'])!=1 or life['round']!=1 or len(life['rounds'])!=1:
        raise RuntimeError('retain exactly the three first paid proposals')
    provider=TokenLedger(root/'ledgers/optimizer_requests.jsonl').summary()
    Ledger,_=transport_modules();mu=Ledger(root/'meta').summarize('os')
    if not provider['overall']['accounting_usable_with_estimates'] or not mu['accounting_complete']:
        raise RuntimeError('pending or unaccounted paid requests')
    if any(g['round']!=1 and g['attempts'] for g in provider['groups'].values()):
        raise RuntimeError('later paid AHE/Life round already began')
    summary=experiment/'runs/iteration_001/evolve/evolve_summary.md'
    if not summary.exists() or (experiment/'runs/iteration_002/evolve/evolve_summary.md').exists():
        raise RuntimeError('exactly one completed AHE evolution required')
    if workspace_hash(experiment/'workspace')!=workspace_hash(experiment/'runs/iteration_002/input/workspace'):
        raise RuntimeError('AHE first evolved workspace changed')
    jobs=sorted((experiment/'runs/iteration_002/input/benchmark').glob('20*/result.json'))
    if len(jobs)!=1:raise RuntimeError('one completed second-batch measurement required')
    records=list(jobs[0].parent.glob('native_*/native_record.json'))
    ids=read(root/'protocol.json')['batches'][1]['task_ids']
    if len(records)!=50 or {int(read(p)['task_id']) for p in records}!=set(ids):
        raise RuntimeError('AHE second-batch measurements incomplete')
    if source_fingerprint()!=read(root/'ahe_source_receipt.json'):
        raise RuntimeError('official AHE sources changed')
    effective={p:digest(p) for p in old['effective_runtime_sources']}
    effective.update({str(p):digest(p) for p in (ROOT/'meta/os_comparison').glob('*.py')})
    artifacts=[root/'meta/os/state.json',root/'life/state.json',summary,previous,root/'frozen_runtime_sources.json']+records
    artifacts.extend(p for p in (experiment/'workspace').rglob('*') if p.is_file() and '.git' not in p.parts)
    save(target,{'user_authorization':'meta/ahe继续六轮并记录结果，life也继续六轮；撤销首轮性能停止条件',
        'completed_first_rounds':True,'prior_receipt_sha256':digest(previous),
        'original_frozen_receipt_sha256':old['original_frozen_receipt_sha256'],
        'effective_runtime_sources':effective,'source_corrections':changed,
        'starting_artifact_hashes':{str(p):digest(p) for p in artifacts},
        'paid_successful_calls_replayed':0,'ahe_resume_iteration':2,'ahe_eval_replayed':False,
        'retained_ahe_first_round_debugger_limitation':'All ten original debugger CLI calls failed before inference because editable package path was outside chroot; first paid evolution retained and subsequent debugger deployment repaired.',
        'first_round_rows':{'meta':meta['rounds'][0],'life':life['rounds'][0]},
        'retained_usage':{'meta':mu,'provider':provider['overall']}})
    print('Audited: retain all first-round proposals and resume AHE at iteration 2 with its existing measurement.')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    audit(parser.parse_args().root.resolve())
