"""Audit and prepare one continuation of the stopped OS integration run, without API calls."""
import argparse
import json
from pathlib import Path
import sys
from meta.tau_comparison.common import AHE_NATIVE,ROOT,read,transport_modules
from .protocol import digest,save,verify_sources
from .recovery import AUTHORIZATION,recover_checkpoint


def audit(root):
    sys.path.insert(0,str(AHE_NATIVE))
    from ahe_native.backend import workspace_hash
    from ahe_native.token_accounting import TokenLedger
    if read(root/'status.json')['phase'] != 'stopped_on_error' or (root/'training_seal.json').exists():
        raise RuntimeError('stopped TRAIN-only ancestor required')
    directory=root/'recovery'
    if (directory/'receipt.json').exists():
        raise FileExistsError('one audited continuation only')
    original=read(root/'frozen_runtime_sources.json')
    allowed={str(AHE_NATIVE/'ahe_native'/name) for name in ['records.py','metered_broker.py','token_accounting.py']}
    allowed|={str(ROOT/'meta/os_comparison'/name) for name in ['run.py','evaluate.py','meta_port.py']}
    changed={p:{'before_sha256':sha,'after_sha256':digest(p)} for p,sha in original.items() if digest(p)!=sha}
    if not set(changed)<=allowed:
        raise RuntimeError('unaudited runtime changes: '+str(set(changed)-allowed))
    verify_sources(read(root/'protocol.json')['native_source_hashes'])
    state_path=root/'meta/os/state.json';meta=read(state_path);life=read(root/'life/state.json')
    if len(meta['rounds'])!=1 or meta['rounds'][0]['candidate_score'] is not None or life['round']!=0 or life['rounds']:
        raise RuntimeError('retain the exact first Meta proposal and interrupted first Life checkpoint')
    row=meta['rounds'][0];candidate=root/'meta/os/workspace/agents'/ (row['candidate']['name']+'.py')
    proposer=read(root/'meta/os/proposer_01/receipt.json')
    if digest(candidate)!=proposer['candidate_sha256']:
        raise RuntimeError('paid Meta candidate changed')
    Ledger,_=transport_modules();meta_usage=Ledger(root/'meta').summarize('os')
    provider=TokenLedger(root/'ledgers/optimizer_requests.jsonl')
    if not meta_usage['accounting_complete'] or any(g['role'] in {'debugger','evolve'} and g['attempts'] for g in provider.summary()['groups'].values()):
        raise RuntimeError('pending Meta calls or already paid AHE steps cannot be replayed')
    experiment=root/'ahe/experiments'/(root.name+'_ahe_os');workspace=experiment/'workspace'
    pending=list((experiment/'runs/iteration_001/input/benchmark/.native_pending').iterdir())
    if len(pending)!=1:raise RuntimeError('one completed baseline required')
    expected={(str(i),0) for i in read(root/'protocol.json')['batches'][0]['task_ids']};found=set();episodes=[]
    for p in sorted(pending[0].glob('*/request.json')):
        req,record=read(p),read(p.parent/'record.json');key=(record['task_id'],record['trial'])
        if key in found or key not in expected or key!=(req['task_id'],req['trial']) or workspace_hash(req['workspace'])!=workspace_hash(workspace):
            raise RuntimeError('AHE episode identity mismatch')
        found.add(key);episodes.append({'folder':str(p.parent),'files':{str(f.relative_to(p.parent)):digest(f) for f in p.parent.rglob('*') if f.is_file()}})
    if found!=expected:raise RuntimeError('all fifty original AHE episodes required')
    directory.mkdir(exist_ok=True)
    save(directory/'meta_state_before_remeasurement.json',meta)
    preflight=root/'meta/os/preflight_01.json'
    if preflight.exists():preflight.rename(directory/'meta_preflight_before_remeasurement.json')
    meta.update(rounds=[],batch=0,incumbent=row['parent'],baseline_phase='batch_00_baseline',status='running')
    save(state_path,meta)
    checkpoint=root/'life/optimizer_sessions/round_01/agent_checkpoint.json'
    recover_checkpoint(checkpoint,provider,root.name,1,directory)
    effective={p:digest(p) for p in original}
    effective.update({str(p):digest(p) for p in (ROOT/'meta/os_comparison').glob('*.py')})
    artifacts=[state_path,root/'life/state.json',checkpoint,candidate,root/'meta/os/pending_01.json',root/'ledgers/optimizer_requests.estimates.json']
    for folder in [root/'optimizer_inputs/round_01',root/'life/evals/incumbent_01']:
        artifacts.extend(p for p in folder.rglob('*') if p.is_file())
    save(directory/'receipt.json',{'authorization_quote':AUTHORIZATION,'reason':'OS exporter allowlist, stdlib candidate validator, longer broker timeout and audited Life timeout retry',
        'original_frozen_receipt_sha256':digest(root/'frozen_runtime_sources.json'),
        'source_corrections':changed,'effective_runtime_sources':effective,
        'starting_artifact_hashes':{str(p):digest(p) for p in artifacts},
        'ahe_baseline_reuse':{'workspace_sha256':workspace_hash(workspace),'episodes':episodes},
        'successful_paid_calls_replayed':0,'paid_meta_candidate_sha256':digest(candidate),
        'api_measured_usage_unchanged':True,'unknown_usage_estimated_separately':True,
        'original_upstream_logic_changed':False})
    print(json.dumps({'passed':True,'successful_paid_calls_replayed':0,'reused_ahe_episodes':50,'timeout_retry_ready':True}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    audit(parser.parse_args().root.resolve())
