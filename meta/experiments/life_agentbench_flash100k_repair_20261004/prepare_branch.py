"""Prepare an independent minimal-rerun branch; never modify predecessor evidence."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from protocol import HERE, ROOT, DOMAINS, atomic_json, bind_fingerprint, digest, object_digest, validate_protocol
from composition import verify_active
from meta.life_optimizer_v25 import profile as optimizer

OLD=HERE.parent/'life_agentbench_six_flash_tokens_20261004'
OLD_PID=2890786

def audit_predecessor():
    process=Path(f'/proc/{OLD_PID}')
    if process.exists():
        if str(OLD/'runner.py').encode() not in (process/'cmdline').read_bytes().split(b'\0'):
            raise RuntimeError('Predecessor PID belongs to unrelated process')
        if (process/'stat').read_text().split(') ',1)[1].split()[0] not in {'T','t'}:
            raise RuntimeError('Predecessor must be safely paused')
    sys.path.insert(0,str(ROOT.parent/'ahe-native'))
    from ahe_native.token_accounting import TokenLedger
    usage=TokenLedger(OLD/'flash_requests.jsonl').summary()
    if not usage['overall']['accounting_complete']:raise RuntimeError('Unsettled predecessor costs')
    for domain in DOMAINS:
        for checkpoint in (OLD/domain/'optimizer_sessions').glob('round_*/agent_checkpoint.json'):
            if json.loads(checkpoint.read_text()).get('inflight'):raise RuntimeError('Unsettled agent checkpoint')
    # Import the frozen predecessor in its own interpreter, not this revision's namespace.
    command=[str(ROOT/'AgentBench/.native/core/bin/python'),'-B','-c',
        "import json; import driver; driver.validate_protocol(); driver.preservation_check(); "
        "[driver.verify_history(d,json.loads((driver.HERE/d/'state.json').read_text())) for d in driver.DOMAINS]; "
        "print('predecessor audited without optimizer calls')"]
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
    result=subprocess.run(command,cwd=OLD,env=env,capture_output=True,text=True,timeout=180)
    if result.returncode:raise RuntimeError('Predecessor audit failed: '+result.stderr[-3000:])
    return usage

def verify_behavior_equivalence():
    old=(ROOT/'meta/agentbench_hooks/runtime.py').read_text()
    new=(ROOT/'meta/agentbench_hooks_v31/runtime.py').read_text()
    if new.replace('life-agentbench-h2345-v3.1','life-agentbench-h2345-v3')!=old:
        raise RuntimeError('Runtime behavior changed; old evaluations cannot be reused')
    receipts={}
    for name in ('agentbench_evaluate.py','agentbench_chain.py','composition.py','tokenizer_worker.py'):
        a=(OLD/name).read_text()
        b=(HERE/name).read_text().replace('meta.agentbench_hooks_v31','meta.agentbench_hooks')
        b=b.replace('meta.life_optimizer_v25','meta.life_optimizer')
        if a.rstrip()!=b.rstrip():raise RuntimeError('Evaluator/composition behavior changed: '+name)
        receipts[name]={'old_sha256':digest(OLD/name),'new_sha256':digest(HERE/name)}
    return receipts

def prepare():
    if (HERE/'protocol.json').exists():
        validate_protocol()
        from driver import preservation_check
        preservation_check()
        return {'status':'existing_branch_verified'}
    usage=audit_predecessor()
    equivalence=verify_behavior_equivalence()
    frozen={}
    def preserve(path):
        path=Path(path).resolve()
        if path.is_file():frozen[str(path)]=digest(path)
    for name in ('protocol.json','flash_requests.jsonl','preservation_before.json'):
        preserve(OLD/name)
    cfg=json.loads((OLD/'protocol.json').read_text())
    for name in cfg['optimizer_source_sha256']:preserve(ROOT/name)
    for name in cfg['interface_source_sha256']:preserve(ROOT/'meta/agentbench_hooks'/name)
    for name in ('protocol.py','driver.py','runner.py','agentbench_evaluate.py','composition.py',
                 'agentbench_chain.py','candidate_contract.py','reporting.py','tokenizer_worker.py'):
        preserve(OLD/name)
    ancestor={'source':str(OLD),'ledger_sha256':digest(OLD/'flash_requests.jsonl'),
        'usage':usage,'include_abandoned_attempts':True,
        'rule':'All predecessor requests retained in total spend; inherited or discarded branch costs are never zeroed.'}
    bind_fingerprint(HERE/'ancestor_usage.json',ancestor)
    inherited={}
    for domain in DOMAINS:
        source=(OLD/domain/'state.json' if domain=='dbbench'
                else OLD/domain/'round_inputs/round_02.json')
        seed=json.loads(source.read_text())
        expected=3 if domain=='dbbench' else 1
        if seed['round']!=expected or len(seed['rounds'])!=expected or any(
            r['outcome']!='evaluated_candidate' or r['candidate_successes'] is None for r in seed['rounds']):
            raise RuntimeError('Unexpected or unqualified inherited prefix')
        destination=HERE/domain;destination.mkdir(exist_ok=True)
        preserve(source)
        for entry in seed['operation_log']:
            for op in entry['operations']:
                if op['op']=='delete':continue
                relative=Path('versions')/entry['bundle_id']/(op['operation_id']+'.py')
                src=OLD/domain/relative;dst=destination/relative
                preserve(src);dst.parent.mkdir(parents=True,exist_ok=True)
                if dst.exists() and digest(dst)!=digest(src):raise RuntimeError('Inherited artifact changed')
                if not dst.exists():shutil.copy2(src,dst)
        verify_active(destination,seed)
        for row in seed['rounds']:
            for kind in ('incumbent','candidate'):
                evaluation=OLD/domain/'evals'/row[kind+'_phase']
                for path in evaluation.glob('*.json'):preserve(path)
            for folder in (OLD/domain/'proposals'/f"round_{row['round']:02d}",
                           OLD/'optimizer_inputs'/domain/f"round_{row['round']:02d}"):
                for path in folder.rglob('*'):
                    if path.is_file():preserve(path)
            preserve(OLD/domain/'round_inputs'/f"round_{row['round']:02d}.json")
        # Freeze interrupted workspaces, not mutable native evaluation logs.
        for checkpoint in (OLD/domain/'optimizer_sessions').glob('round_*'):
            for path in checkpoint.rglob('*'):
                if path.is_file():preserve(path)
        for path in (OLD/domain/'failures').glob('*.json'):preserve(path)
        preserve(OLD/domain/'flash_requests.jsonl')
        seed['status']='running'
        inheritance={'source':str(source),'source_sha256':digest(source),
            'inherited_rounds':copy.deepcopy(seed['rounds']),
            'initial_snapshot_sha256':seed['active_snapshot_sha256'],
            'initial_snapshot':seed['active_snapshot'],
            'excluded_later_rounds':json.loads((OLD/domain/'state.json').read_text())['rounds'][expected:],
            'behavior_equivalence':equivalence}
        bind_fingerprint(destination/'inheritance.json',inheritance)
        bind_fingerprint(destination/'state.json',seed)
        inherited[domain]=expected
    cfg.update(experiment=HERE.name,initialization='verified-repair-continuation',
        flash_max_tokens=100000,optimizer_profile=optimizer.PROFILE,
        optimizer_source_sha256=optimizer.source_hashes(),
        interface_version='life-agentbench-h2345-v3.1',
        interface_source_sha256={name:digest(ROOT/'meta/agentbench_hooks_v31'/name)
                                 for name in cfg['interface_source_sha256']},
        inherited_rounds=inherited,predecessor=str(OLD),
        retry_gate='ALFWorld/WebShop second round: qualified candidate and complete paired 50-task evaluation, known usage',
        proposal_failure_policy='Persist pending same round; no round advancement or silent automatic paid restart.',
        six_round_protocol='Exactly six qualified measured decisions per domain; historical failed attempts charged separately.',
        proposer_budget='Agentic max 32 API requests per repair proposal, including tool calls and preflight fixes; no cumulative token cap.',
        initial_active_snapshot={d:json.loads((HERE/d/'state.json').read_text())['active_snapshot'] for d in DOMAINS},
        comparison_label='repair continuation with inherited prefixes and all predecessor plus repair attempt tokens; not a fresh equal-budget six-round run')
    cfg.pop('first_round_gate',None)
    bind_fingerprint(HERE/'protocol.json',cfg)
    bind_fingerprint(HERE/'preservation.json',frozen)
    atomic_json(HERE/'predecessor_pause.json',{'pid':OLD_PID,'state':'paused_at_settled_checkpoint',
        'accounting_complete':True,'predecessor_ledger_sha256':ancestor['ledger_sha256']})
    validate_protocol()
    from driver import preservation_check, render_training_report
    result=preservation_check();render_training_report()
    return {'status':'prepared','inherited_rounds':inherited,'preservation':result,
            'ancestor_tokens':usage['overall']['total_tokens']}

if __name__=='__main__':
    print(json.dumps(prepare(),ensure_ascii=False))
