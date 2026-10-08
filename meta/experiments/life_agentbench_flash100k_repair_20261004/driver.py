"""Audited repair continuation: a round exists only after a qualified evaluation."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import threading
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from meta.life_optimizer_v25 import profile as optimizer
from meta.life_optimizer_v25.context import ProposalFailure
from composition import commit_bundle, stage_bundle, verify_active, write_active_chain
from agentbench_chain import manifest_records
from protocol import (DOMAINS, HERE, ROOT, ROUNDS, atomic_json, bind_fingerprint,
                      digest, object_digest, protocol, task_ids, validate_protocol)

OLD = HERE.parent/'life_agentbench_six_flash_tokens_20261004'
SOURCES = ('protocol.py','driver.py','runner.py','prepare_branch.py','candidate_contract.py',
           'composition.py','agentbench_chain.py','agentbench_evaluate.py','tokenizer_worker.py')
REPORT_LOCK=threading.RLock()

def progress(domain, **values):
    row={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'domain':domain,**values}
    atomic_json(HERE/domain/'progress.json',row)
    print(json.dumps(row,ensure_ascii=False),flush=True)

def child_environment(domain):
    env={k:v for k,v in os.environ.items() if not any(w in k.upper() for w in ('API_KEY','API_TOKEN','SECRET','PASSWORD'))}
    env.update(TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    if domain=='webshop':
        java=ROOT/'AgentBench/.native/sysroot/usr/lib/jvm/java-11-openjdk-amd64'
        env.update(JAVA_HOME=str(java),JVM_PATH=str(java/'lib/server/libjvm.so'),PATH=str(java/'bin')+':'+env.get('PATH',''),
                   _JAVA_OPTIONS='-Xmx4g',PYTHONPATH=str(ROOT/'AgentBench/.native/webshop-src')+':'+str(ROOT/'AgentBench'))
    return env

def run_fingerprint(domain):
    return {'protocol_sha256':digest(HERE/'protocol.json'),'sources':{n:digest(HERE/n) for n in SOURCES},
            'optimizer_sources':optimizer.source_hashes(),'inheritance_sha256':digest(HERE/domain/'inheritance.json'),
            'transport_sources':{n:digest(ROOT.parent/'ahe-native/ahe_native'/n) for n in
                                 ('api_broker.py','metered_broker.py','token_accounting.py','connection.py')}}

def preservation_check():
    hashes=json.loads((HERE/'preservation.json').read_text())
    changed=[name for name,sha in hashes.items() if digest(Path(name))!=sha]
    if changed:raise RuntimeError('Predecessor or historical evidence changed: '+str(changed[:5]))
    return {'files_checked':len(hashes),'changed':[]}

def read_summary(path):
    from agentbench_evaluate import validate_cached_summary
    summary=json.loads((path/'summary.json').read_text())
    validate_cached_summary(path,summary)
    return summary

def phase(domain, split, name, chain, batch):
    if split=='test':verify_seal()
    destination=HERE/domain/'evals'/name
    # The solver/evaluator behavior is byte-identical apart from validation namespace
    # and version metadata. Reuse only exact content/task matches, with a receipt.
    legacy=OLD/domain/'evals'/name
    if split=='train' and not (destination/'summary.json').exists() and (legacy/'summary.json').exists():
        summary=read_summary(legacy)
        _,records=manifest_records(chain)
        expected=task_ids(validate_protocol(check_files=False)['domains'][domain],batch,'train')
        if (summary['snapshot_sha256']==object_digest(records) and summary['ids']==expected
                and summary['split']=='train' and summary['n']==50):
            receipt={'source':str(legacy),'summary_sha256':digest(legacy/'summary.json'),
                     'snapshot_sha256':summary['snapshot_sha256'],'ids':expected,
                     'reason':'exact content, frozen solver/data and behavior-equivalent runtime; no score rewriting'}
            bind_fingerprint(HERE/domain/'reused_evals'/f'{name}.json',receipt)
            progress(domain,phase=name,status='reused_complete_train_evaluation',successes=summary['successes'])
            return summary,legacy
    python=ROOT/{'alfworld':'AgentBench/.native/alfworld-venv/bin/python',
                 'dbbench':'AgentBench/.native/core/bin/python','webshop':'AgentBench/.native/webshop-venv/bin/python'}[domain]
    progress(domain,phase=name,status='evaluating',batch=batch)
    with (HERE/domain/(name+'.log')).open('a') as log:
        result=subprocess.run([str(python),'-u',str(HERE/'agentbench_evaluate.py'),domain,split,name,str(chain),
                               '--batch',str(batch)],cwd=ROOT,env=child_environment(domain),
                              stdout=log,stderr=subprocess.STDOUT,timeout=21600)
    if result.returncode:raise RuntimeError('Native evaluation failed; inspect '+str(HERE/domain/(name+'.log')))
    return read_summary(destination),destination

def json_operations(bundle):
    return [{k:v for k,v in op.items() if k!='artifact_path'} for op in bundle['operations']]

def verify_history(domain,state):
    inheritance=json.loads((HERE/domain/'inheritance.json').read_text())
    prefix=inheritance['inherited_rounds']
    if state['rounds'][:len(prefix)]!=prefix:raise RuntimeError('Inherited decisions changed')
    if len(state['rounds'])!=state['round']:raise RuntimeError('Unmeasured attempt must not increment round')
    verify_active(HERE/domain,state)
    parent=inheritance['initial_snapshot_sha256']
    for number,row in enumerate(state['rounds'][len(prefix):],len(prefix)+1):
        if row['round']!=number or row['train_batch']!=number-1:
            raise RuntimeError('Round and frozen batch ordering mismatch')
        if row['parent_snapshot_sha256']!=parent or row['outcome']!='evaluated_candidate':
            raise RuntimeError('Round must have a qualified measured candidate and exact ancestry')
        if row['candidate_successes'] is None:raise RuntimeError('Unmeasured round in completed history')
        incumbent=read_summary(Path(row['incumbent_eval']))
        candidate=read_summary(Path(row['candidate_eval']))
        for kind,summary in (('incumbent',incumbent),('candidate',candidate)):
            if digest(Path(row[kind+'_eval'])/'summary.json')!=row[kind+'_summary_sha256']:
                raise RuntimeError('Completed evaluation receipt changed')
            if summary['snapshot_sha256']!=row[kind+'_snapshot_sha256']:
                raise RuntimeError('Evaluated snapshot mismatch')
        if incumbent['snapshot_sha256']!=parent:
            raise RuntimeError('Incumbent differs from retained parent')
        expected=task_ids(validate_protocol(check_files=False)['domains'][domain],row['train_batch'],'train')
        if optimizer.strict_improvement(candidate,incumbent,expected_task_ids=expected)!=row['accepted']:
            raise RuntimeError('Paired decision changed')
        output=HERE/domain/'proposals'/f"round_{row['round']:02d}"
        optimizer.propose(HERE/'optimizer_inputs'/domain/f"round_{row['round']:02d}",output,
                          completion=lambda *_:(_ for _ in ()).throw(RuntimeError('Audit cannot call LLM')))
        receipt=optimizer.verify_inputs(HERE/'optimizer_inputs'/domain/f"round_{row['round']:02d}")
        seed=HERE/domain/'round_inputs'/f"round_{row['round']:02d}.json"
        if (receipt['seed_state_sha256']!=digest(seed)
                or receipt['evaluation_sha256']!=row['incumbent_summary_sha256']
                or json.loads(seed.read_text())['active_snapshot_sha256']!=parent):
            raise RuntimeError('Optimizer parent/evidence provenance changed')
        if digest(output/'validation.json')!=row['validation_sha256']:raise RuntimeError('Validation changed')
        parent=row['active_snapshot_sha256']
    if parent!=state['active_snapshot_sha256']:raise RuntimeError('Final retained snapshot changed')

def round_recovery(domain,number,state):
    source=OLD/domain/'optimizer_sessions'/f'round_{number:02d}'/'workspace'
    if not (source/'inventory.json').exists():return None
    old_inventory=json.loads((source/'inventory.json').read_text())
    return source if old_inventory['snapshot_sha256']==state['active_snapshot_sha256'] else None

def run_domain(domain,completion,stop_after=6):
    validate_protocol()
    directory=HERE/domain
    with (directory/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        bind_fingerprint(directory/'run_config.json',run_fingerprint(domain))
        state=json.loads((directory/'state.json').read_text())
        verify_history(domain,state)
        for number in range(state['round']+1,min(ROUNDS,stop_after)+1):
            batch=number-1
            bind_fingerprint(directory/'round_inputs'/f'round_{number:02d}.json',state)
            chain=write_active_chain(directory,state,f'incumbent_{number:02d}')
            incumbent,incumbent_dir=phase(domain,'train',f'round_{number:02d}_incumbent',chain,batch)
            workspace=HERE/'optimizer_inputs'/domain/f'round_{number:02d}'
            if not workspace.exists():optimizer.prepare(directory,incumbent_dir,workspace,seed=number)
            proposal=directory/'proposals'/f'round_{number:02d}'
            progress(domain,round=number,phase='proposing',incumbent=incumbent['successes'],batch=batch)
            try:
                path=optimizer.propose(workspace,proposal,completion=completion,max_turns=32,
                    checkpoint_dir=directory/'optimizer_sessions'/f'round_{number:02d}',
                    recovery_outputs=round_recovery(domain,number,state))
            except ProposalFailure as exc:
                atomic_json(directory/'pending_failure.json',{'round':number,'code':exc.code,'detail':str(exc),
                    'candidate_measured':False,'round_advanced':False,'incumbent_retained':True})
                progress(domain,round=number,phase='needs_same_round_repair',code=exc.code)
                raise
            bundle,staged,staged_chain=stage_bundle(directory,number,path,state)
            candidate,candidate_dir=phase(domain,'train',f'round_{number:02d}_candidate',staged_chain,batch)
            if candidate['n']!=50 or candidate['infra_errors']==50 or incumbent['infra_errors']==50:
                raise RuntimeError('No complete usable paired evaluation; round is NOT advanced')
            accepted=optimizer.strict_improvement(candidate,incumbent,
                expected_task_ids=task_ids(validate_protocol(check_files=False)['domains'][domain],batch,'train'))
            validation=json.loads((proposal/'validation.json').read_text())
            usage={k:validation['usage'].get(k,0) for k in ('prompt_tokens','completion_tokens','total_tokens')}
            row={'round':number,'train_batch':batch,'outcome':'evaluated_candidate','accepted':accepted,
                 'incumbent_successes':incumbent['successes'],'incumbent_score':incumbent['accuracy'],
                 'candidate_successes':candidate['successes'],'candidate_score':candidate['accuracy'],
                 'incumbent_infra_errors':incumbent['infra_errors'],'candidate_infra_errors':candidate['infra_errors'],
                 'parent_snapshot_sha256':state['active_snapshot_sha256'],'incumbent_eval':str(incumbent_dir),
                 'incumbent_summary_sha256':digest(incumbent_dir/'summary.json'),
                 'candidate_summary_sha256':digest(candidate_dir/'summary.json'),
                 'incumbent_snapshot_sha256':incumbent['snapshot_sha256'],
                 'candidate_snapshot_sha256':candidate['snapshot_sha256'],
                 'candidate_eval':str(candidate_dir),'bundle_id':bundle['bundle_id'],'bundle_hypothesis':bundle['hypothesis'],
                 'operations':json_operations(bundle),'usage':usage,'proposer_requests':validation['proposer_requests'],
                 'validation_sha256':digest(proposal/'validation.json'),'fallback_reason':validation['fallback_reason'],
                 'recovered_qualified':validation['recovered_qualified'],'profile':optimizer.PROFILE}
            if accepted:
                records=commit_bundle(directory,bundle,state)
                state['active_snapshot']=records;state['active_snapshot_sha256']=object_digest(records)
                state['accepted_bundle_ids'].append(bundle['bundle_id'])
                state['operation_log'].append({'bundle_id':bundle['bundle_id'],'round':number,
                    'parent_snapshot_sha256':row['parent_snapshot_sha256'],
                    'committed_snapshot_sha256':state['active_snapshot_sha256'],'operations':row['operations']})
            row['active_snapshot_sha256']=state['active_snapshot_sha256']
            state['rounds'].append(row);state['round']=number
            state['status']='complete' if number==6 else 'running'
            atomic_json(directory/'state.json',state)
            verify_history(domain,state)
            progress(domain,round=number,phase='decision',candidate=candidate['successes'],
                     incumbent=incumbent['successes'],accepted=accepted,n=50)
            render_training_report()
        return state

def seal_value():
    return {'experiment':HERE.name,'state_sha256':{d:digest(HERE/d/'state.json') for d in DOMAINS},
            'run_config_sha256':{d:digest(HERE/d/'run_config.json') for d in DOMAINS},
            'protocol_sha256':digest(HERE/'protocol.json'),'flash_ledger_sha256':digest(HERE/'flash_requests.jsonl'),
            'flash_usage_sha256':digest(HERE/'TOKEN_USAGE.json'),'inheritance_sha256':digest(HERE/'ancestor_usage.json')}

def verify_seal(root=HERE):
    if Path(root)!=HERE or not (HERE/'training_seal.json').exists():raise RuntimeError('Test refused before all training is sealed')
    for domain in DOMAINS:
        state=json.loads((HERE/domain/'state.json').read_text())
        if state['round']!=6 or len(state['rounds'])!=6 or state['status']!='complete':raise RuntimeError('Exactly six measured decisions required')
        if json.loads((HERE/domain/'run_config.json').read_text())!=run_fingerprint(domain):raise RuntimeError('Frozen source changed')
        verify_history(domain,state)
    if json.loads((HERE/'training_seal.json').read_text())!=seal_value():raise RuntimeError('Seal changed')
    return seal_value()

def seal_training():
    for domain in DOMAINS:
        state=json.loads((HERE/domain/'state.json').read_text())
        if state['round']!=6 or len(state['rounds'])!=6 or state['status']!='complete':
            raise RuntimeError('Cannot seal before all six qualified measured decisions')
        verify_history(domain,state)
    bind_fingerprint(HERE/'training_seal.json',seal_value())
    return verify_seal()

def render_training_report():
    with REPORT_LOCK:
        return _render_training_report()

def _render_training_report():
    lines=['# Life repaired continuation: TRAIN only','','Inherited qualified rounds retained; failed attempts do not advance rounds.',
           'Flash max output 100000; local solver total 32768/output 4096; cases read on demand.',
           'New-request tokens below; ancestor spend is separately retained in ancestor_usage.json/TOTAL_SPEND.json.','',
           '| Domain | Round | Batch | Incumbent | Candidate | Accepted | New token usage | Provenance |',
           '|---|---:|---:|---:|---:|---|---:|---|']
    for domain in DOMAINS:
        state=json.loads((HERE/domain/'state.json').read_text())
        count=len(json.loads((HERE/domain/'inheritance.json').read_text())['inherited_rounds'])
        for row in state['rounds']:
            inherited=row['round']<=count
            cost=0 if inherited else row['usage']['total_tokens']
            lines.append(f"| {domain} | {row['round']} | b{row['train_batch']} | {row['incumbent_successes']}/50 | {row['candidate_successes']}/50 | {row['accepted']} | {cost} | {'inherited; original tokens in ancestor ledger' if inherited else 'qualified measured candidate'} |")
    temp=HERE/'TRAINING.md.tmp';temp.write_text('\n'.join(lines)+'\n');temp.replace(HERE/'TRAINING.md')
