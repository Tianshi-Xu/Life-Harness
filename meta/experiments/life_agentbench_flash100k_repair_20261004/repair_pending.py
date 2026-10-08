"""One authorized same-round preflight repair; preserve source fingerprints and all spend."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from protocol import HERE,ROOT,DOMAINS,atomic_json,bind_fingerprint,digest,object_digest,validate_protocol
from meta.life_optimizer_v25 import profile as optimizer
from meta.life_optimizer_v25.context import verify_usage
from driver import preservation_check,verify_history
from runner import build_payload

def fixture_contract(checks):
    return {test['test_id']:{k:copy.deepcopy(test[k]) for k in ('rule_id','kind','context','steps') if k in test}
            for test in checks['tests']}

def aggregate_attempts(previous,successful):
    value=copy.deepcopy(successful)
    attempt={'requests':successful['proposer_requests'],'usage':successful['usage']}
    value['successful_attempt']=attempt
    value['previous_failed_attempt']={k:previous[k] for k in ('requests','usage')}
    value['round_attempts']=2
    value['proposer_requests']=previous['requests']+successful['proposer_requests']
    value['usage']={key:previous['usage'].get(key,0)+successful['usage'].get(key,0)
                   for key in set(previous['usage'])|set(successful['usage'])
                   if type(previous['usage'].get(key,0)) is int and type(successful['usage'].get(key,0)) is int}
    return value

def wait_previous_runner(pid,timeout=21600):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        location=Path(f'/proc/{pid}')
        if not location.exists():return
        command=(location/'cmdline').read_bytes().split(b'\0')
        if str(HERE/'runner.py').encode() not in command:
            raise RuntimeError('Predecessor PID reused by unrelated process')
        state=(location/'stat').read_text().split(') ',1)[1].split()[0]
        if state=='Z':return
        time.sleep(30)
    raise RuntimeError('Existing runner not finished; do not merge a live in-memory ledger')

def merge_ledgers(source,destination):
    from ahe_native.token_accounting import TokenLedger
    first=TokenLedger(destination);second=TokenLedger(source)
    if not first.summary()['overall']['accounting_complete'] or not second.summary()['overall']['accounting_complete']:
        raise RuntimeError('Unknown/pending spend; refuse ledger merge or restart')
    existing={(row['event'],row['request_id']):row for row in first.rows}
    for row in second.rows:
        key=(row['event'],row['request_id'])
        if key in existing:
            if existing[key]!=row:raise RuntimeError('Conflicting request receipt')
        else:
            first.append(row);existing[key]=row
    result=first.publish()
    if not result['overall']['accounting_complete']:raise RuntimeError('Merged ledger incomplete')
    return result

def repair(domain,runner_pid):
    validate_protocol();preservation_check()
    state=json.loads((HERE/domain/'state.json').read_text())
    number=state['round']+1
    if number>6 or (HERE/'training_seal.json').exists():raise RuntimeError('Repair restricted to pending unsealed TRAIN')
    failure_path=HERE/domain/'pending_failure.json'
    failure=json.loads(failure_path.read_text())
    if failure['round']!=number or failure['candidate_measured'] is not False or failure['round_advanced'] is not False:
        raise RuntimeError('No matching genuinely pending round')
    original=HERE/domain/'optimizer_sessions'/f'round_{number:02d}'
    previous=json.loads((original/'agent_checkpoint.json').read_text())
    if previous['inflight']:raise RuntimeError('Do not retry an unsettled request')
    workspace=HERE/'optimizer_inputs'/domain/f'round_{number:02d}'
    receipt=optimizer.verify_inputs(workspace)
    if previous['input_sha256']!=object_digest(receipt):raise RuntimeError('Failed attempt input changed')
    donor=original/'workspace'
    if previous['workspace_hashes']!={str(p.relative_to(donor)):digest(p) for p in donor.rglob('*') if p.is_file()}:
        raise RuntimeError('Failed attempt draft changed')
    report=optimizer.check_candidate(donor,donor)
    if report['passed']:raise RuntimeError('Already qualified: use offline recovery instead of a paid repair')
    expected=fixture_contract(json.loads((donor/'checks.json').read_text()))
    folder=HERE/'same_round_repairs'/f'{domain}_round{number:02d}_attempt02'
    folder.mkdir(parents=True,exist_ok=True)
    snapshot=folder/'source_snapshot.py'
    if snapshot.exists() and digest(snapshot)!=digest(Path(__file__)):
        raise RuntimeError('Repair source changed since attempt started')
    if not snapshot.exists():shutil.copy2(Path(__file__),snapshot)
    bind_fingerprint(folder/'authorization.json',{'user_request':'continue new six rounds and require qualified measured candidates',
        'domain':domain,'round':number,'parent_snapshot_sha256':state['active_snapshot_sha256'],
        'original_checkpoint_sha256':digest(original/'agent_checkpoint.json'),
        'failure_receipt_sha256':digest(failure_path),'source_sha256':digest(Path(__file__)),
        'max_requests_per_attempt':32,'max_output_tokens':100000,
        'guard':'keep all original fixture contracts; repair evolved code and operation metadata via DeepSeek; no manual policy filling',
        'concurrency':'repair broker limit 1 plus existing runner at most 3 domain workers = at most 4'})
    sys.path.insert(0,str(ROOT.parent/'ahe-native'))
    from ahe_native.connection import credentials
    from ahe_native.metered_broker import Broker,RequestGate
    from ahe_native.token_accounting import TokenLedger,scoped_url
    current=TokenLedger(HERE/'flash_requests.jsonl').summary()
    group=current['groups'].get(f'{HERE.name}|{domain}|{number}|life_optimizer')
    if (group is None or not group['accounting_complete']
            or group['total_tokens']!=previous['usage']['total_tokens']):
        raise RuntimeError('Pending round original paid spend does not match settled receipt')
    gate=RequestGate(1,min_interval=1.25)
    key,url=credentials()
    if url!=json.loads((HERE/'protocol.json').read_text())['proposer_host']+'/protocol/openai/v1':
        raise RuntimeError('Unexpected API host')
    broker=Broker(key,url,gate,rate_limit_retries=2,ledger_path=folder/'flash_requests.jsonl')
    key_bytes=key.encode()
    if any(key_bytes in p.read_bytes() for p in folder.rglob('*') if p.is_file()):
        broker.close();raise RuntimeError('Credential present on disk; repair refused')
    del key,key_bytes
    cfg=json.loads((HERE/'protocol.json').read_text())
    log=HERE/domain/'flash_requests.jsonl'
    serial=len(log.read_text().splitlines()) if log.exists() else 0
    first=True
    failed_tests=[test for test in report.get('tests',[]) if not test['passed']]
    def completion(messages,tools):
        nonlocal serial,first
        if first:
            messages.append({'role':'user','content':
                'This is an authorized same-round preflight repair, not a new optimization round. '
                'The previous paid attempt exhausted 32 requests. Keep its diagnosis and all existing checks, '
                'including every counterexample and expected output, unchanged. Repair the generated '
                'artifact code and any invalid bundle-operation metadata to satisfy the existing contracts and inventory. Start with the failed fixture(s), '
                'reuse previous case reads, do not restart exhaustive diagnosis, and call check_candidate. '
                'Actual failed tests: '+json.dumps(failed_tests,ensure_ascii=False)})
            first=False
        messages=copy.deepcopy(messages)
        serial+=1;payload=build_payload(cfg,messages,tools)
        endpoint=scoped_url(broker.url,experiment=HERE.name,domain=domain,round=number,role='life_optimizer')
        metadata={'round':number,'request_number':serial,'repair_attempt':2,'model':payload['model'],
            'max_tokens':100000,'local_context_limit':None,'message_count':len(messages)}
        atomic_json(folder/'progress.json',{'phase':'flash_request',**metadata,'gate':gate.snapshot()})
        request=urllib.request.Request(endpoint+'/chat/completions',method='POST',
            headers={'Authorization':'Bearer EMPTY','Content-Type':'application/json'},
            data=json.dumps(payload,ensure_ascii=False).encode())
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request,timeout=2100) as response:result=json.load(response)
        with log.open('a') as stream:
            stream.write(json.dumps({**metadata,'usage':result.get('usage',{})})+'\n')
            stream.flush();os.fsync(stream.fileno())
        verify_usage(result)
        if not result.get('choices') or result.get('error'):raise RuntimeError('Missing assistant choice')
        return result
    error=None
    try:
        output=folder/'proposal'
        optimizer.propose(workspace,output,completion=completion,max_turns=32,
            checkpoint_dir=folder/'optimizer_session',recovery_outputs=donor)
        checks=json.loads((output/'checks.json').read_text())
        if fixture_contract(checks)!=expected:raise RuntimeError('Repair changed original regression expectations; not eligible')
        validation=json.loads((output/'validation.json').read_text())
        validation=aggregate_attempts(previous,validation)
        validation['repair_receipt_sha256']=digest(folder/'authorization.json')
        validation['original_checkpoint_sha256']=digest(original/'agent_checkpoint.json')
        atomic_json(output/'validation.json',validation)
        destination=HERE/domain/'proposals'/f'round_{number:02d}'
        if destination.exists():
            raise RuntimeError('Round proposal already exists; no overwrite')
        shutil.copytree(output,destination)
        # Validate immutable export again using the original frozen input.
        optimizer.propose(workspace,destination,completion=lambda *_:(_ for _ in ()).throw(RuntimeError('Audit must be offline')))
        atomic_json(folder/'progress.json',{'phase':'qualified_repair_ready','domain':domain,'round':number,
            'attempt_requests':validation['successful_attempt']['requests'],
            'attempt_tokens':validation['successful_attempt']['usage']['total_tokens'],
            'all_round_requests':validation['proposer_requests'],'all_round_tokens':validation['usage']['total_tokens']})
        print(json.dumps({'phase':'qualified_repair_ready','domain':domain,'round':number}),flush=True)
    except Exception as exc:
        error=exc
        atomic_json(folder/'progress.json',{'phase':'repair_failed','type':type(exc).__name__,'detail':str(exc)[:3000]})
    finally:
        broker.close();broker.ledger.publish()
    # Separate ledger prevents concurrent writers with independent in-memory ledgers.
    # Wait for the original runner to close before merging every paid attempt.
    wait_previous_runner(runner_pid)
    with (HERE/'.runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        merged=merge_ledgers(folder/'flash_requests.jsonl',HERE/'flash_requests.jsonl')
        bind_fingerprint(folder/'merge_receipt.json',{'source_ledger_sha256':digest(folder/'flash_requests.jsonl'),
            'all_usage':merged['overall'],'all_attempts_preserved':True})
        from runner import publish,audit_usage
        publish(TokenLedger(HERE/'flash_requests.jsonl'))
        if error:raise error
        preservation_check()
        for d in DOMAINS:verify_history(d,json.loads((HERE/d/'state.json').read_text()))
        audit_usage(TokenLedger(HERE/'flash_requests.jsonl'))
    with (HERE/'runner.log').open('a') as logfile:
        child=subprocess.Popen([str(ROOT/'AgentBench/.native/core/bin/python'),'-B','-u',str(HERE/'runner.py')],
            cwd=HERE,stdin=subprocess.DEVNULL,stdout=logfile,stderr=subprocess.STDOUT,start_new_session=True)
    atomic_json(folder/'resume.json',{'pid':child.pid,'same_sources':True,'only_pending_rounds_will_run':True})
    print(json.dumps({'resumed_pid':child.pid}),flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--domain',choices=DOMAINS,required=True)
    parser.add_argument('--runner-pid',type=int,required=True)
    parser.add_argument('--launch',action='store_true')
    args=parser.parse_args()
    if args.launch:
        log=HERE/f'pending_repair_{args.domain}.log'
        with log.open('a') as stream:
            child=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).resolve()),
                '--domain',args.domain,'--runner-pid',str(args.runner_pid)],cwd=HERE,
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        atomic_json(HERE/f'pending_repair_{args.domain}_launch.json',{'pid':child.pid})
        print(json.dumps({'repair_pid':child.pid,'log':str(log)}))
    else:
        with (HERE/f'.pending_repair_{args.domain}.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            repair(args.domain,args.runner_pid)

if __name__=='__main__':main()
