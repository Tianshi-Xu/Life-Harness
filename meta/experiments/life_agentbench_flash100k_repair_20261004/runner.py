"""Repair failed rounds first; continue only after qualified full paired TRAIN scores."""
import copy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from driver import (DOMAINS,HERE,ROOT,ROUNDS,phase,preservation_check,progress,
                    render_training_report,run_domain,seal_training,verify_seal)
from composition import write_active_chain
from protocol import atomic_json,digest,protocol,validate_protocol
from meta.life_optimizer_v25.context import verify_usage

RETRY_DOMAINS=('alfworld','webshop')

def build_payload(cfg,messages,tools):
    if cfg['proposer_model']!='deepseek-v4-flash' or cfg['flash_max_tokens']!=100000 or cfg['flash_context_limit'] is not None:
        raise RuntimeError('Flash must have 100000 output and no local solver context cap')
    return {'model':cfg['proposer_model'],'messages':copy.deepcopy(messages),
            'tools':tools,'tool_choice':'auto','max_tokens':100000,'stream':False}

def state_for(domain):
    return json.loads((HERE/domain/'state.json').read_text())

def parallel(callback,domains=DOMAINS):
    result,errors={},{}
    with ThreadPoolExecutor(max_workers=len(domains)) as pool:
        futures={pool.submit(callback,d):d for d in domains}
        for future in as_completed(futures):
            d=futures[future]
            try:result[d]=future.result()
            except Exception as exc:errors[d]={'type':type(exc).__name__,'detail':str(exc)[:6000]}
    if errors:
        atomic_json(HERE/'errors.json',errors)
        raise RuntimeError('Same-round repair failed: inspect errors.json; no automatic paid restart')
    return result

def audit_usage(ledger):
    usage=ledger.summary()
    if not usage['overall']['accounting_complete']:raise RuntimeError('Unknown or pending API spend; no blind paid retry')
    for domain in DOMAINS:
        prefix=len(json.loads((HERE/domain/'inheritance.json').read_text())['inherited_rounds'])
        for row in state_for(domain)['rounds'][prefix:]:
            key=f"{HERE.name}|{domain}|{row['round']}|life_optimizer"
            measured=usage['groups'].get(key)
            if measured is None:
                if row['proposer_requests']==0 and row['recovered_qualified'] and row['usage']['total_tokens']==0:continue
                raise RuntimeError('Missing nonzero-request API ledger')
            if not measured['accounting_complete']:raise RuntimeError('Incomplete round ledger')
            for account,validation in (('input_tokens','prompt_tokens'),('output_tokens','completion_tokens'),('total_tokens','total_tokens')):
                if measured[account]!=row['usage'][validation]:raise RuntimeError('API ledger/checkpoint mismatch')
    return usage

def retry_gate_valid(states,usage):
    if set(states)!=set(RETRY_DOMAINS) or not usage['overall']['accounting_complete']:return False
    for state in states.values():
        if state['round']<2 or len(state['rounds'])<2:return False
        row=state['rounds'][1]
        if (row['round']!=2 or row['outcome']!='evaluated_candidate' or row['candidate_successes'] is None
            or row['incumbent_infra_errors']>=50 or row['candidate_infra_errors']>=50):return False
        if not (row['proposer_requests']>0 and row['usage']['total_tokens']>0 or
                row.get('recovered_qualified') and row['proposer_requests']==row['usage']['total_tokens']==0):
            return False
    return True

def publish(ledger,gate=None,broker=None):
    usage=ledger.publish()
    ancestor=json.loads((HERE/'ancestor_usage.json').read_text())['usage']['overall']
    overall=usage['overall']
    atomic_json(HERE/'TOTAL_SPEND.json',{'unit':'API input plus output tokens, cached input included once',
        'ancestor':ancestor,'new_attempts':overall,
        'total_tokens':ancestor['total_tokens']+overall['total_tokens'],
        'input_tokens':ancestor['input_tokens']+overall['input_tokens'],
        'output_tokens':ancestor['output_tokens']+overall['output_tokens'],
        'accounting_complete':ancestor['accounting_complete'] and overall['accounting_complete'],
        'includes_failed_and_abandoned_attempts':True})
    if gate is not None:atomic_json(HERE/'flash_metrics.json',{'gate':gate.snapshot(),
        'rate_limit_responses':broker.rate_limit_responses,'usage':overall})
    render_training_report()
    return usage

def offline_preflight():
    validate_protocol();preservation_check()
    from driver import run_fingerprint,verify_history
    from protocol import bind_fingerprint
    from agentbench_evaluate import validate_cached_summary
    for domain in DOMAINS:
        bind_fingerprint(HERE/domain/'run_config.json',run_fingerprint(domain))
        verify_history(domain,state_for(domain))
        source=HERE.parent/'life_agentbench_six_flash_tokens_20261004'/domain/'evals/setup_smoke'
        summary=json.loads((source/'summary.json').read_text())
        validate_cached_summary(source,summary)
        if summary['n']!=2 or summary['infra_errors']:raise RuntimeError('Previously passing native smoke unavailable')
        bind_fingerprint(HERE/domain/'smoke_reuse.json',{'source':str(source),
            'summary_sha256':digest(source/'summary.json'),'n':2,'infra_errors':0})
    return {'status':'offline_preflight_passed','rounds':ROUNDS,'flash_output':100000,
            'flash_concurrency':8,'inherited_rounds':{d:state_for(d)['round'] for d in DOMAINS}}

def main():
    with (HERE/'.runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        offline_preflight();cfg=protocol()
        if (HERE/'FINAL_RESULTS.json').exists():verify_seal();return
        sys.path.insert(0,str(ROOT.parent/'ahe-native'))
        from ahe_native.metered_broker import Broker,RequestGate
        from ahe_native.token_accounting import TokenLedger,scoped_url
        from ahe_native.connection import credentials
        if not TokenLedger(HERE/'flash_requests.jsonl').summary()['overall']['accounting_complete']:
            raise RuntimeError('Prior unsettled repair request; refuse blind retry')
        key,url=credentials()
        if url!=cfg['proposer_host']+'/protocol/openai/v1':raise RuntimeError('Unexpected API host')
        gate=RequestGate(8,min_interval=cfg['flash_min_interval_seconds'])
        broker=Broker(key,url,gate,rate_limit_retries=2,ledger_path=HERE/'flash_requests.jsonl')
        secret=key.encode()
        if any(secret in p.read_bytes() for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts):
            broker.close();raise RuntimeError('Credential found on disk; startup refused')
        del key,secret
        atomic_json(HERE/'process.json',{'pid':os.getpid(),'pgid':os.getpgid(0),'experiment':HERE.name,
            'flash_concurrency_limit':8,'flash_max_tokens':100000,'flash_local_context_limit':None,
            'solver_workers_per_domain':50,'solver_max_context_tokens':32768,'solver_max_tokens':4096,
            'credential_policy':'existing broker memory only; no key in files or native child envs'})

        def completion(domain):
            log_path=HERE/domain/'flash_requests.jsonl'
            number=len(log_path.read_text().splitlines()) if log_path.exists() else 0
            def call(messages,tools):
                nonlocal number
                round_number=state_for(domain)['round']+1
                if not 1<=round_number<=6 or (HERE/'training_seal.json').exists():raise RuntimeError('API call outside unsealed TRAIN')
                number+=1;payload=build_payload(cfg,messages,tools)
                metadata={'round':round_number,'request_number':number,'model':payload['model'],
                    'max_tokens':100000,'local_context_limit':None,'message_count':len(messages)}
                endpoint=scoped_url(broker.url,experiment=HERE.name,domain=domain,round=round_number,role='life_optimizer')
                progress(domain,phase='flash_request',**metadata,gate=gate.snapshot())
                request=urllib.request.Request(endpoint+'/chat/completions',method='POST',
                    headers={'Authorization':'Bearer EMPTY','Content-Type':'application/json'},
                    data=json.dumps(payload,ensure_ascii=False).encode())
                opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with opener.open(request,timeout=2100) as response:result=json.load(response)
                with log_path.open('a') as log:
                    log.write(json.dumps({**metadata,'usage':result.get('usage',{})})+'\n')
                    log.flush();os.fsync(log.fileno())
                verify_usage(result)
                if not result.get('choices') or result.get('error'):raise RuntimeError('Missing assistant choice')
                return result
            return call

        calls={d:completion(d) for d in DOMAINS}
        stopped=threading.Event()
        def monitor():
            while not stopped.wait(30):
                try:publish(broker.ledger,gate,broker)
                except Exception as exc:
                    atomic_json(HERE/'monitor_error.json',{'type':type(exc).__name__,'detail':str(exc)[:2000]})
        thread=threading.Thread(target=monitor,daemon=True);thread.start()
        try:
            if not (HERE/'training_seal.json').exists():
                atomic_json(HERE/'status.json',{'phase':'retry_failed_second_rounds','pid':os.getpid()})
                states=parallel(lambda d:run_domain(d,calls[d],stop_after=2),RETRY_DOMAINS)
                usage=audit_usage(broker.ledger);passed=retry_gate_valid(states,usage)
                atomic_json(HERE/'retry_gate.json',{'passed':passed,'rule':cfg['retry_gate'],
                    'new_usage':usage['overall'],'domains':{d:states[d]['rounds'][1] for d in RETRY_DOMAINS}})
                if not passed:raise RuntimeError('Retry gate failed; later rounds not started')
                atomic_json(HERE/'status.json',{'phase':'remaining_qualified_training','pid':os.getpid()})
                parallel(lambda d:run_domain(d,calls[d]))
                audit_usage(broker.ledger)
        finally:
            stopped.set();thread.join(timeout=40);broker.close()
            publish(broker.ledger,gate,broker);preservation_check()
        seal_training()
        atomic_json(HERE/'status.json',{'phase':'final_test','pid':os.getpid()})
        ledger_sha=digest(HERE/'flash_requests.jsonl')
        measured=parallel(lambda d:phase(d,'test','final_test',write_active_chain(HERE/d,state_for(d),'final_test'),0))
        if digest(HERE/'flash_requests.jsonl')!=ledger_sha:raise RuntimeError('Optimizer request after seal')
        atomic_json(HERE/'FINAL_RESULTS.json',{'split':'test','domains':{d:r[0] for d,r in measured.items()},
            'checkpoint_policy':'final retained strict TRAIN incumbent','prior_test_exposure':True,
            'test_used_for_optimization':False,'new_optimizer_usage':broker.ledger.summary(),
            'total_spend':json.loads((HERE/'TOTAL_SPEND.json').read_text())})
        preservation_check();atomic_json(HERE/'status.json',{'phase':'complete','pid':os.getpid()})

if __name__=='__main__':
    if sys.argv[1:]==['--preflight']:
        print(json.dumps(offline_preflight()))
    elif sys.argv[1:]==['--launch']:
        with (HERE/'runner.log').open('a') as log:
            child=subprocess.Popen([sys.executable,'-B','-u',str(HERE/'runner.py')],cwd=HERE,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        atomic_json(HERE/'launch.json',{'pid':child.pid,'started_at':time.time()})
        print(json.dumps({'launched_pid':child.pid,'log':str(HERE/'runner.log')}))
    elif not sys.argv[1:]:
        try:main()
        except Exception as exc:
            atomic_json(HERE/'status.json',{'phase':'stopped','pid':os.getpid(),'error':f'{type(exc).__name__}: {exc}'})
            raise
    else:raise SystemExit('usage: runner.py [--launch|--preflight]')

