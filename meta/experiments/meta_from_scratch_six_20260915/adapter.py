"""Six-round identity-starting runs through the cloned original Meta-Harness loop."""
from pathlib import Path
import argparse, ast, copy, hashlib, importlib.util, json, os, re, shutil, subprocess, sys, time, types

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];AB=ROOT/'AgentBench';TAU=ROOT/'TauBench';UP=HERE/'upstream_reference'
sys.path.insert(0,str(HERE));import agentbench_evaluate
MANIFEST=json.loads((HERE/'manifest.json').read_text());TAU_DOMAINS=('airline','retail','telecom');AB_DOMAINS=('alfworld','dbbench','webshop')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,default=str,ensure_ascii=False)+'\n')

def freeze():
 for name,h in MANIFEST['upstream']['files'].items():assert digest(UP/name)==h,'Original Meta-Harness changed: '+name
 for rel,h in MANIFEST['snapshot_hashes'].items():assert digest(HERE/'task_snapshot'/rel)==h,'Task snapshot changed: '+rel
 for rel,h in MANIFEST['data_hashes'].items():assert digest(AB/rel)==h,'AgentBench data changed: '+rel
 for d in (*TAU_DOMAINS,*AB_DOMAINS):assert digest(HERE/d/'workspace/agents/baseline_zero.py')==MANIFEST['zero_sha256'],'Identity baseline changed'
 files=[HERE/n for n in ('adapter.py','agentbench_evaluate.py','tau_evaluate.py','prepare.py','manifest.json')]
 expected={str(p.relative_to(HERE)):digest(p) for p in files};p=HERE/'frozen.json'
 if p.exists():assert json.loads(p.read_text())==expected,'Frozen experiment driver changed'
 else:save(p,expected)

def upstream():
 sys.path.insert(0,str(UP));saved=dict(os.environ)
 if 'dotenv' not in sys.modules:
  try:import dotenv  # noqa: F401
  except ModuleNotFoundError:
   shim=types.ModuleType('dotenv');shim.load_dotenv=lambda *a,**k: False;sys.modules['dotenv']=shim
 spec=importlib.util.spec_from_file_location('original_meta_harness',UP/'meta_harness.py');u=importlib.util.module_from_spec(spec);spec.loader.exec_module(u)
 os.environ.clear();os.environ.update(saved)
 assert all(getattr(u,n).__code__.co_filename==str(UP/'meta_harness.py') for n in ('run_evolve','compute_pass_rates','update_frontier','update_evolution_summary'))
 return u

def score_value(summary):return float(summary['accuracy'])
def public_export(domain,phase,workspace):
 src=HERE/domain/'evals'/phase;summary_file=src/('val.json' if domain in TAU_DOMAINS else 'summary.json');summary=json.loads(summary_file.read_text());dest=workspace/'evidence'/phase;dest.mkdir(parents=True,exist_ok=True)
 if domain in TAU_DOMAINS:
  save(dest/'summary.json',{k:v for k,v in summary.items() if k not in ('save_dir','cmd')})
  raw=json.loads((Path(summary['save_dir'])/'results.json').read_text())
  sims=[]
  for x in raw.get('simulations',[]):
   sims.append({k:copy.deepcopy(x.get(k)) for k in ('task_id','trial','reward_info','termination_reason','messages','agent_system_prompt','agent_tools_schema')})
  save(dest/'public_trajectories.json',sims)
 else:
  save(dest/'summary.json',{k:v for k,v in summary.items() if k!='ids'})
  for task in summary['per_task']:
   row=json.loads((src/f'episode_{task}.json').read_text());history=[{k:m[k] for k in ('role','content','tool_calls','tool_call_id','name') if k in m} for m in row['history'] if isinstance(m,dict) and m.get('role') in ('system','user','assistant','tool') and not str(m.get('content','')).startswith('[HARNESS_TRACE_V1]')]
   calls=[]
   for c in row['calls']:
    response=c.get('response',{}).get('choices',[{}])[0].get('message',{})
    calls.append({'turn':c['turn'],'messages':c['request']['messages'],'tools':c['request']['tools'],'response':{k:response[k] for k in ('role','content','tool_calls') if k in response}})
   save(dest/f'episode_{task}.json',{'task_id':task,'success':row['success'],'status':row['status'],'turns':row['turns'],'tokens':row['tokens'],'history':history,'model_calls':calls,'candidate_hash':summary['plugin_hash']})
 return summary

def phase(domain,split,batch,name,candidate):
 run=HERE/domain;family=MANIFEST['domains'][domain]['family']
 if family=='tau':
  python=sys.executable;cmd=[python,'-u',str(HERE/'tau_evaluate.py'),domain,split,name,str(candidate),'--batch',str(batch)]
 else:
  py={'alfworld':AB/'.native/alfworld-venv/bin/python','dbbench':AB/'.native/core/bin/python','webshop':AB/'.native/webshop-venv/bin/python'}[domain]
  cmd=[str(py),'-u',str(HERE/'agentbench_evaluate.py'),domain,split,name,str(candidate),'--batch',str(batch)]
 env={**os.environ,'META_AGENTBENCH_BATCH':str(batch)}
 env['PATH']='/mnt/data/xts/tools/uv/bin:'+env.get('PATH','')
 if domain=='webshop':
  java=AB/'.native/sysroot/usr/lib/jvm/java-11-openjdk-amd64';env.update(JAVA_HOME=str(java),PATH=str(java/'bin')+':'+env.get('PATH',''),_JAVA_OPTIONS='-Xmx4g',PYTHONPATH=str(AB/'.native/webshop-src')+':'+str(AB))
 with (run/f'{name}.log').open('a') as log:subprocess.run(cmd,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=21600)
 p=run/'evals'/name/('val.json' if domain in TAU_DOMAINS and split=='train' else 'test.json' if domain in TAU_DOMAINS else 'summary.json')
 summary=json.loads(p.read_text());expected=len(MANIFEST['domains'][domain]['batches'][batch].get('task_ids',MANIFEST['domains'][domain]['batches'][batch].get('indices',[]))) if split=='train' else len(MANIFEST['domains'][domain].get('test_task_ids',MANIFEST['domains'][domain].get('test_indices',[])))
 if domain in AB_DOMAINS:assert summary['n']==expected and summary['infra_errors']==0
 else:assert len(summary['per_task'])==expected
 return summary

def parse(job,expected_trials=None):
 p=job/'summary.json'
 if not p.exists():return {}
 data=json.loads(p.read_text());return {k:(v if isinstance(v,list) else [v]) for k,v in data['per_task'].items()}
def metrics(job):return {}

def port_prompt(domain,iteration,batch,workspace,incumbent):
 spec=MANIFEST['domains'][domain];family=spec['family'];n=len(spec['batches'][batch].get('task_ids',spec['batches'][batch].get('indices',[])))
 api=("AgentHarness(public_context=None) with transform_request(request_dict) and transform_response(assistant_dict, public_history). Both return ordinary OpenAI-compatible dictionaries. Optional register()." if family=='agentbench' else "A self-contained importable plugin exporting AgentHarness plus register(). register() may monkeypatch the PLAIN tau2 domain Tools/environment classes; all released harness flags stay disabled.")
 return f'''\nPORT CONTRACT (overrides every Terminal-Bench/KIRA/Claude-specific statement above).\nActual task: {domain}; round {iteration}/6; training sample {batch}; {n} public TRAIN questions; fixed qwen3-4b solver, temperature 0, max tokens fixed by host.\nThis run starts from agents/baseline_zero.py, an identity implementation. Never read, copy, import, or reconstruct released Life-Harness H2/H3/H4/H5.\nCurrent retained implementation: agents/{incumbent}. Build only from that file and public evidence. The search space is free-form Meta-Harness Python and has no four-layer design constraint.\nRuntime API: {api}\nThe host freezes model, task/environment, tools executed by the benchmark, budgets, sampling, and scoring. Runtime code may transform public prompts/tool schemas/model replies, or patch public tool execution only for Tau.\nNo new model/network/subprocess/filesystem calls at runtime; no labels, expected answers, evaluator objects, hidden state, task IDs, instance-specific solution tables, or cross-episode mutable state.\nAllowed candidate imports: stdlib re/json/copy/math/collections/difflib/string/typing/ast/dataclasses/functools/itertools/statistics; Tau candidates may additionally import tau2.\nOne focused mechanism. Analyze both failures and passing counterexamples. Test results are never present here and must not influence proposals.\nWorkspace boundary: {workspace}. Read only agents/, source_view/, evidence/, logs/, jobs/, history.json, upstream_SKILL.md. Do not read parent directories or other experiments.\nUse the original skill's ONE analysis Agent and ONE implementation Agent, with this entire port contract included in both delegated prompts. Do not run benchmarks. Local pure import/prototype checks only.\nWrite one new agents/<lower_snake_case>.py and pending_eval.json with exactly one candidate: name, import_path="agents.<name>:AgentHarness", hypothesis, changes, expected_efficiency.\n'''

def validate_candidate(path,domain):
 text=path.read_text();tree=ast.parse(text)
 forbidden=('h2345','published_h','Life-Harness','tasks.json','split_tasks','simulations','subprocess','requests','urllib','socket','os.environ','open(')
 hit=next((x for x in forbidden if x in text),None)
 if hit:raise ValueError('forbidden candidate reference '+hit)
 if domain in AB_DOMAINS:agentbench_evaluate.validate_plugin(path)
 else:
  code=f"import importlib.util; s=importlib.util.spec_from_file_location('c',r'{path}');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);assert hasattr(m,'AgentHarness') and callable(getattr(m,'register',None));print('OK')"
  env={**os.environ,'UV_CACHE_DIR':'/tmp/uv-cache'};env['PATH']='/mnt/data/xts/tools/uv/bin:'+env.get('PATH','');p=subprocess.run(['uv','run','python','-c',code],cwd=TAU,env=env,capture_output=True,text=True,timeout=180)
  if p.returncode:raise ValueError(p.stderr[-1000:])

def native_round(domain,iteration,batch,incumbent,baseline_phase):
 run=HERE/domain;workspace=run/'workspace';u=upstream();u.EVOLVE_DIR=workspace;u.AGENTS_DIR=workspace/'agents';u.BASELINES=[('incumbent',f'agents.{Path(incumbent).stem}:AgentHarness')];u.BASELINE_AGENT_NAME='incumbent';u.N_EVAL_TASKS=len(MANIFEST['domains'][domain]['batches'][batch].get('task_ids',MANIFEST['domains'][domain]['batches'][batch].get('indices',[])));u.MODEL='qwen3-4b'
 runname=f'round_{iteration:02d}';baselinejob=workspace/'jobs'/runname/'incumbent-t1';baselinejob.mkdir(parents=True,exist_ok=True)
 baseline=json.loads((run/'evals'/baseline_phase/('val.json' if domain in TAU_DOMAINS else 'summary.json')).read_text());save(baselinejob/'summary.json',{'per_task':baseline.get('per_task_trials',baseline['per_task'])})
 renderer=u.render_task_prompt;u.render_task_prompt=lambda i,n:renderer(i,n)+'\n\n'+(workspace/'upstream_SKILL.md').read_text()+port_prompt(domain,iteration,batch,workspace,incumbent)
 def candidate_path(name,ip):
  assert re.fullmatch('[a-z][a-z0-9_]*',name) and ip==f'agents.{name}:AgentHarness';assert name not in ('baseline_zero',Path(incumbent).stem);return workspace/'agents'/f'{name}.py'
 def propose(prompt,i,timeout=5400):
  cached=run/f'pending_{iteration:02d}.json'
  if cached.exists():shutil.copy2(cached,u.PENDING_EVAL);return True
  (run/f'prompt_{iteration:02d}.md').write_text(prompt)
  frozen={str(p.relative_to(workspace)):digest(p) for p in workspace.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
  save(run/f'inputs_{iteration:02d}.json',frozen)
  tools=['Read','Write','Edit','Glob','Grep','Bash','Agent'];cmd=['qoder','-p',prompt,'-w',str(workspace),'-o','json','--permission-mode','accept_edits','--tools',*tools,'--disallowed-tools','Task','-m','DeepSeek-Flash']
  for t in tools:cmd+=['--allowed-tools',t]
  result=subprocess.run(cmd,cwd=workspace,capture_output=True,text=True,timeout=timeout);save(run/f'proposer_{iteration:02d}.json',{'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
  for rel,h in frozen.items():assert (workspace/rel).exists() and digest(workspace/rel)==h,'Proposer modified frozen input '+rel
  if result.returncode or not u.PENDING_EVAL.exists():raise RuntimeError('Proposer failed to create pending_eval.json')
  pending=json.loads(u.PENDING_EVAL.read_text());assert len(pending['candidates'])==1;c=pending['candidates'][0];assert candidate_path(c['name'],c['import_path']).exists();shutil.copy2(u.PENDING_EVAL,cached);return True
 u.propose_claude=propose
 def validate(name,ip):
  path=candidate_path(name,ip);validate_candidate(path,domain);save(run/f'preflight_{iteration:02d}.json',{'candidate_hash':digest(path),'identity_parent_hash':digest(workspace/'agents'/incumbent),'released_h_flags':[],'validation':'pass'});return True
 u.validate_candidate=validate
 def harbor(ip,job_name,n_trials=1,n_concurrent=8):
  name=ip.split(':')[0].split('.')[-1];path=candidate_path(name,ip);summary=phase(domain,'train',batch,f'candidate_{iteration:02d}',path);public_export(domain,f'candidate_{iteration:02d}',workspace);job=workspace/'jobs'/runname/f'{name}-native';save(job/'summary.json',{'per_task':summary.get('per_task_trials',summary['per_task'])});return job,True
 u.harbor_run=harbor;u.parse_job_results=parse;u.parse_trial_metrics=metrics
 summary=workspace/'logs'/runname/'evolution_summary.jsonl'
 if not summary.exists():u.run_evolve(argparse.Namespace(run_name=runname,iterations=1,trials=1,concurrent=8,skip_baseline=True,skip_smoke=True,full_eval=False,fresh=False,propose_timeout=5400))
 assert summary.exists();pending=json.loads((run/f'pending_{iteration:02d}.json').read_text());c=pending['candidates'][0];score=json.loads((run/'evals'/f'candidate_{iteration:02d}'/('val.json' if domain in TAU_DOMAINS else 'summary.json')).read_text());return c,score

def main(domain):
 freeze();run=HERE/domain;workspace=run/'workspace';statefile=run/'state.json';state=json.loads(statefile.read_text()) if statefile.exists() else {'status':'running','incumbent':'baseline_zero.py','batch':0,'rounds':[],'baseline_phase':None,'from_scratch':True,'released_h_flags':[]}
 assert state['from_scratch'] and state['released_h_flags']==[]
 for iteration in range(len(state['rounds'])+1,7):
  freeze();batch=state['batch']
  if state['baseline_phase'] is None:
   name=f'batch_{batch:02d}_baseline';base=phase(domain,'train',batch,name,workspace/'agents'/state['incumbent']);public_export(domain,name,workspace);state['baseline_phase']=name;save(statefile,state)
  else:base=json.loads((run/'evals'/state['baseline_phase']/('val.json' if domain in TAU_DOMAINS else 'summary.json')).read_text())
  # Test fields are deliberately excluded from proposer-visible history.
  save(workspace/'history.json',[{k:v for k,v in row.items() if not k.startswith('test_')} for row in state['rounds']])
  prior_incumbent=state['incumbent']
  candidate,score=native_round(domain,iteration,batch,prior_incumbent,state['baseline_phase']);assert set(base['per_task'])==set(score['per_task'])
  accepted=score_value(score)>score_value(base)
  if accepted:state.update(incumbent=candidate['name']+'.py',baseline_phase=f'candidate_{iteration:02d}')
  elif iteration<6:state.update(batch=min(batch+1,5),baseline_phase=None)
  test_phase=f'test_round_{iteration:02d}';test=phase(domain,'test',batch,test_phase,workspace/'agents'/state['incumbent'])
  old_ids=MANIFEST['domains'][domain]['batches'][batch].get('task_ids',MANIFEST['domains'][domain]['batches'][batch].get('indices',[]));new_ids=MANIFEST['domains'][domain]['batches'][state['batch']].get('task_ids',MANIFEST['domains'][domain]['batches'][state['batch']].get('indices',[]))
  row={'iteration':iteration,'train_batch':batch,'train_n':len(base['per_task']),'baseline_candidate':prior_incumbent,'baseline_score':score_value(base),'candidate':candidate,'candidate_score':score_value(score),'accepted':accepted,'wins':sum(score['per_task'][k]>v for k,v in base['per_task'].items()),'losses':sum(score['per_task'][k]<v for k,v in base['per_task'].items()),'next_batch_changes':not accepted and iteration<6 and old_ids!=new_ids,'retained_candidate':state['incumbent'],'test_phase':test_phase,'test_n':len(test['per_task']),'test_score':score_value(test),'test_successes':test.get('successes'),'released_h_flags':[]}
  state['rounds'].append(row);save(statefile,state);print(json.dumps(row,ensure_ascii=False),flush=True)
 state['status']='complete';save(statefile,state)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('domain',choices=[*TAU_DOMAINS,*AB_DOMAINS]);a=p.parse_args();main(a.domain)
