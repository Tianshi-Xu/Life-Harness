"""Native AgentBench evaluator for a free-form identity-starting AgentHarness."""
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse, ast, asyncio, copy, hashlib, importlib.util, json, multiprocessing, os, sys, time, traceback

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];AB=ROOT/'AgentBench'
MANIFEST=json.loads((HERE/'manifest.json').read_text());TASK=None;PLUGIN=None;DOMAIN=None;SPLIT=None

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,default=str,ensure_ascii=False)+'\n');tmp.replace(path)

def validate_plugin(path):
 tree=ast.parse(Path(path).read_text());allowed={'re','json','copy','math','collections','difflib','string','typing','ast','dataclasses','functools','itertools','statistics'}
 for n in ast.walk(tree):
  if isinstance(n,(ast.Import,ast.ImportFrom)):
   names=[a.name for a in n.names] if isinstance(n,ast.Import) else [n.module or '']
   if any(x.split('.')[0] not in allowed for x in names):raise ValueError('Plugin import not allowed: '+str(names))
  if isinstance(n,ast.Name) and n.id in {'open','eval','exec','compile','__import__','globals','locals','vars','input','breakpoint'}:raise ValueError('Forbidden plugin operation: '+n.id)
  if isinstance(n,ast.Attribute) and n.attr.startswith('__'):raise ValueError('Dunder access forbidden')
 spec=importlib.util.spec_from_file_location('candidate_'+digest(path)[:12],path);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
 obj=m.AgentHarness(None)
 for name in ('transform_request','transform_response'):
  if not callable(getattr(obj,name,None)):raise ValueError('Missing runtime method '+name)
 return m

def initialize(domain,split,plugin):
 global TASK,PLUGIN,DOMAIN,SPLIT
 DOMAIN=domain;SPLIT=split;os.chdir(AB)
 if domain=='webshop':
  ws=AB/'.native/webshop-src';sys.path.insert(0,str(ws))
 snap=HERE/'task_snapshot';sys.path.insert(0,str(AB));sys.path.insert(0,str(snap))
 import yaml
 PLUGIN=validate_plugin(plugin)
 cfg=yaml.safe_load((snap/f'configs/tasks/{domain}.yaml').read_text())['default']['parameters'];cfg.update(name=f'{domain}-meta-{split}',concurrency=1)
 if domain=='alfworld':
  from src.server.tasks.alfworld.task import ALFWorld
  cfg.update(data_path=str(AB/'data/alfworld'),config_path=str(snap/'src/server/tasks/alfworld/configs/base_config.yaml'),prompts_path=str(snap/'src/server/tasks/alfworld/prompts/alfworld_multiturn_plan_first.json'),split='train_valid' if split=='train' else 'new_std',enabled=False,h2=False,h3=False,h4=False,h5=False)
  TASK=ALFWorld(**cfg)
 elif domain=='dbbench':
  from src.server.tasks.dbbench.task import DBBenchTask
  cfg.update(data_file=str(AB/f"data/dbbench/{'db_out_new.jsonl' if split=='train' else 'standard.jsonl'}"),env_driver='native_mysql',env_options={'host':'127.0.0.1','port':13306},db_password='',harness={'enabled':False,'h2':False,'h3':False,'h4':False,'h5':False})
  TASK=DBBenchTask(**cfg)
 else:
  from src.server.tasks.webshop.task import WebShop
  cfg.update(start=0,end=1021,enabled=False,h2=False,h3=False,h4=False,h5=False)
  TASK=WebShop(**cfg)
 assert not TASK.harness_config.enabled

def public_context(index):
 if DOMAIN=='dbbench':
  e=TASK.dataset[index][0];return {k:copy.deepcopy(e[k]) for k in ('description','type','table','evidence','add_description') if k in e}
 return None

class EpisodeSession:
 def __init__(self,index):
  self.history=[];self.calls=[];self.rewards=[];self.turn=0;self.tools=copy.deepcopy(TASK.tools);self.harness=PLUGIN.AgentHarness(public_context(index))
  self.controller=SimpleNamespace(env_output=SimpleNamespace(history=[]))
 def set_tools(self,tools):self.tools=copy.deepcopy(tools)
 def set_full_history(self,value):return None
 def cover(self,history):self.history=copy.deepcopy(history)
 def inject(self,item):
  if isinstance(item,list):
   for x in item:self.inject(x)
  elif isinstance(item,dict) and 'role' in item:self.history.append(copy.deepcopy(item))
  elif hasattr(item,'reward'):self.rewards.append(item.model_dump(mode='json'))
 def sync_action(self):
  import requests
  self.turn+=1;fixed=MANIFEST['agent'];base={k:v for k,v in fixed.items() if k!='url'}
  base.update(messages=copy.deepcopy(self.history),tools=copy.deepcopy(self.tools),tool_choice='auto',parallel_tool_calls=False)
  body=self.harness.transform_request(copy.deepcopy(base));assert isinstance(body,dict)
  # Solver identity and sampling parameters are experiment constants.
  for k in ('model','temperature','max_tokens','chat_template_kwargs'):body[k]=copy.deepcopy(base[k])
  assert isinstance(body.get('messages'),list) and isinstance(body.get('tools'),list)
  call={'turn':self.turn,'request':copy.deepcopy(body)};self.calls.append(call)
  for attempt in range(3):
   try:
    response=requests.post(fixed['url'],headers={'Authorization':'Bearer EMPTY'},json=body,timeout=240);response.raise_for_status();data=response.json();break
   except Exception as exc:
    call.setdefault('transport_errors',[]).append(str(exc))
    if attempt==2:raise
    time.sleep(2*(attempt+1))
  raw=data['choices'][0]['message'];raw={k:raw[k] for k in ('role','content','tool_calls') if k in raw and raw[k] is not None};raw.setdefault('role','assistant')
  msg=self.harness.transform_response(copy.deepcopy(raw),copy.deepcopy(self.history));assert isinstance(msg,dict) and msg.get('role')=='assistant'
  call.update(response=data,usage=data.get('usage',{}),transformed=copy.deepcopy(msg) if msg!=raw else None)
  self.history.append(copy.deepcopy(msg));return SimpleNamespace(messages=[msg])
 async def action(self):return self.sync_action()

async def run_db(index,session):
 try:return await TASK.start_sample(index,session)
 finally:
  if TASK.env_controller_background_task:TASK.env_controller_background_task.cancel();TASK.env_controller_background_task=None

def episode(index):
 start=time.time();session=EpisodeSession(index)
 try:
  result=TASK.sync_start_sample(index,session) if DOMAIN in ('alfworld','webshop') else asyncio.run(run_db(index,session))
  status=result.status.value;raw=result.result or {}
  if DOMAIN=='alfworld':success=bool(raw.get('result')==1)
  elif DOMAIN=='dbbench':success=bool(raw.get('is_correct',False))
  else:success=float(raw.get('reward',0) or 0)>=1.0
  infra=status.replace(' ','_') in ('task_error','cancelled','unknown','running') or any('transport_errors' in x and 'response' not in x for x in session.calls)
  out={'index':index,'success':int(success),'status':status,'infrastructure_error':infra,'task_result_private':raw}
 except Exception:
  out={'index':index,'success':0,'status':'infrastructure_error','infrastructure_error':True,'exception':traceback.format_exc()}
 out.update(wall_seconds=time.time()-start,turns=session.turn,history=session.history,calls=session.calls,rewards=session.rewards,tokens=sum(c.get('usage',{}).get('total_tokens',0) for c in session.calls))
 return out

def evaluate(domain,split,phase,plugin,batch=0):
 plugin=Path(plugin).resolve();validate_plugin(plugin);spec=MANIFEST['domains'][domain]
 for rel,h in MANIFEST['snapshot_hashes'].items():assert digest(HERE/'task_snapshot'/rel)==h,'Frozen source changed: '+rel
 for rel,h in MANIFEST['data_hashes'].items():
  if rel.startswith('data/'+domain) or (domain=='webshop' and rel.startswith('.native/webshop')):assert digest(AB/rel)==h,'Frozen data changed: '+rel
 ids=spec['batches'][batch]['indices'] if split=='train' else spec['test_indices']
 out=HERE/domain/'evals'/phase;out.mkdir(parents=True,exist_ok=True)
 config={'domain':domain,'split':split,'phase':phase,'batch':batch if split=='train' else None,'ids':ids,'plugin_hash':digest(plugin),'manifest_hash':digest(HERE/'manifest.json'),'runner_hash':digest(__file__),'concurrency':8}
 meta=out/'config.json'
 if meta.exists():assert json.loads(meta.read_text())==config,'Resume configuration mismatch'
 else:save(meta,config)
 pending=[]
 for i in ids:
  p=out/f'episode_{i}.json'
  if not p.exists() or json.loads(p.read_text()).get('infrastructure_error'):pending.append(i)
 ctx=multiprocessing.get_context('spawn')
 with ProcessPoolExecutor(max_workers=8,mp_context=ctx,initializer=initialize,initargs=(domain,split,str(plugin))) as pool:
  futures={pool.submit(episode,i):i for i in pending}
  for f in as_completed(futures):
   row=f.result();p=out/f'episode_{row["index"]}.json'
   if p.exists():p.rename(out/f'episode_{row["index"]}.previous_{int(time.time())}.json')
   save(p,row);print(json.dumps({'domain':domain,'split':split,'phase':phase,'index':row['index'],'success':row['success'],'status':row['status'],'turns':row['turns']}),flush=True)
 rows=[json.loads((out/f'episode_{i}.json').read_text()) for i in ids]
 summary={**config,'n':len(rows),'successes':sum(x['success'] for x in rows),'accuracy':sum(x['success'] for x in rows)/len(rows),'infra_errors':sum(x['infrastructure_error'] for x in rows),'tokens':sum(x['tokens'] for x in rows),'turns':sum(x['turns'] for x in rows),'per_task':{str(x['index']):x['success'] for x in rows}}
 save(out/'summary.json',summary);print(json.dumps(summary),flush=True)
 if summary['infra_errors']:raise RuntimeError(f"{summary['infra_errors']} infrastructure errors")
 return summary

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('domain',choices=['alfworld','dbbench','webshop']);p.add_argument('split',choices=['train','test']);p.add_argument('phase');p.add_argument('plugin');p.add_argument('--batch',type=int,default=0);a=p.parse_args();evaluate(a.domain,a.split,a.phase,a.plugin,a.batch)
