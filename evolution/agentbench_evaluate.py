"""Unified current Harness interface; published rules migrated into the plugin."""
from pathlib import Path
import argparse,ast,asyncio,copy,hashlib,importlib.util,json,os,sys,time,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
from types import SimpleNamespace
import multiprocessing
HERE=Path(__file__).resolve().parent; AB=Path(__file__).resolve().parents[1]/'AgentBench'
if os.environ.get('LIFE_AGENTBENCH_RUN'):
 HERE=Path(os.environ['LIFE_AGENTBENCH_RUN']).resolve()
MANIFEST=json.loads((HERE/'manifest.json').read_text())
TASK=None; PLUGIN=None; DOMAIN=None

def save(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,default=str)+'\n');tmp.replace(path)
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def validate_plugin(path):
 tree=ast.parse(Path(path).read_text());allowed={'re','json','copy','math','collections','difflib','string','typing','ast','dataclasses'}
 for n in ast.walk(tree):
  if isinstance(n,(ast.Import,ast.ImportFrom)):
   names=[a.name for a in n.names] if isinstance(n,ast.Import) else [n.module or '']
   if any(x.split('.')[0] not in allowed for x in names):raise ValueError('Plugin import not allowed: '+str(names))
  if isinstance(n,ast.Name) and n.id in {'open','eval','exec','compile','__import__','globals','locals','getattr','setattr','vars','input','breakpoint'}:raise ValueError('Forbidden plugin operation: '+n.id)
  if isinstance(n,ast.Attribute) and n.attr.startswith('__'):raise ValueError('Dunder access forbidden')
 spec=importlib.util.spec_from_file_location('candidate',path);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
 for k in ['h2','h3','h4','h5']:
  if not callable(getattr(m.Harness(),k,None)):raise ValueError('Missing hook '+k)
 return m

def initialize(domain,plugin):
 global TASK,PLUGIN,DOMAIN
 os.chdir(AB);sys.path.insert(0,str(HERE/'task_snapshot'));sys.path.insert(0,str(AB))
 # Snapshot must win over mutable checkout.
 sys.path.remove(str(HERE/'task_snapshot'));sys.path.insert(0,str(HERE/'task_snapshot'))
 import yaml
 sys.path.insert(0,str(HERE))
 if not MANIFEST.get('current_task_snapshot',False):
  from legacy_disabled import install
  install()
 DOMAIN=domain;PLUGIN=validate_plugin(plugin)
 cfg=yaml.safe_load((HERE/f'task_snapshot/configs/tasks/{domain}.yaml').read_text())['default']['parameters']
 cfg.update(name=domain+'-zero-train',concurrency=1)
 if domain=='alfworld':
  from src.server.tasks.alfworld.task import ALFWorld
  cfg.update(data_path=str(AB/'data/alfworld'),config_path=str(HERE/'task_snapshot/src/server/tasks/alfworld/configs/base_config.yaml'),prompts_path=str(HERE/'task_snapshot/src/server/tasks/alfworld/prompts/alfworld_multiturn_plan_first.json'),split='train_valid',enabled=False,h2=False,h3=False,h4=False,h5=False)
  TASK=ALFWorld(**cfg)
 else:
  from src.server.tasks.dbbench.task import DBBenchTask
  cfg.update(data_file=str(AB/'data/dbbench/db_out_new.jsonl'),env_driver='native_mysql',env_options={'host':'127.0.0.1','port':13306},db_password='',harness={'enabled':False,'h2':False,'h3':False,'h4':False,'h5':False})
  TASK=DBBenchTask(**cfg)
 assert not TASK.harness_config.enabled

def public_task_context(index):
 if DOMAIN!='dbbench':return None
 entry=TASK.dataset[index][0]
 return {key:copy.deepcopy(entry[key]) for key in ('description','type','table','evidence','add_description') if key in entry}

class EpisodeSession:
 def __init__(self,index):
  self.history=[];self.trace=[];self.calls=[];self.rewards=[];self.turn=0;self.harness=PLUGIN.Harness();self._h2_suppressed=False
  bind=getattr(self.harness,'bind_task_context',None)
  if callable(bind):bind(public_task_context(index))
  self.h2_prevalidated_actions=True
  self.tools=copy.deepcopy(TASK.tools);before=copy.deepcopy(self.tools);self.tools=self.harness.h3(self.tools)
  def skeleton(tools):
   t=copy.deepcopy(tools)
   for item in t:item['function'].pop('description',None)
   return t
  assert skeleton(before)==skeleton(self.tools),'H3 may edit descriptions only'
  if before!=self.tools:self.trace.append({'layer':'h3','before':before,'after':self.tools})
  self.cold=[];self.limit=50 if DOMAIN=='alfworld' else 15
 def inject(self,item):
  if isinstance(item,list):
   for x in item:self.inject(x)
  elif isinstance(item,dict) and 'role' in item:
   before=copy.deepcopy(item);patch=getattr(self.harness,'h3_message',None)
   if callable(patch):
    item=patch(copy.deepcopy(item))
    assert isinstance(item,dict) and item.get('role')==before.get('role'),'H3 message conditioning must preserve role'
    assert {k:v for k,v in item.items() if k!='content'}=={k:v for k,v in before.items() if k!='content'},'H3 message conditioning may change content only'
    if item!=before:self.trace.append({'layer':'h3','message_before':before,'message_after':copy.deepcopy(item)})
   self.history.append(copy.deepcopy(item))
  elif hasattr(item,'reward'):self.rewards.append(item.model_dump(mode='json'))
 def hints(self,layer,values):
  assert isinstance(values,list) and len(values)<=1 and all(isinstance(x,str) and len(x.split())<=120 for x in values),'At most one <=120 word hint per hook'
  if values:self.trace.append({'turn':self.turn,'layer':layer,'hints':values})
  return values
 def sync_action(self):
  import requests
  self.turn+=1;self._h2_suppressed=False
  if self.turn==1:self.cold=self.hints('h5',self.harness.h5(copy.deepcopy(self.history)))
  public_history=copy.deepcopy(self.history)
  feedback=[]
  if self.turn>1:
   feedback=self.hints('h4',self.harness.h4(copy.deepcopy(self.history),self.limit-self.turn+1))
   if feedback:
    item={'role':'user','content':getattr(self.harness,'h4_message_prefix','Execution guidance: ')+feedback[0]}
    if getattr(self.harness,'h4_persistent',False):
     self.history.append(copy.deepcopy(item))
  payload_history=copy.deepcopy(self.history)
  if self.cold:payload_history[0]['content']+=getattr(self.harness,'h5_message_prefix','\n\nProcedural guidance:\n')+self.cold[0]
  if feedback and not getattr(self.harness,'h4_persistent',False):payload_history.append(item)
  fixed=MANIFEST['agent'];body={k:v for k,v in fixed.items() if k!='url'}
  body.update(messages=payload_history,tools=self.tools,tool_choice='auto',parallel_tool_calls=False)
  call={'turn':self.turn,'request':body};self.calls.append(call)
  for attempt in range(3):
   try:
    response=requests.post(fixed['url'],headers={'Authorization':'Bearer EMPTY'},json=body,timeout=180)
    response.raise_for_status();data=response.json();break
   except Exception as exc:
    call.setdefault('transport_errors',[]).append(str(exc))
    if attempt==2:raise
    time.sleep(2*(attempt+1))
  raw=data['choices'][0]['message'];raw={k:raw[k] for k in ['role','content','tool_calls'] if k in raw and raw[k] is not None}
  raw.setdefault('role','assistant')
  call.update(response=data,usage=data.get('usage',{}))
  msg=self.harness.h2(copy.deepcopy(raw),public_history)
  assert isinstance(msg,dict) and msg.get('role')=='assistant'
  control=msg.pop('_harness_control',None)
  if msg!=raw:self.trace.append({'turn':self.turn,'layer':'h2','before':raw,'after':msg})
  if control:
   assert isinstance(control,dict) and control.get('action')=='suppress_tool','Invalid H2 host control'
   self.history.append(copy.deepcopy(raw))
   for item in control.get('messages',[]):
    assert isinstance(item,dict) and item.get('role') in ('user','tool'),'Invalid H2 control message'
    self.history.append(copy.deepcopy(item))
   self._h2_suppressed=True;return SimpleNamespace(messages=[])
  no_tool=getattr(self.harness,'h2_no_tool_message',None)
  if not (msg.get('tool_calls') or []) and callable(no_tool):
   nudge=no_tool(copy.deepcopy(raw),max(0,self.limit-self.turn+1))
   if nudge:
    self.history.append(copy.deepcopy(raw));self.history.append({'role':'user','content':str(nudge)})
    self._h2_suppressed=True;return SimpleNamespace(messages=[])
  self.history.append(copy.deepcopy(raw if getattr(self.harness,'h2_preserve_raw_history',False) else msg))
  drain=getattr(self.harness,'drain_h2_messages',None);side=drain() if callable(drain) else []
  assert isinstance(side,list),'H2 side messages must be a list'
  for item in side:
   if isinstance(item,str):item={'role':'user','content':item}
   assert isinstance(item,dict) and item.get('role')=='user' and isinstance(item.get('content'),str),'Invalid H2 side message'
   self.history.append(copy.deepcopy(item))
  return SimpleNamespace(messages=[msg])
 async def action(self):return self.sync_action()
 def consume_h2_suppressed(self):
  value=self._h2_suppressed;self._h2_suppressed=False;return value

async def run_db(index,session):
 try:return await TASK.start_sample(index,session)
 finally:
  if TASK.env_controller_background_task:
   TASK.env_controller_background_task.cancel();TASK.env_controller_background_task=None

def episode(index):
 start=time.time();session=EpisodeSession(index)
 try:
  result=TASK.sync_start_sample(index,session) if DOMAIN=='alfworld' else asyncio.run(run_db(index,session))
  status=result.status.value;raw=result.result or {}
  success=bool(raw.get('result')==1) if DOMAIN=='alfworld' else bool(raw.get('is_correct',False))
  infra=status.replace(' ', '_') in ['task_error','cancelled','unknown','running'] or any('transport_errors' in x and 'response' not in x for x in session.calls)
  output={'index':index,'success':int(success),'status':status,'infrastructure_error':infra,'task_result_private':raw}
 except Exception:
  output={'index':index,'success':0,'status':'infrastructure_error','infrastructure_error':True,'exception':traceback.format_exc()}
 output.update(wall_seconds=time.time()-start,turns=session.turn,history=session.history,calls=session.calls,harness_trace=session.trace,rewards=session.rewards,
               tokens=sum(c.get('usage',{}).get('total_tokens',0) for c in session.calls))
 return output

def evaluate(domain,phase,plugin,indices_file=None,heldout=False):
 plugin=Path(plugin).resolve();validate_plugin(plugin)
 for relative,expected in MANIFEST['hashes'].items():assert digest(HERE/'task_snapshot'/relative)==expected,'Frozen source changed: '+relative
 for relative,expected in MANIFEST['data_hashes'].items():assert digest(AB/relative)==expected,'Frozen data changed: '+relative
 out=HERE/domain/'evals'/phase;out.mkdir(parents=True,exist_ok=True)
 config={'domain':domain,'phase':phase,'plugin_hash':digest(plugin),'manifest_hash':digest(HERE/'manifest.json'),'runner_hash':digest(__file__),'pool_indices':MANIFEST[domain]['heldout_indices' if heldout else 'indices'], 'heldout':heldout}
 meta=out/'config.json'
 if meta.exists():assert json.loads(meta.read_text())==config,'Resume configuration mismatch'
 else:save(meta,config)
 ids=MANIFEST[domain]['heldout_indices' if heldout else 'indices']
 assert ids, 'Empty evaluation pool'
 assert not (heldout and indices_file), 'Heldout screening is forbidden'
 if indices_file:
  ids=json.loads(Path(indices_file).read_text())
  assert ids and len(ids)==len(set(ids)) and set(ids)<=set(MANIFEST[domain]['indices']),'Invalid screening subset'
 pending=[]
 for i in ids:
  p=out/f'episode_{i}.json'
  if not p.exists() or json.loads(p.read_text())['infrastructure_error']:pending.append(i)
 with ProcessPoolExecutor(max_workers=MANIFEST['concurrency_per_dataset'],mp_context=multiprocessing.get_context('spawn'),initializer=initialize,initargs=(domain,str(plugin))) as pool:
  futures={pool.submit(episode,i):i for i in pending}
  for f in as_completed(futures):
   r=f.result();p=out/f'episode_{r["index"]}.json'
   if p.exists():p.rename(out/f'episode_{r["index"]}.previous_{int(time.time())}.json')
   save(p,r);print(json.dumps({'domain':domain,'phase':phase,'index':r['index'],'success':r['success'],'status':r['status'],'turns':r['turns']}),flush=True)
 rows=[json.loads((out/f'episode_{i}.json').read_text()) for i in ids]
 summary={**config,'n':len(rows),'successes':sum(x['success'] for x in rows),'accuracy':sum(x['success'] for x in rows)/len(rows),'infra_errors':sum(x['infrastructure_error'] for x in rows),'tokens':sum(x['tokens'] for x in rows),'turns':sum(x['turns'] for x in rows),'per_task':{str(x['index']):x['success'] for x in rows}}
 save(out/('screen_summary.json' if indices_file else 'summary.json'),summary);print(json.dumps(summary),flush=True)
 if summary['infra_errors']:raise RuntimeError('Infrastructure errors; not a valid scored comparison')
 return summary
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('domain',choices=['alfworld','dbbench']);p.add_argument('phase');p.add_argument('plugin');p.add_argument('--indices-file');p.add_argument('--heldout',action='store_true')
 a=p.parse_args();evaluate(a.domain,a.phase,a.plugin,a.indices_file,a.heldout)
