"""Run one frozen TauBench train/test phase with all released H layers off."""
from pathlib import Path
import argparse, hashlib, json, os, subprocess, sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];TAU=ROOT/'TauBench';BENCH=ROOT/'meta/benchmark.py'
MANIFEST=json.loads((HERE/'manifest.json').read_text())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def add_trial_rewards(score):
 from collections import defaultdict
 trials=defaultdict(list);results=Path(score['save_dir'])/'results.json'
 for sim in json.loads(results.read_text()).get('simulations',[]):
  trials[str(sim['task_id'])].append(float((sim.get('reward_info') or {}).get('reward',0)))
 score['per_task_trials']={k:v for k,v in trials.items()}
 return score

def main(domain,split,phase,plugin,batch):
 spec=MANIFEST['domains'][domain];plugin=Path(plugin).resolve()
 user_llm='openai/deepseek-v4-flash' if domain=='telecom' and batch>=2 else 'openai/deepseek-flash'
 ids=spec['batches'][batch]['task_ids'] if split=='train' else spec['test_task_ids']
 d=HERE/domain/'task_ids';d.mkdir(parents=True,exist_ok=True);id_file=d/(f'batch_{batch:02d}.txt' if split=='train' else 'test.txt')
 content='\n'.join(ids)+'\n'
 if id_file.exists():assert id_file.read_text()==content,'Frozen task IDs changed'
 else:id_file.write_text(content)
 out=HERE/domain/'evals'/phase/('val.json' if split=='train' else 'test.json')
 if out.exists():
  score=add_trial_rewards(json.loads(out.read_text()));out.write_text(json.dumps(score,indent=2,ensure_ascii=False)+'\n');print(out.read_text());return
 out.parent.mkdir(parents=True,exist_ok=True)
 trials=spec['train_trials'] if split=='train' else 1
 cmd=[sys.executable,str(BENCH),'--domain',domain,'--split','search' if split=='train' else 'test','--candidate-name',phase,
      '--candidate-file',str(plugin),'--flags','','--run-tag',HERE.name+'_'+domain,'--out',str(out),'--trials',str(trials),
      '--task-id-file',str(id_file),'--concurrency','8','--agent-llm','openai/qwen3-4b','--user-llm',user_llm,
      '--user-api-base',os.environ.get('USER_API_BASE') if str(os.environ.get('USER_API_BASE','')).startswith(('http://','https://')) else 'https://routify-pub.alibaba-inc.com/protocol/openai/v1','--eval-timeout','21600']
 if domain=='retail':cmd.append('--nl')
 env={**os.environ,'UV_CACHE_DIR':'/tmp/uv-cache'}
 env['PATH']='/mnt/data/xts/tools/uv/bin:'+env.get('PATH','')
 if user_llm=='openai/deepseek-v4-flash':env['TAU_USER_LLM_MIN_INTERVAL_SECONDS']='3.75'
 with (out.parent/'driver.log').open('w') as log:
  p=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
 if p.returncode or not out.exists():raise RuntimeError(f'Tau evaluation failed rc={p.returncode}; see {out.parent/"driver.log"}')
 score=add_trial_rewards(json.loads(out.read_text()));score.update(split=split,batch=batch if split=='train' else None,task_ids=ids,plugin_hash=digest(plugin),released_h_flags=[],user_llm=user_llm,user_llm_rpm=16 if user_llm=='openai/deepseek-v4-flash' else None,user_llm_min_interval_seconds=float(env['TAU_USER_LLM_MIN_INTERVAL_SECONDS']) if user_llm=='openai/deepseek-v4-flash' else None)
 out.write_text(json.dumps(score,indent=2,ensure_ascii=False)+'\n');print(out.read_text())

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('domain',choices=['airline','retail','telecom']);p.add_argument('split',choices=['train','test']);p.add_argument('phase');p.add_argument('plugin');p.add_argument('--batch',type=int,default=0);a=p.parse_args();main(a.domain,a.split,a.phase,a.plugin,a.batch)
