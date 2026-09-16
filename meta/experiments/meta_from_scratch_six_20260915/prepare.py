"""Freeze six from-scratch Meta-Harness runs and their train/test samples."""
from pathlib import Path
from collections import defaultdict
import hashlib, json, os, random, shutil, subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AB = ROOT / "AgentBench"
TAU = ROOT / "TauBench"
UP = ROOT.parent / "meta-harness/reference_examples/terminal_bench_2"
DOMAINS = ("airline", "retail", "telecom", "alfworld", "dbbench", "webshop")

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")

def tau_ids():
    code = """
import json
out={}
for d in ('airline','retail','telecom'):
 m=__import__(f'tau2.domains.{d}.environment',fromlist=['get_tasks'])
 out[d]={s:[t.id for t in m.get_tasks(s)] for s in ('train','test')}
print('TASK_IDS_JSON='+json.dumps(out))
"""
    env = {**os.environ, "UV_CACHE_DIR": "/tmp/uv-cache"}
    raw = subprocess.check_output(["uv", "run", "python", "-c", code], cwd=TAU, env=env, text=True)
    return json.loads(next(x.split("=",1)[1] for x in raw.splitlines() if x.startswith("TASK_IDS_JSON=")))

def rotated_batches(ids, size, seed):
    """Different deterministic samples; disjoint while the pool permits it."""
    ids=list(ids); batches=[]; seen=set()
    for n in range(6):
        rng=random.Random(seed+n)
        if len(ids) <= size:
            batch=ids[:]
        else:
            unseen=[x for x in ids if x not in seen]
            rng.shuffle(unseen)
            take=unseen[:size]
            if len(take)<size:
                rest=[x for x in ids if x not in take]
                rng.shuffle(rest);take += rest[:size-len(take)]
            rng.shuffle(take);batch=take
        assert len(batch)==min(size,len(ids)) and len(set(batch))==len(batch)
        if len(ids)>size and batches: assert batch != batches[-1]
        batches.append(batch);seen.update(batch)
    return batches

def copy_snapshot():
    snap=HERE/'task_snapshot'
    for rel in ('src/server/tasks/alfworld','src/server/tasks/dbbench','src/server/tasks/webshop','src/server/harness'):
        shutil.copytree(AB/rel,snap/rel,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for rel in ('src/server/tasks/native_environment.py','configs/tasks/alfworld.yaml','configs/tasks/dbbench.yaml','configs/tasks/webshop.yaml'):
        p=snap/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(AB/rel,p)
    for rel in ('src/__init__.py','src/server/__init__.py','src/server/tasks/__init__.py'):
        p=snap/rel;p.parent.mkdir(parents=True,exist_ok=True)
        if not p.exists():p.write_text('')
    return snap

ZERO='''\"\"\"Identity runtime: the from-scratch Meta-Harness iteration-0 anchor.\"\"\"\n\ndef register():\n    return None\n\nclass AgentHarness:\n    def __init__(self, public_context=None):\n        self.public_context = public_context\n    def transform_request(self, request):\n        return request\n    def transform_response(self, response, public_history):\n        return response\n'''

def main():
    if (HERE/'manifest.json').exists():
        raise SystemExit('manifest exists; sampling is frozen')
    snap=copy_snapshot(); tids=tau_ids()
    upstream_files=['meta_harness.py','claude_wrapper.py','.claude/skills/meta-harness-terminal-bench-2/SKILL.md']
    manifest={
      'name':HERE.name,
      'method':'original cloned Meta-Harness run_evolve/skill, ported to six native tasks; identity anchor; no released H2-H5',
      'iterations':6,'train_target_questions':50,'trials':1,'concurrency':8,
      'acceptance':'strict same-batch score improvement; change sample after non-improvement',
      'test_policy':'full fixed public test/std split after every round; test never affects acceptance or proposals',
      'agent':{'url':'http://127.0.0.1:8400/v1/chat/completions','model':'qwen3-4b','temperature':0.0,'max_tokens':4096,'chat_template_kwargs':{'enable_thinking':False}},
      'teacher':{'model':'openai/deepseek-flash','route_env_file':'/mnt/workspace/xts/skill/test_ap/env.sh'},
      'upstream':{'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=UP,text=True).strip(),'files':{n:digest(UP/n) for n in upstream_files}},
      'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
      'snapshot_hashes':{str(p.relative_to(snap)):digest(p) for p in snap.rglob('*') if p.is_file()},
      'data_hashes':{},'domains':{}
    }
    # Tau: airline has only 30 train questions, so two trials yield 60 trajectories.
    for offset,d in enumerate(('airline','retail','telecom')):
        train=tids[d]['train'];test=tids[d]['test'];batches=rotated_batches(train,50,202609150+offset*10)
        manifest['domains'][d]={'family':'tau','train_pool':len(train),'train_trials':2 if d=='airline' else 1,
          'capacity_exception':('30 public train questions; use all with two trials' if d=='airline' else None),
          'batches':[{'id':i,'task_ids':x} for i,x in enumerate(batches)],'test_task_ids':test}
    # ALFWorld: six balanced, disjoint batches from train, full new_std test.
    train=json.loads((AB/'data/alfworld/train_valid.json').read_text());test=json.loads((AB/'data/alfworld/new_std.json').read_text())
    games=[];groups=defaultdict(list);held={x for rows in test.values() for x in rows}
    for kind,rows in train.items():
        for game in rows:
            idx=len(games);games.append(game)
            if game not in held and game.startswith('json_2.1.1/train/'):groups[kind].append(idx)
    rng=random.Random(202609153)
    for ids in groups.values():rng.shuffle(ids)
    kinds=sorted(groups);base=50//len(kinds);quota={k:base for k in kinds}
    for k in kinds[:50-base*len(kinds)]:quota[k]+=1
    batches=[]
    for b in range(6):
        picked=[]
        for k in kinds:
            part=groups[k][b*quota[k]:(b+1)*quota[k]];assert len(part)==quota[k];picked+=part
        rng.shuffle(picked);batches.append({'id':b,'indices':picked,'types':{str(i):next(k for k,v in groups.items() if i in v) for i in picked}})
    test_games=[x for rows in test.values() for x in rows]
    manifest['domains']['alfworld']={'family':'agentbench','train_pool':sum(map(len,groups.values())),'batches':batches,
      'test_indices':list(range(len(test_games))),'train_split':'train_valid','test_split':'new_std'}
    manifest['data_hashes']['data/alfworld/train_valid.json']=digest(AB/'data/alfworld/train_valid.json')
    manifest['data_hashes']['data/alfworld/new_std.json']=digest(AB/'data/alfworld/new_std.json')
    for i in {x for b in batches for x in b['indices']}:
        p=AB/'data/alfworld'/games[i];manifest['data_hashes'][str(p.relative_to(AB))]=digest(p)
    for game in test_games:
        p=AB/'data/alfworld'/game;manifest['data_hashes'][str(p.relative_to(AB))]=digest(p)
    # DBBench: same broad type quotas previously proven to cover 6x50.
    norm=lambda s:' '.join(s.casefold().split())
    held_desc={norm(json.loads(line)['description']) for n in ('standard.jsonl','dev.jsonl') for line in (AB/f'data/dbbench/{n}').read_text().splitlines() if line.strip()}
    rows=[json.loads(x) for x in (AB/'data/dbbench/db_out_new.jsonl').read_text().splitlines() if x.strip()]
    groups=defaultdict(list)
    for i,row in enumerate(rows):
        if norm(row['description']) not in held_desc:groups[row['type'][0]].append(i)
    quota={'INSERT':17,'UPDATE':17,'SELECT':2,'other':2,'counting':2,'comparison':3,'ranking':3,'aggregation-SUM':1,'aggregation-MIN':1,'aggregation-MAX':1,'aggregation-AVG':1}
    rng=random.Random(202609154)
    for ids in groups.values():rng.shuffle(ids)
    batches=[]
    for b in range(6):
        picked=[]
        for k,n in quota.items():
            part=groups[k][b*n:(b+1)*n];assert len(part)==n,(k,len(part));picked+=part
        rng.shuffle(picked);batches.append({'id':b,'indices':picked,'types':{str(i):rows[i]['type'][0] for i in picked}})
    ntest=sum(1 for x in (AB/'data/dbbench/standard.jsonl').read_text().splitlines() if x.strip())
    manifest['domains']['dbbench']={'family':'agentbench','train_pool':sum(map(len,groups.values())),'batches':batches,'test_indices':list(range(ntest)),'train_file':'db_out_new.jsonl','test_file':'standard.jsonl'}
    for n in ('db_out_new.jsonl','standard.jsonl','dev.jsonl'):manifest['data_hashes'][f'data/dbbench/{n}']=digest(AB/f'data/dbbench/{n}')
    # WebShop public standard is 0:200; training uses 200:1021.
    ids=list(range(200,1021));random.Random(202609155).shuffle(ids)
    batches=[{'id':b,'indices':ids[b*50:(b+1)*50]} for b in range(6)]
    manifest['domains']['webshop']={'family':'agentbench','train_pool':len(ids),'batches':batches,'test_indices':list(range(200)),'train_range':[200,1021],'test_range':[0,200]}
    for rel in ('.native/webshop-src/data/items_shuffle_100k.json','.native/webshop-src/data/items_ins_v2_100k.json','.native/webshop-src/data/items_human_ins.json'):
        p=AB/rel;manifest['data_hashes'][rel]=digest(p)
    skill=UP/'.claude/skills/meta-harness-terminal-bench-2/SKILL.md'
    for d in DOMAINS:
        work=HERE/d/'workspace';(work/'agents').mkdir(parents=True,exist_ok=True);(work/'source_view').mkdir(exist_ok=True)
        (work/'agents/baseline_zero.py').write_text(ZERO)
        shutil.copy2(skill,work/'upstream_SKILL.md')
        if d in ('airline','retail','telecom'):
            (work/'source_view/CONTRACT.md').write_text('Candidate exports AgentHarness and register(). register() may patch only the plain domain environment/Tools path. Runtime starts from identity and every released H2-H5 flag remains disabled.\n')
            for name in ('environment.py','tools.py','data_model.py'):
                src=TAU/f'src/tau2/domains/{d}/{name}'
                if src.exists():shutil.copy2(src,work/'source_view'/name)
            shutil.copy2(TAU/'src/tau2/harness/base.py',work/'source_view/harness_base_protocol.py')
        else:
            (work/'source_view/CONTRACT.md').write_text('Candidate exports AgentHarness(public_context=None) with transform_request(request_dict) and transform_response(assistant_dict, public_history). Runtime starts from identity. No H2-H5 implementation is present or enabled.\n')
    manifest['zero_sha256']=digest(HERE/'airline/workspace/agents/baseline_zero.py')
    save(HERE/'manifest.json',manifest)
    save(HERE/'sampling_summary.json',{d:{'train_pool':manifest['domains'][d]['train_pool'],'batch_sizes':[len(x.get('task_ids',x.get('indices',[]))) for x in manifest['domains'][d]['batches']],'test_size':len(manifest['domains'][d].get('test_task_ids',manifest['domains'][d].get('test_indices',[])))} for d in DOMAINS})
    print((HERE/'sampling_summary.json').read_text())

if __name__=='__main__':main()
