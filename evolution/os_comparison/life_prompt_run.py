"""Fresh Life-only prompt experiment; reuse native OS runtime and six-round loop."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import fcntl
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import urllib.request

from evolution.tau_comparison.common import AHE_NATIVE, ROOT, read
from .protocol import prepare, save, digest
from .results import token_fields
from .run import Run, life_usage


def prepare_experiment(name, reference):
    reference = Path(reference).resolve()
    original = read(reference / 'protocol.json')
    final = read(reference / 'FINAL_RESULTS.json')
    if read(reference / 'status.json')['phase'] != 'complete':
        raise ValueError('completed reference experiment required')
    root = prepare(name, original['seed'])
    spec = read(root / 'protocol.json')
    if spec['batches'] != original['batches'] or spec['solver'] != original['solver']:
        raise ValueError('reference TRAIN schedule or solver changed')
    old_prompt = reference / 'life/optimizer_sessions/round_01/workspace/OPTIMIZER.md'
    archive = root / 'reference_prompt.md'
    shutil.copy2(old_prompt, archive)
    prompt = ROOT / 'evolution/life_evolve/PROMPT.md'
    spec.update(methods=['life'], initialization='empty Life H2-H5; no prior candidate artifacts',
        prior_test_exposure=True,
        test='one repeated native TEST144 after six TRAIN rounds and sealed selection; no TEST evidence in optimizer',
        intervention={'kind':'prompt-only', 'prompt':str(prompt), 'sha256':digest(prompt),
                      'reference_prompt_sha256':digest(archive), 'tools_changed':False},
        reference={'root':str(reference), 'final_results_sha256':digest(reference/'FINAL_RESULTS.json'),
                   'ahe_accuracy':final['methods']['ahe']['accuracy'],
                   'ahe_successes':final['methods']['ahe']['successes'],
                   'life_accuracy':final['methods']['life']['accuracy'],
                   'note':'Same TEST already inspected; this is a repeated comparison, not an unseen holdout.'})
    save(root/'protocol.json',spec)
    save(root/'status.json',{'phase':'prepared','completed_rounds':{'life':0}})
    return root


class LifePromptRun(Run):
    def __init__(self, root):
        spec = read(root/'protocol.json')
        if spec['methods'] != ['life'] or not spec['prior_test_exposure']:
            raise ValueError('fresh Life-only prompt protocol required')
        super().__init__(root)
        if self.recovery:
            raise ValueError('continuation needs a separate paid-artifact audit')
        self.state['prior_test_exposure'] = True
        save(self.work/'state.json',self.state)
        self.ready.set()
        self.mr.blocked_roles.update({'debugger','evolve'})
        self.meta_broker.close()
        self.meta_broker = None

    def verify(self):
        super().verify()
        value=read(self.root/'protocol.json')['intervention']
        if digest(value['prompt']) != value['sha256']:
            raise RuntimeError('frozen prompt changed')

    def wait_first(self):
        super().wait_first()
        if self.first_round.get('life:os') is False:
            raise RuntimeError('Life first-round integration check failed')

    def smoke(self):
        empty=self.compose.write_active_chain(self.work,self.state,'setup_empty')
        score,_=self.evaluate_life('train','setup_smoke',empty,self.spec['batches'][0]['task_ids'][:1])
        if score['infra_errors']:
            raise RuntimeError('native Life smoke failed before paid optimization')
        save(self.root/'setup_smoke.json',{'passed':True,'optimizer_calls':0,'life':score})

    def report(self, phase='training'):
        state=read(self.work/'state.json')
        usage=life_usage(self.mr.ledger.summary(),self.root.name)
        status={'phase':phase,'completed_rounds':{'life':state['round']},
                'utc_unix':time.time(),'gate':self.gate.snapshot()}
        save(self.root/'status.json',status)
        save(self.root/'TRAINING_RESULTS.json',{'life':state['rounds']})
        save(self.root/'OPTIMIZER_TOKEN_USAGE.json',{'life':usage})
        rows=[{'method':'life','round':v['round'],'measurement_batch':v['train_batch'],
            'baseline_accuracy':v['incumbent_score'],'candidate_accuracy':v['candidate_score'],
            'retained_accuracy':v['retained_score'],'accepted':v['accepted'],
            'infra_errors':v['candidate_infra_errors'],**token_fields(v['optimizer_usage'])}
            for v in state['rounds']]
        final_path=self.root/'FINAL_RESULTS.json'
        save(self.root/'BENCHMARK_RESULTS.json',{'phase':phase,'rounds':rows,
            'six_rounds_complete':state['round']==6,'optimizer_usage':{'life':usage},
            'prior_test_exposure':True,'reference':self.protocol['reference'],
            'intervention':self.protocol['intervention'],
            'final_test':read(final_path) if final_path.exists() else None})
        stream=io.StringIO()
        if rows:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        temporary=self.root/'SIX_ROUND_RESULTS.csv.tmp';temporary.write_text(stream.getvalue())
        temporary.replace(self.root/'SIX_ROUND_RESULTS.csv')
        return state['round'],usage

    def training(self):
        with ThreadPoolExecutor(max_workers=1) as pool:
            future=pool.submit(self.run_life)
            while not future.done():
                self.report()
                time.sleep(15)
            future.result()
        count,usage=self.report()
        self.verify()
        if count!=6 or not usage['accounting_usable_with_estimates']:
            raise RuntimeError('six complete Life rounds and usable accounting required')
        if any(v['role']!='life_optimizer' for v in self.mr.ledger.rows if v['event']=='started'):
            raise RuntimeError('unexpected paid optimizer role')
        self.life_final=self.compose.write_active_chain(self.work,self.state,'final')
        save(self.root/'training_seal.json',{'completed_rounds':{'life':6},
             'candidate_sha256':digest(self.life_final),'state_sha256':digest(self.work/'state.json'),
             'candidates':{'life':digest(self.life_final)},
             'state_hashes':{str(self.work/'state.json'):digest(self.work/'state.json')},
             'optimizer_usage':usage,'prior_test_exposure':True})

    def tests(self):
        seal=read(self.root/'training_seal.json')
        if (seal['completed_rounds']!={'life':6} or self.state['round']!=6
                or digest(self.life_final)!=seal['candidate_sha256']
                or digest(self.work/'state.json')!=seal['state_sha256']):
            raise RuntimeError('six-round sealed selection required before TEST')
        self.mr.blocked_roles.add('life_optimizer')
        self.report('testing')
        score,_=self.evaluate_life('test','final_test',self.life_final,self.spec['test_task_ids'])
        seal=read(self.root/'training_seal.json')
        if digest(self.life_final)!=seal['candidate_sha256'] or digest(self.work/'state.json')!=seal['state_sha256']:
            raise RuntimeError('sealed selection changed during TEST')
        reference=self.protocol['reference']['ahe_accuracy']
        save(self.root/'FINAL_RESULTS.json',{'split':'test','methods':{'life':score},
            'prior_test_exposure':True,'ahe_reference_accuracy':reference,
            'matches_or_exceeds_ahe_reference':score['accuracy']>=reference,
            'difference_from_ahe_reference':score['accuracy']-reference})
        self.report('complete')
        print(json.dumps({'phase':'complete','life_test_accuracy':score['accuracy'],
                          'ahe_reference_accuracy':reference}),flush=True)


def launch(root):
    if read(root/'status.json')['phase']!='prepared' or (root/'launch.json').exists():
        raise ValueError('fresh unlaunched prepared experiment required')
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open('http://127.0.0.1:8400/health',timeout=30) as response:health=json.load(response)
    if len(health)!=10 or any(v!=200 for v in health.values()):
        raise RuntimeError('all ten Qwen nodes must be healthy')
    name='life-harness-'+root.name.replace('_','-')
    command=['docker','run','-d','--name',name,'--network','host','--cap-add','SYS_CHROOT','--cap-add','MKNOD',
        '-v',f'{ROOT.parent}:{ROOT.parent}','-v',f'{Path(os.environ["LIFE_SOLVER_TOKENIZER"]).resolve()}:{Path(os.environ["LIFE_SOLVER_TOKENIZER"]).resolve()}:ro',
        '-v','/var/run/docker.sock:/var/run/docker.sock','-w',str(ROOT),'-e',f'PYTHONPATH={ROOT}:{AHE_NATIVE}',
        '-e',f'LIFE_SOLVER_TOKENIZER={Path(os.environ["LIFE_SOLVER_TOKENIZER"]).resolve()}',
        '-e','PYTHONDONTWRITEBYTECODE=1','-e','NO_PROXY=127.0.0.1,localhost','-e','GIT_CONFIG_COUNT=1',
        '-e','GIT_CONFIG_KEY_0=safe.directory','-e',f'GIT_CONFIG_VALUE_0={ROOT / "integrations/agentic-harness-engineering"}',
        'local-os/default',str(AHE_NATIVE/'.venv/bin/python'),'-u','-m','evolution.os_comparison.life_prompt_run',
        '--root',str(root),'--run']
    result=subprocess.run(command,check=True,capture_output=True,text=True)
    save(root/'launch.json',{'container':name,'container_id':result.stdout.strip(),'methods':['life']})
    print(json.dumps({'root':str(root),'container':name}))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--name');parser.add_argument('--reference',type=Path)
    parser.add_argument('--root',type=Path);parser.add_argument('--launch',action='store_true');parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if args.name:
        if args.run or not args.reference:parser.error('--name requires --reference and excludes --run')
        root=prepare_experiment(args.name,args.reference)
    elif args.root:root=args.root.resolve()
    else:parser.error('--name or --root required')
    if args.launch:launch(root);return
    if not args.run:print(root);return
    runner=None
    with (root/'.runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            runner=LifePromptRun(root);runner.smoke();runner.training();runner.tests()
        except Exception as exc:
            save(root/'failure.json',{'type':type(exc).__name__,'detail':str(exc)[:6000]})
            save(root/'status.json',{'phase':'stopped_on_error','error_type':type(exc).__name__})
            raise
        finally:
            if runner is not None:runner.close()

if __name__=='__main__':main()
