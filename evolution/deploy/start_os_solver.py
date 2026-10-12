"""Start two isolated Qwen3-4B servers per selected GPU and the existing proxy."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / "AgentBench/.native/solver/bin/python"
STATE = ROOT / "AgentBench/.native/os_solver"
LOGS = ROOT / "evolution/deploy/logs/os_qwen"


def save(value):
    target=STATE/"status.json"
    temporary=target.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(target)


def healthy(port):
    try:
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f'http://127.0.0.1:{port}/health',timeout=3) as response:
            return response.status==200
    except Exception:
        return False


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpus',default='1,2,3,4,5')
    args=parser.parse_args()
    gpus=[int(x) for x in args.gpus.split(',')]
    if len(set(gpus))!=len(gpus):raise ValueError('duplicate GPU')
    model=Path(os.environ['LIFE_SOLVER_TOKENIZER']).resolve()
    index=json.loads((model/'model.safetensors.index.json').read_text())
    if any(not (model/name).is_file() for name in set(index['weight_map'].values())):
        raise RuntimeError('complete model shards required')
    if (STATE/'status.json').exists():raise FileExistsError('existing deployment; inspect recorded PIDs before restarting')
    nodes=[{'gpu':gpu,'slot':slot,'port':8401+2*i+slot,'phase':'pending'}
           for i,gpu in enumerate(gpus) for slot in range(2)]
    for port in [8400,*[n['port'] for n in nodes]]:
        with socket.socket() as sock:sock.bind(('127.0.0.1',port))
    info=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu',
                                  '--format=csv,noheader,nounits'],text=True)
    free={int(line.split(',')[0]) for line in info.splitlines()
          if int(line.split(',')[1])<1024 and int(line.split(',')[2])<=5}
    if not set(gpus)<=free:raise RuntimeError('selected GPU is no longer idle')
    STATE.mkdir(parents=True,exist_ok=True);LOGS.mkdir(parents=True,exist_ok=True)
    state={'phase':'starting','launcher_pid':os.getpid(),'nodes':nodes,'thinking':False,'context':32768,'nodes_per_gpu':2}
    save(state)
    processes={}
    for slot in range(2):
        for node in [n for n in nodes if n['slot']==slot]:
            env=os.environ.copy()
            for key in list(env):
                if any(word in key.upper() for word in ('KEY','TOKEN','SECRET','PASSWORD','CREDENTIAL','AUTH')):env.pop(key)
            env.update(CUDA_VISIBLE_DEVICES=str(node['gpu']),HF_HUB_OFFLINE='1',
                       PATH=str(PYTHON.parent)+':'+env.get('PATH',''),
                       OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',
                       NO_PROXY='localhost,127.0.0.1',no_proxy='localhost,127.0.0.1')
            log=LOGS/f"node_{node['port']}.log"
            command=[str(PYTHON),'-m','vllm.entrypoints.cli.main','serve',str(model),
                     '--served-model-name','qwen3-4b','--host','127.0.0.1','--port',str(node['port']),
                     '--gpu-memory-utilization','0.43','--max-model-len','32768',
                     '--max-num-seqs','64','--max-num-batched-tokens','8192',
                     '--enable-auto-tool-choice','--tool-call-parser','hermes',
                     '--default-chat-template-kwargs','{"enable_thinking":false}',
                     '--generation-config','vllm']
            with log.open('w') as output:
                process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
            node.update(pid=process.pid,phase='starting',log=str(log));save(state)
            processes[node['port']]=process
        deadline=time.monotonic()+900
        while not all(n['phase']=='healthy' for n in nodes if n['slot']==slot):
            for node in [n for n in nodes if n['slot']==slot and n['phase']!='healthy']:
                if healthy(node['port']):node['phase']='healthy'
                elif processes[node['port']].poll() is not None:
                    state.update(phase='failed',failed_port=node['port']);save(state)
                    raise RuntimeError('Qwen node exited; inspect '+node['log'])
            save(state)
            print(json.dumps({'slot':slot,'healthy':[n['port'] for n in nodes if n['phase']=='healthy']}),flush=True)
            if time.monotonic()>deadline:
                state['phase']='startup_timeout';save(state);raise TimeoutError('node startup timed out')
            time.sleep(15)
    env=os.environ.copy();env['OS_QWEN_BACKENDS']=','.join(f"http://127.0.0.1:{n['port']}" for n in nodes)
    with (LOGS/'proxy.log').open('w') as output:
        process=subprocess.Popen([str(PYTHON),str(ROOT/'evolution/deploy/proxy.py')],cwd=ROOT,
            env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
    state.update(proxy_pid=process.pid,proxy='http://127.0.0.1:8400/v1',phase='nodes_healthy_proxy_starting');save(state)
    for _ in range(15):
        if healthy(8400):
            state['phase']='healthy';save(state);print(json.dumps(state),flush=True);return
        time.sleep(2)
    state['phase']='proxy_failed';save(state);raise RuntimeError('proxy did not become ready')


if __name__=='__main__':main()
