"""Archive only this experiment's interrupted public proposer workspaces."""
import json
from pathlib import Path
import shutil
from protocol import HERE, atomic_json, digest

WORKSPACES = {'dbbench': Path('/tmp/life-optimizer-tchgb62p/workspace'),
              'webshop': Path('/tmp/life-optimizer-l9boinjs/workspace')}

def main():
    rows = {}
    for domain, source in WORKSPACES.items():
        inventory = json.loads((source/'inventory.json').read_text())
        frozen = HERE/'optimizer_inputs'/domain/'round_01'
        if (inventory['domain'] != domain or inventory['records'] != []
                or (source/'receipt.json').read_bytes() != (frozen/'receipt.json').read_bytes()):
            raise RuntimeError('Interrupted workspace identity mismatch')
        receipt = json.loads((source/'receipt.json').read_text())
        for name, sha in receipt['inputs'].items():
            if digest(source/name) != sha:
                raise RuntimeError('Interrupted public input changed')
        target = HERE/'interrupted_proposals'/domain/'workspace'
        if target.exists():
            actual = {str(p.relative_to(target)):digest(p) for p in target.rglob('*') if p.is_file()}
            expected = {str(p.relative_to(source)):digest(p) for p in source.rglob('*') if p.is_file()}
            if actual != expected:
                raise RuntimeError('Archive changed')
        else:
            shutil.copytree(source,target)
        requests = [json.loads(s) for s in (HERE/domain/'flash_requests.jsonl').read_text().splitlines()]
        started = json.loads((HERE/domain/'progress.json').read_text())['domain_request_number']
        usage = {}
        for row in requests:
            for key,value in row['usage'].items():
                if type(value) is int:
                    usage[key]=usage.get(key,0)+value
        rows[domain] = {'requests_started':started,'requests_completed':len(requests),'usage':usage,
            'unknown_inflight_requests':started-len(requests),
            'completion_tokens_reserved':usage.get('completion_tokens',0)+8192*(started-len(requests)),
            'output_truncations':sum(r['finish_reason']=='length' for r in requests),
            'workspace':str(target.relative_to(HERE)),
            'workspace_hashes':{str(p.relative_to(target)):digest(p) for p in target.rglob('*') if p.is_file()}}
    atomic_json(HERE/'interruption.json',{'reason':'repeated 8192-token output truncation before valid candidate',
        'performance_decisions':0,'training_baselines_preserved':True,'domains':rows,
        'repair':'request-level checkpoint; actual reasoning retained; wider per-call output within same round aggregate output budget'})
    print(json.dumps({d:{k:v for k,v in r.items() if k not in ('workspace_hashes','usage')} for d,r in rows.items()}))

if __name__ == '__main__':
    main()
