"""Host-only comparisons; never made visible to the optimizer."""
import fcntl
import json
from pathlib import Path
from protocol import DOMAINS, HERE, ROOT, atomic_json


def compare():
    meta = HERE.parent/'meta_agentbench_six_flash_tokens_20261004_repair2'
    old = HERE.parent/'life_agentbench_six_resume_mr_20261001'
    ahe = ROOT.parent/'ahe-native/batches/ahe_restricted_six_rotating_flash8_20261003_073330_repair1'
    ahe_rows = json.loads((ahe/'round_comparison.json').read_text())['rows']
    rows = []
    lines = ['# Six-round TRAIN comparison', '',
        'All scores are TRAIN, 50 tasks per evaluation. Batches differ across methods, not a paired comparison.',
        'Meta/Life: bN incumbent->candidate; accept/reject. AHE: post-evolve score, usually on the next batch.',
        'Meta/Life new have API usage ledgers; old Life tokens missing; AHE full round tokens missing (Evolve lower bound only).', '',
        '| Domain | Round | New Meta | Old Life | Restricted AHE | New Life | New Life round tokens |',
        '|---|---:|---|---|---|---|---:|']
    for domain in DOMAINS:
        states = {}
        for name, root in [('meta',meta),('life_old',old),('life_new',HERE)]:
            path = root/domain/'state.json'
            states[name] = json.loads(path.read_text())['rounds'] if path.exists() else []
        for number in range(1,7):
            row = dict(domain=domain, round=number)
            cells = {}
            for name, data in states.items():
                value = next((v for v in data if v.get('round',v.get('iteration')) == number), None)
                if value is None:
                    row[name], cells[name] = None, 'pending'
                    continue
                base = value.get('incumbent_score',value.get('baseline_score'))
                candidate = value.get('candidate_score')
                row[name] = {'batch':value['train_batch'], 'incumbent_accuracy':base,
                    'candidate_accuracy':candidate, 'retained_accuracy':candidate if value['accepted'] else base,
                    'accepted':value['accepted'], 'usage':value.get('usage')}
                text = 'unmeasured' if candidate is None else f'{candidate:.0%}'
                cells[name] = f"b{value['train_batch']} {base:.0%}->{text}; {'accept' if value['accepted'] else 'reject'}"
            value = next(v['ahe'] for v in ahe_rows if v['domain']==domain and v['optimization_step']==number)
            row['ahe'] = {k:value[k] for k in ('batch','accuracy','successes','tasks')}
            rows.append(row)
            cost = row['life_new']['usage']['total_tokens'] if row['life_new'] else 'pending'
            lines.append(f"| {domain} | {number} | {cells['meta']} | {cells['life_old']} | b{value['batch']} {value['accuracy']:.0%} | {cells['life_new']} | {cost} |")
    with (HERE/'.comparison.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        atomic_json(HERE/'round_comparison.json', {'split':'train','rows':rows})
        path = HERE/'COMPARISON.md'
        temp = path.with_suffix('.tmp')
        temp.write_text('\n'.join(lines)+'\n')
        temp.replace(path)
