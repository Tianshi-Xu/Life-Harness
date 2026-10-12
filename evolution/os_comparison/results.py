"""Record live and final OS results without running models or changing selection."""
import argparse
import csv
import io
from pathlib import Path
import time
from evolution.tau_comparison.common import read
from .protocol import save,digest


def token_fields(value):
    return {'measured_optimizer_tokens':value.get('total_tokens',0),
            'estimated_optimizer_tokens':value.get('estimated_total_tokens',0),
            'optimizer_tokens_including_estimates':value.get('budget_tokens_including_estimates',value.get('total_tokens',0)),
            'accounting_complete':value.get('accounting_complete',False),
            'accounting_usable_with_estimates':value.get('accounting_usable_with_estimates',value.get('accounting_complete',False))}


def build(root):
    root=Path(root);training=read(root/'TRAINING_RESULTS.json');tokens=read(root/'OPTIMIZER_TOKEN_USAGE.json')
    rows=[]
    for method in ('meta','life'):
        for row in training[method]:
            is_meta=method=='meta';number=row['iteration'] if is_meta else row['round']
            baseline=row['baseline_score'] if is_meta else row['incumbent_score']
            score=row['candidate_score']
            retained=score if row['accepted'] else baseline
            rows.append({'method':method,'round':number,'measurement_batch':row['train_batch'],
                'candidate_accuracy':score,'retained_accuracy':retained,'baseline_accuracy':baseline,
                'accepted':row['accepted'],'infra_errors':row['candidate_infra_errors'],
                'measurement':'same complete TRAIN batch candidate/incumbent comparison',
                **token_fields(row['usage'] if is_meta else row['optimizer_usage'])})
    # Official AHE evaluates the evolved workspace on the NEXT batch. Last
    # evolution has an additional TRAIN measurement; selection stays best-ever.
    evaluation={row['iteration']:row for row in training['ahe']['evaluations']}
    experiment=root/'ahe/experiments'/(root.name+'_ahe_os')
    for number in range(1,7):
        if not (experiment/f'runs/iteration_{number:03d}/evolve/evolve_summary.md').exists():continue
        value=evaluation.get(number+1) if number<6 else training.get('ahe_final_train')
        if value and number<6:
            n=value['tasks'];score=value['successes']/n if n else None;infra=value['infrastructure_errors']
        elif value:
            score=value['accuracy'];infra=value['infra_errors']
        else:score=infra=None
        rows.append({'method':'ahe','round':number,'measurement_batch':min(number,5),
            'candidate_accuracy':score,'retained_accuracy':None,'baseline_accuracy':None,
            'accepted':None,'infra_errors':infra,
            'measurement':'next TRAIN batch evolved workspace; final round measured on batch 5; official best-ever selection',
            **token_fields(training['ahe_round_usage'][str(number)])})
    final_path=root/'FINAL_RESULTS.json'
    final=read(final_path) if final_path.exists() else None
    status=read(root/'status.json')
    report={'schema_version':1,'domain':'os','phase':status['phase'],'six_rounds_complete':all(
        len([r for r in rows if r['method']==m])==6 for m in ('meta','ahe','life')),
        'solver':'Qwen3-4B, thinking disabled','rounds':rows,'optimizer_usage':tokens,
        'final_test':final,'test_started':(root/'training_seal.json').exists(),
        'ahe_first_debugger_deployment_failure':'First-round CLI failed before inference; paid first evolution retained. Debugger package repaired for rounds 2–6.',
        'selection':'Meta/Life strict same-batch improvement; AHE original best-ever TRAIN snapshot',
        'notes':['AHE candidate scores use different batches from Meta/Life; do not present as same-task comparisons.',
                 'Unknown timeout usage is estimated separately; solver tokens are excluded from optimizer totals.',
                 'Pre-six-round and pre-TEST files are progress records, not final experimental results.'],
        'raw_record_hashes':{str(p.relative_to(root)):digest(p) for p in [root/'TRAINING_RESULTS.json',root/'OPTIMIZER_TOKEN_USAGE.json']}}
    return report


def write(root):
    root=Path(root);report=build(root)
    save(root/'BENCHMARK_RESULTS.json',report)
    stream=io.StringIO();fields=list(report['rounds'][0]) if report['rounds'] else []
    writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(report['rounds'])
    path=root/'SIX_ROUND_RESULTS.csv';temporary=path.with_suffix('.tmp');temporary.write_text(stream.getvalue());temporary.replace(path)
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--watch',action='store_true')
    args=parser.parse_args()
    while True:
        report=write(args.root)
        if not args.watch or report['phase'] in {'complete','stopped_on_error'}:break
        time.sleep(45)

if __name__=='__main__':main()
