"""Read historic traces; publish separately without inventing missing API usage."""
import argparse
import json
from pathlib import Path

from .records import write_json

ROOT = Path(__file__).resolve().parents[1]
LIFE = ROOT.parent/'Life-Harness/meta/experiments'
NAME = 'ahe_restricted_six_rotating_flash8_20261003_073330_repair1'
DOMAINS = ('alfworld','dbbench','webshop')


def audit():
    batch = ROOT/'batches'/NAME
    comparisons = json.loads((batch/'round_comparison.json').read_text())['rows']
    domains, all_tokens, all_calls = {}, 0, 0
    for domain in DOMAINS:
        experiment = ROOT/'experiments'/(NAME+'_'+domain)
        rows = []
        for number in range(1,7):
            path = experiment/'runs'/f'iteration_{number:03d}'/'evolve/nexau_in_memory_tracer.cleaned.json'
            trace = json.loads(path.read_text())
            if trace['model'] != 'deepseek-v4-flash' or trace['subagents']:
                raise ValueError('Unexpected model or nested agent accounting')
            train = next(v['ahe'] for v in comparisons
                         if v['domain']==domain and v['optimization_step']==number)
            rows.append({'round':number,'evolve_recorded_tokens':trace['total_tokens'],
                'evolve_recorded_generations':trace['generation_count'],
                'full_round_tokens':None,'full_round_accounting_complete':False,
                'train_accuracy':train['accuracy'],'train_successes':train['successes'],
                'train_n':train['tasks'],'evaluation_batch':train['batch'],
                'source':str(path.relative_to(ROOT)),
                'missing':'Debugger and possibly untraced compaction/retry/unfinished calls'})
        tokens = sum(v['evolve_recorded_tokens'] for v in rows)
        calls = sum(v['evolve_recorded_generations'] for v in rows)
        domains[domain] = {'rounds':rows,'evolve_recorded_tokens':tokens,'evolve_recorded_generations':calls}
        all_tokens += tokens
        all_calls += calls
    receipt = json.loads((batch/'api_transport_receipt.json').read_text())
    requests = receipt['flash_api']['requests']
    difference = requests-all_calls
    sensitivity = [{'assumed_mean_tokens_per_untraced_attempt':mean,
                    'untraced_scenario_tokens':difference*mean,
                    'known_evolve_plus_scenario_tokens':all_tokens+difference*mean}
                   for mean in (20000,50000,100000)]
    return {'schema_version':1,'method':'read-only historical tracer audit',
        'domains':domains,'evolve_recorded_tokens':all_tokens,
        'full_optimizer_tokens':None,'full_accounting_complete':False,
        'current_run_api_attempts':requests,'evolve_recorded_generations':all_calls,
        'request_count_difference':difference,
        'difference_is_not_verified_debugger_only':True,
        'sensitivity_scenarios_not_measurements':sensitivity,
        'estimate_limitations':[
            'Assumed per-call means are not empirically measured, and this is not a confidence interval.',
            'Trace generations and request-gate attempts are not guaranteed one-to-one on failed calls.',
            'Reused first-round WebShop Debugger work predates the current broker and is outside its request count.',
            'Archived pre-repair evolution/startup costs are not included.',
            'Cannot assign missing spend to individual rounds; cannot claim these scenarios are audited full spend.'
        ]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Never overwrite an existing historical audit')
    args.output.mkdir(parents=True)
    value = audit()
    write_json(args.output/'TOKEN_AUDIT.json',value)
    lines = ['# Historical AHE token audit','',
        'Evolve-only known tokens are a lower-bound record, not full optimizer spend.',
        'Debugger had no tracer; the old broker recorded request counts but not API usage.',
        'Sensitivity scenarios below are explicitly assumptions, not recovered measurements.','',
        '| Domain | Round | Recorded Evolve tokens | Post-round TRAIN accuracy | Batch |',
        '|---|---:|---:|---:|---:|']
    for domain,data in value['domains'].items():
        for row in data['rounds']:
            lines.append(f"| {domain} | {row['round']} | {row['evolve_recorded_tokens']} | {row['train_accuracy']:.0%} | {row['evaluation_batch']} |")
    lines += ['',f"Known Evolve total: {value['evolve_recorded_tokens']:,} tokens.",
              f"Current broker attempts: {value['current_run_api_attempts']}; Evolve generations: {value['evolve_recorded_generations']}; difference: {value['request_count_difference']}.",'',
              '| Assumed missing-call mean | Known plus scenario tokens |','|---:|---:|']
    for row in value['sensitivity_scenarios_not_measurements']:
        lines.append(f"| {row['assumed_mean_tokens_per_untraced_attempt']:,} | {row['known_evolve_plus_scenario_tokens']:,} |")
    lines += ['','Limitations:']+['- '+v for v in value['estimate_limitations']]
    (args.output/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:value[k] for k in ('evolve_recorded_tokens','current_run_api_attempts',
        'evolve_recorded_generations','request_count_difference','sensitivity_scenarios_not_measurements')}))


if __name__ == '__main__':
    main()
