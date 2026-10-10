"""Test TRAIN-selected AHE snapshots; never rerun evolution or select on TEST."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait
import os
from pathlib import Path
import shutil
import sys
import time

import yaml

from .common import AHE_NATIVE, ROOT, digest, read, save, source_fingerprint
from .pass_three import Tests, audit as audit_tau, pass_three

TAU_PARENT = ROOT / 'meta/experiments/meta_ahe_taubench_six_flash_tokens_20261004_repair8'
PASS_PARENT = ROOT / 'meta/experiments/meta_ahe_tau_pass3_20261009'
AB_PARENT = AHE_NATIVE / 'heldout_tests/ahe_restricted_six_rotating_flash8_20261003_073330_repair1_finaltest'
AB_BATCH = AHE_NATIVE / 'batches/ahe_restricted_six_rotating_flash8_20261003_073330_repair1'
LIFE_TELECOM = ROOT / 'meta/experiments/life_telecom_r3_fixed_test3_20261010'


def select_final(best_rate, final_successes, final_episodes):
    """Original strict best-ever tie rule, including the paid final TRAIN check."""
    if not 0 <= best_rate <= 1 or final_episodes <= 0 or not 0 <= final_successes <= final_episodes:
        raise ValueError('invalid complete TRAIN score')
    return final_successes / final_episodes > best_rate


def audit():
    sys.path.insert(0, str(AHE_NATIVE))
    from ahe_native.backend import workspace_hash
    from ahe_native.capabilities import restricted_policy, validate_workspace
    from ahe_native.final_test import completed_training, fixed_test_spec
    ab_seal, ab_spec = completed_training(AB_BATCH), fixed_test_spec()
    old_tau = audit_tau(TAU_PARENT)
    tau_spec = read(TAU_PARENT / 'protocol.json')['domains_spec']
    tau_seal = read(TAU_PARENT / 'training_seal.json')
    old_ab_seal = read(AB_PARENT / 'training_seal.json')
    plan, receipts = {}, {}
    for domain in (*ab_spec['task_ids'], *tau_spec):
        is_tau = domain in tau_spec
        if is_tau:
            experiments = TAU_PARENT / 'ahe/experiments'
            experiment = next(p for p in experiments.iterdir() if p.name.endswith('_ahe_' + domain))
            final_path = TAU_PARENT / 'ahe' / domain / 'final_train.json'
            final_train = read(final_path)
            ids = tau_spec[domain]['test_task_ids']
            expected_hash = tau_seal['snapshots']['ahe:' + domain]
            tested = Path(old_tau['ahe:' + domain]['snapshot'])
            final_n = final_train['episodes']
        else:
            experiment = Path(ab_seal['domains'][domain]['workspace']).parent
            final_path = AB_BATCH / (domain + '_final_candidate_train.json')
            final_train = read(final_path)
            ids = ab_spec['task_ids'][domain]
            expected_hash = old_ab_seal['domains'][domain]['workspace_sha256']
            tested = AB_PARENT / domain / 'workspace'
            final_n = final_train['tasks']
        final_hash = workspace_hash(experiment / 'workspace')
        if workspace_hash(tested) != expected_hash or final_hash != expected_hash:
            raise ValueError('historical final TEST snapshot changed: ' + domain)
        best_path = experiment / 'best_ever.json'
        best = read(best_path)
        scores_path = experiment / 'iteration_scores.yaml'
        scores = yaml.safe_load(scores_path.read_text())['scores']
        score = next(row for row in scores if row['iteration'] == best['iteration'])
        if abs(score['pass_rate'] - best['pass_rate']) > 0.0001 or any(
                row['pass_rate'] > best['pass_rate'] + 0.0001 for row in scores):
            raise ValueError('best-ever disagrees with registered TRAIN scores: ' + domain)
        snapshot = experiment / 'runs' / f"iteration_{best['iteration']:03d}" / 'input/workspace'
        if not snapshot.is_dir():
            snapshot = snapshot.parent.parent / 'workspace_snapshot'
        job = experiment / score['job_dir']
        job_receipt = job / 'workspace_receipt.json'
        if workspace_hash(snapshot) != read(job_receipt)['workspace_sha256']:
            raise ValueError('best TRAIN receipt does not bind its snapshot: ' + domain)
        final_job = Path(final_train['job'])
        final_receipt = final_job / 'workspace_receipt.json'
        if read(final_receipt)['workspace_sha256'] != final_hash:
            raise ValueError('final TRAIN receipt does not bind its snapshot: ' + domain)
        for measured_job, expected_n, expected_rate in [(job,
                60 if domain == 'airline' else 50, best['pass_rate']),
                (final_job, final_n, final_train['successes'] / final_n)]:
            records = list(measured_job.glob('native_*/native_record.json'))
            rows = [read(p) for p in records]
            if (len(rows) != expected_n or len({(r['task_id'], r['trial']) for r in rows}) != expected_n
                    or abs(sum(r['success'] for r in rows) / expected_n - expected_rate) > 0.0001):
                raise ValueError('incomplete or inconsistent TRAIN evidence: ' + domain)
            receipts.update({str(p): digest(p) for p in records})
        use_final = select_final(best['pass_rate'], final_train['successes'], final_n)
        selected = experiment / 'workspace' if use_final else snapshot
        selected_hash = workspace_hash(selected)
        validate_workspace(selected, restricted_policy(domain))
        plan['ahe:' + domain] = {'domain': domain, 'ids': ids, 'snapshot': str(selected),
            'snapshot_sha256': selected_hash, 'tested_snapshot_sha256': expected_hash,
            'needs_retest': selected_hash != expected_hash,
            'selected_round': 6 if use_final else best['iteration'] - 1,
            'selected_train_accuracy': final_train['successes'] / final_n if use_final else best['pass_rate'],
            'original_best_iteration': best['iteration'], 'original_best_train_accuracy': best['pass_rate'],
            'final_train_accuracy': final_train['successes'] / final_n,
            'selection': 'strict TRAIN best including paid final candidate; ties retain original best-ever',
            'trial_zero': [], 'new_trials': [0, 1, 2] if is_tau else [0]}
        receipts.update({str(p): digest(p) for p in [best_path, scores_path, final_path, job_receipt, final_receipt]})
    for spec in old_tau.values():
        receipts.update(spec['records_sha256'])
    for domain in ab_spec['task_ids']:
        summary_path = AB_PARENT / domain / 'summary.json'
        summary = read(summary_path)
        receipts[str(summary_path)] = digest(summary_path)
        for relative, sha in summary['episode_sha256'].items():
            path = AB_PARENT / domain / relative
            if digest(path) != sha:
                raise ValueError('old AgentBench TEST record changed')
            receipts[str(path)] = sha
    return plan, receipts


def reused_metrics(plan, receipts):
    """Derive new metrics from complete raw trials without editing old results."""
    from ahe_native.backend import workspace_hash
    results, old = {}, read(PASS_PARENT / 'reuse_receipt.json')['plan']
    for domain in ('airline', 'retail', 'telecom'):
        path = PASS_PARENT / 'meta' / domain / 'native/results.json'
        rows = read(path)['simulations']
        results['meta:' + domain] = pass_three(old['meta:' + domain]['ids'], rows)
        receipts[str(path)] = digest(path)
        spec = plan['ahe:' + domain]
        if not spec['needs_retest']:
            folder = PASS_PARENT / 'ahe' / domain
            if (spec['snapshot_sha256'] != old['ahe:' + domain]['snapshot_sha256']
                    or workspace_hash(folder / 'workspace') != spec['snapshot_sha256']):
                raise ValueError('cannot reuse TEST from another selected snapshot')
            paths = list(folder.glob('*/record.json'))
            rows = old['ahe:' + domain]['trial_zero'] + [{**read(p), 'reward': read(p)['success']} for p in paths]
            results['ahe:' + domain] = pass_three(spec['ids'], rows)
            receipts.update({str(p): digest(p) for p in paths})
    path = LIFE_TELECOM / 'native/results.json'
    protocol = read(LIFE_TELECOM / 'protocol.json')
    results['life:telecom_r3'] = {**pass_three(protocol['ids'], read(path)['simulations']),
        'snapshot_sha256': protocol['snapshot_sha256'], 'selected_round': protocol['selected_round'],
        'note': 'existing fixed round-3 snapshot, not the original six-round or new round-6 candidate'}
    receipts.update({str(p): digest(p) for p in [path, LIFE_TELECOM / 'protocol.json', PASS_PARENT / 'reuse_receipt.json']})
    if any(score['missing_episodes'] for score in results.values()):
        raise ValueError('cannot publish pass@3 from incomplete trials')
    return results


def agentbench(root, domain, spec):
    from ahe_native.backend import workspace_hash
    from ahe_native.final_test import evaluate_domain
    folder = root / 'ahe' / domain
    snapshot = folder / 'workspace'
    shutil.copytree(spec['snapshot'], snapshot, ignore=shutil.ignore_patterns('.git', '__pycache__'))
    if workspace_hash(snapshot) != spec['snapshot_sha256']:
        raise ValueError('selected snapshot changed before TEST')
    score = evaluate_domain(domain, spec['ids'], snapshot, folder, 50)
    score.pop('optimization_step', None)
    score.update(selected_round=spec['selected_round'], optimization_steps_completed=6,
                 candidate_selection=spec['selection'])
    save(folder / 'summary.json', score)
    return score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    if not args.name.replace('_', '').replace('-', '').isalnum():
        raise ValueError('isolated simple run name required')
    plan, receipts = audit()
    reused = reused_metrics(plan, receipts)
    if args.audit_only:
        print({key: {k:spec[k] for k in ('needs_retest', 'selected_round', 'selected_train_accuracy')}
               for key, spec in plan.items()})
        return
    root = ROOT / 'meta/experiments' / args.name
    root.mkdir()  # Never overwrite or automatically replay paid tests.
    save(root / 'selection_receipt.json', {'plan': plan, 'historical_files': receipts})
    save(root / 'REUSED_TAU_METRICS.json', reused)
    frozen = source_fingerprint()
    save(root / 'frozen_sources.json', frozen)
    save(root / 'protocol.json', {'selection': 'TRAIN best only; original AHE evolution untouched',
        'final_paid_train_included': True, 'optimizer_tokens': 0, 'new_life_episodes': 0,
        'retest': [key for key, spec in plan.items() if spec['needs_retest']],
        'agentbench_trials': 1, 'taubench_trials': 3, 'global_mr_concurrency': 8,
        'local_concurrency_per_domain': 50, 'old_results_overwritten': False})
    tests = Tests.__new__(Tests)
    try:
        tests.__init__(root, plan)
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = {}
            for key, spec in plan.items():
                if not spec['needs_retest']:
                    continue
                domain = spec['domain']
                futures[key] = (pool.submit(tests.ahe, domain) if domain in ('airline', 'retail', 'telecom')
                                else pool.submit(agentbench, root, domain, spec))
            while True:
                pending = [future for future in futures.values() if not future.done()]
                save(root / 'status.json', {'phase': 'running' if pending else 'finished', 'pid': os.getpid(),
                    'completed': [key for key, future in futures.items() if future.done() and future.exception() is None],
                    'failed': [key for key, future in futures.items() if future.done() and future.exception() is not None],
                    'mr_gate': tests.gate.snapshot(), 'updated_at': time.time()})
                if not pending:
                    break
                wait(pending, timeout=30)
            for key, future in futures.items():
                future.result()
                reused[key] = read(root / 'ahe' / plan[key]['domain'] / 'summary.json')
        from ahe_native.backend import workspace_hash
        if (source_fingerprint() != frozen
                or any(digest(p) != sha for p, sha in receipts.items())
                or any(workspace_hash(spec['snapshot']) != spec['snapshot_sha256'] for spec in plan.values())):
            raise RuntimeError('frozen source or historical records changed')
        save(root / 'FINAL_RESULTS.json', {'selection': 'TRAIN best, never TEST selected',
            'optimizer_tokens': 0, 'methods': reused})
    except Exception as exc:
        save(root / 'error.json', {'type': type(exc).__name__, 'message': str(exc)})
        raise
    finally:
        tests.close()
        if hasattr(tests, 'gate'):
            save(root / 'gate_closed.json', {'gate': tests.gate.snapshot(), 'closed': True})


if __name__ == '__main__':
    main()
