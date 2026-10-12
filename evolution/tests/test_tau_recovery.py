"""Offline recovery checks; no paid APIs, no historical artifact mutation."""
from contextlib import redirect_stdout
import io
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from evolution.tau_comparison.common import AHE_NATIVE, digest, read, save
sys.path.insert(0, str(AHE_NATIVE))
from evolution.tau_comparison import meta_port
from evolution.tau_comparison.recovery import (audit_meta_cached, audit_meta_zero_api,
    audit_ahe_rounds, audit_completed_evolve, audit_debugger_cache, audit_final_train,
    audit_unscoped_evolve, copy_parent, ledger_complete)
from ahe_native.backend import episode_timeout, execute, workspace_hash
from ahe_native.token_accounting import TokenLedger
from evolution.tau_comparison.run import Comparison, main, select_optimizer_groups


class RecoveryTests(unittest.TestCase):
    def final_train_fixture(self, root):
        import shutil
        from evolution.tau_comparison.common import summarize_rows
        experiment = root / 'ahe/experiments/airline'
        shutil.copytree(AHE_NATIVE / 'baselines/airline', experiment / 'workspace')
        spec = {'batches': [{'task_ids': ['a', 'b']} for _ in range(6)], 'train_trials': 1}
        job = root / 'ahe/final_train/airline/iteration_006/job'
        rows = [{'task_id': task, 'trial': 0, 'success': int(task == 'a'),
                 'infrastructure_error': False} for task in ('a', 'b')]
        for row in rows:
            save(job / ('native_' + row['task_id']) / 'native_record.json', row)
        save(job / 'result.json', {'completed': True})
        save(job / 'workspace_receipt.json', {'workspace_sha256': workspace_hash(experiment / 'workspace')})
        score = summarize_rows(['a', 'b'], 1, [{**row, 'reward': row['success']} for row in rows])
        save(root / 'ahe/airline/final_train.json', {**score, 'job': str(job), 'used_for_selection': False})
        return experiment, spec, job

    def test_final_train_reuse_rejects_incomplete_changed_or_cross_domain_results(self):
        for tamper in (None, 'partial', 'score', 'workspace', 'task', 'job'):
            with self.subTest(tamper=tamper), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                experiment, spec, job = self.final_train_fixture(root)
                receipt = root / 'ahe/airline/final_train.json'
                if tamper == 'partial':
                    receipt.unlink()
                elif tamper == 'score':
                    save(receipt, {**read(receipt), 'successes': 2})
                elif tamper == 'workspace':
                    (experiment / 'workspace/systemprompt.md').write_text('changed')
                elif tamper == 'task':
                    save(job / 'native_b/native_record.json', {'task_id': 'c', 'trial': 0, 'success': 0})
                elif tamper == 'job':
                    save(receipt, {**read(receipt), 'job': str(root / 'other_domain')})
                if tamper:
                    with self.assertRaises((RuntimeError, ValueError)):
                        audit_final_train(root, 'airline', experiment, spec)
                else:
                    cached = audit_final_train(root, 'airline', experiment, spec)
                    self.assertEqual(cached['completed_episodes_reused'], 2)
                    self.assertIn('ahe/airline/final_train.json', cached['files'])
                    comparison = object.__new__(Comparison)
                    comparison.root = root
                    comparison.ahe_configs = {'airline': {'harbor': {'k': 1}}}
                    comparison.protocol = {'domains_spec': {'airline': spec}}
                    comparison.names = {'airline': 'airline'}
                    comparison.continuation = {'ahe_resume': {'airline': {
                        'start_iteration': 7, 'final_train_reused': cached}}}
                    comparison.ahe_module = Mock()
                    comparison.ahe_usage = Mock(return_value={'accounting_usable_with_estimates': True})
                    for number in range(1, 7):
                        (experiment / f'runs/iteration_{number:03d}/evolve').mkdir(parents=True)
                        (experiment / f'runs/iteration_{number:03d}/evolve/evolve_summary.md').write_text('paid')
                    with patch('ahe_native.batch.progress', return_value={'evaluations': [{}] * 6}):
                        comparison.run_ahe('airline')
                    comparison.ahe_module.run_harbor.assert_not_called()

    def ahe_round_fixture(self, root, measured, completed):
        import shutil
        experiment = root / 'host'
        spec = {'batches': [{'task_ids': ['a', 'b']} for _ in range(6)], 'train_trials': 1}
        paid = []
        for number in range(2, measured + 1):
            folder = experiment / f'runs/iteration_{number:03d}'
            shutil.copytree(AHE_NATIVE / 'baselines/airline', folder / 'input/workspace')
            job = folder / 'input/benchmark/20261005_job'
            save(job / 'result.json', {'completed': True})
            save(job / 'workspace_receipt.json', {'workspace_sha256': workspace_hash(folder / 'input/workspace')})
            for task in ('a', 'b'):
                save(job / f'native_{task}/native_record.json', {'task_id': task, 'trial': 0,
                     'success': False, 'infrastructure_error': False})
            if number <= completed:
                (folder / 'evolve').mkdir()
                (folder / 'evolve/evolve_summary.md').write_text('original completed Evolve')
                save(folder / 'evolve/capability_validation.json', {'candidate_valid': True})
                paid.append({'round': number, 'role': 'evolve', 'pending_calls': 0,
                             'unknown_usage_calls': 0, 'total_tokens': 100})
        return experiment, spec, paid

    def test_later_rounds_resume_the_first_unfinished_step_without_replay(self):
        for completed in range(2, 7):
            for measured in sorted({completed, min(completed + 1, 6)}):
                with self.subTest(completed=completed, measured=measured), tempfile.TemporaryDirectory() as directory:
                    experiment, spec, paid = self.ahe_round_fixture(Path(directory), measured, completed)
                    plan = audit_ahe_rounds(experiment, 'airline', spec, paid)
                    self.assertEqual(plan['start_iteration'], completed + 1)
                    self.assertEqual(plan['skip_eval'], measured > completed)
                    self.assertEqual(plan['completed_episodes_reused'], (measured - 1) * 2)
                    self.assertNotIn('reuse_completed_evolve', plan)

    def test_copy_final_train_reparents_job_and_preserves_ancestor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent, destination = root / 'parent', root / 'continued'
            experiment, spec, job = self.final_train_fixture(parent)
            audit = {'meta_paid_cache_reused': True, 'meta_states': {}, 'ahe_resume': {}}
            save(parent / 'protocol.json', {})
            save(parent / 'setup_smoke.json', {'passed': True})
            (parent / 'ledgers').mkdir()
            for domain in ('airline', 'retail', 'telecom'):
                state = {'rounds': [{'iteration': 1}], 'incumbent': 'baseline_zero.py'}
                audit['meta_states'][domain] = state
                audit['ahe_resume'][domain] = {'name': domain}
                save(parent / 'meta' / domain / 'state.json', state)
                (parent / 'ahe/experiments' / domain).mkdir(parents=True, exist_ok=True)
            audit['ahe_resume']['airline']['final_train_reused'] = audit_final_train(parent, 'airline', experiment, spec)
            original = read(parent / 'ahe/airline/final_train.json')
            copy_parent(parent, destination, audit)
            self.assertEqual(read(parent / 'ahe/airline/final_train.json'), original)
            self.assertEqual(read(destination / 'ahe/airline/final_train.json')['job'],
                             str(destination / job.relative_to(parent)))
            self.assertEqual(audit_final_train(destination, 'airline',
                destination / 'ahe/experiments/airline', spec)['completed_episodes_reused'], 2)

    def test_third_round_description_cache_preserves_paid_candidate_and_rejects_structural_change(self):
        import copy
        import shutil
        import yaml
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment, spec, paid = self.ahe_round_fixture(root, 3, 2)
            third = experiment / 'runs/iteration_003'
            private = root / 'episodes/paid/experiment'
            shutil.copytree(third / 'input/workspace', private / 'runs/iteration_003/input/workspace')
            shutil.copytree(third / 'input/workspace', private / 'workspace')
            name = next((private / 'workspace/tool_descriptions').glob('*.yaml')).relative_to(private / 'workspace')
            path = private / 'workspace' / name
            value = yaml.safe_load(path.read_text())
            parameter = next(iter(value['input_schema']['properties'].values()))
            parameter['description'] = 'Original paid parameter guidance.'
            path.write_text(yaml.safe_dump(value))
            save(private / 'runs/iteration_003/evolve/evolve_trace.json', {'original_paid': True})
            save(private.parent / 'evolve_request.json', {'iteration': 3,
                 'config': {'native_backend': {'domain': 'airline'}}})
            save(private.parent / 'evolve_result.json', {'result': 'original paid result'})
            save(third / 'evolve/capability_rejection.json', {'candidate_promoted': False,
                 'error': 'restricted tool descriptions must preserve names, schemas and dispatch settings',
                 'private_candidate': str(private / 'workspace')})
            (third / 'evolve/isolated_process.log').write_text('[evolve] Evolve agent completed')
            details = ['native_a', 'native_b']
            save(third / 'debugger/evidence_projection.json', {'current_iteration': 3,
                 'detailed_tasks_max': 10, 'iterations': [{'iteration': 'iteration_003', 'selected_tasks': details}]})
            (third / 'input/analysis/detail').mkdir(parents=True)
            (third / 'input/analysis/overview.md').write_text('# Debugger Analysis Overview — Iteration 3\noriginal report')
            for task in details:
                (third / 'input/analysis/detail' / (task + '.md')).write_text('original paid analysis')
            for role in ('debugger', 'evolve'):
                paid.append({'round': 3, 'role': role, 'pending_calls': 0,
                             'unknown_usage_calls': 0, 'total_tokens': 100})
            before = workspace_hash(private / 'workspace')
            with patch('ahe_native.isolation.BASE', root):
                plan = audit_ahe_rounds(experiment, 'airline', spec, paid)
                self.assertEqual(plan['start_iteration'], 3)
                self.assertTrue(plan['skip_eval'])
                self.assertEqual(plan['reuse_completed_evolve']['iteration'], 3)
                self.assertEqual(plan['reuse_completed_evolve']['candidate_workspace_sha256'], before)
                self.assertEqual(len(plan['debugger_analysis_reused']), 3)
                self.assertEqual(workspace_hash(private / 'workspace'), before)
                changed = copy.deepcopy(value)
                changed['input_schema']['properties']['new_parameter'] = {'type': 'string'}
                path.write_text(yaml.safe_dump(changed))
                with self.assertRaisesRegex(ValueError, 'descriptions'):
                    audit_ahe_rounds(experiment, 'airline', spec, paid)

    def test_later_round_audit_rejects_partial_changed_unknown_and_future_paid_work(self):
        for case in ('partial', 'snapshot', 'duplicate', 'pending', 'unknown', 'future'):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                measured = 2 if case in ('partial', 'future') else 3
                experiment, spec, paid = self.ahe_round_fixture(Path(directory), measured, measured)
                third = experiment / 'runs/iteration_003'
                if case == 'partial':
                    save(third / 'input/benchmark/.native_pending/job/native_a/record.json', {'task_id': 'a', 'trial': 0})
                elif case == 'snapshot':
                    (third / 'input/workspace/systemprompt.md').write_text('changed snapshot')
                elif case == 'duplicate':
                    save(third / 'input/benchmark/20261005_job/native_duplicate/native_record.json', {'task_id': 'a', 'trial': 0})
                elif case == 'future':
                    paid.append({'round': 5, 'role': 'evolve'})
                else:
                    paid[-1]['pending_calls' if case == 'pending' else 'unknown_usage_calls'] = 1
                with self.assertRaises(RuntimeError):
                    audit_ahe_rounds(experiment, 'airline', spec, paid)

    def test_failed_domain_is_reported_before_other_domains_finish(self):
        with tempfile.TemporaryDirectory() as directory:
            comparison = Comparison.__new__(Comparison)
            comparison.root = Path(directory)
            comparison.abort, comparison.ready = threading.Event(), threading.Event()
            comparison.ready.set()
            comparison.first_round = dict.fromkeys(range(6), True)
            comparison.run_meta, comparison.run_ahe = Mock(), Mock()
            comparison.inspect_first_ahe = Mock()
            failure = Mock()
            failure.done.return_value = True
            failure.exception.return_value = ValueError('fixture compatibility failure')
            pending = Mock()
            pending.done.return_value = False
            pending.exception.return_value = None
            finished = Mock()
            finished.done.return_value = True
            finished.exception.return_value = None

            def observe():
                errors = read(comparison.root / 'errors.json')
                self.assertEqual(errors['meta:airline']['type'], 'ValueError')
                self.assertFalse(pending.done())
                pending.done.return_value = True

            comparison.report = Mock(side_effect=observe)
            with patch('evolution.tau_comparison.run.ThreadPoolExecutor') as executor, \
                    patch('evolution.tau_comparison.run.time.sleep'):
                executor.return_value.__enter__.return_value.submit.side_effect = [failure, pending, *([finished] * 4)]
                with self.assertRaisesRegex(RuntimeError, 'training failed'):
                    comparison.training()
            self.assertTrue(comparison.abort.is_set())
            comparison.report.assert_called_once()

    def test_original_evolve_cache_requires_complete_usage_and_exact_input_domain(self):
        from ahe_native.capabilities import restricted_policy
        import shutil
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment = root / 'host'
            second = experiment / 'runs/iteration_002'
            private = root / 'episodes/paid/experiment'
            baseline = AHE_NATIVE / 'baselines/airline'
            shutil.copytree(baseline, second / 'input/workspace')
            shutil.copytree(baseline, private / 'runs/iteration_002/input/workspace')
            shutil.copytree(baseline, private / 'workspace')
            save(private / 'runs/iteration_002/evolve/evolve_trace.json', {'original_paid': True})
            save(private.parent / 'evolve_request.json', {'iteration': 2, 'config': {'native_backend': {'domain': 'airline'}}})
            save(private.parent / 'evolve_result.json', {'result': 'paid original result'})
            save(second / 'evolve/capability_rejection.json', {'candidate_promoted': False,
                'error': 'restricted getattr accepts public hook fields only',
                'private_candidate': str(private / 'workspace')})
            (second / 'evolve/isolated_process.log').write_text('[evolve] Evolve agent completed')
            usage = {'pending_calls': 0, 'unknown_usage_calls': 0, 'total_tokens': 100}
            with patch('ahe_native.isolation.BASE', root):
                cache = audit_completed_evolve(experiment, 'airline', [usage])
                self.assertEqual(cache['iteration'], 2)
                self.assertEqual(cache['private_experiment'], str(private))
                for field in ('pending_calls', 'unknown_usage_calls'):
                    with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, 'complete paid receipts'):
                        audit_completed_evolve(experiment, 'airline', [{**usage, field: 1}])
                with self.assertRaisesRegex(RuntimeError, 'iteration/domain/input'):
                    audit_completed_evolve(experiment, 'retail', [usage])
                (private / 'runs/iteration_002/input/workspace/systemprompt.md').write_text('changed input')
                with self.assertRaisesRegex(RuntimeError, 'iteration/domain/input'):
                    audit_completed_evolve(experiment, 'airline', [usage])

    def test_date_imports_allowed_without_relaxing_private_or_execution_access(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'candidate.py'
            for source in ('import datetime\ndatetime.datetime.strptime("2026-10-05", "%Y-%m-%d")',
                           'from datetime import datetime, timedelta\ndatetime(2026, 10, 5) + timedelta(days=1)'):
                path.write_text(source)
                with patch.object(meta_port.subprocess, 'run') as process:
                    meta_port.validate_candidate(path, 'airline')
                    process.assert_called_once()
            for source in ('import os', 'from datetime import __builtins__',
                           'object.__setattr__(obj, "x", 1)', 'import datetime\nopen("hidden")'):
                with self.subTest(source=source):
                    path.write_text(source)
                    with patch.object(meta_port.subprocess, 'run') as process, self.assertRaises(ValueError):
                        meta_port.validate_candidate(path, 'airline')
                    process.assert_not_called()

    def test_date_repair_reuses_original_workflow_and_invalidates_only_changed_branch(self):
        for acceptance in (None, 3, 4, 6):
            with self.subTest(acceptance=acceptance), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                run = root / 'meta/airline'
                work = run / 'workspace'
                (work / 'agents').mkdir(parents=True)
                (work / 'agents/kept.py').write_text('identity')
                (work / 'upstream_SKILL.md').write_text(
                    (meta_port.HISTORICAL / 'upstream_reference/.claude/skills/meta-harness-terminal-bench-2/SKILL.md').read_text())
                spec = {'batches': [{'task_ids': ['a', 'b']} for _ in range(6)], 'train_trials': 1}
                baseline = {'n': 2, 'episodes': 2, 'successes': 1, 'accuracy': .5,
                            'infra_errors': 0, 'per_task': {'a': [1], 'b': [0]}}
                rows, candidates = [{'iteration': 1}, {'iteration': 2}], {}
                for i in range(3, 7):
                    source = work / 'agents' / f'date_{i}.py'
                    source.write_text('import datetime')
                    candidates[i] = digest(source)
                    candidate = {'name': f'date_{i}', 'import_path': f'agents.date_{i}:AgentHarness'}
                    save(run / f'pending_{i:02d}.json', {'candidates': [candidate]})
                    save(run / f'preflight_{i:02d}.json', {'passed': False})
                    save(run / f'proposer_{i:02d}/receipt.json', {'paid': True})
                    save(work / f'logs/round_{i:02d}/old.json', {'old': True})
                    save(work / f'jobs/round_{i:02d}/old.json', {'old': True})
                    save(run / f'evals/batch_{i-2:02d}_baseline/summary.json', baseline)
                    save(work / f'evidence/batch_{i-2:02d}_baseline/summary.json', baseline)
                    rows.append({'iteration': i, 'train_batch': i - 2, 'parent': 'kept.py',
                                 'candidate': candidate, 'candidate_score': None})
                original = {'rounds': rows, 'incumbent': 'kept.py', 'batch': 4,
                            'baseline_phase': 'batch_04_baseline', 'status': 'training_complete'}
                save(run / 'state.json', original)
                ledger = Mock()
                ledger.summarize.side_effect = lambda domain, i: {'accounting_complete': True,
                    'total_tokens': i * 100, 'by_role': dict.fromkeys(('parent', 'analysis', 'implementation'))}
                port = meta_port.MetaPort(root, 'airline', spec, meta_port.upstream('airline'),
                                         'unused', ledger, 'unused', 'unused')

                def phase(split, batch, name, candidate, number):
                    if name.endswith('_baseline'):
                        return read(run / 'evals' / name / 'summary.json')
                    score = {**baseline, 'successes': 2 if number == acceptance else 1,
                             'accuracy': 1 if number == acceptance else .5,
                             'per_task': {'a': [1], 'b': [int(number == acceptance)]}}
                    save(run / 'evals' / name / 'summary.json', score)
                    return score

                port.phase = Mock(side_effect=phase)
                with patch.object(meta_port, 'validate_candidate'), \
                     patch.object(meta_port, 'isolated_proposal') as paid, redirect_stdout(io.StringIO()):
                    port.repair_date_rejections(candidates)
                paid.assert_not_called()
                last = acceptance or 6
                self.assertEqual(len(port.state['rounds']), last)
                self.assertEqual(sum(c.args[2].startswith('candidate_')
                                     for c in port.phase.call_args_list), last - 2)
                self.assertEqual(read(run / 'date_import_repair/state_before_repair.json'), original)
                self.assertEqual(port.state['rounds'][-1]['usage']['total_tokens'], last * 100)
                self.assertEqual(port.state['incumbent'], f'date_{last}.py' if acceptance else 'kept.py')
                if acceptance:
                    self.assertEqual(port.state['baseline_phase'], f'candidate_{last:02d}')
                    for i in range(last + 1, 7):
                        self.assertFalse((run / f'pending_{i:02d}.json').exists())
                        self.assertFalse((work / f'agents/date_{i}.py').exists())
                        self.assertTrue((run / 'date_import_repair' / f'pending_{i:02d}.json').exists())
                    self.assertFalse((run / f'evals/batch_{last-1:02d}_baseline').exists())
                else:
                    self.assertEqual(port.state['status'], 'training_complete')

    def test_meta_host_supplies_the_same_dedicated_uid_as_enter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work = root / 'work'
            (work / 'agents').mkdir(parents=True)
            (work / 'agents/baseline.py').write_text('identity')
            isolated = root / 'isolation'
            case = isolated / 'cases/case'
            (case / 'workspace/agents').mkdir(parents=True)
            (case / 'workspace/agents/baseline.py').write_text('identity')
            (case / 'output').mkdir()
            save(case / 'output/result.json', {'success': False})
            with patch('ahe_native.isolation.case_root', return_value=(isolated, case, '/cases/case', 23456)), \
                 patch('evolution.tau_comparison.meta_port.os.chown'), \
                 patch('evolution.tau_comparison.meta_port.subprocess.run', return_value=SimpleNamespace(returncode=1)):
                result = meta_port.isolated_proposal(root, 'retail', 1, work, 'public prompt', 'public contract',
                    'http://localhost/v1', work / 'logs/round_01/pending_eval.json')
            self.assertFalse(result)
            request = json.loads((case / 'request.json').read_text())
            self.assertEqual(request['isolated_uid'], 23456)
            self.assertEqual(request['workspace'], '/cases/case/workspace')
            self.assertNotIn('api_key', request)

    def test_unknown_ancestor_usage_cannot_be_restarted_or_estimated(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'ledger.jsonl'
            path.write_text(json.dumps({'event': 'started', 'request_id': 'id'}) + '\n')
            with self.assertRaisesRegex(RuntimeError, 'usage incomplete'):
                ledger_complete(path)
            with path.open('a') as stream:
                stream.write(json.dumps({'event': 'finished', 'request_id': 'id', 'usage': None}) + '\n')
            with self.assertRaisesRegex(RuntimeError, 'usage incomplete'):
                ledger_complete(path)

    def test_complete_raw_usage_is_preserved_and_summed_once(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'ledger.jsonl'
            path.write_text('\n'.join(json.dumps(r) for r in [
                {'event': 'started', 'request_id': 'id'},
                {'event': 'finished', 'request_id': 'id', 'usage': {'total_tokens': 123}}]) + '\n')
            before = path.read_bytes()
            self.assertEqual(ledger_complete(path), {'attempts': 1, 'tokens': 123})
            self.assertEqual(path.read_bytes(), before)

    def test_authorized_historical_estimate_allows_reuse_without_hiding_missing_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'mr_requests.jsonl'
            ledger = TokenLedger(path)
            scope = dict(experiment='original', domain='telecom', round=2, role='evolve')
            ident = ledger.start(scope, model='deepseek-v4-flash', attempt=1, stream=True)
            ledger.finish(ident, status=200)
            before = path.read_bytes()
            save(path.with_suffix('.estimates.json'), [{'request_id': ident, 'user_authorized': True,
                'basis': 'synthetic neighboring calls', 'usage': {'input_tokens': 100, 'output_tokens': 7},
                'receipt_sha256': hashlib.sha256(json.dumps(ledger.rows, sort_keys=True,
                    separators=(',', ':')).encode()).hexdigest()}])
            costs = ledger_complete(path)
            self.assertEqual((costs['tokens'], costs['estimated_tokens'], costs['tokens_including_estimates']),
                             (0, 107, 107))
            result = select_optimizer_groups(ledger.summary(), 'original', 'telecom', 2)
            self.assertFalse(result['inference_accounting_complete'])
            self.assertTrue(result['accounting_usable_with_estimates'])
            self.assertEqual(result['budget_tokens_including_estimates'], 107)
            self.assertEqual(path.read_bytes(), before)
            extra = ledger.start(scope, model='deepseek-v4-flash', attempt=1, stream=True)
            ledger.finish(extra, status=200)
            self.assertFalse(select_optimizer_groups(ledger.summary(), 'original')['accounting_usable_with_estimates'])
            with self.assertRaisesRegex(RuntimeError, 'usage incomplete'):
                ledger_complete(path)

    def test_unscoped_evolve_is_attributed_only_by_exact_original_trace_without_raw_edits(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            path = parent / 'ledgers/mr_requests.jsonl'
            ledger = TokenLedger(path)
            scope = {'experiment': 'unscoped', 'domain': 'unknown', 'round': None, 'role': 'unknown'}
            ident = ledger.start(scope, model='deepseek-v4-flash', attempt=1, stream=True)
            ledger.finish(ident, status=200, usage={'prompt_tokens': 100, 'completion_tokens': 5})
            trace = parent / 'ahe/experiments/original/runs/iteration_001/evolve/nexau_in_memory_tracer.cleaned.json'
            evidence = {'timestamp': ledger.rows[0]['utc_unix'] - 1, 'latency': 2000,
                'generation_count': 1, 'total_tokens': 105, 'subagents': [], 'model': 'deepseek-v4-flash'}
            save(trace, evidence)
            before = path.read_bytes()
            correction = audit_unscoped_evolve(parent, ledger.summary(), 'original')
            result = select_optimizer_groups(ledger.summary(), 'original', 'telecom', 1, correction)
            self.assertEqual(result['by_role']['evolve']['total_tokens'], 105)
            self.assertTrue(result['inference_accounting_complete'])
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(ledger.rows[0]['role'], 'unknown')
            for field, value in [('total_tokens', 104), ('generation_count', 2), ('timestamp', 0)]:
                save(trace, {**evidence, field: value})
                with self.assertRaisesRegex(RuntimeError, 'uniquely attributed'):
                    audit_unscoped_evolve(parent, ledger.summary(), 'original')
            extra = ledger.start(scope, model='deepseek-v4-flash', attempt=1, stream=True)
            ledger.finish(extra, status=200, usage={'prompt_tokens': 10, 'completion_tokens': 1})
            with self.assertRaisesRegex(RuntimeError, 'ambiguous attribution'):
                select_optimizer_groups(ledger.summary(), 'original', correction=correction)

    def test_paid_meta_attempts_are_not_blindly_replayed(self):
        with tempfile.TemporaryDirectory() as root:
            parent = Path(root)
            save(parent / 'meta/retail/state.json', {'rounds': []})
            (parent / 'meta/retail/flash_requests.jsonl').write_text('{"paid":"attempt"}\n')
            with self.assertRaisesRegex(RuntimeError, 'paid requests'):
                audit_meta_zero_api(parent, 'retail')

    def test_paid_qualified_meta_cache_reused_only_with_complete_matching_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            run = parent / 'meta/telecom'
            source = run / 'workspace/agents/candidate.py'
            source.parent.mkdir(parents=True)
            source.write_text('public candidate')
            baseline_source = source.parent / 'baseline_zero.py'
            baseline_source.write_text('identity')
            state = {'rounds': []}
            save(run / 'state.json', state)
            pending = run / 'pending_01.json'
            save(pending, {'candidates': [{'name': 'candidate'}]})
            save(run / 'proposer_01/receipt.json', {'success': True,
                'delegations': ['analysis', 'implementation'], 'pending_sha256': digest(pending),
                'candidate_sha256': digest(source)})
            save(run / 'preflight_01.json', {'passed': True, 'sha256': digest(source)})
            spec = {'batches': [{'task_ids': ['train']}], 'train_trials': 1}
            save(run / 'evals/batch_00_baseline/summary.json', {'infra_errors': 0,
                'per_task': {'train': [0]}, 'plugin_hash': digest(baseline_source), 'episodes': 1,
                'config': {'domain': 'telecom', 'split': 'train', 'task_ids': ['train'], 'trials': 1,
                    'candidate_sha256': digest(baseline_source), 'solver_total_context': 32768, 'solver_output': 4096}})
            ledger = Mock()
            usage = {'accounting_complete': True, 'by_role': {r: {'total_tokens': 10}
                for r in ('parent', 'analysis', 'implementation')}}
            ledger.summarize.return_value = usage
            ledger.rows.return_value = [{'round': 1}]
            with patch('evolution.tau_comparison.recovery.transport_modules', return_value=(Mock(return_value=ledger), None)):
                self.assertEqual(audit_meta_cached(parent, 'telecom', spec), state)
                usage['accounting_complete'] = False
                with self.assertRaisesRegex(RuntimeError, 'complete first-round'):
                    audit_meta_cached(parent, 'telecom', spec)
                usage['accounting_complete'] = True
                source.write_text('tampered')
                with self.assertRaisesRegex(RuntimeError, 'candidate or preflight'):
                    audit_meta_cached(parent, 'telecom', spec)

    def test_six_paid_meta_rounds_preserve_failed_proposals_and_exact_decisions(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            run = parent / 'meta/retail'
            agents = run / 'workspace/agents'
            agents.mkdir(parents=True)
            (agents / 'baseline_zero.py').write_text('identity')
            spec = {'batches': [{'task_ids': [f'train_{i}']} for i in range(6)], 'train_trials': 1}
            state = {'rounds': [], 'incumbent': 'baseline_zero.py', 'batch': 0, 'status': 'training_complete'}
            phase, usages = 'batch_00_baseline', {}
            for number, outcome in enumerate((1, None, 'invalid', 0, 1, 0), 1):
                batch, incumbent = state['batch'], state['incumbent']
                baseline_path = run / 'evals' / phase / 'summary.json'
                if not baseline_path.exists():
                    save(baseline_path, {'infra_errors': 0, 'missing_episodes': 0, 'episodes': 1,
                        'per_task': {f'train_{batch}': [0]}, 'successes': 0, 'accuracy': 0,
                        'plugin_hash': digest(agents / incumbent), 'config': {
                            'domain': 'retail', 'split': 'train', 'task_ids': spec['batches'][batch]['task_ids'],
                            'trials': 1, 'candidate_sha256': digest(agents / incumbent),
                            'solver_total_context': 32768, 'solver_output': 4096}})
                baseline = read(baseline_path)
                roles = ('parent', 'analysis') if outcome is None else ('parent', 'analysis', 'implementation')
                usages[number] = {'accounting_complete': True,
                                  'by_role': {r: {'total_tokens': number} for r in roles}}
                candidate, score = None, None
                if outcome is None:
                    result = {'success': False, 'error': 'original workflow incomplete'}
                    save(run / f'proposer_{number:02d}/traces/result.json', result)
                    save(run / f'proposer_{number:02d}/rejection.json', {'reason': 'API proposer failed', 'result': result})
                else:
                    candidate = {'name': f'candidate_{number}'}
                    source = agents / (candidate['name'] + '.py')
                    source.write_text(f'public mechanism {number}')
                    pending = run / f'pending_{number:02d}.json'
                    save(pending, {'candidates': [candidate]})
                    save(run / f'proposer_{number:02d}/receipt.json', {'success': True,
                        'delegations': ['analysis', 'implementation'], 'pending_sha256': digest(pending),
                        'candidate_sha256': digest(source)})
                    save(run / f'preflight_{number:02d}.json', {'passed': False, 'error_type': 'ValueError'}
                         if outcome == 'invalid' else {'passed': True, 'sha256': digest(source)})
                    if outcome != 'invalid':
                        score = {**baseline, 'successes': outcome, 'accuracy': outcome,
                            'per_task': {f'train_{batch}': [outcome]}, 'plugin_hash': digest(source),
                            'config': {**baseline['config'], 'candidate_sha256': digest(source)}}
                        save(run / f'evals/candidate_{number:02d}/summary.json', score)
                accepted = score is not None and score['successes'] > baseline['successes']
                if accepted:
                    state['incumbent'], phase = candidate['name'] + '.py', f'candidate_{number:02d}'
                elif number < 6:
                    state['batch'] += 1
                    phase = f'batch_{state["batch"]:02d}_baseline'
                state['rounds'].append({'iteration': number, 'train_batch': batch, 'parent': incumbent,
                    'usage': usages[number], 'baseline_successes': baseline['successes'],
                    'baseline_score': baseline['accuracy'], 'candidate': candidate,
                    'candidate_successes': score['successes'] if score else None,
                    'candidate_score': score['accuracy'] if score else None, 'accepted': accepted,
                    'retained_candidate': state['incumbent']})
            save(run / 'state.json', state)
            ledger = Mock()
            ledger.rows.return_value = [{'round': i} for i in range(1, 7)]
            ledger.summarize.side_effect = lambda domain, number: usages[number]
            with patch('evolution.tau_comparison.recovery.transport_modules', return_value=(Mock(return_value=ledger), None)):
                self.assertEqual(audit_meta_cached(parent, 'retail', spec), state)
                self.assertEqual(len(state['rounds']), 6)
                for field, value in [('candidate_score', 1), ('accepted', True), ('train_batch', 1)]:
                    with self.subTest(field=field):
                        altered = read(run / 'state.json')
                        altered['rounds'][1][field] = value
                        save(run / 'state.json', altered)
                        with self.assertRaises(RuntimeError):
                            audit_meta_cached(parent, 'retail', spec)
                        save(run / 'state.json', state)
                ledger.rows.return_value.append({'round': 7})
                with self.assertRaisesRegex(RuntimeError, 'uncommitted paid'):
                    audit_meta_cached(parent, 'retail', spec)

    def test_debugger_cache_requires_complete_paid_receipts_and_selected_ten_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            second = Path(directory) / 'iteration_002'
            self.assertIsNone(audit_debugger_cache(second, []))
            analysis = second / 'input/analysis'
            details = [f'native_{i}' for i in range(10)]
            save(second / 'debugger/evidence_projection.json', {'current_iteration': 2, 'detailed_tasks_max': 10,
                'iterations': [{'iteration': 'iteration_002', 'selected_tasks': details}]})
            (analysis / 'detail').mkdir(parents=True)
            (analysis / 'overview.md').write_text('# Debugger Analysis Overview — Iteration 2\noriginal report')
            for name in details:
                (analysis / 'detail' / (name + '.md')).write_text('completed original detailed analysis')
            paid = [{'pending_calls': 0, 'unknown_usage_calls': 1,
                     'rate_limit_rejections_without_usage': 1, 'total_tokens': 100}]
            receipt = audit_debugger_cache(second, paid)
            self.assertEqual(len(receipt), 11)
            for field in ('pending_calls', 'unknown_usage_calls'):
                with self.subTest(field=field):
                    changed = {**paid[0], field: paid[0][field] + 1}
                    with self.assertRaisesRegex(RuntimeError, 'incomplete'):
                        audit_debugger_cache(second, [changed])
            with self.assertRaisesRegex(RuntimeError, 'incomplete'):
                audit_debugger_cache(second, [])
            (analysis / 'detail/native_0.md').write_text('')
            with self.assertRaisesRegex(RuntimeError, 'incomplete'):
                audit_debugger_cache(second, paid)

    def test_completed_meta_round_releases_only_five_remaining_proposals(self):
        comparison = Comparison.__new__(Comparison)
        comparison.continuation = {}
        row = {'candidate_score': .3, 'candidate_infra_errors': 0,
               'usage': {'accounting_complete': True, 'by_role': dict.fromkeys(
                   ('parent', 'analysis', 'implementation'))}}
        port = Mock()
        port.state = {'rounds': [row]}
        port.round.side_effect = lambda: port.state['rounds'].append({'next': True})
        comparison.meta_ports = {'retail': port}
        comparison.lock, comparison.first_round = threading.Lock(), {}
        comparison.ready, comparison.abort = Mock(), Mock()
        comparison.abort.is_set.return_value = True
        with self.assertRaisesRegex(RuntimeError, 'integration gate failed'):
            comparison.run_meta('retail')
        port.round.assert_not_called()
        comparison.abort.is_set.return_value = False
        comparison.run_meta('retail')
        self.assertEqual(port.round.call_count, 5)
        self.assertEqual(len(port.state['rounds']), 6)
        comparison.run_meta('retail')
        self.assertEqual(port.round.call_count, 5)

    def test_baseline_repair_uses_original_metadata_functions_without_optimizer_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment = root / 'ahe/experiments/original'
            first = experiment / 'runs/iteration_001'
            (first / 'input/workspace').mkdir(parents=True)
            summary = first / 'evolve/evolve_summary.md'
            summary.parent.mkdir()
            summary.write_text('original paid evolve output')
            job = first / 'input/benchmark/repaired'
            save(job / 'native_1/native_record.json', {'task_id': 'train', 'trial': 0,
                'success': 1, 'infrastructure_error': False})
            comparison = Comparison.__new__(Comparison)
            comparison.root, comparison.names = root, {'telecom': 'original'}
            comparison.protocol = {'domains_spec': {'telecom': {
                'batches': [{'task_ids': ['train']}], 'train_trials': 1}}}
            u = comparison.ahe_module = Mock()
            u.run_harbor.return_value = job
            u.compute_stats.return_value = {'task_results': {'train': 'pass'}, 'pass_rate': 1.,
                                            'n_pass': 1, 'n_total': 1, 'n_exception': 0}
            with patch('ahe_native.capabilities.validate_workspace'):
                comparison.repair_ahe_baseline('telecom', {'native_backend': {}}, {'episodes': []})
            u.run_single_experiment.assert_not_called()
            u.rollback_experiment_metadata.assert_called_once_with(experiment, 1)
            u.update_best_ever.assert_called_once_with(experiment, 1, u.compute_stats.return_value)
            u.update_history_after.assert_called_once_with(experiment, 1, summary.read_text())
            self.assertFalse(read(experiment / 'baseline_repair_receipt.json')['optimizer_replayed'])

    def test_only_excluded_simulator_unknown_usage_can_be_retained(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'ledger.jsonl'
            path.write_text('\n'.join(json.dumps(r) for r in [
                {'event': 'started', 'request_id': 'id', 'role': 'user_simulator'},
                {'event': 'finished', 'request_id': 'id', 'usage': None}]) + '\n')
            self.assertEqual(ledger_complete(path)['excluded_simulator_or_judge_unknown_usage_calls'], 1)
            path.write_text(path.read_text().replace('user_simulator', 'evolve'))
            with self.assertRaisesRegex(RuntimeError, 'usage incomplete'):
                ledger_complete(path)

    def test_tau_deadline_does_not_change_agentbench(self):
        for domain in ('airline', 'retail', 'telecom'):
            self.assertEqual(episode_timeout(domain), 21600)
        for domain in ('alfworld', 'webshop', 'dbbench'):
            self.assertEqual(episode_timeout(domain), 1800)

    def test_two_provider_summaries_do_not_overwrite_each_other(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = TokenLedger(root / 'a.jsonl', root / 'A_USAGE.json')
            b = TokenLedger(root / 'b.jsonl', root / 'B_USAGE.json')
            a.publish()
            b.publish()
            self.assertEqual(read(root / 'A_USAGE.json'), a.summary())
            self.assertEqual(read(root / 'B_USAGE.json'), b.summary())

    def test_rate_limit_rejection_is_not_a_lost_inference_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'ledger.jsonl'
            ledger = TokenLedger(path)
            scope = {'experiment': 'original', 'domain': 'airline', 'round': 1, 'role': 'evolve'}
            rejected = ledger.start(scope, model='deepseek-v4-flash', attempt=1, stream=True)
            ledger.finish(rejected, status=429)
            success = ledger.start(scope, model='deepseek-v4-flash', attempt=2, stream=True)
            ledger.finish(success, status=200, usage={'prompt_tokens': 100, 'completion_tokens': 5})
            result = select_optimizer_groups(ledger.summary(), 'original', 'airline', 1)
            self.assertFalse(result['accounting_complete'])
            self.assertTrue(result['inference_accounting_complete'])
            self.assertEqual(result['unknown_usage_calls'], 1)
            self.assertEqual(result['total_tokens'], 105)
            before = path.read_bytes()
            self.assertEqual(ledger_complete(path)['rate_limit_rejections_without_usage'], 1)
            self.assertEqual(path.read_bytes(), before)
            lost = ledger.start(scope, model='deepseek-v4-flash', attempt=1, stream=True)
            ledger.finish(lost, status=200)
            self.assertFalse(select_optimizer_groups(ledger.summary(), 'original')['inference_accounting_complete'])
            with self.assertRaisesRegex(RuntimeError, 'usage incomplete'):
                ledger_complete(path)

    def test_only_a_dead_unstarted_waiting_launcher_can_be_reloaded(self):
        for state in ('dead_empty', 'live', 'paid', 'approved_date_repair', 'unapproved_date_repair'):
            with self.subTest(state=state), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                run = root / 'evolution/experiments/continuation'
                save(run / 'status.json', {'phase': 'waiting_for_ancestor_quiescence', 'pid': 123,
                                          'new_api_calls': 0})
                save(run / 'launch.json', {'pid': 123})
                if state.endswith('date_repair'):
                    save(run / 'status.json', {'phase': 'stopped_pending_import_contract_choice',
                                              'pid': 123, 'new_api_calls': 0})
                if state == 'paid':
                    (run / 'ledgers').mkdir()
                with patch('evolution.tau_comparison.run.ROOT', root), \
                     patch('evolution.tau_comparison.recovery.live_process', return_value=state == 'live'), \
                     patch('evolution.tau_comparison.run.subprocess.Popen') as process, \
                     patch('sys.argv', ['run', '--name', 'continuation', '--resume-from', 'ancestor', '--launch',
                                        *(['--repair-date-imports'] if state == 'approved_date_repair' else [])]):
                    if state in ('dead_empty', 'approved_date_repair'):
                        process.return_value.pid = 456
                        main()
                        archive_name = ('launch_before_date_import_resume.json' if state == 'approved_date_repair'
                                        else 'waiting_launch_before_accounting_fix.json')
                        self.assertEqual(read(run / archive_name)['pid'], 123)
                        self.assertEqual(read(run / 'launch.json')['pid'], 456)
                    else:
                        with self.assertRaises(FileExistsError):
                            main()
                        process.assert_not_called()

    def cached_request(self, root):
        workspace = root / 'workspace'
        workspace.mkdir()
        (workspace / 'code_agent.yaml').write_text('frozen baseline')
        episode = root / 'old_episode'
        (episode / 'agent').mkdir(parents=True)
        save(episode / 'request.json', {'domain': 'telecom', 'split': 'train', 'task_id': 'task',
            'trial': 0, 'workspace': str(workspace), 'agent_config': 'code_agent.yaml'})
        save(episode / 'record.json', {'task_id': 'task', 'trial': 0, 'success': 0,
            'infrastructure_error': False, 'status': 'context_limit_32768',
            'messages': [{'role': 'user', 'content': 'public request'}]})
        return {'domain': 'telecom', 'split': 'train', 'workspace': str(workspace),
            'agent_config': 'code_agent.yaml', 'output_root': str(root / 'new_job'), 'k': 1, 'concurrency': 50,
            'reuse_completed_episodes': {'workspace_sha256': workspace_hash(workspace), 'episodes': [
                {'folder': str(episode), 'files': {name: digest(episode / name)
                                                  for name in ('request.json', 'record.json')}}]}}

    def test_complete_zero_score_episode_reused_without_model_call(self):
        with tempfile.TemporaryDirectory() as directory:
            request = self.cached_request(Path(directory))
            with patch('ahe_native.backend.validate_request', return_value=['task']), \
                 patch('ahe_native.backend.subprocess.run') as process:
                job = execute(request)
            process.assert_not_called()
            record = read(next(job.glob('native_*/native_record.json')))
            self.assertEqual(record['success'], 0)
            self.assertEqual(record['status'], 'context_limit_32768')
            self.assertEqual(read(job / 'episode_reuse_receipt.json')['new_episodes'], 0)

    def test_changed_cached_record_or_workspace_rejected_before_any_paid_call(self):
        for tamper in ('record', 'workspace'):
            with self.subTest(tamper=tamper), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                request = self.cached_request(root)
                target = root / 'old_episode/record.json' if tamper == 'record' else root / 'workspace/code_agent.yaml'
                target.write_text('tampered')
                with patch('ahe_native.backend.validate_request', return_value=['task']), \
                     patch('ahe_native.backend.subprocess.run') as process:
                    with self.assertRaisesRegex(ValueError, 'changed|different workspace'):
                        execute(request)
                process.assert_not_called()

    def test_transport_repair_reuses_exact_original_meta_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent, destination = root / 'ancestor', root / 'continuation'
            audit = {'meta_states': {}, 'ahe_resume': {}}
            save(parent / 'protocol.json', {'original': True})
            save(parent / 'setup_smoke.json', {'passed': True})
            (parent / 'ledgers').mkdir()
            for domain in ('airline', 'retail', 'telecom'):
                state = {'rounds': [{'candidate': None}], 'batch': 1, 'baseline_phase': None,
                         'incumbent': 'baseline_zero.py'}
                audit['meta_states'][domain] = state
                audit['ahe_resume'][domain] = {'name': domain}
                save(parent / 'meta' / domain / 'state.json', state)
                save(parent / 'meta' / domain / 'workspace/history.json', state['rounds'])
                (parent / 'ahe/experiments' / domain).mkdir(parents=True)
            before = read(parent / 'meta/retail/state.json')
            copy_parent(parent, destination, audit)
            recovered = read(destination / 'meta/retail/state.json')
            self.assertEqual(recovered['batch'], 0)
            self.assertEqual(recovered['baseline_phase'], 'batch_00_baseline')
            self.assertEqual(recovered['rounds'], [])
            self.assertEqual(read(parent / 'meta/retail/state.json'), before)

    def test_paid_recovery_keeps_measured_rounds_and_archives_only_telecom_measurements(self):
        with tempfile.TemporaryDirectory() as directory:
            parent, destination = Path(directory) / 'parent', Path(directory) / 'continued'
            audit = {'meta_paid_cache_reused': True, 'meta_states': {}, 'ahe_resume': {}}
            save(parent / 'protocol.json', {})
            save(parent / 'setup_smoke.json', {'passed': True})
            (parent / 'ledgers').mkdir()
            for domain in ('airline', 'retail', 'telecom'):
                state = {'rounds': [] if domain == 'telecom' else [{'iteration': 1, 'measured': True}],
                         'batch': 1 if domain == 'retail' else 0, 'status': 'running',
                         'baseline_phase': 'batch_00_baseline'}
                audit['meta_states'][domain] = state
                audit['ahe_resume'][domain] = {'name': domain}
                src = parent / 'meta' / domain
                save(src / 'state.json', state)
                save(src / 'workspace/history.json', state['rounds'])
                save(src / 'pending_01.json', {'paid_candidate': domain})
                save(src / 'proposer_01/receipt.json', {'paid': True})
                save(src / 'evals/batch_00_baseline/summary.json', {'legacy_measurement': True})
                (src / 'evals/candidate_01').mkdir()
                (parent / 'ahe/experiments' / domain).mkdir(parents=True)
            copy_parent(parent, destination, audit)
            for domain in ('airline', 'retail'):
                self.assertEqual(read(destination / 'meta' / domain / 'state.json'), audit['meta_states'][domain])
            tel = destination / 'meta/telecom'
            self.assertIsNone(read(tel / 'state.json')['baseline_phase'])
            self.assertTrue((tel / 'simulator_output_repair/batch_00_baseline/summary.json').exists())
            self.assertFalse((tel / 'evals/batch_00_baseline').exists())
            self.assertEqual(read(tel / 'pending_01.json'), {'paid_candidate': 'telecom'})
            self.assertTrue((tel / 'proposer_01/receipt.json').exists())
            self.assertEqual(read(parent / 'meta/telecom/state.json'), audit['meta_states']['telecom'])
            # A later stopped gate retains the now-complete Telecom measurement.
            audit['meta_states']['telecom']['rounds'] = [{'iteration': 1, 'measured': True}]
            another = Path(directory) / 'second_continuation'
            copy_parent(parent, another, audit)
            self.assertEqual(read(another / 'meta/telecom/state.json'), audit['meta_states']['telecom'])
            self.assertTrue((another / 'meta/telecom/evals/batch_00_baseline/summary.json').exists())
            self.assertFalse((another / 'meta/telecom/simulator_output_repair').exists())


if __name__ == '__main__':
    unittest.main()
