"""Original Meta loop/skill, native Tau transport and unchanged two-Agent shell."""
import argparse
import ast
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from .common import (ROOT, TAU, HISTORICAL, META_TRANSPORT, AHE_NATIVE, child_env,
                     digest, load_module, public_simulation, read, save, summarize_rows, user_model)


def prepare_workspace(root, domain):
    work = root / 'meta' / domain / 'workspace'
    if work.exists():
        raise FileExistsError('new Meta workspace required')
    (work / 'agents').mkdir(parents=True)
    (work / 'source_view').mkdir()
    shutil.copy2(HISTORICAL / domain / 'workspace/agents/baseline_zero.py', work / 'agents/baseline_zero.py')
    shutil.copy2(HISTORICAL / 'upstream_reference/.claude/skills/meta-harness-terminal-bench-2/SKILL.md',
                 work / 'upstream_SKILL.md')
    # Do not expose the released H2-H5 implementations embedded in domain sources.
    shutil.copy2(AHE_NATIVE / 'baselines' / domain / 'public_tool_schema.json', work / 'source_view/native_tools.json')
    (work / 'source_view/CONTRACT.md').write_text(
        'Export AgentHarness and register(). register() may patch plain tau2 domain Tools/environment '
        'or the native LLMAgent public message/tool interface. All released H2-H5 flags are disabled. '
        'Native agent module: tau2.agent.llm_agent (LLMAgent, LLMAgentState); '
        'domain modules: tau2.domains.' + domain + '.tools and .environment. '
        'The public native policy and tools occur in training conversations. '
        'No evaluator/user simulator, hidden task state/answers, filesystem/network or new model calls.\n')
    return work


def upstream(domain):
    directory = HISTORICAL / 'upstream_reference'
    sys.path.insert(0, str(directory))
    module = load_module('tau_original_meta_' + domain, directory / 'meta_harness.py')
    if module.run_evolve.__code__.co_filename != str(directory / 'meta_harness.py'):
        raise RuntimeError('must execute original Meta run_evolve')
    return module


def isolated_proposal(root, domain, iteration, workspace, prompt, contract, broker_url, pending):
    from ahe_native.isolation import case_root, copy_unchanged_or_atomic, enter, require_regular_workspace
    trace = root / 'meta' / domain / f'proposer_{iteration:02d}'
    trace.mkdir(parents=True, exist_ok=True)
    isolated_root, case, logical, uid = case_root(workspace, trace)
    worker = isolated_root / 'adapter/tau_meta/api_agent.py'
    worker.parent.mkdir(parents=True, exist_ok=True)
    copy_unchanged_or_atomic(META_TRANSPORT / 'api_agent.py', worker)
    logical_work = logical + '/workspace'
    public = case / 'workspace'
    frozen = {str(p.relative_to(workspace)): digest(p) for p in workspace.rglob('*')
              if p.is_file() and '__pycache__' not in p.parts}
    save(trace / 'frozen_inputs.json', frozen)
    for path in [public, *public.rglob('*')]:
        if path.is_dir():
            relative = path.relative_to(public)
            if not relative.parts or relative.parts[0] == 'agents' or (
                    relative.parts[0] == 'logs' and len(relative.parts) > 1 and
                    relative.parts[1] == f'round_{iteration:02d}'):
                os.chown(path, uid, uid)
                path.chmod(0o700)
    request = {'domain': domain, 'round': iteration, 'workspace': logical_work, 'isolated_uid': uid,
        'broker_url': broker_url, 'timeout_seconds': 5400,
        'prompt': prompt.replace(str(workspace), logical_work),
        'port_contract': contract.replace(str(workspace), logical_work)}
    save(case / 'request.json', request)
    (case / 'request.json').chmod(0o444)
    env = {'PATH': '/runtime/bin:/usr/bin:/bin', 'PYTHONPATH': '/adapter/tau_meta',
           'PYTHONDONTWRITEBYTECODE': '1', 'TMPDIR': '/tmp', 'NO_PROXY': '127.0.0.1,localhost'}
    with (trace / 'process.log').open('a') as log:
        process = subprocess.run(['/runtime/bin/python', '-u', '/adapter/tau_meta/api_agent.py',
            '--request', logical + '/request.json', '--output', logical + '/output'],
            preexec_fn=lambda: enter(str(isolated_root), logical_work, uid), env=env,
            stdout=log, stderr=subprocess.STDOUT, timeout=5520)
    require_regular_workspace(public)
    if (case / 'output').exists():
        shutil.copytree(case / 'output', trace / 'traces', dirs_exist_ok=True)
    changed = [name for name, sha in frozen.items() if not (public / name).is_file() or digest(public / name) != sha]
    if changed:
        save(trace / 'rejection.json', {'reason': 'frozen input mutation', 'paths': changed})
        return False
    result = read(case / 'output/result.json') if (case / 'output/result.json').exists() else {'success': False}
    if process.returncode or not result.get('success'):
        save(trace / 'rejection.json', {'reason': 'API proposer failed', 'result': result})
        return False
    parent = read(case / 'output/parent/session.json')
    if parent['delegations'] != ['analysis', 'implementation']:
        raise RuntimeError('original Meta analysis/implementation Agent workflow missing')
    staged_pending = public / pending.relative_to(workspace)
    if not staged_pending.is_file():
        return False
    document = read(staged_pending)
    if len(document.get('candidates', [])) != 1:
        raise ValueError('original Meta skill requires exactly one candidate')
    candidate = document['candidates'][0]
    name = candidate['name']
    if (not re.fullmatch('[a-z][a-z0-9_]*', name) or candidate['import_path'] != f'agents.{name}:AgentHarness'
            or f'agents/{name}.py' in frozen):
        raise ValueError('invalid or existing candidate path')
    source = public / 'agents' / (name + '.py')
    shutil.copy2(source, workspace / 'agents' / source.name)
    shutil.copy2(staged_pending, pending)
    save(trace / 'receipt.json', {'success': True, 'delegations': parent['delegations'],
        'candidate_sha256': digest(source), 'pending_sha256': digest(staged_pending),
        'api_shell_sha256': digest(META_TRANSPORT / 'api_agent.py'), 'credential_policy': 'EMPTY only'})
    return True


def validate_candidate(path, domain):
    tree = ast.parse(path.read_text())
    safe = {'re', 'json', 'copy', 'math', 'collections', 'difflib', 'string', 'typing',
            'ast', 'dataclasses', 'functools', 'itertools', 'statistics', 'datetime', '__future__'}
    tau = {'tau2', 'tau2.agent.llm_agent', 'tau2.data_model.message', 'tau2.environment.tool',
           f'tau2.domains.{domain}.tools', f'tau2.domains.{domain}.environment', f'tau2.domains.{domain}.data_model'}
    forbidden = {'open', 'eval', 'exec', '__import__', 'globals', 'locals', 'vars',
                 'get_tasks', 'task_id', 'reward_info', 'evaluator', 'user_simulator', 'DATA_DIR'}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(a.name not in safe | tau for a in node.names):
            raise ValueError('forbidden candidate import')
        if isinstance(node, ast.ImportFrom) and (node.level or node.module not in safe | tau or
                any(a.name == '*' or a.name.startswith('__') for a in node.names)):
            raise ValueError('forbidden candidate import')
        if isinstance(node, ast.Name) and node.id in forbidden:
            raise ValueError('forbidden private/runtime execution name')
        if isinstance(node, ast.Attribute) and (node.attr.startswith('__') or node.attr in forbidden or
                any(term in node.attr.lower() for term in ('harness', 'evaluator', 'reward_info'))):
            raise ValueError('forbidden private/released runtime reference')
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and any(term in node.value for term in
                ('h2345', 'published_h', 'Life-Harness', 'tasks.json', 'split_tasks', 'os.environ')):
            raise ValueError('forbidden released/private reference')
    code = ('import importlib.util,sys; s=importlib.util.spec_from_file_location("candidate",sys.argv[1]);'
            'm=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);'
            'assert hasattr(m,"AgentHarness") and callable(getattr(m,"register",None))')
    subprocess.run([str(TAU / '.venv/bin/python'), '-c', code, str(path)], cwd=TAU,
                   env=child_env(), check=True, capture_output=True, timeout=120)


class MetaPort:
    def __init__(self, root, domain, spec, module, broker_url, ledger, user_base, solver_base):
        self.root, self.domain, self.spec, self.module = root, domain, spec, module
        self.run = root / 'meta' / domain
        self.work = self.run / 'workspace'
        self.broker_url, self.ledger, self.user_base, self.solver_base = broker_url, ledger, user_base, solver_base
        path = self.run / 'state.json'
        self.state = read(path) if path.exists() else {'status': 'running', 'incumbent': 'baseline_zero.py',
            'batch': 0, 'baseline_phase': None, 'rounds': [], 'released_h_flags': []}
        self.persist()

    def persist(self):
        save(self.run / 'state.json', self.state)

    def repair_date_rejections(self, candidates):
        """Measure exact paid candidates; only a changed acceptance invalidates later proposals."""
        original = read(self.run / 'state.json')
        first = min(candidates)
        if len(original['rounds']) != 6 or any(
                original['rounds'][i - 1]['candidate_score'] is not None or
                digest(self.work / 'agents' / (original['rounds'][i - 1]['candidate']['name'] + '.py')) != sha
                for i, sha in candidates.items()):
            raise RuntimeError('date repair requires audited, unmeasured original candidates')
        archive = self.run / 'date_import_repair'
        save(archive / 'state_before_repair.json', original)

        def preserve(path):
            if path.exists():
                target = archive / path.relative_to(self.run)
                target.parent.mkdir(parents=True, exist_ok=True)
                path.rename(target)

        for i in range(first, 7):
            for path in (self.work / 'logs' / f'round_{i:02d}',
                         self.work / 'jobs' / f'round_{i:02d}', self.run / f'preflight_{i:02d}.json'):
                preserve(path)
        start = original['rounds'][first - 1]
        self.state.update(rounds=original['rounds'][:first - 1], status='running',
            incumbent=start['parent'], batch=start['train_batch'],
            baseline_phase=f'batch_{start["train_batch"]:02d}_baseline')
        self.persist()
        for i in range(first, 7):
            if i not in candidates or not (self.run / f'pending_{i:02d}.json').is_file():
                raise RuntimeError('refuse to replay an unaudited paid proposal')
            row = self.round()  # Original run_evolve; cached pending makes no optimizer calls.
            row['date_import_repair'] = {'original_candidate_sha256': candidates[i],
                                         'optimizer_replayed': False}
            self.persist()
            if not row['accepted']:
                continue
            for later in original['rounds'][i:]:
                number = later['iteration']
                for name in (f'pending_{number:02d}.json', f'proposer_{number:02d}',
                             f'prompt_{number:02d}.md', f'preflight_{number:02d}.json'):
                    preserve(self.run / name)
                if later['candidate']:
                    preserve(self.work / 'agents' / (later['candidate']['name'] + '.py'))
                preserve(self.run / 'evals' / f'candidate_{number:02d}')
                preserve(self.work / 'evidence' / f'candidate_{number:02d}')
            for batch in range(row['train_batch'] + 1, 6):
                preserve(self.run / 'evals' / f'batch_{batch:02d}_baseline')
                preserve(self.work / 'evidence' / f'batch_{batch:02d}_baseline')
            save(archive / 'branch_change.json', {'accepted_round': i,
                'redo_original_optimizer_rounds': list(range(i + 1, 7)),
                'historical_optimizer_fees_retained': True})
            break
        save(self.work / 'history.json', self.state['rounds'])

    def phase(self, split, batch, name, candidate, round, smoke=False):
        from ahe_native.token_accounting import scoped_url
        directory = self.run / 'evals' / name
        summary_path = directory / 'summary.json'
        ids = list(self.spec['batches'][batch]['task_ids'] if split == 'train' else self.spec['test_task_ids'])
        if smoke:
            ids = ids[:1]
        trials = 1 if smoke or split == 'test' else self.spec['train_trials']
        config = {'domain': self.domain, 'split': split, 'task_ids': ids, 'trials': trials,
            'candidate_sha256': digest(candidate), 'solver_total_context': 32768, 'solver_output': 4096}
        if summary_path.is_file():
            score = read(summary_path)
            if score['config'] != config:
                raise RuntimeError('native phase fingerprint changed')
            return score
        directory.mkdir(parents=True, exist_ok=True)
        id_path = directory / 'task_ids.txt'
        id_path.write_text('\n'.join(ids) + '\n')
        scope = {'experiment': self.root.name + '_meta_' + self.domain, 'domain': self.domain, 'round': round}
        request = {'domain': self.domain, 'split': split, 'task_id_file': str(id_path),
            'trials': trials, 'concurrency': 50, 'candidate': str(candidate), 'solver_base': self.solver_base,
            'user_model': user_model(self.domain), 'user_base': scoped_url(self.user_base, role='user_simulator', **scope),
            'judge_base': scoped_url(self.user_base, role='judge', **scope), 'save_to': str(directory / 'native')}
        save(directory / 'request.json', request)
        with (directory / 'driver.log').open('a') as log:
            process = subprocess.run([str(TAU / '.venv/bin/python'), '-m', 'evolution.tau_comparison.meta_evaluate',
                '--request', str(directory / 'request.json')], cwd=ROOT,
                env=child_env(PYTHONPATH=str(ROOT)), stdout=log, stderr=subprocess.STDOUT, timeout=21600)
        results_path = directory / 'native/results.json'
        if process.returncode or not results_path.is_file():
            raise RuntimeError('native Tau evaluation failed; inspect ' + str(directory / 'driver.log'))
        simulations = read(results_path).get('simulations', [])
        score = summarize_rows(ids, trials, simulations)
        score.update(config=config, native_results_sha256=digest(results_path), split=split,
                     plugin_hash=digest(candidate), user_model=request['user_model'])
        save(summary_path, score)
        if split == 'train' and not smoke:
            dest = self.work / 'evidence' / name
            save(dest / 'summary.json', score)
            save(dest / 'public_trajectories.json', [public_simulation(s) for s in simulations])
        return score

    def contract(self, iteration, batch, parent):
        return f'''
PORT CONTRACT overrides Terminal-Bench/KIRA/Claude specifics.
Domain {self.domain}, round {iteration}/6, TRAIN batch {batch}; {len(self.spec['batches'][batch]['task_ids'])} public tasks,
{self.spec['train_trials']} trials per task. Solver local qwen3-4b, temperature 0, output 4096, total context 32768.
Optimizer deepseek-v4-flash. Start from identity; current incumbent agents/{parent}.
Never read/copy/import/reconstruct released Life H2-H5 or other experiments. Free-form Meta Python, no four-layer constraint.
Export AgentHarness and register(). register() may patch only plain native domain Tools/environment or LLMAgent public
messages/tool descriptions/action arguments; do not change real tools, budgets, user simulator or evaluator.
No new runtime model/network/subprocess/filesystem calls, hidden state/answers, task-ID tables, cross-episode mutable state.
Public interface contract is source_view/CONTRACT.md; standard library plus the documented plain tau2 modules only.
One focused mechanism, failures AND passing counterexamples. No TEST information exists here.
Workspace {self.work}: only agents/, source_view/, evidence/, logs/, jobs/, history.json, upstream_SKILL.md.
Use the original skill's ONE analysis Agent then ONE implementation Agent; host includes this contract in both.
Do not run benchmarks or independent APIs. Pure prototypes use python; no uv. Do not change existing frozen files.
Create one new agents/<lower_snake_case>.py and prescribed pending_eval.json with exactly one candidate:
name, import_path="agents.<name>:AgentHarness", hypothesis, changes, expected_efficiency.
'''

    def round(self):
        iteration = len(self.state['rounds']) + 1
        batch, parent = self.state['batch'], self.state['incumbent']
        if not self.state['baseline_phase']:
            name = f'batch_{batch:02d}_baseline'
            baseline = self.phase('train', batch, name, self.work / 'agents' / parent, iteration)
            self.state['baseline_phase'] = name
            self.persist()
        else:
            baseline = read(self.run / 'evals' / self.state['baseline_phase'] / 'summary.json')
        if baseline['infra_errors'] > 2:
            raise RuntimeError('baseline-wide infrastructure failure: stop before optimization')
        save(self.work / 'history.json', self.state['rounds'])
        u = self.module
        u.EVOLVE_DIR, u.AGENTS_DIR = self.work, self.work / 'agents'
        u.BASELINES = [('incumbent', f'agents.{Path(parent).stem}:AgentHarness')]
        u.BASELINE_AGENT_NAME, u.N_EVAL_TASKS, u.MODEL = 'incumbent', len(baseline['per_task']), 'qwen3-4b'
        round_name = f'round_{iteration:02d}'
        (self.work / 'logs' / round_name).mkdir(parents=True, exist_ok=True)
        save(self.work / 'jobs' / round_name / f'incumbent-t{self.spec["train_trials"]}/summary.json',
             {'per_task': baseline['per_task']})
        contract = self.contract(iteration, batch, parent)
        renderer = u.render_task_prompt
        u.render_task_prompt = lambda i, n: renderer(i, n) + '\n\n' + (self.work / 'upstream_SKILL.md').read_text() + contract
        cached = self.run / f'pending_{iteration:02d}.json'

        def propose(prompt, i, timeout=5400):
            if cached.is_file():
                shutil.copy2(cached, u.PENDING_EVAL)
                return True
            (self.run / f'prompt_{iteration:02d}.md').write_text(prompt)
            result = isolated_proposal(self.root, self.domain, iteration, self.work, prompt, contract,
                                       self.broker_url, u.PENDING_EVAL)
            if result:
                shutil.copy2(u.PENDING_EVAL, cached)
            return result

        def validate(name, import_path):
            if not re.fullmatch('[a-z][a-z0-9_]*', name) or import_path != f'agents.{name}:AgentHarness' or name == Path(parent).stem:
                return False
            try:
                path = self.work / 'agents' / (name + '.py')
                validate_candidate(path, self.domain)
                save(self.run / f'preflight_{iteration:02d}.json', {'passed': True, 'sha256': digest(path)})
                return True
            except Exception as exc:
                save(self.run / f'preflight_{iteration:02d}.json', {'passed': False, 'error_type': type(exc).__name__})
                return False

        def harbor(import_path, job_name, n_trials=1, n_concurrent=50):
            name = import_path.split(':')[0].split('.')[-1]
            score = self.phase('train', batch, f'candidate_{iteration:02d}', self.work / 'agents' / (name + '.py'), iteration)
            job = self.work / 'jobs' / round_name / (name + '-native')
            save(job / 'summary.json', {'per_task': score['per_task']})
            return job, True

        u.propose_claude, u.validate_candidate, u.harbor_run = propose, validate, harbor
        u.parse_job_results = lambda job, expected_trials=None: (
            read(job / 'summary.json')['per_task'] if (job / 'summary.json').is_file() else {})
        u.parse_trial_metrics = lambda job: {}
        try:
            u.run_evolve(argparse.Namespace(run_name=round_name, iterations=1, trials=self.spec['train_trials'],
                concurrent=50, skip_baseline=True, skip_smoke=True, full_eval=False, fresh=False, propose_timeout=5400))
        finally:
            u.render_task_prompt = renderer
        candidate_summary = self.run / 'evals' / f'candidate_{iteration:02d}/summary.json'
        score = read(candidate_summary) if candidate_summary.is_file() else None
        candidate = read(cached)['candidates'][0] if cached.is_file() else None
        usage = self.ledger.summarize(self.domain, iteration)
        if not usage['accounting_complete']:
            raise RuntimeError('optimizer usage unknown; stop before spending again')
        if score and set(score['per_task']) != set(baseline['per_task']):
            raise RuntimeError('candidate and incumbent tasks differ')
        accepted = score is not None and score['successes'] > baseline['successes']
        row = {'iteration': iteration, 'train_batch': batch, 'train_n': baseline['n'],
            'train_episodes': baseline['episodes'], 'parent': parent, 'baseline_successes': baseline['successes'],
            'baseline_score': baseline['accuracy'], 'candidate': candidate,
            'candidate_successes': score['successes'] if score else None,
            'candidate_score': score['accuracy'] if score else None, 'accepted': accepted,
            'candidate_infra_errors': score['infra_errors'] if score else None, 'usage': usage}
        if accepted:
            self.state.update(incumbent=candidate['name'] + '.py', baseline_phase=f'candidate_{iteration:02d}')
        elif iteration < 6:
            self.state.update(batch=min(batch + 1, 5), baseline_phase=None)
        row['retained_candidate'] = self.state['incumbent']
        self.state['rounds'].append(row)
        self.state['status'] = 'training_complete' if iteration == 6 else 'running'
        self.persist()
        return row
