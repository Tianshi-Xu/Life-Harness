"""Unmodified upstream Meta loop with native evaluation and recorded API transport."""
import argparse
import copy
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import ast
import io
import tokenize
import types
from pathlib import Path

from common import AB, DOMAINS, HERE, ROOT, ROUNDS, digest, manifest, progress, save, verify_frozen
from isolation import isolated_proposal


def candidate_restriction(source):
    """Keep the historical ban on code/data, without matching inert prose.

    The historical AST validator still enforces allowed imports, forbidden
    operations and dunder access. Executable string literals remain inspected;
    comments and standalone documentation strings cannot perform network I/O.
    """
    tree = ast.parse(source)
    prose = [(node.value.lineno, node.value.col_offset,
              node.value.end_lineno, node.value.end_col_offset)
             for node in ast.walk(tree)
             if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
             and isinstance(node.value.value, str)]
    tokens = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            continue
        if token.type == tokenize.STRING and any(
                (start_line, start_col) <= token.start and token.end <= (end_line, end_col)
                for start_line, start_col, end_line, end_col in prose):
            continue
        tokens.append(token)
    code = tokenize.untokenize(tokens)
    forbidden = ('h2345', 'published_h', 'Life-Harness', 'tasks.json', 'split_tasks',
                 'simulations', 'subprocess', 'requests', 'urllib', 'socket', 'os.environ', 'open(')
    return next((value for value in forbidden if value in code), None)


def upstream(domain):
    path = HERE / 'upstream_reference'
    sys.path.insert(0, str(path))
    try:
        import dotenv
    except ModuleNotFoundError:
        shim = types.ModuleType('dotenv')
        shim.load_dotenv = lambda *args, **kwargs: False
        sys.modules['dotenv'] = shim
    saved_env = dict(os.environ)
    spec = importlib.util.spec_from_file_location('original_meta_' + domain, path / 'meta_harness.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    os.environ.clear()
    os.environ.update(saved_env)
    assert module.run_evolve.__code__.co_filename == str(path / 'meta_harness.py')
    return module


def child_environment(domain):
    env = {k: v for k, v in os.environ.items()
           if not any(word in k.upper() for word in ('KEY', 'TOKEN', 'SECRET', 'PASSWORD', 'CREDENTIAL', 'AUTH'))}
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['TOKENIZERS_PARALLELISM'] = 'false'
    if domain == 'webshop':
        java = AB / '.native/sysroot/usr/lib/jvm/java-11-openjdk-amd64'
        env.update(JAVA_HOME=str(java), PATH=str(java / 'bin') + ':' + env.get('PATH', ''),
                   _JAVA_OPTIONS='-Xmx4g', PYTHONPATH=str(AB / '.native/webshop-src') + ':' + str(AB))
    return env


def phase(domain, split, batch, name, candidate, smoke_count=None):
    out = HERE / domain / 'evals' / name / 'summary.json'
    py = {'alfworld': AB / '.native/alfworld-venv/bin/python',
          'dbbench': AB / '.native/core/bin/python',
          'webshop': AB / '.native/webshop-venv/bin/python'}[domain]
    cmd = [str(py), '-u', str(HERE / 'evaluator.py'), domain, split, name, str(candidate), '--batch', str(batch)]
    if smoke_count is not None:
        cmd += ['--smoke-count', str(smoke_count)]
    progress(domain, phase=name, split=split, batch=batch)
    with (HERE / domain / (name + '.log')).open('a') as log:
        subprocess.run(cmd, env=child_environment(domain), cwd=ROOT, stdout=log,
                       stderr=subprocess.STDOUT, check=True, timeout=21600)
    summary = json.loads(out.read_text())
    expected = smoke_count if smoke_count is not None else 50 if split == 'train' else len(manifest()['domains'][domain]['test_indices'])
    if summary['n'] != expected or summary['plugin_hash'] != digest(candidate):
        raise RuntimeError('Incomplete evaluation or candidate hash mismatch')
    return summary


def public_export(domain, phase_name):
    src = HERE / domain / 'evals' / phase_name
    summary = json.loads((src / 'summary.json').read_text())
    dest = HERE / domain / 'workspace/evidence' / phase_name
    save(dest / 'summary.json', {k: v for k, v in summary.items() if k != 'ids'})
    for index in summary['per_task']:
        row = json.loads((src / f'episode_{index}.json').read_text())
        history = [{k: m[k] for k in ('role', 'content', 'tool_calls', 'tool_call_id', 'name') if k in m}
                   for m in row['history'] if isinstance(m, dict) and m.get('role') in ('system', 'user', 'assistant', 'tool')]
        calls = []
        for call in row['calls']:
            msg = call.get('response', {}).get('choices', [{}])[0].get('message', {})
            calls.append({'turn': call['turn'], 'messages': call['request']['messages'],
                          'tools': call['request']['tools'],
                          'response': {k: msg[k] for k in ('role', 'content', 'tool_calls') if k in msg}})
        save(dest / f'episode_{index}.json', {'task_id': index, 'success': row['success'],
            'status': row['status'], 'turns': row['turns'], 'tokens': row['tokens'],
            'history': history, 'model_calls': calls, 'candidate_hash': summary['plugin_hash']})
    return summary


def port_contract(domain, iteration, batch, workspace, incumbent):
    return f'''
PORT CONTRACT: overrides every Terminal-Bench/KIRA/Claude-specific statement above.
Actual task: {domain}; round {iteration}/6; training batch {batch}; exactly 50 public TRAIN questions.
Solver: local qwen3-4b, temperature 0, output max 4096, total context max 32768. Optimizer: deepseek-v4-flash.
Start from identity agents/baseline_zero.py. Current retained implementation: agents/{incumbent}.
Never read, copy, import, or reconstruct released Life-Harness H2/H3/H4/H5. Search space is free-form Meta-Harness Python, no four-layer constraint.
Runtime: AgentHarness(public_context=None), synchronous transform_request(request_dict) and transform_response(assistant_dict, public_history).
Both return ordinary OpenAI-compatible dictionaries. One instance is created per episode. Optional register() is not used by this AgentBench port.
The host freezes model/sampling/budgets/environment/scoring/executed benchmark tools. Runtime may transform public requests and replies.
No new runtime LLM, model/network/subprocess/filesystem calls, labels, expected answers, evaluator objects, hidden state, task-ID policies,
instance-specific answer tables, or cross-episode mutable state. Imports: re/json/copy/math/collections/difflib/string/typing/ast/dataclasses/functools/itertools/statistics.
One focused mechanism. Analyze failures AND passing counterexamples. No test results exist in this workspace.
Workspace boundary: {workspace}. Read only agents/, source_view/, evidence/, logs/, jobs/, history.json, upstream_SKILL.md.
All 50 current TRAIN trajectories are under evidence/. Previous attempts and code are preserved here; history.json contains all previous TRAIN decisions.
Use the original skill's ONE analysis Agent and ONE implementation Agent. The host automatically includes this full contract in both delegated prompts.
Do not run benchmarks, access other experiments, or make independent API/network calls. Pure local import/prototype checks use python (not uv).
Create one NEW agents/<lower_snake_case>.py and the prescribed pending_eval.json, with exactly one candidate:
name, import_path="agents.<name>:AgentHarness", hypothesis, changes, expected_efficiency.
Do not modify existing agents or frozen evidence/log/history/skill/source inputs.
'''


def state_for(domain):
    path = HERE / domain / 'state.json'
    if path.exists():
        return json.loads(path.read_text())
    state = {'status': 'running', 'incumbent': 'baseline_zero.py', 'batch': 0,
             'baseline_phase': None, 'from_scratch': True, 'released_h_flags': [], 'rounds': []}
    save(path, state)
    return state


def native_round(domain, iteration, batch, incumbent, baseline, module, broker_url):
    run, work = HERE / domain, HERE / domain / 'workspace'
    u = module
    u.EVOLVE_DIR, u.AGENTS_DIR = work, work / 'agents'
    u.BASELINES = [('incumbent', f'agents.{Path(incumbent).stem}:AgentHarness')]
    u.BASELINE_AGENT_NAME, u.N_EVAL_TASKS, u.MODEL = 'incumbent', 50, 'qwen3-4b'
    round_name = f'round_{iteration:02d}'
    logdir = work / 'logs' / round_name
    logdir.mkdir(parents=True, exist_ok=True)
    seed = work / 'jobs' / round_name / 'incumbent-t1'
    save(seed / 'summary.json', {'per_task': baseline['per_task']})
    summary_path = logdir / 'evolution_summary.jsonl'
    if not summary_path.exists():
        rows = []
        for previous in range(1, iteration):
            p = work / 'logs' / f'round_{previous:02d}' / 'evolution_summary.jsonl'
            if p.exists():
                lines = [json.loads(line) for line in p.read_text().splitlines() if line]
                rows += [r for r in lines if r.get('iteration') == previous]
        summary_path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    contract = port_contract(domain, iteration, batch, work, incumbent)
    renderer = u.render_task_prompt
    # Restore this after the round; never wrap an already wrapped renderer.
    u.render_task_prompt = lambda i, n: renderer(i, n) + '\n\n' + (work / 'upstream_SKILL.md').read_text() + contract
    cached = run / f'pending_{iteration:02d}.json'

    def propose(prompt, i, timeout=5400):
        if cached.exists():
            shutil.copy2(cached, u.PENDING_EVAL)
            return True
        (run / f'prompt_{iteration:02d}.md').write_text(prompt)
        ok = isolated_proposal(domain, iteration, work, prompt, contract, broker_url, u.PENDING_EVAL)
        if ok:
            shutil.copy2(u.PENDING_EVAL, cached)
        return ok

    def validate(name, import_path):
        if import_path != f'agents.{name}:AgentHarness' or name == Path(incumbent).stem:
            raise ValueError('Candidate import contract violation')
        candidate = work / 'agents' / (name + '.py')
        text = candidate.read_text()
        hit = candidate_restriction(text)
        if hit:
            save(run / f'preflight_{iteration:02d}.json', {'pass': False,
                'candidate_sha256': digest(candidate), 'error': 'Historical port restriction: ' + hit})
            return False
        cmd = [str(AB / '.native/core/bin/python'), '-c',
               'import sys;sys.path.insert(0,sys.argv[1]);import evaluator;'
               'evaluator.core.validate_plugin(sys.argv[2]);print("OK")', str(HERE), str(candidate)]
        result = subprocess.run(cmd, env=child_environment(domain), cwd=ROOT, capture_output=True, text=True, timeout=180)
        save(run / f'preflight_{iteration:02d}.json', {'pass': result.returncode == 0,
            'candidate_sha256': digest(candidate), 'parent_sha256': digest(work / 'agents' / incumbent),
            'error': result.stderr[-8000:] if result.returncode else None})
        return result.returncode == 0

    def harbor(import_path, job_name, n_trials=1, n_concurrent=50):
        name = import_path.split(':')[0].split('.')[-1]
        score = phase(domain, 'train', batch, f'candidate_{iteration:02d}', work / 'agents' / (name + '.py'))
        public_export(domain, f'candidate_{iteration:02d}')
        job = work / 'jobs' / round_name / (name + '-native')
        save(job / 'summary.json', {'per_task': score['per_task']})
        return job, True

    def parse(job, expected_trials=None):
        p = job / 'summary.json'
        if not p.exists():
            return {}
        return {k: v if isinstance(v, list) else [v] for k, v in json.loads(p.read_text())['per_task'].items()}

    u.propose_claude, u.validate_candidate = propose, validate
    u.harbor_run, u.parse_job_results, u.parse_trial_metrics = harbor, parse, lambda job: {}
    candidate_summary = run / 'evals' / f'candidate_{iteration:02d}' / 'summary.json'
    try:
        if not (cached.exists() and candidate_summary.exists()):
            u.run_evolve(argparse.Namespace(run_name=round_name, iterations=1, trials=1, concurrent=50,
                skip_baseline=True, skip_smoke=True, full_eval=False, fresh=False, propose_timeout=5400))
    finally:
        u.render_task_prompt = renderer
    candidate = json.loads(cached.read_text())['candidates'][0] if cached.exists() else None
    score = json.loads(candidate_summary.read_text()) if candidate_summary.exists() else None
    return candidate, score


def run_round(domain, module, broker_url, ledger):
    verify_frozen()
    if not ledger.summarize()['accounting_complete']:
        raise RuntimeError('Unknown prior API usage: stop before spending more tokens')
    work = HERE / domain / 'workspace'
    state = state_for(domain)
    iteration = len(state['rounds']) + 1
    if iteration > ROUNDS:
        return state
    batch, parent = state['batch'], state['incumbent']
    if not state['baseline_phase']:
        base_name = f'batch_{batch:02d}_baseline'
        baseline = phase(domain, 'train', batch, base_name, work / 'agents' / parent)
        public_export(domain, base_name)
        state['baseline_phase'] = base_name
        save(HERE / domain / 'state.json', state)
    else:
        baseline = json.loads((HERE / domain / 'evals' / state['baseline_phase'] / 'summary.json').read_text())
    if baseline['infra_errors'] > manifest()['max_batch_infra_errors_before_pause']:
        raise RuntimeError('Batch-wide baseline infrastructure failures; stop before paid optimization')
    save(work / 'history.json', state['rounds'])
    progress(domain, phase='propose', round=iteration, train_batch=batch)
    candidate, score = native_round(domain, iteration, batch, parent, baseline, module, broker_url)
    usage = ledger.summarize(domain, iteration)
    if not usage['accounting_complete']:
        raise RuntimeError('Optimizer accounting incomplete; stop before another round')
    if score is not None and set(baseline['per_task']) != set(score['per_task']):
        raise RuntimeError('Candidate and incumbent task IDs differ')
    accepted = score is not None and score['successes'] > baseline['successes']
    row = {'iteration': iteration, 'train_batch': batch, 'train_n': 50, 'parent': parent,
           'baseline_successes': baseline['successes'], 'baseline_score': baseline['accuracy'],
           'candidate': candidate, 'candidate_successes': score['successes'] if score else None,
           'candidate_score': score['accuracy'] if score else None, 'accepted': accepted,
           'baseline_infra_errors': baseline['infra_errors'], 'candidate_infra_errors': score['infra_errors'] if score else None,
           'usage': usage, 'solver_baseline_tokens': baseline['tokens'],
           'solver_candidate_tokens': score['tokens'] if score else None,
           'wins': sum(score['per_task'][k] > v for k, v in baseline['per_task'].items()) if score else None,
           'losses': sum(score['per_task'][k] < v for k, v in baseline['per_task'].items()) if score else None,
           'outcome': 'accepted' if accepted else 'rejected' if score else 'candidate_unmeasured'}
    if accepted:
        state.update(incumbent=candidate['name'] + '.py', baseline_phase=f'candidate_{iteration:02d}')
    elif iteration < ROUNDS:
        state.update(batch=min(batch + 1, 5), baseline_phase=None)
    row['retained_candidate'] = state['incumbent']
    row['retained_candidate_sha256'] = digest(work / 'agents' / state['incumbent'])
    state['rounds'].append(row)
    if iteration == ROUNDS:
        state['status'] = 'training_complete'
    save(HERE / domain / 'state.json', state)
    progress(domain, phase='round_complete', round=iteration, outcome=row['outcome'], tokens=usage['total_tokens'])
    print(json.dumps({'domain': domain, **row}, ensure_ascii=False), flush=True)
    return state


def seal_training(ledger):
    cfg = verify_frozen()
    states = {d: state_for(d) for d in DOMAINS}
    if any(s['status'] != 'training_complete' or len(s['rounds']) != 6 for s in states.values()):
        raise RuntimeError('Cannot seal incomplete training')
    if not ledger.summarize()['accounting_complete']:
        raise RuntimeError('Cannot seal incomplete token accounting')
    save(HERE / 'training_seal.json', {'states': {d: digest(HERE / d / 'state.json') for d in DOMAINS},
        'retained_candidates': {d: digest(HERE / d / 'workspace/agents' / states[d]['incumbent']) for d in DOMAINS},
        'manifest_sha256': digest(HERE / 'manifest.json'), 'budget_label': cfg['comparison_label'],
        'usage': ledger.summarize()})


def render_report(ledger):
    ledger.report(DOMAINS)
    lines = ['# Meta: six-round API rerun', '',
             'Measured optimizer tokens; no preset cumulative token ceiling. NOT an equal-token-budget comparison.', '',
             '| Domain | Round | Batch | Incumbent /50 | Candidate /50 | Decision | Flash calls | Input tokens | Output tokens | Total tokens | Cumulative tokens |',
             '|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|']
    for d in DOMAINS:
        path = HERE / d / 'state.json'
        if not path.exists():
            continue
        total = 0
        for row in json.loads(path.read_text())['rounds']:
            usage = row['usage']
            total += usage['total_tokens']
            candidate = row['candidate_successes'] if row['candidate_successes'] is not None else 'unmeasured'
            lines.append(f'| {d} | {row["iteration"]} | {row["train_batch"]} | {row["baseline_successes"]} | {candidate} | '
                         f'{row["outcome"]} | {usage["successful_calls"]} | {usage["input_tokens"]} | '
                         f'{usage["output_tokens"]} | {usage["total_tokens"]} | {total} |')
    (HERE / 'TRAINING.md').write_text('\n'.join(lines) + '\n')
