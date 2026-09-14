#!/usr/bin/env python3
"""Current four-hook AgentBench iteration with bounded evidence and staged scoring.

Historical experiment directories are immutable inputs. This entry creates a new
run, freezes its protocol, and uses real evaluations for every candidate score.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import evidence
import proposer
from agentbench_preflight import preflight

META = Path(__file__).resolve().parent
AB = META.parent / 'AgentBench'
save, digest = evidence.save, evidence.digest


def freeze_files(root):
    return {str(p.relative_to(root)): digest(p) for p in root.rglob('*')
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}


def prepare(args):
    source, root = args.source.resolve(), args.run_dir.resolve()
    if root == source or source in root.parents or root in source.parents:
        raise ValueError('Use a separate new run directory')
    if (root / 'protocol.json').exists():
        protocol = json.loads((root / 'protocol.json').read_text())
        requested = settings(args)
        if protocol['settings'] != requested:
            raise ValueError('Resume settings changed; create a new run')
        for rel, expected in protocol['frozen'].items():
            if digest(root / rel) != expected:
                raise ValueError('Frozen input changed: ' + rel)
        for rel, expected in protocol['implementation'].items():
            if digest(META / rel) != expected:
                raise ValueError('Implementation changed; use a new run')
        return protocol
    if root.exists() and any(root.iterdir()):
        raise ValueError('New run directory must be empty')
    manifest = json.loads((source / 'manifest.json').read_text())
    ids = manifest[args.domain]['indices']
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Invalid fixed training pool')
    heldout = json.loads(args.heldout_indices.read_text()) if args.heldout_indices else []
    if args.heldout_indices and (not heldout or len(heldout) != len(set(heldout)) or set(heldout) & set(ids)):
        raise ValueError('Heldout indices must be nonempty, unique and disjoint from training')
    if any(type(i) is not int or i < 0 for i in ids + heldout):
        raise ValueError('Task indices must be nonnegative integers')
    baseline = args.baseline.resolve()
    root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source / 'task_snapshot', root / 'task_snapshot', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(source / 'legacy_disabled.py', root / 'legacy_disabled.py')
    run = root / args.domain
    run.mkdir()
    for name in ('protocol.py', 'DESIGN_GUIDE.md'):
        shutil.copy2(source / args.domain / name, run / name)
    shutil.copy2(baseline, run / 'baseline.py')
    manifest.update(name=root.name, method='bounded public evidence + staged full-confirmed current H2345',
                    iterations=args.rounds, sample_size=len(ids), acceptance='strict full-pool success-count increase')
    manifest[args.domain]['heldout_indices'] = heldout
    save(root / 'manifest.json', manifest)
    protocol = {'settings': settings(args), 'source_manifest_hash': digest(source / 'manifest.json'),
                'baseline_hash': digest(baseline), 'frozen': freeze_files(root),
                'implementation': {name: digest(META / name) for name in
                                   ('agentbench_loop.py', 'agentbench_evaluate.py', 'agentbench_preflight.py', 'evidence.py', 'proposer.py')}}
    save(root / 'protocol.json', protocol)
    return protocol


def settings(args):
    return {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()
            if k not in {'prepare_only', 'finalize'}}


def load_evaluator(root):
    os.environ['LIFE_AGENTBENCH_RUN'] = str(root)
    spec = importlib.util.spec_from_file_location('agentbench_evaluate', META / 'agentbench_evaluate.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def phase(root, domain, name, candidate, ids=None, heldout=False):
    python = AB / ('.native/core/bin/python' if domain == 'dbbench' else '.native/alfworld-venv/bin/python')
    cmd = [str(python), '-u', str(META / 'agentbench_evaluate.py'), domain, name, str(candidate)]
    if heldout:
        cmd.append('--heldout')
    if ids is not None:
        path = root / domain / f'{name}_screen_ids.json'
        if path.exists() and json.loads(path.read_text()) != ids:
            raise ValueError('Screen selection changed on resume')
        save(path, ids)
        cmd += ['--indices-file', str(path)]
    env = {**os.environ, 'LIFE_AGENTBENCH_RUN': str(root)}
    with (root / domain / f'{name}.log').open('a') as log:
        subprocess.run(cmd, cwd=AB, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=10800)
    result = json.loads((root / domain / 'evals' / name / ('screen_summary.json' if ids is not None else 'summary.json')).read_text())
    pool = json.loads((root / 'manifest.json').read_text())[domain]['heldout_indices' if heldout else 'indices']
    expected = {str(i) for i in (ids if ids is not None else pool)}
    if result['infra_errors'] or set(result['per_task']) != expected or result['n'] != len(expected):
        raise ValueError('Incomplete or invalid evaluation')
    return result


def invoke(root, workspace, prompt, tag, model):
    # Every input except the three output contracts is read-only and hash-checked.
    outputs = {'candidate.py', 'proposal.json', 'retrieval_request.json'}
    frozen = {rel: h for rel, h in freeze_files(workspace).items() if rel not in outputs}
    save(root / f'{tag}_inputs.json', frozen)
    (root / f'{tag}_prompt.md').write_text(prompt)
    result = proposer.run(prompt=prompt, cwd=workspace, model=model, log_path=root / f'{tag}_proposer.json')
    if any(not (workspace / rel).exists() or digest(workspace / rel) != h for rel, h in frozen.items()):
        raise ValueError('Proposer changed read-only input')
    if result.exit_code:
        raise RuntimeError('Proposer failed; logs preserved for resume')


def proposal_prompt(domain, stage, budget):
    return f'''Improve the CURRENT unified Harness.h2/h3/h4/h5 for {domain} with ONE focused mechanism.
Read DESIGN_GUIDE.md and protocol.py, current.py, then evidence/index.json and any evidence/details_*.json.
Only Read/Write/Edit/Glob/Grep inside this workspace. No shell, execution, delegation, parent paths, other runs or heldout data.
The host keeps full episodes outside your workspace. Never seek them directly. Evidence includes public messages,
public hook interventions and success bits only; observed signals do NOT establish root causes.
No private labels, correct SQL, hidden state, task-specific names/IDs/answers, I/O or extra model calls in the candidate.
Preserve all four layers and use only their public inputs. H3 may edit tool descriptions only;
H5 once and each H4 call returns [] or one <=120-word string; H2 mechanical repairs preserve tool IDs.
DBBench may define bind_task_context(context) for the host-provided public description, task type, table schema,
evidence and additional table description. It must reject or ignore oracle SQL, labels and expected answers.
Optional h3_message(message), h4_persistent, h4_message_prefix and h5_message_prefix are host-owned transport adapters;
preserve them when they are present in current.py so candidate behavior matches the released Task.
Preserve h2_preserve_raw_history when present: H2 may repair the action sent to the environment while retaining
the model's original assistant message in public history, as the released runtimes did.
The incumbent may also use the host-owned _harness_control return metadata to suppress a blocked tool while
preserving its public assistant/tool history. Preserve that adapter metadata; do not invent new control actions.
Optional h2_no_tool_message(message, remaining) preserves the released no-tool recovery timing and is also
host-owned transport behavior; preserve it when present.
Each tool action uses a normal turn. Keep model, environment, reward, history and turn budgets unchanged.
Fresh Harness per episode; imports limited to re/json/copy/math/collections/difflib/string/typing/ast/dataclasses.
No open/eval/exec/compile/getattr/setattr/globals/locals/vars/dunder attributes. Private helpers allowed.

{stage}
At most {budget} distinct serialized evidence characters will be exposed, independent of task pool size.
This is an exposure limit, not a claim that repeated tool reads or context tokens are hard-capped.
To request detail, write retrieval_request.json as a list of at most 12 objects:
{{"task_id":"ID","start_turn":1,"end_turn":3}} or {{"group":"signal_name","start_turn":1,"end_turn":3}}.
Omit turn bounds for a whole episode. Tool MESSAGE positions are not model-turn numbers.
A group request chooses a representative failure, preferring tasks not yet delivered. Narrow the range after a budget denial.
For the final proposal edit candidate.py and write proposal.json with:
{{"name":"snake_case","target_layer":"h2|h3|h4|h5","changed_symbols":[],
"hypothesis":"one mechanism and public evidence","evidence_tasks":[],"evidence_groups":[],
"counterexamples":"valid cases that must survive","risks":"including unexamined groups"}}.
Host runs contract replay, a fixed representative screen, then full-pool confirmation if screen net >=0
or this is a periodic exploration round. Only strictly greater FULL-pool success accepts. Screen-only results
cannot become an incumbent. Ties reject after full scoring. No performance improvement is guaranteed.
'''


def propose(args, root, iteration, incumbent, eval_dir):
    run = root / args.domain
    workspace = run / f'proposer_{iteration:02d}'
    host = run / f'evidence_{iteration:02d}'
    ready = workspace / 'ready.json'
    if not ready.exists():
        if workspace.exists():
            shutil.rmtree(workspace)
        if host.exists():
            shutil.rmtree(host)
        workspace.mkdir()
        for name in ('protocol.py', 'DESIGN_GUIDE.md'):
            shutil.copy2(run / name, workspace / name)
        shutil.copy2(incumbent, workspace / 'current.py')
        shutil.copy2(incumbent, workspace / 'candidate.py')
        evidence.build_pack(eval_dir, host, workspace / 'evidence', seed=args.seed + iteration,
                            max_cases=args.cases, initial_chars=args.initial_chars, total_chars=args.evidence_chars)
        history = json.loads((run / 'state.json').read_text())['rounds']
        brief = [{'iteration': r['iteration'], 'name': str(r['proposal']['name'])[:120],
                  'accepted': r['accepted'], 'full_successes': r['full_successes'], 'screen': r['screen']}
                 for r in history[-5:]]
        save(workspace / 'round_history.json', brief)
        save(ready, {'incumbent_hash': digest(incumbent)})
    if json.loads(ready.read_text())['incumbent_hash'] != digest(incumbent):
        raise ValueError('Incumbent changed on resume')
    for batch in range(args.retrieval_batches):
        marker = host / f'batch_{batch}.json'
        if marker.exists():
            continue
        request = workspace / 'retrieval_request.json'
        # Preserve completed proposer output across crashes, without rerunning the agent.
        done = host / f'proposer_batch_{batch}.json'
        if not done.exists():
            request.unlink(missing_ok=True)
            invoke(run, workspace, proposal_prompt(args.domain,
                   'Analysis/retrieval stage: request evidence now; final implementation follows after retrieval.', args.evidence_chars),
                   f'round_{iteration:02d}_read_{batch}', args.proposer_model)
            save(done, {'complete': True})
        requests = json.loads(request.read_text()) if request.exists() else []
        # A host-side journal commits retrieval and its exposure ledger for crash recovery.
        ledger_before = json.loads((host / 'ledger.json').read_text())
        existing = next((x for x in ledger_before['requests'] if x['batch'] == batch), None)
        if not existing:
            evidence.retrieve(host, workspace / 'evidence', requests, batch)
        save(marker, {'requests': requests})
    final_marker = host / 'proposal_complete.json'
    if not final_marker.exists():
        invoke(run, workspace, proposal_prompt(args.domain,
               'FINAL stage: no more retrieval. Implement one supported hypothesis and write proposal.json.', args.evidence_chars),
               f'round_{iteration:02d}_final', args.proposer_model)
        metadata = json.loads((workspace / 'proposal.json').read_text())
        if metadata['target_layer'] not in {'h2', 'h3', 'h4', 'h5'} or any(k not in metadata for k in
            ('hypothesis', 'evidence_tasks', 'evidence_groups', 'counterexamples', 'risks', 'changed_symbols', 'name')):
            raise ValueError('Incomplete proposal evidence contract')
        save(final_marker, {'candidate_hash': digest(workspace / 'candidate.py'), 'proposal_hash': digest(workspace / 'proposal.json')})
    final = json.loads(final_marker.read_text())
    if digest(workspace / 'candidate.py') != final['candidate_hash'] or digest(workspace / 'proposal.json') != final['proposal_hash']:
        raise ValueError('Completed proposal changed')
    return workspace / 'candidate.py', json.loads((workspace / 'proposal.json').read_text())


def run(args):
    protocol = prepare(args)
    root = args.run_dir.resolve()
    if args.prepare_only:
        print(json.dumps({'prepared': str(root), 'live_episodes': 0}))
        return
    evaluator = load_evaluator(root)
    domain = args.domain
    folder = root / domain
    state_path = folder / 'state.json'
    if args.finalize and not state_path.exists():
        raise ValueError('Cannot finalize before completing search')
    if state_path.exists():
        state = json.loads(state_path.read_text())
    else:
        base = phase(root, domain, 'baseline', folder / 'baseline.py')
        state = {'status': 'running', 'accepted': 'baseline.py', 'accepted_phase': 'baseline',
                 'accepted_hash': digest(folder / 'baseline.py'), 'score': base['successes'], 'rounds': []}
        save(state_path, state)
    if args.finalize:
        if state['status'] not in {'complete', 'frozen_for_test', 'tested'}:
            raise ValueError('Complete the fixed search before heldout testing')
        if not evaluator.MANIFEST[domain].get('heldout_indices'):
            raise ValueError('Register disjoint heldout indices when preparing the run')
        if digest(folder / state['accepted']) != state['accepted_hash']:
            raise ValueError('Accepted candidate changed')
        state['status'] = 'frozen_for_test'
        save(state_path, state)
        baseline = phase(root, domain, 'heldout_baseline', folder / 'baseline.py', heldout=True)
        final = phase(root, domain, 'heldout_final', folder / state['accepted'], heldout=True)
        save(folder / 'heldout_comparison.json', {'baseline': baseline, 'final': final,
             'wins': sum(final['per_task'][k] > v for k, v in baseline['per_task'].items()),
             'losses': sum(final['per_task'][k] < v for k, v in baseline['per_task'].items())})
        state['status'] = 'tested'
        save(state_path, state)
        return
    if state['status'] in {'frozen_for_test', 'tested'}:
        raise ValueError('Run frozen for heldout testing; further proposals disabled')
    for iteration in range(len(state['rounds']) + 1, args.rounds + 1):
        prepare(args)  # Recheck frozen inputs before every proposal/evaluation cycle.
        incumbent = folder / state['accepted']
        if digest(incumbent) != state['accepted_hash']:
            raise ValueError('Accepted candidate changed')
        eval_dir = folder / 'evals' / state['accepted_phase']
        candidate, metadata = propose(args, root, iteration, incumbent, eval_dir)
        checks = preflight(candidate, domain, root, evaluator)
        save(folder / f'preflight_{iteration:02d}.json', checks)
        dest = folder / f'candidate_{iteration:02d}.py'
        if dest.exists() and digest(dest) != digest(candidate):
            raise ValueError('Scored candidate changed')
        shutil.copy2(candidate, dest)
        catalog = json.loads((folder / f'evidence_{iteration:02d}' / 'catalog.json').read_text())
        episodes = {k: json.loads((folder / f'evidence_{iteration:02d}' / 'episodes' / f'{k}.json').read_text()) for k in catalog}
        ids = [int(k) for k in evidence.screen_ids(episodes, args.screen_size, args.seed + iteration)]
        name = f'iter_{iteration:02d}'
        screen = phase(root, domain, name, dest, ids)
        incumbent_score = json.loads((eval_dir / 'summary.json').read_text())['per_task']
        decision = evidence.screen_decision(screen['per_task'], {k: incumbent_score[k] for k in screen['per_task']}, iteration, args.explore_every)
        row = {'iteration': iteration, 'proposal': metadata, 'screen': decision, 'screen_n': len(ids),
               'screen_tokens': screen['tokens'], 'accepted': False, 'full_successes': None,
               'evidence': json.loads((folder / f'evidence_{iteration:02d}' / 'ledger.json').read_text())}
        if decision['run_full']:
            result = phase(root, domain, name, dest)  # Reuses only THIS candidate's identical-config screen episodes.
            row.update(full_successes=result['successes'], tokens=result['tokens'],
                       wins=sum(result['per_task'][k] > v for k, v in incumbent_score.items()),
                       losses=sum(result['per_task'][k] < v for k, v in incumbent_score.items()))
            row['accepted'] = result['successes'] > state['score']
            if row['accepted']:
                state.update(accepted=dest.name, accepted_phase=name, accepted_hash=digest(dest), score=result['successes'])
        state['rounds'].append(row)
        save(state_path, state)
        print(json.dumps({'iteration': iteration, 'full_successes': row['full_successes'], 'accepted': row['accepted']}), flush=True)
    state['status'] = 'complete'
    save(state_path, state)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True, help='Frozen train-only AgentBench experiment adapter and manifest')
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--domain', choices=['alfworld', 'dbbench'], required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--screen-size', type=int, default=16)
    p.add_argument('--explore-every', type=int, default=3)
    p.add_argument('--cases', type=int, default=12)
    p.add_argument('--initial-chars', type=int, default=24000)
    p.add_argument('--evidence-chars', type=int, default=80000)
    p.add_argument('--retrieval-batches', type=int, default=2)
    p.add_argument('--seed', type=int, default=20260913)
    p.add_argument('--proposer-model', default='DeepSeek-Flash')
    p.add_argument('--heldout-indices', type=Path, help='Predeclared unused holdout IDs in the SAME source dataset; never proposer evidence')
    p.add_argument('--finalize', action='store_true', help='Freeze completed run and compare baseline/final on registered heldout IDs')
    p.add_argument('--prepare-only', action='store_true')
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    if min(args.rounds, args.screen_size, args.cases, args.explore_every) < 1 or args.retrieval_batches < 0:
        raise SystemExit('Rounds, screen/case sizes and exploration interval must be positive')
    if not 2000 <= args.initial_chars <= args.evidence_chars:
        raise SystemExit('Require 2000 <= initial chars <= total evidence chars')
    run(args)
