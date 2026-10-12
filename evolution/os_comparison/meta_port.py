"""OS boundary around the unchanged official Meta optimization loop."""
import json
from pathlib import Path
import shutil
import subprocess

from evolution.tau_comparison.common import HISTORICAL, ROOT, child_env, public_message, read, save
from evolution.tau_comparison.meta_port import MetaPort as OriginalPort
from .evaluate import validate_meta
from .protocol import digest


def prepare_workspace(root, schema):
    work = root / 'meta/os/workspace'
    (work / 'agents').mkdir(parents=True)
    (work / 'source_view').mkdir()
    shutil.copy2(HISTORICAL / 'alfworld/workspace/agents/baseline_zero.py', work / 'agents/baseline_zero.py')
    shutil.copy2(HISTORICAL / 'upstream_reference/.claude/skills/meta-harness-terminal-bench-2/SKILL.md',
                 work / 'upstream_SKILL.md')
    shutil.copy2(schema, work / 'source_view/native_tools.json')
    (work / 'source_view/CONTRACT.md').write_text(
        'AgentHarness(public_context) receives only {description: str}.\n'
        'transform_request(request) returns an OpenAI chat request; transform_response(response, public_history) '
        'returns an assistant message. These methods may edit public messages, tool descriptions, actions, '
        'or episode-local state. register() is a no-op. Pure standard library only.\n'
        'OS execution uses native bash_action, finish_action, answer_action. Tools, images, budgets and '
        'grading remain fixed. No filesystem, subprocess, network, model calls, private state/answers, '
        'task-ID lookup or persistent cross-episode state.\n')
    return work


class MetaPort(OriginalPort):
    def round(self):
        from evolution.tau_comparison import meta_port as boundary
        original = boundary.validate_candidate
        boundary.validate_candidate = validate_candidate
        try:
            return super().round()
        finally:
            boundary.validate_candidate = original

    def phase(self, split, batch, name, candidate, round, smoke=False):
        ids = (self.spec['batches'][batch]['task_ids'] if split == 'train' else self.spec['test_task_ids'])
        if smoke:
            ids = ids[:1]
        target = self.run / 'evals' / name / 'summary.json'
        if target.exists():
            score = read(target)
            if score['candidate_sha256'] != digest(candidate) or score['ids'] != ids:
                raise RuntimeError('cached native measurement identity changed')
            return score
        native = self.root / 'evolution/evals' / name
        if (native / 'summary.json').exists():
            config = read(native / 'config.json')
            if config['candidate_sha256'] != digest(candidate) or config['ids'] != ids or config['split'] != split:
                raise RuntimeError('audited native cache differs from candidate or task batch')
        else:
            log = self.run / 'logs' / (name + '.log')
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open('w') as output:
                subprocess.run([str(ROOT / 'AgentBench/.native/core/bin/python'), '-u', '-m',
                    'evolution.os_comparison.evaluate', '--root', str(self.root), '--method', 'meta',
                    '--split', split, '--phase', name, '--ids', json.dumps(ids), '--plugin', str(candidate),
                    '--solver-base', self.solver_base], cwd=ROOT, env=child_env(PYTHONPATH=str(ROOT)),
                    stdout=output, stderr=subprocess.STDOUT, timeout=21600, check=True)
        score = read(native / 'summary.json')
        score['episodes'] = score['n']
        score['per_task'] = {key: [value] for key, value in score['per_task'].items()}
        save(target, score)
        if split == 'train' and not smoke:
            public = []
            for index in ids:
                row = read(native / f'episode_{index}.json')
                public.append({'task_id': str(index), 'success': row['success'],
                    'public_task_context': row['public_task_context'],
                    'messages': [m for value in row['history'] if (m := public_message(value)) is not None],
                    'status': row['status']})
            save(self.work / 'evidence' / name / 'public_trajectories.json', public)
            save(self.work / 'evidence' / name / 'summary.json', score)
        return score

    def contract(self, iteration, batch, parent):
        return f'''
PORT CONTRACT overrides Terminal-Bench/KIRA/Claude details.
Native OS, round {iteration}/6, TRAIN batch {batch}, 50 tasks, one trial each.
Solver qwen3-4b, thinking disabled, temperature 0, output 4096, total context 32768, 16 turns.
Current incumbent agents/{parent}. Start from identity. Free-form Meta; no four-layer constraint.
Export AgentHarness(public_context), transform_request(request), transform_response(response, public_history),
and no-op register(). Exact public interface: source_view/CONTRACT.md and native_tools.json.
Use pure standard library only; do not import task modules or alter environment/tools/grader/budgets.
No runtime filesystem, subprocess, network/model calls, hidden state/answers, task-ID tables or cross-episode state.
Never read/copy/import/reconstruct released Life strategies, H2-H5 implementations or other experiments.
One focused mechanism with failures and passing counterexamples. Only TRAIN evidence exists here.
Workspace {self.work}: agents/, source_view/, evidence/, logs/, jobs/, history.json, upstream_SKILL.md.
Use the original skill's ONE analysis Agent then ONE implementation Agent. No benchmark/API calls.
Do not change frozen inputs. Write one new agents/<lower_snake_case>.py and pending_eval.json with
exactly one candidate: name, import_path="agents.<name>:AgentHarness", hypothesis, changes, expected_efficiency.
'''


def validate_candidate(path, domain):
    if domain != 'os':
        raise ValueError('OS adapter only')
    validate_meta(path)
