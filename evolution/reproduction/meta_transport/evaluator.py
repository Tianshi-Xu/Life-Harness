"""Original free-form Meta AgentHarness port with the shared Qwen evaluation budget.

The benchmark, disabled published layers, reward extraction and candidate validator
come from the immutable historical evaluator. New infrastructure adds a total
32768 solver context bound, 50 native workers, fail-isolated episodes, and hashing.
"""
import argparse
import copy
import importlib.util
import json
import os
import multiprocessing
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

from common import AB, DOMAINS, HERE, OLD, ROOT, digest, manifest, object_digest, save, verify_frozen

spec = importlib.util.spec_from_file_location('meta_original_native_evaluator', OLD / 'agentbench_evaluate.py')
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
core.HERE, core.ROOT, core.AB, core.MANIFEST = HERE, ROOT, AB, manifest()
TOKENIZER = None


class TokenizerProxy:
    """ALFWorld's older venv has no transformers; do not mutate that environment."""
    def __init__(self):
        self.process = subprocess.Popen([
            str(AB / '.native/core/bin/python'), '-u', str(HERE / 'tokenizer_worker.py')],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)

    def apply_chat_template(self, messages, **kwargs):
        self.process.stdin.write(json.dumps({'messages': messages, 'tools': kwargs['tools']}) + '\n')
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('Local tokenizer process exited')
        value = json.loads(line)
        if 'error' in value:
            raise ValueError(value['error'])
        return range(value['length'])


class SolverContextLimit(ValueError):
    pass


def bounded(messages, tools):
    messages = copy.deepcopy(messages)
    budget = 32768 - 4096 - 256

    def count(values):
        return len(TOKENIZER.apply_chat_template(values, tools=tools, tokenize=True,
                   add_generation_prompt=True, enable_thinking=False))

    if count(messages) <= budget:
        return messages
    if core.DOMAIN != 'webshop':
        raise SolverContextLimit('solver_context_limit_32768')
    anchors, groups, current = messages[:2], [], []
    for message in messages[2:]:
        if message.get('role') == 'assistant' and current:
            groups.append(current)
            current = []
        current.append(message)
    if current:
        groups.append(current)
    while len(groups) > 1 and count(anchors + [m for g in groups for m in g]) > budget:
        groups.pop(0)
    result = anchors + [m for g in groups for m in g]
    if count(result) > budget:
        raise SolverContextLimit('solver_context_limit_32768')
    return result


class Session(core.EpisodeSession):
    def sync_action(self):
        import requests
        self.turn += 1
        fixed = core.MANIFEST['agent']
        base = {k: copy.deepcopy(v) for k, v in fixed.items() if k != 'url'}
        base.update(messages=copy.deepcopy(self.history), tools=copy.deepcopy(self.tools),
                    tool_choice='auto', parallel_tool_calls=False)
        body = self.harness.transform_request(copy.deepcopy(base))
        assert isinstance(body, dict) and isinstance(body.get('messages'), list) and isinstance(body.get('tools'), list)
        for key in ('model', 'temperature', 'max_tokens', 'chat_template_kwargs'):
            body[key] = copy.deepcopy(base[key])
        body['messages'] = bounded(body['messages'], body['tools'])
        call = {'turn': self.turn, 'request': copy.deepcopy(body)}
        self.calls.append(call)
        for attempt in range(3):
            try:
                response = requests.post(fixed['url'], headers={'Authorization': 'Bearer EMPTY'},
                                         json=body, timeout=600)
                if response.status_code != 200:
                    call.setdefault('transport_statuses', []).append({
                        'attempt': attempt + 1, 'status': response.status_code,
                        'error_body': response.text[:2000]})
                response.raise_for_status()
                data = response.json()
                if not data.get('choices'):
                    raise ValueError('Local Qwen response has no choices')
                break
            except (requests.RequestException, ValueError) as exc:
                call.setdefault('transport_errors', []).append({
                    'attempt': attempt + 1, 'error': f'{type(exc).__name__}: {exc}'})
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))
        raw = data['choices'][0]['message']
        raw = {k: raw[k] for k in ('role', 'content', 'tool_calls') if k in raw and raw[k] is not None}
        raw.setdefault('role', 'assistant')
        message = self.harness.transform_response(copy.deepcopy(raw), copy.deepcopy(self.history))
        assert isinstance(message, dict) and message.get('role') == 'assistant'
        call.update(response=data, usage=data.get('usage', {}), transformed=copy.deepcopy(message) if message != raw else None)
        self.history.append(copy.deepcopy(message))
        return SimpleNamespace(messages=[message])


core.EpisodeSession = Session


def initialize(domain, split, plugin):
    global TOKENIZER
    core.initialize(domain, split, plugin)
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError:
        if domain != 'alfworld':
            raise
        TOKENIZER = TokenizerProxy()
    else:
        TOKENIZER = AutoTokenizer.from_pretrained(os.environ.get('LIFE_SOLVER_TOKENIZER', 'Qwen/Qwen3-4B'), local_files_only=True)


def episode(index):
    row = core.episode(index)
    if row.get('exception') and 'solver_context_limit_32768' in row['exception']:
        row.update(success=0, status='context_limit', infrastructure_error=False,
                   failure_kind='context_limit')
    if row['infrastructure_error']:
        row['success'] = 0
    return row


def valid_episode(path, index, config_sha):
    if not path.is_file():
        return False
    row = json.loads(path.read_text())
    return (row.get('index') == index and row.get('config_sha256') == config_sha
            and row.get('success') in (0, 1) and isinstance(row.get('infrastructure_error'), bool))


def validate_test_gate(domain, plugin):
    seal = json.loads((HERE / 'training_seal.json').read_text())
    for d in DOMAINS:
        state = json.loads((HERE / d / 'state.json').read_text())
        if (state['status'] != 'training_complete' or len(state['rounds']) != 6
                or digest(HERE / d / 'state.json') != seal['states'][d]):
            raise RuntimeError('All six TRAIN rounds must be sealed before TEST')
    if digest(plugin) != seal['retained_candidates'][domain]:
        raise RuntimeError('TEST candidate is not the sealed retained implementation')


def evaluate(domain, split, phase, plugin, batch=0, smoke_count=None):
    cfg = verify_frozen()
    plugin = Path(plugin).resolve()
    core.validate_plugin(plugin)
    if split == 'test':
        if phase != 'final_test' or smoke_count is not None:
            raise RuntimeError('Only one complete final TEST is permitted')
        validate_test_gate(domain, plugin)
    spec = cfg['domains'][domain]
    ids = spec['batches'][batch]['indices'] if split == 'train' else spec['test_indices']
    if smoke_count is not None:
        if phase != 'setup_smoke':
            raise RuntimeError('Partial task sets are setup checks only')
        ids = ids[:smoke_count]
    body = {'domain': domain, 'split': split, 'phase': phase, 'batch': batch if split == 'train' else None,
            'ids': ids, 'plugin_hash': digest(plugin), 'manifest_hash': digest(HERE / 'manifest.json'),
            'runner_hash': digest(__file__), 'historical_evaluator_hash': digest(OLD / 'agentbench_evaluate.py'),
            'concurrency': 50, 'max_context_tokens': 32768, 'max_output_tokens': 4096,
            'infra_error_policy': 'preserve once and count zero'}
    config = {**body, 'config_sha256': object_digest(body)}
    out = HERE / domain / 'evals' / phase
    config_path = out / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError('Evaluation resume configuration mismatch')
    save(config_path, config)
    pending = [i for i in ids if not valid_episode(out / f'episode_{i}.json', i, config['config_sha256'])]
    if pending:
        # The failed startup changed only request retry handling. Already
        # completed episodes have exactly the same model/source/IDs/budget.
        source_root = cfg.get('reused_baseline_source')
        if source_root and phase == 'batch_00_baseline' and plugin.name == 'baseline_zero.py':
            source = Path(source_root) / domain / 'evals/batch_00_baseline'
            old_config = json.loads((source / 'config.json').read_text())
            if (old_config['ids'] != ids or old_config['plugin_hash'] != digest(plugin)
                    or old_config['max_context_tokens'] != 32768 or old_config['max_output_tokens'] != 4096):
                raise RuntimeError('Reused baseline differs in tasks, identity or solver budget')
            expected = cfg['reused_baseline_episode_hashes'][domain]
            for index in pending[:]:
                name = f'episode_{index}.json'
                if name not in expected:
                    continue
                original = source / name
                if digest(original) != expected[name]:
                    raise RuntimeError('Reused episode changed')
                row = json.loads(original.read_text())
                if row['infrastructure_error']:
                    raise RuntimeError('Never reuse an infrastructure-error episode')
                row['reuse_provenance'] = {'source': str(original), 'source_sha256': digest(original),
                    'original_config_sha256': row['config_sha256']}
                row['config_sha256'] = config['config_sha256']
                save(out / name, row)
                pending.remove(index)
        print(json.dumps({'domain': domain, 'phase': phase, 'fresh_episodes': len(pending),
                          'reused_episodes': len(ids) - len(pending)}), flush=True)
    if pending:
        with ProcessPoolExecutor(max_workers=min(50, len(pending)),
                mp_context=multiprocessing.get_context('spawn'), initializer=initialize,
                initargs=(domain, split, str(plugin))) as pool:
            futures = {pool.submit(episode, i): i for i in pending}
            for future in as_completed(futures):
                row = future.result()
                if row['index'] != futures[future]:
                    raise RuntimeError('Episode ID mismatch')
                row['config_sha256'] = config['config_sha256']
                path = out / f'episode_{row["index"]}.json'
                if path.exists():
                    raise RuntimeError('Refuse to overwrite a prior episode')
                save(path, row)
                print(json.dumps({k: row[k] for k in ('index', 'success', 'status')}), flush=True)
    rows = [json.loads((out / f'episode_{i}.json').read_text()) for i in ids]
    summary = {**config, 'n': len(rows), 'successes': sum(r['success'] for r in rows),
               'accuracy': sum(r['success'] for r in rows) / len(rows),
               'infra_errors': sum(r['infrastructure_error'] for r in rows),
               'tokens': sum(r['tokens'] for r in rows), 'turns': sum(r['turns'] for r in rows),
               'new_solver_tokens': sum(r['tokens'] for r in rows if not r.get('reuse_provenance')),
               'reused_episodes': sum(bool(r.get('reuse_provenance')) for r in rows),
               'per_task': {str(r['index']): r['success'] for r in rows}}
    save(out / 'summary.json', summary)
    print(json.dumps({k: summary[k] for k in ('domain', 'phase', 'n', 'successes', 'infra_errors', 'tokens')}), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('domain', choices=DOMAINS)
    parser.add_argument('split', choices=('train', 'test'))
    parser.add_argument('phase')
    parser.add_argument('plugin', type=Path)
    parser.add_argument('--batch', type=int, default=0)
    parser.add_argument('--smoke-count', type=int)
    args = parser.parse_args()
    evaluate(args.domain, args.split, args.phase, args.plugin, args.batch, args.smoke_count)
