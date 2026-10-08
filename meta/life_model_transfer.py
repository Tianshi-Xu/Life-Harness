"""Read-only transfer of sealed six-round AgentBench harnesses to Qwen2.5."""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import json
import multiprocessing
from multiprocessing.util import Finalize
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

from meta.tau_comparison.common import ROOT, child_env, digest, load_module, read, save

EXPERIMENT = ROOT / 'meta/experiments/life_six_qwen25_transfer_20261007'
BASELINE = EXPERIMENT
COMPATIBILITY = False
OLD = ROOT / 'meta/experiments/life_agentbench_flash100k_repair_20261004'
WEB = ROOT / 'meta/experiments/life_webshop_six_current_20261007_binding_resume'
DOMAINS = ('alfworld', 'dbbench', 'webshop')
MODELS = {
    '7b': {'model': 'qwen2.5-7b-instruct', 'port': 8701, 'path': '/mnt/data/xts/models/Qwen2.5-7B-Instruct'},
    '14b': {'model': 'qwen2.5-14b-instruct', 'port': 8702, 'path': '/mnt/data/xts/models/Qwen2.5-14B-Instruct'},
}
PYTHONS = {d: ROOT / 'AgentBench/.native' / v / 'bin/python' for d, v in
           (('alfworld', 'alfworld-venv'), ('dbbench', 'core'), ('webshop', 'webshop-venv'))}
_worker = None


def normalize_tool_arguments(message, tools):
    """Repair only unambiguous single-parameter envelopes, never their values."""
    result = copy.deepcopy(message)
    schemas = {t['function']['name']: t['function']['parameters'] for t in tools}
    errors = []
    for call in result.get('tool_calls') or []:
        function = call.get('function', {})
        schema = schemas.get(function.get('name'))
        if schema is None:
            continue  # Unknown tool names retain the native task's handling.
        try:
            value = json.loads(function.get('arguments', '{}'))
        except (TypeError, ValueError):
            value = None
        if isinstance(value, dict):
            continue  # No extra policy checks on already-object arguments.
        properties = schema.get('properties', {})
        required = schema.get('required', [])
        if len(properties) == len(required) == 1 and required[0] in properties:
            name = required[0]
            kind = properties[name].get('type')
            if kind == 'string' and isinstance(value, list) and len(value) == 1:
                value = value[0]
            matches = ((kind == 'string' and isinstance(value, str)) or
                       (kind == 'array' and isinstance(value, list)))
            if matches:
                function['arguments'] = json.dumps({name: value}, ensure_ascii=False)
                continue
        errors.append({'role': 'tool', 'tool_call_id': call.get('id', ''),
            'content': 'Error: tool arguments must be a JSON object with the declared parameter names. '
                       'Nothing was executed; correct the argument format and retry.'})
    if errors:
        # Suppress the entire turn, so a bad later call cannot leave partial writes.
        bad_ids = {e['tool_call_id'] for e in errors}
        errors.extend({'role': 'tool', 'tool_call_id': c.get('id', ''),
            'content': 'Not executed: another tool call in this turn had invalid argument format.'}
            for c in result.get('tool_calls') or [] if c.get('id', '') not in bad_ids)
        result['_harness_control'] = {'action': 'suppress_tool', 'messages': errors}
    return result


def install_compatibility(task_module, domain, manifest):
    """Worker-local transport fix; selected artifacts and frozen task stay intact."""
    from meta.agentbench_hooks_v31 import runtime
    receipt = runtime.install(task_module, domain, manifest,
                              options={'h2_preserve_raw_history': False})
    base = task_module.FourHookSession

    class CompatibleSession(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            original_h2 = self.harness.h2

            def h2(message, history):
                repaired = normalize_tool_arguments(message, self.session.tools)
                if repaired.get('_harness_control'):
                    return repaired
                return original_h2(repaired, history)

            self.harness.h2 = h2

    task_module.FourHookSession = CompatibleSession
    return receipt


def enable_compatibility():
    global COMPATIBILITY, EXPERIMENT, DOMAINS
    COMPATIBILITY = True
    EXPERIMENT = ROOT / 'meta/experiments/life_six_qwen25_compat_20261007'
    DOMAINS = ('alfworld', 'dbbench')


def native_module():
    sys.path.insert(0, str(OLD))
    return load_module('life_qwen25_native', OLD / 'agentbench_evaluate.py')


def native_environment():
    java = ROOT / 'AgentBench/.native/sysroot/usr/lib/jvm/java-11-openjdk-amd64'
    return child_env(PYTHONPATH=str(ROOT), JAVA_HOME=str(java), JVM_PATH=str(java / 'lib/server/libjvm.so'),
        PATH=str(java / 'bin') + ':' + os.environ.get('PATH', ''), _JAVA_OPTIONS='-Xmx4g',
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')


def model_body(key):
    model = MODELS[key]
    return {'model': model['model'], 'url': f"http://127.0.0.1:{model['port']}/v1/chat/completions",
            'max_tokens': 4096, 'temperature': 0}


def selected(domain, native):
    from meta import harness_hooks
    root = WEB if domain == 'webshop' else OLD
    state_path = root / domain / 'state.json'
    state, seal = read(state_path), read(root / 'training_seal.json')
    manifest = root / domain / ('runtime/final/manifest.json' if domain == 'webshop'
                              else 'runtime/final_test/manifest.json')
    _, records = (harness_hooks.manifest_records(manifest) if domain == 'webshop'
                  else native.manifest_records(manifest))
    expected = seal['state_sha256'] if domain == 'webshop' else seal['state_sha256'][domain]
    if (state['round'] != 6 or len(state['rounds']) != 6 or digest(state_path) != expected
            or records != state['active_snapshot']
            or native.object_digest(records) != state['active_snapshot_sha256']
            or (domain == 'webshop' and records != seal['snapshot'])):
        raise ValueError('Selected chain does not match the sealed six-round incumbent')
    artifact_root = Path(read(manifest)['artifact_root'])
    hashes = {str(manifest): digest(manifest), str(state_path): digest(state_path),
              str(root / 'training_seal.json'): digest(root / 'training_seal.json')}
    for record in records:
        path = artifact_root / record['artifact_path']
        if digest(path) != record['sha256']:
            raise ValueError('Selected harness content changed')
        hashes[str(path)] = record['sha256']
    return {'manifest': str(manifest), 'snapshot_sha256': native.object_digest(records),
            'source_hashes': hashes, 'interface': 'unified-v2' if domain == 'webshop' else 'legacy-v31'}


class TokenizerService:
    """Exact Qwen2 tokenizer in the existing core env for old ALFWorld Python."""
    def __init__(self, model_path):
        self.process = subprocess.Popen([str(PYTHONS['dbbench']), '-u', '-m',
            'meta.life_model_transfer', '--tokenizer-worker', '--model-path', model_path],
            cwd=ROOT, env=native_environment(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1)
        Finalize(self, self.close, exitpriority=10)

    def apply_chat_template(self, messages, **kwargs):
        self.process.stdin.write(json.dumps({'messages': messages, 'tools': kwargs['tools']}) + '\n')
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('Qwen2.5 tokenizer worker exited')
        value = json.loads(line)
        if 'error' in value:
            raise ValueError(value['error'])
        return range(value['length'])

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=5)


def initialize_worker(domain, split, manifest, model_key, compatibility=False):
    global _worker
    _worker = native_module()
    _worker.HISTORICAL = copy.deepcopy(_worker.HISTORICAL)
    _worker.HISTORICAL['agent'] = model_body(model_key)
    if compatibility:
        _worker.inject_task_module = lambda d, task, path: install_compatibility(task, d, path)
    elif domain == 'webshop':
        from meta import harness_hooks
        _worker.inject_task_module = lambda d, task, path: harness_hooks.install_agentbench(task, d, path)
    model_path = MODELS[model_key]['path']
    if domain == 'alfworld':
        _worker.TokenizerProxy = lambda: TokenizerService(model_path)
        _worker.initialize(domain, split, manifest)
    else:
        from transformers import AutoTokenizer
        loader = AutoTokenizer.from_pretrained
        with patch.object(AutoTokenizer, 'from_pretrained',
                          side_effect=lambda *a, **kw: loader(model_path, local_files_only=True)):
            _worker.initialize(domain, split, manifest)


def episode_worker(index):
    row = _worker.episode(index)
    # The frozen DB task assumes JSON-object arguments and throws AttributeError
    # on valid JSON scalars. That is solver schema noncompliance, not a dead DB.
    # Do not coerce/fix the answer or replay it; keep its original zero outcome.
    if row['infrastructure_error'] and row['status'] == 'task error' and not row.get('exception'):
        calls = row.get('calls', [])
        message = (calls[-1].get('response', {}).get('choices', [{}])[0].get('message', {})
                   if calls else {})
        for call in message.get('tool_calls') or []:
            try:
                arguments = json.loads(call['function']['arguments'])
            except (KeyError, TypeError, ValueError):
                continue
            if not isinstance(arguments, dict):
                row.update(infrastructure_error=False, native_infrastructure_error=True,
                           failure_kind='model_tool_argument_schema', success=0)
                break
    return row


def prepare():
    native = native_module()
    original = native.validate_protocol()
    # A model-only replay must not require obsolete optimizer-broker source files.
    # Verify the original seal/history fully; record (do not silently accept) only
    # unused external optimizer transport differences from the old fingerprint.
    import driver
    if read(OLD / 'training_seal.json') != driver.seal_value():
        raise ValueError('Original training seal changed')
    transport_changes = {}
    for domain in DOMAINS:
        before, now = read(OLD / domain / 'run_config.json'), driver.run_fingerprint(domain)
        if {k: v for k, v in before.items() if k != 'transport_sources'} != {
                k: v for k, v in now.items() if k != 'transport_sources'}:
            raise ValueError('Original evaluator/optimizer source fingerprint changed')
        driver.verify_history(domain, read(OLD / domain / 'state.json'))
        transport_changes[domain] = {p: {'original': sha, 'current': now['transport_sources'][p]}
            for p, sha in before['transport_sources'].items() if sha != now['transport_sources'][p]}
    choices = {d: selected(d, native) for d in DOMAINS}
    frozen = {str(Path(__file__)): digest(__file__)}
    for name in ('agentbench_evaluate.py', 'agentbench_chain.py', 'protocol.py'):
        frozen[str(OLD / name)] = digest(OLD / name)
    for name in ('meta/harness_hooks.py', 'meta/life_agentbench_run.py'):
        frozen[str(ROOT / name)] = digest(ROOT / name)
    protocol = {'models': MODELS, 'domains': choices,
        'test_ids': {d: native.task_ids(original['domains'][d], 0, 'test') for d in DOMAINS},
        'smoke_ids': {d: native.task_ids(original['domains'][d], 0, 'train')[:1] for d in DOMAINS},
        'source_hashes': frozen, 'context_tokens': 32768, 'output_tokens': 4096,
        'temperature': 0, 'workers_per_model': 50, 'optimizer_calls': 0,
        'unused_optimizer_transport_changes': transport_changes,
        'selection': 'existing sealed TRAIN-selected six-round artifacts; no new selection',
        'prior_test_exposure': True, 'original_task_manifest_sha256': digest(native.HISTORICAL_ROOT / 'manifest.json')}
    if COMPATIBILITY:
        protocol.update(compatibility={'tool_arguments': 'schema-driven single-parameter envelope only',
            'h2_preserve_raw_history': False, 'policy_artifacts_changed': False},
            baseline_results_sha256=digest(BASELINE / 'FINAL_RESULTS.json'))
        protocol['smoke_ids'] = {d: native.task_ids(original['domains'][d], 0, 'train')[:50]
                                 for d in DOMAINS}
    if [len(protocol['test_ids'][d]) for d in DOMAINS] != ([109, 300] if COMPATIBILITY else [109, 300, 200]):
        raise ValueError('Full frozen TEST denominator changed')
    path = EXPERIMENT / 'protocol.json'
    if path.exists() and read(path) != protocol:
        raise ValueError('Model transfer protocol changed on resume')
    save(path, protocol)
    return protocol


def evaluate(model_key, domain, smoke=False):
    native = native_module()
    protocol = read(EXPERIMENT / 'protocol.json')
    choice = selected(domain, native)
    if choice != protocol['domains'][domain] or any(digest(p) != h for p, h in protocol['source_hashes'].items()):
        raise ValueError('Frozen model transfer sources changed')
    model = MODELS[model_key]
    model_path = Path(model['path'])
    ids = protocol['smoke_ids' if smoke else 'test_ids'][domain]
    phase, split = ('smoke', 'train') if smoke else ('final_test', 'test')
    directory = EXPERIMENT / model_key / domain / phase
    body = {'evaluator_schema_version': 1, 'model_key': model_key, 'agent': model_body(model_key),
        'domain': domain, 'split': split, 'phase': phase, 'ids': ids,
        'selected_harness': choice, 'protocol_sha256': digest(EXPERIMENT / 'protocol.json'),
        'model_files_sha256': {n: digest(model_path / n) for n in
            ('config.json', 'tokenizer_config.json', 'tokenizer.json', 'model.safetensors.index.json')},
        'concurrency': min(50, len(ids)), 'max_context_tokens': 32768, 'max_output_tokens': 4096}
    if COMPATIBILITY:
        body['compatibility'] = protocol['compatibility']
    config = {**body, 'config_sha256': native.object_digest(body)}
    if (directory / 'config.json').exists() and read(directory / 'config.json') != config:
        raise ValueError('Resume evaluation configuration mismatch')
    save(directory / 'config.json', config)
    if (directory / 'summary.json').exists():
        result = read(directory / 'summary.json')
        native.validate_cached_summary(directory, result)
        return result
    pending = [i for i in ids if not native._episode_valid(directory / f'episode_{i}.json', i, config['config_sha256'])]
    initargs = (domain, split, choice['manifest'], model_key)
    if COMPATIBILITY:
        initargs += (True,)
    with ProcessPoolExecutor(max_workers=config['concurrency'], mp_context=multiprocessing.get_context('spawn'),
            initializer=initialize_worker, initargs=initargs) as pool:
        futures = {pool.submit(episode_worker, i): i for i in pending}
        for future in as_completed(futures):
            row = native._jsonable(future.result())
            if row['index'] != futures[future]:
                raise ValueError('Worker returned a wrong episode')
            row.update(episode_schema_version=1, config_sha256=config['config_sha256'])
            save(directory / f"episode_{row['index']}.json", row)
            print(json.dumps({k: row[k] for k in ('index', 'success', 'status', 'infrastructure_error')}), flush=True)
    rows = [read(directory / f'episode_{i}.json') for i in ids]
    result = {**config, 'n': len(ids), 'successes': sum(r['success'] for r in rows),
        'accuracy': sum(r['success'] for r in rows) / len(ids),
        'infra_errors': sum(r['infrastructure_error'] for r in rows),
        'tokens': sum(r['tokens'] for r in rows), 'turns': sum(r['turns'] for r in rows),
        'per_task': {str(r['index']): r['success'] for r in rows}}
    save(directory / 'summary.json', result)
    native.validate_cached_summary(directory, result)
    return result


def run_model(model_key):
    for smoke in (True, False):
        for domain in DOMAINS:
            phase = 'smoke' if smoke else 'final_test'
            log = EXPERIMENT / model_key / f'{domain}_{phase}.log'
            log.parent.mkdir(parents=True, exist_ok=True)
            command = [str(PYTHONS[domain]), '-u', '-m', 'meta.life_model_transfer',
                       '--model', model_key, '--domain', domain] + (['--smoke'] if smoke else [])
            if COMPATIBILITY:
                command.append('--compatibility')
            with log.open('a') as stream:
                subprocess.run(command, cwd=ROOT, env=native_environment(), stdout=stream,
                               stderr=subprocess.STDOUT, check=True)
            summary = read(EXPERIMENT / model_key / domain / phase / 'summary.json')
            print(json.dumps({'model': model_key, 'domain': domain, 'phase': phase,
                              **{k: summary[k] for k in ('n', 'successes', 'accuracy', 'infra_errors')}}), flush=True)
            if smoke and summary['infra_errors']:
                raise RuntimeError('Native smoke infrastructure failure; inspect before full TEST')


def tokenizer_worker(model_path):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    for line in sys.stdin:
        try:
            value = json.loads(line)
            result = {'length': len(tokenizer.apply_chat_template(value['messages'], tools=value['tools'],
                tokenize=True, add_generation_prompt=True, enable_thinking=False))}
        except Exception as exc:
            result = {'error': str(exc)}
        print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--model', choices=MODELS)
    parser.add_argument('--domain', choices=DOMAINS)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--tokenizer-worker', action='store_true')
    parser.add_argument('--model-path')
    parser.add_argument('--compatibility', action='store_true')
    args = parser.parse_args()
    if args.compatibility:
        enable_compatibility()
    if args.tokenizer_worker:
        tokenizer_worker(args.model_path)
    elif args.prepare:
        print(json.dumps(prepare()))
    elif args.run:
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(run_model, MODELS))
    else:
        print(json.dumps(evaluate(args.model, args.domain, args.smoke)))


if __name__ == '__main__':
    main()
