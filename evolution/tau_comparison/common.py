"""Frozen protocol, public export and credential-free infrastructure helpers."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
TAU = ROOT / 'TauBench'
AHE_NATIVE = ROOT / 'integrations/ahe-native'
HISTORICAL = ROOT / 'evolution/reproduction/reference'
META_TRANSPORT = ROOT / 'evolution/reproduction/meta_transport'
DOMAINS = ('airline', 'retail', 'telecom')
ALI_URL = 'https://maas.qianwenaiapi.com/compatible-mode/v1'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def transport_modules():
    # Reuse the audited Meta API shell and host ledger without changing its files.
    sys.path.insert(0, str(META_TRANSPORT))
    ledger = load_module('tau_meta_ledger', META_TRANSPORT / 'usage.py')
    broker = load_module('tau_meta_broker', META_TRANSPORT / 'api_broker.py')
    return ledger.Ledger, broker.TokenBroker


def child_env(**updates):
    env = {k: v for k, v in os.environ.items() if not any(
        word in k.upper() for word in ('KEY', 'TOKEN', 'SECRET', 'PASSWORD', 'CREDENTIAL', 'AUTH'))}
    env.update(PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false',
               NO_PROXY='127.0.0.1,localhost', no_proxy='127.0.0.1,localhost',
               OPENAI_API_KEY='EMPTY', AGENT_API_KEY='EMPTY', TAU_USER_LLM_MIN_INTERVAL_SECONDS='0')
    env.update(updates)
    return env


def aliyun_key():
    key = os.environ.get('DASHSCOPE_API_KEY')
    if not key:
        value = subprocess.run(['bash', '-c', 'source "$1"; printf "%s" "$DASHSCOPE_API_KEY"',
            'bash', str(ROOT / 'env.sh')], capture_output=True, timeout=30, check=True)
        key = value.stdout.decode().strip()
    if not key:
        raise RuntimeError('DASHSCOPE_API_KEY is missing')
    return key


def user_model(domain):
    if domain not in DOMAINS:
        raise ValueError('unknown Tau domain')
    return 'openai/deepseek-v4-flash' if domain == 'telecom' else 'openai/qwen3.8-flash'


def public_message(message):
    if not isinstance(message, dict) or message.get('role') not in ('system', 'user', 'assistant', 'tool'):
        return None
    if str(message.get('content', '')).startswith('[HARNESS_TRACE_V1]'):
        return None
    result = {k: message[k] for k in ('role', 'content', 'tool_call_id', 'name') if k in message}
    if message.get('tool_calls'):
        result['tool_calls'] = []
        for call in message['tool_calls']:
            function = call.get('function') or call
            arguments = function.get('arguments')
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, ensure_ascii=False)
            result['tool_calls'].append({'id': call.get('id'), 'type': 'function',
                'function': {'name': function.get('name'), 'arguments': arguments}})
    if message['role'] == 'tool' and 'tool_call_id' not in result and message.get('id'):
        result['tool_call_id'] = message['id']
    return result


def public_simulation(simulation):
    reward = float((simulation.get('reward_info') or {}).get('reward', 0))
    messages = [m for raw in simulation.get('messages', []) if (m := public_message(raw)) is not None]
    return {'task_id': str(simulation['task_id']), 'trial': simulation.get('trial', 0),
        'reward': reward, 'termination_reason': simulation.get('termination_reason'), 'messages': messages,
        'agent_system_prompt': simulation.get('agent_system_prompt'),
        'agent_tools_schema': simulation.get('agent_tools_schema')}


def summarize_rows(ids, trials, simulations):
    """Keep fixed task/trial denominators, including native episodes that failed."""
    outcomes = {str(task): [] for task in ids}
    infra = 0
    seen = set()
    for row in simulations:
        key = (str(row['task_id']), row.get('trial', 0))
        if key in seen or key[0] not in outcomes or type(key[1]) is not int or not 0 <= key[1] < trials:
            raise ValueError('unexpected or duplicate native task/trial')
        seen.add(key)
        reward = float((row.get('reward_info') or {}).get('reward', row.get('reward', 0)))
        if reward not in (0.0, 1.0):
            raise ValueError('native pass@1 requires binary complete reward')
        if row.get('infrastructure_error'):
            infra += 1
            if reward:
                raise ValueError('infra outcomes must be zero')
        outcomes[key[0]].append(reward)
    missing = len(ids) * trials - len(seen)
    for values in outcomes.values():
        values.extend([0.0] * (trials - len(values)))
    total = sum(sum(values) for values in outcomes.values())
    return {'n': len(ids), 'episodes': len(ids) * trials, 'successes': int(total),
        'accuracy': total / (len(ids) * trials), 'infra_errors': infra + missing,
        'missing_episodes': missing, 'per_task': outcomes}


def source_fingerprint():
    paths = list(Path(__file__).parent.glob('*.py'))
    paths += [META_TRANSPORT / name for name in ('api_agent.py', 'api_broker.py', 'usage.py', 'common.py')]
    paths += list((AHE_NATIVE / 'ahe_native').glob('*.py'))
    paths += [p for p in (TAU / 'src/tau2').rglob('*.py') if '__pycache__' not in p.parts]
    paths += [TAU / 'scripts/eval_harness.py', HISTORICAL / 'manifest.json']
    paths += [p for d in DOMAINS for p in (TAU / 'data/tau2/domains' / d).rglob('*')
              if p.is_file() and '__pycache__' not in p.parts]
    return {str(p): digest(p) for p in sorted(set(paths))}
