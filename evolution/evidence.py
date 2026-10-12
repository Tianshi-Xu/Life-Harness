"""Public-only evidence packs and deterministic, budgeted retrieval for AgentBench.

Budgets bound distinct serialized evidence exposed, not repeated tool reads or
model context. Full episodes remain host-side. Signals are observations, not causes.
"""
from __future__ import annotations

import collections
import hashlib
import json
import random
import re
from pathlib import Path


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def encoded(value):
    return json.dumps(value, ensure_ascii=False, indent=2) + '\n'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def public_episode(row):
    messages = []
    turn = 0
    for ordinal, raw in enumerate(row['history']):
        if raw.get('role') not in {'system', 'user', 'assistant', 'tool'}:
            continue
        if str(raw.get('content', '')).startswith('[HARNESS_TRACE_V1]'):
            continue
        if raw['role'] == 'assistant':
            turn += 1
        msg = {k: raw[k] for k in ('role', 'content', 'tool_call_id', 'name') if k in raw}
        if raw.get('tool_calls'):
            msg['tool_calls'] = [
                {'id': c.get('id'), 'type': c.get('type', 'function'),
                 'function': {k: c.get('function', {}).get(k) for k in ('name', 'arguments')}}
                for c in raw['tool_calls']]
        messages.append({'message_index': ordinal, 'model_turn': turn or None, 'message': msg})
    # Only assign model turn numbers when one executed assistant matches each call.
    aligned = turn == row['turns'] and [c['turn'] for c in row['calls']] == list(range(1, turn + 1))
    if not aligned:
        for msg in messages:
            msg['model_turn'] = None
    interventions = []
    for event in row.get('harness_trace', []):
        layer = event.get('layer')
        if layer in {'h2', 'h3', 'h4', 'h5'}:
            # These fields contain only hook input/output from the public interface.
            interventions.append({k: event[k] for k in ('layer', 'turn', 'before', 'after', 'hints') if k in event})
    return {'task_id': str(row['index']), 'success': int(row['success']),
            'status': row['status'], 'turns': row['turns'], 'tokens': row.get('tokens', 0),
            'turn_alignment_verified': aligned, 'messages': messages, 'interventions': interventions}


def classify(episode):
    text = '\n'.join(str(x['message'].get('content') or '') for x in episode['messages']
                     if x['message']['role'] in {'tool', 'user'})
    signals = set()
    for pattern, signal in [
        (r'\b1064\b|SQL syntax', 'sql_syntax'),
        (r'\b1054\b|unknown column', 'sql_unknown_column'),
        (r'\b1146\b|table .*doesn.t exist', 'sql_missing_table'),
        (r'invalid action|nothing happens|not a valid action', 'invalid_action'),
    ]:
        if re.search(pattern, text, re.I):
            signals.add(signal)
    actions = [json.dumps(x['message'].get('tool_calls') or x['message'].get('content'), sort_keys=True)
               for x in episode['messages'] if x['message']['role'] == 'assistant']
    if any(a == b == c for a, b, c in zip(actions, actions[1:], actions[2:])):
        signals.add('repeated_action')
    if any(x['message']['role'] == 'assistant' and not x['message'].get('tool_calls') for x in episode['messages']):
        signals.add('assistant_without_tool_call')
    if 'limit' in episode['status'].lower():
        signals.add('budget_exhausted')
    return sorted(signals or {'unclassified'})


def signature(episode):
    # Public instruction features only; no dataset task-type labels.
    user = next((str(m['message'].get('content') or '') for m in episode['messages']
                 if m['message']['role'] == 'user'), '')
    return set(re.findall(r'[a-z_]+', user.lower()))


def representatives(episodes, maximum, seed):
    groups = collections.defaultdict(list)
    for task, ep in episodes.items():
        if not ep['success']:
            for signal in classify(ep):
                groups[signal].append(task)
    rng = random.Random(seed)
    selected = []
    # Alternating frequent and rare groups protects minority failures.
    names = sorted(groups, key=lambda g: (-len(groups[g]), g))
    order = []
    while names:
        order.append(names.pop(0))
        if names:
            order.append(names.pop())
    successes = [k for k, e in episodes.items() if e['success']]
    for group in order:
        candidates = sorted(groups[group])
        rng.shuffle(candidates)
        task = next((k for k in candidates if k not in selected), None)
        if task is None:
            continue
        selected.append(task)
        if successes:
            a = signature(episodes[task])
            match = max(successes, key=lambda k: (len(a & signature(episodes[k])) / max(1, len(a | signature(episodes[k]))), k))
            if match not in selected:
                selected.append(match)
    # Reserve roughly one quarter for random spot checks.
    core = selected[:max(1, maximum * 3 // 4)]
    remaining = sorted(set(episodes) - set(core))
    rng.shuffle(remaining)
    return (core + remaining)[:maximum]


def build_pack(eval_dir, host_dir, visible_dir, *, seed=0, max_cases=12, initial_chars=24000, total_chars=80000):
    eval_dir, host_dir, visible_dir = map(Path, (eval_dir, host_dir, visible_dir))
    if initial_chars > total_chars or initial_chars < 2000 or max_cases < 1:
        raise ValueError('Invalid evidence budgets')
    summary = json.loads((eval_dir / 'summary.json').read_text())
    if summary['infra_errors'] or summary['n'] != len(summary['per_task']):
        raise ValueError('Evidence requires a complete, valid evaluation')
    episodes = {}
    for task in summary['per_task']:
        row = json.loads((eval_dir / f'episode_{task}.json').read_text())
        if row['infrastructure_error'] or str(row['index']) != task or row['success'] != summary['per_task'][task]:
            raise ValueError('Episode/summary mismatch')
        episodes[task] = public_episode(row)
    groups = collections.Counter(g for e in episodes.values() if not e['success'] for g in classify(e))
    hooks = collections.Counter(x['layer'] for e in episodes.values() for x in e['interventions'])
    styles = collections.defaultdict(lambda: {'n': 0, 'successes': 0})
    for ep in episodes.values():
        instruction = next((str(m['message'].get('content') or '') for m in ep['messages'] if m['message']['role'] == 'user'), '')
        word = re.search(r'[a-z]+', instruction.lower())
        first = word.group() if word else 'empty'
        style = first if first in {'find', 'what', 'which', 'how', 'insert', 'update', 'add', 'change', 'list', 'get', 'count', 'calculate', 'your'} else 'other'
        styles[style]['n'] += 1
        styles[style]['successes'] += ep['success']
    provenance = {k: summary[k] for k in ('plugin_hash', 'runner_hash', 'manifest_hash')}
    index = {'schema': 1, 'provenance': provenance, 'n': summary['n'], 'successes': summary['successes'],
             'tokens': sum(e['tokens'] for e in episodes.values()), 'turns': sum(e['turns'] for e in episodes.values()),
             'status_counts': dict(collections.Counter(e['status'] for e in episodes.values())),
             'failure_signals': dict(groups), 'hook_counts': dict(hooks), 'public_instruction_styles': dict(styles),
             'note': 'Signals overlap and do not establish causes. Task types are not oracle labels.', 'cases': []}
    host_dir.mkdir(parents=True, exist_ok=True)
    visible_dir.mkdir(parents=True, exist_ok=True)
    for task, ep in episodes.items():
        save(host_dir / 'episodes' / f'{task}.json', {**ep, 'provenance': provenance})
    for task in representatives(episodes, max_cases, seed):
        ep = episodes[task]
        case = {'task_id': task, 'success': ep['success'], 'signals': classify(ep), 'turns': ep['turns'],
                'preview': [{'message_index': m['message_index'], 'model_turn': m['model_turn'],
                             'role': m['message']['role'], 'text': json.dumps(m['message'], ensure_ascii=False)[:700]}
                            for m in ep['messages'][-3:]]}
        index['cases'].append(case)
        if len(encoded(index)) > initial_chars:
            index['cases'].pop()
            break
    if len(encoded(index)) > initial_chars:
        raise ValueError('Summary alone exceeds initial budget')
    save(visible_dir / 'index.json', index)
    # Per-task metadata is host-side, for group-based retrieval without a huge visible index.
    save(host_dir / 'catalog.json', {k: {'success': e['success'], 'signals': classify(e), 'turns': e['turns']} for k, e in episodes.items()})
    ledger = {'total_chars': total_chars, 'exposed_chars': len(encoded(index)), 'requests': [], 'delivered': [], 'seed': seed}
    save(host_dir / 'ledger.json', ledger)
    return index


def retrieve(host_dir, visible_dir, requests, request_id):
    """Host-mediated request: task_id OR group, optionally inclusive turn range.

    Returns whole JSON objects or a budget denial; never silently truncates traces.
    Requests and repeated invocations are deterministic and recorded for resume.
    """
    host_dir, visible_dir = Path(host_dir), Path(visible_dir)
    journal = host_dir / f'retrieval_{request_id}.json'
    if journal.exists():
        committed = json.loads(journal.read_text())
        if committed['requests'] != requests:
            raise ValueError('Retrieval batch changed on resume')
        save(visible_dir / f'details_{request_id}.json', committed['result'])
        current = json.loads((host_dir / 'ledger.json').read_text())
        if not any(r['batch'] == request_id for r in current['requests']):
            save(host_dir / 'ledger.json', committed['ledger'])
        return committed['result']
    ledger = json.loads((host_dir / 'ledger.json').read_text())
    catalog = json.loads((host_dir / 'catalog.json').read_text())
    if not isinstance(requests, list) or len(requests) > 12:
        raise ValueError('At most 12 retrieval requests per batch')
    remaining = ledger['total_chars'] - ledger['exposed_chars']
    out = [{'request_index': i, 'denied': 'budget; request fewer/narrower excerpts'}
           for i in range(len(requests))]
    if len(encoded({'results': out})) > remaining:
        out = []
        result = {'results': [], 'denied': 'remaining budget exhausted'}
        if len(encoded(result)) > remaining:
            return None
    else:
        for index, request in enumerate(requests):
            if not isinstance(request, dict) or set(request) - {'task_id', 'group', 'start_turn', 'end_turn'}:
                raise ValueError('Invalid retrieval request')
            if ('task_id' in request) == ('group' in request):
                raise ValueError('Specify task_id OR group')
            task = str(request.get('task_id', ''))
            if 'group' in request:
                matches = sorted(k for k, v in catalog.items() if not v['success'] and request['group'] in v['signals'])
                matches.sort(key=lambda k: (k in ledger['delivered'], k))
                task = matches[0] if matches else ''
            delivered = False
            if task not in catalog:
                payload = {'request': request, 'denied': 'unknown task/group'}
            else:
                ep = json.loads((host_dir / 'episodes' / f'{task}.json').read_text())
                start, end = request.get('start_turn', 1), request.get('end_turn', ep['turns'])
                if type(start) is not int or type(end) is not int or not 1 <= start <= end <= ep['turns']:
                    payload = {'request': request, 'denied': 'invalid inclusive model-turn range',
                               'selected_task_id': task, 'available_turns': [1, ep['turns']]}
                elif (start, end) != (1, ep['turns']) and not ep['turn_alignment_verified']:
                    payload = {'request': request, 'denied': 'model-turn mapping unavailable; request whole episode'}
                else:
                    if (start, end) != (1, ep['turns']):
                        ep['messages'] = [m for m in ep['messages'] if m['model_turn'] is None or start <= m['model_turn'] <= end]
                        ep['interventions'] = [e for e in ep['interventions'] if 'turn' not in e or start <= e['turn'] <= end]
                        ep['excerpt_turn_range'] = [start, end]
                    payload = {'request': request, 'episode': ep}
                    delivered = True
            # Account for the EXACT final serialization including all remaining denials.
            # A large request must never discard already admitted, smaller excerpts.
            proposed = out[:index] + [payload] + out[index + 1:]
            if len(encoded({'results': proposed})) <= remaining:
                out = proposed
                if delivered:
                    ledger['delivered'].append(task)
        result = {'results': out}
    cost = len(encoded(result))
    ledger['exposed_chars'] += cost
    ledger['requests'].append({'batch': request_id, 'requests': requests, 'chars': cost})
    save(journal, {'requests': requests, 'result': result, 'ledger': ledger})
    save(visible_dir / f'details_{request_id}.json', result)
    save(host_dir / 'ledger.json', ledger)
    return result


def screen_ids(episodes, size, seed):
    if size < 1:
        raise ValueError('Screen size must be positive')
    return representatives(episodes, min(size, len(episodes)), seed)


def screen_decision(candidate, incumbent, iteration, explore_every=3):
    if set(candidate) != set(incumbent) or not candidate:
        raise ValueError('Screen pairs must match')
    wins = sum(candidate[k] > incumbent[k] for k in candidate)
    losses = sum(candidate[k] < incumbent[k] for k in candidate)
    explore = explore_every > 0 and iteration % explore_every == 0
    # Ties / no signal are allowed through: a small screen cannot disprove long-tail gains.
    return {'wins': wins, 'losses': losses, 'exploration': explore, 'run_full': wins >= losses or explore}
