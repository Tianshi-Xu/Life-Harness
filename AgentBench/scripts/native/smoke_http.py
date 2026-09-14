"""Exercise the controller/worker HTTP protocol with scripted tool calls."""
import json
import requests

BASE = 'http://127.0.0.1:15020/api'
results = []
workers = requests.get(BASE + '/list_workers', timeout=20)
workers.raise_for_status()
for name in ('dbbench-std', 'alfworld-std'):
    assert name in workers.json(), name
    response = requests.post(BASE + '/start_sample', json={'name': name, 'index': 0}, timeout=90)
    response.raise_for_status()
    sid = response.headers.get('session_id') or response.json().get('session_id')
    assert sid is not None, response.text
    headers = {'session_id': str(sid)}
    calls = [('execute_sql', {'query': 'SELECT 1'}), ('commit_final_answer', {'answers': ['1']})] if name == 'dbbench-std' else [('take_action', {'action': 'inventory'})]
    statuses = [response.json().get('status')]
    try:
        for i, (tool, args) in enumerate(calls):
            payload = {'messages': [{'role': 'assistant', 'content': '', 'tool_calls': [{'id': f'http-smoke-{i}', 'type': 'function', 'function': {'name': tool, 'arguments': json.dumps(args)}}]}]}
            response = requests.post(BASE + '/interact', json=payload, headers=headers, timeout=90)
            response.raise_for_status()
            output = response.json()
            assert 'error' not in str(output.get('status', '')).lower(), output
            statuses.append(output.get('status'))
        if name == 'dbbench-std':
            assert output.get('finish'), output
        results.append({'name': name, 'indices': len(workers.json()[name]['indices']), 'statuses': statuses, 'scripted_tool_calls': len(calls)})
    finally:
        # A finished DB task may already be removed. ALFWorld must be cancelled.
        cancel = requests.post(BASE + '/cancel', json={}, headers=headers, timeout=30)
        if name == 'alfworld-std':
            cancel.raise_for_status()
print(json.dumps(results, indent=2))
