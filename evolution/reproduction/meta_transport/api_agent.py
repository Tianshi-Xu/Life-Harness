"""API execution shell only: the frozen upstream prompt owns the Meta workflow.

Implements the historical Read/Write/Edit/Glob/Grep/Bash/Agent tool surface.
An Agent call creates an independent context, as the upstream skill requires.
No failure diagnosis, mechanism selection, case quota, or Life policy is baked in.
"""
import copy
import json
import os
import re
import subprocess
import time
import urllib.request
import uuid
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def tool(name, description, properties, required):
    return {'type': 'function', 'function': {'name': name, 'description': description,
        'parameters': {'type': 'object', 'properties': properties, 'required': required}}}


STRING = {'type': 'string'}
TOOLS = [
    tool('Read', 'Read a public workspace file with optional 1-based offset and line limit. '
         'Omitting the limit reads the complete file. No case quota.',
         {'file_path': STRING, 'offset': {'type': 'integer'}, 'limit': {'type': 'integer'}}, ['file_path']),
    tool('Write', 'Write a new candidate Python file or the prescribed pending_eval.json.',
         {'file_path': STRING, 'content': STRING}, ['file_path', 'content']),
    tool('Edit', 'Replace an exact string in a writable newly generated file.',
         {'file_path': STRING, 'old_string': STRING, 'new_string': STRING,
          'replace_all': {'type': 'boolean'}}, ['file_path', 'old_string', 'new_string']),
    tool('Glob', 'List public files matching a glob in the workspace.',
         {'pattern': STRING, 'path': STRING}, ['pattern']),
    tool('Grep', 'Search public workspace text using a regular expression.',
         {'pattern': STRING, 'path': STRING, 'glob': STRING, 'output_mode': STRING,
          'head_limit': {'type': 'integer'}, 'case_insensitive': {'type': 'boolean'}}, ['pattern']),
    tool('Bash', 'Run a local pure import/prototype check inside the isolated public workspace. '
         'No benchmark, network call, package installation, or other experiment access. Python is available; uv is not.',
         {'command': STRING, 'timeout': {'type': 'integer'}, 'description': STRING}, ['command']),
    tool('Agent', 'Launch the upstream skill\'s ONE analysis Agent, then ONE implementation Agent. '
         'They use independent contexts and share this workspace. No nested delegation.',
         {'prompt': STRING, 'description': STRING, 'subagent_type': STRING}, ['prompt']),
]


class AgentRunner:
    def __init__(self, workspace, output, request, *, completion=None, allow_bash=False):
        self.workspace, self.output = Path(workspace).resolve(), Path(output)
        self.request = request
        self.deadline = time.monotonic() + request.get('timeout_seconds', 5400)
        self.completion = completion or self.api_call
        self.allow_bash = allow_bash
        self.initial = {str(p.relative_to(self.workspace)): p.read_bytes()
                        for p in self.workspace.rglob('*') if p.is_file()}

    def api_call(self, messages, tools, role, request_id):
        payload = {'model': 'deepseek-v4-flash', 'messages': messages, 'tools': tools,
                   'tool_choice': 'auto', 'max_tokens': 60000, 'stream': False}
        req = urllib.request.Request(self.request['broker_url'] + '/chat/completions', method='POST',
            data=json.dumps(payload, ensure_ascii=False).encode(),
            headers={'Authorization': 'Bearer EMPTY', 'Content-Type': 'application/json',
                     'X-Meta-Domain': self.request['domain'], 'X-Meta-Round': str(self.request['round']),
                     'X-Meta-Role': role, 'X-Meta-Request-ID': request_id})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req, timeout=max(1, min(2100, self.deadline - time.monotonic()))) as response:
            return json.load(response)

    def path(self, value, write=False):
        path = Path(value)
        if not path.is_absolute():
            path = self.workspace / path
        path = path.resolve()
        try:
            rel = path.relative_to(self.workspace)
        except ValueError:
            raise ValueError('Path escapes the public workspace')
        if rel.parts and rel.parts[0] not in (
                'agents', 'source_view', 'evidence', 'logs', 'jobs', 'history.json', 'upstream_SKILL.md'):
            raise ValueError('Path is outside the historical public tool contract')
        if write:
            if str(rel) in self.initial:
                raise ValueError('An existing frozen input cannot be modified')
            candidate = rel.parts[0] == 'agents' and len(rel.parts) == 2 and path.suffix == '.py'
            pending = path.name == 'pending_eval.json' and rel.parts[0] == 'logs'
            if not (candidate or pending):
                raise ValueError('Write only a new agents/*.py or prescribed pending_eval.json')
        return path

    def dispatch(self, name, args, state, role):
        if name == 'Read':
            lines = self.path(args['file_path']).read_text().splitlines()
            start = max(0, args.get('offset', 1) - 1)
            limit = args.get('limit')
            end = start + limit if limit is not None else len(lines)
            return '\n'.join(f'{i+1}: {line}' for i, line in enumerate(lines[start:end], start))
        if name == 'Write':
            path = self.path(args['file_path'], write=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(args['content'])
            return 'Written ' + str(path.relative_to(self.workspace))
        if name == 'Edit':
            path = self.path(args['file_path'], write=True)
            text, old = path.read_text(), args['old_string']
            if not old or old not in text or (text.count(old) != 1 and not args.get('replace_all')):
                raise ValueError('old_string is absent or ambiguous')
            path.write_text(text.replace(old, args['new_string'], -1 if args.get('replace_all') else 1))
            return 'Edited ' + str(path.relative_to(self.workspace))
        if name in ('Glob', 'Grep'):
            root = self.path(args.get('path') or '.')
            paths = ([root] if root.is_file() else root.glob(args.get('pattern', '**/*'))
                     if name == 'Glob' else root.glob(args.get('glob', '**/*')))
            output = []
            pattern = (re.compile(args['pattern'], re.I if args.get('case_insensitive') else 0)
                       if name == 'Grep' else None)
            for path in sorted(paths):
                if not path.is_file():
                    continue
                path = self.path(str(path))
                rel = str(path.relative_to(self.workspace))
                if name == 'Glob':
                    output.append(rel)
                    continue
                try:
                    lines = path.read_text().splitlines()
                except UnicodeDecodeError:
                    continue
                hits = [f'{rel}:{n}: {line}' for n, line in enumerate(lines, 1) if pattern.search(line)]
                if args.get('output_mode') == 'files_with_matches':
                    if hits:
                        output.append(rel)
                elif args.get('output_mode') == 'count':
                    if hits:
                        output.append(f'{rel}: {len(hits)}')
                else:
                    output += hits
            if args.get('head_limit'):
                output = output[:args['head_limit']]
            return '\n'.join(output)
        if name == 'Bash':
            if not self.allow_bash:
                raise ValueError('Bash requires the dedicated chroot worker')
            command = args['command']
            if any(x in command.casefold() for x in ('curl ', 'wget ', 'pip install', 'benchmark',
                    'agentbench', '/chat/completions', 'http://', 'https://')):
                raise ValueError('Only local pure checks are allowed')
            result = subprocess.run(['/bin/bash', '-c', command], cwd=self.workspace,
                capture_output=True, text=True, timeout=max(1, min(args.get('timeout', 120),
                    self.deadline - time.monotonic())), env={
                        'PATH': '/runtime/bin:/usr/bin:/bin', 'PYTHONPATH': str(self.workspace),
                        'PYTHONDONTWRITEBYTECODE': '1', 'TMPDIR': '/tmp'})
            return json.dumps({'exit_code': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr})
        if name == 'Agent':
            if role != 'parent' or len(state['delegations']) >= 2:
                raise ValueError('Exactly one analysis and one implementation Agent; no nested agents')
            child = ('analysis', 'implementation')[len(state['delegations'])]
            prompt = args['prompt'] + '\n\n' + self.request['port_contract']
            result = self.run(child, prompt)
            state['delegations'].append(child)
            return result
        raise ValueError('Unknown tool ' + name)

    def run(self, role, prompt):
        path = self.output / role / 'session.json'
        state = json.loads(path.read_text()) if path.exists() else {
            'messages': [{'role': 'system', 'content':
                'You are the coding-agent execution backend for the original Meta-Harness skill. '
                'Follow its workflow and the overriding native PORT CONTRACT. '
                'Use tools to inspect public training evidence and implement the requested candidate. '
                'All tool paths must stay in the public workspace. Local import checks use python. '
                'Do not run benchmarks, request additional LLMs yourself, or access another experiment.'},
                {'role': 'user', 'content': prompt}],
            'delegations': [], 'done': False, 'pending': None, 'cursor': 0, 'request_id': None}
        if state['done']:
            return state['result']
        tools = TOOLS if role == 'parent' else [t for t in TOOLS if t['function']['name'] != 'Agent']
        while time.monotonic() < self.deadline:
            if state['pending'] is None:
                if state['request_id'] is None:
                    state['request_id'] = uuid.uuid4().hex
                    write_json(path, state)
                response = self.completion(copy.deepcopy(state['messages']), tools, role, state['request_id'])
                choice = response['choices'][0]
                if choice.get('finish_reason') == 'length':
                    raise RuntimeError('DeepSeek output truncated at configured 60000 tokens')
                raw = choice['message']
                message = {k: raw[k] for k in ('role', 'content', 'tool_calls', 'reasoning_content') if k in raw}
                message['role'] = 'assistant'
                if message.get('tool_calls') and not message.get('reasoning_content'):
                    # Preserve a genuine empty reasoning continuation, as the
                    # shared Flash transport does; never fabricate thought text.
                    message['reasoning_content'] = ''
                state['messages'].append(message)
                state.update(pending=message, cursor=0, request_id=None)
                write_json(path, state)
            calls = state['pending'].get('tool_calls') or []
            if not calls:
                if role == 'parent' and state['delegations'] != ['analysis', 'implementation']:
                    raise RuntimeError('Upstream two-Agent workflow was not completed')
                state.update(done=True, result=state['pending'].get('content') or '')
                write_json(path, state)
                return state['result']
            for index in range(state['cursor'], len(calls)):
                call = calls[index]
                try:
                    value = self.dispatch(call['function']['name'], json.loads(call['function']['arguments']), state, role)
                except (ValueError, OSError, subprocess.SubprocessError, re.error, TypeError, KeyError) as exc:
                    value = json.dumps({'tool_error': f'{type(exc).__name__}: {exc}'})
                state['messages'].append({'role': 'tool', 'tool_call_id': call['id'], 'content': value})
                state['cursor'] = index + 1
                write_json(path, state)
            state.update(pending=None, cursor=0)
            write_json(path, state)
        raise TimeoutError('Original proposer wall-time limit exhausted')


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    if os.getuid() != request['isolated_uid'] or os.getuid() == 0:
        raise RuntimeError('Optimizer shell must run as its dedicated unprivileged chroot UID')
    runner = AgentRunner(request['workspace'], args.output, request, allow_bash=True)
    try:
        result = runner.run('parent', request['prompt'])
        write_json(args.output / 'result.json', {'success': True, 'text': result})
    except Exception as exc:
        write_json(args.output / 'result.json', {'success': False, 'error': f'{type(exc).__name__}: {exc}'})
        raise


if __name__ == '__main__':
    main()
