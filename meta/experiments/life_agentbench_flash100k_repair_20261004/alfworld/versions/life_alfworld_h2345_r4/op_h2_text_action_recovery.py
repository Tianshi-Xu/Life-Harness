"""H2 ALFWorld recovery: a text-channel action becomes the real call; a proven no-effect repeat is rerouted."""
from json import dumps, loads
from re import I, S, findall, finditer, search

TARGET_LAYER = 'h2'

_AVAIL = r'available actions:\s*(.*)'
_WORDS = r'[a-z0-9]+'
_BLOCK = r'<tool_call>\s*(\{.+?\})\s*</tool_call>'
_FENCE = r'```[a-z]*\s*(\{.+?\})\s*```'
_OBJECT = r'\{[^{}]*\}'
_KEY_ACTION = r'"action"\s*:\s*"([^"]+)"'
_KEY_NAME = r'"name"\s*:\s*"([^"]+)"'
_STOP = ('the', 'a', 'an')
TOOL = 'take_action'
CALL_ID = 'h2-text-action'
MAX_TEXT = 120


def _norm(value):
    return ' '.join(str(value).lower().split())


def _tokens(text):
    return findall(_WORDS, str(text).lower())


def _allowed(history):
    latest = None
    if not isinstance(history, list):
        return None
    for message in history:
        if not isinstance(message, dict) or message.get('role') not in ('tool', 'user'):
            continue
        content = message.get('content')
        if not isinstance(content, str):
            continue
        found = search(_AVAIL, content, I | S)
        if found is None:
            continue
        actions = []
        for line in found.group(1).split('\n'):
            line = line.strip().rstrip('.')
            if line:
                actions.append(line)
        if actions:
            latest = actions
    return latest


def _action_of(call):
    if not isinstance(call, dict):
        return None
    function = call.get('function')
    if not isinstance(function, dict):
        return None
    raw = function.get('arguments')
    if not isinstance(raw, str):
        return None
    try:
        data = loads(raw)
    except ValueError:
        return None
    if isinstance(data, dict) and isinstance(data.get('action'), str):
        return data['action']
    return None


def _repair(call, action):
    function = call.get('function')
    if isinstance(function, dict) and isinstance(function.get('arguments'), str):
        function['arguments'] = dumps({'action': action})


def _quoted(text, pattern):
    out = []
    for item in finditer(pattern, text):
        value = item.group(1)
        if value and value not in out:
            out.append(value)
    return out


def _fields(blob):
    out = _quoted(blob, _KEY_ACTION)
    for value in _quoted(blob, _KEY_NAME):
        if value not in out:
            out.append(value)
    return out


def _candidates(content):
    out = []
    for item in finditer(_BLOCK, content):
        for value in _fields(item.group(1)):
            if value not in out:
                out.append(value)
    for item in finditer(_FENCE, content):
        for value in _fields(item.group(1)):
            if value not in out:
                out.append(value)
    if not out:
        for item in finditer(_OBJECT, content):
            for value in _fields(item.group(0)):
                if value not in out:
                    out.append(value)
    if not out:
        text = content.strip()
        if text and '\n' not in text and len(text) <= MAX_TEXT:
            out.append(text)
    return out


def _completion(action, allowed):
    words = [word for word in _tokens(action) if word not in _STOP]
    if not words:
        return None
    hits = []
    for item in allowed:
        item_words = _tokens(item)
        if not item_words or item_words[0] != words[0]:
            continue
        enough = True
        for word in words:
            if word not in item_words:
                enough = False
                break
        if not enough:
            continue
        if item not in hits:
            hits.append(item)
    if len(hits) == 1:
        return hits[0]
    return None


def _resolve(candidate, allowed):
    want = _norm(candidate)
    if not want:
        return None
    for item in allowed:
        if _norm(item) == want:
            return item
    return _completion(candidate, allowed)


def _last_action(history):
    if not isinstance(history, list):
        return None
    for message in reversed(history):
        if not isinstance(message, dict) or message.get('role') != 'assistant':
            continue
        calls = message.get('tool_calls')
        if isinstance(calls, list) and calls:
            return _action_of(calls[0])
    return None


class Harness:
    def __init__(self):
        self.state = {}
        self.context = None

    def bind_shared_state(self, state):
        self.state = state

    def bind_task_context(self, context):
        self.context = context

    def h2(self, message, history):
        if not isinstance(message, dict):
            return message
        allowed = _allowed(history)
        calls = message.get('tool_calls')
        if not isinstance(calls, list) or not calls:
            return self._from_text(message, allowed)
        action = _action_of(calls[0])
        if action is None or not allowed:
            return message
        legal = [_norm(item) for item in allowed]
        if _norm(action) not in legal:
            return message
        previous = _last_action(history)
        if not isinstance(previous, str) or _norm(previous) != _norm(action):
            return message
        if _norm(action) not in self._no_effect():
            return message
        published = self._published()
        if not isinstance(published, str) or not published:
            return message
        if _norm(published) == _norm(action) or _norm(published) not in legal:
            return message
        _repair(calls[0], published)
        self._record('noop_reroute', published)
        return message

    def _from_text(self, message, allowed):
        content = message.get('content')
        if not isinstance(content, str) or not content.strip() or not allowed:
            return message
        target = None
        for candidate in _candidates(content):
            found = _resolve(candidate, allowed)
            if found is not None:
                target = found
                break
        if target is None:
            return message
        message['tool_calls'] = [{'id': CALL_ID, 'type': 'function',
                                  'function': {'name': TOOL, 'arguments': dumps({'action': target})}}]
        self._record('text_action', target)
        return message

    def _no_effect(self):
        if not isinstance(self.state, dict):
            return []
        progress = self.state.get('alfworld_progress')
        if not isinstance(progress, dict):
            return []
        value = progress.get('no_effect')
        out = []
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    out.append(_norm(item))
        return out

    def _published(self):
        if not isinstance(self.state, dict):
            return None
        progress = self.state.get('alfworld_progress')
        if isinstance(progress, dict):
            value = progress.get('reroute')
            if isinstance(value, str) and value:
                return value
        decision = self.state.get('alfworld_decision')
        if isinstance(decision, dict):
            value = decision.get('action')
            if isinstance(value, str) and value:
                return value
        return None

    def _record(self, result, action):
        if isinstance(self.state, dict):
            self.state['alfworld_gate'] = {'result': result, 'action': action}
