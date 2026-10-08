"""H2 ALFWorld action gate: verbatim-list check, published-reroute repair, goal-ordered veto."""
from json import dumps, loads
from re import I, S, findall, search

TARGET_LAYER = 'h2'

_AVAIL = r'available actions:\s*(.*)'
_WORDS = r'[a-z0-9]+'
NUDGE = ('ALFWorld: an assistant turn without a take_action call performs nothing and is discarded. '
         'Reply with exactly one take_action call whose action string is copied verbatim from the '
         'most recent AVAILABLE ACTIONS list.')
SUGGEST = ' Suggested next action: '
VETO_HEAD = 'ALFWorld: the action '
VETO_MID = (' is not in the current AVAILABLE ACTIONS, so it would do nothing. '
            'Submit exactly one of these strings, copied verbatim: ')
VETO_TAIL = (' Keep to the task goal object and its goal receptacle; only those actions make '
             'progress.')
HAND = ('open ', 'take ', 'put ', 'cool ', 'heat ', 'clean ', 'use ')
MAX_LIST = 12


def _norm(value):
    return ' '.join(str(value).lower().split())


def _tokens(text):
    return findall(_WORDS, str(text).lower())


def _texts(value):
    out = []
    if isinstance(value, str) and value:
        out.append(value)
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, str) and item:
                out.append(item)
    return out


def _hit(action, names):
    if not names:
        return False
    tokens = _tokens(action)
    for name in names:
        if name in tokens:
            return True
    return False


def _score(action, obj, dest):
    if _hit(action, obj):
        return 0
    if _hit(action, dest):
        return 1
    low = _norm(action)
    for prefix in HAND:
        if low.startswith(prefix):
            return 2
    if low.startswith('go to '):
        return 3
    return 4


def _ordered(allowed, obj, dest):
    buckets = ([], [], [], [], [])
    for action in allowed:
        buckets[_score(action, obj, dest)].append(action)
    out = []
    for bucket in buckets:
        out.extend(bucket)
    return out


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


def _pick(allowed):
    for action in allowed:
        low = _norm(action)
        for prefix in HAND:
            if low.startswith(prefix):
                return action
    return None


class Harness:
    def __init__(self):
        self.state = {}
        self.context = None
        self._allowed = []

    def bind_shared_state(self, state):
        self.state = state

    def bind_task_context(self, context):
        self.context = context

    def h2(self, message, history):
        if not isinstance(message, dict):
            return message
        allowed = _allowed(history)
        self._allowed = list(allowed) if allowed else []
        calls = message.get('tool_calls')
        if not isinstance(calls, list) or not calls:
            return message
        action = _action_of(calls[0])
        if action is None or not allowed:
            return message
        legal = [_norm(item) for item in allowed]
        if _norm(action) in legal:
            return message
        reroute = self._published()
        if isinstance(reroute, str) and _norm(reroute) in legal:
            _repair(calls[0], reroute)
            return message
        obj, dest = self._goal_names()
        shown = _ordered(allowed, obj, dest)[:MAX_LIST]
        veto = VETO_HEAD + action + VETO_MID + ' | '.join(shown) + VETO_TAIL
        message['_harness_control'] = {'action': 'suppress_tool',
                                       'messages': [{'role': 'user', 'content': veto}]}
        return message

    def h2_no_tool_message(self, message, remaining):
        suggestion = self._published()
        if suggestion is None:
            suggestion = _pick(self._allowed)
        if suggestion:
            return NUDGE + SUGGEST + suggestion + '.'
        return NUDGE

    def drain_h2_messages(self):
        return []

    def _progress(self):
        if isinstance(self.state, dict) and isinstance(self.state.get('alfworld_progress'), dict):
            return self.state['alfworld_progress']
        return {}

    def _published(self):
        value = self._progress().get('reroute')
        if isinstance(value, str) and value:
            return value
        return None

    def _goal_names(self):
        progress = self._progress()
        obj = _texts(progress.get('goal_object_candidates'))
        if not obj:
            obj = _texts(progress.get('goal_object'))
        dest = _texts(progress.get('goal_destination_candidates'))
        if not dest:
            dest = _texts(progress.get('goal_destination'))
        return obj, dest
