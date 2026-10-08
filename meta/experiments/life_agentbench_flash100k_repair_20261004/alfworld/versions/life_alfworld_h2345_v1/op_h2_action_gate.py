"""H2 ALFWorld action gate: verbatim-list check, H4 reroute substitution, veto, empty-turn nudge."""
from json import dumps, loads
from re import I, S, search

TARGET_LAYER = 'h2'

_AVAIL = r'available actions:\s*(.*)'
NUDGE = ('ALFWorld: an assistant turn without a take_action call performs nothing and is discarded. '
         'Reply with exactly one take_action call whose action string is copied verbatim from the '
         'most recent AVAILABLE ACTIONS list.')
SUGGEST = ' Suggested next action: '
VETO_HEAD = 'ALFWorld: the action '
VETO_MID = (' is not in the current AVAILABLE ACTIONS, so it would do nothing. '
            'Submit exactly one of these strings, copied verbatim: ')
PROGRESS_PREFIXES = ('take ', 'put ', 'clean ', 'use ', 'heat ', 'cool ')


def _norm(value):
    return ' '.join(str(value).lower().split())


def _allowed(history):
    latest = None
    if not isinstance(history, list):
        return None
    for message in history:
        if not isinstance(message, dict) or message.get('role') not in ('tool', 'user'):
            continue
        if not isinstance(message.get('content'), str):
            continue
        found = search(_AVAIL, message['content'], I | S)
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
        for prefix in PROGRESS_PREFIXES:
            if action.startswith(prefix):
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
        reroute = self._reroute()
        if isinstance(reroute, str) and _norm(reroute) in legal:
            _repair(calls[0], reroute)
            return message
        veto = VETO_HEAD + action + VETO_MID + ' | '.join(allowed[:8])
        message['_harness_control'] = {'action': 'suppress_tool',
                                       'messages': [{'role': 'user', 'content': veto}]}
        return message

    def h2_no_tool_message(self, message, remaining):
        suggestion = self._reroute()
        if suggestion is None:
            suggestion = _pick(self._allowed)
        if suggestion:
            return NUDGE + SUGGEST + suggestion + '.'
        return NUDGE

    def drain_h2_messages(self):
        return []

    def _reroute(self):
        if isinstance(self.state, dict):
            progress = self.state.get('alfworld_progress')
            if isinstance(progress, dict) and isinstance(progress.get('reroute'), str):
                return progress['reroute']
        return None
