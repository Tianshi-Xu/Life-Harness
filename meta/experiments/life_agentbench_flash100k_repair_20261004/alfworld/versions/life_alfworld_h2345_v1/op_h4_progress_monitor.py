"""H4 ALFWorld progress monitor: public-goal reading, state tracking, one hint per turn."""
from json import loads
from re import I, S, compile, finditer, search, split

TARGET_LAYER = 'h4'

_TASK = compile(r'your task is to:?\s*([^\n]+)', I)
_AVAIL = compile(r'available actions:\s*(.*)', I | S)
_FACING = compile(r'you are facing the (.+?)[.,]', I)
_ON = compile(r'on the (.+?), you see', I)
_PICK = compile(r'you pick up the (.+?) from the', I)
_PUT = compile(r'you put the (.+?) in/on the (.+?)[.\n]', I)

NOISE = ('a', 'an', 'the', 'some', 'it', 'in', 'on', 'under', 'with', 'to', 'and', 'of',
         'hot', 'cool', 'cold', 'clean', 'cooled', 'heated', 'warm')
APPLIANCE = {'cool': 'fridge', 'heat': 'microwave', 'clean': 'sinkbasin'}
PREFIX = '[alfworld-monitor] '
SUFFIX = '; next action (copy verbatim): '
REASON_TAKE = 'the goal object is not in your inventory yet'
REASON_LIGHT = 'you are holding the goal object but the lamp is still off'
REASON_PROCESS = 'the goal object still needs its cool/heat/clean step'
REASON_PLACE = 'you are holding the goal object and the goal receptacle is reachable'
REASON_RECOVER = 'your recent actions changed nothing'
PLACED_NOTE = ('the goal object is already in the goal receptacle, so do not take it back; '
               'if the task has not ended, satisfy the remaining part of the goal')


def _norm(value):
    return ' '.join(str(value).lower().split())


def _joined(messages):
    parts = []
    for message in messages:
        if isinstance(message, dict) and isinstance(message.get('content'), str):
            parts.append(message['content'])
    return '\n'.join(parts)


def _role_contents(messages, role):
    contents = []
    for message in messages:
        if isinstance(message, dict) and message.get('role') == role:
            if isinstance(message.get('content'), str):
                contents.append(message['content'])
    return contents


def _head(phrase):
    words = [word for word in split(r'[^a-z0-9]+', str(phrase).lower()) if word]
    words = [word for word in words if word not in NOISE]
    return words[0] if words else None


def _goal(text):
    found = search(_TASK, text)
    if found is None:
        return None
    sentence = _norm(found.group(1))
    cut = sentence.find('available actions')
    if cut != -1:
        sentence = sentence[:cut]
    sentence = sentence.rstrip('. ')
    record = {'text': sentence, 'mode': 'place', 'object': None, 'destination': None, 'light': None}
    if 'look at' in sentence:
        record['mode'] = 'look_at_light'
        rest = sentence.split('look at', 1)[1]
        if ' under ' in rest:
            head, tail = rest.split(' under ', 1)
        elif ' with ' in rest:
            head, tail = rest.split(' with ', 1)
        else:
            head, tail = rest, 'desklamp'
        record['object'] = _head(head)
        record['light'] = _head(tail) or 'desklamp'
        return record
    if 'cool' in sentence:
        record['mode'] = 'cool'
    elif 'heat' in sentence or 'hot' in sentence:
        record['mode'] = 'heat'
    elif 'clean' in sentence:
        record['mode'] = 'clean'
    for marker in ('put some ', 'put a ', 'put the ', 'cool some ', 'cool a ',
                   'heat some ', 'heat a ', 'clean some ', 'clean the '):
        if marker in sentence:
            record['object'] = _head(sentence.split(marker, 1)[1])
            break
    for marker in (' in ', ' on '):
        if marker in sentence:
            record['destination'] = _head(sentence.split(marker, 1)[1].split(' and ')[0])
    return record


def _allowed(messages):
    latest = None
    for message in messages:
        if not isinstance(message, dict) or message.get('role') not in ('tool', 'user'):
            continue
        if not isinstance(message.get('content'), str):
            continue
        found = search(_AVAIL, message['content'])
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


def _used_actions(messages):
    used = set()
    for message in messages:
        if not isinstance(message, dict) or message.get('role') != 'assistant':
            continue
        calls = message.get('tool_calls')
        if not isinstance(calls, list):
            continue
        for call in calls:
            action = _action_of(call)
            if action:
                used.add(_norm(action))
    return used


def _held(tool_texts):
    held = None
    for content in tool_texts:
        low = content.lower()
        if low.startswith('you are carrying:'):
            tail = content.split(':', 1)[1].split('AVAILABLE ACTIONS')[0]
            if 'not carrying' in low:
                held = None
            else:
                chunk = tail.split(',')[0].strip().rstrip('.')
                words = [word for word in chunk.split() if word not in ('a', 'an', 'the')]
                held = ' '.join(words) if words else None
        elif low.startswith('you pick up the'):
            found = search(_PICK, content)
            if found:
                held = found.group(1).strip()
        elif low.startswith('you put the'):
            held = None
    return held


def _location(tool_texts):
    location = None
    for content in tool_texts:
        found = search(_ON, content)
        if found:
            location = found.group(1).strip().lower()
        found = search(_FACING, content)
        if found:
            location = found.group(1).split(',')[0].strip().lower()
    return location


def _lighted(tool_texts):
    for content in tool_texts:
        if 'you turn on the' in content.lower():
            return True
    return False


def _processed(tool_texts, record):
    obj = record.get('object')
    appliance = APPLIANCE.get(record.get('mode'))
    if not obj or not appliance:
        return False
    for content in tool_texts:
        low = content.lower()
        if obj in low and appliance in low:
            if low.startswith('you put the') or 'you clean the' in low:
                return True
    return False


def _placed(tool_texts, record):
    obj = record.get('object')
    dest = record.get('destination')
    if not obj or not dest:
        return False
    for content in tool_texts:
        for found in finditer(_PUT, content):
            if obj in found.group(1).lower() and dest in found.group(2).lower():
                return True
    return False


def _stall(tool_texts):
    keys = [_norm(content) for content in tool_texts]
    count = 0
    for index in range(len(keys) - 1, -1, -1):
        key = keys[index]
        if key and key in keys[:index]:
            count += 1
        else:
            break
    return count


def _first_take(allowed, obj):
    for action in allowed:
        if action.startswith('take ') and obj in action.lower():
            return action
    return None


def _first_prefix(allowed, prefixes, token):
    for action in allowed:
        low = action.lower()
        if token and token in low:
            for prefix in prefixes:
                if low.startswith(prefix):
                    return action
    return None


def _first_held(allowed, prefixes, held_text, token):
    for action in allowed:
        low = action.lower()
        if not held_text or held_text not in low:
            continue
        if token and token in low:
            for prefix in prefixes:
                if low.startswith(prefix):
                    return action
    return None


def _choose(allowed, record, held, tool_texts):
    obj = record.get('object')
    dest = record.get('destination')
    mode = record.get('mode')
    held_text = held.lower() if isinstance(held, str) else ''
    has_object = bool(obj) and obj in held_text
    if obj and not has_object:
        found = _first_take(allowed, obj)
        if found:
            return found, REASON_TAKE
    if mode == 'look_at_light' and has_object and not _lighted(tool_texts):
        found = _first_prefix(allowed, ('use ',), record.get('light') or 'desklamp')
        if found:
            return found, REASON_LIGHT
    if obj and has_object and mode in APPLIANCE and not _processed(tool_texts, record):
        found = _first_held(allowed, ('put ', 'clean '), held_text, APPLIANCE[mode])
        if found:
            return found, REASON_PROCESS
    if obj and has_object and dest:
        found = _first_held(allowed, ('put ',), held_text, dest)
        if found:
            return found, REASON_PLACE
        found = _first_prefix(allowed, ('go to ',), dest)
        if found:
            return found, REASON_PLACE
    return None, None


def _pick_unused(allowed, used):
    fresh = []
    for action in allowed:
        low = _norm(action)
        if low in used or low in ('look', 'inventory'):
            continue
        fresh.append(action)
    for prefix in ('go to ', 'open ', 'examine ', 'take ', 'clean ', 'use '):
        for action in fresh:
            if action.startswith(prefix):
                return action
    return fresh[0] if fresh else None


class Harness:
    def __init__(self):
        self.state = {}
        self.context = None
        self._last = None

    def bind_shared_state(self, state):
        self.state = state

    def bind_task_context(self, context):
        self.context = context

    def h4(self, messages, remaining):
        if not isinstance(messages, list) or not messages:
            return []
        record = _goal(_joined(messages))
        if record is None and isinstance(self.state, dict):
            stored = self.state.get('alfworld_goal')
            if isinstance(stored, dict):
                record = stored
        tool_texts = _role_contents(messages, 'tool')
        allowed = _allowed(messages)
        held = _held(tool_texts)
        location = _location(tool_texts)
        placed = False
        if record is not None and _placed(tool_texts, record):
            obj = record.get('object')
            placed = not (obj and isinstance(held, str) and obj in held.lower())
        stall = _stall(tool_texts)
        reroute = None
        reason = None
        if record is not None and allowed:
            if placed:
                if stall >= 1:
                    reason = PLACED_NOTE
            else:
                reroute, reason = _choose(allowed, record, held, tool_texts)
                if reroute is None and stall >= 2:
                    reroute = _pick_unused(allowed, _used_actions(messages))
                    if reroute is not None:
                        reason = REASON_RECOVER
        hint = None
        if reroute is not None and reason is not None:
            hint = PREFIX + reason + SUFFIX + reroute
        elif reason is not None and record is not None:
            hint = PREFIX + reason + ': ' + str(record.get('text', ''))
        if hint is not None and isinstance(remaining, int) and not isinstance(remaining, bool):
            if remaining <= 3:
                hint = 'Only ' + str(remaining) + ' turns left. ' + hint
        self._store(held, location, placed, stall, allowed, reroute, reason)
        if hint is None or hint == self._last:
            return []
        self._last = hint
        return [hint]

    def _store(self, held, location, placed, stall, allowed, reroute, reason):
        if not isinstance(self.state, dict):
            return
        self.state['alfworld_progress'] = {
            'held': held,
            'location': location,
            'placed': placed,
            'stall': stall,
            'allowed': list(allowed) if allowed else [],
            'reroute': reroute,
            'reason': reason,
        }
