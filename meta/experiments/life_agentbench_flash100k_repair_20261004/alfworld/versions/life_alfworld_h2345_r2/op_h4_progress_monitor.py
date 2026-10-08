"""H4 ALFWorld goal monitor: token-identity goal matching, ordered recipes, allowed-action search."""
from json import loads
from re import I, S, compile, findall, finditer, search

TARGET_LAYER = 'h4'

_TASK = compile(r'your task is to:?\s*([^\n]+)', I)
_AVAIL = compile(r'available actions:\s*(.*)', I | S)
_WORDS = compile(r'[a-z0-9]+')
_PICK = compile(r'you pick up the (.+?) from the (.+?)[.\n]', I)
_PUT = compile(r'you put the (.+?)(?: in/on | in | on )the (.+?)[.\n]', I)
_PROC = compile(r'you (cool|heat|clean) the (.+?) (?:using|with) the (.+?)[.\n]', I)
_CARRY = compile(r'you are carrying:?\s*([^\n]*)', I)
_SEE = compile(r'(?:on|in) the (.+?), you see', I)
_FACING = compile(r'you are facing the (.+?)[.,]', I)
_OPENED = compile(r'you open the (.+?)[.\n]', I)
_TAKE_A = compile(r'^take (.+?) from (.+)$', I)
_PUT_A = compile(r'^put (.+?) (?:in/on|in|on) (.+)$', I)
_PROC_A = compile(r'^(cool|heat|clean) (.+?) (?:using|with) (.+)$', I)
_GO_A = compile(r'^go to (.+)$', I)
_OPEN_A = compile(r'^open (.+)$', I)
_EXAMINE_A = compile(r'^examine (.+)$', I)
_USE_A = compile(r'^use (.+)$', I)
_VERB = compile(r'\b(cool|heat|clean)\b', I)
_PUTWORD = compile(r'\bput\b', I)

APPLIANCE = {'cool': 'fridge', 'heat': 'microwave', 'clean': 'sinkbasin'}
SKIP = ('a', 'an', 'the', 'some', 'in', 'on', 'to', 'under', 'with', 'and', 'of', 'it',
        'hot', 'cold', 'cool', 'cooled', 'heated', 'warm', 'clean', 'cleaned')
ADJECTIVE = ('hot', 'warm', 'cold', 'cool', 'cooled', 'heated', 'clean', 'cleaned')
PREFIX = '[alfworld-monitor] '
SUFFIX = '; next action (copy verbatim): '
HAND = ('open ', 'take ', 'put ', 'cool ', 'heat ', 'clean ', 'use ')
ARTICLE = ('a', 'an', 'the')


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


def _candidates(phrase):
    words = [word for word in _tokens(phrase) if word not in SKIP]
    out = []
    if words:
        out.append(words[0])
        if len(words) > 1:
            out.append(words[0] + words[1])
            out.append(words[1])
    return out


def _mentions(text, names):
    if not names:
        return False
    tokens = _tokens(text)
    for name in names:
        if name in tokens:
            return True
    return False


def _related(text, names):
    tokens = _tokens(text)
    for token in tokens:
        for name in names:
            if len(name) >= 4 and token.startswith(name):
                return True
            if len(token) >= 4 and name.startswith(token):
                return True
    return False


def _first_name(phrase):
    words = [word for word in _tokens(phrase) if word not in ARTICLE]
    if not words:
        return None
    if len(words) > 1 and words[1].isdigit():
        return words[0] + ' ' + words[1]
    return words[0]


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


def _goal(text):
    found = search(_TASK, text)
    if found is None:
        return None
    sentence = _norm(found.group(1))
    cut = sentence.find('available actions')
    if cut != -1:
        sentence = sentence[:cut]
    sentence = sentence.rstrip('. ')
    if not sentence:
        return None
    record = {'text': sentence, 'mode': 'place', 'object': None, 'destination': None,
              'object_candidates': [], 'destination_candidates': [], 'light': None}
    lowered = sentence
    if ('look at' in lowered or 'examine' in lowered) and 'lamp' in lowered:
        rest = lowered
        for marker in ('look at', 'examine'):
            if marker in rest:
                rest = rest.split(marker, 1)[1]
                break
        head = rest
        for marker in (' under ', ' with '):
            if marker in head:
                head = head.split(marker, 1)[0]
                break
        record['mode'] = 'look_at_light'
        record['object_candidates'] = _candidates(head)
        record['object'] = record['object_candidates'][0] if record['object_candidates'] else None
        record['light'] = 'desklamp'
        return record
    verb = search(_VERB, lowered)
    putword = search(_PUTWORD, lowered)
    mode = None
    if verb is not None and (putword is None or verb.start() < putword.start()):
        mode = verb.group(1)
        tail = lowered[verb.end():]
        for marker in (' and ', ' in ', ' on '):
            if marker in tail:
                tail = tail.split(marker, 1)[0]
                break
        record['object_candidates'] = _candidates(tail)
    elif putword is not None:
        tail = lowered[putword.end():]
        for marker in (' in ', ' on '):
            if marker in tail:
                tail = tail.split(marker, 1)[0]
                break
        words = _tokens(tail)
        for adjective in ADJECTIVE:
            if adjective in words:
                if adjective in ('hot', 'warm'):
                    mode = 'heat'
                elif adjective in ('clean', 'cleaned'):
                    mode = 'clean'
                else:
                    mode = 'cool'
                break
        record['object_candidates'] = _candidates(tail)
    record['mode'] = mode or 'place'
    record['object'] = record['object_candidates'][0] if record['object_candidates'] else None
    for marker in (' in ', ' on '):
        if marker in sentence:
            tail = sentence.split(marker, 1)[1].split(' and ')[0]
            found = _candidates(tail)
            if found:
                record['destination_candidates'] = found
                record['destination'] = found[0]
            break
    return record


def _allowed(messages):
    latest = None
    for message in messages:
        if not isinstance(message, dict) or message.get('role') not in ('tool', 'user'):
            continue
        content = message.get('content')
        if not isinstance(content, str):
            continue
        found = search(_AVAIL, content)
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


def _call_action(call):
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
            action = _call_action(call)
            if action:
                used.add(_norm(action))
    return used


def _held(tool_texts):
    held = None
    for content in tool_texts:
        found = search(_CARRY, content)
        if found is not None:
            held = _first_name(found.group(1))
        for item in finditer(_PICK, content):
            held = _norm(item.group(1))
        for item in finditer(_PUT, content):
            held = None
    return held


def _location(tool_texts):
    location = None
    for content in tool_texts:
        found = search(_OPENED, content)
        if found is not None:
            location = _norm(found.group(1))
        found = search(_SEE, content)
        if found is not None:
            location = _norm(found.group(1))
        found = search(_FACING, content)
        if found is not None:
            location = _norm(found.group(1))
    return location


def _light_on(tool_texts):
    for content in tool_texts:
        lowered = content.lower()
        if 'you turn on the' in lowered or 'desklamp 1 is on' in lowered:
            return True
    return False


def _processed(tool_texts, mode, names, appliance):
    for content in tool_texts:
        for found in finditer(_PROC, content):
            if found.group(1).lower() == mode and _mentions(found.group(2), names):
                if appliance in _norm(found.group(3)):
                    return True
    return False


def _placed(tool_texts, names, dest):
    for content in tool_texts:
        for found in finditer(_PUT, content):
            if _mentions(found.group(1), names) and _mentions(found.group(2), dest):
                return True
    return False


def _parts(action):
    text = _norm(action)
    found = search(_TAKE_A, text)
    if found is not None:
        return ('take', found.group(1), found.group(2))
    found = search(_PUT_A, text)
    if found is not None:
        return ('put', found.group(1), found.group(2))
    found = search(_PROC_A, text)
    if found is not None:
        return (found.group(1), found.group(2), found.group(3))
    found = search(_GO_A, text)
    if found is not None:
        return ('go', '', found.group(1))
    found = search(_OPEN_A, text)
    if found is not None:
        return ('open', '', found.group(1))
    found = search(_EXAMINE_A, text)
    if found is not None:
        return ('examine', '', found.group(1))
    found = search(_USE_A, text)
    if found is not None:
        return ('use', '', found.group(1))
    return (text, '', '')


def _take(allowed, names):
    for action in allowed:
        parts = _parts(action)
        if parts[0] == 'take' and _mentions(parts[1], names):
            return action
    return None


def _put(allowed, names, dest):
    if not dest:
        return None
    for action in allowed:
        parts = _parts(action)
        if parts[0] == 'put' and _mentions(parts[1], names) and _mentions(parts[2], dest):
            return action
    return None


def _process(allowed, names, appliance):
    for action in allowed:
        parts = _parts(action)
        if parts[0] in ('cool', 'heat', 'clean') and appliance in parts[2]:
            if _mentions(parts[1], names):
                return action
    return None


def _named(allowed, kinds, names):
    for action in allowed:
        parts = _parts(action)
        if parts[0] in kinds and _mentions(parts[2], names):
            return action
    return None


def _holding(allowed, held, names):
    if held and _mentions(held, names):
        return True
    for action in allowed:
        parts = _parts(action)
        if parts[0] == 'put' and _mentions(parts[1], names):
            return True
    return False


def _search(allowed, used, names, dest, location, held):
    note = ''
    if held:
        note = ' (you are holding ' + str(held) + ', which is not the goal object)'
    if location:
        loc_names = _candidates(location)
        action = _named(allowed, ('open',), loc_names)
        if action is not None and _norm(action) not in used:
            return action, 'open the container in front of you to look for the goal object' + note
    fallback = None
    for names_now in (names, dest):
        for action in allowed:
            parts = _parts(action)
            if parts[0] != 'go':
                continue
            if _norm(action) in used:
                continue
            if parts[2] and _related(parts[2], names_now):
                return action, 'go to the place named in the goal' + note
    for action in allowed:
        parts = _parts(action)
        if parts[0] != 'go' or _norm(action) in used:
            continue
        if fallback is None:
            fallback = action
    if fallback is not None:
        return fallback, 'the goal object is not here yet, so search other receptacles' + note
    return None, None


def _plan(record, allowed, tool_texts, used, held, location):
    names = _texts(record.get('object_candidates')) or _texts(record.get('object'))
    dest = _texts(record.get('destination_candidates')) or _texts(record.get('destination'))
    mode = record.get('mode') if isinstance(record.get('mode'), str) else 'place'
    if mode == 'look_at_light':
        light = _texts(record.get('light')) or ['desklamp']
        if not _light_on(tool_texts):
            action = _named(allowed, ('use',), light)
            if action is not None:
                return action, 'the lamp is still off and the goal object must be examined with it'
        if names:
            action = _take(allowed, names)
            if action is not None:
                return action, 'take the goal object so you can examine it under the lamp'
            if _light_on(tool_texts):
                action = _named(allowed, ('examine',), names)
                if action is not None:
                    return action, 'the lamp is on, so examine the goal object now'
        return None, None
    appliance = APPLIANCE.get(mode)
    placed = bool(names and dest and _placed(tool_texts, names, dest))
    if appliance:
        if _processed(tool_texts, mode, names, appliance):
            if not placed:
                action = _put(allowed, names, dest)
                if action is not None:
                    return action, 'the goal object is ready, so put it in the goal receptacle'
                action = _named(allowed, ('go',), dest)
                if action is not None:
                    return action, 'the goal object is ready, so go to the goal receptacle'
            return None, None
        if placed:
            action = _take(allowed, names)
            if action is not None:
                return action, ('the goal object is already in the receptacle but its ' + mode +
                                ' step is missing, so it does not count yet: take it out and ' +
                                mode + ' it')
            action = _named(allowed, ('go',), dest)
            if action is not None:
                return action, ('the goal object is in the receptacle without its ' + mode +
                                ' step, so go back for it and ' + mode + ' it')
            return None, ('the goal object is in the receptacle without its ' + mode +
                          ' step, so the task is not finished; do not leave it there')
    if names:
        holding = _holding(allowed, held, names)
        if not holding:
            if placed:
                return None, None
            action = _take(allowed, names)
            if action is not None:
                return action, 'the goal object is not in your inventory yet'
            return _search(allowed, used, names, dest, location, held)
        if appliance:
            action = _process(allowed, names, appliance)
            if action is not None:
                return action, 'the goal object still needs its ' + mode + ' step'
            action = _named(allowed, ('open',), [appliance])
            if action is not None:
                return action, 'the ' + appliance + ' must be open for the ' + mode + ' step'
            action = _named(allowed, ('go',), [appliance])
            if action is not None:
                return action, 'go to the ' + appliance + ' for the ' + mode + ' step'
            if _put(allowed, names, dest) is not None:
                return None, ('stop, the goal object still needs its ' + mode + ' step in the ' +
                              appliance + ', so placing it now would not count')
            return None, None
        action = _put(allowed, names, dest)
        if action is not None:
            return action, 'you hold the goal object and the goal receptacle is reachable'
        action = _named(allowed, ('go',), dest)
        if action is not None:
            return action, 'go to the goal receptacle and put the goal object there'
        return None, None
    return _search(allowed, used, names, dest, location, held)


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
        if record is not None and isinstance(self.state, dict):
            if not isinstance(self.state.get('alfworld_goal'), dict):
                self.state['alfworld_goal'] = record
        tool_texts = _role_contents(messages, 'tool')
        allowed = _allowed(messages)
        held = _held(tool_texts)
        location = _location(tool_texts)
        reroute = None
        reason = None
        if record is not None and allowed:
            reroute, reason = _plan(record, allowed, tool_texts, _used_actions(messages), held, location)
        hint = None
        if reroute is not None and reason is not None:
            hint = PREFIX + reason + SUFFIX + reroute
        elif reason is not None:
            hint = PREFIX + reason
        if hint is not None and isinstance(remaining, int) and not isinstance(remaining, bool):
            if remaining <= 3:
                hint = 'Only ' + str(remaining) + ' turns left. ' + hint
        self._store(record, held, location, allowed, reroute, reason)
        if hint is None or hint == self._last:
            return []
        self._last = hint
        return [hint]

    def _store(self, record, held, location, allowed, reroute, reason):
        if not isinstance(self.state, dict):
            return
        goal = record if isinstance(record, dict) else {}
        self.state['alfworld_progress'] = {
            'held': held,
            'location': location,
            'goal_object': goal.get('object'),
            'goal_object_candidates': _texts(goal.get('object_candidates')),
            'goal_destination': goal.get('destination'),
            'goal_destination_candidates': _texts(goal.get('destination_candidates')),
            'goal_mode': goal.get('mode'),
            'goal_text': goal.get('text'),
            'allowed': list(allowed) if allowed else [],
            'reroute': reroute,
            'reason': reason,
        }
