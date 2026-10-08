"""H4 ALFWorld monitor: counted-goal program, exact-referent steps, no-effect stall guard."""
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
_STATE = compile(r'(?:^|\. |\n)(?:the )?([a-z][a-z0-9 ]*?) is (closed|open)\b', I)
_TAKE_A = compile(r'^take (.+?) from (.+)$', I)
_PUT_A = compile(r'^put (.+?) (?:in/on|in|on) (.+)$', I)
_PROC_A = compile(r'^(cool|heat|clean) (.+?) (?:using|with) (.+)$', I)
_GO_A = compile(r'^go to (.+)$', I)
_OPEN_A = compile(r'^open (.+)$', I)
_EXAMINE_A = compile(r'^examine (.+)$', I)
_USE_A = compile(r'^use (.+)$', I)
_VERB = compile(r'\b(cool|heat|clean)\b', I)
_PUTWORD = compile(r'\bput\b', I)
_QTY_AFTER = compile(r'\b(?:put|place|pick up|pick)\s+(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\b', I)
_LEAD_QTY = compile(r'^(?:one|two|three|four|five|six|seven|eight|nine|ten)\s+', I)
_PICKUP = compile(r'pick up\s+(.+?)(?:\s+and\s+|\s*$)', I)
_LAMP_WORD = compile(r'([a-z]*lamp)', I)
_LAMP_ON = compile(r'you turn on the (?:the )?([a-z0-9 ]*lamp[a-z0-9 ]*?)[.\n]', I)
_LAMP_STATE = compile(r'the ([a-z0-9 ]*lamp[a-z0-9 ]*?) is on\b', I)
_NOINFO = compile(r"nothing happens|there'?s nothing special about|next to it, you see nothing", I)

QTY = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6,
       'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10}
APPLIANCE = {'cool': 'fridge', 'heat': 'microwave', 'clean': 'sinkbasin'}
SKIP = ('a', 'an', 'the', 'some', 'in', 'on', 'to', 'under', 'with', 'and', 'of', 'it',
        'them', 'both', 'all', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight',
        'nine', 'ten', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
        'hot', 'cold', 'cool', 'cooled', 'heated', 'warm', 'clean', 'cleaned')
ADJECTIVE = ('hot', 'warm', 'cold', 'cool', 'cooled', 'heated', 'clean', 'cleaned')
PREFIX = '[alfworld-monitor] '
SUFFIX = '; next action (copy verbatim): '
STALL = 'choose a genuinely different action from the newest AVAILABLE ACTIONS list'
HAND = ('open ', 'take ', 'put ', 'cool ', 'heat ', 'clean ', 'use ')
ARTICLE = ('a', 'an', 'the')
MAX_STUCK = 4


def _norm(value):
    return ' '.join(str(value).lower().split())


def _tokens(text):
    return findall(_WORDS, str(text).lower())


def _contains(text, want):
    """Whole-token containment, so 'cabinet 1' never matches inside 'cabinet 10'."""
    if not want:
        return False
    if text == want:
        return True
    if text.startswith(want + ' '):
        return True
    if text.endswith(' ' + want):
        return True
    if (' ' + want + ' ') in text:
        return True
    return False


def _texts(value):
    out = []
    if isinstance(value, str) and value:
        out.append(value)
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, str) and item:
                out.append(item)
    return out


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


def _quantity(text):
    found = search(_QTY_AFTER, str(text))
    if found is None:
        return 1
    raw = found.group(1).lower()
    if raw in QTY:
        return QTY[raw]
    if raw.isdigit():
        value = int(raw)
        if 1 <= value <= 10:
            return value
    return 1


def _strip_quantity(phrase):
    text = _norm(phrase)
    found = search(_LEAD_QTY, text)
    if found is not None:
        text = text[found.end():]
    return text


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


def _number(phrase):
    for token in _tokens(phrase):
        if token.isdigit():
            return token
        if token in QTY:
            return str(QTY[token])
    return None


def _match(target, phrase):
    if not target or not phrase:
        return False
    want = _norm(phrase)
    text = _norm(target)
    if not want:
        return False
    if _contains(text, want):
        return True
    if not _mentions(target, _candidates(want)):
        return False
    number = _number(want)
    if number is not None and number not in _tokens(target):
        return False
    return True


def _score(target, phrase):
    if not target or not phrase:
        return None
    want = _norm(phrase)
    text = _norm(target)
    if _contains(text, want):
        return 0
    if not _match(target, phrase):
        return None
    if _number(phrase) is not None:
        return 1
    return 2


def _first_name(phrase):
    words = [word for word in _tokens(phrase) if word not in ARTICLE]
    if not words:
        return None
    if len(words) > 1 and words[1].isdigit():
        return words[0] + ' ' + words[1]
    return words[0]


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


def _stuck_actions(messages):
    stuck = []
    seen = {}
    pending = None
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get('role') == 'assistant':
            pending = None
            calls = message.get('tool_calls')
            if isinstance(calls, list) and calls:
                action = _call_action(calls[0])
                if action:
                    pending = _norm(action)
            continue
        if message.get('role') != 'tool' or pending is None:
            continue
        content = message.get('content')
        action = pending
        pending = None
        if not isinstance(content, str):
            continue
        if search(_NOINFO, content) is not None or seen.get(action) == content:
            if action not in stuck:
                stuck.append(action)
        seen[action] = content
    return stuck


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
    states = {}
    order = []
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
        for item in finditer(_STATE, content):
            place = _norm(item.group(1))
            states[place] = item.group(2).lower()
            order.append(place)
    if order:
        location = order[-1]
    return location, states


def _closed_place(states, phrase):
    if not phrase:
        return False
    for place in states:
        if states[place] == 'closed' and _match(place, phrase):
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


def _find(allowed, kinds, slot, phrase, used, stuck):
    best = None
    best_key = None
    for action in allowed:
        parts = _parts(action)
        if parts[0] not in kinds:
            continue
        score = _score(parts[slot], phrase)
        if score is None:
            continue
        text = _norm(action)
        key = (1 if text in stuck else 0, score, 1 if text in used else 0)
        if best_key is None or key < best_key:
            best_key = key
            best = action
    if best is None or _norm(best) in stuck:
        return None
    return best


def _object_step(allowed, phrase, used, stuck):
    return _find(allowed, ('take',), 1, phrase, used, stuck)


def _dest_step(allowed, kinds, phrase, used, stuck):
    return _find(allowed, kinds, 2, phrase, used, stuck)


def _put_step(allowed, obj_phrase, dest_phrase, used, stuck):
    best = None
    best_key = None
    for action in allowed:
        parts = _parts(action)
        if parts[0] != 'put' or not _match(parts[1], obj_phrase):
            continue
        score = _score(parts[2], dest_phrase)
        if score is None:
            continue
        text = _norm(action)
        key = (1 if text in stuck else 0, score, 1 if text in used else 0)
        if best_key is None or key < best_key:
            best_key = key
            best = action
    if best is None or _norm(best) in stuck:
        return None
    return best


def _proc_step(allowed, obj_phrase, mode, appliance, used, stuck):
    for action in allowed:
        parts = _parts(action)
        if parts[0] != mode or appliance not in _norm(parts[2]):
            continue
        if not _match(parts[1], obj_phrase):
            continue
        if _norm(action) in stuck:
            continue
        return action
    return None


def _holding(allowed, held, phrase):
    if held and _match(held, phrase):
        return True
    for action in allowed:
        parts = _parts(action)
        if parts[0] == 'put' and _match(parts[1], phrase):
            return True
    return False


def _placed_objects(tool_texts, obj_phrase, dest_phrase):
    placed = []
    for content in tool_texts:
        for item in finditer(_PUT, content):
            obj = _norm(item.group(1))
            if not _match(obj, obj_phrase) or not _match(item.group(2), dest_phrase):
                continue
            if obj not in placed:
                placed.append(obj)
    return placed


def _processed(tool_texts, mode, obj_phrase, appliance):
    for content in tool_texts:
        for item in finditer(_PROC, content):
            if item.group(1).lower() != mode:
                continue
            if not _match(item.group(2), obj_phrase):
                continue
            if appliance in _norm(item.group(3)):
                return True
    return False


def _lamp_on(tool_texts, lamp):
    for content in tool_texts:
        for item in finditer(_LAMP_ON, content):
            if lamp in _norm(item.group(1)):
                return True
        for item in finditer(_LAMP_STATE, content):
            if lamp in _norm(item.group(1)):
                return True
    return False


def _stall_note(stuck):
    return 'nothing new comes back from ' + ', '.join(stuck[:MAX_STUCK]) + '; ' + STALL


def _search_plan(allowed, used, stuck, obj_phrase, dest_phrase, location, held):
    note = ''
    if held:
        note = ' (you are holding ' + str(held) + ', which is not the goal object)'
    if location:
        action = _dest_step(allowed, ('open',), location, used, stuck)
        if action is not None:
            return action, 'open the container in front of you to look for the goal object' + note, 'search'
    for phrase in (obj_phrase, dest_phrase):
        if not phrase:
            continue
        action = _dest_step(allowed, ('go',), phrase, used, stuck)
        if action is not None:
            return action, 'go to the place named in the goal' + note, 'search'
    for action in allowed:
        parts = _parts(action)
        if parts[0] != 'go':
            continue
        text = _norm(action)
        if text in used or text in stuck:
            continue
        return action, 'the goal object is not here yet, so search other receptacles' + note, 'search'
    if stuck:
        return None, _stall_note(stuck), 'stall'
    return None, None, 'idle'


def _lamp_search(allowed, used, stuck, location):
    if location:
        action = _dest_step(allowed, ('examine',), location, used, stuck)
        if action is not None:
            return action, 'examine the receptacle in front of you to look for the lamp', 'lamp-search'
    for action in allowed:
        parts = _parts(action)
        if parts[0] != 'go':
            continue
        text = _norm(action)
        if text in used or text in stuck:
            continue
        return action, 'the lamp has not been found yet, so survey the next place', 'lamp-search'
    if stuck:
        return None, _stall_note(stuck), 'stall'
    return None, ('the lamp has not been turned on yet and no new place is reachable; look for a '
                  'lamp action in the newest AVAILABLE ACTIONS list'), 'warn'


def _light_plan(record, allowed, tool_texts, used, stuck, held, location, obj_phrase):
    lamp = _norm(record.get('light') or 'desklamp') or 'desklamp'
    lamp_on = _lamp_on(tool_texts, lamp)
    holding = _holding(allowed, held, obj_phrase) if obj_phrase else False
    if obj_phrase and not holding:
        action = _object_step(allowed, obj_phrase, used, stuck)
        if action is not None:
            return action, 'take the goal object so you can look at it in the light of the ' + lamp, 'take', 0
        result = _search_plan(allowed, used, stuck, obj_phrase, obj_phrase, location, held)
        return result[0], result[1], result[2], 0
    if not lamp_on:
        action = _dest_step(allowed, ('use',), lamp, used, stuck)
        if action is not None:
            return action, ('the ' + lamp + ' is still off, and turning it on is the step this task '
                            'needs'), 'lamp', 0
        result = _lamp_search(allowed, used, stuck, location)
        return result[0], result[1], result[2], 0
    action = _find(allowed, ('examine',), 2, obj_phrase, used, stuck) if obj_phrase else None
    if action is not None:
        return action, 'the ' + lamp + ' is on, so look at the goal object now', 'examine', 0
    return None, None, 'idle', 0


def _plan(record, allowed, tool_texts, used, stuck, held, location, states):
    obj_phrase = _norm(record.get('object_phrase') or record.get('object') or '')
    dest_phrase = _norm(record.get('destination_phrase') or record.get('destination') or '')
    mode = record.get('mode') if isinstance(record.get('mode'), str) else 'place'
    quantity = record.get('quantity') if isinstance(record.get('quantity'), int) else 1
    if quantity < 1:
        quantity = 1
    if mode == 'look_at_light':
        return _light_plan(record, allowed, tool_texts, used, stuck, held, location, obj_phrase)
    placed = _placed_objects(tool_texts, obj_phrase, dest_phrase) if (obj_phrase and dest_phrase) else []
    count = len(placed)
    appliance = APPLIANCE.get(mode)
    if count >= quantity:
        return None, None, 'done', count
    if appliance:
        if _processed(tool_texts, mode, obj_phrase, appliance):
            action = _put_step(allowed, obj_phrase, dest_phrase, used, stuck)
            if action is not None:
                return action, 'the goal object is ready, so put it in the goal receptacle', 'put', count
            action = _dest_step(allowed, ('go',), dest_phrase, used, stuck)
            if action is not None:
                return action, 'the goal object is ready, so go to the goal receptacle', 'go', count
        if placed:
            action = _object_step(allowed, obj_phrase, used, stuck)
            if action is not None:
                return action, ('the goal object is already in the receptacle but its ' + mode +
                                ' step is missing, so it does not count yet: take it out and ' + mode +
                                ' it'), 'recover', count
            action = _dest_step(allowed, ('go',), dest_phrase, used, stuck)
            if action is not None:
                return action, ('the goal object is in the receptacle without its ' + mode +
                                ' step, so go back for it and ' + mode + ' it'), 'recover', count
            return None, ('the goal object is in the receptacle without its ' + mode +
                          ' step, so the task is not finished; do not leave it there'), 'warn', count
        if not _holding(allowed, held, obj_phrase):
            action = _object_step(allowed, obj_phrase, used, stuck)
            if action is not None:
                return action, 'the goal object is not in your inventory yet', 'take', count
            result = _search_plan(allowed, used, stuck, obj_phrase, dest_phrase, location, held)
            return result[0], result[1], result[2], count
        action = _proc_step(allowed, obj_phrase, mode, appliance, used, stuck)
        if action is not None:
            return action, 'the goal object still needs its ' + mode + ' step', 'process', count
        action = _dest_step(allowed, ('open',), appliance, used, stuck)
        if action is not None:
            return action, 'the ' + appliance + ' must be open for the ' + mode + ' step', 'open', count
        action = _dest_step(allowed, ('go',), appliance, used, stuck)
        if action is not None:
            return action, 'go to the ' + appliance + ' for the ' + mode + ' step', 'go', count
        if _put_step(allowed, obj_phrase, dest_phrase, used, stuck) is not None:
            return None, ('stop, the goal object still needs its ' + mode + ' step in the ' + appliance +
                          ', so placing it now would not count'), 'warn', count
        return None, None, 'idle', count
    if _holding(allowed, held, obj_phrase):
        action = _put_step(allowed, obj_phrase, dest_phrase, used, stuck)
        if action is not None:
            return action, 'you hold the goal object and the goal receptacle is reachable', 'put', count
        if _closed_place(states, dest_phrase):
            action = _dest_step(allowed, ('open',), dest_phrase, used, stuck)
            if action is not None:
                return action, ('the goal receptacle is closed, so open it before putting the goal '
                                'object in'), 'open', count
        action = _dest_step(allowed, ('go',), dest_phrase, used, stuck)
        if action is not None:
            return action, 'go to the goal receptacle and put the goal object there', 'go', count
        return None, None, 'idle', count
    if quantity == 1 and placed:
        return None, None, 'done', count
    action = _object_step(allowed, obj_phrase, used, stuck) if obj_phrase else None
    if action is not None:
        if quantity > 1:
            if count:
                reason = (str(count) + ' of ' + str(quantity) + ' objects of type ' + obj_phrase +
                          ' are placed; take another one')
            else:
                reason = ('this goal needs ' + str(quantity) + ' objects of type ' + obj_phrase +
                          ' and you hold none yet')
            return action, reason, 'take', count
        return action, 'the goal object is not in your inventory yet', 'take', count
    result = _search_plan(allowed, used, stuck, obj_phrase, dest_phrase, location, held)
    return result[0], result[1], result[2], count


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
    record = {'text': sentence, 'mode': 'place', 'object': None, 'object_phrase': None,
              'destination': None, 'destination_phrase': None, 'light': None,
              'quantity': _quantity(sentence), 'object_candidates': [],
              'destination_candidates': []}
    lamp = search(_LAMP_WORD, sentence)
    if ('look at' in sentence or 'examine' in sentence) and 'lamp' in sentence:
        rest = sentence
        for marker in ('look at', 'examine'):
            if marker in rest:
                rest = rest.split(marker, 1)[1]
                break
        head = rest
        for marker in (' under ', ' with '):
            if marker in head:
                head = head.split(marker, 1)[0]
                break
        head = _strip_quantity(head)
        record['mode'] = 'look_at_light'
        record['object_candidates'] = _candidates(head)
        record['object'] = record['object_candidates'][0] if record['object_candidates'] else None
        record['object_phrase'] = head if head else None
        record['light'] = lamp.group(1).lower() if lamp is not None else 'desklamp'
        return record
    verb = search(_VERB, sentence)
    putword = search(_PUTWORD, sentence)
    mode = None
    obj_phrase = None
    if verb is not None and (putword is None or verb.start() < putword.start()):
        mode = verb.group(1)
        tail = sentence[verb.end():]
        for marker in (' and ', ' in ', ' on '):
            if marker in tail:
                tail = tail.split(marker, 1)[0]
                break
        obj_phrase = _strip_quantity(tail)
    elif putword is not None:
        tail = sentence[putword.end():]
        for marker in (' in ', ' on '):
            if marker in tail:
                tail = tail.split(marker, 1)[0]
                break
        obj_phrase = _strip_quantity(tail)
        words = _tokens(obj_phrase)
        for adjective in ADJECTIVE:
            if adjective in words:
                if adjective in ('hot', 'warm'):
                    mode = 'heat'
                elif adjective in ('clean', 'cleaned'):
                    mode = 'clean'
                else:
                    mode = 'cool'
                break
    if obj_phrase is None or not _candidates(obj_phrase):
        pickup = search(_PICKUP, sentence)
        if pickup is not None:
            obj_phrase = _strip_quantity(pickup.group(1))
    record['mode'] = mode or 'place'
    record['object_phrase'] = _norm(obj_phrase) if obj_phrase else None
    record['object_candidates'] = _candidates(obj_phrase) if obj_phrase else []
    record['object'] = record['object_candidates'][0] if record['object_candidates'] else None
    for marker in (' in ', ' on '):
        if marker in sentence:
            tail = sentence.split(marker, 1)[1].split(' and ')[0]
            tail = _strip_quantity(tail)
            found = _candidates(tail)
            if found:
                record['destination_phrase'] = _norm(tail)
                record['destination_candidates'] = found
                record['destination'] = found[0]
            break
    return record


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
        used = _used_actions(messages)
        stuck = _stuck_actions(messages)
        if isinstance(self.state, dict):
            gate = self.state.get('alfworld_gate')
            if isinstance(gate, dict) and gate.get('result') == 'veto':
                value = gate.get('action')
                if isinstance(value, str) and value and _norm(value) not in stuck:
                    stuck.append(_norm(value))
        held = _held(tool_texts)
        location, states = _location(tool_texts)
        reroute = None
        reason = None
        stage = 'idle'
        count = 0
        if record is not None and allowed:
            reroute, reason, stage, count = _plan(record, allowed, tool_texts, used, stuck, held,
                                                  location, states)
        if reroute is None and reason is None and stuck:
            reason = ('your recent actions returned no new information (' + ', '.join(stuck[:MAX_STUCK]) +
                      '); ' + STALL)
            stage = 'stall'
        quantity = 1
        if isinstance(record, dict) and isinstance(record.get('quantity'), int):
            quantity = record.get('quantity')
        hint = None
        if reroute is not None and reason is not None:
            hint = PREFIX + reason + SUFFIX + reroute
        elif reason is not None:
            hint = PREFIX + reason
        if hint is not None and isinstance(remaining, int) and not isinstance(remaining, bool):
            if remaining <= 3:
                hint = 'Only ' + str(remaining) + ' turns left. ' + hint
        self._store(record, held, location, allowed, reroute, reason, count, stuck, quantity, stage)
        if hint is None or hint == self._last:
            return []
        self._last = hint
        return [hint]

    def _store(self, record, held, location, allowed, reroute, reason, count, stuck, quantity, stage):
        if not isinstance(self.state, dict):
            return
        goal = record if isinstance(record, dict) else {}
        self.state['alfworld_progress'] = {
            'held': held,
            'location': location,
            'goal_object': goal.get('object'),
            'goal_object_candidates': _texts(goal.get('object_candidates')),
            'goal_object_phrase': goal.get('object_phrase'),
            'goal_destination': goal.get('destination'),
            'goal_destination_candidates': _texts(goal.get('destination_candidates')),
            'goal_destination_phrase': goal.get('destination_phrase'),
            'goal_mode': goal.get('mode'),
            'goal_text': goal.get('text'),
            'quantity': quantity,
            'placed_count': count,
            'allowed': list(allowed) if allowed else [],
            'reroute': reroute,
            'reason': reason,
        }
        self.state['alfworld_decision'] = {
            'action': reroute,
            'reason': reason,
            'mode': goal.get('mode'),
            'quantity': quantity,
            'placed_count': count,
            'stuck': list(stuck),
            'stage': stage,
        }
