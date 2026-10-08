"""H5 ALFWorld cold-start skill: counted-goal and lamp programs plus the parsed public focus."""
from re import I, compile, findall, search

TARGET_LAYER = 'h5'

_TASK = compile(r'your task is to:?\s*([^\n]+)', I)
_WORDS = compile(r'[a-z0-9]+')
_VERB = compile(r'\b(cool|heat|clean)\b', I)
_PUTWORD = compile(r'\bput\b', I)
_QTY_AFTER = compile(r'\b(?:put|place|pick up|pick)\s+(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\b', I)
_LEAD_QTY = compile(r'^(?:one|two|three|four|five|six|seven|eight|nine|ten)\s+', I)
_PICKUP = compile(r'pick up\s+(.+?)(?:\s+and\s+|\s*$)', I)
_LAMP_WORD = compile(r'([a-z]*lamp)', I)
MARKERS = ('AVAILABLE ACTIONS', 'Interact with a household')
QTY = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6,
       'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10}
SKIP = ('a', 'an', 'the', 'some', 'in', 'on', 'to', 'under', 'with', 'and', 'of', 'it',
        'them', 'both', 'all', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight',
        'nine', 'ten', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
        'hot', 'cold', 'cool', 'cooled', 'heated', 'warm', 'clean', 'cleaned')
ADJECTIVE = ('hot', 'warm', 'cold', 'cool', 'cooled', 'heated', 'clean', 'cleaned')
SKILL = (
    'ALFWorld procedure (apply before acting):\n'
    '1. Every turn is exactly one take_action call, and its action string must be copied verbatim '
    'from the AVAILABLE ACTIONS list of the newest tool message; anything else changes nothing.\n'
    '2. The task names one goal object type and one goal receptacle type. Only the named object type '
    'counts: taking, cooling, heating, cleaning or placing a different object is not progress, and an '
    'object that was never cooled/heated/cleaned does not finish the task even inside the goal '
    'receptacle.\n'
    '3. Placement goals (put some X in/on Y): take an X, then put X in/on Y; open Y first if it is '
    'closed.\n'
    '4. Counted goals (put two X in Y): that many different X objects must end up in Y, so after the '
    'first one is placed you must find and place a second one; putting the same object in twice does '
    'not count.\n'
    '5. Process goals (cool/heat/clean some X and put it in Y): take X, go to fridge 1 for cool, '
    'microwave 1 for heat or sinkbasin 1 for clean, open it if it is closed, run the cool/heat/clean '
    'action on X, then go to Y and put X there; never place X in Y before that step.\n'
    '6. Light goals (look at or examine X with the desklamp or under it): take the goal object '
    'first, then go to the lamp named in the task and use it once (use desklamp 1) while that '
    'object is still in your inventory; if no lamp action is listed, move to another receptacle '
    'until the lamp is reachable; never put the goal object down before the lamp is on, and '
    'examining the goal object itself does not finish this goal.\n'
    '7. Survey discipline: if the goal object is not in the current room, move to another '
    'receptacle, open it if it is closed, and never repeat an action whose observation did not '
    'change; when a repeated action returns the same text, choose a different action from the '
    'newest list. Search first the places where that kind of object is usually kept - countertops, '
    'the fridge, the diningtable or the microwave for food; countertops, shelves, cabinets or the '
    'sinkbasin for bottles and cleaning supplies; sidetables, shelves or drawers for small items - '
    'and if an earlier observation showed the goal object somewhere, go back to that place instead '
    'of opening a new one.\n'
    '8. Never end a turn with plain text; an assistant turn without a tool call is discarded.'
)


def _joined(messages):
    parts = []
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict) and isinstance(message.get('content'), str):
                parts.append(message['content'])
    return '\n'.join(parts)


def _norm(value):
    return ' '.join(str(value).lower().split())


def _tokens(text):
    return findall(_WORDS, str(text).lower())


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


def _focus(record):
    if not isinstance(record, dict):
        return ''
    parts = []
    if record.get('object'):
        parts.append('goal object: ' + str(record['object']))
    if record.get('destination'):
        parts.append('goal receptacle: ' + str(record['destination']))
    if record.get('mode') and record.get('mode') != 'place':
        parts.append('required step: ' + str(record['mode']))
    quantity = record.get('quantity')
    if isinstance(quantity, int) and quantity > 1:
        parts.append('objects required: ' + str(quantity))
    if not parts:
        return ''
    return '\nGoal focus from the public task text - ' + '; '.join(parts) + '.'


class Harness:
    def __init__(self):
        self.state = {}
        self.context = None
        self._emitted = False

    def bind_shared_state(self, state):
        self.state = state

    def bind_task_context(self, context):
        self.context = context

    def h5(self, messages):
        if not isinstance(messages, list) or not messages:
            return []
        text = _joined(messages)
        if not any(marker in text for marker in MARKERS):
            return []
        record = _goal(text)
        if record is not None and isinstance(self.state, dict):
            if not isinstance(self.state.get('alfworld_goal'), dict):
                self.state['alfworld_goal'] = record
        if self._emitted:
            return []
        self._emitted = True
        return [SKILL + _focus(record)]
