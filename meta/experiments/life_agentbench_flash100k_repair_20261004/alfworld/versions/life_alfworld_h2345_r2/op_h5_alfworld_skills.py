"""H5 ALFWorld cold-start skill: one call per turn, ordered recipes, goal-object focus."""
from re import I, compile, findall, search

TARGET_LAYER = 'h5'

_TASK = compile(r'your task is to:?\s*([^\n]+)', I)
_WORDS = compile(r'[a-z0-9]+')
_VERB = compile(r'\b(cool|heat|clean)\b', I)
_PUTWORD = compile(r'\bput\b', I)
MARKERS = ('AVAILABLE ACTIONS', 'Interact with a household')
SKIP = ('a', 'an', 'the', 'some', 'in', 'on', 'to', 'under', 'with', 'and', 'of', 'it',
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
    '3. Placement goals (put some X in/on Y): take an X, then put X in/on Y.\n'
    '4. Process goals (cool/heat/clean some X and put it in Y): take X, go to fridge 1 for cool, '
    'microwave 1 for heat or sinkbasin 1 for clean, open it if it is closed, run the cool/heat/clean '
    'action on X, then go to Y and put X there; never place X in Y before that step.\n'
    '5. Light goals (look at or examine X with the desklamp or under it): turn the lamp on and take X.\n'
    '6. If the goal object is not in the current room, search: move to another receptacle, open it if '
    'it is closed, and never repeat a survey action whose observation did not change.\n'
    '7. Never end a turn with plain text; an assistant turn without a tool call is discarded.'
)


def _joined(messages):
    parts = []
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict) and isinstance(message.get('content'), str):
                parts.append(message['content'])
    return '\n'.join(parts)


def _tokens(text):
    return findall(_WORDS, str(text).lower())


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
    sentence = ' '.join(found.group(1).lower().split())
    cut = sentence.find('available actions')
    if cut != -1:
        sentence = sentence[:cut]
    sentence = sentence.rstrip('. ')
    if not sentence:
        return None
    record = {'text': sentence, 'mode': 'place', 'object': None, 'destination': None,
              'object_candidates': [], 'destination_candidates': [], 'light': None}
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
        record['mode'] = 'look_at_light'
        record['object_candidates'] = _candidates(head)
        record['object'] = record['object_candidates'][0] if record['object_candidates'] else None
        record['light'] = 'desklamp'
        return record
    verb = search(_VERB, sentence)
    putword = search(_PUTWORD, sentence)
    mode = None
    if verb is not None and (putword is None or verb.start() < putword.start()):
        mode = verb.group(1)
        tail = sentence[verb.end():]
        for marker in (' and ', ' in ', ' on '):
            if marker in tail:
                tail = tail.split(marker, 1)[0]
                break
        record['object_candidates'] = _candidates(tail)
    elif putword is not None:
        tail = sentence[putword.end():]
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
