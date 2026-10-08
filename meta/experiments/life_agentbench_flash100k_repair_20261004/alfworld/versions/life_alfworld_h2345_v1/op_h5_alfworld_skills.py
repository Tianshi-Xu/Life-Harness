"""H5 ALFWorld cold-start skill bank: protocol, survey discipline, per-family recipes."""
from re import I, compile, search, split

TARGET_LAYER = 'h5'

_TASK = compile(r'your task is to:?\s*([^\n]+)', I)
MARKERS = ('AVAILABLE ACTIONS', 'Interact with a household')
NOISE = ('a', 'an', 'the', 'some', 'it', 'in', 'on', 'under', 'with', 'to', 'and', 'of',
         'hot', 'cool', 'cold', 'clean', 'cooled', 'heated', 'warm')
SKILL = (
    'ALFWorld procedure (apply before acting):\n'
    '1. Every turn must be one take_action call; copy the action string verbatim from the '
    'AVAILABLE ACTIONS list of the newest tool message, because any other string changes nothing.\n'
    '2. Survey (look, inventory, examine) only to reveal a choice you do not already have; never '
    'repeat a survey whose observation is unchanged.\n'
    '3. Placement goals (put some X on/in Y): take an X you can see, then put X in/on Y.\n'
    '4. Look-at goals (look at X under the desklamp): take X and also use the desklamp; only one '
    'of the two is not enough.\n'
    '5. Process goals (cool/heat/clean X and put it in Y): take X, process it in fridge 1 or '
    'microwave 1 or clean X with sinkbasin 1, take it back out, then put it in/on Y.\n'
    '6. Never end a turn with plain text; an assistant turn without a tool call is discarded and '
    'wastes a turn.'
)


def _joined(messages):
    parts = []
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict) and isinstance(message.get('content'), str):
                parts.append(message['content'])
    return '\n'.join(parts)


def _head(phrase):
    words = [word for word in split(r'[^a-z0-9]+', str(phrase).lower()) if word]
    words = [word for word in words if word not in NOISE]
    return words[0] if words else None


def _goal(text):
    found = search(_TASK, text)
    if found is None:
        return None
    sentence = ' '.join(found.group(1).lower().split())
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


class Harness:
    def __init__(self):
        self.state = {}
        self.context = None

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
        return [SKILL]
