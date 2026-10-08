"""H2 -- requirement-gated purchase control for WebShop.

Public goal -> episode-local progress -> next action -> feedback:

* goal: the episode instruction read from the public history;
* progress: the option groups of the page in the latest public observation,
  which instruction-backed option values were already clicked on that page (the
  model-visible history plus the values this layer selected itself), and the
  turn budget H4 keeps in the shared dict;
* next action: the pending assistant action is replaced only when the evidence
  is unambiguous -- a 'buy now' click while an instruction-backed option of the
  current page is still unselected, a click value that is not one of the page's
  clickables while a required value is available, or a tool call the solver
  wrote as plain text instead of a tool call;
* feedback: drain_h2_messages() reports the replacement once (one-shot), so the
  solver can see which value was selected and finish with 'buy now'.

State owned here: 'seq' (replacement counter), 'note' (one-shot feedback) and
'h2_page' / 'h2_selected' (values selected here, read by H4, reset when the
page changes). State read from H4: 'guard_off' (turn budget nearly exhausted ->
no action override) and 'h4_values' / 'h4_required' (current page snapshot used
by the no-tool corrective message). Nothing is kept across episodes: the shared
dict and this instance are created per episode by the runtime.
"""
from json import dumps, loads
from re import I, S, search, sub

TARGET_LAYER = 'h2'

_WORDS = {
    'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5', 'six': '6',
    'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10', 'eleven': '11',
    'twelve': '12', 'twenty': '20', 'thirty': '30', 'forty': '40', 'fifty': '50',
}
_OPT_STOP = ('a', 'an', 'the', 'of', 'with', 'and', 'for', 'in', 'on', 'to', 'at', 'or', 'x')
_TRAILERS = ('description', 'features', 'reviews', 'buy now')
_NAV = ('back to search', 'prev', 'next', 'search')
_PRODUCT = r'^b0[0-9a-z]{8}$'


def _norm(value):
    text = value.lower().replace('&', ' and ').replace('#', ' ')
    text = sub(r'[^a-z0-9]+', ' ', text)
    return sub(r'\s+', ' ', text).strip()


def _digitize(text):
    for word in _WORDS:
        text = sub(r'\b' + word + r'\b', _WORDS[word], text)
    return text


def _strip_price(text):
    text = sub(r'(\d)\.(\d)', r'\1 \2', text.lower())
    text = sub(r'(and\s+|with\s+|a\s+|the\s+)*price\s+(is\s+|are\s+)?[^.;]*', ' ', text)
    text = sub(r'\b(lower|less|cheaper)\s+than\s+[^.;]*', ' ', text)
    text = sub(r'\b(under|below)\s+[\$]?\s*\d+(\s+\d+)?\s*(dollars|usd|bucks)?', ' ', text)
    text = sub(r'[\$]\s*\d+(\s+\d+)?', ' ', text)
    return sub(r'\b\d+(\s+\d+)?\s*(dollars|usd|bucks)\b', ' ', text)


def _compact(text):
    return _norm(text).replace(' ', '')


def _instruction_compact(instruction):
    if not instruction:
        return ''
    return _compact(_digitize(_strip_price(instruction)))


def _tok_in(token, ci):
    if token in ci:
        return True
    if len(token) > 3 and token.endswith('s') and token[:-1] in ci:
        return True
    if len(token) > 4 and token.endswith('es') and token[:-2] in ci:
        return True
    return False


def _lcs(left, right):
    left = left[:60]
    right = right[:260]
    if not left or not right:
        return 0
    best = 0
    previous = [0] * (len(right) + 1)
    for i in range(1, len(left) + 1):
        current = [0] * (len(right) + 1)
        char = left[i - 1]
        for j in range(1, len(right) + 1):
            if char == right[j - 1]:
                value = previous[j - 1] + 1
                current[j] = value
                if value > best:
                    best = value
        previous = current
    return best


def _match_value(value, ci):
    """None when the option value has no support in the instruction text."""
    if not isinstance(value, str) or not ci:
        return None
    nv = _digitize(_norm(value))
    if not nv:
        return None
    cv = nv.replace(' ', '')
    tokens = []
    for token in nv.split():
        if token in _OPT_STOP:
            continue
        if len(token) < 2 and not token.isdigit():
            continue
        tokens.append(token)
    if not tokens:
        return None
    matched = [token for token in tokens if _tok_in(token, ci)]
    if len(cv) > 2 and cv in ci:
        return {'phrase': True, 'n': len(matched), 'lcs': len(cv)}
    if matched and len(matched) * 2 >= len(tokens):
        return {'phrase': False, 'n': len(matched), 'lcs': _lcs(cv, ci)}
    tail = search(r'^(.*?)[\s\-_#]?\d{1,2}$', nv)
    if tail and tail.group(1).strip():
        base = _match_value(tail.group(1), ci)
        if base and (base['phrase'] or (base['n'] >= 1 and base['lcs'] >= 3)):
            return base
    return None


def _engaged(value, ci):
    """True when a click already addressed this option group.

    Used only to decide whether a group may still be touched: a value that
    matches the instruction, or one that shares a distinctive word with it,
    means the solver has already chosen a variant of that attribute.
    """
    if _match_value(value, ci):
        return True
    if not isinstance(value, str) or not ci:
        return False
    for token in _digitize(_norm(value)).split():
        if len(token) >= 4 and _tok_in(token, ci):
            return True
    return False


def _arguments(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = loads(value)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _scan_quoted(text, position):
    items = []
    index = position
    size = len(text)
    while index < size:
        char = text[index]
        if char == ']':
            break
        if char == "'" or char == '"':
            end = index + 1
            buffer = []
            while end < size:
                current = text[end]
                if current == '\\' and end + 1 < size:
                    buffer.append(text[end + 1])
                    end += 2
                    continue
                if current == char:
                    break
                buffer.append(current)
                end += 1
            items.append(''.join(buffer))
            index = end + 1
            continue
        index += 1
    return items


def _page(text):
    view = {'valid': False, 'is_product': False, 'clickables': [], 'groups': [], 'title': ''}
    if not isinstance(text, str) or 'Available Actions:' not in text:
        return view
    marker = text.find('clickables')
    begin = text.find('[', marker) if marker >= 0 else -1
    if begin < 0:
        return view
    values = _scan_quoted(text, begin)
    if not values:
        return view
    lookup = {}
    for item in values:
        lookup[_norm(item)] = item
    view['clickables'] = values
    view['valid'] = True
    view['is_product'] = 'buy now' in lookup
    tokens = text.split('[SEP]')
    start = -1
    for position in range(len(tokens)):
        if tokens[position].strip().lower() == '< prev':
            start = position + 1
    if start < 0:
        return view
    stop = len(tokens)
    for position in range(start, len(tokens)):
        low = tokens[position].lower()
        if 'price:' in low or 'rating:' in low or _norm(tokens[position]) in _TRAILERS:
            stop = position
            break
    view['title'] = tokens[stop - 1].strip() if stop - 1 >= start else ''
    end = stop - 1 if (stop - 1 > start and _norm(tokens[stop - 1]) not in lookup) else stop
    groups = []
    label = ''
    bucket = []
    for position in range(start, end):
        raw = tokens[position].strip()
        if not raw:
            continue
        if _norm(raw) in lookup:
            bucket.append(raw)
            continue
        if bucket:
            groups.append((label, bucket))
            bucket = []
        label = raw
    if bucket:
        groups.append((label, bucket))
    view['groups'] = groups
    return view


def _page_key(view):
    parts = []
    for item in view['clickables']:
        parts.append(_norm(item))
    return '|'.join(parts)[:300]


def _last_observation(messages):
    if not isinstance(messages, list):
        return ''
    for message in reversed(messages):
        if isinstance(message, dict):
            content = message.get('content')
            if isinstance(content, str) and 'Available Actions:' in content:
                return content
    return ''


def _instruction(messages, context):
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            content = message.get('content')
            if not isinstance(content, str):
                continue
            found = search(r'Instruction:\s*\[SEP\]\s*(.*?)\s*\[SEP\]', content, S)
            if found and found.group(1).strip():
                return found.group(1).strip()
    if isinstance(context, dict):
        value = context.get('instruction')
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ''


def _clicked(messages):
    clicked = set()
    if not isinstance(messages, list):
        return clicked
    for message in messages:
        if not isinstance(message, dict) or message.get('role') != 'assistant':
            continue
        for call in message.get('tool_calls') or []:
            function = call.get('function') if isinstance(call, dict) else None
            if not isinstance(function, dict):
                continue
            name = function.get('name')
            if name == 'search_action':
                clicked = set()
                continue
            if name != 'click_action':
                continue
            value = _arguments(function.get('arguments')).get('value')
            if not isinstance(value, str):
                continue
            key = _norm(value)
            if key in _NAV or search(_PRODUCT, key):
                clicked = set()
                continue
            if key in _TRAILERS:
                continue
            clicked.add(key)
    return clicked


def _required(view, ci, clicked):
    """Ranked instruction-backed option values of this page that are unselected.

    A group the solver already engaged with (a matched value, or one sharing a
    distinctive word with the instruction) is left alone: the variant was its
    choice and a click of another value of the same group would replace it.
    """
    ranked = []
    if not ci:
        return []
    order = 0
    for label, values in view.get('groups') or []:
        filled = False
        for item in values:
            if _norm(item) in clicked and _engaged(item, ci):
                filled = True
                break
        for item in values:
            order += 1
            if filled or _norm(item) in clicked:
                continue
            match = _match_value(item, ci)
            if not match:
                continue
            ranked.append((match, item, order))
    ranked.sort(key=_rank)
    return [row[1] for row in ranked]


def _rank(row):
    match = row[0]
    return (-match['lcs'], -match['n'], 0 if match['phrase'] else 1, row[2])


def _canonical(value, values):
    key = _norm(value) if isinstance(value, str) else ''
    if not key:
        return None
    for item in values:
        if _norm(item) == key:
            return item
    return None


def _join_values(items, limit):
    shown = []
    for item in items[:limit]:
        shown.append(repr(item))
    if len(items) > limit:
        shown.append('...')
    return ', '.join(shown)


def _quoted_field(text, key):
    marker = search(r'["\']?\b' + key + r'\b["\']?\s*[:=]\s*', text, I)
    if not marker:
        return None
    rest = text[marker.end():]
    if not rest:
        return None
    quote = rest[0] if rest[0] == '"' or rest[0] == "'" else ''
    if not quote:
        return rest.split('}')[0].strip().strip(',')
    last = rest.rfind(quote)
    if last <= 0:
        return None
    return rest[1:last]


def _pseudo_call(text):
    """Best-effort recovery of a tool call the solver wrote as reply text."""
    if not isinstance(text, str) or not text.strip():
        return None, None
    if 'click_action' in text:
        value = _quoted_field(text, 'value')
        if not value:
            value = _quoted_field(text, 'click')
        if not value:
            found = search(r'click_action\s*[\(\[\{]\s*["\']?(.*?)["\']?\s*[\)\]\}]', text, I)
            value = found.group(1) if found else None
        if isinstance(value, str) and value.strip():
            return 'click_action', {'value': value.strip()}
        return None, None
    if 'search_action' in text:
        keywords = _quoted_field(text, 'keywords')
        if not keywords:
            keywords = _quoted_field(text, 'query')
        if not keywords:
            found = search(r'search_action\s*[\(\[\{]\s*["\']?(.*?)["\']?\s*[\)\]\}]', text, I)
            keywords = found.group(1) if found else None
        if isinstance(keywords, str) and keywords.strip():
            return 'search_action', {'keywords': keywords.strip()}
    return None, None


def _click_message(value, state):
    state['seq'] = int(state.get('seq', 0)) + 1
    call_id = 'h2sel%02d' % state['seq']
    return {'role': 'assistant',
            'tool_calls': [{'id': call_id, 'type': 'function',
                            'function': {'name': 'click_action',
                                         'arguments': dumps({'value': value})}}]}


def _search_message(keywords, state):
    state['seq'] = int(state.get('seq', 0)) + 1
    call_id = 'h2sea%02d' % state['seq']
    return {'role': 'assistant',
            'tool_calls': [{'id': call_id, 'type': 'function',
                            'function': {'name': 'search_action',
                                         'arguments': dumps({'keywords': keywords})}}]}


class Harness:
    def bind_shared_state(self, state):
        self.state = state

    def bind_task_context(self, context):
        self.context = context

    def _state(self):
        current = self.state if hasattr(self, 'state') else None
        if not isinstance(current, dict):
            current = {}
            self.state = current
        return current

    def _context(self):
        return self.context if hasattr(self, 'context') else None

    def _selected(self, state, view):
        signature = _page_key(view)
        if state.get('h2_page') != signature:
            state['h2_page'] = signature
            state['h2_selected'] = []
        return set(state.get('h2_selected') or [])

    def _record(self, state, value):
        chosen = list(state.get('h2_selected') or [])
        key = _norm(value)
        if key not in chosen:
            chosen.append(key)
        state['h2_selected'] = chosen

    def _targets(self, view, history, state):
        if not view['valid']:
            return []
        ci = _instruction_compact(_instruction(history, self._context()))
        clicked = _clicked(history)
        clicked.update(self._selected(state, view))
        return _required(view, ci, clicked)

    def _selection_note(self, value, view, history, state):
        ci = _instruction_compact(_instruction(history, self._context()))
        clicked = _clicked(history)
        clicked.update(self._selected(state, view))
        clicked.add(_norm(value))
        left = _required(view, ci, clicked)
        if left:
            tail = 'Values still unselected on this page: %s.' % _join_values(left, 4)
        else:
            tail = 'No required value is left on this page, so click "buy now" to complete the purchase.'
        return '[H2] %r was selected because the instruction asks for it. %s' % (value, tail)

    def h2(self, message, history):
        state = self._state()
        if not isinstance(message, dict):
            return message
        calls = message.get('tool_calls') or []
        if not calls:
            return self._recover(message, history, state)
        if len(calls) != 1 or not isinstance(calls[0], dict):
            return message
        function = calls[0].get('function')
        if not isinstance(function, dict) or function.get('name') != 'click_action':
            return message
        value = _arguments(function.get('arguments')).get('value')
        if not isinstance(value, str) or not value.strip():
            return message
        view = _page(_last_observation(history))
        if not view['valid']:
            return message
        picked = _canonical(value, view['clickables'])
        if picked is None:
            targets = self._targets(view, history, state)
            if targets:
                state['note'] = self._selection_note(targets[0], view, history, state)
                self._record(state, targets[0])
                return _click_message(targets[0], state)
            return message
        if _norm(picked) == 'buy now':
            if state.get('guard_off'):
                return message
            targets = self._targets(view, history, state)
            if targets:
                state['note'] = self._selection_note(targets[0], view, history, state)
                self._record(state, targets[0])
                return _click_message(targets[0], state)
            return message
        if _norm(picked) != _norm(value):
            state['note'] = '[H2] The clickable value is spelled %r; it was clicked exactly.' % picked
            return _click_message(picked, state)
        return message

    def _recover(self, message, history, state):
        name, payload = _pseudo_call(message.get('content'))
        if name is None:
            return message
        if name == 'click_action':
            view = _page(_last_observation(history))
            if not view['valid']:
                return message
            picked = _canonical(payload.get('value'), view['clickables'])
            if picked is None:
                targets = self._targets(view, history, state)
                if not targets:
                    return message
                picked = targets[0]
            state['note'] = ('[H2] The tool call in your reply was executed: click_action(%r). '
                             'Write actions as tool calls, not prose.' % picked)
            self._record(state, picked)
            return _click_message(picked, state)
        keywords = payload.get('keywords')
        if not isinstance(keywords, str) or not keywords.strip():
            return message
        state['note'] = ('[H2] The tool call in your reply was executed: search_action(%r). '
                         'Write actions as tool calls, not prose.' % keywords)
        return _search_message(keywords, state)

    def h2_no_tool_message(self, message, remaining):
        state = self._state()
        state['note'] = None
        values = state.get('h4_values')
        left = state.get('h4_required')
        lines = ['[H2] No environment action was executed: your reply contained no tool call. '
                 'Reply with exactly one tool call now: search_action(keywords=...) or '
                 'click_action(value=...).']
        if isinstance(values, list) and values:
            lines.append('Clickable values on the current page: %s.' % _join_values(values, 8))
        if isinstance(left, list) and left:
            lines.append('Instruction-required values still unselected: %s. Click one of them now.'
                         % _join_values(left, 5))
        elif isinstance(values, list) and values:
            lines.append('If the product and its options are right, click "buy now".')
        return ' '.join(lines)

    def drain_h2_messages(self):
        state = self._state()
        note = state.get('note')
        state['note'] = None
        if isinstance(note, str) and note.strip():
            return [note]
        return []
