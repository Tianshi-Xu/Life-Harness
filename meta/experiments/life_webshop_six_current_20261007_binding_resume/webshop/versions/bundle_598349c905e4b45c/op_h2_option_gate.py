"""H2 -- instruction-variant purchase control for WebShop.

Public goal -> episode-local progress -> next action -> feedback:

* goal: the episode instruction read from the public history (falling back to
  the public task context);
* progress: the option groups of the page in the latest public observation,
  which instruction-named option values were already clicked on that page (the
  model-visible history plus the values this layer selected itself), and the
  turn budget H4 keeps in the shared dict;
* next action: the pending assistant action is replaced only when the evidence
  is specific -- a 'buy now' click while an instruction-named option value of
  the current page is still unselected, a click on an option value that is not
  the value the instruction names for that group, a click value that is not one
  of the page's clickables, or a tool call the solver wrote as plain text;
* feedback: drain_h2_messages() reports the replacement once (one-shot), so the
  solver sees which value was selected and can finish with 'buy now'.

An option value counts as instruction-named only when its own words appear as
words of the instruction (a contiguous run of instruction words, or a majority
of the value's words including at least one non-generic word). Substring hits
such as 'blue' inside 'bluetooth' are not evidence, and values of one option
group are alternatives: exactly one of them is the variant that gets bought.

State owned here: 'seq' (replacement counter), 'note' (one-shot feedback) and
'h2_page' / 'h2_selected' (values selected here, read by H4, reset when the
page changes). State read from H4: 'guard_off' (turn budget nearly exhausted ->
no buy-now override) and 'h4_values' / 'h4_required' (current page snapshot
used by the no-tool corrective message). The shared dict is created per episode
by the runtime, starts empty and is never shared across episodes.
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
_GENERIC = (
    'fast', 'quick', 'long', 'lasting', 'new', 'best', 'good', 'great', 'nice',
    'easy', 'hard', 'soft', 'high', 'low', 'top', 'quality', 'premium', 'super',
    'ultra', 'extra', 'multi', 'plus', 'free', 'set', 'type', 'use', 'used',
    'ideal', 'perfect', 'gift', 'women', 'men', 'kids', 'adult', 'size', 'sized',
    'pack', 'packs', 'piece', 'pieces', 'color', 'colour', 'style',
)
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


def _words_of(text):
    if not isinstance(text, str):
        return []
    return _norm(_digitize(_norm(text))).split()


def _same(left, right):
    if left == right:
        return True
    if len(left) > 3 and left[-1] == 's' and left[:-1] == right:
        return True
    if len(right) > 3 and right[-1] == 's' and right[:-1] == left:
        return True
    if len(left) > 4 and left[-2:] == 'es' and left[:-2] == right:
        return True
    if len(right) > 4 and right[-2:] == 'es' and right[:-2] == left:
        return True
    return False


def _has_word(token, words):
    for word in words:
        if _same(token, word):
            return True
    return False


def _contiguous(tokens, words):
    size = len(tokens)
    if not size or size > len(words):
        return False
    for start in range(len(words) - size + 1):
        hit = True
        for offset in range(size):
            if not _same(tokens[offset], words[start + offset]):
                hit = False
                break
        if hit:
            return True
    return False


def _content(value):
    tokens = []
    for token in _words_of(value):
        if token in _OPT_STOP:
            continue
        if len(token) < 2 and not token.isdigit():
            continue
        tokens.append(token)
    return tokens


def _instruction_words(instruction):
    if not instruction:
        return []
    return _words_of(_strip_price(instruction))


def _match(value, words):
    """None when the option value has no word-level support in the instruction."""
    if not isinstance(value, str) or not words:
        return None
    tokens = _words_of(value)
    if not tokens:
        return None
    if _contiguous(tokens, words):
        return {'level': 0, 'n': len(tokens), 'chars': len(''.join(tokens))}
    content = _content(value)
    if not content:
        return None
    matched = [token for token in content if _has_word(token, words)]
    if not matched:
        return None
    if len(matched) * 2 < len(content):
        return None
    strong = [token for token in matched if token not in _GENERIC]
    if not strong:
        return None
    return {'level': 1, 'n': len(matched), 'chars': sum(len(token) for token in strong)}


def _engaged(value, words):
    """True when a click already addressed this option group.

    Used only to decide whether a group may still be touched: a value the
    instruction names, or one sharing a distinctive word with it, means the
    solver has already chosen a variant of that attribute.
    """
    if _match(value, words):
        return True
    for token in _content(value):
        if len(token) >= 4 and token not in _GENERIC and _has_word(token, words):
            return True
    return False


def _rank(row):
    match = row[0]
    return (match['level'], -match['n'], -match['chars'], row[2])


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


def _required(view, words, clicked):
    """Best instruction-named, still unselected value of every option group.

    One value per group: the values inside a group are alternatives of the same
    attribute and the last click is the variant that gets bought, so asking for
    several of them would only keep flipping the variant. A group the solver
    already engaged with is left alone.
    """
    ranked = []
    if not words:
        return []
    order = 0
    for label, values in view.get('groups') or []:
        filled = False
        for item in values:
            if _norm(item) in clicked and _engaged(item, words):
                filled = True
                break
        best = None
        for item in values:
            order += 1
            if filled or _norm(item) in clicked:
                continue
            match = _match(item, words)
            if not match:
                continue
            row = (match, item, order)
            if best is None or _rank(row) < _rank(best):
                best = row
        if best is not None:
            ranked.append(best[1])
    return ranked


def _variant_override(view, words, value):
    """Named value of the clicked group, when the solver clicked a different one.

    None when the clicked value is itself named by the instruction, when it is
    not part of any option group, or when its group names nothing.
    """
    key = _norm(value) if isinstance(value, str) else ''
    if not key or not words:
        return None
    if _match(value, words):
        return None
    for label, values in view.get('groups') or []:
        found = False
        for item in values:
            if _norm(item) == key:
                found = True
                break
        if not found:
            continue
        best = None
        order = 0
        for item in values:
            order += 1
            if _norm(item) == key:
                continue
            match = _match(item, words)
            if not match:
                continue
            row = (match, item, order)
            if best is None or _rank(row) < _rank(best):
                best = row
        return best[1] if best is not None else None
    return None


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
    if search(r'click_action', text, I):
        value = _quoted_field(text, 'value')
        if not value:
            value = _quoted_field(text, 'click')
        if not value:
            found = search(r'click_action\s*[\(\[\{]\s*["\']?(.*?)["\']?\s*[\)\]\}]', text, I)
            value = found.group(1) if found else None
        if isinstance(value, str) and value.strip():
            return 'click_action', {'value': value.strip()}
        return None, None
    if search(r'search_action', text, I):
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

    def _words(self, history):
        return _instruction_words(_instruction(history, self._context()))

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
        words = self._words(history)
        clicked = _clicked(history)
        clicked.update(self._selected(state, view))
        return _required(view, words, clicked)

    def _selection_note(self, value, view, history, state):
        words = self._words(history)
        clicked = _clicked(history)
        clicked.update(self._selected(state, view))
        clicked.add(_norm(value))
        left = _required(view, words, clicked)
        if left:
            tail = 'Values still unselected on this page: %s.' % _join_values(left, 3)
        else:
            tail = 'No instruction-named value is left unselected, so click "buy now" to complete the purchase.'
        return '[H2] %r was selected because the instruction names it. %s' % (value, tail)

    def h2(self, message, history):
        try:
            return self._control(message, history, self._state())
        except Exception:
            return message

    def _control(self, message, history, state):
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
        better = _variant_override(view, self._words(history), picked)
        if better is not None:
            state['note'] = ('[H2] The instruction names %r for that option group, so %r was clicked '
                             'instead of %r.' % (better, better, picked))
            self._record(state, better)
            return _click_message(better, state)
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
        lines = ['[H2] No environment action was executed: your reply contained no tool call. '
                 'Answer with exactly one tool call now: search_action(keywords=...) or '
                 'click_action(value=...).']
        values = state.get('h4_values')
        left = state.get('h4_required')
        if isinstance(left, list) and left:
            lines.append('Instruction-named values still unselected on this page: %s. Click %r now '
                         '(one value per option group), then click "buy now".'
                         % (_join_values(left, 4), left[0]))
        elif isinstance(values, list) and values:
            lines.append('The current page offers the clickable values %s. If the product and its '
                         'options match the instruction, click "buy now"; otherwise click '
                         '"back to search" and open a better product.' % _join_values(values, 8))
        else:
            lines.append('If the current observation is the product page, click "buy now"; if it only '
                         'shows Description/Features/Reviews, click "< prev" first.')
        return ' '.join(lines)

    def drain_h2_messages(self):
        state = self._state()
        note = state.get('note')
        state['note'] = None
        if isinstance(note, str) and note.strip():
            return [note]
        return []
