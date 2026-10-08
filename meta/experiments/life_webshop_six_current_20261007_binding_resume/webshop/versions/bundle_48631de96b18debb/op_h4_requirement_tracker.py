"""H4 -- episode-stage requirement tracker for WebShop.

Public goal -> episode-local progress -> next action -> feedback:

* goal: the episode instruction, read from the public history (or from the
  public task context as a fallback);
* progress: which stage the episode is in (search bar, result page, product
  page, product sub-page), the instruction-named option values of the current
  product page that are still unselected, whether the named value of an option
  group is missing from this product, how many result pages were already shown
  without opening a product, and the remaining budget;
* next action: at most one short imperative instruction, emitted only when
  something is actionable -- click the best unselected instruction-named option
  value (one value per option group, because the values of one group are
  alternatives), open the title that matches the instruction best when the
  result list has already been seen or paged through without opening anything,
  click '< prev' after a description/features/reviews view, or click 'buy now'
  when the budget is nearly exhausted and nothing is left to select;
* feedback: the solver's next observation shows the effect, and the snapshot
  written here ('h4_values', 'h4_required') is reused by H2 for its no-tool
  corrective message.

Guidance never re-judges an executed click against the page that replaced it: a
product link, a description/features view or 'back to search' legitimately
removes the clicked value from the clickables list, so warning there would be
false. Guidance is also never a verdict about the result page: a list of
everyday products is not an "accessory" page just because their titles contain
a word such as 'case', 'cover', 'mount' or 'band', and the requested product
itself ('pillow covers', 'floating shelves', 'fitbit bands') must never be
reported as an accessory. Descriptive verdicts of that shape invited the solver
to repeat them as its own reply, which cost whole turns, so every emitted line
is a staged next action with a clickable value in it.

State owned here: 'remaining', 'guard_off' (turn budget), 'h4_values',
'h4_required' (current page snapshot, read by H2), 'h4_page' / 'h4_stage' /
'h4_seen' (per-page episode-local progress) and 'h4_hint' (dedupe of the last
emitted guidance). State read from H2: 'h2_page' / 'h2_selected', the option
values H2 selected itself. The dict is the runtime-provided episode-local
shared state: it starts empty for every episode and is never shared across
episodes.
"""
from json import loads
from re import S, search, sub

TARGET_LAYER = 'h4'

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
_FUNCTION = _OPT_STOP + (
    'is', 'are', 'was', 'were', 'be', 'am', 'do', 'does', 'it', 'its', 'that',
    'this', 'these', 'those', 'my', 'me', 'i', 'you', 'your', 'we', 'they',
    'them', 'he', 'she', 'his', 'her', 'there', 'here', 'also', 'then', 'only',
    'should', 'would', 'must', 'can', 'will', 'come', 'comes', 'coming', 'have',
    'has', 'had', 'need', 'needs', 'want', 'wants', 'looking', 'look', 'search',
    'searching', 'like', 'prefer', 'prefers', 'choose', 'choice', 'price',
    'dollar', 'dollars', 'lower', 'less', 'than', 'size', 'sized', 'inch',
    'inches', 'foot', 'feet', 'about', 'please', 'some', 'any', 'one', 'pair',
)
_ALT = ('or', 'either')
_TRAILERS = ('description', 'features', 'reviews', 'buy now')
_NAV = ('back to search', 'prev', 'next', 'search')
_PRODUCT = r'^b0[0-9a-z]{8}$'
_UNIT = (
    'inch', 'inches', 'feet', 'foot', 'size', 'sized', 'color', 'colour',
    'pack', 'packs', 'price', 'dollars', 'lower', 'than', 'style', 'piece',
    'pieces', 'with', 'that', 'this', 'your', 'should', 'come', 'comes',
)
_COLOURS = (
    'black', 'white', 'grey', 'gray', 'blue', 'red', 'green', 'pink', 'purple',
    'yellow', 'orange', 'brown', 'gold', 'silver', 'navy', 'beige', 'teal',
    'tan', 'ivory', 'multicolor', 'clear',
)


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


def _run(tokens, words):
    """(start, end) of the first contiguous run of instruction words, or None."""
    size = len(tokens)
    if not size or size > len(words):
        return None
    for start in range(len(words) - size + 1):
        hit = True
        for offset in range(size):
            if not _same(tokens[offset], words[start + offset]):
                hit = False
                break
        if hit:
            return (start, start + size)
    return None


def _contiguous(tokens, words):
    return _run(tokens, words) is not None


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


def _named_phrase(tokens, words):
    """None when a contiguous hit only repeats part of a longer instruction phrase."""
    span = _run(tokens, words)
    if span is None:
        return False
    start, end = span
    if len(tokens) == 1:
        if start > 0 and words[start - 1] in _ALT:
            return False
        if end < len(words) and words[end] in _ALT:
            return False
    if start > 0:
        before = words[start - 1]
        if (len(before) >= 3 and before not in _FUNCTION and before not in _GENERIC
                and not _has_word(before, tokens)):
            return False
    return True


def _match(value, words):
    """None when the option value has no word-level support in the instruction."""
    if not isinstance(value, str) or not words:
        return None
    tokens = _words_of(value)
    if not tokens:
        return None
    if _contiguous(tokens, words) and _named_phrase(tokens, words):
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


def _opened(messages):
    """Product links this episode already opened, in order."""
    found = []
    if not isinstance(messages, list):
        return found
    for message in messages:
        if not isinstance(message, dict) or message.get('role') != 'assistant':
            continue
        for call in message.get('tool_calls') or []:
            function = call.get('function') if isinstance(call, dict) else None
            if not isinstance(function, dict) or function.get('name') != 'click_action':
                continue
            value = _arguments(function.get('arguments')).get('value')
            if not isinstance(value, str):
                continue
            key = _norm(value)
            if search(_PRODUCT, key) and key not in found:
                found.append(key)
    return found


def _result_entries(text):
    """(asin, title) pairs shown on a result page."""
    entries = []
    if not isinstance(text, str):
        return entries
    pieces = text.split('[SEP]')
    for index in range(len(pieces) - 1):
        key = _norm(pieces[index])
        if search(_PRODUCT, key):
            title = pieces[index + 1].strip()
            if title:
                entries.append((key, title))
    return entries


def _result_pages(messages):
    """Signatures of the result pages shown since the last product page."""
    pages = []
    if not isinstance(messages, list):
        return pages
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get('content')
        if not isinstance(content, str) or 'Available Actions:' not in content:
            continue
        view = _page(content)
        if not view['valid']:
            continue
        keys = set()
        for item in view['clickables']:
            keys.add(_norm(item))
        if view['is_product']:
            pages = []
        elif 'next' in keys:
            pages.append(_page_key(view))
    return pages


def _distinctive(words):
    picked = []
    for word in words:
        if len(word) < 4 or word in _GENERIC or word in _UNIT:
            continue
        if word not in picked:
            picked.append(word)
    return picked


def _covered(title, words):
    tokens = _words_of(title)
    hits = []
    for word in _distinctive(words):
        if _has_word(word, tokens):
            hits.append(word)
    return hits


def _best_result(text, words, opened):
    """(-hits, position, asin, title, hits) of the best unopened title match."""
    best = None
    if not words:
        return None
    for position, entry in enumerate(_result_entries(text)):
        asin, title = entry[0], entry[1]
        if asin in opened:
            continue
        hits = _covered(title, words)
        if len(hits) < 2:
            continue
        row = (-len(hits), position, asin, title, hits)
        if best is None or row < best:
            best = row
    return best


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


def _carried(state, view):
    if state.get('h2_page') == _page_key(view):
        return set(state.get('h2_selected') or [])
    return set()


def _absent_colour(view, words):
    """(group label, colour, offered) when the instruction names a colour this page lacks."""
    wanted = []
    for word in words:
        if word in _COLOURS and word not in wanted:
            wanted.append(word)
    if len(wanted) != 1:
        return None
    want = wanted[0]
    label = None
    seen = False
    offered = []
    for name, values in view.get('groups') or []:
        low = _norm(name)
        if 'color' not in low and 'colour' not in low:
            continue
        if not values:
            continue
        seen = True
        for item in values:
            offered.append(item)
            if _match(item, [want]):
                return None
        label = name
    if not seen or label is None:
        return None
    return (label, want, offered[:5])


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

    def h4(self, messages, remaining):
        state = self._state()
        if isinstance(remaining, bool) or not isinstance(remaining, int):
            state['remaining'] = None
            state['guard_off'] = False
        else:
            state['remaining'] = remaining
            state['guard_off'] = remaining <= 2
        try:
            return self._guide(state, messages, remaining)
        except Exception:
            return []

    def _guide(self, state, messages, remaining):
        text = _last_observation(messages)
        view = _page(text)
        words = _instruction_words(_instruction(messages, self._context()))
        clicked = _clicked(messages)
        clicked.update(_carried(state, view))
        keys = set()
        for item in view['clickables']:
            keys.add(_norm(item))
        stage = 'other'
        if view['valid']:
            if 'buy now' in keys:
                stage = 'product'
            elif 'next' in keys:
                stage = 'results'
            elif 'prev' in keys:
                stage = 'detail'
            elif 'search' in keys:
                stage = 'start'
        values = []
        for item in view['clickables']:
            key = _norm(item)
            if key in _TRAILERS or key in _NAV:
                continue
            if search(_PRODUCT, key):
                continue
            values.append(item)
        targets = _required(view, words, clicked) if stage == 'product' else []
        state['h4_values'] = values[:12]
        state['h4_required'] = targets[:4]
        state['h4_stage'] = stage
        signature = _page_key(view)
        state['h4_page'] = signature
        seen = list(state.get('h4_seen') or [])
        lines = []
        if stage == 'detail':
            key = 'detail|' + signature
            if key not in seen:
                seen.append(key)
                lines.append("The current view is the product's description/features/reviews section, not "
                             "the product page: click '< prev' to get back to the product page, then click "
                             '"buy now" if the product matches the instruction, otherwise click "back to '
                             'search".')
        elif stage == 'product':
            if targets:
                key = 'options|' + signature + '|' + _join_values(targets[:2], 2)
                if key not in seen:
                    seen.append(key)
                    lines.append('The instruction names the option value %s and it is not selected here '
                                 'yet: click %r now (one value per option group -- clicking another value '
                                 'of the same group replaces it), then click "buy now".'
                                 % (_join_values(targets[:2], 2), targets[0]))
            else:
                missing = _absent_colour(view, words)
                if missing is not None and not state.get('guard_off'):
                    key = 'colour|' + signature + '|' + missing[1]
                    if key not in seen:
                        seen.append(key)
                        lines.append('Click "back to search" now: this product cannot satisfy the '
                                     'instruction, because its colour options (%s) do not include %r, '
                                     'which the instruction asks for. Open the result whose title '
                                     'matches the instruction best instead of this one (%r).'
                                     % (_join_values(missing[2], 5), missing[1], view['title'][:60]))
                elif remaining is not None and remaining <= 4:
                    key = 'ready|' + signature
                    if key not in seen:
                        seen.append(key)
                        lines.append('Nothing is left to select on this page: click "buy now" now to '
                                     'complete the purchase, only %s turns are left.' % remaining)
        elif stage == 'results':
            pages = _result_pages(messages)
            repeated = pages.count(signature) >= 2
            if repeated or len(pages) >= 3:
                best = _best_result(text, words, _opened(messages))
                if best is not None:
                    asin, title, hits = best[2], best[3], best[4]
                    key = 'best|' + signature + '|' + asin
                    if key not in seen:
                        seen.append(key)
                        lines.append('Click %s now (its title %r covers %s of the instruction), then '
                                     'check its option values and click "buy now". Do not search or '
                                     'page again from this list.'
                                     % (asin, title[:70], _join_values(hits, 4)))
        state['h4_seen'] = seen[-8:]
        if not lines:
            return []
        hint = '\n'.join(lines)
        if hint == state.get('h4_hint'):
            return []
        state['h4_hint'] = hint
        return [hint]
