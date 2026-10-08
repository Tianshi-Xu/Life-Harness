"""H4 -- episode-stage requirement tracker for WebShop.

Public goal -> episode-local progress -> next action -> feedback:

* goal: the episode instruction, read from the public history (or from the
  public task context as a fallback);
* progress: which stage the episode is in (search bar, result page, product
  page, product sub-page), the instruction-named option values of the current
  product page that are still unselected, whether the value the instruction
  names for an option family is absent from this product, how many result pages
  were already shown without opening a product, and the remaining budget;
* next action: at most one short imperative instruction, emitted only when
  something is actionable -- leave a product whose colour options cannot offer
  the colour phrase the instruction names and open one concrete better title
  instead, click the best unselected instruction-named option value (one value
  per option group, because the values of one group are alternatives), open the
  title that matches the instruction best when the list has already been seen or
  paged through without opening anything, click '< prev' after a
  description/features/reviews view, or click 'buy now' when the budget is
  nearly exhausted and nothing is left to select;
* feedback: the solver's next observation shows the effect, and the snapshot
  written here ('h4_values', 'h4_required', 'h4_best', 'h4_stage') is reused by
  H2 for its no-tool corrective message.

Option evidence is word-level and quantity-aware: a glued number and unit is
tokenised apart ('200ft' matches '200 foot'), glue words and bare units are not
evidence ('3 ft' is not the value named by '135 ft'), a value whose own numbers
contradict the instruction is not named unless another distinctive word of the
value really is a matched instruction word ('40x24 inch' is not '40x32'), and
named alternatives are ranked by how many of the instruction's own numbers they
offer before page order, so '52" w by 90" l' wins over the '52" w by 52" l' that
merely comes first.

The colour obligation of the previous revision is generalised: it covers every
colour phrase the instruction names, a phrase being a colour word together with
the modifier or colour word in front of it ('rose gold', 'light gray', 'silver
gray'), and it is checked with whole-word evidence against the page's colour
options before any option is named. A page whose colour options exist but offer
neither phrase cannot satisfy the instruction, so the next action is 'back to
search' plus one concrete matching title from the result list.

Guidance never re-judges an executed click against the page that replaced it,
and it is never a descriptive verdict about a result page: every emitted line is
a staged next action naming a clickable value or a product link.

State owned here: 'remaining', 'guard_off' (turn budget), 'h4_values',
'h4_required', 'h4_best', 'h4_stage' (current page snapshot, read by H2),
'h4_page' / 'h4_seen' (per-page episode-local progress) and 'h4_hint' (dedupe of
the last emitted guidance). State read from H2: 'h2_page' / 'h2_selected', the
option values H2 selected itself. The dict is the runtime-provided episode-local
shared state: it starts empty for every episode and is never shared across
episodes.
"""
from json import loads
from re import S, findall, search, sub

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
_UNIT_WORDS = (
    'ft', 'foot', 'feet', 'inch', 'inches', 'in', 'cm', 'mm', 'meter', 'meters',
    'oz', 'ounce', 'ounces', 'lb', 'lbs', 'pound', 'pounds', 'ml', 'l', 'ghz',
    'mhz', 'hz', 'watt', 'watts', 'volt', 'volts', 'amp', 'amps', 'pack',
    'packs', 'piece', 'pieces', 'pc', 'pcs', 'count', 'slot', 'slots', 'tb',
    'gb', 'mb', 'gallon', 'gallons',
)
_GLUE = (
    'by', 'w', 'h', 'd', 'and', 'or', 'with', 'of', 'for', 'the', 'a', 'an',
    'to', 'at', 'on', 'in', 'is', 'are', 'be', 'it', 'its',
)
_WEAK = set(_GENERIC) | set(_UNIT_WORDS) | set(_GLUE) | set(_FUNCTION)
_COLOURS = (
    'black', 'white', 'grey', 'gray', 'blue', 'red', 'green', 'pink', 'purple',
    'yellow', 'orange', 'brown', 'gold', 'silver', 'navy', 'beige', 'teal',
    'tan', 'ivory', 'multicolor', 'clear', 'wine', 'cream', 'violet', 'maroon',
)
_COLOUR_MODIFIERS = (
    'rose', 'light', 'dark', 'deep', 'pale', 'bright', 'antique', 'baby',
    'neon', 'pastel', 'dusty', 'burnt', 'medium', 'natural', 'ash', 'bleach',
    'chestnut', 'ginger', 'auburn', 'royal', 'sky', 'sea', 'stone', 'ice',
    'sand', 'off', 'mid',
)


def _norm(value):
    text = value.lower().replace('&', ' and ').replace('#', ' ')
    text = sub(r'[^a-z0-9]+', ' ', text)
    return sub(r'\s+', ' ', text).strip()


def _digitize(text):
    for word in _WORDS:
        text = sub(r'\b' + word + r'\b', _WORDS[word], text)
    return text


def _split_glue(text):
    """Keep a glued number and unit matchable: '200ft' -> '200 ft'."""
    text = sub(r'(?<=\d)(?=[a-z])', ' ', text)
    return sub(r'(?<=[a-z])(?=\d)', ' ', text)


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
    return _split_glue(_norm(_digitize(_norm(text)))).split()


def _numbers(text):
    if not isinstance(text, str):
        return []
    return [int(token) for token in findall(r'\d+', text)]


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
    if len(left) > 4 and left[-3:] == 'ies' and left[:-3] + 'y' == right:
        return True
    if len(right) > 4 and right[-3:] == 'ies' and right[:-3] + 'y' == left:
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


def _strong(tokens):
    return [token for token in tokens
            if token not in _WEAK and (len(token) > 1 or token.isdigit())]


def _instruction_words(instruction):
    if not instruction:
        return []
    return _words_of(_strip_price(instruction))


def _named_phrase(tokens, words):
    """False when a contiguous hit is only part of a longer instruction phrase."""
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


def _agrees(value, words):
    """False when the value's own numbers rule it out for the instruction."""
    want = _numbers(' '.join(words))
    have = _numbers(value)
    if not want or not have:
        return True
    if not [item for item in have if item in want]:
        return False
    extra = [item for item in have if item not in want]
    if not extra:
        return True
    for token in _content(value):
        if token.isdigit() or token in _WEAK:
            continue
        if _has_word(token, words):
            return True
    return False


def _num_cover(value, words):
    """How many of the instruction's own numbers the value offers."""
    want = set(_numbers(' '.join(words)))
    if not want:
        return 0
    return len(want.intersection(_numbers(value)))


def _match(value, words):
    """None when the option value has no solid word-level support in the instruction."""
    if not isinstance(value, str) or not words:
        return None
    tokens = _words_of(value)
    if not tokens or not _agrees(value, words):
        return None
    if _contiguous(tokens, words) and _named_phrase(tokens, words) and _strong(tokens):
        return {'level': 0, 'n': len(tokens), 'chars': len(''.join(tokens)),
                'cover': _num_cover(value, words)}
    content = _content(value)
    if not content:
        return None
    matched = [token for token in content if _has_word(token, words)]
    if not matched:
        return None
    if len(matched) * 2 < len(content):
        return None
    strong = [token for token in matched if token not in _WEAK]
    if not strong:
        return None
    return {'level': 1, 'n': len(matched), 'chars': sum(len(token) for token in strong),
            'cover': _num_cover(value, words)}


def _engaged(value, words):
    """True when a click already addressed this option group."""
    if _match(value, words):
        return True
    for token in _content(value):
        if len(token) >= 4 and token not in _WEAK and _has_word(token, words):
            return True
    return False


def _rank(row):
    match = row[0]
    return (match['level'], -match['cover'], -match['n'], -match['chars'], row[2])


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
        if len(word) < 4 or word in _WEAK:
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


def _best_result(text, words, opened, minimum):
    """(-hits, position, asin, title, hits) of the best unopened title match."""
    best = None
    if not words:
        return None
    for position, entry in enumerate(_result_entries(text)):
        asin, title = entry[0], entry[1]
        if asin in opened:
            continue
        hits = _covered(title, words)
        if len(hits) < minimum:
            continue
        row = (-len(hits), position, asin, title, hits)
        if best is None or row < best:
            best = row
    return best


def _last_result(messages, words, opened):
    """Best matching unopened title of the most recent result list."""
    if not isinstance(messages, list):
        return None
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        content = message.get('content')
        if not isinstance(content, str) or 'Available Actions:' not in content:
            continue
        view = _page(content)
        if not view['valid'] or view['is_product']:
            continue
        keys = set()
        for item in view['clickables']:
            keys.add(_norm(item))
        if 'next' not in keys:
            continue
        best = _best_result(content, words, opened, 2)
        if best is not None:
            return best
    return None


def _required(view, words, clicked):
    """Best instruction-named, still unselected value of every option group.

    One value per group: the values inside a group are alternatives of the same
    attribute and the last click is the variant that gets bought. A group the
    solver already engaged with is left alone.
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


def _colour_wants(words):
    """Colour phrases the instruction names; a phrase keeps its leading modifier."""
    wants = []
    for index, word in enumerate(words):
        if word not in _COLOURS:
            continue
        previous = words[index - 1] if index else ''
        if previous in _COLOURS or previous in _COLOUR_MODIFIERS:
            pair = (previous, word)
            if pair not in wants:
                wants.append(pair)
            continue
        if word not in wants:
            wants.append(word)
    return wants


def _offers(value, needed):
    tokens = _words_of(value)
    for word in needed:
        if not _has_word(word, tokens):
            return False
    return True


def _phrase(want):
    if isinstance(want, str):
        return repr(want)
    return ' '.join(repr(word) for word in want)


def _missing_colour(view, words):
    """(label, want, offered) when the instruction names a colour this page lacks."""
    wants = _colour_wants(words)
    if not wants:
        return None
    for label, values in view.get('groups') or []:
        low = _norm(label) if isinstance(label, str) else ''
        if 'color' not in low and 'colour' not in low:
            continue
        if not values:
            continue
        for want in wants:
            needed = (want,) if isinstance(want, str) else want
            if not any(_offers(item, needed) for item in values):
                return (label, want, values[:5])
    return None


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
        best = None
        if stage == 'results':
            best = _best_result(text, words, _opened(messages), 2)
        state['h4_values'] = values[:12]
        state['h4_required'] = targets[:4]
        state['h4_stage'] = stage
        state['h4_best'] = [best[2], best[3]] if best is not None else None
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
            missing = _missing_colour(view, words)
            if missing is not None and not state.get('guard_off'):
                key = 'colour|' + signature + '|' + repr(missing[1])
                if key not in seen:
                    seen.append(key)
                    tail = ''
                    candidate = _last_result(messages, words, _opened(messages))
                    if candidate is not None:
                        tail = (' Click %s now instead: its title %r matches the instruction best.'
                                % (candidate[2], candidate[3][:70]))
                    lines.append('Click "back to search" now: this product cannot satisfy the '
                                 'instruction, because its colour options (%s) do not include %s, '
                                 'which the instruction asks for.%s'
                                 % (_join_values(missing[2], 5), _phrase(missing[1]), tail))
            elif targets:
                key = 'options|' + signature + '|' + _join_values(targets[:2], 2)
                if key not in seen:
                    seen.append(key)
                    lines.append('The instruction names the option value %s and it is not selected here '
                                 'yet: click %r now (one value per option group -- clicking another value '
                                 'of the same group replaces it), then click "buy now".'
                                 % (_join_values(targets[:2], 2), targets[0]))
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
                if best is None and repeated:
                    best = _best_result(text, words, _opened(messages), 1)
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
