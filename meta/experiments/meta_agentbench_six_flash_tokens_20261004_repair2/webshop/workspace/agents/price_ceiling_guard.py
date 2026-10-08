"""Hard price-ceiling guard runtime (round-2 candidate).

Mechanism (single new hypothesis): a stated hard budget constraint is never
violated by a finalised purchase.

This runtime is the round-1 incumbent ``option_first_buy`` (a response-side
guard that defers a premature checkout until the request-matching attribute
options are selected) plus one additional, provable guard on the request's
stated price ceiling.

Every WebShop instruction here carries a hard constraint of the form
``... and price lower than Y dollars``.  Products that cost more than ``Y``
can never be the target product, so any purchase that finalises such a
product has guaranteed-zero reward.  The incumbent guard could not touch
these cases because they usually expose no request-matching option at all
(it abstains rather than guess).  The price guard acts on the *selection*
instead:

* When the policy clicks a product ASIN on a search-results page and the
  listing's advertised price exceeds the request ceiling, the click is
  rewritten to the best in-ceiling product on the same results page
  (highest request-token overlap in the title, ties broken by the earlier
  -- i.e. more relevant -- listing position).
* When the policy emits the checkout click on a product page whose
  advertised (minimum) price exceeds the request ceiling, the click is
  rewritten to ``back to search`` so a provably-wrong purchase is never
  finalised.

The guard is a strict no-op on every episode whose every candidate product
is within budget and on every episode with no stated ceiling: the incumbent
checkout-deferral logic is then applied unchanged.  Any exception or
unexpected shape returns the response unchanged.
"""

import re
import json
import copy
import ast


# ---------------------------------------------------------------------------
# Message text extraction
# ---------------------------------------------------------------------------

def _text_of(msg):
    """Robustly extract text from an OpenAI-style message dict."""
    if not isinstance(msg, dict):
        return ''
    content = msg.get('content')
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces = []
        for part in content:
            if isinstance(part, dict):
                pieces.append(str(part.get('text', '')))
            else:
                pieces.append(str(part))
        return ''.join(pieces)
    return str(content)


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

_NUMBER_WORDS = {
    'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
    'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10',
    'eleven': '11', 'twelve': '12', 'thirteen': '13', 'fourteen': '14',
    'fifteen': '15', 'sixteen': '16', 'seventeen': '17', 'eighteen': '18',
    'nineteen': '19', 'twenty': '20', 'thirty': '30', 'forty': '40',
    'fifty': '50', 'sixty': '60', 'seventy': '70', 'eighty': '80',
    'ninety': '90', 'hundred': '100',
}

_DIGIT_LETTER = re.compile(r'(?<=[0-9])(?=[a-z])')
_LETTER_DIGIT = re.compile(r'(?<=[a-z])(?=[0-9])')
_SPLIT_NONALNUM = re.compile(r'[^a-z0-9]+')


def _tokenize(s):
    """Lower-case, split into alphanumeric tokens, normalise number words."""
    if not isinstance(s, str):
        s = str(s)
    s = s.lower()
    s = _DIGIT_LETTER.sub(' ', s)
    s = _LETTER_DIGIT.sub(' ', s)
    out = []
    for tok in _SPLIT_NONALNUM.split(s):
        if not tok:
            continue
        out.append(_NUMBER_WORDS.get(tok, tok))
    return out


_STOPWORDS = frozenset({
    'a', 'an', 'the', 'and', 'or', 'of', 'off', 'to', 'for', 'in', 'on',
    'at', 'by', 'with', 'from', 'into', 'onto', 'over', 'up', 'out',
    'is', 'are', 'was', 'were', 'am', 'be', 'been', 'being',
    'do', 'does', 'did', 'having', 'have', 'has', 'had',
    'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'she', 'they',
    'them', 'their', 'it', 'its',
    'this', 'that', 'these', 'those', 'there', 'here',
    'what', 'which', 'who', 'when', 'where', 'why', 'how', 'about',
    'as', 'so', 'such', 'very', 'more', 'most', 'much', 'many', 'few',
    'own', 'same', 'too', 'also', 'just', 'not', 'no', 'yes',
    'want', 'need', 'look', 'looking', 'find', 'finding', 'get',
    'getting', 'would', 'like', 'please', 'can', 'could', 'should',
    'will', 'price', 'prices', 'dollar', 'dollars', 'lower', 'than',
    'less', 'under', 'cheap', 'cost', 'buy', 'purchase', 'item',
    'product', 'items', 'things', 'new', 'some', 'any', 'all',
})

_PRICE_TOKEN = re.compile(r'^\d+\.\d\d$')


# ---------------------------------------------------------------------------
# Request extraction
# ---------------------------------------------------------------------------

_INSTRUCTION_SEG = re.compile(r'Instruction:\s*(?:\[SEP\])?\s*(.*?)\s*\[SEP\]', re.DOTALL)


def _request_text(text):
    if not isinstance(text, str):
        text = str(text)
    m = _INSTRUCTION_SEG.search(text)
    if m:
        return m.group(1)
    if 'Instruction:' in text:
        return text.split('Instruction:', 1)[1]
    return text


def _request_tokens(text):
    """Salient (non-stopword, non-price) tokens of the request description."""
    seg = _request_text(text)
    out = set()
    for t in _tokenize(seg):
        if t in _STOPWORDS:
            continue
        if _PRICE_TOKEN.match(t):
            continue
        out.add(t)
    return out


_CEILING = re.compile(r'(?:lower than|less than|under|below)\s*\$?\s*([0-9]+(?:\.[0-9]+)?)')


def _request_ceiling(text):
    """The numeric price ceiling stated in the request, or None."""
    seg = _request_text(text)
    if not isinstance(seg, str):
        return None
    m = _CEILING.search(seg.lower())
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Observation parsing
# ---------------------------------------------------------------------------

_CLICKABLES_OBJ = re.compile(r'\{.*\}', re.DOTALL)

_NAV = {
    'back to search', '< prev', 'next >', 'description', 'features',
    'reviews', 'buy now', 'search', 'back to results',
}


def _clickables(content):
    if not isinstance(content, str):
        return None
    marker = 'Available Actions'
    if marker not in content:
        return None
    tail = content[content.index(marker):]
    m = _CLICKABLES_OBJ.search(tail)
    if not m:
        return None
    try:
        data = ast.literal_eval(m.group(0))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    clicks = data.get('clickables', [])
    if not isinstance(clicks, list):
        return None
    return list(clicks)


_PRICE_LINE = re.compile(
    r'Price:\s*\$?\s*([0-9]+(?:\.[0-9]+)?)'
    r'(?:\s*to\s*\$?\s*([0-9]+(?:\.[0-9]+)?))?'
)


def _page_min_price(content):
    """Minimum advertised price on a product page, or None."""
    if not isinstance(content, str):
        return None
    prices = []
    for m in _PRICE_LINE.finditer(content):
        try:
            a = float(m.group(1))
            b = float(m.group(2)) if m.group(2) else a
        except Exception:
            continue
        prices.append(min(a, b))
    return min(prices) if prices else None


_ASIN = re.compile(r'^[a-z0-9]{10}$')


def _result_listings(content):
    """Parse a search-results page into a list of (asin, title, price | None).

    Returns None when the observation is not a paginated results page.
    """
    if not isinstance(content, str):
        return None
    if 'Total results' not in content:
        return None
    body = content.split('Next >', 1)[1] if 'Next >' in content else content
    body = body.split('Available Actions')[0]
    parts = [p.strip() for p in body.split('[SEP]') if p.strip()]
    out = []
    i = 0
    while i < len(parts):
        if _ASIN.match(parts[i].lower()):
            asin = parts[i].lower()
            title = parts[i + 1] if i + 1 < len(parts) else ''
            price = None
            if i + 2 < len(parts) and parts[i + 2].startswith('$'):
                try:
                    price = float(parts[i + 2][1:])
                except Exception:
                    price = None
            out.append((asin, title, price))
            i += 3
        else:
            i += 1
    return out if out else None


def _item_groups(content):
    """Group advertised attribute options under their group headers.

    Returns a list of (header, [option, ...]) for a product page, else None.
    """
    cl = _clickables(content)
    if cl is None:
        return None
    lower_cl = set()
    for c in cl:
        try:
            lower_cl.add(str(c).lower())
        except Exception:
            return None
    if 'buy now' not in lower_cl:
        return None
    if '< Prev' in content:
        seg = content.split('< Prev', 1)[1]
    elif 'Instruction:' in content:
        seg = content.split('Instruction:', 1)[1]
    else:
        seg = content
    seg = seg.split('Price:', 1)[0]
    parts = [p.strip() for p in seg.split('[SEP]') if p.strip()]
    groups = []
    current = None
    for i, p in enumerate(parts):
        k = p.lower()
        if k in lower_cl and k not in _NAV:
            if current is None:
                current = ['_', []]
                groups.append(current)
            current[1].append(p)
        else:
            nxt = parts[i + 1].lower() if i + 1 < len(parts) else None
            if nxt is not None and nxt in lower_cl and nxt not in _NAV:
                current = [k, []]
                groups.append(current)
    return [(h, opts) for (h, opts) in groups if opts]


# ---------------------------------------------------------------------------
# Matching / selection
# ---------------------------------------------------------------------------

def _option_match(option, req):
    ot = [t for t in _tokenize(option) if t not in _STOPWORDS]
    if not ot:
        return (0, 0.0)
    count = sum(1 for t in ot if t in req)
    return (count, count / len(ot))


def _title_overlap(title, req):
    tt = [t for t in _tokenize(title) if t not in _STOPWORDS]
    return sum(1 for t in tt if t in req)


def _best_inceiling(listings, req, ceiling, exclude):
    """Highest-overlap in-ceiling listing, ties broken by earlier position."""
    best = None
    best_key = None
    for pos, (asin, title, price) in enumerate(listings):
        if asin == exclude:
            continue
        if price is None or price > ceiling:
            continue
        key = (_title_overlap(title, req), -pos)
        if best_key is None or key > best_key:
            best_key = key
            best = asin
    return best


def _best_option(options, req):
    best = None
    best_score = (0, 0.0, 0)
    for opt in options:
        mc, frac = _option_match(opt, req)
        score = (mc, frac, len(opt) if isinstance(opt, str) else 0)
        if best is None or score > best_score:
            best = opt
            best_score = score
    return (best, best_score)


def _decide_option(content, req, selected):
    """Best still-needed request-matching option, or None (incumbent logic)."""
    groups = _item_groups(content)
    if not groups:
        return None
    needs = []
    for _header, options in groups:
        sel_in_group = [o for o in options
                        if isinstance(o, str) and o.lower() in selected]
        if any(_option_match(o, req)[0] > 0 for o in sel_in_group):
            continue
        best, score = _best_option(options, req)
        if best is not None and score[0] > 0 and str(best).lower() not in selected:
            needs.append((score[0], score[1], len(best) if isinstance(best, str) else 0, best))
    if not needs:
        return None
    chosen = None
    chosen_key = None
    for need in needs:
        key = need[:3]
        if chosen_key is None or key > chosen_key:
            chosen_key = key
            chosen = need[3]
    return chosen


# ---------------------------------------------------------------------------
# History helpers
# ---------------------------------------------------------------------------

def _selected_values(public_history):
    selected = set()
    if not public_history:
        return selected
    for msg in public_history:
        if not isinstance(msg, dict):
            continue
        calls = msg.get('tool_calls')
        if not isinstance(calls, list):
            continue
        for call in calls:
            if not isinstance(call, dict):
                continue
            fn = call.get('function')
            if not isinstance(fn, dict):
                continue
            args = fn.get('arguments')
            parsed = None
            if isinstance(args, str):
                try:
                    parsed = json.loads(args)
                except Exception:
                    parsed = None
            elif isinstance(args, dict):
                parsed = args
            if not isinstance(parsed, dict):
                continue
            if 'value' in parsed:
                try:
                    selected.add(str(parsed['value']).strip().lower())
                except Exception:
                    continue
    return selected


def _parse_args(call):
    if not isinstance(call, dict):
        return None
    fn = call.get('function')
    if not isinstance(fn, dict):
        return None
    args = fn.get('arguments')
    parsed = None
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
        except Exception:
            parsed = None
    elif isinstance(args, dict):
        parsed = args
    return parsed if isinstance(parsed, dict) else None


def _rewrite_value(response, old_lower, new_value):
    """Deep-copy response, replacing any click argument equal to old_lower."""
    new_response = copy.deepcopy(response)
    new_args = json.dumps({'value': new_value})
    for call in new_response.get('tool_calls', []):
        if not isinstance(call, dict):
            continue
        fn = call.get('function')
        if not isinstance(fn, dict):
            continue
        parsed = _parse_args(call)
        if not parsed:
            continue
        value = parsed.get('value')
        if isinstance(value, str) and value.strip().lower() == old_lower:
            fn['arguments'] = new_args
    return new_response


_CHECKOUT = 'buy now'


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------

class AgentHarness:
    """Price-ceiling guard layered on the incumbent option-first guard."""

    def __init__(self, public_context=None):
        self.public_context = public_context

    def transform_request(self, request):
        # Identity: this mechanism is purely response-side.
        return request

    def transform_response(self, response, public_history):
        """Apply the price-ceiling guard, then the option-first guard."""
        try:
            if not isinstance(response, dict):
                return response
            if not public_history:
                return response

            last_obs = None
            for msg in public_history:
                txt = _text_of(msg)
                if 'Available Actions' in txt:
                    last_obs = txt
            if last_obs is None:
                return response

            req = _request_tokens(last_obs)
            if not req:
                for msg in public_history:
                    txt = _text_of(msg)
                    if 'Instruction:' in txt:
                        req = _request_tokens(txt)
                        if req:
                            break
            if not req:
                return response

            tool_calls = response.get('tool_calls')
            if not isinstance(tool_calls, list) or not tool_calls:
                return response

            # ---- Guard 1: hard price ceiling -----------------------------
            ceiling = _request_ceiling(last_obs)
            if ceiling is not None:
                listings = _result_listings(last_obs)
                page_min = _page_min_price(last_obs)
                clickables_lower = None
                cl = _clickables(last_obs)
                if cl is not None:
                    try:
                        clickables_lower = set(str(c).lower() for c in cl)
                    except Exception:
                        clickables_lower = None

                for call in tool_calls:
                    parsed = _parse_args(call)
                    if not parsed:
                        continue
                    value = parsed.get('value')
                    if not isinstance(value, str):
                        continue
                    v = value.strip().lower()

                    # (a) checkout a product page that is provably over budget
                    if (v == _CHECKOUT and page_min is not None
                            and page_min > ceiling):
                        if clickables_lower and 'back to search' in clickables_lower:
                            return _rewrite_value(response, _CHECKOUT, 'back to search')

                    # (b) open a results listing that is provably over budget
                    if listings is not None and _ASIN.match(v):
                        hit = [p for p in listings if p[0] == v]
                        if hit and hit[0][2] is not None and hit[0][2] > ceiling:
                            target = _best_inceiling(listings, req, ceiling, exclude=v)
                            if target is not None:
                                return _rewrite_value(response, v, target)

            # ---- Guard 2: incumbent option-first checkout deferral -------
            buy_calls = False
            for call in tool_calls:
                parsed = _parse_args(call)
                if not parsed:
                    continue
                value = parsed.get('value')
                if isinstance(value, str) and value.strip().lower() == _CHECKOUT:
                    buy_calls = True
                    break
            if not buy_calls:
                return response

            selected = _selected_values(public_history)
            chosen = _decide_option(last_obs, req, selected)
            if chosen is None:
                return response
            return _rewrite_value(response, _CHECKOUT, chosen)
        except Exception:
            return response


def register():
    return None
