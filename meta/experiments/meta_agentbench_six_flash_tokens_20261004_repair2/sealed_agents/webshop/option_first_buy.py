"""Option-first checkout guard runtime.

Mechanism (single hypothesis): deferring a premature purchase until
request-matching attribute options are selected.

This is a response-side guard. On a product page that still exposes
attribute options described by the task request (e.g. a colour, size or
style), a policy that jumps straight to the checkout action can complete
the purchase in a default/unselected configuration and miss the required
variant. When the policy emits a click whose value is the checkout action
on such a page, we rewrite that click into a click on the best-matching
option instead, so the purchase can only happen after the option is
selected.

The guard is self-disabling: once a matching option has already been
selected (as observed from the prior action history), later checkout
clicks pass through unchanged. Any unexpected shape or error falls back
to returning the response unchanged.
"""

import re
import json
import copy
import ast

# ---------------------------------------------------------------------------
# Message text extraction
# ---------------------------------------------------------------------------


def _text_of(msg):
    """Robustly extract text from an OpenAI-style message dict.

    ``content`` may be a str, None, or a list of parts. List parts that are
    dicts contribute their ``text`` field; any other part is stringified.
    Returns '' when no text can be recovered.
    """
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
    """Lower-case, split into alphanumeric tokens, and normalise number words.

    A space boundary is first inserted between any adjacent digit and letter
    (in both directions) so that mixed alphanumeric codes (e.g. a model or
    style code) are split consistently. English number words are mapped to
    their digit form so that numeric choice options compare equal to the
    numeric forms used in requests.
    """
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


# Common English function words plus shopping filler that carry no
# discriminating content for matching an option against a request.
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

# A decimal price such as 12.34.
_PRICE_TOKEN = re.compile(r'^\d+\.\d\d$')


# ---------------------------------------------------------------------------
# Request extraction
# ---------------------------------------------------------------------------

_INSTRUCTION_SEG = re.compile(r'Instruction:\s*(?:\[SEP\])?\s*(.*?)\s*\[SEP\]', re.DOTALL)


def _request_tokens(text):
    """Extract the salient (non-stopword, non-price) tokens of the request.

    The request description is taken from the 'Instruction:' field of the
    observation. If the field markers are malformed, everything after
    'Instruction:' is used. Function words and price literals are dropped so
    that only content-bearing description terms remain.
    """
    if not isinstance(text, str):
        text = str(text)
    m = _INSTRUCTION_SEG.search(text)
    if m:
        seg = m.group(1)
    elif 'Instruction:' in text:
        seg = text.split('Instruction:', 1)[1]
    else:
        seg = text
    toks = _tokenize(seg)
    out = set()
    for t in toks:
        if t in _STOPWORDS:
            continue
        if _PRICE_TOKEN.match(t):
            continue
        out.add(t)
    return out


# ---------------------------------------------------------------------------
# Observation parsing
# ---------------------------------------------------------------------------

_CLICKABLES_OBJ = re.compile(r'\{.*\}', re.DOTALL)

# Navigation / page-chrome labels that are clickable but are not attribute
# options and must never be treated as selectable choices.
_NAV = {
    'back to search', '< prev', 'next >', 'description', 'features',
    'reviews', 'buy now', 'search', 'back to results',
}


def _clickables(content):
    """Return the list of clickable labels advertised in the observation.

    The labels follow an 'Available Actions' marker in a Python-literal dict.
    Returns None when the block is absent or cannot be parsed safely.
    """
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


def _item_groups(content):
    """Group the attribute options advertised on a product page.

    Returns a list of ``(header, [option, ...])`` for every attribute group
    that has at least one option, or None when the page is not a product
    page (i.e. no checkout action is available).

    The page body is isolated (after the back-navigation marker, before the
    price line) and split on the field separator. A part is an option when it
    matches an advertised clickable label; a part is an attribute-group header
    when the immediately following part is such a label.
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

    # Only a product page offers the checkout action.
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
            # An advertised, non-navigation label -> an attribute option.
            if current is None:
                current = ['_', []]
                groups.append(current)
            current[1].append(p)
        else:
            nxt = parts[i + 1].lower() if i + 1 < len(parts) else None
            if nxt is not None and nxt in lower_cl and nxt not in _NAV:
                # This part introduces a run of options -> group header.
                current = [k, []]
                groups.append(current)

    return [(h, opts) for (h, opts) in groups if opts]


# ---------------------------------------------------------------------------
# Option / request matching
# ---------------------------------------------------------------------------


def _option_match(option, req):
    """Score how well an option string matches the request token set.

    Returns ``(match_count, matched_fraction)`` where ``match_count`` is the
    number of content-bearing option tokens that appear in the request, and
    ``matched_fraction`` normalises that count by the option's token length.
    """
    ot = [t for t in _tokenize(option) if t not in _STOPWORDS]
    if not ot:
        return (0, 0.0)
    count = 0
    for t in ot:
        if t in req:
            count += 1
    return (count, count / len(ot))


def _best_option(options, req):
    """Pick the option with the strongest request match.

    Ordering key is ``(match_count, matched_fraction, option_length)`` with a
    strict '>' comparison while iterating in the given order, so ties resolve
    to the earliest option. Returns ``(option_or_None, score_tuple)``.
    """
    best = None
    best_score = (0, 0.0, 0)
    for opt in options:
        mc, frac = _option_match(opt, req)
        score = (mc, frac, len(opt) if isinstance(opt, str) else 0)
        if best is None or score > best_score:
            best = opt
            best_score = score
    return (best, best_score)


# ---------------------------------------------------------------------------
# Prior-selection tracking
# ---------------------------------------------------------------------------


def _selected_values(public_history):
    """Collect the option values the policy has already clicked.

    Selections are not echoed back in observations, so the only reliable
    signal is the sequence of prior tool-call arguments. Returns a set of
    lower-cased, stripped string values.
    """
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


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------


def _decide(obs_content, req, selected):
    """Choose an option that must be selected before checkout, or None.

    For every attribute group that is not already satisfied by a
    request-matching prior selection, compute the best-matching option. If any
    group still needs a matching option to be chosen, return the single most
    confident such option (largest match_count, then fraction, then length).
    """
    groups = _item_groups(obs_content)
    if not groups:
        return None

    needs = []
    for _header, options in groups:
        sel_in_group = [o for o in options
                        if isinstance(o, str) and o.lower() in selected]
        already_satisfied = False
        for o in sel_in_group:
            if _option_match(o, req)[0] > 0:
                already_satisfied = True
                break
        if already_satisfied:
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
# Runtime
# ---------------------------------------------------------------------------

_CHECKOUT_VALUE = 'buy now'


class AgentHarness:
    """Response-side guard that defers checkout until options are selected."""

    def __init__(self, public_context=None):
        self.public_context = public_context

    def transform_request(self, request):
        # Identity: this mechanism is purely response-side.
        return request

    def transform_response(self, response, public_history):
        """Rewrite a premature checkout click into an option-selection click.

        Any error or unexpected shape yields the response unchanged.
        """
        try:
            if not isinstance(response, dict):
                return response
            if not public_history:
                return response

            # Last observation that advertises available actions.
            last_obs = None
            for msg in public_history:
                txt = _text_of(msg)
                if 'Available Actions' in txt:
                    last_obs = txt
            if last_obs is None:
                return response

            req = _request_tokens(last_obs)
            if not req:
                # Fall back to the first message that carries the instruction.
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

            # Identify checkout clicks among the emitted calls.
            buy_calls = []
            for call in tool_calls:
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
                value = parsed.get('value')
                if isinstance(value, str) and value.strip().lower() == _CHECKOUT_VALUE:
                    buy_calls.append(call)

            if not buy_calls:
                return response

            # Only prior history counts as an already-made selection.
            selected = _selected_values(public_history)

            chosen = _decide(last_obs, req, selected)
            if chosen is None:
                return response

            new_response = copy.deepcopy(response)
            new_args = json.dumps({'value': chosen})
            for call in new_response.get('tool_calls', []):
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
                value = parsed.get('value')
                if isinstance(value, str) and value.strip().lower() == _CHECKOUT_VALUE:
                    fn['arguments'] = new_args
            return new_response
        except Exception:
            return response


def register():
    return None
