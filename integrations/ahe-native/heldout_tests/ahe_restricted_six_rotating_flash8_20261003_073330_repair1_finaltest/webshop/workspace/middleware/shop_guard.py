"""Episode-local WebShop action repair, turn budget and factual feedback.

Evidence base: the ten iteration-3 traces plus the per-task harness records of
all 50 evaluated tasks.  Three defects of the previous revision are fixed here.

1. WRONG PAGE CLASSIFIER.  The search-bar page -- the page reached with
   ``click_action(value="back to search")`` -- prints no ``Price:`` field and no
   ``Page N (Total results: M)`` marker, so the old classifier called it a
   description/features/reviews view and then ordered
   ``NEXT ACTION: ... click_action(value="< prev")``, an action that page does
   not offer.  Traces obeyed and burned turns on a ``< prev`` that returned the
   same page.  The search bar is now recognised by ``'has_search_bar': True``.

2. ENDLESS PAGING.  The old search note ended with "Use 'next >' when no
   listing on this page fits", and the skill text repeated it.  Every long
   episode then clicked ``next >`` until the ~20 turn episode budget was gone,
   including on blank pages 6..10 (the store returns 50 results, ten per page,
   and pages after the fifth hold nothing).  Four of the ten analysed tasks and
   14 of the 50 evaluated tasks ended with harness status ``task limit
   reached`` and therefore scored 0.  The note no longer recommends paging: it
   reports the real page count, states plainly when a page holds no product,
   and prefers a fresh search over more paging.

3. BUYING THE WRONG THING.  Purchases happened with the instruction's own words
   absent from the title (a baby toothbrush for "blue toothbrushes that help
   maintain oral hygiene", an apricot-strawberry bar for "banana hemp cereal")
   or with an attribute the instruction names left unset.  Product pages now
   report which instruction words the title carries and which it does not, and
   which option values the instruction names; a ``buy now`` that leaves a named
   option unset is redirected to that option first.

 The episode budget is visible in every note, and the last turn of an episode
 never ends without a purchase: a purchase is scored, an unfinished episode is
 worth zero.

Iteration-4 evidence (ten traced failures) changed two of those rules.

4. A PAGE THE INSTRUCTION DOES NOT DESCRIBE IS NOT BOUGHT.  The iteration-4
   revision stopped ordering a retreat whenever instruction words were absent
   from a title, because store titles omit filler adjectives (``high``,
   ``performance``, ``screen``, ``cruelty``, ``long``, ``lasting``,
   ``relaxed``) and the warning fired on correct pages, driving a retreat loop
   that never bought.  The iteration-5 traces show the opposite loss: every
   traced failure bought a look-alike on turn 2-4 -- a black earbud for
   ``silver ... earbuds``, a ``10x6.5ft`` backdrop for a ``6 foot by 4 foot``
   one, a lone foundation for a ``mattress set that has a 4" foundation``, a
   generic toothpaste for ``fluoride free ... two times the repair``.  In every
   case the instruction's own numbers or attribute values were simply not on
   the page.  The test is therefore narrowed to the instruction's
   *discriminating* content -- its numbers, its named attribute values and its
   non-filler words -- while the filler adjectives stay ignorable.  A page that
   fails the test is a look-alike: a ``buy now`` on it is served as
   ``back to search`` instead, and the rejected page is recorded so it is never
   bought later in the same episode.

5. EXPLORATION IS BOUNDED, PURCHASE IS GUARANTEED.  A retreat from a product
   page is served at most ``max_retreats`` times per episode; past the deadline,
   on a page already visited, or past that cap it is served as
   ``click_action(value="buy now")`` while the open page is inside the budget.
       A wrong purchase and no purchase both score zero, so the purchase is never
       the worse choice and it removes the never-bought class of loss outright.
   
   Iteration-6 evidence (ten traced failures) exposed three defects in the option
   matching itself, all fixed below.
   
   6. NUMBER AND SIZE FOLDING.  Instructions spell numbers and sizes out ("ten
      pack", "3 feet", "extra large") while product pages print digits and
      abbreviations ("10 pack", "x-large").  The old matcher compared the raw
      words, so for "ten pack ... 3 feet" it labelled the WRONG option -- "3 feet
      (3 pack)" -- as the instruction's value and never saw "3 feet (10 pack)".
      Number words now fold onto digits, size modifiers onto abbreviations, and the
      option matcher is order aware: a number must bind to the unit the
      instruction binds it to ("10 feet (3 pack)" is not "3 feet (10 pack)").
   
   7. PER-GROUP BEST OPTION.  The old rule only acted when exactly ONE option in a
      group consisted entirely of instruction words, which meant the groups whose
      values carry units or a second number ("16 ounce (pack of 24)",
      "38 | 40 | 41 mm", "40x32 inch") were never recognised and the purchase went
      through with the option unset.  Each group now has a scored best option; when
      it is uniquely best, buying with it unset or clicking a weaker option of the
      same group is served as a click on that option.  Near-identical spellings
      ("3 feet (10 pack)" / "3 feet (10-pack)") are collapsed so they cannot fake a
      tie.
   
   8. LOOK-ALIKE VERDICT NARROWED, BUY REDIRECT REMOVED.  The verdict is a fuzzy
      word test and its word-count rule fired on correct pages; three traced
      failures obeyed the resulting "buy now -> back to search" redirect and looped
      without ever buying.  A page is now a look-alike only when it omits one of the
      instruction's numbers or carries none of its words, and a buy is never
      rewritten into a retreat -- the verdict is reported, and the agent's own
            retreats stay bounded by ``max_retreats`` and the turn deadline.
"""

from __future__ import annotations

import re

from nexau.archs.main_sub.execution.hooks import HookResult, Middleware

_SEP = "[SEP]"
_BUY = "buy now"
_BACK = "back to search"

# Clickable values that are navigation chrome rather than product attributes.
_NAV = frozenset([
    "back to search", "< prev", "next >", "next", "< next", "description",
    "features", "reviews", "buy now", "search", "back",
])

# Clickable values that move between result pages of the same search.
_PAGING = frozenset(["next >", "next", "< next", "< prev"])

_STOP = frozenset([
    "a", "an", "the", "and", "or", "for", "with", "in", "on", "to", "at", "by",
    "from", "as", "of", "is", "are", "be", "it", "its", "this", "that", "these",
    "those", "i", "im", "m", "s", "my", "me", "we", "you", "your", "one", "some",
    "want", "wanted", "would", "like", "look", "looking", "need", "needed",
    "find", "finding", "buy", "purchase", "get", "please", "brand", "price",
    "prices", "lower", "less", "cheaper", "cheap", "under", "below", "than",
    "dollar", "dollars", "usd", "pack", "packs", "pcs", "pc", "set", "sets",
    "new", "size", "sizes", "color", "colour", "colours", "style", "styles",
    "name", "flavor", "flavour", "flavors", "flavours", "type", "model", "item",
    "am", "our", "will", "can", "could", "should", "has", "have", "do", "does",
    "did", "not", "but", "also", "all", "any", "well", "very", "sure", "good",
    "must", "choose", "pair", "two", "interested", "searching", "shopping",
    "hoping", "prefer", "really", "particular", "specifically", "gonna",
    "something", "anything",
])

# Instruction words that store titles routinely omit.  These are the filler
# adjectives an instruction is padded with; their absence from a product page
# is not evidence that the page is a different product.  Everything else the
# instruction names -- the object type, the numbers, the attribute values --
# has to be on the page.
_SOFT = frozenset([
    "high", "performance", "screen", "quality", "long", "last", "lasting",
    "easy", "assemble", "assembly", "cruelty", "free", "relax", "relaxed",
    "fit", "fitting", "accessory", "premium", "durable", "strong", "light",
    "lightweight", "comfort", "comfortable", "soft", "cute", "fashion",
    "fashionable", "modern", "great", "best", "ideal", "nice", "beautiful",
    "stylish", "trendy", "classic", "simple", "basic", "extra", "super",
])

_PRICE_PHRASE = re.compile(
    r"(?:\s*,?\s*and\s+)?(?:\s*a\s+)?(?:\s*price\s+)?(?:\s*is\s+)?"
    r"(?:lower|less|cheaper|under|below)\s+than\s+\$?\s*[0-9][0-9.,]*\s*"
    r"(?:us\s+)?dollars?",
    re.IGNORECASE)

_PRICE_LIMIT = re.compile(
    r"(?:lower|less|cheaper|under|below)\s+than\s+\$?\s*([0-9][0-9.,]*)\s*"
    r"(?:us\s+)?dollars?",
    re.IGNORECASE)

_NUMBER = re.compile(r"[0-9]+(?:[.,][0-9]+)?")
_WORD = re.compile(r"[a-z0-9]+(?:[.][0-9]+)?")
_WS = re.compile(r"\s+")
_CLICKABLES = re.compile(r"'clickables'\s*:\s*\[(.*?)\]", re.DOTALL)
_ITEM = re.compile(r"'([^']*)'|\"([^\"]*)\"")
_ASIN = re.compile(r"^b[0-9a-z]{9}$")
_PAGE_NO = re.compile(r"page\s+([0-9]+)\s*\(\s*total results", re.IGNORECASE)
_TOTAL = re.compile(r"total results\s*:\s*([0-9]+)", re.IGNORECASE)
_HAS_BAR = re.compile(r"'has_search_bar'\s*:\s*True")

# Measurement/unit words.  An option value carries units the instruction may
# not repeat ("16 ounce (pack of 24)" for "a 16 ounce drink"), so a unit word
# that the instruction does not name is not evidence against the option.
_UNIT = frozenset([
    "inch", "inches", "ft", "feet", "foot", "mm", "cm", "oz", "ounce", "ounces",
    "fl", "lb", "lbs", "gram", "grams", "kg", "ml", "liter", "litre", "meters",
    "meter", "metre", "metres", "wide", "width", "deep", "depth", "tall",
    "height", "diameter", "length", "count", "pcs", "pc", "piece", "pieces",
    "pack", "packs", "set", "sets", "size", "sizes", "model", "models",
    "style", "styles", "color", "colour", "colours", "flavor", "flavour",
    "flavors", "flavours", "name", "type", "item", "assorted", "multi",
])

# An instruction spells numbers out ("ten pack") while the store prints digits
# ("10 pack").  The token comparison used to treat those as different words, so
# the HDMI option "3 feet (10 pack)" read as "not carried here: ten" and the
# matcher instead labelled the wrong option "3 feet (3 pack)" as the
# instruction's value.  Folding small spelled numbers onto their digits removes
# that whole class of mismatch.
_NUMWORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
    "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18",
    "nineteen": "19", "twenty": "20", "thirty": "30", "forty": "40",
    "fifty": "50", "sixty": "60", "seventy": "70", "eighty": "80",
    "ninety": "90", "single": "1", "pair": "2", "dozen": "12",
}

# Size modifiers the store abbreviates: an instruction "extra large" is the
# option "x-large", not the option "large".
_ALIAS = {"extra": "x", "double": "xx", "triple": "xxx"}


def _norm(value):
    return _WS.sub(" ", value.strip().lower())


def _stem(token):
    """Very light suffix folding so 'bagels' matches 'bagel' and vice versa."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _tokens(value):
    return _WORD.findall(value.lower())


def _content_tokens(value):
    kept = []
    for token in _tokens(value):
        if token in _STOP:
            continue
        stemmed = _stem(token)
        if stemmed in _STOP:
            continue
        if len(token) > 1 or token.isdigit():
            kept.append(stemmed)
    return kept


def _fold_token(token):
    """Fold a spelled notation onto the way the store writes it.

    Number words become digits ('ten' -> '10') and size modifiers become the
    store's abbreviation ('extra large' -> 'x-large'), so the instruction and
    the option value share the same token.
    """
    return _NUMWORDS.get(token, _ALIAS.get(token, token))


def _value_tokens(value):
    """Tokens of a value with spelled numbers folded onto digits.

    Unit words are kept even when they are also stop words (``pack``), so a
    number keeps the unit it binds to: ``3 feet (10 pack)`` stays
    ``['3', 'feet', '10', 'pack']`` and can be told apart from
    ``10 feet (3 pack)``.
    """
    kept = []
    for token in _tokens(value):
        folded = _fold_token(token)
        if folded in _UNIT:
            kept.append(folded)
            continue
        if folded in _STOP or _stem(folded) in _STOP:
            continue
        if len(folded) > 1 or folded.isdigit() or folded == "x":
            kept.append(_stem(folded))
    return kept


_PAIR_UNITS = frozenset([
    "inch", "inches", "ft", "feet", "foot", "mm", "cm", "oz", "ounce", "ounces",
    "fl", "lb", "lbs", "gram", "grams", "kg", "ml", "liter", "litre", "meters",
    "meter", "metre", "metres", "mile", "miles", "yard", "yards", "pack",
    "packs", "count", "pcs", "px", "pc", "piece", "pieces", "set", "sets",
    "watt", "watts", "volt", "volts", "ml", "cl",
])


def _measure_pairs(tokens):
    """(number, unit) bindings of a token stream, e.g. ('3', 'feet').

    Used to tell ``3 feet (10 pack)`` from ``10 feet (3 pack)``: both carry the
    tokens 3, feet, 10, pack, but only the first binds the numbers to the units
    the instruction uses.
    """
    pairs = []
    for index, token in enumerate(tokens):
        if not any(ch.isdigit() for ch in token):
            continue
        if index + 1 < len(tokens) and tokens[index + 1] in _PAIR_UNITS:
            pairs.append((token, tokens[index + 1]))
    return pairs


def _loose(left, right):
    """True when two content words are the same word, plural or not."""
    if left == right:
        return True
    short, long = (left, right) if len(left) <= len(right) else (right, left)
    if len(short) < 4:
        return False
    return long.startswith(short) and len(long) - len(short) <= 2


def _to_float(raw):
    try:
        return float(raw.replace(",", ""))
    except Exception:
        return None


def _fmt(value):
    if value is None:
        return "?"
    return "%.2f" % value


def _clip(text, size):
    flat = _WS.sub(" ", text).strip()
    if len(flat) <= size:
        return flat
    return flat[: size - 3] + "..."


def _strip_price_phrase(value):
    cleaned = _PRICE_PHRASE.sub(" ", value)
    cleaned = _WS.sub(" ", cleaned).strip()
    return cleaned.strip(" ,;:-")


def _clean(value):
    """Drop the backslash escapes the environment puts around quotes.

    The action dictionary is printed with escaped quotes (``\\'back to search\\'``)
    and one apostrophe inside an option value is enough to shift every later
    quote pair, which silently loses all options that follow it.
    """
    return value.replace("\\'", "'").replace('\\"', '"')


def _clickables(text):
    match = _CLICKABLES.search(_clean(text))
    if match is None:
        return []
    items = []
    # The environment quotes values with '...' but switches to "..." for any
    # value containing an apostrophe ("grandma's fruit basket"), so both
    # quoting styles have to be accepted or the list is silently truncated.
    for single, double in _ITEM.findall(match.group(1)):
        items.append(single if single else double)
    return [item for item in items if item]


def _asins(values):
    return [item for item in values if _ASIN.match(item)]


def _price_of(segment):
    """Smallest number printed in a `Price: ...` segment."""
    head = segment.split("Available Actions")[0].strip()
    if head.lower().startswith("price:"):
        head = head[6:]
    values = []
    for raw in _NUMBER.findall(head):
        value = _to_float(raw)
        if value is not None:
            values.append(value)
    if not values:
        return None
    return min(values)


def _groups(region, options):
    """Split the product-page header into (label, [option, ...]) groups.

    ``region`` runs from just after ``< prev`` up to the title, ``options`` maps
    a normalised clickable to its verbatim form.  Every option clickable is
    attached to the most recent non-option part, which is the group label the
    environment prints (``color``, ``size``, ``flavor name``, ...).
    """
    groups = []
    for part in region:
        key = _norm(_clean(part))
        if key in options:
            if not groups:
                groups.append(("", []))
            groups[-1][1].append(options[key])
        else:
            groups.append((_clean(part), []))
    return [(label, opts) for label, opts in groups if opts]


_SEARCH_MARKER = re.compile(r"^page\s+[0-9]+\s*\(\s*total results", re.IGNORECASE)


def _is_search_page(parts):
    for part in parts:
        if _SEARCH_MARKER.match(part.strip()):
            return True
    return False


class ShopGuardMiddleware(Middleware):
    """Deterministic action repair, turn budget and factual feedback."""

    def __init__(self, max_redirects=20, turn_budget=20, final_push=True,
                 max_retreats=3, deadline_turns=4):
        self.max_redirects = max_redirects
        self.max_turns = turn_budget
        self.final_push = bool(final_push)
        # How many times an episode may leave a product page for the result
        # list.  Beyond it the retreat is served as 'buy now', because an
        # episode that never buys is worth zero while a purchase can still
        # score.
        self.max_retreats = max(0, int(max_retreats))
        self.retreats = 0
        self.deadline_turns = max(0, int(deadline_turns))
        # instruction state (episode-local)
        self.instruction = ""
        self.instruction_words = []
        self.instruction_bag = []
        self.limit = None
        self.numbers = []
        # current page state
        self.page = ""
        self.product_key = ""
        self.title = ""
        self.price = None
        self.over_budget = False
        self.groups = []
        self.auto_options = []
        self.mentioned_options = []
        self.matched_options = {}
        self.option_group = {}
        self.option_sig = {}
        self.opt_scores = {}
        self.desired = {}
        self.clicked_options = set()
        self.asins = []
        self.page_number = None
        self.total_pages = None
        self.invalid_click = False
        # bookkeeping
        self.turn = 0
        self.repairs = 0
        self.seen = {}
        self.paged = 0
        self.last_click = ""
        # Product keys of pages already judged to be look-alikes.  A page on
        # this list is never bought in the same episode.
        self.rejected = set()

    # ------------------------------------------------------------------ hooks

    def before_tool(self, hook_input):
        raw = hook_input.tool_input
        if not isinstance(raw, dict):
            return HookResult.no_changes()
        try:
            if hook_input.tool_name == "search_action":
                return self._repair_search(dict(raw))
            if hook_input.tool_name == "click_action":
                return self._repair_click(dict(raw))
        except Exception:
            # A repair must never abort an episode.
            return HookResult.no_changes()
        return HookResult.no_changes()

    def after_tool(self, hook_input):
        if hook_input.tool_name not in ("search_action", "click_action"):
            return HookResult.no_changes()
        output = hook_input.tool_output
        if not isinstance(output, dict):
            return HookResult.no_changes()
        text = output.get("result")
        if not isinstance(text, str) or not text:
            return HookResult.no_changes()

        params = hook_input.tool_input if isinstance(hook_input.tool_input, dict) else {}
        try:
            self._read_instruction(text)
            self.turn += 1
            if hook_input.tool_name == "click_action":
                value = params.get("value")
                self.last_click = _norm(value) if isinstance(value, str) else ""
            else:
                self.last_click = ""

            note = self._refresh(text, params, hook_input.tool_name)
            if self._is_loop():
                note = note + [self._repeat_note()]
            lines = [self._header()] + note
        except Exception:
            return HookResult.no_changes()
        updated = dict(output)
        updated["result"] = text + "\n\n" + "\n".join(lines)
        return HookResult.with_modifications(tool_output=updated)

    # ------------------------------------------------------------ action gate

    def _repair_search(self, params):
        keywords = params.get("keywords")
        if not isinstance(keywords, str):
            return HookResult.no_changes()
        cleaned = _strip_price_phrase(keywords)
        if cleaned and cleaned != keywords:
            params["keywords"] = cleaned
            return HookResult.with_modifications(tool_input=params)
        return HookResult.no_changes()

    def _repair_click(self, params):
        value = params.get("value")
        if not isinstance(value, str):
            return HookResult.no_changes()
        key = _norm(value)

        # The store returns 50 results, ten per page, and pages after the last
        # one are blank: paging on from the final result page can only waste a turn.
        if key in ("next >", "next", "< next") and self.page == "search" \
                and self.page_number is not None and self.total_pages is not None \
                and self.page_number >= self.total_pages:
            params["value"] = _BACK
            return HookResult.with_modifications(tool_input=params)

        # The last turn of an episode is never spent without buying: with no turn
        # left to follow up, only 'buy now' can still score.
        if self.final_push and self.page == "product" and key != _BUY \
                and self.page_ready() and self.turn >= self.max_turns - 1:
            if key in self.matched_options:
                self.clicked_options.add(key)
            params["value"] = _BUY
            return HookResult.with_modifications(tool_input=params)

        # Leaving a product page for the result list is the action behind the
        # episode-ending loops: the agent obeyed a note that ordered a retreat,
        # re-searched, re-opened a listing, and the twenty turns ran out with
        # nothing bought, which scores zero.  Exploration is therefore bounded:
        # past the deadline, on a page already visited, or after
        # ``max_retreats`` retreats, the retreat is served as 'buy now' because
        # a purchase can still score while an unfinished episode cannot.
        if key in (_BACK, "< prev") and self.page == "product" \
                and self.page_ready() and self.repairs < self.max_redirects:
            if self._verdict()[0]:
                # The agent is itself leaving a page that cannot be the product:
                # remember it so a later re-open is called out as a repeat.
                self.rejected.add(self.product_key)
            left = self.max_turns - self.turn
            revisited = self.seen.get("product|" + self.product_key, 0) >= 2
            if revisited or left <= self.deadline_turns or self.retreats >= self.max_retreats:
                pending = self._pending_options()
                self.repairs += 1
                if pending and left > 1:
                    self.clicked_options.add(_norm(pending[0][1]))
                    params["value"] = pending[0][1]
                else:
                    params["value"] = _BUY
                return HookResult.with_modifications(tool_input=params)
            self.retreats += 1

        # An option click is remembered for this product page.  When the group
        # has one clearly best-matching option and the agent clicked a weaker
        # option of the same group, the agent's click is served as that best
        # option instead: the traced HDMI failure clicked "3 feet (3 pack)" and
        # "3 pack" for a "ten pack ... 3 feet" instruction and bought the wrong
        # configuration.
        if key not in _NAV and key != _BUY and self.page == "product":
            label = self.option_group.get(key)
            if label is not None:
                best = self.desired.get(label)
                if best is not None and _norm(best) != key \
                        and self.opt_scores.get(key, 0.0) \
                        < self.opt_scores.get(_norm(best), 0.0):
                    self.clicked_options.add(_norm(best))
                    params["value"] = best
                    return HookResult.with_modifications(tool_input=params)
                self.clicked_options.add(key)
                return HookResult.no_changes()

        if key == _BUY and self.page == "product" and self.over_budget \
                and self.repairs < self.max_redirects:
            self.repairs += 1
            params["value"] = _BACK
            return HookResult.with_modifications(tool_input=params)

        # NOTE: a "look-alike" verdict no longer rewrites a buy into a retreat.
        # The verdict is a fuzzy word/number test, and a false positive used to
        # turn a correct purchase into a retreat loop that never bought (three
        # of the ten iteration-5 traced failures).  The verdict is reported, but
        # the decision to leave a page stays with the agent and is bounded by the
        # retreat cap and the turn deadline above.

        # An option the instruction names must not be left unset before buying.
        if key == _BUY and self.page == "product":
            pending = self._pending_options()
            if pending:
                self.clicked_options.add(_norm(pending[0][1]))
                params["value"] = pending[0][1]
                return HookResult.with_modifications(tool_input=params)

        # Clicking a group label selects nothing; use the option instead.
        if key not in _NAV and self.page == "product":
            for label, _opts in self.groups:
                if label and _norm(label) == key:
                    named = [opt for lab, opt in self._pending_options()
                             if _norm(lab) == key]
                    if len(named) == 1:
                        self.clicked_options.add(_norm(named[0]))
                        params["value"] = named[0]
                        return HookResult.with_modifications(tool_input=params)
                    break

        return HookResult.no_changes()

    def page_ready(self):
        """True when the open product page can be bought from."""
        return self.price is not None and not self.over_budget

    def _is_clicked(self, opt):
        """True when this option, or a same-signature spelling of it, was clicked."""
        key = _norm(opt)
        if key in self.clicked_options:
            return True
        sig = self.option_sig.get(key)
        if sig is None:
            return False
        return any(self.option_sig.get(k) == sig for k in self.clicked_options)

    def _pending_options(self):
        return [(label, opt) for label, opt in self.auto_options
                if not self._is_clicked(opt)]

    # -------------------------------------------------------------- loop gate

    def _fingerprint(self):
        if self.page == "search":
            return "search|" + ",".join(sorted(self.asins))
        if self.page == "product":
            return "product|" + self.product_key
        if self.page == "home":
            return "home|" + self.last_click
        return self.page + "|" + self.product_key + "|" + self.last_click

    def _is_loop(self):
        if self.last_click == "< prev":
            # Going back is deliberate, never a loop.
            return False
        key = self._fingerprint()
        count = self.seen.get(key, 0) + 1
        self.seen[key] = count
        return count >= 2

    def _repeat_note(self):
        lines = [
            "REPEAT: this exact page has already been shown to you, so an action that "
            "leads back here changes nothing."
        ]
        if self.page == "search":
            if not self.asins:
                lines.append(
                    "There are no products on this page and no further result page: "
                    "click_action(value=\"back to search\") and search with DIFFERENT "
                    "keywords."
                )
            else:
                lines.append(
                    "Click a listing on this page now, or click_action(value=\"back to "
                    "search\") and use DIFFERENT keywords. Do not page again."
                )
        elif self.page == "product":
            pending = self._pending_options()
            look_alike, reasons = self._verdict()
            if look_alike:
                text = ('This page is NOT the product the instruction describes ('
                        + "; ".join(reasons[:2]) + '): click_action(value="back to '
                        'search") and open a DIFFERENT listing.')
                if self.product_key in self.rejected:
                    text = ("You have already left this listing once and it is still "
                            "not the product asked for. " + text)
                lines.append(text)
            elif pending:
                lines.append(
                    'The instruction names option "' + pending[0][1]
                    + '": click it, then click_action(value="buy now").'
                )
            elif self.groups:
                lines.append(
                    'Every value the instruction names is already set on this page: '
                    'click_action(value="buy now") now, and do not leave the page again.'
                )
            else:
                lines.append(
                    'Click_action(value="buy now") now - you have already seen everything '
                    'this page offers.'
                )
        elif self.page == "home":
            lines.append(
                "You are at the search bar: use search_action with keywords that differ "
                "from the keywords you sent before."
            )
        else:
            lines.append('Click_action(value="< prev") to return to the product page.')
        return "\n".join(lines)

    # ------------------------------------------------------------- page state

    def _reset(self):
        self.instruction_words = []
        self.instruction_bag = []
        self.limit = None
        self.numbers = []
        self.page = ""
        self.product_key = ""
        self.title = ""
        self.price = None
        self.over_budget = False
        self.groups = []
        self.auto_options = []
        self.mentioned_options = []
        self.matched_options = {}
        self.option_group = {}
        self.option_sig = {}
        self.opt_scores = {}
        self.desired = {}
        self.clicked_options = set()
        self.asins = []
        self.page_number = None
        self.total_pages = None
        self.invalid_click = False
        self.turn = 0
        self.repairs = 0
        self.retreats = 0
        self.seen = {}
        self.paged = 0
        self.last_click = ""
        self.rejected = set()

    def _read_instruction(self, text):
        parts = text.split(_SEP)
        if len(parts) < 3:
            return
        candidate = parts[1].strip()
        # The search-bar page echoes the label "Instruction:" as parts[1]; the
        # real instruction is the next field.
        if not candidate or candidate.rstrip(":").lower() in ("instruction", ""):
            candidate = parts[2].strip() if len(parts) > 2 else ""
        if not candidate or candidate.rstrip(":").lower() == "instruction":
            return
        candidate = _WS.sub(" ", candidate).strip()
        if self.instruction and candidate != self.instruction:
            # A new episode in the same process: drop all previous state.
            self._reset()
        if self.instruction == candidate:
            return
        self.instruction = candidate
        core = _strip_price_phrase(candidate)
        bag = _value_tokens(core)
        words = []
        for token in bag:
            if len(token) > 1 and token not in words:
                words.append(token)
        self.instruction_words = words
        self.instruction_bag = bag
        numbers = []
        for raw in _NUMBER.findall(core):
            value = _to_float(raw)
            if value is not None and value not in numbers:
                numbers.append(value)
        for raw_token in _tokens(core):
            folded = _fold_token(raw_token)
            if folded == raw_token or not folded.isdigit():
                continue
            value = _to_float(folded)
            if value is not None and value not in numbers:
                numbers.append(value)
        self.numbers = numbers
        match = _PRICE_LIMIT.search(candidate)
        if match is not None:
            self.limit = _to_float(match.group(1))

    def _word_in_instruction(self, token):
        folded = _stem(token)
        for word in self.instruction_words:
            if _loose(folded, _stem(word)):
                return True
        return False

    def _number_in_instruction(self, raw):
        value = _to_float(raw)
        if value is None:
            return False
        for known in self.numbers:
            if abs(known - value) < 1e-6:
                return True
        return False

    def _refresh(self, text, params, tool_name):
        parts = [part.strip() for part in text.split(_SEP)]
        clickables = _clickables(text)
        self.asins = _asins(clickables)
        self.invalid_click = (
            tool_name == "click_action"
            and self.last_click not in _PAGING
            and self.last_click != _BACK
            and self.last_click != ""
        )

        price_index = -1
        for index, part in enumerate(parts):
            if part.lower().startswith("price:"):
                price_index = index
                break

        if _HAS_BAR.search(text):
            self._enter_home()
            return self._home_note(tool_name)
        if price_index < 0:
            if _is_search_page(parts):
                return self._enter_search(parts, params, tool_name)
            # A genuine description/features/reviews view of the open product.
            self.page = "detail"
            return self._detail_note()
        return self._enter_product(parts, clickables, price_index)

    # ------------------------------------------------------------------ notes

    def _header(self):
        left = self.max_turns - self.turn
        head = "[SHOP CHECK] turn " + str(self.turn) + " of about " + str(self.max_turns)
        if left <= 3:
            return (head + " | DEADLINE: about " + str(left) + " turns left. Buy the best "
                    "matching product you can reach NOW - an episode that never buys "
                    "scores 0.")
        if left <= 6:
            return head + " | " + str(left) + " turns left: stop exploring and get to 'buy now'."
        return head

    def _clear_page(self):
        """Drop the per-page state.  ``asins`` belongs to the page being parsed
        and is managed by the caller."""
        self.product_key = ""
        self.title = ""
        self.price = None
        self.over_budget = False
        self.groups = []
        self.auto_options = []
        self.mentioned_options = []
        self.matched_options = {}
        self.option_group = {}
        self.option_sig = {}
        self.opt_scores = {}
        self.desired = {}
        self.clicked_options = set()
        self.page_number = None
        self.total_pages = None

    def _enter_home(self):
        self.page = "home"
        self._clear_page()
        self.asins = []
        if self.last_click == _BACK:
            # A deliberate return to the search bar: this is not a fresh search.
            self.paged = 0

    def _home_note(self, tool_name):
        lines = ["page=SEARCH BAR (start page)"]
        if self.invalid_click:
            lines.append(
                "NOTE: your click value was not an action of the previous page, so the "
                "environment ran a SEARCH for it instead. Nothing was opened."
            )
        lines.append(
            "Only 'search' works here and '< prev' does not exist on this page. Use "
            "search_action with keywords that DIFFER from the last ones: the same words "
            "return the identical page."
        )
        return lines

    def _enter_search(self, parts, params, tool_name):
        previous = self.page
        previous_number = self.page_number if previous == "search" else None
        number = None
        for part in parts:
            match = _PAGE_NO.search(part)
            if match is not None:
                number = int(match.group(1))
                break
        total = None
        for part in parts:
            match = _TOTAL.search(part)
            if match is not None:
                total = int(match.group(1))
                break
        self._clear_page()
        self.page = "search"
        if tool_name == "search_action":
            # A fresh search starts a fresh result list.
            self.paged = 0
        elif previous_number is not None and number is not None and number > previous_number:
            self.paged += 1
        if number is not None:
            self.page_number = number
        if total is not None:
            pages = (total + 9) // 10
            self.total_pages = pages if pages > 0 else None
        return self._search_note(params)

    def _search_note(self, params):
        lines = []
        if not self.asins:
            lines.append("page=SEARCH RESULTS - NO PRODUCTS ON THIS PAGE")
            lines.append(
                "This result list is exhausted: 'next >' keeps returning blank pages. Do "
                "NOT click 'next >' again. Click_action(value=\"back to search\") and use "
                "DIFFERENT keywords (another product noun, fewer words)."
            )
            return lines
        where = "page " + str(self.page_number) if self.page_number is not None else "a page"
        if self.total_pages:
            where = where + " of about " + str(self.total_pages)
        lines.append("page=SEARCH RESULTS (" + where + ", ten titles per page)")
        lines.append(self._budget_line())
        keywords = ""
        if isinstance(params, dict):
            raw = params.get("keywords")
            if isinstance(raw, str):
                keywords = raw
        if keywords:
            lines.append("keywords actually sent: " + _clip(keywords, 120))
        if self.instruction_words:
            lines.append(
                "instruction words a listing title should carry: "
                + ", ".join(self.instruction_words[:12])
            )
        if self.invalid_click:
            lines.append(
                "NOTE: your click value was not an action of the previous page, so the "
                "environment ran a SEARCH for it instead. These listings are not the "
                "product you meant to open."
            )
        lines.append(
            "Next step: click the ASIN of the title that carries the product type plus the "
            "most of those words, when its price is inside the budget. Do not settle for "
            "an accessory, a look-alike or another size/model."
        )
        if self.paged >= 3:
            lines.append(
                "You have paged " + str(self.paged) + " times on this search; another "
                "'next >' only shows more of the same 50 results, so searching with "
                "different keywords is now the better move."
            )
        return lines

    def _detail_note(self):
        return [
            "page=DETAIL VIEW (description / features / reviews)",
            "This view only re-prints text about the product already open; it changes "
            "nothing. Click_action(value=\"< prev\") to return to the product page.",
        ]

    def _enter_product(self, parts, clickables, price_index):
        previous_key = self.product_key if self.page == "product" else ""
        self.page = "product"
        self.asins = []
        self.page_number = None
        self.total_pages = None
        prev_index = -1
        for index, part in enumerate(parts):
            if part.lower() == "< prev":
                prev_index = index
                break
        title = parts[price_index - 1] if price_index > 0 else ""
        region = parts[prev_index + 1: price_index - 1] if prev_index >= 0 else []
        options = {}
        for item in clickables:
            if _norm(item) in _NAV:
                continue
            options[_norm(_clean(item))] = _clean(item)
        self.groups = _groups(region, options)
        self.title = title
        self.product_key = _norm(title)[:100]
        if self.product_key != previous_key:
            self.clicked_options = set()
        self.price = _price_of(parts[price_index])
        self.over_budget = (
            self.price is not None
            and self.limit is not None
            and self.price > self.limit
        )
        self._collect_options()
        return self._product_note()

    def _option_score(self, opt):
        """(score, shared, carries_number, specific) of one option value.

        ``shared`` counts the instruction tokens the option carries, with
        multiplicity; ``score`` subtracts half a point for every option token
        the instruction does not name, so ``3 feet (3 pack)`` loses to
        ``3 feet (10 pack)`` for a ``ten pack ... 3 feet`` instruction.
        ``carries_number`` reports whether the option carries a number the
        instruction names; ``specific`` reports whether it shares a distinctive
        word (a colour or size word, not filler), which is enough to make a
        single-word value like ``green`` the group's value.
        """
        tokens = _value_tokens(opt)
        if not tokens:
            return 0.0, 0, False, False
        pool = list(self.instruction_bag)
        shared = 0
        carries_number = False
        specific = False
        noise = 0
        for token in tokens:
            if token in pool:
                pool.remove(token)
                shared += 1
                if any(ch.isdigit() for ch in token):
                    carries_number = True
                elif token not in _SOFT and token not in _UNIT and len(token) > 2:
                    specific = True
            elif token not in _UNIT and not any(ch.isdigit() for ch in token):
                noise += 1
        # Order-aware check: a number the instruction binds to a unit must be
        # bound to that same unit here, or the option mis-uses the number
        # ('10 feet (3 pack)' for a 'ten pack ... 3 feet' instruction).
        instruction_pairs = set(_measure_pairs(self.instruction_bag))
        bound_numbers = {number for number, _unit in instruction_pairs}
        misbound = 0
        for number, unit in _measure_pairs(tokens):
            if number in bound_numbers and (number, unit) not in instruction_pairs:
                misbound += 1
        return shared - 0.5 * noise - misbound, shared, carries_number, specific

    def _collect_options(self):
        self.auto_options = []
        self.mentioned_options = []
        self.matched_options = {}
        self.option_group = {}
        self.option_sig = {}
        self.opt_scores = {}
        self.desired = {}
        for label, opts in self.groups:
            scored = []
            seen = {}
            for opt in opts:
                key = _norm(opt)
                self.matched_options[key] = opt
                self.option_group[key] = label
                score, shared, carries_number, specific = self._option_score(opt)
                self.opt_scores[key] = score
                sig = tuple(_value_tokens(opt))
                self.option_sig[key] = sig
                if sig in seen:
                    # '3 feet (10 pack)' and '3 feet (10-pack)' are the same
                    # value spelled two ways; two entries must not form a tie.
                    continue
                seen[sig] = opt
                scored.append((score, shared, carries_number, specific, opt))
            if not scored:
                continue
            scored.sort(key=lambda item: item[0], reverse=True)
            best_score, best_shared, best_number, best_specific, best_opt = scored[0]
            ties = sum(1 for score, _sh, _n, _sp, _o in scored
                       if abs(score - best_score) < 1e-9)
            for score, _sh, _n, _sp, opt in scored[:3]:
                self.mentioned_options.append((label, opt, score))
            # A single clearly-related option is the group's value: it carries
            # an instruction number, or shares two instruction words / one
            # distinctive word, and nothing else ties it.
            if ties == 1 and best_score >= 1.0 \
                    and (best_number or best_shared >= 2 or best_specific):
                self.desired[label] = best_opt
                self.auto_options.append((label, best_opt))

    def _budget_line(self):
        if self.limit is None:
            return "budget: none stated in the instruction"
        return "budget: price must stay below $" + _fmt(self.limit)

    def _title_report(self):
        page_tokens = _value_tokens(self.title)
        for _label, opts in self.groups:
            for opt in opts:
                for token in _value_tokens(opt):
                    page_tokens.append(token)
        present = []
        missing = []
        for word in self.instruction_words:
            folded = _stem(word)
            found = False
            for token in page_tokens:
                if _loose(_stem(token), folded):
                    found = True
                    break
            if found:
                present.append(word)
            else:
                missing.append(word)
        return present, missing

    def _page_numbers(self):
        """Every number printed in the open product's title and option values."""
        raw_parts = [_NUMBER.findall(self.title)]
        for _label, opts in self.groups:
            for opt in opts:
                raw_parts.append(_NUMBER.findall(opt))
        values = []
        for group in raw_parts:
            for raw in group:
                value = _to_float(raw)
                if value is not None:
                    values.append(value)
        return values

    def _missing_numbers(self):
        """Instruction numbers that the open product page does not carry.

        A size, a dimension, a pack count or a model number the instruction
        names is part of the product's identity; a listing that does not print
        it is a different variant (``10x6.5ft`` is not ``6 foot by 4 foot``).
        """
        if not self.numbers:
            return []
        page = self._page_numbers()
        return [value for value in self.numbers
                if not any(abs(value - other) < 1e-6 for other in page)]

    def _verdict(self):
        """Is the open product page a look-alike?  Returns (bool, [reason, ...]).

        Only high-precision evidence counts: the instruction names a number
        (a size, a dimension, a pack count, a model) that the page does not
        print anywhere, or the page carries not one instruction word.  A page
        that merely omits a descriptive adjective is NOT a look-alike: store
        titles drop those routinely, and the word-count rule behind the old
        verdict fired on correct pages and drove retreat loops that never
        bought.
        """
        if self.over_budget:
            return True, ["the printed price is above the instruction's limit"]
        present, missing = self._title_report()
        reasons = []
        if not present:
            reasons.append("not one word of the instruction is on this page")
        missing_numbers = self._missing_numbers()
        if missing_numbers:
            reasons.append(
                "the instruction's number(s) "
                + ", ".join(_fmt(value) for value in missing_numbers)
                + " are not printed on this page"
            )
        return bool(reasons), reasons

    def _product_note(self):
        lines = ["page=PRODUCT PAGE"]
        lines.append("price $" + _fmt(self.price) + " | " + self._budget_line())
        if self.over_budget:
            lines.append(
                "WARNING: the price is above the instruction's limit, so buying this "
                "product can never satisfy the instruction. Click_action(value=\"back to "
                "search\") and pick a cheaper listing."
            )
            return lines

        present, missing = self._title_report()
        total = len(present) + len(missing)
        if total:
            lines.append(
                "instruction words this page carries: " + ", ".join(present[:8])
                + " (" + str(len(present)) + " of " + str(total) + ")"
            )
            if missing:
                lines.append("NOT carried here: " + ", ".join(missing[:6]))
        look_alike, reasons = self._verdict()
        if look_alike:
            lines.append(
                "VERDICT: this page is NOT the product the instruction describes ("
                + "; ".join(reasons[:2]) + "). The instruction is written from the "
                "product it wants, so the right listing carries its numbers and its "
                'attribute values. click_action(value="back to search") and open a '
                "DIFFERENT listing - buying this page scores 0."
            )
        else:
            lines.append(
                "VERDICT: this page carries the instruction's object type, numbers "
                "and attribute values, so it is the product to buy."
            )

        clicked = [_norm(opt) for _label, opt in self.auto_options]
        pending = self._pending_options()
        if self.groups:
            shown = []
            for label, _opts in self.groups:
                shown.append("'" + label + "'" if label else "(unnamed group)")
            lines.append("attribute groups on this page: " + ", ".join(shown))
            for label, _opts in self.groups:
                best = self.desired.get(label)
                if best is None:
                    lines.append(
                        "group '" + (label or "unnamed") + "': no option matches any word "
                        "or number of the instruction. If the instruction names a value "
                        "for this group, this listing cannot provide it."
                    )
                elif self._is_clicked(best):
                    lines.append(
                        "group '" + (label or "unnamed") + "': the instruction's value \""
                        + best + "\" is already clicked."
                    )
                else:
                    lines.append(
                        "group '" + (label or "unnamed") + "': the instruction's value is "
                        'option "' + best + '": click_action(value="' + best
                        + '") before \'buy now\'.'
                    )
            if pending:
                lines.append(
                    'The instruction still names an unset option: click_action(value="'
                    + pending[0][1] + '") now.'
                )
            elif clicked:
                lines.append(
                    "Every instruction-named option is set, so the next action must be "
                    "click_action(value=\"buy now\")."
                )
            else:
                lines.append(
                    "The page never shows what is already selected, so keep your own list "
                    "of clicked options."
                )
        else:
            if look_alike:
                lines.append(
                    "no attribute group on this page and it is not the product the "
                    'instruction describes: click_action(value="back to search") and '
                    "open a listing that carries "
                    + ", ".join(self.instruction_words[:4]) + "."
                )
            else:
                lines.append(
                    "no attribute group on this page: there is nothing to select, so the "
                    "only remaining action is click_action(value=\"buy now\")."
                )
        return lines
