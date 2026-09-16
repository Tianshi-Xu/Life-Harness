"""Per-turn constraint ledger runtime with repetition-triggered progress guidance.

Mechanism: before each model call, derive a compact checklist from the task
instruction and the current page, then append it to the last observation.
The checklist restates the price bound, the explicit attribute values the
instruction asks for (color / size / quantity), whether each value is offered
on the current page, and whether it has already been clicked on this page.

On top of that, when the model's most recent action re-clicks a value it has
already clicked earlier in the transcript (a product id, or a paging /
navigation control), a short progress note is appended after the checklist:
how often the value was clicked, which products were opened so far, which
requested value is still unsatisfied on the current page, and one concrete
redirect (open an unopened visible result, page on, or change one keyword).

Everything is computed statelessly from the public message list: no task ids,
no product tables, no cross-episode state.

A further stateless check ("Variant check") fires when the last observation is
a product page whose clickables offer a not-yet-clicked option carrying every
digit-bearing identifier of the budget-stripped instruction (decimals,
alphanumeric variant codes, '#n' shades, dimension marks).
"""

import ast
import copy
import json
import re

# --------------------------------------------------------------------------
# generic vocabularies
# --------------------------------------------------------------------------

_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
    "dozen": 12,
}
_NUMBER_RE = re.compile(
    r"\b(" + "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True)) + r")\b"
)

# Common color names (generic shopping vocabulary only).
_COLORS = frozenset({
    "black", "white", "gray", "grey", "red", "blue", "green", "yellow",
    "orange", "purple", "pink", "brown", "gold", "silver", "beige", "tan",
    "ivory", "teal", "navy", "maroon", "magenta", "turquoise", "violet",
    "lavender", "olive", "khaki", "aqua", "charcoal", "bronze", "copper",
    "indigo", "cyan", "crimson", "burgundy", "fuchsia",
})
# Words that are colors but also common object/flavor nouns: only trusted when
# they sit next to an explicit "color/colored" marker in the instruction.
_RISKY_COLORS = frozenset({
    "mint", "denim", "clay", "peach", "lilac", "cream", "coral", "rose",
    "plum", "sage", "sand", "chocolate", "mustard", "wine", "pearl",
})
_COLOR_HINTS = frozenset({"color", "colour", "colored", "coloured", "colors", "colours"})
_COLOR_MODS = ("dark", "light", "pale", "deep", "bright", "royal", "hot",
               "neon", "pastel", "metallic", "matte", "true")
_MULTI_RE = re.compile(r"\bmulti[\s-]*colou?red\b|\bmulti[\s-]*color\b")

# Units grouped into compatibility classes used for quantity matching.
_UNIT_CLASS = {
    "pack": "pack", "packs": "pack", "pk": "pack", "set": "pack",
    "sets": "pack", "count": "pack", "ct": "pack", "piece": "pack",
    "pieces": "pack", "pc": "pack", "pcs": "pack", "tablet": "pack",
    "tablets": "pack", "capsule": "pack", "capsules": "pack",
    "stick": "pack", "sticks": "pack", "pod": "pack", "pods": "pack",
    "strip": "pack", "strips": "pack", "box": "pack", "case": "pack",
    "bundle": "pack", "nos": "pack",
    "ounce": "ounce", "ounces": "ounce", "oz": "ounce", "floz": "ounce",
    "inch": "inch", "inches": "inch", "in": "inch",
    "foot": "foot", "feet": "foot", "ft": "foot",
    "pound": "pound", "pounds": "pound", "lb": "pound", "lbs": "pound",
    "gram": "gram", "grams": "gram", "kg": "gram", "mg": "gram",
    "ml": "liter", "liter": "liter", "liters": "liter", "litre": "liter",
    "litres": "liter", "milliliter": "liter", "milliliters": "liter",
    "watt": "watt", "watts": "watt", "volt": "volt", "volts": "volt",
    "light": "light", "lights": "light",
}
_UNIT_CLASSES = frozenset(_UNIT_CLASS.values())

_SIZE_TOKEN_MAP = {
    "small": "s", "medium": "m", "large": "l", "xsmall": "xs",
    "one": "1",
}
_SIZE_VOCAB = frozenset({
    "x", "xs", "s", "m", "l", "xl", "xxl", "xxxl", "xxs", "small",
    "medium", "large", "x-large", "xsmall", "x-small", "xx-large",
    "xxx-large", "plus", "wide",
})
_STANDALONE_SIZE_RE = re.compile(
    r"\b(?:\d{1,2}\s*x\s*-?\s*large|\d{1,2}xl|xxl|xxxl|xx-large|x-large|xxx-large|x-small|xs|xxs)\b"
)

_NAV = frozenset({
    "search", "back to search", "< prev", "prev", "previous", "next >",
    "next", "buy now", "description", "features", "reviews", "rating",
})
_PROD_ID_RE = re.compile(r"^b0[a-z0-9]{7,10}$", re.I)

# Attribute-group labels that can precede option values on a product page.
_LABEL_CATEGORY = {
    "color": "color", "colour": "color", "colors": "color",
    "colours": "color", "color name": "color", "colour name": "color",
    "size": "size", "sizes": "size", "size name": "size",
    "ring size": "size", "shoe size": "size",
    "style": "style", "styles": "style", "style name": "style",
    "pattern": "pattern", "patterns": "pattern",
    "pattern name": "pattern", "theme": "theme",
    "flavor": "flavor", "flavour": "flavor", "flavors": "flavor",
    "flavours": "flavor", "flavor name": "flavor", "flavour name": "flavor",
    "scent": "scent", "scent name": "scent", "scented": "scent",
    "material": "material", "material name": "material",
    "shape": "shape", "finish": "finish", "edition": "edition",
    "model": "model", "configuration": "configuration", "metal": "metal",
    "metal type": "metal", "gemstone": "gemstone", "count": "quantity",
    "quantity": "quantity", "unit count": "quantity",
    "number of pieces": "quantity", "item package quantity": "quantity",
    "weight": "weight", "capacity": "capacity", "length": "length",
    "width": "width", "height": "height", "diameter": "diameter",
    "number of lights": "quantity", "light source": "other",
    "wattage": "other", "voltage": "other", "bundle": "other",
    "option": "other", "options": "other", "type": "other",
}
# An instruction color that only lands inside one of these groups is a word
# inside another attribute's value, not a selectable color.
_INCOMPATIBLE_COLOR_GROUPS = frozenset({"flavor", "scent", "style", "pattern", "theme", "material"})

_BUDGET_STRIP_RES = [
    re.compile(r"between\s+\$?\s*[0-9][0-9,]*(?:\.[0-9]+)?\s+and\s+\$?\s*[0-9][0-9,]*(?:\.[0-9]+)?", re.I),
    re.compile(r"(?:and\s+)?price\s+(?:is\s+)?(?:lower|less|cheaper)\s+than\s+\$?\s*[0-9][0-9,]*(?:\.[0-9]+)?\s*(?:dollars?|bucks)?", re.I),
    re.compile(r"(?:lower|less|cheaper)\s+than\s+\$?\s*[0-9][0-9,]*(?:\.[0-9]+)?\s*(?:dollars?|bucks)?", re.I),
    re.compile(r"(?:no\s+more\s+than|not\s+more\s+than|under|below|within|for\s+under)\s+\$?\s*[0-9][0-9,]*(?:\.[0-9]+)?\s*(?:dollars?|bucks)?", re.I),
    re.compile(r"(?:over|above|more\s+than)\s+\$?\s*[0-9][0-9,]*(?:\.[0-9]+)?\s*(?:dollars?|bucks)?", re.I),
]
_PRICE_RE = re.compile(
    r"Price:\s*\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)"
    r"(?:\s*to\s*\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?))?"
)
_RESULT_COUNT_RE = re.compile(r"Total results", re.I)

_QUANTITY_PATTERNS = [
    (re.compile(r"([0-9]+(?:\.[0-9]+)?)[-\s]*(?:pack|packs|pk)\b", re.I), "pack"),
    (re.compile(r"\b(?:pack|set|box|case|bundle)\s+of\s+([0-9]+(?:\.[0-9]+)?)", re.I), "pack"),
    (re.compile(r"([0-9]+(?:\.[0-9]+)?)[-\s]*(?:count|ct|pcs?|pieces?|tablets?|capsules?|sticks?|pods?|strips?|nos?)\b", re.I), "pack"),
    (re.compile(r"([0-9]+(?:\.[0-9]+)?)[-\s]*(?:fl\.?\s*oz\.?|fluid\s*ounces?|ounces?|oz)\b", re.I), "ounce"),
    (re.compile(r"([0-9]+(?:\.[0-9]+)?)[-\s]*(?:inch(?:es)?)\b", re.I), "inch"),
    (re.compile(r"([0-9]+(?:\.[0-9]+)?)\s+in\b", re.I), "inch"),
    (re.compile(r"([0-9]+(?:\.[0-9]+)?)[-\s]*(?:foot|feet|ft)\b", re.I), "foot"),
    (re.compile(r"([0-9]+(?:\.[0-9]+)?)[-\s]*(?:pounds?|lbs?)\b", re.I), "pound"),
    (re.compile(r"([0-9]+(?:\.[0-9]+)?)[-\s]*(?:grams?|kg|mg)\b", re.I), "gram"),
    (re.compile(r"([0-9]+(?:\.[0-9]+)?)[-\s]*(?:ml|milliliters?|liters?|litres?)\b", re.I), "liter"),
    (re.compile(r"([0-9]+(?:\.[0-9]+)?)[-\s]*watts?\b", re.I), "watt"),
    (re.compile(r"([0-9]+(?:\.[0-9]+)?)[-\s]*volts?\b", re.I), "volt"),
    (re.compile(r"([0-9]+(?:\.[0-9]+)?)[-\s]*lights\b", re.I), "light"),
]
_SIZE_AFTER_RE = re.compile(r"\bsizes?\s+([a-z0-9][a-z0-9./ +x-]{0,20})", re.I)
# '[...] pack of 2 ounce [...]' means "a pack of [items of] 2 ounce", not a
# quantity of 2 packs; a measurement unit right after the number proves it.
_UNIT_TAIL_RE = re.compile(
    r"\s*-?\s*(?:fl\.?\s*oz\.?|fluid\s*ounces?|ounces?|oz|inch(?:es)?|foot|feet|ft|pounds?|lbs?|grams?|kg|mg|ml|milliliters?|liters?|litres?|watts?|volts?)\b",
    re.I,
)

_INSTRUCTION_RE = re.compile(r"Instruction:\s*\[SEP\]\s*(.*?)\s*\[SEP\]", re.DOTALL)


# --------------------------------------------------------------------------
# text helpers (pure)
# --------------------------------------------------------------------------

def _replace_number_words(text):
    return _NUMBER_RE.sub(lambda m: str(_NUMBER_WORDS[m.group(1)]), text)


def _norm(text):
    """Lowercase, fold '&' into 'and', expand number words, squeeze spaces."""
    s = text.lower().replace("&", " and ")
    s = _replace_number_words(s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _tokens(text):
    return re.findall(r"[a-z]+|[0-9]+(?:\.[0-9]+)?", text.lower())


def _canon_units(text):
    """Canonicalize packaging / measurement phrases inside a value or title."""
    s = _norm(text)
    s = re.sub(r"([0-9]+(?:\.[0-9]+)?)\s*-\s*(?:pack|packs|pk|count|ct|pcs?|pieces?|tablets?|capsules?|sticks?|pods?|strips?)\b", r"\1 pack", s)
    s = re.sub(r"([0-9]+(?:\.[0-9]+)?)\s*(?:pack|packs|pk|count|ct|pcs?|pieces?|tablets?|capsules?|sticks?|pods?|strips?|nos?)\b", r"\1 pack", s)
    s = re.sub(r"\b(?:pack|set|box|case|bundle)\s+of\s+([0-9]+(?:\.[0-9]+)?)", r"\1 pack", s)
    s = re.sub(r"([0-9]+(?:\.[0-9]+)?)\s*(?:fl\.?\s*oz\.?|fluid\s*ounces?|ounces?|oz)\b", r"\1 ounce", s)
    s = re.sub(r"([0-9]+(?:\.[0-9]+)?)\s*(?:inch(?:es)?|in\b)", r"\1 inch", s)
    s = re.sub(r"([0-9]+(?:\.[0-9]+)?)\s*(?:foot|feet|ft)\b", r"\1 foot", s)
    s = re.sub(r"([0-9]+(?:\.[0-9]+)?)\s*(?:pounds?|lbs?)\b", r"\1 pound", s)
    s = re.sub(r"([0-9]+(?:\.[0-9]+)?)\s*(?:grams?|kg|mg)\b", r"\1 gram", s)
    s = re.sub(r"([0-9]+(?:\.[0-9]+)?)\s*(?:ml|milliliters?|liters?|litres?)\b", r"\1 liter", s)
    s = re.sub(r"([0-9]+(?:\.[0-9]+)?)\s*watts?\b", r"\1 watt", s)
    s = re.sub(r"([0-9]+(?:\.[0-9]+)?)\s*volts?\b", r"\1 volt", s)
    s = re.sub(r"([0-9]+(?:\.[0-9]+)?)\s*lights?\b", r"\1 light", s)
    return re.sub(r"\s+", " ", s).strip()


def _canon_size(text):
    s = _norm(text)
    s = re.sub(r"([0-9]+)\s*x\s*-?\s*large\b", r"\1xl", s)
    s = re.sub(r"([0-9]+)\s*x\s*-?\s*l\b", r"\1xl", s)
    s = s.replace("xxx-large", "3xl").replace("xx-large", "xxl").replace("x-large", "xl")
    s = s.replace("x-small", "xs").replace("xsmall", "xs")
    return " ".join(_SIZE_TOKEN_MAP.get(t, t) for t in _tokens(s))


def _contains_all(required, candidate):
    """True when every token of `required` occurs in `candidate` (multiset)."""
    counts = {}
    for t in candidate:
        counts[t] = counts.get(t, 0) + 1
    for t in required:
        if counts.get(t, 0) <= 0:
            return False
        counts[t] -= 1
    return True


def _numbers(tokens):
    return [float(t) for t in tokens if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", t)]


def _unit_classes(tokens):
    return set(_UNIT_CLASS[t] for t in tokens if t in _UNIT_CLASS)


# --------------------------------------------------------------------------
# instruction parsing
# --------------------------------------------------------------------------

def _instruction_text(messages):
    text = None
    for m in messages:
        if not isinstance(m, dict):
            continue
        c = m.get("content")
        if not isinstance(c, str):
            continue
        for mt in _INSTRUCTION_RE.finditer(c):
            text = mt.group(1)
    return text


def _strip_budget(text):
    out = text
    for pat in _BUDGET_STRIP_RES:
        out = pat.sub(" ", out)
    return out


def _to_float(numstr):
    try:
        return float(numstr.replace(",", ""))
    except (TypeError, ValueError):
        return None


def _parse_budget(text):
    """Return {'lo': float|None, 'hi': float|None} or None."""
    m = re.search(r"between\s+\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s+and\s+\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)", text, re.I)
    if m:
        lo, hi = _to_float(m.group(1)), _to_float(m.group(2))
        if lo is not None and hi is not None:
            return {"lo": min(lo, hi), "hi": max(lo, hi)}
    for pat in (
        r"(?:lower|less|cheaper)\s+than\s+\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
        r"(?:no\s+more\s+than|not\s+more\s+than|under|below|within)\s+\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
    ):
        m = re.search(pat, text, re.I)
        if m:
            hi = _to_float(m.group(1))
            if hi is not None:
                return {"lo": None, "hi": hi}
    m = re.search(r"(?:over|above|more\s+than)\s+\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)", text, re.I)
    if m:
        lo = _to_float(m.group(1))
        if lo is not None:
            return {"lo": lo, "hi": None}
    return None


def _extract_colors(text):
    s = _norm(text)
    words = re.findall(r"[a-z]+", s)
    values = []
    for i, w in enumerate(words):
        if w == "multi":
            continue
        if w not in _COLORS and w not in _RISKY_COLORS:
            continue
        if w in _RISKY_COLORS:
            neighbors = words[max(0, i - 2):i + 3]
            if not any(n in _COLOR_HINTS for n in neighbors if n != w):
                continue
        phrase = [w]
        if i > 0 and words[i - 1] in _COLOR_MODS:
            phrase.insert(0, words[i - 1])
        values.append(" ".join(phrase))
    if _MULTI_RE.search(s):
        values.append("multicolor")
    out, seen = [], set()
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _extract_sizes(text):
    values = []
    norm_text = _norm(text)
    for m in _SIZE_AFTER_RE.finditer(norm_text):
        phrase = m.group(1)
        toks = _tokens(phrase)
        take = []
        for t in toks:
            if t.isdigit() or re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", t) or t in _SIZE_VOCAB:
                take.append(t)
            else:
                break
        if take:
            values.append(" ".join(take))
    for m in _STANDALONE_SIZE_RE.finditer(norm_text):
        values.append(m.group(0))
    out, seen = [], set()
    for v in values:
        key = tuple(_tokens(_canon_size(v)))
        if key and key not in seen:
            seen.add(key)
            out.append(v)
    return out


def _extract_quantities(text):
    norm_text = _norm(text)
    values = []
    for pat, cls in _QUANTITY_PATTERNS:
        for m in pat.finditer(norm_text):
            if m.group(0).strip().lower().startswith("pack of") and _UNIT_TAIL_RE.match(norm_text[m.end():]):
                continue
            values.append((m.group(0).strip(), cls))
    out, seen = [], set()
    for raw, _cls in values:
        toks = _tokens(_canon_units(raw))
        key = (tuple(sorted(_numbers(toks))), tuple(sorted(_unit_classes(toks))))
        if key[0] and key not in seen:
            seen.add(key)
            out.append(raw.strip())
    return out


def _extract_required(instruction):
    """Values explicitly asked for by the instruction, budget clause removed."""
    clean = _strip_budget(instruction)
    reqs = []
    for c in _extract_colors(clean):
        reqs.append({"kind": "color", "raw": c, "tokens": _tokens(_norm(c))})
    for s in _extract_sizes(clean):
        reqs.append({"kind": "size", "raw": s if s.startswith("size") else "size " + s, "tokens": _tokens(_canon_size(s))})
    for q in _extract_quantities(clean):
        toks = _tokens(_canon_units(q))
        reqs.append({"kind": "qty", "raw": q, "tokens": toks,
                     "numbers": _numbers(toks), "classes": _unit_classes(toks)})
    return reqs


# --------------------------------------------------------------------------
# page parsing
# --------------------------------------------------------------------------

def _parse_clickables(content):
    m = re.search(r"\{'has_search_bar':[^\n]*\}", content)
    if m:
        try:
            data = ast.literal_eval(m.group(0))
            vals = data.get("clickables")
            if isinstance(vals, list):
                return [str(v) for v in vals], bool(data.get("has_search_bar"))
        except (ValueError, SyntaxError):
            pass
    m = re.search(r"'clickables':\s*(\[[^\]]*\])", content)
    if m:
        try:
            vals = ast.literal_eval(m.group(1))
            if isinstance(vals, list):
                return [str(v) for v in vals], ("'has_search_bar': True" in content)
        except (ValueError, SyntaxError):
            pass
    return None, False


def _offered_values(clickables):
    out = []
    for v in clickables:
        low = v.strip().lower()
        if low in _NAV or low in _LABEL_CATEGORY or _PROD_ID_RE.match(low):
            continue
        out.append(v)
    return out


def _page_price(content):
    m = _PRICE_RE.search(content)
    if not m:
        return None
    lo = _to_float(m.group(1))
    hi = _to_float(m.group(2)) if m.group(2) else None
    if lo is None:
        return None
    return (lo, hi)


def _product_title(content):
    idx = content.find("Price:")
    region = content[:idx] if idx != -1 else content
    items = [x.strip() for x in region.split("[SEP]")]
    title = ""
    skip_next = False
    for it in items:
        if skip_next:
            skip_next = False
            continue
        low = it.lower()
        if not it or low in _NAV or low in _LABEL_CATEGORY:
            continue
        if "instruction:" in low:
            skip_next = True
            continue
        if _PROD_ID_RE.match(low) or len(it) < 8:
            continue
        title = it
    return title


def _value_groups(content, offered):
    """Map offered value (lowercase) -> attribute category, when inferable."""
    groups = {}
    if not offered:
        return groups
    if "Back to Search" in content:
        region = content[content.rfind("Back to Search"):]
    else:
        region = content
    idx = region.find("Price:")
    if idx != -1:
        region = region[:idx]
    lowvals = set(v.strip().lower() for v in offered)
    cur = None
    for it in [x.strip() for x in region.split("[SEP]")]:
        low = it.lower()
        if not low or low in _NAV:
            continue
        if low in lowvals:
            groups.setdefault(low, cur)
            continue
        cat = _LABEL_CATEGORY.get(low)
        if cat:
            cur = cat
            continue
        if len(it) <= 30 and not re.search(r"[0-9,().]", it) and re.fullmatch(r"[A-Za-z][A-Za-z /#&-]{0,29}", it):
            cur = "other"
            continue
        break
    return groups


# --------------------------------------------------------------------------
# value matching
# --------------------------------------------------------------------------

def _color_tokens_match(req_tokens, cand_tokens):
    if req_tokens == ["multicolor"]:
        return any(t.startswith("multi") for t in cand_tokens)
    return _contains_all(req_tokens, cand_tokens)


def _qty_match(req, cand_tokens):
    cand_nums = _numbers(cand_tokens)
    for n in req.get("numbers", []):
        if n not in cand_nums:
            return False
    classes = req.get("classes") or set()
    if not classes:
        return bool(req.get("numbers"))
    return bool(classes & _unit_classes(cand_tokens))


def _value_matches(req, raw_value, groups):
    if isinstance(req, dict):
        toks = _tokens(_canon_units(raw_value))
        if req["kind"] == "color":
            cat = groups.get(raw_value.strip().lower())
            if cat in _INCOMPATIBLE_COLOR_GROUPS:
                return False
            return _color_tokens_match(req["tokens"], _tokens(_norm(raw_value)))
        if req["kind"] == "size":
            return _contains_all(req["tokens"], _tokens(_canon_size(raw_value)))
        return _qty_match(req, toks)
    return False


def _value_in_title(req, title):
    if not title:
        return False
    if req["kind"] == "color":
        return _color_tokens_match(req["tokens"], _tokens(_norm(title)))
    if req["kind"] == "size":
        return _contains_all(req["tokens"], _tokens(_canon_size(title)))
    toks = _tokens(_canon_units(title))
    return _qty_match(req, toks)


def _best_candidate(req, candidates, groups):
    def key(v):
        exact = 0 if _same_value(req, v, groups) else 1
        return (exact, len(_tokens(v)), v.lower())
    return sorted(candidates, key=key)[0]


def _same_value(req, raw_value, groups):
    if req["kind"] == "color":
        return _tokens(_norm(raw_value)) == req["tokens"]
    if req["kind"] == "size":
        return _tokens(_canon_size(raw_value)) == req["tokens"]
    toks = _tokens(_canon_units(raw_value))
    return sorted(_numbers(toks)) == sorted(req["numbers"]) and _unit_classes(toks) == req["classes"]


def _obs_body(content):
    """Observation text with the 'Action: ...' echo stripped, so that two
    observations of the same page compare equal regardless of the click."""
    idx = content.find("Observation:")
    return content[idx:] if idx != -1 else content


def _clicked_values(messages):
    """Click values made since the last page change (page-local, stateless)."""
    clicked = []
    prev_tool = None
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role == "tool":
            body = _obs_body(m.get("content") or "")
            if prev_tool is not None and body != prev_tool:
                clicked = []
            prev_tool = body
        elif role == "assistant":
            for tc in m.get("tool_calls") or []:
                fn = (tc or {}).get("function") or {}
                if fn.get("name") != "click_action":
                    continue
                args = fn.get("arguments")
                val = None
                if isinstance(args, str):
                    try:
                        val = json.loads(args).get("value")
                    except (ValueError, AttributeError):
                        val = None
                elif isinstance(args, dict):
                    val = args.get("value")
                if isinstance(val, str) and val.strip():
                    clicked.append(val.strip())
    return clicked


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

def _money(x):
    return "$%.2f" % x


def _render_budget(budget, price):
    if budget is None:
        return []
    lo, hi = budget.get("lo"), budget.get("hi")
    if lo is not None and hi is not None:
        bound = "%s-%s" % (_money(lo), _money(hi))
        ok = price is not None and lo - 1e-9 <= price[0] and (price[1] or price[0]) <= hi + 1e-9
        over = price is not None and not ok
    elif hi is not None:
        bound = "under %s" % _money(hi)
        over = price is not None and price[0] > hi + 1e-9
    else:
        bound = "over %s" % _money(lo)
        over = price is not None and price[0] < lo - 1e-9
    if price is None:
        return ["Budget: must be %s (no price shown on this page)." % bound]
    if price[1] is not None and price[1] > price[0]:
        price_txt = "%s-%s" % (_money(price[0]), _money(price[1]))
    else:
        price_txt = _money(price[0])
    if over:
        return ["Budget: must be %s — page price %s: OVER BUDGET — do not buy this product; click 'Back to Search' and pick another."
                % (bound, price_txt)]
    return ["Budget: must be %s — page price %s: OK" % (bound, price_txt)]


def _render_values(reqs, offered, groups, title, clicked):
    lines = []
    for req in reqs:
        matches = [v for v in offered if _value_matches(req, v, groups)]
        raw = req["raw"]
        if matches:
            best = _best_candidate(req, matches, groups)
            hits = [c for c in clicked if _value_matches(req, c, groups)]
            if hits:
                lines.append("- %s: offered as '%s' — already clicked." % (raw, hits[0]))
            else:
                lines.append("- %s: offered as '%s' — not clicked yet: click '%s'." % (raw, best, best))
        elif _value_in_title(req, title):
            lines.append("- %s: already stated in the product title — no option needed." % raw)
        elif req["kind"] == "color" and any(
            _color_tokens_match(req["tokens"], _tokens(_norm(v)))
            for v in offered
        ):
            continue  # color word only appears inside another attribute's values
        else:
            lines.append("- %s: NOT offered on this page — do not buy this product; click 'Back to Search' and pick another." % raw)
    return lines


def _render_product(content, reqs, clicked):
    offered = _offered_values(_parse_clickables(content)[0] or [])
    groups = _value_groups(content, offered)
    title = _product_title(content)
    price = _page_price(content)
    return _render_values(reqs, offered, groups, title, clicked), price, offered


# --------------------------------------------------------------------------
# ledger assembly
# --------------------------------------------------------------------------

def _ledger_block(messages):
    instruction = _instruction_text(messages)
    if not instruction:
        return None
    last = None
    for m in reversed(messages):
        if isinstance(m, dict) and m.get("role") in ("tool", "user") and isinstance(m.get("content"), str):
            last = m
            break
    if last is None:
        return None
    content = last["content"]
    budget = _parse_budget(instruction)
    reqs = _extract_required(instruction)
    if not reqs and not budget:
        return None
    clickables, has_search_bar = _parse_clickables(content)
    clicked = _clicked_values(messages)
    is_product = ("Price:" in content) or (clickables is not None and not has_search_bar and not _RESULT_COUNT_RE.search(content))

    lines = ["[Constraint check]"]
    if is_product:
        value_lines, price, offered = _render_product(content, reqs, clicked)
        lines += _render_budget(budget, price)
        if reqs:
            lines.append("Requested values: " + "; ".join(r["raw"] for r in reqs) + ".")
            lines += value_lines
            lines.append("Gate: click 'buy now' only after every requested value is clicked and the budget is OK.")
    else:
        lines += _render_budget(budget, None)
        if reqs:
            lines.append("Requested values: " + "; ".join(r["raw"] for r in reqs) + ".")
            lines.append("Open a result whose title matches ALL requested values, then click the matching options before buying.")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# progress guidance (repetition-triggered)
# --------------------------------------------------------------------------

# Navigation controls for which a repeat may still be intentional (paging).
_REPEATABLE_NAV = frozenset({"back to search", "next >", "< prev"})
_BLOCK_MARKERS = ("\n\n[Constraint check]", "\n\n[Progress check]", "\n\nVariant check:")


def _norm_click(value):
    return re.sub(r"\s+", " ", value).strip().lower()


def _click_trace(messages):
    """Every click_action value so far, in message order, normalized, repeats kept."""
    trace = []
    for m in messages:
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            fn = (tc or {}).get("function") or {}
            if fn.get("name") != "click_action":
                continue
            args = fn.get("arguments")
            val = None
            if isinstance(args, str):
                try:
                    val = json.loads(args).get("value")
                except (ValueError, AttributeError):
                    val = None
            elif isinstance(args, dict):
                val = args.get("value")
            if isinstance(val, str) and val.strip():
                trace.append(_norm_click(val))
    return trace


def _last_action_name(messages):
    """Name of the most recent tool call, or None."""
    name = None
    for m in messages:
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            fn = (tc or {}).get("function") or {}
            if isinstance(fn.get("name"), str):
                name = fn["name"]
    return name


def _opened_ids(trace):
    """Product-id-like values clicked so far, in order, deduplicated."""
    ids, seen = [], set()
    for v in trace:
        if _PROD_ID_RE.match(v) and v not in seen:
            seen.add(v)
            ids.append(v)
    return ids


def _latest_results_clickables(messages):
    """Clickables of the most recent search-results observation, or None."""
    for m in reversed(messages):
        if not isinstance(m, dict) or m.get("role") not in ("tool", "user"):
            continue
        content = m.get("content")
        if not isinstance(content, str) or not _RESULT_COUNT_RE.search(content):
            continue
        clickables, _has_bar = _parse_clickables(content)
        if clickables:
            return clickables
    return None


def _unmet_values(messages, content):
    """Raw requested values still unsatisfied on the current product page."""
    instruction = _instruction_text(messages)
    if not instruction:
        return []
    reqs = _extract_required(instruction)
    if not reqs:
        return []
    offered = _offered_values(_parse_clickables(content)[0] or [])
    groups = _value_groups(content, offered)
    title = _product_title(content)
    clicked = _clicked_values(messages)
    unmet = []
    for req in reqs:
        if any(_value_matches(req, v, groups) for v in offered):
            if not any(_value_matches(req, c, groups) for c in clicked):
                unmet.append("%s (not clicked)" % req["raw"])
        elif _value_in_title(req, title):
            continue
        elif req["kind"] == "color" and any(
            _color_tokens_match(req["tokens"], _tokens(_norm(v))) for v in offered
        ):
            continue
        else:
            unmet.append("%s (not offered)" % req["raw"])
    return unmet


def _progress_block(messages):
    """Guidance block when the latest action re-clicks an already-clicked value."""
    if not isinstance(messages, list):
        return None
    trace = _click_trace(messages)
    if not trace:
        return None
    if _last_action_name(messages) != "click_action":
        return None
    last = trace[-1]
    times = trace.count(last)
    if times < 2:
        return None
    is_pid = bool(_PROD_ID_RE.match(last))
    if not is_pid and last not in _REPEATABLE_NAV:
        return None

    if is_pid:
        lines = ["[Progress check] '%s' clicked %d times; repeating the same product does not make progress." % (last, times)]
    else:
        lines = ["[Progress check] '%s' clicked %d times; repeating it does not make progress (paging on is fine if intentional)." % (last, times)]

    opened = _opened_ids(trace)
    if opened:
        shown = ", ".join(opened[:4]) + (", ..." if len(opened) > 4 else "")
        lines.append("Opened so far: %s." % shown)
    else:
        lines.append("Opened so far: none.")

    last_obs = None
    for m in reversed(messages):
        if isinstance(m, dict) and m.get("role") in ("tool", "user") and isinstance(m.get("content"), str):
            last_obs = m["content"]
            break
    if last_obs is not None and "Price:" in last_obs:
        unmet = _unmet_values(messages, last_obs)
        if unmet:
            lines.append("Still missing here: %s." % "; ".join(unmet))

    example, clickables = None, _latest_results_clickables(messages)
    if clickables:
        opened_set = set(opened)
        for v in clickables:
            cand = v.strip()
            if _PROD_ID_RE.match(cand) and cand.lower() not in opened_set:
                example = cand
                break
    if example:
        lines.append("Try: click 'Back to Search', open a result you have NOT opened yet (e.g. '%s'), or click 'Next >' for more results." % example)
    else:
        lines.append("Try: click 'Back to Search' and change one keyword, then search again, or click 'Next >' for more results.")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# variant check (digit-bearing identifiers on a product page)
# --------------------------------------------------------------------------

# Boundary-safe tokenizer for clickable values: hyphens, quotes, periods and
# slashes stay inside a token, so variant codes, shade numbers, dimension
# marks and decimals survive intact instead of being split apart.
_VARIANT_TOKEN_RE = re.compile(r"#?[a-z0-9]+(?:[-'./\"][a-z0-9]*)*", re.I)

# Ordered identifier patterns; each match is blanked out of the working text
# so lower-priority patterns cannot re-cut an identifier that already matched.
_VARIANT_ID_RES = (
    re.compile(r"#\s*[0-9a-z]+", re.I),                                         # shade code
    re.compile(r"[0-9]+(?:\.[0-9]+)?\s*'\s*[0-9]+(?:\.[0-9]+)?\s*(?:\"|'')"),  # feet + inches
    re.compile(r"[0-9]+(?:\.[0-9]+)?\s*(?:\"|'')"),                             # inch mark
    re.compile(r"[0-9]+(?:\.[0-9]+)?\s*'"),                                     # foot mark
    re.compile(r"[0-9]+\.[0-9]+"),                                              # decimal value
    re.compile(r"(?<![a-z0-9])[0-9a-z]+(?:[-'][0-9a-z]+)*(?![a-z0-9])", re.I),  # alphanumeric code
)
_BARE_INT_RE = re.compile(r"[0-9]+")
_VARIANT_MARK_RE = re.compile(r"[#.'\"]")
# "<count>-<package word>" identifiers are quantity expressions, already
# handled by the quantity side of the ledger; the variant check targets
# codes, measurements and shade numbers instead.
_PACKAGE_IDENT_RE = re.compile(
    r"[0-9]+(?:[.][0-9]+)?-(?:pack|packs|pk|pcs|pc|ct|count|piece|pieces|"
    r"set|sets|pair|pairs|item|items)"
)


def _variant_identifiers(text):
    """Digit-bearing identifiers of the (budget-stripped) instruction, kept
    intact: decimals, alphanumeric codes, '#n' shades, dimension marks.
    Bare integers are ignored."""
    found = []
    work = text or ""
    for pat in _VARIANT_ID_RES:
        def _keep(m, _found=found):
            _found.append(m.group(0))
            return " " * len(m.group(0))
        work = pat.sub(_keep, work)
    out, seen = [], set()
    for raw in found:
        ident = re.sub(r"\s+", "", raw).lower()
        if not ident or not re.search(r"[0-9]", ident):
            continue
        if _BARE_INT_RE.fullmatch(ident):
            continue
        if _PACKAGE_IDENT_RE.fullmatch(ident):
            continue
        if not (re.search(r"[a-z]", ident) or _VARIANT_MARK_RE.search(ident)):
            continue
        if ident not in seen:
            seen.add(ident)
            out.append(ident)
    return out


def _variant_tokens(value):
    return [t.lower() for t in _VARIANT_TOKEN_RE.findall(value or "")]


def _carries_ident(value, ident):
    """Whole-token or token-prefix match of `ident` in a clickable value."""
    if not ident:
        return False
    for tok in _variant_tokens(value):
        if tok == ident or tok.startswith(ident):
            return True
    # tolerate whitespace inside multi-part identifiers ("2' 3\"", "# 6")
    compact = re.sub(r"\s+", "", (value or "").lower())
    start = 0
    while True:
        pos = compact.find(ident, start)
        if pos == -1:
            return False
        end = pos + len(ident)
        left_ok = pos == 0 or not (ident[0].isalnum() and compact[pos - 1].isalnum())
        right_ok = end >= len(compact) or not (ident[-1].isalnum() and compact[end].isalnum())
        if left_ok and right_ok:
            return True
        start = pos + 1


def _exact_ident_token(value, idents):
    toks = set(_variant_tokens(value))
    return any(i in toks for i in idents)


def _variant_block(messages):
    """Stateless check: the instruction carries digit-bearing identifiers and
    this product page offers an unclicked option carrying all of them."""
    if not isinstance(messages, list):
        return None
    instruction = _instruction_text(messages)
    if not instruction:
        return None
    content = None
    for m in reversed(messages):
        if isinstance(m, dict) and m.get("role") in ("tool", "user") and isinstance(m.get("content"), str):
            content = m["content"]
            break
    if content is None:
        return None
    if _RESULT_COUNT_RE.search(content):
        return None
    clickables, _has_bar = _parse_clickables(content)
    if not clickables or not _PRICE_RE.search(content):
        return None
    idents = _variant_identifiers(_strip_budget(instruction))
    if not idents:
        return None
    offered = _offered_values(clickables)
    if not offered:
        return None
    offered_by_norm = {}
    for v in offered:
        offered_by_norm.setdefault(_norm_click(v), v)
    # Every click_action so far whose value is still offered on this page
    # counts as clicked here (survives option-click page re-renders).
    clicked = set(c for c in _click_trace(messages) if c and c in offered_by_norm)
    if any(all(_carries_ident(v, i) for i in idents) for v in clicked):
        return None
    candidates = [
        raw for norm, raw in offered_by_norm.items()
        if norm not in clicked and all(_carries_ident(raw, i) for i in idents)
    ]
    if not candidates:
        return None
    best = sorted(candidates, key=lambda v: (0 if _exact_ident_token(v, idents) else 1, len(v), v.lower()))[0]
    ids_txt = ", ".join(idents)
    return ("Variant check: your instruction specifies %s; offered here as '%s' — click it and verify before buying "
            "(no clicked option carries %s)." % (ids_txt, best, ids_txt))


# --------------------------------------------------------------------------
# runtime entry points
# --------------------------------------------------------------------------

def _last_text_index(messages):
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if isinstance(m, dict) and m.get("role") in ("tool", "user") and isinstance(m.get("content"), str):
            return i
    return None


def _strip_blocks(content):
    """Drop a trailing injected block region so re-injection stays idempotent."""
    cut = len(content)
    for marker in _BLOCK_MARKERS:
        pos = content.find(marker)
        if pos != -1 and pos < cut:
            cut = pos
    return content[:cut]


def _with_blocks(content, blocks):
    """Append the given blocks (in order) after removing any stale block region."""
    content = _strip_blocks(content)
    for block in blocks:
        if block:
            content = content + "\n\n" + block
    return content


def _with_block(content, block):
    return _with_blocks(content, [block])


def register():
    return None


class AgentHarness:
    def __init__(self, public_context=None):
        self.public_context = public_context

    def transform_request(self, request):
        if isinstance(request, list):
            ledger = _ledger_block(request)
            progress = _progress_block(request)
            variant = _variant_block(request)
            if not ledger and not progress and not variant:
                return request
            out = copy.deepcopy(request)
            idx = _last_text_index(out)
            if idx is None:
                return request
            out[idx]["content"] = _with_blocks(out[idx]["content"], [ledger, progress, variant])
            return out
        if isinstance(request, dict) and isinstance(request.get("messages"), list):
            messages = request["messages"]
            ledger = _ledger_block(messages)
            progress = _progress_block(messages)
            variant = _variant_block(messages)
            if not ledger and not progress and not variant:
                return request
            out = copy.deepcopy(request)
            idx = _last_text_index(out["messages"])
            if idx is None:
                return request
            out["messages"][idx]["content"] = _with_blocks(out["messages"][idx].get("content", ""), [ledger, progress, variant])
            return out
        return request

    def transform_response(self, response, public_history):
        return response
