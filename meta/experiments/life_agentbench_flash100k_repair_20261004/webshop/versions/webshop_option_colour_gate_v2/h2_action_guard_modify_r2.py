"""H2 checkout guard v2 (webshop): keep one native tool call per turn, snap a
click value onto the page's own clickables, and keep the checkout honest - a
stated colour value that the page cannot provide, an instruction-matching option
value that is still unclicked (including the page's own compound or numeric
spelling such as 'cube (7)' or '6.5-7') and a price above the stated ceiling all
block 'buy now'."""
from json import dumps, loads
from re import DOTALL, escape, findall, fullmatch, search, split, sub

TARGET_LAYER = 'h2'

TOOL_NAMES = ('search_action', 'click_action')
STANDARD_CLICKS = ('back to search', '< prev', 'next >', 'description',
                   'features', 'reviews', 'buy now', 'search')
PRODUCT_ID_PATTERN = r"b[0-9a-z]{9}"
CEILING_PATTERN = (
    r"(?:lower than|less than|under|below|at most|no more than)\s*(?:usd\s*)?\$?\s*(\d+(?:\.\d+)?)")
PRICE_CLAUSE_PATTERN = (
    r"(?:price\s*)?(?:lower than|less than|under|below|at most|no more than)\s*(?:usd\s*)?\$?\s*"
    r"\d+(?:\.\d+)?(?:\s*(?:dollars|usd))?")
INSTRUCTION_PATTERN = r"Instruction:\s*\[SEP\]\s*(.*?)\s*\[SEP\]"
TOOL_CALL_PATTERN = r"<tool_call>\s*(.*?)\s*</tool_call>"
NUMBER_TOKEN_PATTERN = r"\d+(?:\.\d+)?"
NUMBER_PATTERN = r"(\d+(?:\.\d+)?)"
QUOTED_COLOUR_PATTERN = r"[\"']([a-z][a-z\-]{2,20})[\"']\s+colou?rs?\b"
PLAIN_COLOUR_PATTERN = r"([a-z][a-z\-]{2,20})\s+colou?rs?\b"
COLOUR_WORDS = ('granite', 'charcoal', 'ivory', 'maroon', 'burgundy', 'teal', 'turquoise',
                'aqua', 'coral', 'mint', 'plum', 'lavender', 'khaki', 'olive', 'mustard',
                'rust', 'wine', 'sand', 'stone', 'slate', 'graphite', 'camel', 'taupe',
                'mauve', 'peach', 'champagne', 'emerald', 'sapphire', 'ruby', 'amber',
                'cocoa', 'walnut', 'cherry', 'chocolate', 'espresso', 'bronze', 'copper',
                'brass', 'nickel', 'cream', 'tan', 'rose', 'jade', 'sage', 'navy', 'denim',
                'mocha', 'caramel', 'beige', 'black', 'white', 'grey', 'gray', 'brown',
                'red', 'blue', 'green', 'yellow', 'orange', 'pink', 'purple', 'gold',
                'silver', 'multicolor')
COLOUR_STOPWORDS = ('the', 'and', 'for', 'with', 'that', 'this', 'same', 'any', 'all',
                    'new', 'one', 'two', 'color', 'colour', 'colors', 'colours')
HEADER_NAMES = ('color', 'colors', 'colour', 'colours', 'size', 'sizes', 'style', 'pattern',
                'flavor', 'scent', 'quantity', 'count', 'pack', 'capacity', 'material',
                'finish', 'width', 'length', 'height', 'edition', 'version', 'model',
                'color name', 'size name')
SKIP_SEGMENTS = ('back to search', '< prev', 'next >', 'description', 'features',
                 'reviews', 'buy now', 'search', '')
TEXT_ONLY_NUDGE = (
    "TEXT-ONLY TURN REJECTED: this reply carries no native tool call. Re-issue the intended "
    "action as exactly one native tool call (search_action with keywords, or click_action with a "
    "value copied from the current Available Actions clickables list), and never write the call "
    "inside the message text.")


def _flat(text):
    return " ".join(str(text).split())


def _lower(text):
    return _flat(str(text).lower())


def _loads(text):
    if isinstance(text, dict):
        return text
    if not isinstance(text, str):
        return {}
    try:
        value = loads(text)
    except ValueError:
        return {}
    if isinstance(value, dict):
        return value
    return {}


def _instruction_source_text(messages):
    if not isinstance(messages, list):
        return ""
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str) and "Instruction:" in content:
            return content
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content
    return ""


def _instruction_payload(text):
    match = search(INSTRUCTION_PATTERN, text, DOTALL)
    if match:
        return _lower(match.group(1))
    return _lower(text)


def _ceiling_of(text):
    values = findall(CEILING_PATTERN, text or "")
    if values:
        return float(values[0])
    return None


def _without_price(text):
    return sub(PRICE_CLAUSE_PATTERN, " ", text or "")


def _base_phrase(value):
    text = _lower(value)
    position = text.find("(")
    if position > 0:
        text = text[:position]
    while text.endswith("-") or text.endswith(" "):
        text = text[:-1].strip()
    return text


def _numbers(text):
    return findall(NUMBER_TOKEN_PATTERN, text or "")


def _clickables(text):
    marker = "'clickables': ["
    start = text.rfind(marker)
    if start < 0:
        marker = '"clickables": ['
        start = text.rfind(marker)
        if start < 0:
            return []
    index = start + len(marker)
    values = []
    buffer = []
    quote = ""
    inside = False
    while index < len(text):
        char = text[index]
        if inside:
            if char == "\\" and index + 1 < len(text):
                buffer.append(text[index + 1])
                index += 2
                continue
            if char == quote:
                inside = False
                values.append(_flat("".join(buffer)))
                buffer = []
                index += 1
                continue
            buffer.append(char)
            index += 1
            continue
        if char == "'" or char == '"':
            inside = True
            quote = char
            index += 1
            continue
        if char == "]":
            break
        index += 1
    return [value for value in values if value]


def _newest_observation(messages):
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "tool":
            content = message.get("content")
            if isinstance(content, str):
                return content
    return ""


def _clicked_values(messages):
    found = []
    if not isinstance(messages, list):
        return found
    for message in messages:
        if not isinstance(message, dict):
            continue
        calls = message.get("tool_calls")
        if not isinstance(calls, list):
            continue
        for call in calls:
            if not isinstance(call, dict):
                continue
            function = call.get("function")
            if not isinstance(function, dict) or function.get("name") != "click_action":
                continue
            arguments = _loads(function.get("arguments"))
            value = arguments.get("value")
            if isinstance(value, str):
                found.append(_lower(value))
    return found


def _bounded(needle, haystack):
    return search(r"(?<![0-9a-z.])" + escape(needle) + r"(?![0-9a-z.])", haystack) is not None


def _value_matches(value, instruction, attribute_text):
    raw = _lower(value)
    if not raw:
        return False
    if _bounded(raw, instruction):
        return True
    base = _base_phrase(raw)
    if base and base != raw and _bounded(base, instruction):
        return True
    for number in _numbers(raw):
        if _bounded(number, attribute_text):
            return True
    return False


def _matches(value, instruction, attribute_text):
    if _value_matches(value, instruction, attribute_text):
        return True
    raw = _lower(value)
    parts = [part.strip() for part in split(r"[|/&]", raw) if part.strip()]
    if len(parts) > 1 and all(_bounded(part, instruction) for part in parts):
        return True
    return False


def _wanted_colours(instruction):
    text = instruction or ""
    found = []
    for word in findall(QUOTED_COLOUR_PATTERN, text):
        if word not in COLOUR_STOPWORDS and word not in found:
            found.append(word)
    for word in findall(PLAIN_COLOUR_PATTERN, text):
        if word in COLOUR_WORDS and word not in found:
            found.append(word)
    return found


def _colour_value_matches(value, wanted):
    raw = _lower(value)
    if not raw or not wanted:
        return False
    return _bounded(wanted, raw) or _bounded(raw, wanted)


def _price_of(text):
    position = text.find("Price:")
    if position < 0:
        return None
    numbers = findall(NUMBER_PATTERN, text[position:position + 40])
    if not numbers:
        return None
    return min(float(number) for number in numbers)


def _groups(text):
    segments = [segment.strip() for segment in text.split("[SEP]")]
    title = -1
    for index, segment in enumerate(segments):
        if segment.startswith("Price:"):
            title = index - 1
            break
    if title < 1:
        return {}
    result = {}
    current = ""
    for index in range(1, title):
        segment = segments[index]
        key = _lower(segment)
        if key in HEADER_NAMES:
            current = key
            if current not in result:
                result[current] = []
            continue
        if key in SKIP_SEGMENTS or index == 1:
            continue
        if current and segment:
            result[current].append(segment)
    return result


def _colour_group(groups):
    for name in ('color', 'colors', 'colour', 'colours'):
        values = groups.get(name)
        if values:
            return values
    return []


def _audit(observation, instruction, ceiling, clicked):
    kind = "other"
    if "Price:" in observation:
        kind = "product"
    elif "has_search_bar" in observation or "Total results" in observation or "Next >" in observation:
        kind = "search"
    attribute_text = _without_price(instruction)
    options = []
    colours = []
    if kind == "product":
        for value in _clickables(observation):
            key = _lower(value)
            if key in STANDARD_CLICKS or fullmatch(PRODUCT_ID_PATTERN, key):
                continue
            options.append(value)
        colours = _colour_group(_groups(observation))
    matched = [value for value in options if _matches(value, instruction, attribute_text)]
    pending = [value for value in matched if _lower(value) not in clicked]
    missing = []
    for wanted in _wanted_colours(instruction):
        if colours and not [value for value in colours if _colour_value_matches(value, wanted)]:
            missing.append(wanted)
    price = _price_of(observation) if kind == "product" else None
    violation = bool(ceiling is not None and price is not None and price > ceiling)
    return {"kind": kind, "options": options, "matched": matched, "pending": pending,
            "colours": colours, "missing_colours": missing, "price": price,
            "price_violation": violation}


def _call_from_payload(payload):
    if not isinstance(payload, dict):
        return None
    name = payload.get("name")
    arguments = payload.get("arguments")
    if isinstance(arguments, str):
        arguments = _loads(arguments)
    if name not in TOOL_NAMES or not isinstance(arguments, dict):
        return None
    if name == "search_action" and not isinstance(arguments.get("keywords"), str):
        return None
    if name == "click_action" and not isinstance(arguments.get("value"), str):
        return None
    return {"id": "h2-recovered-call", "type": "function",
            "function": {"name": name, "arguments": dumps(arguments)}}


def _call_id(call):
    value = call.get("id")
    if isinstance(value, str) and value:
        return value
    return "h2-guard-call"


def _repair_click_value(calls, history):
    first = calls[0]
    if not isinstance(first, dict):
        return None
    function = first.get("function")
    if not isinstance(function, dict) or function.get("name") != "click_action":
        return None
    arguments = _loads(function.get("arguments"))
    value = arguments.get("value")
    if not isinstance(value, str) or not value.strip():
        return None
    observation = _newest_observation(history)
    if not observation:
        return None
    clickables = _clickables(observation)
    if not clickables:
        return None
    lowered = _lower(value)
    for item in clickables:
        if _lower(item) == lowered:
            return None
    candidates = [item for item in clickables if _bounded(lowered, _lower(item))]
    if len(candidates) != 1:
        return None
    arguments = dict(arguments)
    arguments["value"] = candidates[0]
    replacement = {"id": _call_id(first), "type": "function",
                   "function": {"name": "click_action", "arguments": dumps(arguments)}}
    return [replacement] + list(calls[1:])


class Harness:
    def __init__(self):
        self.state = {}
        self.context = None

    def bind_shared_state(self, state):
        self.state = state

    def bind_task_context(self, context):
        self.context = context

    def h2(self, message, history):
        if not isinstance(message, dict):
            return message
        calls = message.get("tool_calls")
        if not (isinstance(calls, list) and calls):
            recovered = self._recover(message)
            if recovered is None:
                return message
            message = dict(message)
            message["tool_calls"] = recovered
            calls = recovered
        repaired = _repair_click_value(calls, history)
        if repaired is not None:
            message = dict(message)
            message["tool_calls"] = repaired
        return self._guard(message, history)

    def h2_no_tool_message(self, message, remaining):
        if not isinstance(message, dict):
            return None
        calls = message.get("tool_calls")
        if isinstance(calls, list) and calls:
            return None
        return TEXT_ONLY_NUDGE

    def _recover(self, message):
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            return None
        for block in findall(TOOL_CALL_PATTERN, content, DOTALL):
            call = _call_from_payload(_loads(block))
            if call is not None:
                return [call]
        call = _call_from_payload(_loads(content.strip()))
        if call is not None:
            return [call]
        return None

    def _instruction(self, history):
        state = self.state
        if isinstance(state, dict):
            text = state.get("instruction_text")
            if isinstance(text, str) and text.strip():
                return _lower(text)
        context = self.context
        if isinstance(context, dict):
            text = context.get("instruction")
            if isinstance(text, str) and text.strip():
                return _instruction_payload(text)
        return _instruction_payload(_instruction_source_text(history))

    def _ceiling(self, history):
        state = self.state
        if isinstance(state, dict):
            value = state.get("price_ceiling")
            if isinstance(value, bool):
                value = None
            elif isinstance(value, (int, float)):
                return float(value)
        return _ceiling_of(self._instruction(history))

    def _guard(self, message, history):
        calls = message.get("tool_calls")
        if not isinstance(calls, list) or not calls:
            return message
        first = calls[0]
        if not isinstance(first, dict):
            return message
        function = first.get("function")
        if not isinstance(function, dict) or function.get("name") != "click_action":
            return message
        arguments = _loads(function.get("arguments"))
        value = arguments.get("value")
        if not isinstance(value, str) or _lower(value) != "buy now":
            return message
        observation = _newest_observation(history)
        if not observation:
            return message
        page = _audit(observation, self._instruction(history), self._ceiling(history),
                      _clicked_values(history))
        if page["kind"] != "product":
            return message
        cached = self.state.get("page_state") if isinstance(self.state, dict) else None
        if (isinstance(cached, dict) and cached.get("kind") == "product"
                and cached.get("options") == page["options"]):
            for key in ("pending", "missing_colours"):
                values = cached.get(key)
                if isinstance(values, list):
                    page[key] = [item for item in values if isinstance(item, str)]
            if isinstance(cached.get("price_violation"), bool):
                page["price_violation"] = cached["price_violation"]
        if page["price_violation"] or page["missing_colours"]:
            return self._rewrite(message, "back to search")
        if page["pending"]:
            return self._rewrite(message, page["pending"][0])
        return message

    def _rewrite(self, message, value):
        calls = message.get("tool_calls")
        first = calls[0] if isinstance(calls, list) and calls else None
        call_id = "h2-guard-call"
        if isinstance(first, dict) and isinstance(first.get("id"), str) and first.get("id"):
            call_id = first["id"]
        replacement = {"id": call_id, "type": "function",
                       "function": {"name": "click_action",
                                    "arguments": dumps({"value": value})}}
        repaired = dict(message)
        if isinstance(calls, list) and calls:
            repaired["tool_calls"] = [replacement] + list(calls[1:])
        else:
            repaired["tool_calls"] = [replacement]
        return repaired
