"""H4 monitor (webshop): audit the newest public observation, publish it as
shared page_state, and give the solver a price-ceiling or option/attribute check."""
from json import loads
from re import DOTALL, escape, findall, fullmatch, search, split

TARGET_LAYER = 'h4'

STANDARD_CLICKS = ('back to search', '< prev', 'next >', 'description',
                   'features', 'reviews', 'buy now', 'search')
PRODUCT_ID_PATTERN = r"b[0-9a-z]{9}"
CEILING_PATTERN = (
    r"(?:lower than|less than|under|below|at most|no more than)\s*(?:usd\s*)?\$?\s*(\d+(?:\.\d+)?)")
INSTRUCTION_PATTERN = r"Instruction:\s*\[SEP\]\s*(.*?)\s*\[SEP\]"
NUMBER_PATTERN = r"(\d+(?:\.\d+)?)"
HEADER_NAMES = ('color', 'colors', 'colour', 'colours', 'size', 'sizes', 'style', 'pattern',
                'flavor', 'scent', 'quantity', 'count', 'pack', 'capacity', 'material',
                'finish', 'width', 'length', 'height', 'edition', 'version', 'model',
                'color name', 'size name')
SKIP_SEGMENTS = ('back to search', '< prev', 'next >', 'description', 'features',
                 'reviews', 'buy now', 'search', '')
COLORS = ('black', 'white', 'grey', 'gray', 'beige', 'brown', 'red', 'blue', 'green',
          'yellow', 'orange', 'pink', 'purple', 'gold', 'silver', 'ivory', 'navy',
          'teal', 'cream', 'tan', 'bronze', 'brass', 'copper', 'nickel', 'cherry',
          'espresso', 'walnut', 'oak', 'multicolor')
PRICE_PRODUCT = ("PRICE CHECK: the current product page shows a price of ${price:.2f}, which is "
                 "above the ceiling of ${ceiling:.2f} stated by the instruction. Click "
                 "'back to search' and open a product whose price is below ${ceiling:.2f}.")
PRICE_REMINDER = ("PRICE REMINDER: the instruction allows at most ${ceiling:.2f}. Compare the "
                  "listed price of each candidate with that ceiling and open a product whose "
                  "price is below ${ceiling:.2f} before clicking 'buy now'.")
OPTION_CHECK = ("OPTION CHECK: this product page lists option values that match the instruction "
                "and are still unselected: {listing}. Click one of them now, and click 'buy now' "
                "only after every listed value has been clicked.")
ATTRIBUTE_CHECK = ("ATTRIBUTE CHECK: the instruction asks for {wanted} but this product page only "
                   "offers {offered}. Click 'back to search' and open a product whose options "
                   "include {wanted}.")


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


def _matches(value, instruction):
    value = _lower(value)
    if not value:
        return False
    if _bounded(value, instruction):
        return True
    parts = [part.strip() for part in split(r"[|/&]", value) if part.strip()]
    if len(parts) > 1:
        return all(_bounded(part, instruction) for part in parts)
    return False


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


def _listing(values):
    return ", ".join("'" + value + "'" for value in values)


class Harness:
    def __init__(self):
        self.state = {}
        self.context = None

    def bind_shared_state(self, state):
        self.state = state

    def bind_task_context(self, context):
        self.context = context

    def h4(self, messages, remaining):
        observation = _newest_observation(messages)
        if not observation:
            return []
        instruction = self._instruction(messages)
        ceiling = self._ceiling(instruction)
        page = self._audit(observation, instruction, ceiling, _clicked_values(messages))
        colours = _colour_group(_groups(observation)) if page["kind"] == "product" else []
        if isinstance(self.state, dict):
            self.state["page_state"] = {"kind": page["kind"], "price": page["price"],
                                        "options": page["options"], "pending": page["pending"],
                                        "price_violation": page["price_violation"]}
        if page["kind"] == "product":
            if page["price_violation"]:
                return [PRICE_PRODUCT.format(price=page["price"], ceiling=ceiling)]
            wanted = [colour for colour in COLORS if _bounded(colour, instruction)]
            if colours and wanted and not [value for value in colours
                                           if _matches(value, instruction)]:
                return [ATTRIBUTE_CHECK.format(wanted=_listing(wanted),
                                               offered=_listing(colours))]
            if page["pending"]:
                return [OPTION_CHECK.format(listing=_listing(page["pending"]))]
            return []
        if page["kind"] == "search" and ceiling is not None:
            return [PRICE_REMINDER.format(ceiling=ceiling)]
        return []

    def _instruction(self, messages):
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
        return _instruction_payload(_instruction_source_text(messages))

    def _ceiling(self, instruction):
        state = self.state
        if isinstance(state, dict):
            value = state.get("price_ceiling")
            if isinstance(value, bool):
                value = None
            elif isinstance(value, (int, float)):
                return float(value)
        return _ceiling_of(instruction)

    def _audit(self, observation, instruction, ceiling, clicked):
        kind = "other"
        if "Price:" in observation:
            kind = "product"
        elif ("has_search_bar" in observation or "Total results" in observation
              or "Next >" in observation):
            kind = "search"
        options = []
        if kind == "product":
            for value in _clickables(observation):
                key = _lower(value)
                if key in STANDARD_CLICKS or fullmatch(PRODUCT_ID_PATTERN, key):
                    continue
                options.append(value)
        matched = [value for value in options if _matches(value, instruction)]
        pending = [value for value in matched if _lower(value) not in clicked]
        price = _price_of(observation) if kind == "product" else None
        violation = bool(ceiling is not None and price is not None and price > ceiling)
        return {"kind": kind, "options": options, "matched": matched, "pending": pending,
                "price": price, "price_violation": violation}
