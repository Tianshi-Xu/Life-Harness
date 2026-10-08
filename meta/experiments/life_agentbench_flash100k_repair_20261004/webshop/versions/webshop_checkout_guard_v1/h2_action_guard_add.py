"""H2 action guard (webshop): recover a lost native tool call and keep the
checkout honest - instruction-matching option values are clicked before
'buy now', and no purchase is made above the stated price ceiling."""
from json import dumps, loads
from re import DOTALL, escape, findall, fullmatch, search, split

TARGET_LAYER = 'h2'

TOOL_NAMES = ('search_action', 'click_action')
STANDARD_CLICKS = ('back to search', '< prev', 'next >', 'description',
                   'features', 'reviews', 'buy now', 'search')
PRODUCT_ID_PATTERN = r"b[0-9a-z]{9}"
CEILING_PATTERN = (
    r"(?:lower than|less than|under|below|at most|no more than)\s*(?:usd\s*)?\$?\s*(\d+(?:\.\d+)?)")
INSTRUCTION_PATTERN = r"Instruction:\s*\[SEP\]\s*(.*?)\s*\[SEP\]"
TOOL_CALL_PATTERN = r"<tool_call>\s*(.*?)\s*</tool_call>"
NUMBER_PATTERN = r"(\d+(?:\.\d+)?)"
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


def _audit(observation, instruction, ceiling, clicked):
    kind = "other"
    if "Price:" in observation:
        kind = "product"
    elif "has_search_bar" in observation or "Total results" in observation or "Next >" in observation:
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
        if isinstance(calls, list) and calls:
            return self._guard(message, history)
        recovered = self._recover(message)
        if recovered is None:
            return message
        repaired = dict(message)
        repaired["tool_calls"] = recovered
        return self._guard(repaired, history)

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
            pending = cached.get("pending")
            if isinstance(pending, list):
                page["pending"] = [item for item in pending if isinstance(item, str)]
            if isinstance(cached.get("price_violation"), bool):
                page["price_violation"] = cached["price_violation"]
        if page["price_violation"]:
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
