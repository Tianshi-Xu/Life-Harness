"""H5 cold start (webshop): seed the shared instruction/ceiling and inject the
purchase procedure on the first solver turn."""
from re import DOTALL, findall, search

TARGET_LAYER = 'h5'

CEILING_PATTERN = (
    r"(?:lower than|less than|under|below|at most|no more than)\s*(?:usd\s*)?\$?\s*(\d+(?:\.\d+)?)")
INSTRUCTION_PATTERN = r"Instruction:\s*\[SEP\]\s*(.*?)\s*\[SEP\]"
GUIDANCE = (
    "WebShop purchase procedure:\n"
    "1. Read the instruction: list the product type and every descriptive word (brand, colour, "
    "size, quantity, style), and record the price ceiling when it says \"price lower than X "
    "dollars\" or \"less than $X\".\n"
    "2. Act first with search_action, using the product type plus those descriptive words and no "
    "price phrase or dollar amount.\n"
    "3. On a results page compare every title with ALL the descriptive words, then click the "
    "product id (b...) of the best full match whose price is below the ceiling; if no visible "
    "result qualifies, click 'next >'.\n"
    "4. On a product page check the price against the ceiling, then click EVERY option value "
    "listed for the product that matches a descriptive word (colour, size, quantity ...). One "
    "click selects one option value, so click one unclicked matching value per turn and keep "
    "going.\n"
    "5. Click 'buy now' only when no matching option value on that page is still unclicked and "
    "the price is below the ceiling.\n"
    "6. Every turn must contain exactly one native tool call - search_action or click_action - "
    "with the click value copied exactly from the current Available Actions clickables list. "
    "Never write a tool call inside the message text.")


def _flat(text):
    return " ".join(str(text).split())


def _lower(text):
    return _flat(str(text).lower())


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


class Harness:
    def __init__(self):
        self.state = {}
        self.context = None

    def bind_shared_state(self, state):
        self.state = state

    def bind_task_context(self, context):
        self.context = context

    def h5(self, messages):
        text = ""
        context = self.context
        if isinstance(context, dict):
            value = context.get("instruction")
            if isinstance(value, str) and value.strip():
                text = value
        if not text:
            text = _instruction_source_text(messages)
        instruction = _instruction_payload(text) if text else ""
        if isinstance(self.state, dict) and instruction:
            stored = self.state.get("instruction_text")
            if not isinstance(stored, str) or not stored.strip():
                self.state["instruction_text"] = instruction
            if self.state.get("price_ceiling") is None:
                ceiling = _ceiling_of(instruction)
                if ceiling is not None:
                    self.state["price_ceiling"] = ceiling
        return [GUIDANCE]
