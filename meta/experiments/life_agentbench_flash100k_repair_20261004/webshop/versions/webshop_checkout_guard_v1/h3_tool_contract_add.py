"""H3 tool conditioning (webshop): keep the two native tool schemas untouched and
state the click-value / search-keyword contract in their descriptions."""
TARGET_LAYER = 'h3'

SEARCH_DESCRIPTION = (
    "Use search functionality with specified keywords. Give the product type plus every "
    "descriptive word of the instruction (brand, colour, size, quantity, material); never "
    "include a price phrase or a dollar amount. The reply lists product ids (b...), titles and "
    "prices with a 'next >' link, and every accepted value is echoed in Available Actions. Call "
    "this native tool instead of writing the call in the message text.")
CLICK_DESCRIPTION = (
    "Click a button or link with a specified value. Copy the value exactly from the clickables "
    "list of the current observation: a product id (b...) or 'next >' on a results page; an "
    "option value such as 'vintage grey', 'black & gold' or '2-pack', or 'back to search' on a "
    "product page; and 'buy now' only after every option value that matches the instruction has "
    "been clicked. Call this native tool instead of writing the call in the message text.")


class Harness:
    def bind_shared_state(self, state):
        self.state = state

    def bind_task_context(self, context):
        self.context = context

    def h3(self, tools):
        if not isinstance(tools, list):
            return tools
        result = []
        for tool in tools:
            item = tool
            if isinstance(tool, dict):
                item = dict(tool)
                function = item.get("function")
                if isinstance(function, dict):
                    name = function.get("name")
                    if name == "search_action":
                        copied = dict(function)
                        copied["description"] = SEARCH_DESCRIPTION
                        item["function"] = copied
                    elif name == "click_action":
                        copied = dict(function)
                        copied["description"] = CLICK_DESCRIPTION
                        item["function"] = copied
            result.append(item)
        return result
