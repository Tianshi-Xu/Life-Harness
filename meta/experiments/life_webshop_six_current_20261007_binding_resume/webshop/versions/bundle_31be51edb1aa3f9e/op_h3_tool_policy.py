"""H3 -- tool-description conditioning for WebShop (content only).

Names, schemas, order and dispatch stay exactly as supplied. The two native
descriptions say nothing about keyword choice or about the option values that
gate a valid purchase, which is where the public TRAIN traces diverge.
"""
TARGET_LAYER = 'h3'

_DESCRIPTIONS = {
    'search_action': (
        "Search the shop by keywords. Use the product name/brand plus the distinctive "
        "descriptive words of the instruction (e.g. 'black dress imported zipper'); leave "
        "out prices, 'dollars', 'lower than' and filler words."
    ),
    'click_action': (
        "Click one value from the current 'clickables' list, copied exactly as written. On a "
        "product page, click each option value the instruction asks for (color, size, style, "
        "flavor, pack) before clicking 'buy now'."
    ),
}


class Harness:
    def bind_shared_state(self, state):
        self.state = state

    def bind_task_context(self, context):
        self.context = context

    def h3(self, tools):
        if not isinstance(tools, list):
            return tools
        patched = []
        for tool in tools:
            function = tool.get('function') if isinstance(tool, dict) else None
            text = _DESCRIPTIONS.get(function.get('name')) if isinstance(function, dict) else None
            if text is None:
                patched.append(tool)
                continue
            updated = dict(tool)
            inner = dict(function)
            inner['description'] = text
            updated['function'] = inner
            patched.append(updated)
        return patched
