"""H3 -- tool-description conditioning for WebShop (content only).

Names, schemas, order and dispatch stay exactly as supplied. The two native
descriptions say nothing about keyword choice or about the option values that
gate a valid purchase, which is where the public TRAIN traces diverge: searches
that return a different kind of item keep the target out of reach, several
values of one option group flip the bought variant, and a product page viewed
through Description/Features/Reviews has no 'buy now'.
"""
TARGET_LAYER = 'h3'

_DESCRIPTIONS = {
    'search_action': (
        "Search the shop by keywords. Use the product name or type plus the instruction's "
        "distinctive words (e.g. 'black dress imported zipper'); leave out prices, 'dollars', "
        "'lower than' and filler words. If the results are a different kind of item or only "
        "accessories (chargers, cables, cases when you need a phone), search again with a "
        "shorter query that names the product."
    ),
    'click_action': (
        "Click exactly one value from the current 'clickables' list, copied as written. On a "
        "product page, click each option value the instruction names, one value per option "
        "group -- values in a group are alternatives and the last click is the variant you buy "
        "-- then click 'buy now'. If the page only shows Description/Features/Reviews, click "
        "'< prev' first."
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
