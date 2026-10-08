"""H3 -- tool-description conditioning for WebShop (content only).

Names, schemas, order and dispatch stay exactly as supplied. The two native
descriptions say nothing about keyword choice or about the option values that
gate a valid purchase, which is where the public TRAIN traces diverge: searches
that return a different kind of item keep the target out of reach, several
values of one option group flip the bought variant, a product page viewed
through Description/Features/Reviews has no 'buy now', and a solver that is
told to "search again with a shorter query" tends to re-submit the keywords it
already used, sees the identical result list and burns the whole turn budget.

The description therefore names the two loops it must not enter -- repeating
search keywords, and answering without a tool call -- and points at the
concrete alternative actions (open the closest title on the page, drop the
least distinctive keyword, click 'next >' only when the page has nothing
closer) instead of inviting another search.
"""
TARGET_LAYER = 'h3'

_DESCRIPTIONS = {
    'search_action': (
        "Search the shop by keywords. Use the product name or type plus the instruction's "
        "distinctive words (e.g. 'black dress imported zipper'); leave out prices, 'dollars', "
        "'lower than' and filler words. Never submit keywords you already submitted: the same "
        "result list comes back and the turn is lost. If the results are a different kind of "
        "item, open the closest title on the page, or search with different and fewer words, "
        "dropping the least distinctive keyword."
    ),
    'click_action': (
        "Click exactly one value from the current 'clickables' list, copied as written. Answer "
        "with exactly one tool call every turn and never with text alone. On a product page, "
        "click each option value the instruction names, one value per option group -- values in a "
        "group are alternatives and the last click is the variant you buy -- then click 'buy "
        "now'. If the page only shows Description/Features/Reviews, click '< prev' first; if the "
        "product clearly cannot satisfy the instruction (wrong kind of item, or a colour it rules "
        "out), click 'back to search' and open a better title instead of buying."
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
