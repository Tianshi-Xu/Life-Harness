"""H5 -- cold-start purchase skills for WebShop (payload only).

The first solver turn receives this procedure appended to the system message.
It encodes the purchase contract that the public TRAIN traces show to matter:
one tool call per turn, search keywords that name the product, reading result
titles before clicking, selecting exactly one instruction-named value per
option group (group values are alternatives, so the last click is the bought
variant), verifying the product before buying, returning from a
description/features/reviews sub-page with '< prev', and finishing with
'buy now' well before the turn limit. No shared state is used here; per-episode
progress lives in H4 and action control lives in H2.
"""
TARGET_LAYER = 'h5'

_SKILL = """WebShop purchase procedure - apply it on every turn.

1. Exactly one tool call per turn: search_action(keywords) or click_action(value). Never answer with prose only, and never repeat this procedure text back as your reply.
2. search_action keywords: the product name or type plus the instruction's distinctive words (e.g. 'black dress imported zipper'); leave out prices, 'dollars', 'lower than' and filler words. If the results are a different kind of item or only accessories (chargers, cables, cases when you need a phone), search again with a shorter query that names the product.
3. Result page: click the product whose title contains the instruction's product words and as many of its attribute words as possible; if no title on the page does, click 'next >' or search again. Never open a product whose title names a different colour, size or style than the instruction.
4. Product page: click every option value the instruction names, but exactly ONE value per option group - the values in one group are alternatives of the same attribute and the last one you click is the variant you buy ('x-small' is not 'small'). An option counts as named only when the instruction really names that value: 'blue' is not wanted because the instruction says 'bluetooth'. Match by meaning, not spelling: '3 pack' = 'pack of 3', 'three x large' = '3x-large', 'pink | white' = pink.
5. Verify before buying: the title, the options and the description must cover the instruction's distinctive attributes (e.g. 'heavy duty steel frame', 'lumbar support', 'height adjustable', 'paraben free'). If a named attribute is missing from this product, click 'back to search' and open a better product instead of buying.
6. Sub-pages: if the observation shows Description, Features or Reviews instead of the product page, click '< prev' to return to the product page.
7. Finish: once the right product is open and all of its instruction-named option values are selected, click_action('buy now'). Do not spend turns re-searching for a product you already found, and do not wait for the turn limit.
"""


class Harness:
    def bind_shared_state(self, state):
        self.state = state

    def bind_task_context(self, context):
        self.context = context

    def h5(self, messages):
        return [_SKILL]
