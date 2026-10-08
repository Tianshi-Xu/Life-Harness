"""H5 -- cold-start purchase skills for WebShop (payload only).

The first solver turn receives this procedure appended to the system message.
It encodes the purchase contract that the public TRAIN traces show to matter:
one tool call per turn, search keywords that name the product, never repeating a
query that was already submitted, opening the closest title on a result page
instead of paging through the list, selecting exactly one instruction-named
value per option group (group values are alternatives, so the last click is the
bought variant), verifying the product before buying, returning from a
description/features/reviews sub-page with '< prev', and finishing with
'buy now' well before the turn limit.

Two wordings of the earlier version are removed because the traces show the
solver parroting them: "search again with a shorter query that names the
product" was repeated verbatim as a plan and then executed as the identical
query (819, 611, 511, 463, 799, 766, 352), and the pre-buying verification rule
demanded that every named attribute be visible in the product's own text, which
the successful purchases 777 and 569 do not satisfy. Verification now only
rules out products that clearly contradict the instruction. No shared state is
used here; per-episode progress lives in H4 and action control lives in H2.
"""
TARGET_LAYER = 'h5'

_SKILL = """WebShop purchase procedure - apply it on every turn.

1. Your reply is exactly one tool call and nothing else: search_action(keywords) or click_action(value). Harness feedback arrives in messages marked [H4] or [H2]; never copy, paraphrase or answer those notes, and never write your own note about the page. A reply without a tool call is discarded and costs one of your turns.
2. search_action keywords: the product name or type plus the instruction's distinctive words (e.g. 'black dress imported zipper'); leave out prices, 'dollars', 'lower than' and filler words. Never submit keywords you already submitted - the identical result list comes back. If the results are a different kind of item, open the closest title on the page, or search with different and fewer words (drop the least distinctive one).
3. Result page: click the product whose title contains the instruction's product words and as many of its attribute words as possible. If no title contains all of them, open the title that contains the most of them rather than paging further; click 'next >' only when no title on the page shares even two of the instruction's words.
4. Product page: click every option value the instruction names, but exactly ONE value per option group - the values in one group are alternatives of the same attribute and the last one you click is the variant you buy ('x-small' is not 'small'). An option counts as named only when the instruction really names that value: 'blue' is not wanted because the instruction says 'bluetooth', 'gold' is not wanted because the instruction says 'rose gold'. Match by meaning, not spelling: '3 pack' = 'pack of 3', 'three x large' = '3x-large', 'pink | white' = pink.
5. Verify before buying: the title, the options and the description must not contradict the instruction (a different kind of item, or a colour or size the instruction rules out). If this product clearly cannot satisfy the instruction, click 'back to search' and open a better title; do not keep searching with the same words.
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
