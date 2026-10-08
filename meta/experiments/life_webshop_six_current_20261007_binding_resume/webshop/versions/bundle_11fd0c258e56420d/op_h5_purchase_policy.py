"""H5 -- cold-start purchase skills for WebShop (payload only).

The first solver turn receives this procedure appended to the system message.
It encodes the purchase contract that the public TRAIN traces show to matter:
one tool call per turn, search keywords that name the product, never repeating a
query that was already submitted, opening the closest title on a result page
instead of searching again, selecting exactly one instruction-named value per
option group (group values are alternatives, so the last click is the bought
variant), matching an option value by meaning rather than spelling, returning
from a description/features/reviews view with '< prev', and finishing with
'buy now' well before the turn limit.

Three wordings of the earlier versions are removed because the traces show the
solver parroting them and then burning turns on them: "search again with a
shorter query that names the product" was executed as the identical query, and
"search with different and fewer words, dropping the least distinctive keyword"
was paraphrased as a plan and followed by another identical query (764, 292,
589, 819, 611, 511, 463, 799, 766, 352). The pre-buying verification rule is
also narrowed, because the solver read "if the observation shows Description,
Features or Reviews" as true of an ordinary product page, which lists those
clickables, and clicked '< prev' out of a usable product page (560). No shared
state is used here; per-episode progress lives in H4 and action control in H2.
"""
TARGET_LAYER = 'h5'

_SKILL = """WebShop purchase procedure - apply it on every turn.

1. Your reply is exactly one tool call and nothing else: search_action(keywords) or click_action(value). Harness feedback arrives in messages marked [H4] or [H2]; never copy, paraphrase or answer those notes, and never write your own note about the page. A reply without a tool call is discarded and costs one of your turns.
2. search_action keywords: the product name or type plus the instruction's distinctive words (e.g. 'black dress imported zipper'); leave out prices, 'dollars', 'lower than' and filler words. Never submit keywords you already submitted - the identical result list comes back and the turn is lost. If the results are a different kind of item than asked for, open the closest title on the page instead of searching again.
3. Result page: click the product whose title contains the instruction's product words and as many of its attribute words as possible. If no title contains all of them, open the title that contains the most of them rather than paging further; click 'next >' only when no title on the page shares even two of the instruction's words.
4. Product page: click every option value the instruction names, but exactly ONE value per option group - the values in one group are alternatives of the same attribute and the last one you click is the variant you buy ('x-small' is not 'small'). An option counts as named only when the instruction really names that value: 'blue' is not wanted because the instruction says 'bluetooth', 'gold' is not wanted because the instruction says 'rose gold', and a size the instruction does not mention (a 3 ft option when the instruction asks for 135 ft) is not wanted either. Match by meaning, not spelling: '3 pack' = 'pack of 3', 'three x large' = '3x-large', 'pink | white' = pink, '200ft' = '200 foot'.
5. Verify before buying: the title and the option values must not contradict the instruction (a different kind of item, or a colour or size the instruction rules out). If this product clearly cannot satisfy the instruction, click 'back to search' and open a better title; do not keep searching with the same words. You do not need to open Description, Features or Reviews to verify.
6. Sub-pages: if the observation has no 'buy now' and no option values and instead shows the product's Description, Features or Reviews text, click '< prev' to return to the product page.
7. Finish: once the right product is open and all of its instruction-named option values are selected, click_action('buy now'). Do not spend turns re-searching for a product you already found, and do not wait for the turn limit.
"""


class Harness:
    def bind_shared_state(self, state):
        self.state = state

    def bind_task_context(self, context):
        self.context = context

    def h5(self, messages):
        return [_SKILL]
