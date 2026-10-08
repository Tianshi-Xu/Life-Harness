"""H5 -- cold-start purchase skills for WebShop (payload only).

The first solver turn receives this procedure appended to the system message.
It encodes the purchase contract that the public TRAIN traces show to matter:
one tool call per turn, instruction keywords for searching, and -- above all --
selecting every option value the instruction names *before* clicking 'buy now'.
No shared state is used here; per-episode progress lives in H4 and action
control lives in H2.
"""
TARGET_LAYER = 'h5'

_SKILL = """WebShop purchase procedure - apply it on every turn.

1. Exactly one tool call per turn: search_action(keywords) or click_action(value). Never reply with prose only.
2. search_action keywords: the product name/brand plus the distinctive descriptive words of the instruction (e.g. 'black dress imported zipper'). Leave out prices, 'dollars', 'lower than' and filler words.
3. Click a product whose title contains the instruction's product type and its distinctive words; do not click the first result by default.
4. A product page lists option values among its clickables (color, size, style, flavor name, pack size). The instruction names the values it wants, e.g. 'navy', 'x-large', '2-pack', 'lemon ginger', '225zf-4', '#6', '10 in'.
5. Click EVERY option value the instruction asks for, one click per turn, copying the value exactly as it is written in the clickables. Match by meaning, not spelling: '3 pack' = 'pack of 3', '3xl' = '3x-large', 'three x large' = '3x-large big', 'pack of nine' = 'pack of 9'.
6. Do not select option values the instruction does not ask for, and never click 'buy now' while an asked-for option value is still unselected.
7. If the page offers no option that fits a required attribute (wrong color, wrong size, wrong length, wrong style), click 'back to search' and open a better product instead of buying.
8. Finish with click_action('buy now') once the right product is open and all of its asked-for option values are selected.
"""


class Harness:
    def bind_shared_state(self, state):
        self.state = state

    def bind_task_context(self, context):
        self.context = context

    def h5(self, messages):
        return [_SKILL]
