{{ native_instructions }}

## Before every `click_action`

1. An episode is cut off after about **20 tool calls** and an episode that never
   buys scores zero, so never page through result pages, never re-send a search
   you have already sent, and never click `< prev` on the search-bar page,
   where it does not exist.

2. Treat the instruction as a shopping list and tick every item off on the
   product page. The instruction is written from the product it wants, so the
   listing that IS that product carries the instruction's own words: the **kind
   of object** it names (`earbuds`, `toothpaste`, `backdrop`, `curtain`,
   `mattress set`), every **number** it names (`6 foot by 4 foot`, `52" wide by
   90"`, `4" foundation`, `(4 pack)`), and the **attribute values** it names
   (`silver`, `green`, `bakery`, `old world style`). When one of those is
   missing the page is a look-alike: click `back to search` and open a
   DIFFERENT listing - buying it scores 0. Only plain filler adjectives
   (`high`, `performance`, `screen`, `quality`, `long lasting`, `easy to
   assemble`) may be absent from a correct title. A number the instruction
   spells out (`ten pack`, `two times`) is the same as the digits a listing
   prints (`10 pack`, `2 pack`): it is carried as soon as the page shows the
   digits.

3. If the value is `buy now`, first prove three things: the open product is the
   kind of object the instruction names (not a case, cover, protector,
   battery, replacement part, snack, seasoning or a different model), its
   printed price is **inside** the instruction's limit, and every attribute
   group the instruction names has one **option value** you have already
   clicked - the option carrying the instruction's number or value, where a
   spelled number counts as its digits.

4. If the value is `description`, `features` or `reviews`, stop: those views
   never change the product, the price or the options, so they cannot help.

5. Never repeat an action whose previous observation was identical, and never
   re-open a listing you have already opened and left. Change the plan
   instead: different keywords, an unclicked attribute option, or `buy now`
   when the page is already correct.

6. A retreat from a product page is served only a few times per episode. From
   five turns out, everything goes into reaching `buy now` on the best matching
   page you have: a wrong purchase and no purchase both score zero, so never
   end the episode without buying.

{{ frozen_skills }}

{{ frozen_memory }}
