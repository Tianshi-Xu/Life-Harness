# Long-term memory

Distilled from past WebShop episodes. These are the mistakes that actually cost
episodes; treat them as strong priors.

## The episode has a turn budget

- An episode is cut off after about **20 tool calls**. The only scoring outcome
  is the purchase, so an episode that spends its turns paging, re-searching or
  re-opening pages ends at zero.
- Every wasted turn is a twentieth of the episode. Three wasted turns are
  already a sixth of the budget.
- With five turns or fewer left, buy the best matching product that is open or
  reachable: a wrong purchase and no purchase both score zero, so the purchase
  is never the worse choice.

## The purchase is irreversible

- A WebShop episode ends the moment `click_action(value="buy now")` runs, and
  the resulting purchase is the score. There is no second attempt.
- Therefore `buy now` must only be clicked once the product and every attribute
  option named by the instruction have been selected - but an unfinished
  episode scores zero as well, so do not stall either.

## Wrong product is the most expensive mistake

- Listings that share the brand or a keyword but are a different **kind** of
  object are wrong: a phone *case*, a screen *protector*, a *battery*, a
  *replacement part* is not the phone; a *nuts snack* or a *seasoning* is not
  the *bagel*; a *7x5ft* backdrop is not the requested *7x7ft* backdrop.
- Match the instruction's numbers exactly (size, dimensions, pack count,
  volume, model number). A listing that carries an extra number the
  instruction does not mention is very often the wrong variant.
- Compare the listing title with the words of the instruction before opening
  it: the title that carries the product type **and** the most distinguishing
  words of the instruction is the one to open.

## Attribute selection

- Product pages print every attribute group (`color`, `size`, `style`,
  `flavor name`, ...) and its option values. Click the option VALUE that the
  instruction names (`bakery`, `black`, `medium`), never the group label.
- A number the instruction spells out is the digits the page prints: `ten
  pack` is the `10 pack` option, `two pack` is `2 pack`, `six foot` is `6
  feet`. Match the numbers, and never accept a different one (`3 pack` is not
  `10 pack`, `10 feet (3 pack)` is not `3 feet (10 pack)`).
- When the `[SHOP CHECK]` block names an option value for a group (`group
  'size': the instruction's value is option "3 feet (10 pack)"`), click exactly
  that value before `buy now`. When it says a group has **no** matching option,
  do not click the nearest option instead.
- If no option in a group can satisfy the instruction's value, the product is
  wrong: go `back to search` instead of clicking the nearest option. If the
  instruction names no value for the group, stay and buy.
- `description`, `features` and `reviews` only re-print the same product text.
  They never reveal a hidden price or selection state, so never click them.

## The test before buying a product page

- The instruction is written from the product it wants, so the page that IS that
  product carries the instruction's own content: the **product type noun** and
  the instruction's **numbers** (size, count, volume, model), and the
  **attribute values** it names (`silver`, `green`, `bakery`, `old world
  style`, `fluoride free`). All three must be on the page.
- A page that fails that test is a look-alike and buying it scores 0: a
  `10x6.5ft` backdrop is not the `6 foot by 4 foot` one; a bare mattress
  foundation is not a `mattress set that has a 4" foundation`; a colour list
  without `green` cannot be the `green` hoodie. Leave with `back to search` and
  open a different listing.
- Only plain filler adjectives are exempt: `high`, `performance`, `screen`,
  `quality`, `long lasting`, `easy to assemble`, `cruelty free`, `relaxed
  fit`, `accessories`. The correct title omits those too, so do not leave a
  page over one of them alone.
- Judging every page by a missing adjective, re-searching and re-opening until
  the twenty turns are gone, is the opposite expensive habit: an episode that
  never buys scores zero even if the page under it was right. Leave a page only
  for the reasons above, and never more than a few times per episode.
- A product page you have already opened and left is identical the second
  time. Do not re-open it, and never buy it.
- Retreating from a product page to the search list is allowed only a few
  times per episode; past that the retreat is served as `buy now`, because a
  purchase can score while an unfinished episode cannot.


## Search and paging

- Never repeat the instruction's price wording inside `search_action`
  keywords; price words dilute the retrieval. Nothing in the search page
  filters by price, so compare prices by hand.
- There are about 50 results, ten per page, so the result list is five pages
  long. Every page after the fifth is **blank**; a blank page means the list is
  exhausted, and clicking `next >` again only returns another blank page.
- Paging is expensive and often pointless: if two or three pages of titles hold
  nothing that matches the instruction, search with **different** keywords
  instead of paging on.
- Repeating the same search never yields a different page. If a search did not
  help, change the words instead of re-sending them.
- The search-bar page (the page that offers `has_search_bar`) has only one
  action: `search`. `< prev` does not exist there, and clicking it wastes a
  turn.
- Clicking a value that is not one of the current page's actions does not fail
  loudly: the environment runs it as a search. If your click produces a page
  full of ten listings instead of one product, that is what happened.
