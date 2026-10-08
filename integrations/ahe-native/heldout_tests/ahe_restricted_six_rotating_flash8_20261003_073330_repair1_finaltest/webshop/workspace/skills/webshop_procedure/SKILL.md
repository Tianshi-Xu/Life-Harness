# Skill: WebShop purchase procedure

You shop in a WebShop text environment. The only two actions are
`search_action(keywords="...")` and `click_action(value="...")`.
Emit exactly one tool call per turn.

## The episode budget is the hard limit

- An episode is cut off after about **20 tool calls**. When that happens the
  episode ends wherever it is, and an episode that never bought anything scores
  zero no matter how good the product you were looking at was.
- Count your turns. After 15 of them, stop exploring and get to `buy now` on
  the best matching product you can reach. The final turn of an episode must
  never be spent on anything but `buy now` on a product page that is inside the
  budget.
- The environment ends the episode the instant `click_action(value="buy now")`
  is executed, and the purchase is scored against the instruction.

Every action costs one twentieth of the episode, so the two ways to lose are
(a) buying the wrong product and (b) burning turns without buying.

## Turn loop

1. SEARCH
   - `search_action` with the product **type noun** the instruction asks for
     plus its distinguishing words: material, flavour, colour, size, count,
     brand. Example: `low carb bagel`, `samsung galaxy a71 mobile`.
   - Never put the budget into the keywords (no "price lower than 60 dollars",
     no "under $40"). Price words only dilute the retrieval.
   - A search page lists `ASIN | TITLE | PRICE` triples, **10 per page**, and
     there are about 50 results, so there are five pages of listings. Pages
     after the fifth are **empty**: a blank result page means the list is
     exhausted, not that the next page is coming.
   - Read the ten titles on the page you are on before paging. Paging costs
     turns of a twenty-turn episode. If nothing on the page fits, a search with
     **different** words is usually better than `next >`; if the titles look
     close, open the closest one instead of paging.
   - Re-sending the same keywords returns the identical page. Change at least
     one word.

2. PICK A PRODUCT
    - The listing TITLE must contain the **product type the instruction names**.
      That type noun is the word that says what the object *is*: `tablet`,
      `eyeliner`, `dresser`, `battery`, `sofa table`. A title that does not carry
      it is a different kind of object.
    - The title must also carry every **number** the instruction names: size,
      dimension, count, volume, model number. `10x6.5ft` is not `6 foot by 4
      foot`; `7x5ft` is not `7 foot by 7 foot`; `(2 pack)` is not `(3 pack)`;
      `12 count` is not `24 count`. A listing whose numbers differ is a
      different variant and scores 0, however close its other words look.
    - The title should carry the instruction's **attribute values** too - the
      colour, flavour, style, material or model the instruction names
      (`silver`, `green`, `bakery`, `fluoride free`, `old world style`). When
      the listing shows its option groups on the product page, the option the
      instruction names must be among them; if it is not, another listing
      carries it.
    - Only plain filler adjectives are **not** part of this test: `high`,
      `performance`, `screen`, `quality`, `long lasting`, `easy to assemble`,
      `cruelty free`, `relaxed fit`, `accessories`. Store titles often omit
      those on the correct product too, so do not reject a listing over one of
      them alone. Every other word the instruction names - the object type, the
      numbers, the attribute values - is evidence, and its absence means the
      listing is a look-alike.
    - Preference order on a result page: the title that carries the object type
      **and** the instruction's numbers **and** the most of its other words.
      Do not settle for the first plausible listing.
    - Reject look-alikes that are a different kind of object:
      * a **case / cover / sleeve / screen protector / strap / holder / mount /
        replacement part / battery / charger** is NOT the phone, tablet or watch
        itself;
      * a **chips / salt / seasoning / nuts / flour** item is NOT the bagel;
      * a listing for a different size, width, capacity or model is a different
        product even if the brand matches.
    - Open the chosen listing with `click_action(value="<lowercase asin>")`.
      Open **one** listing at a time, and never re-open a listing you already
      opened and rejected: the second look shows the identical page and costs a
      turn.
    - You may return to the search list from a product page **at most three
      times** in an episode. Past that you must buy from the page you are on.
      Retreating over and over is how episodes end with nothing bought.

3. PRODUCT PAGE - CHECK THE PRICE, THEN FILL EVERY NAMED ATTRIBUTE
    - Read `Price: $X`. If X is above the instruction's limit the purchase can
      never be correct: click `back to search` and pick a cheaper product.
    - The header before the title lists every attribute group (`color`, `size`,
      `style`, `flavor name`, ...) together with its option values. ONE option
      of every group the instruction names must be clicked before buying:
      * click the option value that matches the instruction: instruction
        "i want the bakery flavor" -> `click_action(value="bakery")`;
        instruction "please choose black one" -> `click_action(value="black")`;
        instruction "size 9" -> the option carrying `9`.
      * a number the instruction spells out is the digits in the option:
        instruction "ten pack ... 3 feet" -> the option carrying both `10`
        and `3 feet` (`3 feet (10 pack)`), NOT `3 feet (3 pack)`.
      * when the `[SHOP CHECK]` block names the option for a group (`group
        'size': the instruction's value is option "..."`), click exactly that
        value before `buy now`; when it reports no matching option, do not
        click the nearest one instead.
      * click the option VALUE, never the group label such as `color`.
      * one option per turn; the page never displays what is already selected,
        so keep your own list of clicked options.
      * an option value can be a compound label such as
        `38 | 40 | 41mm s | m (small)`; click the whole string.
    - If the instruction names a value and **no option on the page is that
      value** (e.g. the instruction says `black` but the only options are
      `gold` and `rose gold`), this product cannot satisfy the instruction:
      click `back to search` and look for a listing that can. If the
      instruction names no value for a group, do not leave: pick the option
      whose words come closest to the instruction and then buy.
    - If the page shows **no attribute group at all**, there is nothing to
      select: the words of the title and the price are the whole decision.
    - The buy test is a checklist, not a feeling. Buy only when the page
      carries (a) the **kind of object** the instruction names, (b) every
      **number** it names, and (c) every **attribute value** it names as a
      clickable option. If (a), (b) or (c) fails, this page cannot score:
      click `back to search` and open a **different** listing. A title that
      omits only a filler adjective (`high`, `quality`, `long lasting`) is
      still the product - do not leave over that alone.
    - Retreading the same pages does not help. A listing you have already
      opened and left is identical the second time, and a search you have
      already sent returns the identical page.
4. BUY
   - Click `buy now` as soon as (a) the product is the one the instruction
     describes and (b) every attribute group it names has one clicked option
     and (c) the price is inside the budget.
   - Do not end the episode without buying. Buying a product that is a
     reasonable match is better than running out of turns: both score zero, but
     the purchase is the only one that can score one.

## Reading the observation

- The instruction is echoed at the top of every observation, between the first
  `[SEP]` markers. Re-read it before every action.
- The `Available Actions` block lists exactly the strings that `click_action`
  accepts. Any other string is not an action of the page; the environment then
  runs it as a **search** instead, so a "product page" that looks like a list
  of ten listings means your click opened a search, not a product.
- On the **search-bar page** (the page with `has_search_bar: True`, reached
  with `back to search`) only `search` works. `< prev` does not exist there.
- **`description`, `features` and `reviews` only re-print text about the
  product you already opened.** They never change the price, the options or
  the product, so they are never useful for verifying anything. Do not click
  them.
- A `[SHOP CHECK]` block may be appended to an observation. It is factual
  feedback about the page you are looking at: the budget, the attribute groups
  and their option values, which instruction words this page does and does not
  carry, and how many turns are left. Use it; you still decide the action. When
  the turns run short, a retreat it reports is served as `buy now` instead, so
  the episode still ends in a scored purchase.

## Recurring mistakes that lose the episode

- Buying an accessory, a different model, or a different size than the
  instruction names.
- Buying a look-alike: a listing whose title does **not** carry the
  instruction's numbers (a `10x6.5ft` backdrop for a `6 foot by 4 foot` one) or
  whose option groups do **not** offer the value the instruction names (no
  `green` in a hoodie's colour list, no `silver` on an earbud page). This is
  the mistake behind most lost episodes: the purchase ends the episode, so a
  wrong page scores 0. Check the numbers and the named values first.
- Re-opening a listing you already opened and left: the page is identical and
  the turn is gone.
- Clicking `buy now` immediately after opening a product page whose attribute
  groups are still unset.
- Clicking a group label (`color`) instead of an option value (`black`).
- Buying a product whose price is above the instruction's limit.
- Paging through empty result pages, or re-sending the same search keywords,
  until the twenty-turn budget is gone.
- Clicking `< prev` on the search-bar page, where it does not exist.
- Looping: re-issuing an action that already produced the same observation. If
  you catch yourself repeating, buy the page you are on.
