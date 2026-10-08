# ALFWorld playbook

This is a text household simulator. A single episode is one goal, e.g.
"put some pen on sidetable", "cool some mug and put it in coffeemachine",
"put a clean soapbar in cart", "put two keychain in sofa". The environment
stops the episode after roughly 50 actions: an unfinished task is a lost task,
so every action has to either reveal a new receptacle or advance the goal.

## Goal families (read the goal line first)

- **put one object**: `put some pen on sidetable` -> carry one pen to the sidetable.
- **put two objects**: `find two cloth and put them in shelf`, `put two kettle in
  cabinet` -> the goal needs **two different objects of the same type**. Put the first
  one at the goal place, then go and find a SECOND, different one (e.g. `kettle 2` when
  `kettle 1` is already delivered). `placed=k/2` in the [PLAN] block tracks progress.
- **clean / heat / cool then put**: `cool some pot and put it in cabinet` -> take the
  object to the appliance first (`sinkbasin` to clean, `microwave`/`stoveburner` to
  heat, `fridge` to cool), apply the change, and only then carry it to the goal place.
- **examine with a tool / look at under a light**: `examine the pillow with the desklamp`
  and `look at the pen under the desklamp` are the same family. Take the object, walk to
  the surface that carries the lamp (`use <lamp>` is ONLY offered there), `use <lamp>` to
  switch it on - exactly once, because a lamp is a toggle and a second `use` turns it back
  off - and then `examine` the object. Never `put` the object down anywhere else.

## The destination rule (biggest source of lost episodes)

The place named at the **end** of the goal is the ONLY place that counts.

- Put the goal object there - never "park" it back where you found it, never put it on a
  nearer surface, never put it in the first container you happen to pass.
- A placement anywhere else does not count and usually loses the object. The harness will
  rewrite any `put <goal object>` that targets a different place.
- Once an object of the goal type sits at the goal place it is done. **Never take it out
  again** - the object you still need is elsewhere. Re-taking a delivered object is the
  single most common way these episodes run out of budget.

## Action grammar (exact forms that appear in AVAILABLE ACTIONS)

- `look` - re-describe the current location.
- `inventory` - list what you are carrying (max one object).
- `go to <place>` - move. You cannot "go to" the place you are already at.
- `examine <place>` - list what is on/inside a place.
- `open <container>` / `close <container>` - drawers, cabinets, fridge, microwave.
- `take <object> from <place>` - pick up an object (needs an empty hand).
- `put <object> in/on <place>` - put down what you carry (needs to actually carry it).
- `heat <obj> with <microwave|stoveburner> 1`, `cool <obj> with fridge 1`,
  `clean <obj> with sinkbasin 1`.

The last lines of the AVAILABLE ACTIONS block are the important ones: the
`take ... from ...` lines name every object that is within reach *right now*.
Scan those lines before choosing a move - they tell you where an object is
without any guessing. When a `take <goal object> ...` line appears, take it.

## Search order for a missing object

Search by receptacle type, most likely first (this is the order the [PLAN] block
uses):

- **clean / heat / cool goals, first of all**: the object is a kitchen item, so sweep the
  open kitchen surfaces before anything else - countertop, stoveburner, sinkbasin,
  diningtable, coffeemachine, toaster, shelf. Only then fall back to the class list below.
- food / produce (lettuce, tomato, potato, egg, apple, bread, ...):
  fridge, countertop, diningtable, microwave, sinkbasin, cabinet.
- kettle / teapot (heatable vessels): countertop, stoveburner, sinkbasin, diningtable,
  coffeemachine, shelf, cabinet.
- crockery and other drinks (mug, cup, glass, vase, bottle, pitcher):
  cabinet, countertop, shelf, coffeetable, diningtable, coffeemachine.
- cutlery (spoon, fork, knife, butterknife):
  drawer, countertop, diningtable, sinkbasin, cabinet.
- stationery and small flat items (pen, pencil, book, newspaper, paper):
  drawer, desk, sidetable, coffeetable, shelf, countertop.
- hygiene / cleaning (soapbar, soapbottle, spraybottle, cloth, dishsponge,
  toiletpaper, tissuebox): cabinet, countertop, drawer, sinkbasin,
  bathtubbasin, toilet, towelholder, handtowelholder, toiletpaperhanger.
- small valuables and gadgets (watch, alarmclock, keychain, creditcard,
  cellphone, cd, remotecontrol): drawer, sidetable, dresser, desk,
  coffeetable, shelf, sofa, armchair, tvstand.
- decoration (statue, houseplant, picture, television, laptop, box):
  shelf, cabinet, coffeetable, dresser, desk, sidetable, tvstand.

Surfaces are usually open (going there shows the contents directly). Containers
(drawer, cabinet, fridge, microwave) must be `open`ed first - open every closed
receptacle you arrive at, otherwise its contents stay invisible. Because a container
costs two actions and a surface costs one, surfaces are worth checking first whenever
you have no better information.

A *bank* of identical containers (e.g. 20 cabinets or 12 drawers in a large kitchen)
is the single most expensive thing in a scene: ruling out every member costs far more
than the whole 50-action budget, and the goal object is very rarely inside one of them.
The [PLAN] block therefore schedules a container kind with three or more instances
**after** every other receptacle, and it still visits a lone container (the one fridge,
the one microwave, a single drawer) in its normal position, because those carry the
strong object-specific priors (food lives in the fridge). Follow the `[PLAN]` "next
action" line instead of walking a bank of cabinets one by one; if you do end up in a
bank, that means every surface has already been checked.

## Re-visiting what you have already seen

Keep a mental list of every object you have laid eyes on and where it was.

- If the goal object appeared in an earlier observation, **go straight back to that place**.
  The harness tracks this too and will say
  `[PLAN] you already saw the <object> at <place> -> go back there first`; when that line
  appears, that is your next move.
- This matters most for `two <object>` goals: the second object is often sitting in the same
  receptacle as the first one. After delivering `kettle 1` from `cabinet 1`, look at what
  was *also* in `cabinet 1` before you open anything else - do not restart the search in
  new containers.
- A place you have already visited is still allowed to be visited again. "Already checked"
  only means you looked there; it does not mean the object you still need is gone.

## Common mistakes to avoid

- Calling a tool named after the action instead of `take_action`.
- Taking an object whose type is not the goal type: `soapbar` != `soapbottle`,
  `pen` != `keychain`, `statue` != `watch`, `butterknife` != `spoon`,
  `cup` != `potato`, `pencil` != `pen`. Taking the wrong object is never
  progress; keep searching.
- `put` before `take` (you cannot put an object you are not carrying).
- Repeating the same action when the observation did not change, or bouncing
  between two places that you have already inspected.
- Forgetting the transformation step on "clean / heat / cool" goals, or moving
  the object to the goal place before the transformation is applied.
- Saying "done" instead of finishing the placement. The environment ends the
  episode by itself once the goal is reached.
