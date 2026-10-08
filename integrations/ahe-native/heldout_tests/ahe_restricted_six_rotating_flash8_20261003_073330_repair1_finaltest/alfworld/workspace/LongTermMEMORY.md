# Long-term memory

Recurring pitfalls observed in this environment (ALFWorld text transport, tool: `take_action`).

1. The only tool is `take_action`. Its name never changes. The action text belongs in the
   `action` argument - it is NOT the tool name. A call named "go to sidetable 1" fails with
   "Tool not found" and the episode never advances.
2. Every action must be copied verbatim from the most recent "AVAILABLE ACTIONS" list.
   Actions that are not on that list are ignored / counted as invalid.
3. Repeating an action never helps in this deterministic world. If the observation did not
   change, pick something else.
4. `put` requires `take` first; the hand holds at most one object.
5. `clean` / `heat` / `cool` goals need the transformation step before the final placement,
   and the transformation happens at the appliance (`sinkbasin` to clean, `microwave` or
   `stoveburner` to heat, `fridge` to cool) - go there while holding the object.
6. Match the object type exactly: pen vs pencil, cup vs mug, soapbar vs soapbottle vs
   spraybottle, statue vs watch, butterknife vs spoon. An object of the wrong type is never
   progress, however plausible it looks.
7. The episode has a hard budget of about 50 actions. Finishing the search is more important
   than handling anything you have already found: sweep the unchecked receptacles, opening
   closed containers, and never return to a place you have already inspected.
8. The observation ends with the AVAILABLE ACTIONS list, and the `take ... from ...` lines
   near the end of that list name every object within reach. If the goal object appears
   there, take it instead of walking anywhere else.
9. Each observation also carries a `[PLAN]` block (goal, carried object, placed count,
   unchecked places, recommended next action). It is maintained by the harness and is
   always accurate; prefer its "next action" line over your own recollection.
10. A goal that says "two" needs two DIFFERENT objects of that type: after delivering
    `kettle 1` the next object is `kettle 2`, which is some other place. `placed=k/n` in
    the [PLAN] block shows how many are done.
11. The only place that counts is the place named at the end of the goal. Putting the goal
    object back where you found it, or on any other surface/container, loses it. Never
    take an object of the goal type out of the goal place once it is delivered - that is
    the classic way to loop `take kettle 1 from cabinet 1` / `put kettle 1 in/on cabinet 1`
    until the budget runs out.
12. "examine X with Y" and "look at X under Y" are the same light family (different from
    every "put" goal): take X, carry it to Y (usually a `desklamp`, which sits on a
    table/sidetable), switch Y on with `use Y` - that action only appears while you stand
    at the surface carrying Y - and then `examine X`. Y is a toggle, so `use` it exactly
    once; a second `use` switches the light back off. Do not `put` X anywhere else; only
    if the `[PLAN]` line explicitly asks for it is X left on the lamp's surface.
13. Remember where you have already seen things. If an object of the goal type appeared in
    an earlier observation, going back to that place beats opening containers you have
    never seen. This is the difference between a 12-action episode and a lost one. For a
    "two <object>" goal the second object is frequently in the very same receptacle as
    the first one, so re-read the contents you already saw before searching anything new.
14. An open surface (`countertop`, `diningtable`, `stoveburner`, `shelf`, `sinkbasin`,
    `coffeemachine`, `toaster`) reveals its contents in one action (`go to`), while a
    closed container (`cabinet`, `drawer`, `fridge`, `microwave`) takes two (`go to` +
    `open`). With no other information, sweep the surfaces first: in a room with a dozen
    cabinets, opening every cabinet before looking at the countertops can consume the whole
    budget on its own. The [PLAN] block enforces this - a container kind that the room
    offers three or more times (a bank of 20 cabinets, 12 drawers) is scheduled after
    every other receptacle, while a *lone* container keeps its normal position so that
    object-specific priors still work (food is in the one fridge). If the "next action"
    line walks you into a bank of containers, every surface has already been checked.
15. A goal that says clean / heat / cool can only name a kitchen item, and those objects
    are found on the kitchen surfaces (`countertop`, `stoveburner`, `sinkbasin`,
    `diningtable`, `coffeemachine`, `toaster`, `shelf`), not buried in cabinets. Sweep
    those first, then apply the transformation at the appliance, then carry the object to
    the goal place.
