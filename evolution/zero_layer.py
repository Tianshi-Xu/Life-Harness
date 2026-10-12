"""Zero-state layer for --from-scratch runs (life_loop chain layer 0).

Clears all released H2-H5 harness CONTENT while leaving the hook machinery
intact: the mixin protocols, tool-description mixins, and skill retrieval
still work — there is simply no released content loaded. The loop evolves
all harness content from zero via subsequent plugin layers.
"""


def register() -> None:
    from tau2.harness.base import HarnessedToolKitMixin
    # This built-in H4 intervention is content too. Keep tracking machinery,
    # but disable its released alert threshold until an evolved layer sets it.
    HarnessedToolKitMixin._STUCK_LOOP_THRESHOLD = float("inf")
    import tau2.harness.airline as airline_h
    import tau2.harness.banking_knowledge as banking_h
    import tau2.harness.h3_tools as h3_tools
    import tau2.harness.retail as retail_h
    import tau2.harness.skills as skills
    import tau2.harness.telecom as telecom_h

    # H2 rules / H4 annotators: class-level registries on the Tools classes.
    for mod in (airline_h, retail_h, telecom_h, banking_h):
        for obj in vars(mod).values():
            if not isinstance(obj, type):
                continue
            for attr in ("harness_rules", "harness_annotators"):
                registry = obj.__dict__.get(attr)
                if isinstance(registry, dict):
                    registry.clear()

    # H3 hints: module-level per-domain hint dicts.
    for name, obj in vars(h3_tools).items():
        if isinstance(obj, dict) and name.endswith("_H3_HINTS"):
            obj.clear()

    # H5 skill bank (rebinding is read at call time by _build_index).
    skills.DOMAIN_SKILLS = {}
