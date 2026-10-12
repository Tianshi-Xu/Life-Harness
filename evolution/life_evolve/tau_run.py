"""Bind current Life validation to immutable candidate composition."""
import sys
from pathlib import Path
from evolution.life_evolve import profile
from evolution.tau_comparison.common import ROOT, load_module

COMPOSITION = ROOT / "evolution/reproduction/life_runtime"


def composition_module():
    # Reuse proven immutable staging/CAS/commit, not a historical optimizer/runtime.
    sys.path.insert(0, str(COMPOSITION))
    module = load_module("life_tau_composition", COMPOSITION / "composition.py")

    def validate(document, domain, records, snapshot, workspace):
        bundle, rejected = profile.filter_operations(document, domain, records, snapshot, workspace)
        if rejected:
            raise ValueError("qualified proposal changed before staging")
        for op in bundle["operations"]:
            if op["op"] != "delete":
                op["artifact_path"] = Path(workspace) / op["artifact"]
        return bundle

    module.validate_bundle = validate
    return module
