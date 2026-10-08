"""Run the frozen evaluator with the declared hint contract at the Task boundary."""
from copy import deepcopy

import agentbench_evaluate as evaluator
import protocol
import runtime_support

ORIGINAL_INITIALIZE = evaluator.initialize
ORIGINAL_ATOMIC = evaluator.atomic_json


def uncapped_hints(self, layer, hints):
    if not (isinstance(hints, list) and len(hints) <= 1
            and all(isinstance(value, str) for value in hints)):
        raise ValueError("H4/H5 require at most one string hint")
    if hints:
        self.trace[layer].append({"turn": self.turn, "hints": deepcopy(hints)})
    return hints


def initialize(domain, split, chain_manifest):
    ORIGINAL_INITIALIZE(domain, split, chain_manifest)
    from src.server.harness.session import FourHookSession
    FourHookSession._hints = uncapped_hints


def atomic_json(path, value):
    if path.name == "config.json":
        ORIGINAL_ATOMIC(path.parent / "runtime_amendment.json",
                        {"config_sha256": value["config_sha256"],
                         "runtime_amendment_sha256": protocol.digest(protocol.HERE / "runtime_amendment.json")})
    return ORIGINAL_ATOMIC(path, value)


if __name__ == "__main__":
    runtime_support.bind_amendment()
    evaluator.initialize = initialize
    evaluator.atomic_json = atomic_json
    evaluator.validate_cached_summary = runtime_support.verify_evaluation
    evaluator.main()
