"""Run the existing bundle, hook and proposer regressions against continuation code."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from unittest.mock import patch

import agentbench_chain
import agentbench_evaluate
import candidate_contract
import driver
import protocol
import status

previous_path = list(sys.path)
spec = importlib.util.spec_from_file_location("life_shared_regressions",
    protocol.HERE.parent / "life_agentbench_bundle_five_mr_filter_20261001/test_adapter.py")
original = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = original
spec.loader.exec_module(original)
sys.path[:] = previous_path


def empty_state(domain):
    return {"status": "running", "domain": domain, "round": 0, "batch": 0,
            "active_snapshot": [], "active_snapshot_sha256": driver.snapshot_sha256([]),
            "accepted_bundle_ids": [], "operation_log": [], "rounds": [],
            "baseline_phase": None, "baseline_summary": None}


def set_up(self):
    fixture = patch.object(driver, "initial_state", empty_state)
    fixture.start()
    self.addCleanup(fixture.stop)


for name in ("BundleTests", "HookContractTests", "ProposerIsolationTests"):
    globals()[name] = type(name, (getattr(original, name),),
                          {"__module__": __name__, "setUp": set_up})


class CoreRules(unittest.TestCase):
    def test_strict_acceptance_and_rotation(self):
        original.ProtocolTests("test_strict_acceptance_and_rotation").test_strict_acceptance_and_rotation()

    def test_episode_cache_includes_infrastructure_failure(self):
        original.ReportingTests("test_episode_cache_requires_schema_index_and_config_hash").test_episode_cache_requires_schema_index_and_config_hash()

    def test_evaluator_rejects_unknown_split(self):
        original.ReportingTests("test_evaluator_rejects_unknown_split_before_loading_tasks").test_evaluator_rejects_unknown_split_before_loading_tasks()


if __name__ == "__main__":
    unittest.main()
