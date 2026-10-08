"""Prove the amendment changes only the hint word cap, not ordinary traces."""
import importlib.util
import unittest
from types import SimpleNamespace

import corrected_evaluate
import protocol


class RuntimeTests(unittest.TestCase):
    def test_long_hint_allowed_and_single_string_shape_retained(self):
        for layer in ("h4", "h5"):
            session = SimpleNamespace(trace={"h4": [], "h5": []}, turn=1)
            hints = ["word " * 300]
            self.assertEqual(corrected_evaluate.uncapped_hints(session, layer, hints), hints)
            self.assertEqual(session.trace[layer], [{"turn": 1, "hints": hints}])
        for value in (None, "text", ["one", "two"], [3]):
            with self.assertRaises(ValueError):
                corrected_evaluate.uncapped_hints(SimpleNamespace(), "h5", value)

    def test_short_and_empty_hints_identical_to_frozen_session(self):
        path = protocol.HISTORICAL_ROOT / "task_snapshot/src/server/harness/session.py"
        spec = importlib.util.spec_from_file_location("frozen_session", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for hints in ([], [""], ["a b c"], ["word " * 120]):
            for layer in ("h4", "h5"):
                old = SimpleNamespace(trace={"h4": [], "h5": []}, turn=3)
                new = SimpleNamespace(trace={"h4": [], "h5": []}, turn=3)
                expected = module.FourHookSession._hints(old, layer, hints)
                actual = corrected_evaluate.uncapped_hints(new, layer, hints)
                self.assertEqual(expected, actual)
                self.assertEqual(old.trace, new.trace)

    def test_amendment_hashes_are_bound(self):
        import runtime_support
        amendment = runtime_support.bind_amendment()
        self.assertEqual(amendment["frozen_core_sha256"]["driver.py"],
                         protocol.digest(protocol.HERE / "driver.py"))
        self.assertIn("identical cached proposal", amendment["rerun"])


if __name__ == "__main__":
    unittest.main()
