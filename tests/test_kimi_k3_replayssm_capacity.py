from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

_BENCHMARK = Path(__file__).parents[1] / "benchmarks" / "kimi_k3_replayssm_capacity.py"
_SPEC = importlib.util.spec_from_file_location("kimi_k3_replayssm_capacity", _BENCHMARK)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


class ReplaySSMCapacityModelTest(unittest.TestCase):
    def test_tp2_width_three_reduces_transaction_scratch(self):
        result = _MODULE.capacity_model()

        self.assertEqual(result.full_state_history_bytes, 325_582_848)
        self.assertEqual(result.raw_history_bytes, 5_097_168)
        self.assertEqual(result.conv_history_bytes, 11_446_272)
        self.assertGreater(result.reduction_ratio, 20.0)

    def test_acceptance_break_even_includes_all_three_step_costs(self):
        required = _MODULE.acceptance_break_even(
            target_step_ms=130.0,
            draft_step_ms=10.0,
            replay_step_ms=1.0,
        )
        self.assertAlmostEqual(required, 2.397)


if __name__ == "__main__":
    unittest.main()
