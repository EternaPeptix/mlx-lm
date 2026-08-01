from __future__ import annotations

import importlib.util
import json
import subprocess
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

    def test_break_even_distinguishes_emitted_from_accepted_drafts(self):
        emitted = _MODULE.emitted_tokens_break_even(
            target_step_ms=130.0,
            draft_step_ms=10.0,
            replay_step_ms=1.0,
        )
        accepted = _MODULE.accepted_draft_tokens_break_even(
            target_step_ms=130.0,
            draft_step_ms=10.0,
            replay_step_ms=1.0,
        )

        self.assertAlmostEqual(emitted, 2.397)
        self.assertAlmostEqual(accepted, 1.397)
        self.assertAlmostEqual(
            _MODULE.acceptance_break_even(
                target_step_ms=130.0,
                draft_step_ms=10.0,
                replay_step_ms=1.0,
            ),
            emitted,
        )

    def test_expected_prefix_uses_survival_not_independent_probabilities(self):
        tier = _MODULE.SpeculativeTier(
            verify_width=3,
            target_step_ms=132.64,
            draft_step_ms=8.0,
            replay_step_ms=0.0,
        )

        estimate = _MODULE.estimate_speculative_tier(tier, (0.9, 0.72))

        self.assertAlmostEqual(estimate.expected_accepted_draft_tokens, 1.62)
        self.assertAlmostEqual(estimate.expected_emitted_tokens, 2.62)
        self.assertAlmostEqual(
            estimate.expected_tokens_per_second,
            2.62 / 0.14064,
        )

    def test_cost_aware_policy_can_select_ordinary_width_three_or_width_eight(self):
        tiers = (
            _MODULE.SpeculativeTier(3, 132.64, 8.0, 0.0),
            _MODULE.SpeculativeTier(8, 255.13, 8.0, 0.0),
        )

        low, _ = _MODULE.select_cost_aware_tier(
            ordinary_step_ms=69.893,
            tiers=tiers,
            acceptance_survival=(0.4, 0.1, 0.05, 0.02, 0.01, 0.0, 0.0),
        )
        medium, _ = _MODULE.select_cost_aware_tier(
            ordinary_step_ms=69.893,
            tiers=tiers,
            acceptance_survival=(0.95, 0.75, 0.35, 0.15, 0.05, 0.02, 0.01),
        )
        high, _ = _MODULE.select_cost_aware_tier(
            ordinary_step_ms=69.893,
            tiers=tiers,
            acceptance_survival=(0.99, 0.96, 0.92, 0.88, 0.84, 0.80, 0.76),
        )

        self.assertEqual(low.name, "ordinary")
        self.assertEqual(medium.name, "width3")
        self.assertEqual(high.name, "width8")

    def test_survival_curve_and_tier_contract_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "non-increasing"):
            _MODULE.validate_acceptance_survival((0.5, 0.6))
        with self.assertRaisesRegex(ValueError, "shorter"):
            _MODULE.estimate_speculative_tier(
                _MODULE.SpeculativeTier(3, 1.0, 1.0, 1.0),
                (0.5,),
            )
        with self.assertRaisesRegex(ValueError, "at least two"):
            _MODULE.validate_tier(_MODULE.SpeculativeTier(1, 1.0, 0.0, 0.0))

    def test_cli_reports_corrected_break_even_and_selected_tier(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(_BENCHMARK),
                "--target-step-ms",
                "132.64",
                "--draft-step-ms",
                "8",
                "--ordinary-step-ms",
                "69.893",
                "--acceptance-survival",
                "0.95,0.75,0.35,0.15,0.05,0.02,0.01",
                "--tier",
                "3:132.64:8:0",
                "--tier",
                "8:255.13:8:0",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)

        self.assertAlmostEqual(
            payload["throughput_model"][
                "accepted_draft_tokens_per_step_required_for_17_tps"
            ],
            1.39088,
        )
        self.assertEqual(payload["adaptive_tier_model"]["selected"], "width3")


if __name__ == "__main__":
    unittest.main()
