import os
import unittest

import mlx.core as mx

from mlx_lm.models.kimi_k3_prefill_route_combine import (
    PREFILL_ROUTE_COMBINE_ENV,
    fused_sorted_route_combine,
    prefill_route_combine_enabled,
)


class TestKimiK3PrefillRouteCombine(unittest.TestCase):
    def tearDown(self):
        os.environ.pop(PREFILL_ROUTE_COMBINE_ENV, None)
        prefill_route_combine_enabled.cache_clear()

    def test_env_is_strict_and_default_off(self):
        self.assertFalse(prefill_route_combine_enabled())
        prefill_route_combine_enabled.cache_clear()
        os.environ[PREFILL_ROUTE_COMBINE_ENV] = "1"
        self.assertTrue(prefill_route_combine_enabled())
        prefill_route_combine_enabled.cache_clear()
        os.environ[PREFILL_ROUTE_COMBINE_ENV] = "yes"
        with self.assertRaisesRegex(ValueError, "must be 0 or 1"):
            prefill_route_combine_enabled()

    @unittest.skipUnless(mx.metal.is_available(), "Metal is required")
    def test_fused_combine_matches_bfloat16_top16_reduction(self):
        tokens = 2
        routes = tokens * 16
        width = 3584
        mx.random.seed(20260805)
        unsorted = mx.random.uniform(
            low=-1.0,
            high=1.0,
            shape=(tokens, 16, width),
        ).astype(mx.bfloat16)
        weights = mx.random.uniform(
            low=0.0,
            high=1.0,
            shape=(tokens, 16),
        ).astype(mx.bfloat16)
        weights = (
            weights / weights.astype(mx.float32).sum(axis=-1, keepdims=True)
        ).astype(mx.bfloat16)
        order = (mx.arange(routes, dtype=mx.uint32) * 17 + 13) % routes
        inverse = mx.argsort(order)
        sorted_routes = unsorted.reshape(routes, width)[order]

        expected = (unsorted * weights[..., None]).sum(axis=-2)[None]
        actual = fused_sorted_route_combine(sorted_routes, inverse, weights)
        mx.eval(expected, actual)
        self.assertTrue(mx.array_equal(expected, actual).item())

    @unittest.skipUnless(mx.metal.is_available(), "Metal is required")
    def test_fused_combine_rejects_non_bfloat16_weights(self):
        routes = mx.zeros((16, 3584), dtype=mx.bfloat16)
        inverse = mx.arange(16, dtype=mx.uint32)
        weights = mx.ones((1, 16), dtype=mx.float32)
        with self.assertRaisesRegex(ValueError, "invalid K3 BF16"):
            fused_sorted_route_combine(routes, inverse, weights)


if __name__ == "__main__":
    unittest.main()
