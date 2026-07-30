from __future__ import annotations

import os
import unittest

import mlx.core as mx

from mlx_lm.models.kimi_k3 import (
    KimiK3SparseMoE,
    TextArgs,
    _group_expert_select,
)
from mlx_lm.models.kimi_k3_fused_router import (
    FUSED_ROUTER_ENV,
    _metal_available,
    fused_k3_router_enabled,
    maybe_fused_k3_router,
)


def _reference(gates, bias):
    return _group_expert_select(gates, bias, 16, 1, 1, 1.0, True)


def _candidate(gates, bias, **overrides):
    parameters = {
        "top_k": 16,
        "n_group": 1,
        "topk_group": 1,
        "routed_scaling_factor": 1.0,
        "renormalize": True,
        "training": False,
    }
    parameters.update(overrides)
    return maybe_fused_k3_router(gates, bias, **parameters)


def _assert_weights_equal(test, reference, candidate):
    equal = (reference == candidate) | (mx.isnan(reference) & mx.isnan(candidate))
    test.assertTrue(bool(mx.all(equal).item()))


def _released_router_sparse_moe():
    args = TextArgs(
        hidden_size=64,
        intermediate_size=128,
        num_experts=896,
        num_experts_per_token=16,
        num_expert_group=1,
        topk_group=1,
        num_shared_experts=0,
        moe_intermediate_size=16,
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
    )
    module = KimiK3SparseMoE(args)
    module.set_dtype(mx.bfloat16)
    module.e_score_correction_bias = mx.zeros((896,), dtype=mx.float32)
    module.eval()
    mx.eval(module.parameters())
    return module


@unittest.skipUnless(_metal_available(), "requires Metal")
class FusedRouterTest(unittest.TestCase):
    def setUp(self):
        os.environ[FUSED_ROUTER_ENV] = "1"
        fused_k3_router_enabled.cache_clear()

    def tearDown(self):
        os.environ.pop(FUSED_ROUTER_ENV, None)
        fused_k3_router_enabled.cache_clear()

    def assertExact(self, gates, bias):
        reference_indices, reference_weights = _reference(gates, bias)
        candidate = _candidate(gates, bias)
        self.assertIsNotNone(candidate)
        candidate_indices, candidate_weights = candidate
        mx.eval(
            reference_indices,
            reference_weights,
            candidate_indices,
            candidate_weights,
        )
        self.assertTrue(
            bool(mx.all(reference_indices == candidate_indices).item()),
            (reference_indices.tolist(), candidate_indices.tolist()),
        )
        _assert_weights_equal(self, reference_weights, candidate_weights)

    def test_random_scores_and_bias_are_bit_exact(self):
        for seed in range(32):
            with self.subTest(seed=seed):
                mx.random.seed(seed)
                gates = mx.random.normal((1, 1, 896), dtype=mx.bfloat16)
                bias = 0.25 * mx.random.normal((896,), dtype=mx.float32)
                self.assertExact(gates, bias)

    def test_batched_decode_rows_are_bit_exact(self):
        mx.random.seed(127)
        gates = mx.random.normal((3, 1, 896), dtype=mx.bfloat16)
        bias = 0.25 * mx.random.normal((896,), dtype=mx.float32)
        self.assertExact(gates, bias)

    def test_sparse_moe_integration_is_bit_exact(self):
        mx.random.seed(211)
        module = _released_router_sparse_moe()
        gates = mx.random.normal((1, 1, 64), dtype=mx.bfloat16)

        os.environ.pop(FUSED_ROUTER_ENV)
        fused_k3_router_enabled.cache_clear()
        reference = module(gates)

        os.environ[FUSED_ROUTER_ENV] = "1"
        fused_k3_router_enabled.cache_clear()
        candidate = module(gates)
        mx.eval(reference, candidate)
        self.assertTrue(bool(mx.all(reference == candidate).item()))

    def test_ties_use_lower_expert_index(self):
        gates = mx.zeros((1, 1, 896), dtype=mx.bfloat16)
        bias = mx.zeros((896,), dtype=mx.float32)
        self.assertExact(gates, bias)

        gates[0, 0, 100:120] = 1
        self.assertExact(gates, bias)

        gates = mx.zeros((1, 1, 896), dtype=mx.bfloat16)
        bias[200:225] = 2
        self.assertExact(gates, bias)

    def test_extremes_and_nan_follow_stock_argpartition(self):
        gates = mx.zeros((1, 1, 896), dtype=mx.bfloat16)
        bias = mx.zeros((896,), dtype=mx.float32)
        gates[0, 0, 5] = float("nan")
        gates[0, 0, 7] = float("inf")
        gates[0, 0, 9] = -float("inf")
        bias[11] = float("nan")
        bias[13] = float("inf")
        bias[15] = -float("inf")
        self.assertExact(gates, bias)

        all_nan = mx.full((1, 1, 896), float("nan"), dtype=mx.bfloat16)
        self.assertExact(all_nan, mx.zeros((896,), dtype=mx.float32))

    def test_disabled_by_default(self):
        os.environ.pop(FUSED_ROUTER_ENV)
        fused_k3_router_enabled.cache_clear()
        gates = mx.zeros((1, 1, 896), dtype=mx.bfloat16)
        bias = mx.zeros((896,), dtype=mx.float32)
        self.assertIsNone(_candidate(gates, bias))

    def test_unsupported_contracts_fall_back(self):
        gates = mx.zeros((1, 1, 896), dtype=mx.bfloat16)
        bias = mx.zeros((896,), dtype=mx.float32)
        cases = (
            (mx.zeros((1, 2, 896), dtype=mx.bfloat16), bias, {}),
            (mx.zeros((1, 1, 895), dtype=mx.bfloat16), bias[:-1], {}),
            (gates.astype(mx.float32), bias, {}),
            (gates, bias.astype(mx.bfloat16), {}),
            (gates, None, {}),
            (gates, bias, {"top_k": 8}),
            (gates, bias, {"n_group": 8}),
            (gates, bias, {"topk_group": 2}),
            (gates, bias, {"routed_scaling_factor": 2.5}),
            (gates, bias, {"renormalize": False}),
            (gates, bias, {"training": True}),
        )
        for candidate_gates, candidate_bias, overrides in cases:
            with self.subTest(
                shape=candidate_gates.shape,
                dtype=candidate_gates.dtype,
                overrides=overrides,
            ):
                self.assertIsNone(
                    _candidate(candidate_gates, candidate_bias, **overrides)
                )


if __name__ == "__main__":
    unittest.main()
