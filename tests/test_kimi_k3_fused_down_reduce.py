from __future__ import annotations

import unittest

import mlx.core as mx

from mlx_lm.models.kimi_k3_derived_bias import derived_affine2_biases
from mlx_lm.models.kimi_k3_fused_down_reduce import (
    K3_DOWN_INPUT_WIDTH,
    K3_DOWN_OUTPUT_WIDTH,
    K3_TOP_K,
    _metal_available,
    fused_down_reduce_decode,
    supports_fused_down_reduce,
)


def _packed_projection(
    *,
    experts: int = K3_TOP_K,
) -> tuple[mx.array, mx.array, mx.array]:
    weight = mx.full(
        (experts, K3_DOWN_OUTPUT_WIDTH, K3_DOWN_INPUT_WIDTH // 16),
        0x24681357,
        dtype=mx.uint32,
    )
    scales = mx.full(
        (experts, K3_DOWN_OUTPUT_WIDTH, K3_DOWN_INPUT_WIDTH // 128),
        0.015625,
        dtype=mx.bfloat16,
    )
    return weight, scales, mx.zeros_like(scales)


def _stock_down_reduce(
    activated: mx.array,
    indices: mx.array,
    router_weights: mx.array,
    projection: tuple[mx.array, mx.array, mx.array],
) -> mx.array:
    expert_outputs = mx.gather_qmm(
        activated,
        *projection,
        rhs_indices=indices,
        transpose=True,
        group_size=128,
        bits=2,
        mode="affine",
    ).squeeze(-2)
    return (expert_outputs * router_weights[..., None]).sum(axis=-2)


@unittest.skipUnless(_metal_available(), "requires Metal")
class FusedDownReduceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.projection = _packed_projection()
        cls.indices = mx.arange(K3_TOP_K, dtype=mx.uint32).reshape(1, 1, K3_TOP_K)
        mx.eval(*cls.projection, cls.indices)

    def test_k3_geometry_is_bit_exact_for_all_supported_tiles(self):
        mx.random.seed(37)
        activated = mx.random.normal(
            (1, 1, K3_TOP_K, 1, K3_DOWN_INPUT_WIDTH),
            dtype=mx.bfloat16,
        )
        router_weights = mx.random.uniform(
            shape=(1, 1, K3_TOP_K),
            dtype=mx.bfloat16,
        )
        reference = _stock_down_reduce(
            activated,
            self.indices,
            router_weights,
            self.projection,
        )
        configurations = ((2, 8), (4, 4), (4, 8), (4, 16), (8, 8))
        for results, simdgroups in configurations:
            candidate = fused_down_reduce_decode(
                activated,
                self.indices,
                router_weights,
                self.projection,
                results_per_threadgroup=results,
                simdgroups_per_threadgroup=simdgroups,
            )
            mx.eval(reference, candidate)
            self.assertEqual(candidate.shape, (1, 1, K3_DOWN_OUTPUT_WIDTH))
            self.assertTrue(
                bool(mx.all(reference == candidate).item()),
                f"results={results}, simdgroups={simdgroups}",
            )

    def test_router_weights_remain_dynamic_on_cached_kernel(self):
        mx.random.seed(41)
        activated = mx.random.normal(
            (1, 1, K3_TOP_K, 1, K3_DOWN_INPUT_WIDTH),
            dtype=mx.bfloat16,
        )
        first_weights = mx.full(
            (1, 1, K3_TOP_K),
            0.03125,
            dtype=mx.bfloat16,
        )
        second_weights = mx.arange(K3_TOP_K, dtype=mx.bfloat16).reshape(
            1, 1, K3_TOP_K
        ) / mx.array(256, dtype=mx.bfloat16)
        first = fused_down_reduce_decode(
            activated,
            self.indices,
            first_weights,
            self.projection,
        )
        second = fused_down_reduce_decode(
            activated,
            self.indices,
            second_weights,
            self.projection,
        )
        reference = _stock_down_reduce(
            activated,
            self.indices,
            second_weights,
            self.projection,
        )
        mx.eval(first, second, reference)
        self.assertTrue(bool(mx.all(reference == second).item()))
        self.assertTrue(bool(mx.any(first != second).item()))

    def test_derived_bias_is_bit_exact_against_incumbent(self):
        mx.random.seed(43)
        weight, scales, _ = self.projection
        projection = weight, scales, derived_affine2_biases(scales)
        activated = mx.random.normal(
            (1, 1, K3_TOP_K, 1, K3_DOWN_INPUT_WIDTH),
            dtype=mx.bfloat16,
        )
        router_weights = mx.random.uniform(
            shape=(1, 1, K3_TOP_K),
            dtype=mx.bfloat16,
        )
        incumbent = fused_down_reduce_decode(
            activated,
            self.indices,
            router_weights,
            projection,
            results_per_threadgroup=4,
            simdgroups_per_threadgroup=16,
            derive_bias=False,
        )
        candidate = fused_down_reduce_decode(
            activated,
            self.indices,
            router_weights,
            projection,
            results_per_threadgroup=4,
            simdgroups_per_threadgroup=16,
            derive_bias=True,
        )
        mx.eval(incumbent, candidate)
        self.assertTrue(bool(mx.array_equal(incumbent, candidate).item()))

    def test_derive_bias_argument_is_strictly_boolean(self):
        activated = mx.zeros(
            (1, 1, K3_TOP_K, 1, K3_DOWN_INPUT_WIDTH),
            dtype=mx.bfloat16,
        )
        router_weights = mx.zeros(
            (1, 1, K3_TOP_K),
            dtype=mx.bfloat16,
        )
        with self.assertRaisesRegex(TypeError, "derive_bias must be a bool"):
            fused_down_reduce_decode(
                activated,
                self.indices,
                router_weights,
                self.projection,
                derive_bias=1,
            )

    def test_unsupported_shapes_and_dtypes_fail_closed(self):
        activated = mx.zeros(
            (1, 1, K3_TOP_K, 1, K3_DOWN_INPUT_WIDTH),
            dtype=mx.bfloat16,
        )
        router_weights = mx.zeros(
            (1, 1, K3_TOP_K),
            dtype=mx.bfloat16,
        )
        cases = (
            (
                activated.astype(mx.float32),
                self.indices,
                router_weights,
            ),
            (
                activated,
                self.indices[..., :-1],
                router_weights[..., :-1],
            ),
            (
                activated,
                self.indices,
                router_weights.astype(mx.float32),
            ),
        )
        for values, indices, weights in cases:
            with self.subTest(
                activation_shape=values.shape,
                activation_dtype=values.dtype,
                top_k=indices.shape[-1],
                router_dtype=weights.dtype,
            ):
                self.assertFalse(
                    supports_fused_down_reduce(
                        values,
                        indices,
                        weights,
                        self.projection,
                        results_per_threadgroup=4,
                        simdgroups_per_threadgroup=8,
                    )
                )


if __name__ == "__main__":
    unittest.main()
