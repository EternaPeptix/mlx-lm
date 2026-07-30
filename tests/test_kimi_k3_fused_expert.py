from __future__ import annotations

import os
import unittest

import mlx.core as mx

from mlx_lm.models.kimi_k3_fused_expert import (
    FUSED_EXPERT_ENV,
    fused_k3_experts_enabled,
    maybe_fused_k3_switch_glu,
)
from mlx_lm.models.kimi_k3_fused_switch_glu import _metal_available


class _Activation:
    beta = 4.0
    linear_beta = 25.0


class _Projection:
    bits = 2
    group_size = 128
    mode = "affine"

    def __init__(self, weight):
        self.weight, self.scales, self.biases = mx.quantize(
            weight, group_size=128, bits=2, mode="affine"
        )

    @classmethod
    def packed(
        cls,
        *,
        experts: int,
        input_width: int,
        output_width: int,
        packed_value: int,
    ):
        projection = cls.__new__(cls)
        projection.weight = mx.full(
            (experts, output_width, input_width // 16),
            packed_value,
            dtype=mx.uint32,
        )
        projection.scales = mx.full(
            (experts, output_width, input_width // 128),
            0.015625,
            dtype=mx.bfloat16,
        )
        projection.biases = mx.zeros_like(projection.scales)
        return projection

    def __contains__(self, name):
        return False

    def __getitem__(self, name):
        return getattr(self, name)

    def get(self, name):
        return getattr(self, name, None)

    def __call__(self, x, indices, sorted_indices=False):
        del sorted_indices
        return mx.gather_qmm(
            x,
            self.weight,
            self.scales,
            self.biases,
            rhs_indices=indices,
            transpose=True,
            group_size=128,
            bits=2,
            mode="affine",
        )


class _Switch:
    training = False
    activation = _Activation()

    def __init__(self):
        self.gate_proj = _Projection(mx.random.normal((8, 512, 512), dtype=mx.bfloat16))
        self.up_proj = _Projection(mx.random.normal((8, 512, 512), dtype=mx.bfloat16))
        self.down_proj = _Projection(mx.random.normal((8, 512, 512), dtype=mx.bfloat16))

    @classmethod
    def bounded_tp2_geometry(cls):
        """Keep K3's TP2 dimensions while bounding the inactive expert table."""

        switch = cls.__new__(cls)
        switch.gate_proj = _Projection.packed(
            experts=16,
            input_width=3584,
            output_width=1536,
            packed_value=0x12345678,
        )
        switch.up_proj = _Projection.packed(
            experts=16,
            input_width=3584,
            output_width=1536,
            packed_value=0x76543210,
        )
        switch.down_proj = _Projection.packed(
            experts=16,
            input_width=1536,
            output_width=3584,
            packed_value=0x24681357,
        )
        return switch

    def stock(self, x, indices):
        expanded = mx.expand_dims(x, (-2, -3))
        up = self.up_proj(expanded, indices).astype(mx.float32)
        gate = self.gate_proj(expanded, indices).astype(mx.float32)
        activated = (
            4.0 * mx.tanh(gate / 4.0) * mx.sigmoid(gate) * (25.0 * mx.tanh(up / 25.0))
        ).astype(mx.bfloat16)
        return self.down_proj(activated, indices).squeeze(-2)


@unittest.skipUnless(_metal_available(), "requires Metal")
class IntegrationTest(unittest.TestCase):
    def setUp(self):
        os.environ[FUSED_EXPERT_ENV] = "1"
        fused_k3_experts_enabled.cache_clear()

    def tearDown(self):
        os.environ.pop(FUSED_EXPERT_ENV, None)
        fused_k3_experts_enabled.cache_clear()

    def test_fused_switch_is_bit_exact_on_cached_second_call(self):
        mx.random.seed(19)
        switch = _Switch()
        x = mx.random.normal((1, 1, 512), dtype=mx.bfloat16)
        indices = mx.array([[[1, 3, 5, 7]]], dtype=mx.uint32)
        reference = switch.stock(x, indices)
        first = maybe_fused_k3_switch_glu(switch, x, indices)
        second = maybe_fused_k3_switch_glu(switch, x, indices)
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        mx.eval(reference, first, second)
        self.assertTrue(bool(mx.all(reference == first).item()))
        self.assertTrue(bool(mx.all(reference == second).item()))

    def test_compiled_cache_keeps_weights_dynamic(self):
        mx.random.seed(23)
        first_switch = _Switch()
        mx.random.seed(29)
        second_switch = _Switch()
        x = mx.random.normal((1, 1, 512), dtype=mx.bfloat16)
        indices = mx.array([[[0, 2, 4, 6]]], dtype=mx.uint32)

        first = maybe_fused_k3_switch_glu(first_switch, x, indices)
        second = maybe_fused_k3_switch_glu(second_switch, x, indices)
        reference = second_switch.stock(x, indices)
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        mx.eval(first, second, reference)
        self.assertTrue(bool(mx.all(reference == second).item()))
        self.assertTrue(bool(mx.any(first != second).item()))

    def test_bounded_tp2_geometry_is_bit_exact(self):
        switch = _Switch.bounded_tp2_geometry()
        x = mx.random.normal((1, 1, 3584), dtype=mx.bfloat16)
        indices = mx.arange(16, dtype=mx.uint32).reshape(1, 1, 16)
        reference = switch.stock(x, indices)
        candidate = maybe_fused_k3_switch_glu(switch, x, indices)
        self.assertIsNotNone(candidate)
        mx.eval(reference, candidate)
        self.assertEqual(candidate.shape, (1, 1, 16, 3584))
        self.assertTrue(bool(mx.all(reference == candidate).item()))

    def test_non_decode_shapes_and_dtypes_fall_back(self):
        switch = _Switch()
        indices = mx.array([[[1, 3, 5, 7]]], dtype=mx.uint32)
        cases = (
            (
                mx.zeros((1, 2, 512), dtype=mx.bfloat16),
                mx.broadcast_to(indices, (1, 2, 4)),
            ),
            (mx.zeros((1, 1, 512), dtype=mx.float32), indices),
        )
        for x, routed_indices in cases:
            with self.subTest(shape=x.shape, dtype=x.dtype):
                self.assertIsNone(maybe_fused_k3_switch_glu(switch, x, routed_indices))

    def test_training_falls_back(self):
        switch = _Switch()
        switch.training = True
        x = mx.zeros((1, 1, 512), dtype=mx.bfloat16)
        indices = mx.array([[[1, 3, 5, 7]]], dtype=mx.uint32)
        self.assertIsNone(maybe_fused_k3_switch_glu(switch, x, indices))

    def test_disabled_by_default(self):
        os.environ.pop(FUSED_EXPERT_ENV)
        fused_k3_experts_enabled.cache_clear()
        switch = _Switch()
        x = mx.zeros((1, 1, 512), dtype=mx.bfloat16)
        indices = mx.array([[[1, 3, 5, 7]]], dtype=mx.uint32)
        self.assertIsNone(maybe_fused_k3_switch_glu(switch, x, indices))


if __name__ == "__main__":
    unittest.main()
