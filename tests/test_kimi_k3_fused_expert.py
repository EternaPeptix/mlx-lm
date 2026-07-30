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

    def test_fused_switch_is_bit_exact(self):
        mx.random.seed(19)
        switch = _Switch()
        x = mx.random.normal((1, 1, 512), dtype=mx.bfloat16)
        indices = mx.array([[[1, 3, 5, 7]]], dtype=mx.uint32)
        reference = switch.stock(x, indices)
        candidate = maybe_fused_k3_switch_glu(switch, x, indices)
        self.assertIsNotNone(candidate)
        mx.eval(reference, candidate)
        self.assertTrue(bool(mx.all(reference == candidate).item()))

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
