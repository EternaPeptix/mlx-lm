from __future__ import annotations

import unittest

import mlx.core as mx

from mlx_lm.models.kimi_k3_fused_switch_glu import (
    _metal_available,
    fused_switch_situ_decode,
    supports_fused_switch_situ,
)


def _projection(experts: int, output: int, input_width: int):
    weight = mx.random.normal((experts, output, input_width), dtype=mx.bfloat16)
    return tuple(mx.quantize(weight, group_size=128, bits=2, mode="affine"))


def _reference(x, indices, up, gate):
    def qmm(projection):
        return mx.gather_qmm(
            mx.expand_dims(x, (-2, -3)),
            *projection,
            rhs_indices=indices,
            transpose=True,
            group_size=128,
            bits=2,
            mode="affine",
        )

    up_value = qmm(up).astype(mx.float32)
    gate_value = qmm(gate).astype(mx.float32)
    activation = 4.0 * mx.tanh(gate_value / 4.0) * mx.sigmoid(gate_value)
    up_value = 25.0 * mx.tanh(up_value / 25.0)
    return (activation * up_value).astype(mx.bfloat16)


@unittest.skipUnless(_metal_available(), "requires Metal")
class FusedSwitchGLUTest(unittest.TestCase):
    def setUp(self):
        mx.random.seed(7)
        self.x = mx.random.normal((1, 1, 512), dtype=mx.bfloat16)
        self.indices = mx.array([[[1, 3, 5, 7]]], dtype=mx.uint32)
        self.up = _projection(8, 16, 512)
        self.gate = _projection(8, 16, 512)
        mx.eval(self.x, self.indices, *self.up, *self.gate)

    def test_matches_native_quantized_path(self):
        for width in (1, 2):
            x = mx.random.normal((1, width, 512), dtype=mx.bfloat16)
            indices = mx.concatenate(
                [mx.roll(self.indices, shift, axis=-1) for shift in range(width)],
                axis=1,
            )
            reference = _reference(x, indices, self.up, self.gate)
            for results, simds in ((2, 1), (4, 2), (4, 4), (8, 1), (8, 2)):
                candidate = fused_switch_situ_decode(
                    x,
                    indices,
                    self.up,
                    self.gate,
                    results_per_simdgroup=results,
                    simdgroups=simds,
                )
                mx.eval(reference, candidate)
                self.assertEqual(reference.shape, candidate.shape)
                self.assertTrue(
                    bool(mx.all(reference == candidate).item()),
                    (width, results, simds),
                )

    def test_contract_rejects_unproven_verification_widths(self):
        for width in (3, 8):
            prefill = mx.broadcast_to(self.x, (1, width, 512))
            indices = mx.broadcast_to(self.indices, (1, width, 4))
            self.assertFalse(
                supports_fused_switch_situ(prefill, indices, self.up, self.gate)
            )


if __name__ == "__main__":
    unittest.main()
