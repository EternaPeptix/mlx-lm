from __future__ import annotations

import unittest

import mlx.core as mx

from mlx_lm.models.kimi_k3_derived_bias import derived_affine2_biases
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


def _adversarial_projection(experts: int, output: int, input_width: int):
    weight = mx.full(
        (experts, output, input_width // 16),
        0x24681357,
        dtype=mx.uint32,
    )
    # Signed minimum/maximum finite normals plus representative interior values.
    pattern = mx.array(
        [0x0080, 0x8080, 0x3F80, 0xBF80, 0x7F7F, 0xFF7F, 0x3C00, 0xBC00],
        dtype=mx.uint16,
    ).view(mx.bfloat16)
    size = experts * output * (input_width // 128)
    scales = pattern[mx.arange(size, dtype=mx.uint32) % pattern.size].reshape(
        experts,
        output,
        input_width // 128,
    )
    return weight, scales, derived_affine2_biases(scales)


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
        for width in (1, 2, 3):
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
        for width in (4, 8):
            prefill = mx.broadcast_to(self.x, (1, width, 512))
            indices = mx.broadcast_to(self.indices, (1, width, 4))
            self.assertFalse(
                supports_fused_switch_situ(prefill, indices, self.up, self.gate)
            )

    def test_derived_bias_matches_incumbent_at_finite_normal_bf16_limits(self):
        up = _adversarial_projection(8, 16, 512)
        gate = _adversarial_projection(8, 16, 512)
        incumbent = fused_switch_situ_decode(
            self.x,
            self.indices,
            up,
            gate,
            results_per_simdgroup=4,
            simdgroups=2,
            derive_bias=False,
        )
        candidate = fused_switch_situ_decode(
            self.x,
            self.indices,
            up,
            gate,
            results_per_simdgroup=4,
            simdgroups=2,
            derive_bias=True,
        )
        mx.eval(incumbent, candidate)
        self.assertTrue(
            bool(
                mx.array_equal(
                    incumbent.view(mx.uint8), candidate.view(mx.uint8)
                ).item()
            )
        )

    def test_derive_bias_argument_is_strictly_boolean(self):
        with self.assertRaisesRegex(TypeError, "derive_bias must be a bool"):
            fused_switch_situ_decode(
                self.x,
                self.indices,
                self.up,
                self.gate,
                derive_bias=1,
            )


if __name__ == "__main__":
    unittest.main()
