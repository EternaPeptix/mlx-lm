from __future__ import annotations

import unittest

import mlx.core as mx

from mlx_lm.models.kimi_k3_derived_bias import derived_affine2_biases
from mlx_lm.models.kimi_k3_tuned_gather_qmv import (
    _metal_available,
    tuned_gather_qmv,
)


@unittest.skipUnless(_metal_available(), "requires Metal")
class TunedGatherQMVTest(unittest.TestCase):
    def test_all_tiles_are_bit_exact_for_broadcast_and_per_expert_inputs(self):
        mx.random.seed(11)
        experts, output, input_width, top_k = 8, 32, 512, 4
        source = mx.random.normal((experts, output, input_width), dtype=mx.bfloat16)
        projection = tuple(mx.quantize(source, group_size=128, bits=2, mode="affine"))
        indices = mx.array([[[1, 3, 5, 7]]], dtype=mx.uint32)
        for broadcast in (True, False):
            rows = 1 if broadcast else top_k
            x = mx.random.normal((rows, input_width), dtype=mx.bfloat16)
            native_x = (
                x.reshape(1, 1, 1, 1, input_width)
                if broadcast
                else x.reshape(1, 1, top_k, 1, input_width)
            )
            reference = mx.gather_qmm(
                native_x,
                *projection,
                rhs_indices=indices,
                transpose=True,
                group_size=128,
                bits=2,
                mode="affine",
            )
            for results, simds in ((2, 1), (4, 2), (8, 2), (16, 1)):
                candidate = tuned_gather_qmv(
                    x,
                    indices,
                    projection,
                    results_per_simdgroup=results,
                    simdgroups=simds,
                    broadcast_x=broadcast,
                )
                mx.eval(reference, candidate)
                self.assertTrue(
                    bool(mx.all(reference == candidate).item()),
                    (broadcast, results, simds),
                )

    def test_derived_bias_matches_incumbent_for_adversarial_bf16_metadata(self):
        experts, output, input_width, top_k = 8, 32, 512, 4
        weight = mx.full(
            (experts, output, input_width // 16),
            0x13579BDF,
            dtype=mx.uint32,
        )
        pattern = mx.array(
            [
                0x0080,
                0x8080,
                0x3F80,
                0xBF80,
                0x7F7F,
                0xFF7F,
                0x3C00,
                0xBC00,
            ],
            dtype=mx.uint16,
        ).view(mx.bfloat16)
        size = experts * output * (input_width // 128)
        scales = pattern[mx.arange(size, dtype=mx.uint32) % pattern.size].reshape(
            experts,
            output,
            input_width // 128,
        )
        projection = weight, scales, derived_affine2_biases(scales)
        indices = mx.array([[[1, 3, 5, 7]]], dtype=mx.uint32)
        x = mx.random.normal((top_k, input_width), dtype=mx.bfloat16)
        incumbent = tuned_gather_qmv(
            x,
            indices,
            projection,
            results_per_simdgroup=4,
            simdgroups=2,
            broadcast_x=False,
            derive_bias=False,
        )
        candidate = tuned_gather_qmv(
            x,
            indices,
            projection,
            results_per_simdgroup=4,
            simdgroups=2,
            broadcast_x=False,
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
        experts, output, input_width = 1, 8, 512
        source = mx.zeros((experts, output, input_width), dtype=mx.bfloat16)
        projection = tuple(mx.quantize(source, group_size=128, bits=2, mode="affine"))
        with self.assertRaisesRegex(TypeError, "derive_bias must be a bool"):
            tuned_gather_qmv(
                mx.zeros((1, input_width), dtype=mx.bfloat16),
                mx.zeros((1, 1, 1), dtype=mx.uint32),
                projection,
                results_per_simdgroup=4,
                simdgroups=2,
                broadcast_x=True,
                derive_bias=1,
            )


if __name__ == "__main__":
    unittest.main()
