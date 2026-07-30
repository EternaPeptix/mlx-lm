from __future__ import annotations

import unittest

import mlx.core as mx

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


if __name__ == "__main__":
    unittest.main()
