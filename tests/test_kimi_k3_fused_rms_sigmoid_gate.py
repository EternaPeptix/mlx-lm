from __future__ import annotations

import os
import unittest

import mlx.core as mx
import numpy as np

from mlx_lm.models.kimi_k3_fused_rms_sigmoid_gate import (
    FUSED_RMS_SIGMOID_GATE_ENV,
    fused_rms_sigmoid_gate_enabled,
    maybe_fused_rms_sigmoid_gate,
)


_EPS = 1e-5
_SHAPE = (1, 1, 96, 128)


def _reference(x, gate, weight):
    return mx.fast.rms_norm(x, weight, _EPS) * mx.sigmoid(gate)


def _inputs(dtype=mx.bfloat16, seed=7):
    mx.random.seed(seed)
    x = mx.random.normal(_SHAPE).astype(dtype)
    gate = (8.0 * mx.random.normal(_SHAPE)).astype(dtype)
    weight = mx.random.normal((_SHAPE[-1],)).astype(dtype)
    return x, gate, weight


@unittest.skipUnless(mx.metal.is_available(), "requires Metal")
class FusedRmsSigmoidGateTest(unittest.TestCase):
    def setUp(self):
        os.environ[FUSED_RMS_SIGMOID_GATE_ENV] = "1"
        fused_rms_sigmoid_gate_enabled.cache_clear()

    def tearDown(self):
        os.environ.pop(FUSED_RMS_SIGMOID_GATE_ENV, None)
        fused_rms_sigmoid_gate_enabled.cache_clear()

    def assertExact(self, x, gate, weight):
        expected = _reference(x, gate, weight)
        actual = maybe_fused_rms_sigmoid_gate(
            x,
            gate,
            weight,
            _EPS,
            training=False,
        )
        self.assertIsNotNone(actual)
        mx.eval(expected, actual)
        np.testing.assert_array_equal(
            np.asarray(expected.astype(mx.float32)),
            np.asarray(actual.astype(mx.float32)),
        )

    def test_released_geometry_is_bit_exact_across_dtypes_and_seeds(self):
        for dtype in (mx.bfloat16, mx.float16, mx.float32):
            for seed in range(32):
                with self.subTest(dtype=dtype, seed=seed):
                    self.assertExact(*_inputs(dtype, seed))

    def test_compiled_released_geometry_is_bit_exact(self):
        x, gate, weight = _inputs()

        @mx.compile
        def reference(a, b, w):
            return _reference(a, b, w)

        @mx.compile
        def candidate(a, b, w):
            return maybe_fused_rms_sigmoid_gate(
                a,
                b,
                w,
                _EPS,
                training=False,
            )

        expected = reference(x, gate, weight)
        actual = candidate(x, gate, weight)
        mx.eval(expected, actual)
        self.assertTrue(bool(mx.all(expected == actual).item()))

    def test_feature_is_default_off(self):
        os.environ.pop(FUSED_RMS_SIGMOID_GATE_ENV)
        fused_rms_sigmoid_gate_enabled.cache_clear()
        self.assertIsNone(
            maybe_fused_rms_sigmoid_gate(
                *_inputs(),
                _EPS,
                training=False,
            )
        )

    def test_training_and_unsupported_contracts_fail_closed(self):
        x, gate, weight = _inputs()
        cases = (
            (x, gate, weight, _EPS, True),
            (x[:, :, :95], gate[:, :, :95], weight, _EPS, False),
            (x, gate.astype(mx.float16), weight, _EPS, False),
            (x, gate, weight[:-1], _EPS, False),
            (x, gate, weight.astype(mx.float16), _EPS, False),
            (x, gate, weight, 0.0, False),
        )
        for candidate_x, candidate_gate, candidate_weight, eps, training in cases:
            with self.subTest(
                shape=candidate_x.shape,
                dtype=candidate_x.dtype,
                eps=eps,
                training=training,
            ):
                self.assertIsNone(
                    maybe_fused_rms_sigmoid_gate(
                        candidate_x,
                        candidate_gate,
                        candidate_weight,
                        eps,
                        training=training,
                    )
                )


if __name__ == "__main__":
    unittest.main()
