from __future__ import annotations

import os
import unittest
from unittest import mock

import mlx.core as mx

import mlx_lm.models.gated_delta as gated_delta
from mlx_lm.models.gated_delta import (
    _EXPERIMENTAL_KDA_ROW_DECODE_ENV,
    _EXPERIMENTAL_KDA_ROW_PREFILL_ENV,
    experimental_kda_row_decode_enabled,
    experimental_kda_row_eligible,
    experimental_kda_row_prefill_kernel,
    gated_delta_kernel,
    gated_delta_update,
)


@unittest.skipUnless(mx.metal.is_available(), "requires Metal")
class KimiK3KDADecodeTest(unittest.TestCase):
    @staticmethod
    def _inputs(tokens: int = 1, key_heads: int = 2, value_heads: int = 4):
        batch, dim = 1, 128
        mx.random.seed(29)
        q = (mx.random.normal((batch, tokens, key_heads, dim)) / dim**0.5).astype(
            mx.bfloat16
        )
        k = (mx.random.normal((batch, tokens, key_heads, dim)) / dim**0.5).astype(
            mx.bfloat16
        )
        v = (0.05 * mx.random.normal((batch, tokens, value_heads, dim))).astype(
            mx.bfloat16
        )
        gate = mx.full(
            (batch, tokens, value_heads, dim),
            0.98,
            dtype=mx.float32,
        )
        beta = mx.full(
            (batch, tokens, value_heads),
            0.4,
            dtype=mx.bfloat16,
        )
        state = mx.random.normal(
            (batch, value_heads, dim, dim),
            dtype=mx.float32,
        )
        return q, k, v, gate, beta, state

    def test_decode_gate_is_independent_and_default_off(self):
        q, k, v, gate, _, state = self._inputs()
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(experimental_kda_row_decode_enabled())
            self.assertFalse(experimental_kda_row_eligible(q, k, v, gate, state))

        with mock.patch.dict(
            os.environ,
            {_EXPERIMENTAL_KDA_ROW_PREFILL_ENV: "1"},
            clear=True,
        ):
            self.assertFalse(experimental_kda_row_eligible(q, k, v, gate, state))

        with mock.patch.dict(
            os.environ,
            {_EXPERIMENTAL_KDA_ROW_DECODE_ENV: "1"},
            clear=True,
        ):
            self.assertTrue(experimental_kda_row_eligible(q, k, v, gate, state))
            wide = self._inputs(tokens=128)
            self.assertFalse(
                experimental_kda_row_eligible(
                    wide[0], wide[1], wide[2], wide[3], wide[5]
                )
            )

    def test_decode_row_tiles_are_bit_exact(self):
        inputs = self._inputs()
        reference = gated_delta_kernel(*inputs)
        mx.eval(*reference)
        for rows in (1, 2, 4, 8):
            candidate = experimental_kda_row_prefill_kernel(
                *inputs,
                rows_per_simd=rows,
            )
            mx.eval(*candidate)
            self.assertTrue(bool(mx.all(reference[0] == candidate[0]).item()))
            self.assertTrue(bool(mx.all(reference[1] == candidate[1]).item()))

    def test_opt_in_update_dispatch_is_bit_exact(self):
        q, k, v, _, _, state = self._inputs()
        batch, tokens, heads, dim = v.shape
        a = mx.zeros((batch, tokens, heads, dim), dtype=mx.bfloat16)
        b = mx.zeros((batch, tokens, heads), dtype=mx.bfloat16)
        A_log = mx.zeros((heads, 1), dtype=mx.float32)
        dt_bias = mx.zeros((heads, dim), dtype=mx.float32)

        with mock.patch.dict(os.environ, {}, clear=True):
            reference = gated_delta_update(
                q,
                k,
                v,
                a,
                b,
                A_log,
                dt_bias,
                state,
                lower_bound=-5.0,
            )
        with mock.patch.dict(
            os.environ,
            {_EXPERIMENTAL_KDA_ROW_DECODE_ENV: "1"},
            clear=True,
        ), mock.patch.object(
            gated_delta,
            "experimental_kda_row_prefill_kernel",
            wraps=experimental_kda_row_prefill_kernel,
        ) as row_kernel:
            candidate = gated_delta_update(
                q,
                k,
                v,
                a,
                b,
                A_log,
                dt_bias,
                state,
                lower_bound=-5.0,
            )
            self.assertEqual(row_kernel.call_args.kwargs["rows_per_simd"], 2)

        mx.eval(*reference, *candidate)
        self.assertTrue(bool(mx.all(reference[0] == candidate[0]).item()))
        self.assertTrue(bool(mx.all(reference[1] == candidate[1]).item()))


if __name__ == "__main__":
    unittest.main()
