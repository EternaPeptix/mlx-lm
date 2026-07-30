from __future__ import annotations

import os
import unittest
from unittest import mock

import mlx.core as mx

from mlx_lm.models.gated_delta import (
    _EXPERIMENTAL_KDA_ROW_PREFILL_ENV,
    experimental_kda_row_prefill_eligible,
    experimental_kda_row_prefill_enabled,
    experimental_kda_row_prefill_kernel,
    gated_delta_kernel,
    gated_delta_update,
)


@unittest.skipUnless(mx.metal.is_available(), "requires Metal")
class KimiK3KDAPrefillTest(unittest.TestCase):
    @staticmethod
    def _inputs(tokens=128, key_heads=2, value_heads=4):
        batch, dim = 1, 128
        mx.random.seed(19)
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
        state = mx.zeros(
            (batch, value_heads, dim, dim),
            dtype=mx.float32,
        )
        return q, k, v, gate, beta, state

    def test_default_off_and_fail_closed_dispatch(self):
        inputs = self._inputs()
        q, k, v, gate, _, state = inputs
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(experimental_kda_row_prefill_enabled())
            self.assertFalse(
                experimental_kda_row_prefill_eligible(q, k, v, gate, state)
            )

        with mock.patch.dict(
            os.environ,
            {_EXPERIMENTAL_KDA_ROW_PREFILL_ENV: "1"},
            clear=True,
        ):
            self.assertTrue(experimental_kda_row_prefill_eligible(q, k, v, gate, state))
            self.assertFalse(
                experimental_kda_row_prefill_eligible(
                    q[:, :127],
                    k[:, :127],
                    v[:, :127],
                    gate[:, :127],
                    state,
                )
            )
            self.assertFalse(
                experimental_kda_row_prefill_eligible(
                    q,
                    k,
                    v,
                    gate,
                    state,
                    mask=mx.ones((1, 128), dtype=mx.bool_),
                )
            )
            self.assertFalse(
                experimental_kda_row_prefill_eligible(
                    q,
                    k,
                    v,
                    gate,
                    state,
                    return_state_history=True,
                )
            )

    def test_row_tiles_are_bit_exact_with_recurrent_kernel(self):
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
                lower_bound=-0.1,
            )
        with mock.patch.dict(
            os.environ,
            {_EXPERIMENTAL_KDA_ROW_PREFILL_ENV: "1"},
            clear=True,
        ):
            candidate = gated_delta_update(
                q,
                k,
                v,
                a,
                b,
                A_log,
                dt_bias,
                state,
                lower_bound=-0.1,
            )

        mx.eval(*reference, *candidate)
        self.assertTrue(bool(mx.all(reference[0] == candidate[0]).item()))
        self.assertTrue(bool(mx.all(reference[1] == candidate[1]).item()))

    def test_invalid_tile_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "rows_per_simd"):
            experimental_kda_row_prefill_kernel(
                *self._inputs(),
                rows_per_simd=3,
            )


if __name__ == "__main__":
    unittest.main()
