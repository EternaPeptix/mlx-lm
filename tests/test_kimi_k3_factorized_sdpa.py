# Copyright © 2026 Apple Inc.

import math
import os
import unittest
from unittest import mock

import mlx.core as mx

from mlx_lm.models import kimi_k3
from mlx_lm.models.cache import KVCache


def _released_tp2_attention():
    args = kimi_k3.TextArgs(
        hidden_size=8,
        num_attention_heads=48,
        num_key_value_heads=48,
        q_lora_rank=None,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        mla_use_nope=True,
        mla_use_output_gate=False,
    )
    attention = kimi_k3.KimiK3MLAAttention(args)
    attention.set_dtype(mx.bfloat16)
    attention.eval()
    return attention


def _exact_output(q, _, value, **kwargs):
    return mx.zeros((*q.shape[:-1], value.shape[-1]), dtype=q.dtype)


class _PlainBitsCache:
    bits = 4
    group_size = 64

    def update_and_fetch(self, keys, values):
        return keys, values


@unittest.skipUnless(mx.metal.is_available(), "requires Metal")
class TestKimiK3FactorizedSDPA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.attention = _released_tp2_attention()

    def setUp(self):
        self.attention.eval()

    @staticmethod
    def _input(length=9, dtype=mx.bfloat16):
        return mx.zeros((1, length, 8), dtype=dtype)

    @staticmethod
    def _mask(query_length=9, key_length=9):
        offset = key_length - query_length
        return mx.arange(key_length)[None, :] <= (
            mx.arange(query_length)[:, None] + offset
        )

    def test_default_is_off_and_does_not_resolve_the_symbol(self):
        exact = mock.Mock(side_effect=_exact_output)
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(
                kimi_k3,
                "_factorized_sdpa_prefill_primitive",
            ) as resolve,
            mock.patch.object(kimi_k3, "scaled_dot_product_attention", exact),
        ):
            output = self.attention(self._input(), self._mask())

        self.assertEqual(output.shape, (1, 9, 8))
        resolve.assert_not_called()
        exact.assert_called_once()

    def test_dispatches_exact_factorization_heads_scales_and_mask(self):
        calls = []

        def factorized(*args, **kwargs):
            calls.append((args, kwargs))
            q0, _, value, _, _ = args
            return mx.zeros((*q0.shape[:-1], value.shape[-1]), dtype=q0.dtype)

        mask = self._mask()
        with (
            mock.patch.dict(
                os.environ,
                {kimi_k3.FACTORIZED_SDPA_PREFILL_ENV: "1"},
                clear=True,
            ),
            mock.patch.object(
                kimi_k3,
                "_factorized_sdpa_prefill_primitive",
                return_value=factorized,
            ),
            mock.patch.object(
                kimi_k3,
                "scaled_dot_product_attention",
                side_effect=AssertionError("exact path must not run"),
            ),
        ):
            output = self.attention(self._input(), mask)

        self.assertEqual(output.shape, (1, 9, 8))
        self.assertEqual(len(calls), 1)
        (q0, k0, value, q1, k1), kwargs = calls[0]
        self.assertEqual(q0.shape, (1, 48, 9, 128))
        self.assertEqual(k0.shape, (1, 48, 9, 128))
        self.assertEqual(value.shape, (1, 48, 9, 128))
        self.assertEqual(q1.shape, (1, 48, 9, 64))
        self.assertEqual(k1.shape, (1, 1, 9, 64))
        self.assertTrue(all(x.dtype == mx.bfloat16 for x in (q0, k0, value, q1, k1)))
        self.assertAlmostEqual(kwargs["scale0"], 1.0 / math.sqrt(192))
        self.assertAlmostEqual(kwargs["scale1"], 1.0 / math.sqrt(192))
        self.assertIs(kwargs["mask"], mask)

    def test_selected_primitive_errors_do_not_enter_exact_fallback(self):
        exact = mock.Mock(side_effect=_exact_output)
        with (
            mock.patch.dict(
                os.environ,
                {kimi_k3.FACTORIZED_SDPA_PREFILL_ENV: "1"},
                clear=True,
            ),
            mock.patch.object(
                kimi_k3,
                "_factorized_sdpa_prefill_primitive",
                return_value=mock.Mock(side_effect=RuntimeError("primitive failed")),
            ),
            mock.patch.object(kimi_k3, "scaled_dot_product_attention", exact),
        ):
            with self.assertRaisesRegex(RuntimeError, "primitive failed"):
                self.attention(self._input(), self._mask())
        exact.assert_not_called()

    def test_factorized_algebra_matches_the_default_exact_path(self):
        mx.random.seed(193)
        inputs = mx.random.normal((1, 9, 8)).astype(mx.bfloat16)
        mask = self._mask()

        with mock.patch.dict(os.environ, {}, clear=True):
            expected = self.attention(inputs, mask)

        def factorized(q0, k0, value, q1, k1, *, scale0, scale1, mask):
            scores = (q0 * scale0) @ k0.swapaxes(-1, -2)
            scores = scores + (q1 * scale1) @ k1.swapaxes(-1, -2)
            if mask is not None:
                scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
            return mx.softmax(scores, axis=-1, precise=True) @ value

        with (
            mock.patch.dict(
                os.environ,
                {kimi_k3.FACTORIZED_SDPA_PREFILL_ENV: "1"},
                clear=True,
            ),
            mock.patch.object(
                kimi_k3,
                "_factorized_sdpa_prefill_primitive",
                return_value=factorized,
            ),
        ):
            actual = self.attention(inputs, mask)

        mx.eval(expected, actual)
        self.assertTrue(mx.allclose(expected, actual, rtol=2e-2, atol=2e-2).item())

    def test_populated_cache_is_updated_before_dispatch(self):
        cache = KVCache()
        cache.state = (
            mx.zeros((1, 1, 4, 512), dtype=mx.bfloat16),
            mx.zeros((1, 1, 4, 64), dtype=mx.bfloat16),
        )
        mask = self._mask(query_length=9, key_length=13)
        calls = []

        def factorized(*args, **kwargs):
            calls.append((args, kwargs))
            q0, _, value, _, _ = args
            return mx.zeros((*q0.shape[:-1], value.shape[-1]), dtype=q0.dtype)

        with (
            mock.patch.dict(
                os.environ,
                {kimi_k3.FACTORIZED_SDPA_PREFILL_ENV: "1"},
                clear=True,
            ),
            mock.patch.object(
                kimi_k3,
                "_factorized_sdpa_prefill_primitive",
                return_value=factorized,
            ),
            mock.patch.object(
                kimi_k3,
                "scaled_dot_product_attention",
                side_effect=AssertionError("exact path must not run"),
            ),
        ):
            self.attention(self._input(), mask, cache)

        self.assertEqual(cache.offset, 13)
        (q0, k0, value, q1, k1), kwargs = calls[0]
        self.assertEqual(q0.shape[-2], 9)
        self.assertEqual(q1.shape[-2], 9)
        self.assertEqual(k0.shape[-2], 13)
        self.assertEqual(value.shape[-2], 13)
        self.assertEqual(k1.shape[-2], 13)
        self.assertIs(kwargs["mask"], mask)

    def test_decode_training_quantized_cache_and_missing_symbol_fall_back(self):
        cases = (
            ("decode", self._input(1), None, None),
            ("short prefill", self._input(8), self._mask(8, 8), None),
            ("quantized cache", self._input(), self._mask(), _PlainBitsCache()),
            ("missing symbol", self._input(), self._mask(), None),
        )
        for name, inputs, mask, cache in cases:
            with self.subTest(name=name):
                exact = mock.Mock(side_effect=_exact_output)
                primitive = None if name == "missing symbol" else mock.Mock()
                with (
                    mock.patch.dict(
                        os.environ,
                        {kimi_k3.FACTORIZED_SDPA_PREFILL_ENV: "1"},
                        clear=True,
                    ),
                    mock.patch.object(
                        kimi_k3,
                        "_factorized_sdpa_prefill_primitive",
                        return_value=primitive,
                    ) as resolve,
                    mock.patch.object(
                        kimi_k3,
                        "scaled_dot_product_attention",
                        exact,
                    ),
                ):
                    self.attention(inputs, mask, cache)

                exact.assert_called_once()
                if name in ("decode", "short prefill", "quantized cache"):
                    resolve.assert_not_called()
                else:
                    resolve.assert_called_once()
                if primitive is not None:
                    primitive.assert_not_called()

        self.attention.train()
        exact = mock.Mock(side_effect=_exact_output)
        with (
            mock.patch.dict(
                os.environ,
                {kimi_k3.FACTORIZED_SDPA_PREFILL_ENV: "1"},
                clear=True,
            ),
            mock.patch.object(
                kimi_k3,
                "_factorized_sdpa_prefill_primitive",
            ) as resolve,
            mock.patch.object(kimi_k3, "scaled_dot_product_attention", exact),
        ):
            self.attention(self._input(), self._mask())
        resolve.assert_not_called()
        exact.assert_called_once()

    def test_eligibility_fails_closed_for_contract_mismatches(self):
        def arrays(
            *,
            heads=48,
            query_length=9,
            key_length=9,
            dtype=mx.bfloat16,
        ):
            return (
                mx.zeros((1, heads, query_length, 128), dtype=dtype),
                mx.zeros((1, heads, key_length, 128), dtype=dtype),
                mx.zeros((1, heads, key_length, 128), dtype=dtype),
                mx.zeros((1, heads, query_length, 64), dtype=dtype),
                mx.zeros((1, 1, key_length, 64), dtype=dtype),
            )

        valid = arrays()
        self.assertTrue(
            kimi_k3._can_use_factorized_sdpa_prefill(
                *valid,
                mask=self._mask(),
                cache=None,
            )
        )
        mismatches = (
            (arrays(heads=47), self._mask(), None),
            (arrays(query_length=8, key_length=8), self._mask(8, 8), None),
            (arrays(dtype=mx.float32), self._mask(), None),
            (valid, mx.zeros((9, 9), dtype=mx.float32), None),
            (valid, mx.ones((2, 9, 9), dtype=mx.bool_), None),
            (valid, self._mask(), _PlainBitsCache()),
        )
        for sources, mask, cache in mismatches:
            with self.subTest(
                shapes=tuple(source.shape for source in sources),
                mask_dtype=mask.dtype,
                cache=type(cache).__name__,
            ):
                self.assertFalse(
                    kimi_k3._can_use_factorized_sdpa_prefill(
                        *sources,
                        mask=mask,
                        cache=cache,
                    )
                )


if __name__ == "__main__":
    unittest.main()
