# Copyright © 2026 Apple Inc.

import os
import unittest
from types import SimpleNamespace
from unittest import mock

import mlx.core as mx

from mlx_lm.models import kimi_k3
from mlx_lm.models.cache import ArraysCache, KVCache


def _mask(query_length: int, key_length: int) -> mx.array:
    offset = key_length - query_length
    return mx.arange(key_length)[None, :] <= (mx.arange(query_length)[:, None] + offset)


def _attention(*, enabled: bool) -> kimi_k3.KimiK3MLAAttention:
    args = kimi_k3.TextArgs(
        hidden_size=8,
        num_attention_heads=48,
        num_key_value_heads=48,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        mla_use_nope=True,
        mla_use_output_gate=True,
    )
    with mock.patch.dict(
        os.environ,
        {
            kimi_k3.K3_TP2_SEQUENTIAL_ABSORBED_Q3_ENV: "1" if enabled else "0",
        },
        clear=True,
    ):
        attention = kimi_k3.KimiK3MLAAttention(args)
    attention.set_dtype(mx.bfloat16)
    attention.eval()
    mx.eval(attention.parameters())
    return attention


def _cache(
    latent: mx.array,
    rope: mx.array,
    *,
    projected: bool,
) -> KVCache:
    cache = kimi_k3.KimiK3ProjectedKVCache() if projected else KVCache()
    cache.state = (latent, rope)
    return cache


class _SpeculativeOwner:
    begin_speculative_cache = kimi_k3.LanguageModel.begin_speculative_cache
    cancel_speculative_cache = kimi_k3.LanguageModel.cancel_speculative_cache

    def __init__(self):
        self.model = SimpleNamespace(pipeline_size=1)
        self.layers = [
            SimpleNamespace(is_linear=True),
            SimpleNamespace(is_linear=False),
        ]


@unittest.skipUnless(mx.metal.is_available(), "requires Metal")
class TestKimiK3SequentialAbsorbedQ3(unittest.TestCase):
    def test_selector_is_default_off_and_invalid_values_fail_closed(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(kimi_k3._k3_tp2_sequential_absorbed_q3_requested())
        for value in ("", "yes", "2", "true"):
            with (
                self.subTest(value=value),
                mock.patch.dict(
                    os.environ,
                    {kimi_k3.K3_TP2_SEQUENTIAL_ABSORBED_Q3_ENV: value},
                    clear=True,
                ),
                self.assertRaisesRegex(ValueError, "must be exactly '0' or '1'"),
            ):
                kimi_k3._k3_tp2_sequential_absorbed_q3_requested()

    def test_selector_accepts_only_cached_released_q3_geometry(self):
        attention = _attention(enabled=True)
        cache = _cache(
            mx.zeros((1, 1, 5, 512), dtype=mx.bfloat16),
            mx.zeros((1, 1, 5, 64), dtype=mx.bfloat16),
            projected=True,
        )
        cache.begin_projected_transaction(object(), 3)
        base = {
            "attention": attention,
            "x": mx.zeros((1, 3, 8), dtype=mx.bfloat16),
            "mask": _mask(3, 8),
            "cache": cache,
            "requested": True,
            "previous_offset": 5,
        }
        self.assertTrue(kimi_k3._can_use_k3_tp2_sequential_absorbed_q3(**base))

        plain = _cache(*cache.state, projected=False)
        self.assertTrue(
            kimi_k3._can_use_k3_tp2_sequential_absorbed_q3(**{**base, "cache": plain})
        )

        unmarked = _cache(*cache.state, projected=True)
        cases = {
            "default off": {"requested": False},
            "training": {
                "attention": mock.Mock(**{**attention.__dict__, "training": True})
            },
            "no cache": {"cache": None},
            "unmarked projected cache": {"cache": unmarked},
            "prefill": {
                "cache": _cache(
                    mx.zeros((1, 1, 0, 512), dtype=mx.bfloat16),
                    mx.zeros((1, 1, 0, 64), dtype=mx.bfloat16),
                    projected=False,
                ),
                "previous_offset": 0,
                "mask": _mask(3, 3),
            },
            "Q1": {
                "x": mx.zeros((1, 1, 8), dtype=mx.bfloat16),
                "mask": _mask(1, 6),
            },
            "Q2": {
                "x": mx.zeros((1, 2, 8), dtype=mx.bfloat16),
                "mask": _mask(2, 7),
            },
            "Q4": {
                "x": mx.zeros((1, 4, 8), dtype=mx.bfloat16),
                "mask": _mask(4, 9),
            },
            "wrong mask length": {"mask": _mask(3, 7)},
            "non-boolean mask": {"mask": mx.zeros((3, 8), dtype=mx.bfloat16)},
            "float32 hidden": {"x": mx.zeros((1, 3, 8), dtype=mx.float32)},
        }
        for name, changes in cases.items():
            with self.subTest(name=name):
                self.assertFalse(
                    kimi_k3._can_use_k3_tp2_sequential_absorbed_q3(
                        **{**base, **changes}
                    )
                )

    def test_q3_matches_three_actual_q1_calls_and_authoritative_cache(self):
        mx.random.seed(20260806)
        attention = _attention(enabled=True)
        x = (0.125 * mx.random.normal((1, 3, 8))).astype(mx.bfloat16)
        initial_latent = (0.125 * mx.random.normal((1, 1, 11, 512))).astype(mx.bfloat16)
        initial_rope = (0.125 * mx.random.normal((1, 1, 11, 64))).astype(mx.bfloat16)
        mx.eval(x, initial_latent, initial_rope)

        reference_cache = _cache(initial_latent, initial_rope, projected=True)
        attention.use_sequential_absorbed_q3 = False
        expected_rows = []
        for row in range(3):
            expected_rows.append(
                attention(x[:, row : row + 1, :], None, reference_cache)
            )
        expected = mx.concatenate(expected_rows, axis=1)

        candidate_cache = _cache(initial_latent, initial_rope, projected=True)
        candidate_cache.projected_keys = mx.full(
            (1, 48, 11, 128),
            7,
            dtype=mx.bfloat16,
        )
        candidate_cache.projected_values = mx.full(
            (1, 48, 11, 128),
            -3,
            dtype=mx.bfloat16,
        )
        candidate_cache.projected_capacity = 11
        candidate_cache.projected_valid_offset = 11
        candidate_cache.projected_owner_id = 123
        projected_snapshot = candidate_cache.snapshot_projected()
        array_cache = ArraysCache(size=1)
        array_cache.cache = [mx.zeros((1, 4), dtype=mx.bfloat16)]
        owner = _SpeculativeOwner()
        transaction = owner.begin_speculative_cache(
            [array_cache, candidate_cache],
            width=3,
        )
        transaction_token = transaction.projected_transaction_token
        attention.use_sequential_absorbed_q3 = True
        actual = attention(x, _mask(3, 14), candidate_cache)

        mx.eval(expected, actual, reference_cache.state, candidate_cache.state)
        self.assertTrue(mx.array_equal(actual, expected).item())
        for candidate, reference in zip(
            candidate_cache.state,
            reference_cache.state,
            strict=True,
        ):
            self.assertTrue(mx.array_equal(candidate, reference).item())
        self.assertEqual(candidate_cache.offset, 14)
        self.assertIsNone(candidate_cache.projected_keys)
        self.assertIsNone(candidate_cache.projected_values)
        self.assertEqual(candidate_cache.projected_capacity, 0)
        self.assertEqual(candidate_cache.projected_valid_offset, 0)
        self.assertIsNone(candidate_cache.projected_owner_id)
        self.assertTrue(
            candidate_cache.matches_projected_transaction(transaction_token, 3)
        )

        owner.cancel_speculative_cache(transaction)
        self.assertEqual(candidate_cache.offset, 11)
        self.assertIs(candidate_cache.projected_keys, projected_snapshot[0])
        self.assertIs(candidate_cache.projected_values, projected_snapshot[1])
        self.assertEqual(candidate_cache.projected_capacity, 11)
        self.assertEqual(candidate_cache.projected_valid_offset, 11)
        self.assertEqual(candidate_cache.projected_owner_id, 123)
        self.assertIsNone(candidate_cache._projected_transaction_token)
        self.assertEqual(array_cache.speculative_width, 0)

    def test_append_once_and_latent_q1_prefix_and_mask_lengths(self):
        attention = _attention(enabled=True)
        cache = _cache(
            mx.zeros((1, 1, 5, 512), dtype=mx.bfloat16),
            mx.zeros((1, 1, 5, 64), dtype=mx.bfloat16),
            projected=True,
        )
        cache.begin_projected_transaction(object(), 3)
        calls = []

        def sdpa(query, key, value, **kwargs):
            calls.append(
                (
                    query.shape,
                    key.shape,
                    value.shape,
                    kwargs["mask"].shape,
                    cache.offset,
                )
            )
            return mx.zeros((*query.shape[:-1], value.shape[-1]), query.dtype)

        with (
            mock.patch.object(
                cache,
                "update_and_fetch",
                wraps=cache.update_and_fetch,
            ) as update,
            mock.patch.object(
                kimi_k3,
                "scaled_dot_product_attention",
                side_effect=sdpa,
            ),
        ):
            output = attention(
                mx.zeros((1, 3, 8), dtype=mx.bfloat16),
                _mask(3, 8),
                cache,
            )
            mx.eval(output)

        self.assertEqual(update.call_count, 1)
        self.assertEqual(
            calls,
            [
                ((1, 48, 1, 512), (1, 1, 6, 512), (1, 1, 6, 512), (1, 48, 1, 6), 8),
                ((1, 48, 1, 512), (1, 1, 7, 512), (1, 1, 7, 512), (1, 48, 1, 7), 8),
                ((1, 48, 1, 512), (1, 1, 8, 512), (1, 1, 8, 512), (1, 48, 1, 8), 8),
            ],
        )

    def test_selector_off_keeps_current_projected_q3_source_path(self):
        attention = _attention(enabled=False)
        attention.embed_q = attention.embed_q.to_quantized(64, 6)
        attention.unembed_out = attention.unembed_out.to_quantized(64, 6)
        mx.eval(attention.embed_q.parameters(), attention.unembed_out.parameters())
        cache = _cache(
            mx.zeros((1, 1, 32, 512), dtype=mx.bfloat16),
            mx.zeros((1, 1, 32, 64), dtype=mx.bfloat16),
            projected=True,
        )
        token = object()
        cache.begin_projected_transaction(token, 3)
        env = {
            kimi_k3.PROJECTED_KV_CACHE_ENV: "1",
            kimi_k3.PROJECTED_KV_CACHE_MAX_TOKENS_ENV: "32768",
            kimi_k3.K3_TP2_SEQUENTIAL_ABSORBED_Q3_ENV: "0",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            output = attention(
                mx.zeros((1, 3, 8), dtype=mx.bfloat16),
                _mask(3, 35),
                cache,
            )
            mx.eval(output, cache.projected_keys, cache.projected_values)

        self.assertEqual(cache.projected_valid_offset, 35)
        self.assertEqual(cache.projected_owner_id, id(attention))
        self.assertEqual(cache.projected_keys.shape, (1, 48, 256, 128))
        self.assertEqual(cache.projected_values.shape, (1, 48, 256, 128))
        self.assertTrue(cache.matches_projected_transaction(token, 3))


if __name__ == "__main__":
    unittest.main()
