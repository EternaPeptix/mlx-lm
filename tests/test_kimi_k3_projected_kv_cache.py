# Copyright © 2026 Apple Inc.

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import mlx.core as mx

from mlx_lm.models import kimi_k3
from mlx_lm.generate import _make_cache
from mlx_lm.models.cache import (
    ArraysCache,
    BatchKVCache,
    KVCache,
    load_prompt_cache,
    save_prompt_cache,
)


def _released_tp2_attention():
    args = kimi_k3.TextArgs(
        hidden_size=8,
        num_attention_heads=96,
        num_key_value_heads=96,
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
    attention.embed_q = attention.embed_q.to_quantized(64, 6)
    attention.unembed_out = attention.unembed_out.to_quantized(64, 6)
    attention.embed_q.apply(lambda value: value[:48])
    attention.unembed_out.apply(lambda value: value[:48])
    attention.num_heads = 48
    attention.eval()
    mx.eval(attention.parameters())
    return attention


class _SpeculativeOwner:
    """Minimal duck-typed owner for the K3 transaction methods."""

    begin_speculative_cache = kimi_k3.LanguageModel.begin_speculative_cache
    _validate_speculative_transaction = (
        kimi_k3.LanguageModel._validate_speculative_transaction
    )
    resolve_speculative_cache = kimi_k3.LanguageModel.resolve_speculative_cache
    cancel_speculative_cache = kimi_k3.LanguageModel.cancel_speculative_cache

    def __init__(self):
        self.model = SimpleNamespace(pipeline_size=1)
        self.layers = [
            SimpleNamespace(is_linear=True),
            SimpleNamespace(is_linear=False),
        ]


class TestKimiK3ProjectedKVCachePortability(unittest.TestCase):
    def test_prompt_cache_round_trip_serializes_as_portable_base_cache(self):
        cache = kimi_k3.KimiK3ProjectedKVCache()
        cache.state = (
            mx.zeros((1, 1, 7, 512), dtype=mx.bfloat16),
            mx.zeros((1, 1, 7, 64), dtype=mx.bfloat16),
        )
        cache.projected_keys = mx.zeros((1, 48, 7, 128), dtype=mx.bfloat16)
        cache.projected_values = mx.zeros((1, 48, 7, 128), dtype=mx.bfloat16)

        with tempfile.NamedTemporaryFile(suffix=".safetensors") as file:
            save_prompt_cache(file.name, [cache])
            loaded = load_prompt_cache(file.name)

        self.assertEqual(len(loaded), 1)
        self.assertIs(type(loaded[0]), KVCache)
        self.assertEqual(loaded[0].offset, cache.offset)
        mx.eval(*loaded[0].state, *cache.state)
        for actual, expected in zip(loaded[0].state, cache.state, strict=True):
            self.assertTrue(mx.array_equal(actual, expected).item())

    def test_batch_cache_conversion_falls_back_to_batch_kv_cache(self):
        class ProjectedCacheModel:
            @staticmethod
            def make_cache():
                return [kimi_k3.KimiK3ProjectedKVCache()]

        cache = _make_cache(ProjectedCacheModel(), [0, 2], None)
        self.assertEqual(len(cache), 1)
        self.assertIs(type(cache[0]), BatchKVCache)


@unittest.skipUnless(mx.metal.is_available(), "requires Metal")
class TestKimiK3ProjectedKVCache(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mx.random.seed(20260805)
        cls.attention = _released_tp2_attention()

    def setUp(self):
        self.attention.eval()
        self.selector = mock.patch.dict(
            os.environ,
            {
                kimi_k3.PROJECTED_KV_CACHE_ENV: "1",
                kimi_k3.PROJECTED_KV_CACHE_MAX_TOKENS_ENV: "32768",
            },
            clear=True,
        )
        self.selector.start()

    def tearDown(self):
        self.selector.stop()

    @staticmethod
    def _latent(tokens):
        return mx.random.normal((1, 1, tokens, 512)).astype(mx.bfloat16)

    @staticmethod
    def _rope(tokens):
        return mx.random.normal((1, 1, tokens, 64)).astype(mx.bfloat16)

    def _append(self, cache, tokens, attention=None, *, activate=True):
        attention = self.attention if attention is None else attention
        if activate and cache._projected_transaction_token is None:
            cache.begin_projected_transaction(object(), tokens)
        previous = cache.offset
        latent, _ = cache.update_and_fetch(self._latent(tokens), self._rope(tokens))
        result = kimi_k3._maybe_projected_kv(
            attention,
            cache,
            latent,
            batch_size=1,
            query_length=tokens,
            previous_offset=previous,
        )
        self.assertIsNotNone(result)
        return latent, result

    def _transaction_fixture(self, prefix=64):
        array_cache = ArraysCache(size=1)
        array_cache.cache = [mx.zeros((1, 4), dtype=mx.bfloat16)]
        projected_cache = kimi_k3.KimiK3ProjectedKVCache()
        projected_cache.update_and_fetch(self._latent(prefix), self._rope(prefix))
        owner = _SpeculativeOwner()
        return owner, array_cache, projected_cache, [array_cache, projected_cache]

    def test_ordinary_q3_without_transaction_does_not_allocate_projection(self):
        cache = kimi_k3.KimiK3ProjectedKVCache()
        cache.update_and_fetch(self._latent(64), self._rope(64))

        previous = cache.offset
        latent, _ = cache.update_and_fetch(self._latent(3), self._rope(3))
        projected = kimi_k3._maybe_projected_kv(
            self.attention,
            cache,
            latent,
            batch_size=1,
            query_length=3,
            previous_offset=previous,
        )

        self.assertIsNone(projected)
        self.assertIsNone(cache.projected_keys)
        self.assertIsNone(cache.projected_values)
        self.assertIsNone(cache._projected_transaction_token)
        self.assertEqual(cache._projected_transaction_width, 0)

    def test_actual_begin_activates_q3_and_cancel_clears_marker(self):
        owner, array_cache, projected_cache, cache = self._transaction_fixture()
        transaction = owner.begin_speculative_cache(cache, width=3)
        token = transaction.projected_transaction_token

        self.assertIsNotNone(token)
        self.assertTrue(projected_cache.matches_projected_transaction(token, 3))
        latent, (keys, values) = self._append(projected_cache, 3)
        expected_keys = self.attention.embed_q(latent, transpose=False)
        expected_values = self.attention.unembed_out(latent)
        mx.eval(keys, values, expected_keys, expected_values)
        self.assertTrue(mx.array_equal(keys, expected_keys).item())
        self.assertTrue(mx.array_equal(values, expected_values).item())

        owner.cancel_speculative_cache(transaction)
        self.assertIsNone(projected_cache._projected_transaction_token)
        self.assertEqual(projected_cache._projected_transaction_width, 0)
        self.assertEqual(array_cache.speculative_width, 0)

    def test_nested_begin_fails_closed_without_replacing_marker(self):
        owner, array_cache, projected_cache, cache = self._transaction_fixture()
        transaction = owner.begin_speculative_cache(cache, width=3)
        token = transaction.projected_transaction_token

        with self.assertRaisesRegex(ValueError, "already active"):
            owner.begin_speculative_cache(cache, width=3)

        self.assertTrue(projected_cache.matches_projected_transaction(token, 3))
        self.assertEqual(array_cache.speculative_width, 3)
        owner.cancel_speculative_cache(transaction)
        self.assertIsNone(projected_cache._projected_transaction_token)

    def test_resolve_clears_projected_transaction_marker(self):
        owner, array_cache, projected_cache, cache = self._transaction_fixture()
        initial_array = array_cache.cache[0]
        initial_offset = projected_cache.offset
        transaction = owner.begin_speculative_cache(cache, width=3)

        history = mx.stack([initial_array + 1, initial_array + 2, initial_array + 3])
        array_cache.cache = [history[-1]]
        array_cache.capture_speculative([history])
        self._append(projected_cache, 3)
        owner.resolve_speculative_cache(transaction, consumed=1)

        mx.eval(array_cache.cache[0])
        self.assertTrue(mx.array_equal(array_cache.cache[0], history[0]).item())
        self.assertEqual(projected_cache.offset, initial_offset + 1)
        self.assertIsNone(projected_cache._projected_transaction_token)
        self.assertEqual(projected_cache._projected_transaction_width, 0)

    def test_width_two_forward_and_resolve_remain_unmarked(self):
        owner, array_cache, projected_cache, cache = self._transaction_fixture()
        initial_array = array_cache.cache[0]
        initial_offset = projected_cache.offset
        transaction = owner.begin_speculative_cache(cache, width=2)

        self.assertIsNone(transaction.projected_transaction_token)
        self.assertIsNone(projected_cache._projected_transaction_token)
        history = mx.stack([initial_array + 1, initial_array + 2])
        array_cache.cache = [history[-1]]
        array_cache.capture_speculative([history])
        previous = projected_cache.offset
        latent, _ = projected_cache.update_and_fetch(
            self._latent(2),
            self._rope(2),
        )
        projected = kimi_k3._maybe_projected_kv(
            self.attention,
            projected_cache,
            latent,
            batch_size=1,
            query_length=2,
            previous_offset=previous,
        )
        self.assertIsNone(projected)
        self.assertIsNone(projected_cache._projected_transaction_token)

        owner.resolve_speculative_cache(transaction, consumed=1)
        self.assertEqual(projected_cache.offset, initial_offset + 1)
        self.assertEqual(array_cache.speculative_width, 0)
        self.assertIsNone(projected_cache._projected_transaction_token)

    def test_unsupported_q3_geometry_preserves_marker_until_resolve(self):
        owner, array_cache, projected_cache, cache = self._transaction_fixture()
        initial_array = array_cache.cache[0]
        transaction = owner.begin_speculative_cache(cache, width=3)
        token = transaction.projected_transaction_token

        history = mx.stack([initial_array + 1, initial_array + 2, initial_array + 3])
        array_cache.cache = [history[-1]]
        array_cache.capture_speculative([history])
        previous = projected_cache.offset
        latent, _ = projected_cache.update_and_fetch(
            self._latent(3),
            self._rope(3),
        )
        projected = kimi_k3._maybe_projected_kv(
            SimpleNamespace(training=False, num_heads=47),
            projected_cache,
            latent,
            batch_size=1,
            query_length=3,
            previous_offset=previous,
        )
        self.assertIsNone(projected)
        self.assertTrue(projected_cache.matches_projected_transaction(token, 3))

        owner.resolve_speculative_cache(transaction, consumed=1)
        self.assertEqual(array_cache.speculative_width, 0)
        self.assertIsNone(projected_cache._projected_transaction_token)

    def test_resolve_error_cancels_and_clears_marker(self):
        owner, array_cache, projected_cache, cache = self._transaction_fixture()
        initial_offset = projected_cache.offset
        transaction = owner.begin_speculative_cache(cache, width=3)

        with self.assertRaisesRegex(ValueError, "checkpoints are incomplete"):
            owner.resolve_speculative_cache(transaction, consumed=1)

        self.assertFalse(transaction.active)
        self.assertEqual(array_cache.speculative_width, 0)
        self.assertEqual(projected_cache.offset, initial_offset)
        self.assertIsNone(projected_cache._projected_transaction_token)
        self.assertEqual(projected_cache._projected_transaction_width, 0)

    def test_changed_transaction_identity_fails_closed_and_clears_marker(self):
        owner, array_cache, projected_cache, cache = self._transaction_fixture()
        transaction = owner.begin_speculative_cache(cache, width=3)
        projected_cache.clear_projected_transaction()
        projected_cache.begin_projected_transaction(object(), width=3)

        with self.assertRaisesRegex(ValueError, "transaction marker changed"):
            owner.resolve_speculative_cache(transaction, consumed=1)

        self.assertFalse(transaction.active)
        self.assertEqual(array_cache.speculative_width, 0)
        self.assertIsNone(projected_cache._projected_transaction_token)
        self.assertEqual(projected_cache._projected_transaction_width, 0)

    def test_state_restore_clears_active_transaction_marker(self):
        owner, _, projected_cache, cache = self._transaction_fixture()
        transaction = owner.begin_speculative_cache(cache, width=3)
        self.assertIsNotNone(projected_cache._projected_transaction_token)

        state = projected_cache.state
        projected_cache.state = state
        self.assertIsNone(projected_cache._projected_transaction_token)
        self.assertEqual(projected_cache._projected_transaction_width, 0)
        self.assertIsNone(projected_cache.projected_keys)
        self.assertIsNone(projected_cache.projected_values)

        owner.cancel_speculative_cache(transaction)

    def test_tp2_sharded_attention_uses_48_head_projected_cache_exactly(self):
        self.assertIsInstance(
            self.attention.embed_q,
            kimi_k3.QuantizedMultiLinear,
        )
        self.assertIsInstance(
            self.attention.unembed_out,
            kimi_k3.QuantizedMultiLinear,
        )
        for module in (
            self.attention.embed_q,
            self.attention.unembed_out,
        ):
            self.assertEqual(module.weight.shape[0], 48)
            self.assertEqual(module.scales.shape[0], 48)
            self.assertEqual(module.biases.shape[0], 48)
            self.assertTrue(kimi_k3._released_affine6_multilinear(module))

        cache = kimi_k3.KimiK3ProjectedKVCache()
        cache.update_and_fetch(self._latent(64), self._rope(64))

        latent, (keys, values) = self._append(cache, 3, self.attention)
        expected_keys = self.attention.embed_q(latent, transpose=False)
        expected_values = self.attention.unembed_out(latent)
        mx.eval(keys, values, expected_keys, expected_values)
        self.assertEqual(keys.shape, (1, 48, 67, 128))
        self.assertEqual(values.shape, (1, 48, 67, 128))
        self.assertTrue(mx.array_equal(keys, expected_keys).item())
        self.assertTrue(mx.array_equal(values, expected_values).item())

    def test_tp2_capacity_growth_is_exact_and_cancel_restores_old_handles(self):
        cache = kimi_k3.KimiK3ProjectedKVCache()
        cache.update_and_fetch(self._latent(253), self._rope(253))
        _, _ = self._append(cache, 3, self.attention)
        mx.eval(cache.projected_keys, cache.projected_values)
        self.assertEqual(cache.offset, 256)
        self.assertEqual(cache.projected_capacity, 256)

        # Commit only the anchor, leaving two stale speculative rows.  The next
        # Q3 both overwrites that tail and crosses the 256-row capacity edge.
        cache.offset = 254
        initial_keys = cache.keys
        initial_values = cache.values
        initial_offset = cache.offset
        initial_projected = cache.snapshot_projected()
        latent, (keys, values) = self._append(cache, 3, self.attention)
        expected_keys = self.attention.embed_q(latent, transpose=False)
        expected_values = self.attention.unembed_out(latent)
        mx.eval(keys, values, expected_keys, expected_values)
        self.assertEqual(cache.projected_capacity, 512)
        self.assertIsNot(cache.projected_keys, initial_projected[0])
        self.assertIsNot(cache.projected_values, initial_projected[1])
        self.assertTrue(mx.array_equal(keys, expected_keys).item())
        self.assertTrue(mx.array_equal(values, expected_values).item())

        owner = kimi_k3.LanguageModel.__new__(kimi_k3.LanguageModel)
        object.__setattr__(
            owner,
            "model",
            SimpleNamespace(layers=[None], start_idx=0, end_idx=1),
        )
        transaction = kimi_k3.KimiK3SpeculativeCacheTransaction(
            owner_id=id(owner),
            cache=[cache],
            width=3,
            array_states=[],
            kv_states=[
                (
                    0,
                    cache,
                    initial_keys,
                    initial_values,
                    initial_offset,
                )
            ],
            projected_kv_states=[(0, cache, initial_projected)],
        )
        owner.cancel_speculative_cache(transaction)
        self.assertIs(cache.keys, initial_keys)
        self.assertIs(cache.values, initial_values)
        self.assertEqual(cache.offset, initial_offset)
        self.assertIs(cache.projected_keys, initial_projected[0])
        self.assertIs(cache.projected_values, initial_projected[1])
        self.assertEqual(cache.projected_capacity, 256)

    def test_tp2_cached_prefix_is_exact_across_dispatch_boundaries(self):
        for prefix in (32, 33, 252, 253, 254, 255, 256, 257, 510, 511, 512, 513):
            with self.subTest(prefix=prefix):
                cache = kimi_k3.KimiK3ProjectedKVCache()
                cache.update_and_fetch(self._latent(prefix), self._rope(prefix))
                _, _ = self._append(cache, 3, self.attention)

                # Keep only the anchor so the next call must reuse the cached
                # prefix across a different total-row dispatch and overwrite
                # the rejected speculative tail.
                cache.offset = prefix + 1
                latent, (keys, values) = self._append(
                    cache,
                    3,
                    self.attention,
                )
                expected_keys = self.attention.embed_q(
                    latent,
                    transpose=False,
                )
                expected_values = self.attention.unembed_out(latent)
                mx.eval(keys, values, expected_keys, expected_values)
                self.assertTrue(mx.array_equal(keys, expected_keys).item())
                self.assertTrue(mx.array_equal(values, expected_values).item())

    def test_partial_resolution_clamps_projected_valid_offset(self):
        cache = kimi_k3.KimiK3ProjectedKVCache()
        cache.update_and_fetch(self._latent(64), self._rope(64))
        initial_keys = cache.keys
        initial_values = cache.values
        initial_offset = cache.offset
        initial_projected = cache.snapshot_projected()
        _, _ = self._append(cache, 3, self.attention)
        self.assertEqual(cache.projected_valid_offset, 67)

        owner = kimi_k3.LanguageModel.__new__(kimi_k3.LanguageModel)
        object.__setattr__(
            owner,
            "model",
            SimpleNamespace(layers=[None], start_idx=0, end_idx=1),
        )
        transaction = kimi_k3.KimiK3SpeculativeCacheTransaction(
            owner_id=id(owner),
            cache=[cache],
            width=3,
            array_states=[],
            kv_states=[
                (
                    0,
                    cache,
                    initial_keys,
                    initial_values,
                    initial_offset,
                )
            ],
            projected_kv_states=[(0, cache, initial_projected)],
        )
        owner.resolve_speculative_cache(transaction, consumed=1)
        self.assertEqual(cache.offset, 65)
        self.assertEqual(cache.projected_valid_offset, 65)

    def test_padded_q3_append_is_bit_exact_after_partial_commit(self):
        cache = kimi_k3.KimiK3ProjectedKVCache()
        cache.update_and_fetch(self._latent(64), self._rope(64))

        latent, (keys, values) = self._append(cache, 3)
        expected_keys = self.attention.embed_q(latent, transpose=False)
        expected_values = self.attention.unembed_out(latent)
        mx.eval(keys, values, expected_keys, expected_values)
        self.assertTrue(mx.array_equal(keys, expected_keys).item())
        self.assertTrue(mx.array_equal(values, expected_values).item())

        # Commit only anchor + zero accepted drafts. Rejected projected rows
        # remain beyond the logical offset and must be overwritten exactly.
        cache.offset = 65
        latent, (keys, values) = self._append(cache, 3)
        expected_keys = self.attention.embed_q(latent, transpose=False)
        expected_values = self.attention.unembed_out(latent)
        mx.eval(keys, values, expected_keys, expected_values)
        self.assertTrue(mx.array_equal(keys, expected_keys).item())
        self.assertTrue(mx.array_equal(values, expected_values).item())

    def test_state_restore_clears_nonserialized_projection_and_counts_bytes(self):
        cache = kimi_k3.KimiK3ProjectedKVCache()
        cache.update_and_fetch(self._latent(64), self._rope(64))
        _, _ = self._append(cache, 3)
        mx.eval(cache.projected_keys, cache.projected_values)

        base_bytes = cache.keys.nbytes + cache.values.nbytes
        projected_bytes = cache.projected_keys.nbytes + cache.projected_values.nbytes
        self.assertEqual(cache.nbytes, base_bytes + projected_bytes)

        state = cache.state
        restored_base_bytes = state[0].nbytes + state[1].nbytes
        cache.state = state
        self.assertEqual(cache.offset, 67)
        self.assertIsNone(cache.projected_keys)
        self.assertIsNone(cache.projected_values)
        self.assertEqual(cache.nbytes, restored_base_bytes)

    def test_cancel_restores_projected_handles_and_capacity(self):
        cache = kimi_k3.KimiK3ProjectedKVCache()
        cache.update_and_fetch(self._latent(64), self._rope(64))
        _, _ = self._append(cache, 3)
        initial_projected = cache.snapshot_projected()
        initial_keys = cache.keys
        initial_values = cache.values
        initial_offset = cache.offset

        owner = kimi_k3.LanguageModel.__new__(kimi_k3.LanguageModel)
        transaction = kimi_k3.KimiK3SpeculativeCacheTransaction(
            owner_id=id(owner),
            cache=[cache],
            width=3,
            array_states=[],
            kv_states=[
                (
                    0,
                    cache,
                    initial_keys,
                    initial_values,
                    initial_offset,
                )
            ],
            projected_kv_states=[(0, cache, initial_projected)],
        )

        cache.projected_keys = mx.zeros((1, 48, 512, 128), mx.bfloat16)
        cache.projected_values = mx.zeros((1, 48, 512, 128), mx.bfloat16)
        cache.projected_capacity = 512
        cache.offset += 3
        owner.cancel_speculative_cache(transaction)

        self.assertIs(cache.keys, initial_keys)
        self.assertIs(cache.values, initial_values)
        self.assertEqual(cache.offset, initial_offset)
        self.assertIs(cache.projected_keys, initial_projected[0])
        self.assertIs(cache.projected_values, initial_projected[1])
        self.assertEqual(cache.projected_capacity, initial_projected[2])

    def test_ordinary_query_clears_projected_cache(self):
        cache = kimi_k3.KimiK3ProjectedKVCache()
        cache.update_and_fetch(self._latent(64), self._rope(64))
        _, _ = self._append(cache, 3)
        self.assertIsNotNone(cache.projected_keys)
        token = cache._projected_transaction_token

        previous = cache.offset
        latent, _ = cache.update_and_fetch(self._latent(1), self._rope(1))
        projected = kimi_k3._maybe_projected_kv(
            self.attention,
            cache,
            latent,
            batch_size=1,
            query_length=1,
            previous_offset=previous,
        )
        self.assertIsNone(projected)
        self.assertIsNone(cache.projected_keys)
        self.assertIsNone(cache.projected_values)
        self.assertIs(cache._projected_transaction_token, token)
        self.assertEqual(cache._projected_transaction_width, 3)

    def test_selector_and_memory_cap_are_strict(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(kimi_k3._projected_kv_cache_requested())
            self.assertEqual(kimi_k3._projected_kv_cache_max_tokens(), 32768)
        with mock.patch.dict(
            os.environ,
            {kimi_k3.PROJECTED_KV_CACHE_ENV: "yes"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "must be exactly"):
                kimi_k3._projected_kv_cache_requested()
        for invalid in ("not-an-int", "31", "131073"):
            with mock.patch.dict(
                os.environ,
                {kimi_k3.PROJECTED_KV_CACHE_MAX_TOKENS_ENV: invalid},
                clear=True,
            ):
                with self.assertRaises(ValueError):
                    kimi_k3._projected_kv_cache_max_tokens()


if __name__ == "__main__":
    unittest.main()
