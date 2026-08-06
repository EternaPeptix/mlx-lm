# Copyright © 2026 Apple Inc.

import copy
import os
import unittest
from unittest import mock

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_lm.models import kimi_k3


def _config(
    num_layers=8,
    kda_layers=(1, 2, 3, 5, 6, 7),
    block_size=4,
):
    full_attn_layers = [i for i in range(1, num_layers + 1) if i not in kda_layers]
    return {
        "model_type": "kimi_k3",
        "vocab_size": 1024,
        "num_hidden_layers": num_layers,
        "text_config": {
            "model_type": "kimi_linear",
            "vocab_size": 1024,
            "hidden_size": 64,
            "num_hidden_layers": num_layers,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "intermediate_size": 96,
            "rms_norm_eps": 1e-5,
            "hidden_act": "situ",
            "activation_situ_beta": 4.0,
            "activation_situ_linear_beta": 25.0,
            "linear_attn_config": {
                "kda_layers": list(kda_layers),
                "full_attn_layers": full_attn_layers,
                "num_heads": 2,
                "head_dim": 32,
                "short_conv_kernel_size": 4,
                "gate_lower_bound": -5.0,
                "use_full_rank_gate": True,
            },
            "num_experts": 8,
            "moe_intermediate_size": 32,
            "q_lora_rank": 24,
            "kv_lora_rank": 16,
            "qk_nope_head_dim": 16,
            "qk_rope_head_dim": 8,
            "v_head_dim": 16,
            "mla_use_nope": True,
            "mla_use_output_gate": True,
            "num_experts_per_token": 2,
            "num_shared_experts": 1,
            "first_k_dense_replace": 1,
            "routed_expert_hidden_size": 32,
            "latent_moe_use_norm": True,
            "attn_res_block_size": block_size,
        },
    }


def _make_model(config=None, dtype=mx.bfloat16):
    mx.random.seed(0)
    args = kimi_k3.ModelArgs.from_dict(config or _config())
    model = kimi_k3.Model(args)
    model.set_dtype(dtype)
    model.eval()
    mx.eval(model.parameters())
    return model


def _warm_cache(model, length=8):
    cache = model.make_cache()
    tokens = mx.arange(length, dtype=mx.int32)[None] % model.args.text_config.vocab_size
    logits = model(tokens, cache=cache)
    mx.eval(logits, [c.state for c in cache])
    return cache


def _as_batch_cache(cache):
    merged = [type(c).merge([c]) for c in cache]
    mx.eval([c.state for c in merged])
    return merged


def _assert_tree_equal(test, left, right):
    left = tree_flatten(left)
    right = tree_flatten(right)
    test.assertEqual([k for k, _ in left], [k for k, _ in right])
    for (_, lhs), (_, rhs) in zip(left, right, strict=True):
        if isinstance(lhs, mx.array):
            test.assertIsInstance(rhs, mx.array)
            test.assertTrue(mx.array_equal(lhs, rhs).item())
        else:
            test.assertEqual(lhs, rhs)


def _assert_optional_array_equal(test, left, right):
    if left is None or right is None:
        test.assertIs(left, right)
    else:
        test.assertEqual(left.shape, right.shape)
        test.assertEqual(left.dtype, right.dtype)
        test.assertTrue(mx.array_equal(left, right).item())


def _assert_cache_equal(test, left, right):
    test.assertEqual([type(c) for c in left], [type(c) for c in right])
    for lhs, rhs in zip(left, right, strict=True):
        if type(lhs) is kimi_k3.ArraysCache:
            _assert_tree_equal(test, lhs.cache, rhs.cache)
            _assert_optional_array_equal(test, lhs.lengths, rhs.lengths)
            _assert_optional_array_equal(test, lhs.left_padding, rhs.left_padding)
        elif isinstance(lhs, kimi_k3.KVCache):
            test.assertEqual(lhs.offset, rhs.offset)
            _assert_optional_array_equal(test, lhs.keys, rhs.keys)
            _assert_optional_array_equal(test, lhs.values, rhs.values)
        elif type(lhs) is kimi_k3.BatchKVCache:
            test.assertEqual(lhs._idx, rhs._idx)
            _assert_optional_array_equal(test, lhs.keys, rhs.keys)
            _assert_optional_array_equal(test, lhs.values, rhs.values)
            _assert_optional_array_equal(test, lhs.offset, rhs.offset)
            _assert_optional_array_equal(test, lhs.left_padding, rhs.left_padding)
            _assert_optional_array_equal(
                test,
                lhs._right_padding,
                rhs._right_padding,
            )
        else:
            test.fail(f"Unsupported cache type in equality helper: {type(lhs)}")


class TestKimiK3CompiledDecodeSelector(unittest.TestCase):
    def test_selector_accepts_all_none_indices_and_inclusive_ranges(self):
        parse = kimi_k3._parse_compiled_decode_segments

        self.assertEqual(parse("all", 25), frozenset(range(25)))
        self.assertEqual(parse("none", 25), frozenset())
        self.assertEqual(parse("0, 3-5, 24", 25), frozenset((0, 3, 4, 5, 24)))
        self.assertEqual(parse("2-2,2", 25), frozenset((2,)))

    def test_selector_rejects_ambiguous_or_out_of_range_values(self):
        parse = kimi_k3._parse_compiled_decode_segments

        for selector in (
            "",
            " ",
            "0,,1",
            "-1",
            "1-",
            "1-2-3",
            "3-2",
            "25",
            "all,1",
            "*",
        ):
            with self.subTest(selector=selector):
                with self.assertRaises(ValueError):
                    parse(selector, 25)

        with self.assertRaises(ValueError):
            parse("none", -1)
        with self.assertRaises(ValueError):
            parse("0", 0)


class TestKimiK3AsyncDecodeBoundarySelector(unittest.TestCase):
    def test_selector_accepts_named_ladders_and_explicit_ranges(self):
        parse = kimi_k3._parse_async_decode_boundaries

        self.assertEqual(parse("none", 93), frozenset())
        self.assertEqual(
            parse("laguna8", 93),
            frozenset((1, 7, 15, 23, 31, 39, 47, 55, 63, 71, 79, 87)),
        )
        self.assertEqual(
            parse("block8", 93),
            frozenset((7, 15, 23, 31, 39, 47, 55, 63, 71, 79, 87)),
        )
        self.assertEqual(parse("all", 4), frozenset((0, 1, 2)))
        self.assertEqual(parse("0, 3-5, 7", 9), frozenset((0, 3, 4, 5, 7)))

    def test_selector_rejects_empty_ambiguous_and_final_boundaries(self):
        parse = kimi_k3._parse_async_decode_boundaries

        for selector in ("", " ", "0,,1", "-1", "1-", "3-2", "8", "*"):
            with self.subTest(selector=selector):
                with self.assertRaises(ValueError):
                    parse(selector, 9)

        with self.assertRaises(ValueError):
            parse("none", -1)
        with self.assertRaises(ValueError):
            parse("0", 1)


@unittest.skipUnless(mx.metal.is_available(), "requires Metal")
class TestKimiK3CompiledDecode(unittest.TestCase):
    def setUp(self):
        self._env = mock.patch.dict(
            os.environ,
            {
                kimi_k3.COMPILED_DECODE_ENV: "0",
                kimi_k3.COMPILED_DECODE_SEGMENTS_ENV: "all",
                "MLX_LM_KIMI_K3_FUSED_EXPERTS": "0",
            },
            clear=False,
        )
        self._env.start()
        kimi_k3.fused_k3_experts_enabled.cache_clear()

    def tearDown(self):
        kimi_k3.fused_k3_experts_enabled.cache_clear()
        self._env.stop()

    def test_flag_is_snapshotted_per_model(self):
        with mock.patch.dict(
            os.environ,
            {kimi_k3.COMPILED_DECODE_ENV: "1"},
            clear=False,
        ):
            enabled = _make_model()
        with mock.patch.dict(
            os.environ,
            {kimi_k3.COMPILED_DECODE_ENV: "0"},
            clear=False,
        ):
            disabled = _make_model()

        self.assertTrue(enabled.model._compiled_decode_enabled)
        self.assertFalse(disabled.model._compiled_decode_enabled)

    def test_segment_selector_is_snapshotted_per_model(self):
        with mock.patch.dict(
            os.environ,
            {
                kimi_k3.COMPILED_DECODE_ENV: "1",
                kimi_k3.COMPILED_DECODE_SEGMENTS_ENV: "0,2",
            },
            clear=False,
        ):
            model = _make_model()

        with mock.patch.dict(
            os.environ,
            {kimi_k3.COMPILED_DECODE_SEGMENTS_ENV: "all"},
            clear=False,
        ):
            self.assertEqual(
                model.model._compiled_decode_segments,
                frozenset((0, 2)),
            )

    def test_invalid_segment_selector_fails_before_serving(self):
        with mock.patch.dict(
            os.environ,
            {
                kimi_k3.COMPILED_DECODE_ENV: "1",
                kimi_k3.COMPILED_DECODE_SEGMENTS_ENV: "0-99",
            },
            clear=False,
        ):
            with self.assertRaises(ValueError):
                _make_model()

    def test_disabled_feature_ignores_segment_selector(self):
        with mock.patch.dict(
            os.environ,
            {
                kimi_k3.COMPILED_DECODE_ENV: "0",
                kimi_k3.COMPILED_DECODE_SEGMENTS_ENV: "not-a-selector",
            },
            clear=False,
        ):
            model = _make_model()

        self.assertFalse(model.model._compiled_decode_enabled)
        self.assertEqual(
            model.model._compiled_decode_segments,
            frozenset(range(3)),
        )

    def test_guard_accepts_only_supported_decode_state(self):
        model = _make_model()
        cache = _warm_cache(model)
        text_model = model.model
        text_model._compiled_decode_enabled = True
        h = text_model.embed_tokens(mx.array([[11]], dtype=mx.int32))
        layers = text_model.layers

        self.assertTrue(
            text_model._compiled_decode_eligible(h, cache, None, layers)
        )
        projected_cache = []
        for layer, layer_cache in zip(layers, cache, strict=True):
            if layer.is_linear:
                projected_cache.append(layer_cache)
            else:
                projected = kimi_k3.KimiK3ProjectedKVCache()
                projected.state = layer_cache.state
                projected_cache.append(projected)
        self.assertTrue(
            text_model._compiled_decode_eligible(
                h,
                projected_cache,
                None,
                layers,
            )
        )
        self.assertFalse(
            text_model._compiled_decode_eligible(
                mx.broadcast_to(h, (1, 2, h.shape[-1])),
                cache,
                None,
                layers,
            )
        )
        self.assertFalse(
            text_model._compiled_decode_eligible(
                mx.broadcast_to(h, (2, 1, h.shape[-1])),
                cache,
                None,
                layers,
            )
        )
        self.assertFalse(
            text_model._compiled_decode_eligible(
                h,
                cache,
                mx.ones((1, 1), dtype=mx.bool_),
                layers,
            )
        )

        kda_cache = cache[0]
        kda_cache.lengths = mx.array([1])
        self.assertFalse(
            text_model._compiled_decode_eligible(h, cache, None, layers)
        )
        kda_cache.lengths = None

        empty_cache = model.make_cache()
        self.assertFalse(
            text_model._compiled_decode_eligible(h, empty_cache, None, layers)
        )

        malformed_cache = copy.deepcopy(cache)
        malformed_cache[0][0] = malformed_cache[0][0][:, :-1, :]
        self.assertFalse(
            text_model._compiled_decode_eligible(
                h,
                malformed_cache,
                None,
                layers,
            )
        )

        with mock.patch.object(
            kimi_k3,
            "fused_k3_experts_enabled",
            return_value=True,
        ):
            self.assertFalse(
                text_model._compiled_decode_eligible(h, cache, None, layers)
            )

        text_model.train()
        self.assertFalse(
            text_model._compiled_decode_eligible(h, cache, None, layers)
        )

    def test_matches_eager_with_kv_and_single_batch_caches(self):
        model = _make_model()
        base_cache = _warm_cache(model)
        for batched in (False, True):
            with self.subTest(batched=batched):
                cache = _as_batch_cache(base_cache) if batched else base_cache
                eager_cache = copy.deepcopy(cache)
                compiled_cache = copy.deepcopy(cache)

                for token in (17, 23, 29, 31):
                    inputs = mx.array([[token]], dtype=mx.int32)
                    model.model._compiled_decode_enabled = False
                    eager = model(inputs, cache=eager_cache)
                    mx.eval(eager, [c.state for c in eager_cache])

                    model.model._compiled_decode_enabled = True
                    compiled = model(inputs, cache=compiled_cache)
                    mx.eval(compiled, [c.state for c in compiled_cache])

                    self.assertTrue(mx.array_equal(eager, compiled).item())
                    _assert_cache_equal(self, eager_cache, compiled_cache)

        schedule = model.model._compiled_decode_schedule
        self.assertEqual(schedule.prefix_kda_indices, (0, 1, 2))
        self.assertEqual(schedule.mla_indices, (3, 7))
        self.assertEqual(len(schedule.transitions), 1)
        self.assertEqual(schedule.transitions[0].kda_indices, (4, 5, 6))

    def test_projected_cache_preserves_compiled_q1_and_is_invalidated(self):
        model = _make_model()
        base_cache = _warm_cache(model)
        projected_cache = []
        for layer, layer_cache in zip(
            model.model.layers,
            base_cache,
            strict=True,
        ):
            if layer.is_linear:
                projected_cache.append(layer_cache)
                continue
            attention = layer.self_attn
            projected = kimi_k3.KimiK3ProjectedKVCache()
            projected.state = layer_cache.state
            projected.projected_keys = mx.zeros(
                (
                    1,
                    attention.num_heads,
                    projected.offset,
                    attention.qk_nope_head_dim,
                ),
                dtype=projected.keys.dtype,
            )
            projected.projected_values = mx.zeros(
                (
                    1,
                    attention.num_heads,
                    projected.offset,
                    attention.v_head_dim,
                ),
                dtype=projected.values.dtype,
            )
            projected.projected_capacity = projected.offset
            projected.projected_valid_offset = projected.offset
            projected.projected_owner_id = id(attention)
            projected_cache.append(projected)

        eager_cache = copy.deepcopy(projected_cache)
        compiled_cache = copy.deepcopy(projected_cache)
        inputs = mx.array([[17]], dtype=mx.int32)

        model.model._compiled_decode_enabled = False
        eager = model(inputs, cache=eager_cache)
        mx.eval(eager, [cache.state for cache in eager_cache])

        model.model._compiled_decode_enabled = True
        compiled = model(inputs, cache=compiled_cache)
        mx.eval(compiled, [cache.state for cache in compiled_cache])

        self.assertTrue(mx.array_equal(eager, compiled).item())
        _assert_cache_equal(self, eager_cache, compiled_cache)
        self.assertIsNotNone(model.model._compiled_decode_schedule)
        for cache in (*eager_cache, *compiled_cache):
            if isinstance(cache, kimi_k3.KimiK3ProjectedKVCache):
                self.assertIsNone(cache.projected_keys)
                self.assertIsNone(cache.projected_values)

    def test_mixed_compiled_and_eager_segment_schedules_match_eager(self):
        selectors = ("0", "1", "2", "0,2", "0-1", "1-2")
        for selector in selectors:
            with self.subTest(selector=selector):
                with mock.patch.dict(
                    os.environ,
                    {
                        kimi_k3.COMPILED_DECODE_ENV: "1",
                        kimi_k3.COMPILED_DECODE_SEGMENTS_ENV: selector,
                    },
                    clear=False,
                ):
                    model = _make_model()

                base_cache = _warm_cache(model)
                eager_cache = copy.deepcopy(base_cache)
                mixed_cache = copy.deepcopy(base_cache)
                eager_outputs = []
                mixed_outputs = []
                for token in (67, 71, 73):
                    inputs = mx.array([[token]], dtype=mx.int32)
                    model.model._compiled_decode_enabled = False
                    eager_outputs.append(model(inputs, cache=eager_cache))
                    model.model._compiled_decode_enabled = True
                    mixed_outputs.append(model(inputs, cache=mixed_cache))

                mx.eval(
                    eager_outputs,
                    mixed_outputs,
                    [c.state for c in eager_cache],
                    [c.state for c in mixed_cache],
                )
                for eager, mixed in zip(
                    eager_outputs,
                    mixed_outputs,
                    strict=True,
                ):
                    self.assertTrue(mx.array_equal(eager, mixed).item())
                _assert_cache_equal(self, eager_cache, mixed_cache)

                selected = kimi_k3._parse_compiled_decode_segments(selector, 3)
                schedule = model.model._compiled_decode_schedule
                self.assertIsNotNone(schedule)
                self.assertEqual(schedule.prefix is not None, 0 in selected)
                self.assertEqual(
                    schedule.transitions[0].step is not None,
                    1 in selected,
                )
                self.assertEqual(schedule.tail is not None, 2 in selected)

    def test_none_selector_preserves_the_original_eager_path(self):
        with mock.patch.dict(
            os.environ,
            {
                kimi_k3.COMPILED_DECODE_ENV: "1",
                kimi_k3.COMPILED_DECODE_SEGMENTS_ENV: "none",
            },
            clear=False,
        ):
            model = _make_model()

        cache = _warm_cache(model)
        logits = model(mx.array([[79]], dtype=mx.int32), cache=cache)
        mx.eval(logits, [c.state for c in cache])

        self.assertEqual(model.model._compiled_decode_segments, frozenset())
        self.assertIsNone(model.model._compiled_decode_schedule)

    def test_unselected_kda_groups_use_the_layer_eager_path(self):
        cases = {
            "1": (0, 1, 2),
            "0,2": (4, 5, 6),
        }
        for selector, expected_layers in cases.items():
            with self.subTest(selector=selector):
                with mock.patch.dict(
                    os.environ,
                    {
                        kimi_k3.COMPILED_DECODE_ENV: "1",
                        kimi_k3.COMPILED_DECODE_SEGMENTS_ENV: selector,
                    },
                    clear=False,
                ):
                    model = _make_model()
                cache = _warm_cache(model)
                original_call = kimi_k3.KimiK3DeltaAttention.__call__
                eager_layer_calls = []

                def counted_call(attention, *args, **kwargs):
                    eager_layer_calls.append(attention.layer_idx)
                    return original_call(attention, *args, **kwargs)

                with mock.patch.object(
                    kimi_k3.KimiK3DeltaAttention,
                    "__call__",
                    new=counted_call,
                ):
                    logits = model(
                        mx.array([[83]], dtype=mx.int32),
                        cache=cache,
                    )
                    mx.eval(logits, [c.state for c in cache])

                self.assertEqual(tuple(eager_layer_calls), expected_layers)

    def test_final_adjacent_mla_boundary_matches_eager(self):
        config = _config(
            num_layers=5,
            kda_layers=(1, 2, 3),
            block_size=10,
        )
        model = _make_model(config)
        base_cache = _warm_cache(model)
        eager_cache = copy.deepcopy(base_cache)
        compiled_cache = copy.deepcopy(base_cache)
        inputs = mx.array([[37]], dtype=mx.int32)

        model.model._compiled_decode_enabled = False
        eager = model(inputs, cache=eager_cache)
        mx.eval(eager, [c.state for c in eager_cache])
        model.model._compiled_decode_enabled = True
        compiled = model(inputs, cache=compiled_cache)
        mx.eval(compiled, [c.state for c in compiled_cache])

        self.assertTrue(mx.array_equal(eager, compiled).item())
        _assert_cache_equal(self, eager_cache, compiled_cache)
        schedule = model.model._compiled_decode_schedule
        self.assertEqual(schedule.mla_indices, (3, 4))
        self.assertEqual(schedule.transitions[0].kda_indices, ())

    def test_mla_calls_and_cache_updates_remain_eager(self):
        model = _make_model()
        cache = _warm_cache(model)
        mla_indices = [i for i, layer in enumerate(model.layers) if not layer.is_linear]
        starting_offsets = {i: cache[i].offset for i in mla_indices}
        original_call = kimi_k3.KimiK3MLAAttention.__call__
        calls = []

        def counted_call(attention, *args, **kwargs):
            calls.append(attention)
            return original_call(attention, *args, **kwargs)

        model.model._compiled_decode_enabled = True
        with mock.patch.object(
            kimi_k3.KimiK3MLAAttention,
            "__call__",
            new=counted_call,
        ):
            for token in (47, 53, 59):
                logits = model(
                    mx.array([[token]], dtype=mx.int32),
                    cache=cache,
                )
                mx.eval(logits, [c.state for c in cache])

        self.assertEqual(len(calls), 3 * len(mla_indices))
        for idx in mla_indices:
            self.assertEqual(cache[idx].offset, starting_offsets[idx] + 3)

    def test_tp2_full_and_mixed_compiled_collectives_match_eager(self):
        group = mx.distributed.init()
        if group.size() != 2:
            self.skipTest("requires mlx.launch with exactly two ranks")

        for selector in ("all", "0,2"):
            with self.subTest(selector=selector):
                with mock.patch.dict(
                    os.environ,
                    {
                        kimi_k3.COMPILED_DECODE_ENV: "1",
                        kimi_k3.COMPILED_DECODE_SEGMENTS_ENV: selector,
                    },
                    clear=False,
                ):
                    model = _make_model()
                model.shard(group)
                base_cache = _warm_cache(model)
                eager_cache = copy.deepcopy(base_cache)
                compiled_cache = copy.deepcopy(base_cache)
                inputs = mx.array([[61]], dtype=mx.int32)

                model.model._compiled_decode_enabled = False
                eager = model(inputs, cache=eager_cache)
                mx.eval(eager, [c.state for c in eager_cache])
                model.model._compiled_decode_enabled = True
                compiled = model(inputs, cache=compiled_cache)
                mx.eval(compiled, [c.state for c in compiled_cache])

                self.assertTrue(mx.array_equal(eager, compiled).item())
                _assert_cache_equal(self, eager_cache, compiled_cache)

    def test_kv_growth_boundary_matches_eager(self):
        with (
            mock.patch.object(kimi_k3.KVCache, "step", 4),
            mock.patch.object(kimi_k3.BatchKVCache, "step", 4),
        ):
            model = _make_model()
            base_cache = _warm_cache(model, length=3)
            for batched in (False, True):
                with self.subTest(batched=batched):
                    cache = _as_batch_cache(base_cache) if batched else base_cache
                    eager_cache = copy.deepcopy(cache)
                    compiled_cache = copy.deepcopy(cache)

                    eager_outputs = []
                    for token in (41, 43):
                        inputs = mx.array([[token]], dtype=mx.int32)
                        model.model._compiled_decode_enabled = False
                        eager_outputs.append(model(inputs, cache=eager_cache))

                    compiled_outputs = []
                    for token in (41, 43):
                        inputs = mx.array([[token]], dtype=mx.int32)
                        model.model._compiled_decode_enabled = True
                        compiled_outputs.append(model(inputs, cache=compiled_cache))

                    mx.eval(
                        eager_outputs,
                        compiled_outputs,
                        [c.state for c in eager_cache],
                        [c.state for c in compiled_cache],
                    )
                    for eager, compiled in zip(
                        eager_outputs,
                        compiled_outputs,
                        strict=True,
                    ):
                        self.assertTrue(mx.array_equal(eager, compiled).item())
                    _assert_cache_equal(self, eager_cache, compiled_cache)

                    for layer, eager_layer_cache, compiled_layer_cache in zip(
                        model.layers,
                        eager_cache,
                        compiled_cache,
                        strict=True,
                    ):
                        if layer.is_linear:
                            continue
                        if batched:
                            self.assertEqual(eager_layer_cache._idx, 5)
                            self.assertEqual(compiled_layer_cache._idx, 5)
                            expected_capacity = 7
                        else:
                            self.assertEqual(eager_layer_cache.offset, 5)
                            self.assertEqual(compiled_layer_cache.offset, 5)
                            expected_capacity = 8
                        self.assertEqual(
                            eager_layer_cache.keys.shape[2],
                            expected_capacity,
                        )
                        self.assertEqual(
                            compiled_layer_cache.keys.shape[2],
                            expected_capacity,
                        )


@unittest.skipUnless(mx.metal.is_available(), "requires Metal")
class TestKimiK3AsyncDecodeBoundaries(unittest.TestCase):
    def setUp(self):
        self._env = mock.patch.dict(
            os.environ,
            {
                kimi_k3.COMPILED_DECODE_ENV: "0",
                kimi_k3.ASYNC_DECODE_BOUNDARIES_ENV: "none",
                "MLX_LM_KIMI_K3_FUSED_EXPERTS": "0",
            },
            clear=False,
        )
        self._env.start()
        kimi_k3.fused_k3_experts_enabled.cache_clear()

    def tearDown(self):
        kimi_k3.fused_k3_experts_enabled.cache_clear()
        self._env.stop()

    def test_boundaries_are_snapshotted_and_conflicts_fail_closed(self):
        with mock.patch.dict(
            os.environ,
            {
                kimi_k3.ASYNC_DECODE_BOUNDARIES_ENV: "1,5",
                kimi_k3.ASYNC_DECODE_STATE_ENV: "hidden",
            },
            clear=False,
        ):
            model = _make_model()
        self.assertEqual(
            model.model._async_decode_boundaries,
            frozenset((1, 5)),
        )
        self.assertEqual(model.model._async_decode_state, "hidden")

        with mock.patch.dict(
            os.environ,
            {
                kimi_k3.COMPILED_DECODE_ENV: "1",
                kimi_k3.ASYNC_DECODE_BOUNDARIES_ENV: "1",
            },
            clear=False,
        ):
            with self.assertRaises(ValueError):
                _make_model()

        with mock.patch.dict(
            os.environ,
            {
                kimi_k3.ASYNC_DECODE_BOUNDARIES_ENV: "none",
                kimi_k3.ASYNC_DECODE_STATE_ENV: "hidden",
            },
            clear=False,
        ):
            with self.assertRaises(ValueError):
                _make_model()

        with mock.patch.dict(
            os.environ,
            {
                kimi_k3.ASYNC_DECODE_BOUNDARIES_ENV: "1",
                kimi_k3.ASYNC_DECODE_STATE_ENV: "invalid",
            },
            clear=False,
        ):
            with self.assertRaises(ValueError):
                _make_model()

    def test_guard_requires_populated_single_token_eager_decode(self):
        model = _make_model()
        text_model = model.model
        text_model._async_decode_boundaries = frozenset((1, 5))
        cache = _warm_cache(model)
        h = text_model.embed_tokens(mx.array([[11]], dtype=mx.int32))
        layers = text_model.layers

        self.assertTrue(
            text_model._async_decode_boundary_eligible(
                h,
                cache,
                None,
                layers,
            )
        )
        self.assertFalse(
            text_model._async_decode_boundary_eligible(
                mx.broadcast_to(h, (1, 2, h.shape[-1])),
                cache,
                None,
                layers,
            )
        )
        self.assertFalse(
            text_model._async_decode_boundary_eligible(
                h,
                model.make_cache(),
                None,
                layers,
            )
        )

        text_model.train()
        self.assertFalse(
            text_model._async_decode_boundary_eligible(
                h,
                cache,
                None,
                layers,
            )
        )

    def test_boundaries_preserve_exact_logits_and_cache(self):
        model = _make_model()
        base_cache = _warm_cache(model)
        for state in ("hidden", "residual"):
            with self.subTest(state=state):
                eager_cache = copy.deepcopy(base_cache)
                boundary_cache = copy.deepcopy(base_cache)

                for token in (17, 23, 29, 31):
                    inputs = mx.array([[token]], dtype=mx.int32)

                    model.model._async_decode_boundaries = frozenset()
                    eager = model(inputs, cache=eager_cache)
                    mx.eval(eager, [c.state for c in eager_cache])

                    model.model._async_decode_boundaries = frozenset((1, 5))
                    model.model._async_decode_state = state
                    candidate = model(inputs, cache=boundary_cache)
                    mx.eval(candidate, [c.state for c in boundary_cache])

                    self.assertTrue(mx.array_equal(eager, candidate).item())
                    _assert_cache_equal(self, eager_cache, boundary_cache)


if __name__ == "__main__":
    unittest.main()
