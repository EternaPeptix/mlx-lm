from __future__ import annotations

import unittest
from unittest import mock

import mlx.core as mx

from mlx_lm.generate import maybe_quantize_kv_cache
from mlx_lm.models.cache import ArraysCache, BatchKVCache, SpeculativeReplayState
from mlx_lm.models.gated_delta import gated_delta_kernel
from mlx_lm.models.kimi_k3 import (
    REPLAYSSM_SPECULATIVE_ENV,
    KimiK3DeltaAttention,
    KimiK3ShortConv,
    Model,
    ModelArgs,
    TextArgs,
    VocabParallelHead,
    replayssm_speculative_enabled,
)


def _tiny_k3_model_and_cache():
    args = ModelArgs.from_dict(
        {
            "model_type": "kimi_k3",
            "vocab_size": 128,
            "text_config": {
                "model_type": "kimi_linear",
                "vocab_size": 128,
                "hidden_size": 64,
                "num_hidden_layers": 4,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "intermediate_size": 96,
                "rms_norm_eps": 1e-5,
                "hidden_act": "situ",
                "activation_situ_beta": 4.0,
                "activation_situ_linear_beta": 25.0,
                "linear_attn_config": {
                    "kda_layers": [1, 2, 3],
                    "full_attn_layers": [4],
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
                "attn_res_block_size": 2,
            },
        }
    )
    model = Model(args)
    prompt_cache = model.make_cache()
    for layer_cache in prompt_cache:
        if isinstance(layer_cache, ArraysCache):
            layer_cache.cache = [
                mx.zeros((1, 2), dtype=mx.bfloat16),
                mx.zeros((1, 2, 2), dtype=mx.float32),
            ]
        else:
            keys = mx.zeros((1, 1, 1, 16), dtype=mx.bfloat16)
            values = mx.zeros((1, 1, 1, 16), dtype=mx.bfloat16)
            layer_cache.update_and_fetch(keys, values)
    mx.eval([layer_cache.state for layer_cache in prompt_cache])
    return model, prompt_cache


def _tiny_delta_attention():
    args = TextArgs(
        hidden_size=64,
        num_attention_heads=2,
        num_key_value_heads=2,
        linear_attn_config={
            "num_heads": 2,
            "head_dim": 32,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0,
            "use_full_rank_gate": True,
        },
    )
    attention = KimiK3DeltaAttention(args, layer_idx=0)
    attention.eval()
    mx.eval(attention.parameters())
    return attention


def _stage_transaction(transaction):
    width = transaction.width
    for _, layer_cache, initial_states in transaction.array_states:
        histories = []
        for initial in initial_states:
            histories.append(
                mx.stack(
                    [initial + float(position + 1) for position in range(width)]
                )
            )
        layer_cache.cache = [history[-1] for history in histories]
        layer_cache.capture_speculative(histories)
    for _, layer_cache, _, _, _ in transaction.kv_states:
        keys = mx.zeros((1, 1, width, 16), dtype=mx.bfloat16)
        values = mx.zeros((1, 1, width, 16), dtype=mx.bfloat16)
        layer_cache.update_and_fetch(keys, values)


class ArraysCacheCheckpointTest(unittest.TestCase):
    def test_partial_resolution_materializes_requested_state(self):
        cache = ArraysCache(size=2)
        cache.cache = [
            mx.array([[1.0, 2.0]], dtype=mx.float32),
            mx.array([[3.0, 4.0]], dtype=mx.float32),
        ]
        history = [
            mx.arange(6, dtype=mx.float32).reshape(3, 1, 2),
            mx.arange(6, 12, dtype=mx.float32).reshape(3, 1, 2),
        ]
        cache.begin_speculative(3)
        cache.cache = [values[-1] for values in history]
        cache.capture_speculative(history)

        materialized = cache.resolve_speculative(2)
        mx.eval(*materialized)

        self.assertEqual(cache.speculative_width, 0)
        self.assertFalse(cache.speculative_ready)
        self.assertEqual(len(materialized), 2)
        for actual, expected in zip(cache.cache, history, strict=True):
            self.assertTrue(bool(mx.all(actual == expected[1]).item()))

    def test_full_resolution_keeps_final_state(self):
        cache = ArraysCache(size=2)
        cache.cache = [
            mx.array([[1.0]], dtype=mx.float32),
            mx.array([[2.0]], dtype=mx.float32),
        ]
        history = [
            mx.array([[[3.0]], [[4.0]]], dtype=mx.float32),
            mx.array([[[5.0]], [[6.0]]], dtype=mx.float32),
        ]
        cache.begin_speculative(2)
        final = [values[-1] for values in history]
        cache.cache = final
        cache.capture_speculative(history)

        materialized = cache.resolve_speculative(2)
        self.assertEqual(len(materialized), 2)
        for actual, expected in zip(cache.cache, final, strict=True):
            self.assertTrue(bool(mx.all(actual == expected).item()))

    def test_cancel_restores_initial_state(self):
        cache = ArraysCache(size=2)
        initial = [
            mx.array([[1.0]], dtype=mx.float32),
            mx.array([[2.0]], dtype=mx.float32),
        ]
        cache.cache = initial
        history = [
            mx.array([[[3.0]], [[4.0]]], dtype=mx.float32),
            mx.array([[[5.0]], [[6.0]]], dtype=mx.float32),
        ]
        cache.begin_speculative(2)
        cache.cache = [values[-1] for values in history]
        cache.capture_speculative(history)

        cache.cancel_speculative()

        self.assertEqual(cache.speculative_width, 0)
        self.assertFalse(cache.speculative_ready)
        for actual, expected in zip(cache.cache, initial, strict=True):
            self.assertIs(actual, expected)

    def test_from_state_initializes_speculative_bookkeeping(self):
        state = [
            mx.array([[1.0]], dtype=mx.float32),
            mx.array([[2.0]], dtype=mx.float32),
        ]
        cache = ArraysCache.from_state(state, "")

        self.assertEqual(cache.speculative_width, 0)
        self.assertFalse(cache.speculative_ready)
        cache.begin_speculative(2)
        cache.cancel_speculative()
        self.assertEqual(cache.speculative_width, 0)

    def test_guards_fail_closed(self):
        cache = ArraysCache(size=2)
        with self.assertRaisesRegex(ValueError, "greater than one"):
            cache.begin_speculative(1)
        with self.assertRaisesRegex(ValueError, "populated"):
            cache.begin_speculative(2)

    def test_quantization_rejects_an_active_transaction(self):
        cache = ArraysCache(size=1)
        cache.cache = [mx.array([[1.0]], dtype=mx.float32)]
        cache.begin_speculative(2)
        try:
            with self.assertRaisesRegex(ValueError, "active speculative"):
                maybe_quantize_kv_cache([cache], 0, 64, 4)
        finally:
            cache.cancel_speculative()

    def test_raw_replay_materializes_only_the_accepted_prefix(self):
        cache = ArraysCache(size=1)
        initial = mx.array([[1.0, 2.0]], dtype=mx.float32)
        raw = mx.array([[[3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]])

        def replay(initial_state, raw_inputs, consumed):
            return initial_state + mx.sum(raw_inputs[0][:, :consumed], axis=1)

        cache.cache = [initial]
        cache.begin_speculative(3)
        final = replay(initial, (raw,), 3)
        cache.cache = [final]
        cache.capture_speculative(
            [
                SpeculativeReplayState(
                    width=3,
                    raw_inputs=(raw,),
                    history_axis=1,
                    state_shape=tuple(final.shape),
                    state_dtype=final.dtype,
                    replay=replay,
                )
            ]
        )

        materialized = cache.resolve_speculative(2)
        expected = initial + mx.sum(raw[:, :2], axis=1)
        mx.eval(materialized, expected)

        self.assertTrue(bool(mx.all(cache.cache[0] == expected).item()))
        self.assertFalse(cache.speculative_ready)
        self.assertIsNone(cache._speculative_initial_state)

    def test_raw_replay_failure_restores_initial_state(self):
        cache = ArraysCache(size=1)
        initial = mx.array([[1.0]], dtype=mx.float32)
        raw = mx.ones((1, 2, 1), dtype=mx.float32)

        def fail_replay(_initial_state, _raw_inputs, _consumed):
            raise RuntimeError("injected replay failure")

        cache.cache = [initial]
        cache.begin_speculative(2)
        cache.cache = [initial + 2]
        cache.capture_speculative(
            [
                SpeculativeReplayState(
                    width=2,
                    raw_inputs=(raw,),
                    history_axis=1,
                    state_shape=tuple(initial.shape),
                    state_dtype=initial.dtype,
                    replay=fail_replay,
                )
            ]
        )

        with self.assertRaisesRegex(RuntimeError, "injected replay failure"):
            cache.resolve_speculative(1)

        self.assertIs(cache.cache[0], initial)
        self.assertFalse(cache.speculative_ready)

    def test_raw_replay_rejects_an_incompatible_initial_state(self):
        cache = ArraysCache(size=1)
        initial = mx.ones((1, 1), dtype=mx.float32)
        raw = mx.ones((1, 2, 1), dtype=mx.float32)

        cache.cache = [initial]
        cache.begin_speculative(2)
        cache._speculative_initial_state = [mx.ones((1, 2), dtype=mx.float32)]
        cache.cache = [initial + 2]
        cache.capture_speculative(
            [
                SpeculativeReplayState(
                    width=2,
                    raw_inputs=(raw,),
                    history_axis=1,
                    state_shape=tuple(initial.shape),
                    state_dtype=initial.dtype,
                    replay=lambda state, _raw, _consumed: state,
                )
            ]
        )

        with self.assertRaisesRegex(ValueError, "initial state"):
            cache.resolve_speculative(1)

        self.assertFalse(cache.speculative_ready)

    def test_replayssm_environment_is_strict_and_default_off(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertFalse(replayssm_speculative_enabled())
        with mock.patch.dict(
            "os.environ",
            {REPLAYSSM_SPECULATIVE_ENV: "1"},
            clear=True,
        ):
            self.assertTrue(replayssm_speculative_enabled())
        with mock.patch.dict(
            "os.environ",
            {REPLAYSSM_SPECULATIVE_ENV: "true"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "must be 0 or 1"):
                replayssm_speculative_enabled()


class KimiK3CacheTransactionTest(unittest.TestCase):
    def test_begin_rejects_pipeline_parallel_models(self):
        model, prompt_cache = _tiny_k3_model_and_cache()
        model.language_model.model.pipeline_size = 2

        with self.assertRaisesRegex(ValueError, "pipeline_size == 1"):
            model.begin_speculative_cache(prompt_cache, 2)

        arrays = [entry for entry in prompt_cache if isinstance(entry, ArraysCache)]
        self.assertTrue(all(entry.speculative_width == 0 for entry in arrays))

    def test_begin_rejects_batch_kv_cache(self):
        model, prompt_cache = _tiny_k3_model_and_cache()
        kv_index = next(
            index
            for index, entry in enumerate(prompt_cache)
            if not isinstance(entry, ArraysCache)
        )
        prompt_cache[kv_index] = BatchKVCache([0])

        with self.assertRaisesRegex(ValueError, "BatchKVCache"):
            model.begin_speculative_cache(prompt_cache, 2)

        arrays = [entry for entry in prompt_cache if isinstance(entry, ArraysCache)]
        self.assertTrue(all(entry.speculative_width == 0 for entry in arrays))

    def test_begin_preflights_every_layer_before_activation(self):
        model, prompt_cache = _tiny_k3_model_and_cache()
        arrays = [entry for entry in prompt_cache if isinstance(entry, ArraysCache)]
        arrays[-1].lengths = mx.array([1])

        with self.assertRaisesRegex(ValueError, "padded"):
            model.begin_speculative_cache(prompt_cache, 2)

        self.assertTrue(all(entry.speculative_width == 0 for entry in arrays))
        self.assertTrue(all(not entry.speculative_ready for entry in arrays))

    def test_cancel_restores_kda_and_mla_state(self):
        model, prompt_cache = _tiny_k3_model_and_cache()
        original_arrays = [
            list(entry.cache)
            for entry in prompt_cache
            if isinstance(entry, ArraysCache)
        ]
        kv_cache = next(
            entry for entry in prompt_cache if not isinstance(entry, ArraysCache)
        )
        original_keys = kv_cache.keys
        original_values = kv_cache.values
        original_offset = kv_cache.offset

        transaction = model.begin_speculative_cache(prompt_cache, 8)
        _stage_transaction(transaction)
        model.cancel_speculative_cache(transaction)

        self.assertFalse(transaction.active)
        actual_arrays = [
            list(entry.cache)
            for entry in prompt_cache
            if isinstance(entry, ArraysCache)
        ]
        for actual, expected in zip(actual_arrays, original_arrays, strict=True):
            for actual_state, expected_state in zip(actual, expected, strict=True):
                self.assertIs(actual_state, expected_state)
        self.assertIs(kv_cache.keys, original_keys)
        self.assertIs(kv_cache.values, original_values)
        self.assertEqual(kv_cache.offset, original_offset)

    def test_eval_failure_rolls_back_the_entire_transaction(self):
        model, prompt_cache = _tiny_k3_model_and_cache()
        original_arrays = [
            list(entry.cache)
            for entry in prompt_cache
            if isinstance(entry, ArraysCache)
        ]
        kv_cache = next(
            entry for entry in prompt_cache if not isinstance(entry, ArraysCache)
        )
        original_offset = kv_cache.offset
        transaction = model.begin_speculative_cache(prompt_cache, 2)
        _stage_transaction(transaction)

        with mock.patch(
            "mlx_lm.models.kimi_k3.mx.eval",
            side_effect=RuntimeError("injected evaluation failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                model.resolve_speculative_cache(transaction, 1)

        self.assertFalse(transaction.active)
        actual_arrays = [
            list(entry.cache)
            for entry in prompt_cache
            if isinstance(entry, ArraysCache)
        ]
        for actual, expected in zip(actual_arrays, original_arrays, strict=True):
            for actual_state, expected_state in zip(actual, expected, strict=True):
                self.assertIs(actual_state, expected_state)
        self.assertEqual(kv_cache.offset, original_offset)

    def test_incomplete_capture_rolls_back_every_layer(self):
        model, prompt_cache = _tiny_k3_model_and_cache()
        original_arrays = [
            list(entry.cache)
            for entry in prompt_cache
            if isinstance(entry, ArraysCache)
        ]
        transaction = model.begin_speculative_cache(prompt_cache, 2)
        _stage_transaction(transaction)
        transaction.array_states[-1][1]._speculative_state_history = None

        with self.assertRaisesRegex(ValueError, "incomplete"):
            model.resolve_speculative_cache(transaction, 1)

        self.assertFalse(transaction.active)
        actual_arrays = [
            list(entry.cache)
            for entry in prompt_cache
            if isinstance(entry, ArraysCache)
        ]
        for actual, expected in zip(actual_arrays, original_arrays, strict=True):
            for actual_state, expected_state in zip(actual, expected, strict=True):
                self.assertIs(actual_state, expected_state)

    def test_repeated_width_two_and_eight_rounds_release_history(self):
        model, prompt_cache = _tiny_k3_model_and_cache()
        expected_offset = next(
            entry.offset
            for entry in prompt_cache
            if not isinstance(entry, ArraysCache)
        )

        for round_index in range(24):
            width = 2 if round_index % 2 == 0 else 8
            consumed = 1 if round_index % 3 else width
            transaction = model.begin_speculative_cache(prompt_cache, width)
            _stage_transaction(transaction)
            model.resolve_speculative_cache(transaction, consumed)
            expected_offset += consumed

            self.assertFalse(transaction.active)
            self.assertEqual(transaction.array_states, [])
            self.assertEqual(transaction.kv_states, [])
            for entry in prompt_cache:
                if isinstance(entry, ArraysCache):
                    self.assertEqual(entry.speculative_width, 0)
                    self.assertFalse(entry.speculative_ready)
                    self.assertIsNone(entry._speculative_initial_state)
                else:
                    self.assertEqual(entry.offset, expected_offset)


@unittest.skipUnless(mx.metal.is_available(), "requires Metal")
class MetalCheckpointKernelTest(unittest.TestCase):
    def test_target_aux_hidden_taps_preserve_logits(self):
        mx.set_default_device(mx.gpu)
        mx.random.seed(71)
        model, _ = _tiny_k3_model_and_cache()
        model.eval()
        inputs = mx.array([[1, 2, 3]])

        expected_logits = model(inputs)
        result = model.forward_with_aux_hidden_states(
            inputs,
            cache=None,
            layer_ids=(0, 2),
        )
        mx.eval(expected_logits, result.logits, result.aux_hidden_states)

        self.assertTrue(bool(mx.all(result.logits == expected_logits).item()))
        self.assertEqual(len(result.aux_hidden_states), 2)
        self.assertEqual(result.aux_hidden_states[0].shape, (1, 3, 64))
        self.assertEqual(result.aux_hidden_states[1].shape, (1, 3, 64))

    def test_compact_target_verifier_matches_masked_full_logits(self):
        mx.set_default_device(mx.gpu)
        mx.random.seed(72)
        model, _ = _tiny_k3_model_and_cache()
        model.eval()
        inputs = mx.array([[1, 2, 3]])
        assert model.language_model.lm_head is not None
        model.language_model.lm_head.weight = mx.zeros_like(
            model.language_model.lm_head.weight
        )

        full = model.forward_with_aux_hidden_states(
            inputs,
            cache=None,
            layer_ids=(0, 2),
        )
        masked_logits = full.logits
        masked_logits[..., 0] = float("-inf")
        expected = mx.argmax(masked_logits, axis=-1).astype(mx.uint32)

        group = mx.distributed.init()
        self.assertEqual(group.size(), 1)
        model.language_model.lm_head = VocabParallelHead(
            model.language_model.lm_head,
            group,
        )
        compact = model.forward_with_aux_hidden_states_greedy(
            inputs,
            cache=None,
            layer_ids=(0, 2),
            banned_token_ids=(0,),
        )
        mx.eval(expected, compact.tokens, compact.aux_hidden_states)

        self.assertTrue(mx.array_equal(expected, compact.tokens))
        self.assertEqual(compact.tokens.tolist(), [[1, 1, 1]])
        self.assertEqual(len(compact.aux_hidden_states), 2)
        for expected_hidden, actual_hidden in zip(
            full.aux_hidden_states,
            compact.aux_hidden_states,
            strict=True,
        ):
            self.assertTrue(mx.array_equal(expected_hidden, actual_hidden))

    def test_k3_replayssm_width_three_matches_full_history_exactly(self):
        mx.set_default_device(mx.gpu)
        mx.random.seed(73)
        attention = _tiny_delta_attention()
        x = mx.random.normal((1, 3, 64), dtype=mx.float32)
        initial_conv = mx.random.normal((1, 3, 192), dtype=mx.float32)
        initial_ssm = mx.random.normal((1, 2, 32, 32), dtype=mx.float32)
        mx.eval(x, initial_conv, initial_ssm)

        def stage(replay_enabled):
            cache = ArraysCache(size=2)
            cache.cache = [initial_conv, initial_ssm]
            cache.begin_speculative(3)
            with mock.patch.dict(
                "os.environ",
                {REPLAYSSM_SPECULATIVE_ENV: "1" if replay_enabled else "0"},
                clear=False,
            ):
                output = attention(x, cache=cache)
            sources = list(cache._speculative_state_history)
            states = [cache.prepare_speculative(consumed) for consumed in (1, 2, 3)]
            source_arrays = []
            for source in sources:
                if isinstance(source, SpeculativeReplayState):
                    source_arrays.extend(source.raw_inputs)
                else:
                    source_arrays.append(source)
            mx.eval(output, source_arrays, states)
            return cache, output, sources, states

        baseline_cache, baseline_output, baseline_sources, baseline_states = stage(False)
        replay_cache, replay_output, replay_sources, replay_states = stage(True)
        try:
            self.assertIsInstance(replay_sources[1], SpeculativeReplayState)
            self.assertNotIsInstance(baseline_sources[1], SpeculativeReplayState)
            self.assertEqual(len(replay_sources[1].raw_inputs), 4)
            self.assertLess(replay_sources[1].nbytes, baseline_sources[1].nbytes)
            self.assertTrue(bool(mx.all(replay_output == baseline_output).item()))
            for expected, actual in zip(
                baseline_states,
                replay_states,
                strict=True,
            ):
                self.assertTrue(bool(mx.all(actual[0] == expected[0]).item()))
                self.assertTrue(bool(mx.all(actual[1] == expected[1]).item()))
        finally:
            baseline_cache.cancel_speculative()
            replay_cache.cancel_speculative()

    def test_gated_delta_history_matches_sequential_steps(self):
        mx.random.seed(31)
        q = mx.random.normal((1, 3, 2, 32), dtype=mx.bfloat16)
        k = mx.random.normal((1, 3, 2, 32), dtype=mx.bfloat16)
        v = mx.random.normal((1, 3, 2, 32), dtype=mx.bfloat16)
        g = mx.sigmoid(mx.random.normal((1, 3, 2), dtype=mx.float32))
        beta = mx.sigmoid(mx.random.normal((1, 3, 2), dtype=mx.float32))
        state = mx.random.normal((1, 2, 32, 32), dtype=mx.float32)

        wide_y, wide_state, history = gated_delta_kernel(
            q,
            k,
            v,
            g,
            beta,
            state,
            return_state_history=True,
        )
        sequential_y = []
        sequential_history = []
        sequential_state = state
        for position in range(q.shape[1]):
            y, sequential_state = gated_delta_kernel(
                q[:, position : position + 1],
                k[:, position : position + 1],
                v[:, position : position + 1],
                g[:, position : position + 1],
                beta[:, position : position + 1],
                sequential_state,
            )
            sequential_y.append(y)
            sequential_history.append(sequential_state)
        sequential_y = mx.concatenate(sequential_y, axis=1)
        sequential_history = mx.stack(sequential_history, axis=2)
        mx.eval(
            wide_y,
            wide_state,
            history,
            sequential_y,
            sequential_state,
            sequential_history,
        )

        self.assertTrue(bool(mx.all(wide_y == sequential_y).item()))
        self.assertTrue(bool(mx.all(wide_state == sequential_state).item()))
        self.assertTrue(bool(mx.all(history == sequential_history).item()))
        self.assertTrue(bool(mx.all(history[:, :, -1] == wide_state).item()))

    def test_k3_vector_gate_history_width_two_and_eight(self):
        for width in (2, 8):
            with self.subTest(width=width):
                mx.random.seed(41 + width)
                heads = 48
                dimension = 128
                q = mx.random.normal(
                    (1, width, heads, dimension), dtype=mx.bfloat16
                )
                k = mx.random.normal(
                    (1, width, heads, dimension), dtype=mx.bfloat16
                )
                v = mx.random.normal(
                    (1, width, heads, dimension), dtype=mx.bfloat16
                )
                g = mx.sigmoid(
                    mx.random.normal(
                        (1, width, heads, dimension), dtype=mx.float32
                    )
                )
                beta = mx.sigmoid(
                    mx.random.normal((1, width, heads), dtype=mx.float32)
                )
                state = mx.random.normal(
                    (1, heads, dimension, dimension), dtype=mx.float32
                )

                wide_y, wide_state, history = gated_delta_kernel(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    state,
                    return_state_history=True,
                )
                sequential_y = []
                sequential_history = []
                sequential_state = state
                for position in range(width):
                    y, sequential_state = gated_delta_kernel(
                        q[:, position : position + 1],
                        k[:, position : position + 1],
                        v[:, position : position + 1],
                        g[:, position : position + 1],
                        beta[:, position : position + 1],
                        sequential_state,
                    )
                    sequential_y.append(y)
                    sequential_history.append(sequential_state)
                sequential_y = mx.concatenate(sequential_y, axis=1)
                sequential_history = mx.stack(sequential_history, axis=2)
                mx.eval(
                    wide_y,
                    wide_state,
                    history,
                    sequential_y,
                    sequential_state,
                    sequential_history,
                )

                self.assertTrue(bool(mx.all(wide_y == sequential_y).item()))
                self.assertTrue(bool(mx.all(wide_state == sequential_state).item()))
                self.assertTrue(bool(mx.all(history == sequential_history).item()))

    def test_short_conv_history_matches_sequential_steps(self):
        for width in (2, 8):
            with self.subTest(width=width):
                mx.random.seed(37 + width)
                channels = 18432
                conv = KimiK3ShortConv(channels=channels, kernel_size=4)
                conv.eval()
                conv.conv.weight = conv.conv.weight.astype(mx.bfloat16)
                x = mx.random.normal((1, width, channels), dtype=mx.bfloat16)
                state = mx.random.normal((1, 3, channels), dtype=mx.bfloat16)

                wide_y, wide_state, history = conv(
                    x,
                    state,
                    return_state_history=True,
                )
                sequential_y = []
                sequential_history = []
                sequential_state = state
                for position in range(x.shape[1]):
                    y, sequential_state = conv(
                        x[:, position : position + 1],
                        sequential_state,
                    )
                    sequential_y.append(y)
                    sequential_history.append(sequential_state)
                sequential_y = mx.concatenate(sequential_y, axis=1)
                sequential_history = mx.stack(sequential_history, axis=1)
                mx.eval(
                    wide_y,
                    wide_state,
                    history,
                    sequential_y,
                    sequential_state,
                    sequential_history,
                )

                self.assertTrue(bool(mx.all(wide_y == sequential_y).item()))
                self.assertTrue(bool(mx.all(wide_state == sequential_state).item()))
                self.assertTrue(bool(mx.all(history == sequential_history).item()))
                self.assertTrue(bool(mx.all(history[:, -1] == wide_state).item()))


if __name__ == "__main__":
    unittest.main()
