from __future__ import annotations

import json
import multiprocessing as mp
import os
import socket
import tempfile
import unittest
from queue import Empty
from typing import Any
from unittest import mock

import mlx.core as mx

from mlx_lm.generate import maybe_quantize_kv_cache
from mlx_lm.models.cache import ArraysCache, BatchKVCache, SpeculativeReplayState
from mlx_lm.models.gated_delta import gated_delta_kernel
from mlx_lm.models.kimi_k3 import (
    REPLAYSSM_SPECULATIVE_ENV,
    KimiK3AuxPrefill,
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


def _small_k3_model(
    *,
    num_hidden_layers: int = 4,
    kda_layers: tuple[int, ...] = (1, 2, 3),
    full_attn_layers: tuple[int, ...] = (4,),
    attn_res_block_size: int = 2,
) -> Model:
    args = ModelArgs.from_dict(
        {
            "model_type": "kimi_k3",
            "vocab_size": 32,
            "text_config": {
                "model_type": "kimi_linear",
                "vocab_size": 32,
                "hidden_size": 16,
                "num_hidden_layers": num_hidden_layers,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "intermediate_size": 24,
                "rms_norm_eps": 1e-5,
                "hidden_act": "situ",
                "activation_situ_beta": 4.0,
                "activation_situ_linear_beta": 25.0,
                "linear_attn_config": {
                    "kda_layers": list(kda_layers),
                    "full_attn_layers": list(full_attn_layers),
                    "num_heads": 2,
                    "head_dim": 8,
                    "short_conv_kernel_size": 4,
                    "gate_lower_bound": -5.0,
                    "use_full_rank_gate": True,
                },
                "num_experts": 2,
                "moe_intermediate_size": 8,
                "q_lora_rank": 8,
                "kv_lora_rank": 8,
                "qk_nope_head_dim": 4,
                "qk_rope_head_dim": 4,
                "v_head_dim": 4,
                "mla_use_nope": True,
                "mla_use_output_gate": True,
                "num_experts_per_token": 1,
                "num_shared_experts": 1,
                "first_k_dense_replace": 1,
                "routed_expert_hidden_size": 8,
                "latent_moe_use_norm": True,
                "attn_res_block_size": attn_res_block_size,
                "tie_word_embeddings": False,
            },
        }
    )
    model = Model(args)
    model.eval()
    return model


def _cache_roots(cache: list[Any]) -> tuple[mx.array, ...]:
    return tuple(value for layer_cache in cache for value in layer_cache.state)


def _assert_cache_equal(
    test: unittest.TestCase,
    expected_cache: list[Any],
    actual_cache: list[Any],
    expected_offset: int | None,
) -> None:
    test.assertEqual(len(expected_cache), len(actual_cache))
    observed_offsets = set()
    for layer_index, (expected_layer, actual_layer) in enumerate(
        zip(expected_cache, actual_cache, strict=True)
    ):
        test.assertIs(type(expected_layer), type(actual_layer))
        expected_state = tuple(expected_layer.state)
        actual_state = tuple(actual_layer.state)
        test.assertEqual(len(expected_state), len(actual_state))
        for state_index, (expected_value, actual_value) in enumerate(
            zip(expected_state, actual_state, strict=True)
        ):
            test.assertEqual(expected_value.shape, actual_value.shape)
            test.assertEqual(expected_value.dtype, actual_value.dtype)
            test.assertTrue(
                mx.array_equal(expected_value, actual_value),
                f"cache layer {layer_index} state {state_index} disagreed",
            )
        expected_layer_offset = getattr(expected_layer, "offset", None)
        actual_layer_offset = getattr(actual_layer, "offset", None)
        test.assertEqual(expected_layer_offset, actual_layer_offset)
        if expected_layer_offset is not None:
            observed_offsets.add(expected_layer_offset)
        test.assertEqual(
            getattr(expected_layer, "speculative_width", None),
            getattr(actual_layer, "speculative_width", None),
        )
        test.assertEqual(
            getattr(expected_layer, "speculative_ready", None),
            getattr(actual_layer, "speculative_ready", None),
        )
        test.assertIs(
            getattr(expected_layer, "_speculative_initial_state", None),
            getattr(actual_layer, "_speculative_initial_state", None),
        )
        test.assertIs(
            getattr(expected_layer, "_speculative_state_history", None),
            getattr(actual_layer, "_speculative_state_history", None),
        )
    if expected_offset is not None:
        test.assertEqual(observed_offsets, {expected_offset})


class _RecordingHead:
    def __init__(self, delegate: Any):
        self.delegate = delegate
        self.hidden: mx.array | None = None

    def __call__(self, hidden: mx.array) -> mx.array:
        self.hidden = hidden
        return self.delegate(hidden)


def _unused_local_ports(count: int) -> list[int]:
    listeners = []
    try:
        for _ in range(count):
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listener.bind(("127.0.0.1", 0))
            listeners.append(listener)
        return [listener.getsockname()[1] for listener in listeners]
    finally:
        for listener in listeners:
            listener.close()


def _run_aux_prefill_tp2_rank(
    rank: int,
    hostfile: str,
    result_queue: Any,
) -> None:
    os.environ["MLX_HOSTFILE"] = hostfile
    os.environ["MLX_RANK"] = str(rank)
    try:
        mx.set_default_device(mx.gpu)
        group = mx.distributed.init(backend="ring", strict=True)
        mx.random.seed(79)
        model = _small_k3_model()
        model.set_dtype(mx.bfloat16)
        model.shard(group)
        model.shard_vocab_head(group)
        full_cache = model.make_cache()
        aux_cache = model.make_cache()
        inputs = mx.array([[1, 2, 3]], dtype=mx.uint32)
        layer_ids = (0, 2)

        full = model.forward_with_aux_hidden_states(
            inputs,
            cache=full_cache,
            layer_ids=layer_ids,
        )
        mx.eval(full.logits, *full.aux_hidden_states, *_cache_roots(full_cache))

        with (
            mock.patch.object(
                VocabParallelHead,
                "__call__",
                side_effect=AssertionError("TP2 aux prefill must not gather logits"),
            ),
            mock.patch.object(
                VocabParallelHead,
                "greedy_token",
                side_effect=AssertionError("TP2 aux prefill must not sample"),
            ),
        ):
            aux = model.forward_aux_hidden_states_for_cache(
                inputs,
                cache=aux_cache,
                layer_ids=layer_ids,
            )
            mx.eval(
                aux.final_hidden_state,
                *aux.aux_hidden_states,
                *_cache_roots(aux_cache),
            )

        aux_logits = model.language_model.lm_head(aux.final_hidden_state)
        mx.eval(aux_logits)
        taps_equal = all(
            bool(mx.array_equal(expected, actual).item())
            for expected, actual in zip(
                full.aux_hidden_states,
                aux.aux_hidden_states,
                strict=True,
            )
        )
        cache_equal = all(
            type(expected_layer) is type(actual_layer)
            and getattr(expected_layer, "offset", None)
            == getattr(actual_layer, "offset", None)
            and all(
                bool(mx.array_equal(expected, actual).item())
                for expected, actual in zip(
                    expected_layer.state,
                    actual_layer.state,
                    strict=True,
                )
            )
            for expected_layer, actual_layer in zip(
                full_cache,
                aux_cache,
                strict=True,
            )
        )

        full_next = model(mx.array([[4]], dtype=mx.uint32), cache=full_cache)
        aux_next = model(mx.array([[4]], dtype=mx.uint32), cache=aux_cache)
        mx.eval(
            full_next,
            aux_next,
            *_cache_roots(full_cache),
            *_cache_roots(aux_cache),
        )
        result_queue.put(
            (
                rank,
                bool(mx.array_equal(full.logits, aux_logits).item())
                and taps_equal
                and cache_equal
                and bool(mx.array_equal(full_next, aux_next).item()),
                "ok",
            )
        )
    except Exception as error:
        result_queue.put((rank, False, repr(error)))


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
    def test_aux_only_prefill_skips_heads_and_returns_requested_taps(self):
        mx.set_default_device(mx.gpu)
        mx.random.seed(74)
        model, _ = _tiny_k3_model_and_cache()
        model.eval()
        inputs = mx.array([[1, 2, 3]])
        layer_ids = (0, 2)

        expected = model.model(
            inputs,
            None,
            aux_hidden_state_layer_ids=layer_ids,
        )
        self.assertIsInstance(expected, tuple)
        expected_final, expected_taps = expected
        mx.eval(expected_final, *expected_taps)

        class ForbiddenHead:
            def __init__(self):
                self.calls = 0

            def __call__(self, _hidden):
                self.calls += 1
                raise AssertionError("ordinary LM head must not run during aux prefill")

        ordinary_head = model.language_model.lm_head
        forbidden_head = ForbiddenHead()
        model.language_model.lm_head = forbidden_head
        actual = model.forward_aux_hidden_states_for_cache(
            inputs,
            cache=None,
            layer_ids=layer_ids,
        )
        mx.eval(actual.final_hidden_state, *actual.aux_hidden_states)
        self.assertIsInstance(actual, KimiK3AuxPrefill)
        self.assertEqual(forbidden_head.calls, 0)
        self.assertEqual(len(actual.aux_hidden_states), len(layer_ids))
        self.assertTrue(mx.array_equal(expected_final, actual.final_hidden_state))
        for expected_hidden, actual_hidden in zip(
            expected_taps,
            actual.aux_hidden_states,
            strict=True,
        ):
            self.assertTrue(mx.array_equal(expected_hidden, actual_hidden))

        assert ordinary_head is not None
        group = mx.distributed.init()
        self.assertEqual(group.size(), 1)
        model.language_model.lm_head = VocabParallelHead(ordinary_head, group)
        with (
            mock.patch.object(
                VocabParallelHead,
                "__call__",
                side_effect=AssertionError(
                    "vocabulary-parallel full head must not run during aux prefill"
                ),
            ),
            mock.patch.object(
                VocabParallelHead,
                "greedy_token",
                side_effect=AssertionError(
                    "vocabulary-parallel greedy head must not run during aux prefill"
                ),
            ),
        ):
            vocab_parallel = model.forward_aux_hidden_states_for_cache(
                inputs,
                cache=None,
                layer_ids=layer_ids,
            )
            mx.eval(
                vocab_parallel.final_hidden_state,
                *vocab_parallel.aux_hidden_states,
            )
        self.assertTrue(
            mx.array_equal(expected_final, vocab_parallel.final_hidden_state)
        )
        for expected_hidden, actual_hidden in zip(
            expected_taps,
            vocab_parallel.aux_hidden_states,
            strict=True,
        ):
            self.assertTrue(mx.array_equal(expected_hidden, actual_hidden))

    def test_aux_only_prefill_matches_full_path_cache_and_following_decode(self):
        mx.set_default_device(mx.gpu)
        mx.random.seed(75)
        model, _ = _tiny_k3_model_and_cache()
        model.eval()
        full_cache = model.make_cache()
        aux_cache = model.make_cache()
        layer_ids = (0, 2)
        ordinary_head = model.language_model.lm_head
        self.assertIsNotNone(ordinary_head)

        expected_offset = 0
        for chunk in (mx.array([[1, 2]]), mx.array([[3, 4, 5]])):
            recorder = _RecordingHead(ordinary_head)
            model.language_model.lm_head = recorder
            try:
                full = model.forward_with_aux_hidden_states(
                    chunk,
                    cache=full_cache,
                    layer_ids=layer_ids,
                )
            finally:
                model.language_model.lm_head = ordinary_head
            self.assertIsNotNone(recorder.hidden)
            aux = model.forward_aux_hidden_states_for_cache(
                chunk,
                cache=aux_cache,
                layer_ids=layer_ids,
            )
            full_cache_roots = _cache_roots(full_cache)
            aux_cache_roots = _cache_roots(aux_cache)
            mx.eval(
                full.logits,
                *full.aux_hidden_states,
                *full_cache_roots,
            )
            mx.eval(
                aux.final_hidden_state,
                *aux.aux_hidden_states,
                *aux_cache_roots,
            )

            expected_offset += chunk.shape[1]
            self.assertTrue(mx.array_equal(recorder.hidden, aux.final_hidden_state))
            self.assertEqual(len(full.aux_hidden_states), len(aux.aux_hidden_states))
            for expected_hidden, actual_hidden in zip(
                full.aux_hidden_states,
                aux.aux_hidden_states,
                strict=True,
            ):
                self.assertTrue(mx.array_equal(expected_hidden, actual_hidden))
            _assert_cache_equal(self, full_cache, aux_cache, expected_offset)

        full_next = model(mx.array([[6]]), cache=full_cache)
        aux_next = model(mx.array([[6]]), cache=aux_cache)
        mx.eval(
            full_next,
            aux_next,
            *_cache_roots(full_cache),
            *_cache_roots(aux_cache),
        )
        self.assertTrue(mx.array_equal(full_next, aux_next))
        self.assertTrue(
            mx.array_equal(
                mx.argmax(full_next, axis=-1),
                mx.argmax(aux_next, axis=-1),
            )
        )
        _assert_cache_equal(self, full_cache, aux_cache, expected_offset + 1)

    def test_aux_only_prefill_width_matrix_is_bit_exact(self):
        mx.set_default_device(mx.gpu)
        mx.random.seed(77)
        model = _small_k3_model()
        model.set_dtype(mx.bfloat16)
        mx.eval(model.parameters())
        ordinary_head = model.language_model.lm_head
        self.assertIsNotNone(ordinary_head)
        layer_ids = (0, 2)

        for width in (1, 2, 256, 2048, 4096):
            with self.subTest(width=width):
                full_cache = model.make_cache()
                aux_cache = model.make_cache()
                inputs = ((mx.arange(width) % 31) + 1).astype(mx.uint32)[None]
                recorder = _RecordingHead(ordinary_head)
                model.language_model.lm_head = recorder
                try:
                    full = model.forward_with_aux_hidden_states(
                        inputs,
                        cache=full_cache,
                        layer_ids=layer_ids,
                    )
                finally:
                    model.language_model.lm_head = ordinary_head
                self.assertIsNotNone(recorder.hidden)
                aux = model.forward_aux_hidden_states_for_cache(
                    inputs,
                    cache=aux_cache,
                    layer_ids=layer_ids,
                )
                mx.eval(
                    full.logits,
                    *full.aux_hidden_states,
                    *_cache_roots(full_cache),
                )
                mx.eval(
                    aux.final_hidden_state,
                    *aux.aux_hidden_states,
                    *_cache_roots(aux_cache),
                )

                self.assertTrue(mx.array_equal(recorder.hidden, aux.final_hidden_state))
                for expected_hidden, actual_hidden in zip(
                    full.aux_hidden_states,
                    aux.aux_hidden_states,
                    strict=True,
                ):
                    self.assertTrue(mx.array_equal(expected_hidden, actual_hidden))
                _assert_cache_equal(self, full_cache, aux_cache, width)

                next_token = mx.array([[7]], dtype=mx.uint32)
                full_next = model(next_token, cache=full_cache)
                aux_next = model(next_token, cache=aux_cache)
                mx.eval(
                    full_next,
                    aux_next,
                    *_cache_roots(full_cache),
                    *_cache_roots(aux_cache),
                )
                self.assertTrue(mx.array_equal(full_next, aux_next))
                self.assertTrue(
                    mx.array_equal(
                        mx.argmax(full_next, axis=-1),
                        mx.argmax(aux_next, axis=-1),
                    )
                )
                _assert_cache_equal(self, full_cache, aux_cache, width + 1)
                mx.clear_cache()

    def test_aux_only_prefill_multichunk_crosses_kv_growth_boundary(self):
        mx.set_default_device(mx.gpu)
        mx.random.seed(78)
        model = _small_k3_model()
        model.set_dtype(mx.bfloat16)
        mx.eval(model.parameters())
        ordinary_head = model.language_model.lm_head
        self.assertIsNotNone(ordinary_head)
        full_cache = model.make_cache()
        aux_cache = model.make_cache()
        layer_ids = (0, 2)
        expected_offset = 0

        for width, expected_capacity in ((255, 256), (2, 511), (3, 511)):
            start = expected_offset
            chunk = ((mx.arange(width) + start) % 31 + 1).astype(mx.uint32)[None]
            recorder = _RecordingHead(ordinary_head)
            model.language_model.lm_head = recorder
            try:
                full = model.forward_with_aux_hidden_states(
                    chunk,
                    cache=full_cache,
                    layer_ids=layer_ids,
                )
            finally:
                model.language_model.lm_head = ordinary_head
            self.assertIsNotNone(recorder.hidden)
            aux = model.forward_aux_hidden_states_for_cache(
                chunk,
                cache=aux_cache,
                layer_ids=layer_ids,
            )
            mx.eval(
                full.logits,
                *full.aux_hidden_states,
                *_cache_roots(full_cache),
            )
            mx.eval(
                aux.final_hidden_state,
                *aux.aux_hidden_states,
                *_cache_roots(aux_cache),
            )

            expected_offset += width
            self.assertTrue(mx.array_equal(recorder.hidden, aux.final_hidden_state))
            for expected_hidden, actual_hidden in zip(
                full.aux_hidden_states,
                aux.aux_hidden_states,
                strict=True,
            ):
                self.assertTrue(mx.array_equal(expected_hidden, actual_hidden))
            _assert_cache_equal(self, full_cache, aux_cache, expected_offset)
            for cache in (full_cache, aux_cache):
                kv_entries = [entry for entry in cache if hasattr(entry, "keys")]
                self.assertEqual(len(kv_entries), 1)
                self.assertEqual(kv_entries[0].keys.shape[2], expected_capacity)

    def test_aux_only_prefill_production_layer_and_tap_topology(self):
        mx.set_default_device(mx.gpu)
        mx.random.seed(80)
        full_attn_layers = tuple(range(4, 93, 4)) + (93,)
        kda_layers = tuple(
            layer for layer in range(1, 94) if layer not in full_attn_layers
        )
        target_layer_ids = (7, 23, 51, 67, 83)
        model = _small_k3_model(
            num_hidden_layers=93,
            kda_layers=kda_layers,
            full_attn_layers=full_attn_layers,
            attn_res_block_size=12,
        )
        model.set_dtype(mx.bfloat16)
        mx.eval(model.parameters())

        self.assertEqual(len(model.layers), 93)
        self.assertEqual(
            tuple(
                index + 1 for index, layer in enumerate(model.layers) if layer.is_linear
            ),
            kda_layers,
        )
        self.assertEqual(
            tuple(
                index + 1
                for index, layer in enumerate(model.layers)
                if not layer.is_linear
            ),
            full_attn_layers,
        )
        self.assertTrue(
            all(not model.layers[layer_id].is_linear for layer_id in target_layer_ids)
        )

        full_cache = model.make_cache()
        aux_cache = model.make_cache()
        inputs = mx.array([[1, 2]], dtype=mx.uint32)
        ordinary_head = model.language_model.lm_head
        self.assertIsNotNone(ordinary_head)
        recorder = _RecordingHead(ordinary_head)
        model.language_model.lm_head = recorder
        try:
            full = model.forward_with_aux_hidden_states(
                inputs,
                cache=full_cache,
                layer_ids=target_layer_ids,
            )
        finally:
            model.language_model.lm_head = ordinary_head
        self.assertIsNotNone(recorder.hidden)
        aux = model.forward_aux_hidden_states_for_cache(
            inputs,
            cache=aux_cache,
            layer_ids=target_layer_ids,
        )
        mx.eval(
            full.logits,
            *full.aux_hidden_states,
            *_cache_roots(full_cache),
        )
        mx.eval(
            aux.final_hidden_state,
            *aux.aux_hidden_states,
            *_cache_roots(aux_cache),
        )

        self.assertEqual(len(aux.aux_hidden_states), 5)
        self.assertEqual(aux.final_hidden_state.shape, (1, 2, 16))
        self.assertTrue(mx.array_equal(recorder.hidden, aux.final_hidden_state))
        for expected_hidden, actual_hidden in zip(
            full.aux_hidden_states,
            aux.aux_hidden_states,
            strict=True,
        ):
            self.assertEqual(actual_hidden.shape, (1, 2, 16))
            self.assertTrue(mx.array_equal(expected_hidden, actual_hidden))
        _assert_cache_equal(self, full_cache, aux_cache, 2)

        full_next = model(mx.array([[3]], dtype=mx.uint32), cache=full_cache)
        aux_next = model(mx.array([[3]], dtype=mx.uint32), cache=aux_cache)
        mx.eval(
            full_next,
            aux_next,
            *_cache_roots(full_cache),
            *_cache_roots(aux_cache),
        )
        self.assertTrue(mx.array_equal(full_next, aux_next))
        _assert_cache_equal(self, full_cache, aux_cache, 3)

    def test_aux_only_prefill_real_tp2_ring(self):
        ports = _unused_local_ports(2)
        hosts = [f"127.0.0.1:{port}" for port in ports]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as file:
            json.dump(hosts, file)
            hostfile = file.name

        context = mp.get_context("spawn")
        result_queue: Any = context.Queue()
        processes = [
            context.Process(
                target=_run_aux_prefill_tp2_rank,
                args=(rank, hostfile, result_queue),
            )
            for rank in range(2)
        ]
        try:
            for process in processes:
                process.start()
            for process in processes:
                process.join(timeout=60)

            timed_out = [process.pid for process in processes if process.is_alive()]
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
            self.assertEqual(timed_out, [], f"TP2 rank timed out: {timed_out}")
            self.assertEqual(
                [process.exitcode for process in processes],
                [0, 0],
            )

            results: dict[int, tuple[bool, str]] = {}
            try:
                for _ in range(2):
                    rank, passed, detail = result_queue.get(timeout=2)
                    results[rank] = (passed, detail)
            except Empty:
                pass
            self.assertEqual(len(results), 2, f"missing TP2 result: {results}")
            for rank in range(2):
                passed, detail = results[rank]
                self.assertTrue(passed, f"TP2 rank {rank} failed: {detail}")
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
            os.unlink(hostfile)

    def test_aux_only_prefill_matches_aux_capture_validation_contract(self):
        mx.set_default_device(mx.gpu)
        model, _ = _tiny_k3_model_and_cache()
        model.eval()
        inputs = mx.array([[1]])

        for layer_ids in (
            None,
            (),
            (0, 0),
            (2, 0),
            (-1,),
            (4,),
            (0.0,),
        ):
            with self.subTest(layer_ids=layer_ids):
                with self.assertRaises(Exception) as expected:
                    model.forward_with_aux_hidden_states(
                        inputs,
                        cache=None,
                        layer_ids=layer_ids,
                    )
                with self.assertRaises(type(expected.exception)) as actual:
                    model.forward_aux_hidden_states_for_cache(
                        inputs,
                        cache=None,
                        layer_ids=layer_ids,
                    )
                self.assertEqual(str(actual.exception), str(expected.exception))

        model.model.pipeline_size = 2
        with self.assertRaises(ValueError) as expected:
            model.forward_with_aux_hidden_states(
                inputs,
                cache=None,
                layer_ids=(0,),
            )
        with self.assertRaises(type(expected.exception)) as actual:
            model.forward_aux_hidden_states_for_cache(
                inputs,
                cache=None,
                layer_ids=(0,),
            )
        self.assertEqual(str(actual.exception), str(expected.exception))

    def test_aux_only_prefill_uses_existing_eager_capture_path(self):
        mx.set_default_device(mx.gpu)
        mx.random.seed(76)
        model, _ = _tiny_k3_model_and_cache()
        model.eval()
        inputs = mx.array([[1]])

        with mock.patch.object(
            model.model,
            "_compiled_decode_eligible",
            side_effect=AssertionError(
                "auxiliary capture must bypass compiled decode eligibility"
            ),
        ):
            expected = model.forward_with_aux_hidden_states(
                inputs,
                cache=None,
                layer_ids=(0,),
            )
            actual = model.forward_aux_hidden_states_for_cache(
                inputs,
                cache=None,
                layer_ids=(0,),
            )
            mx.eval(
                expected.aux_hidden_states,
                actual.final_hidden_state,
                actual.aux_hidden_states,
            )
        self.assertTrue(
            mx.array_equal(
                expected.aux_hidden_states[0],
                actual.aux_hidden_states[0],
            )
        )

        def aux_roots(value):
            result = model.forward_aux_hidden_states_for_cache(
                value,
                cache=None,
                layer_ids=(0,),
            )
            return result.final_hidden_state, result.aux_hidden_states

        compiled = mx.compile(aux_roots)
        compiled_final_hidden, compiled_hidden_states = compiled(inputs)
        mx.eval(compiled_final_hidden, compiled_hidden_states)
        self.assertEqual(len(compiled_hidden_states), 1)
        self.assertTrue(
            mx.allclose(
                actual.final_hidden_state,
                compiled_final_hidden,
                rtol=1e-5,
                atol=1e-6,
            )
        )
        self.assertTrue(
            mx.allclose(
                actual.aux_hidden_states[0],
                compiled_hidden_states[0],
                rtol=1e-5,
                atol=1e-6,
            )
        )

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
