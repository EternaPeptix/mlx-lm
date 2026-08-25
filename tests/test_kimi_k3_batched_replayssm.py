from __future__ import annotations

import gc
import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_lm.models.cache import ArraysCache, SpeculativeReplayState
from mlx_lm.models.kimi_k3 import (
    BATCHED_REPLAYSSM_COMMIT_ENV,
    BATCHED_REPLAYSSM_EXPECTED_LAYERS_ENV,
    FULL_ACCEPT_IDENTITY_COMMIT_ENV,
    REPLAYSSM_SPECULATIVE_ENV,
    KimiK3DeltaAttention,
    Model,
    ModelArgs,
    TextArgs,
    _increment_batched_replayssm_counter,
    _prepare_batched_replayssm_commit,
    batched_replayssm_commit_counters,
    batched_replayssm_commit_telemetry,
    full_accept_identity_commit_enabled,
    reset_batched_replayssm_commit_counters,
)


def _tiny_delta_attention(layer_index: int = 0) -> KimiK3DeltaAttention:
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
    attention = KimiK3DeltaAttention(args, layer_idx=layer_index)
    attention.eval()
    mx.eval(attention.parameters())
    return attention


def _synthetic_array_states(layer_count: int, width: int):
    states = []
    for layer_index in range(layer_count):
        attention = _tiny_delta_attention(layer_index)
        initial_conv = mx.random.normal((1, 3, 192), dtype=mx.bfloat16)
        initial_ssm = mx.random.normal((1, 2, 32, 32), dtype=mx.float32)
        conv_history = mx.random.normal(
            (width, 1, 3, 192), dtype=mx.bfloat16
        )
        raw_v = mx.random.normal((1, width, 2, 32), dtype=mx.bfloat16)
        raw_k = mx.random.normal((1, width, 2, 32), dtype=mx.bfloat16)
        post_exp_gk = mx.sigmoid(
            mx.random.normal((1, width, 2, 32), dtype=mx.float32)
        )
        beta = mx.sigmoid(mx.random.normal((1, width, 2), dtype=mx.float32))
        cache = ArraysCache(size=2)
        cache.cache = [initial_conv, initial_ssm]
        cache.begin_speculative(width)
        cache.cache = [conv_history[-1], initial_ssm]
        cache.capture_speculative(
            [
                conv_history,
                SpeculativeReplayState(
                    width=width,
                    raw_inputs=(raw_v, raw_k, post_exp_gk, beta),
                    history_axis=1,
                    state_shape=tuple(initial_ssm.shape),
                    state_dtype=initial_ssm.dtype,
                    replay=attention._replay_speculative_ssm,
                ),
            ]
        )
        states.append((layer_index, cache, [initial_conv, initial_ssm]))
    mx.eval(
        [
            value
            for _, cache, _ in states
            for history in cache._speculative_state_history
            for value in (
                history.raw_inputs
                if isinstance(history, SpeculativeReplayState)
                else (history,)
            )
        ],
        [initial for _, _, initial in states for initial in initial],
    )
    return states


def _tiny_model() -> Model:
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
    model.eval()
    mx.eval(model.parameters())
    return model


def _cache_arrays(cache):
    return [value for _, value in tree_flatten([entry.state for entry in cache])]


def _assert_raw_bits_equal(
    test: unittest.TestCase,
    expected: mx.array,
    actual: mx.array,
) -> None:
    mx.eval(expected, actual)
    test.assertEqual(actual.shape, expected.shape)
    test.assertEqual(actual.dtype, expected.dtype)
    test.assertTrue(
        bool(mx.array_equal(expected.view(mx.uint8), actual.view(mx.uint8)).item())
    )


@unittest.skipUnless(mx.metal.is_available(), "requires Metal")
class BatchedReplaySSMTest(unittest.TestCase):
    def setUp(self):
        mx.set_default_device(mx.gpu)
        mx.random.seed(812)
        reset_batched_replayssm_commit_counters()

    def test_three_layer_batch_is_bit_exact_for_every_consumed_prefix(self):
        width = 3
        array_states = _synthetic_array_states(3, width)
        for consumed in (1, 2, 3):
            with self.subTest(consumed=consumed), mock.patch.dict(
                os.environ,
                {BATCHED_REPLAYSSM_EXPECTED_LAYERS_ENV: "3"},
                clear=False,
            ):
                stock = [
                    cache.prepare_speculative(consumed)
                    for _, cache, _ in array_states
                ]
                batched, attestation = _prepare_batched_replayssm_commit(
                    array_states, width, consumed
                )
                self.assertIsNotNone(batched)
                self.assertTrue(attestation["used_batched_path"])
                self.assertEqual(attestation["observed_layers"], 3)
                self.assertEqual(attestation["consumed"], consumed)
                self.assertEqual(attestation["batched_replay_launches"], 1)
                self.assertNotIn("required_target_digest_sha256", attestation)
                batched_states = [states for _, states in batched]
                mx.eval(stock, batched_states)
                for expected_layer, actual_layer in zip(
                    stock, batched_states, strict=True
                ):
                    for expected, actual in zip(
                        expected_layer, actual_layer, strict=True
                    ):
                        self.assertTrue(mx.array_equal(expected, actual).item())

        for _, cache, _ in array_states:
            cache.cancel_speculative()

    def test_full_accept_identity_commit_reuses_terminal_states_and_is_exact(self):
        model = _tiny_model()
        stock_cache = model.make_cache()
        identity_cache = model.make_cache()
        prompt = mx.array([[1, 2]], dtype=mx.uint32)
        stock_prefill = model(prompt, cache=stock_cache)
        identity_prefill = model(prompt, cache=identity_cache)
        mx.eval(
            stock_prefill,
            identity_prefill,
            _cache_arrays(stock_cache),
            _cache_arrays(identity_cache),
        )

        verify_tokens = mx.array([[3, 4, 5]], dtype=mx.uint32)
        common_env = {
            REPLAYSSM_SPECULATIVE_ENV: "1",
            BATCHED_REPLAYSSM_COMMIT_ENV: "1",
            BATCHED_REPLAYSSM_EXPECTED_LAYERS_ENV: "3",
        }
        with mock.patch.dict(
            os.environ,
            {**common_env, FULL_ACCEPT_IDENTITY_COMMIT_ENV: "0"},
            clear=False,
        ):
            stock_transaction = model.begin_speculative_cache(stock_cache, 3)
            stock_wide = model(verify_tokens, cache=stock_cache)
            mx.eval(stock_wide)
            model.resolve_speculative_cache(stock_transaction, consumed=3)

        with mock.patch.dict(
            os.environ,
            {**common_env, FULL_ACCEPT_IDENTITY_COMMIT_ENV: "1"},
            clear=False,
        ):
            identity_transaction = model.begin_speculative_cache(identity_cache, 3)
            identity_wide = model(verify_tokens, cache=identity_cache)
            mx.eval(identity_wide)
            terminal_states = {
                index: tuple(entry.cache)
                for index, entry in enumerate(identity_cache)
                if isinstance(entry, ArraysCache)
            }
            model.resolve_speculative_cache(identity_transaction, consumed=3)

        attestation = identity_transaction.batched_replayssm_attestation
        self.assertEqual(
            attestation["schema"], "kimi-k3-full-accept-identity-commit-v1"
        )
        self.assertEqual(attestation["status"], "identity_committed")
        self.assertTrue(attestation["used_identity_path"])
        self.assertEqual(attestation["expected_layers"], 3)
        self.assertEqual(attestation["observed_layers"], 3)
        self.assertEqual(attestation["state_references_reused"], 6)
        self.assertEqual(attestation["detach_nodes"], 0)
        self.assertTrue(attestation["raw_reference_commit"])
        for index, states in terminal_states.items():
            for expected_object, committed_object in zip(
                states, identity_cache[index].cache, strict=True
            ):
                self.assertIs(committed_object, expected_object)

        stock_arrays = _cache_arrays(stock_cache)
        identity_arrays = _cache_arrays(identity_cache)
        mx.eval(stock_arrays, identity_arrays)
        _assert_raw_bits_equal(self, stock_wide, identity_wide)
        for expected, actual in zip(stock_arrays, identity_arrays, strict=True):
            _assert_raw_bits_equal(self, expected, actual)

        next_token = mx.array([[6]], dtype=mx.uint32)
        stock_next = model(next_token, cache=stock_cache)
        identity_next = model(next_token, cache=identity_cache)
        mx.eval(stock_next, identity_next)
        _assert_raw_bits_equal(self, stock_next, identity_next)

        counters = batched_replayssm_commit_counters()
        self.assertEqual(counters["attempted_prepares"], 2)
        self.assertEqual(counters["batched_commits"], 1)
        self.assertEqual(counters["identity_prepares"], 1)
        self.assertEqual(counters["identity_commits"], 1)
        self.assertEqual(counters["layers_identity_committed"], 3)

    def test_identity_commit_is_not_used_for_partial_acceptance(self):
        model = _tiny_model()
        cache = model.make_cache()
        prefill = model(mx.array([[1, 2]], dtype=mx.uint32), cache=cache)
        mx.eval(prefill, _cache_arrays(cache))
        with mock.patch.dict(
            os.environ,
            {
                REPLAYSSM_SPECULATIVE_ENV: "1",
                BATCHED_REPLAYSSM_COMMIT_ENV: "1",
                BATCHED_REPLAYSSM_EXPECTED_LAYERS_ENV: "3",
                FULL_ACCEPT_IDENTITY_COMMIT_ENV: "1",
            },
            clear=False,
        ):
            transaction = model.begin_speculative_cache(cache, 3)
            wide = model(mx.array([[3, 4, 5]], dtype=mx.uint32), cache=cache)
            mx.eval(wide)
            model.resolve_speculative_cache(transaction, consumed=2)
        self.assertEqual(
            transaction.batched_replayssm_attestation["status"],
            "batched_committed",
        )

    def test_identity_commit_requires_batched_selector_and_expected_layers(self):
        cases = (
            (
                "missing batched selector",
                {
                    BATCHED_REPLAYSSM_COMMIT_ENV: "0",
                    BATCHED_REPLAYSSM_EXPECTED_LAYERS_ENV: "3",
                },
                f"{BATCHED_REPLAYSSM_COMMIT_ENV}=1",
            ),
            (
                "unexpected layer count",
                {
                    BATCHED_REPLAYSSM_COMMIT_ENV: "1",
                    BATCHED_REPLAYSSM_EXPECTED_LAYERS_ENV: "4",
                },
                "requires exactly 4 KDA layers, got 3",
            ),
        )
        for label, selector_environment, error_pattern in cases:
            with self.subTest(label=label):
                model = _tiny_model()
                cache = model.make_cache()
                prefill = model(mx.array([[1, 2]], dtype=mx.uint32), cache=cache)
                mx.eval(prefill, _cache_arrays(cache))
                initial_array_states = {
                    index: list(entry.cache)
                    for index, entry in enumerate(cache)
                    if isinstance(entry, ArraysCache)
                }
                environment = {
                    REPLAYSSM_SPECULATIVE_ENV: "1",
                    FULL_ACCEPT_IDENTITY_COMMIT_ENV: "1",
                    **selector_environment,
                }
                with mock.patch.dict(os.environ, environment, clear=False):
                    transaction = model.begin_speculative_cache(cache, 3)
                    wide = model(mx.array([[3, 4, 5]], dtype=mx.uint32), cache=cache)
                    mx.eval(wide)
                    with self.assertRaisesRegex(ValueError, error_pattern):
                        model.resolve_speculative_cache(transaction, consumed=3)

                self.assertFalse(transaction.active)
                self.assertIsNone(transaction.batched_replayssm_attestation)
                for index, expected in initial_array_states.items():
                    for actual_state, expected_state in zip(
                        cache[index].cache, expected, strict=True
                    ):
                        self.assertIs(actual_state, expected_state)

        counters = batched_replayssm_commit_counters()
        self.assertEqual(counters["identity_prepares"], 0)
        self.assertEqual(counters["identity_commits"], 0)
        self.assertEqual(counters["identity_errors"], 0)

    def test_full_accept_identity_commit_has_flat_active_memory(self):
        model = _tiny_model()
        cache = model.make_cache()
        prefill = model(mx.array([[1, 2]], dtype=mx.uint32), cache=cache)
        mx.eval(prefill, _cache_arrays(cache))
        environment = {
            REPLAYSSM_SPECULATIVE_ENV: "1",
            BATCHED_REPLAYSSM_COMMIT_ENV: "1",
            BATCHED_REPLAYSSM_EXPECTED_LAYERS_ENV: "3",
            FULL_ACCEPT_IDENTITY_COMMIT_ENV: "1",
        }

        def full_accept_round(start: int) -> None:
            transaction = model.begin_speculative_cache(cache, 3)
            tokens = mx.array([[start, start + 1, start + 2]], dtype=mx.uint32)
            wide = model(tokens, cache=cache)
            mx.eval(wide)
            model.resolve_speculative_cache(transaction, consumed=3)

        with mock.patch.dict(os.environ, environment, clear=False):
            for start in range(3, 15, 3):
                full_accept_round(start)
            gc.collect()
            mx.clear_cache()

            active_samples = []
            for start in range(15, 51, 3):
                full_accept_round(start)
                gc.collect()
                mx.clear_cache()
                active_samples.append(mx.get_active_memory())

        # The MLA cache stays inside its first allocation block at these
        # lengths.  Retaining a wide graph therefore appears as monotonic
        # active-memory growth; a detached terminal state stays flat apart
        # from one small allocator page.
        tolerance = 256 * 1024
        self.assertLessEqual(
            max(active_samples) - min(active_samples),
            tolerance,
            f"active-memory samples drifted: {active_samples}",
        )

    def test_identity_commit_selector_is_fail_closed(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(full_accept_identity_commit_enabled())
        with mock.patch.dict(
            os.environ, {FULL_ACCEPT_IDENTITY_COMMIT_ENV: "0"}, clear=False
        ):
            self.assertFalse(full_accept_identity_commit_enabled())
        with mock.patch.dict(
            os.environ, {FULL_ACCEPT_IDENTITY_COMMIT_ENV: "2"}, clear=False
        ):
            with self.assertRaisesRegex(ValueError, "must be 0 or 1"):
                full_accept_identity_commit_enabled()

    def test_layer_count_mismatch_falls_back_before_graph_build(self):
        array_states = _synthetic_array_states(3, width=3)
        with mock.patch.dict(
            os.environ,
            {BATCHED_REPLAYSSM_EXPECTED_LAYERS_ENV: "4"},
            clear=False,
        ):
            prepared, attestation = _prepare_batched_replayssm_commit(
                array_states, width=3, consumed=2
            )
        self.assertIsNone(prepared)
        self.assertFalse(attestation["used_batched_path"])
        self.assertEqual(
            attestation["fallback_reason"], "unexpected_kda_layer_count"
        )
        for _, cache, _ in array_states:
            cache.cancel_speculative()

    def test_full_tiny_model_commit_matches_stock_and_attests(self):
        model = _tiny_model()
        stock_cache = model.make_cache()
        batched_cache = model.make_cache()
        fallback_cache = model.make_cache()
        prompt = mx.array([[1, 2]], dtype=mx.uint32)
        stock_prefill = model(prompt, cache=stock_cache)
        batched_prefill = model(prompt, cache=batched_cache)
        fallback_prefill = model(prompt, cache=fallback_cache)
        mx.eval(
            stock_prefill,
            batched_prefill,
            fallback_prefill,
            _cache_arrays(stock_cache),
            _cache_arrays(batched_cache),
            _cache_arrays(fallback_cache),
        )

        verify_tokens = mx.array([[3, 4, 5]], dtype=mx.uint32)
        common_env = {
            REPLAYSSM_SPECULATIVE_ENV: "1",
            BATCHED_REPLAYSSM_EXPECTED_LAYERS_ENV: "3",
        }
        with mock.patch.dict(
            os.environ,
            {**common_env, BATCHED_REPLAYSSM_COMMIT_ENV: "0"},
            clear=False,
        ):
            stock_transaction = model.begin_speculative_cache(stock_cache, 3)
            stock_wide = model(verify_tokens, cache=stock_cache)
            mx.eval(stock_wide)
            model.resolve_speculative_cache(stock_transaction, consumed=2)

        with mock.patch.dict(
            os.environ,
            {**common_env, BATCHED_REPLAYSSM_COMMIT_ENV: "1"},
            clear=False,
        ):
            batched_transaction = model.begin_speculative_cache(batched_cache, 3)
            batched_wide = model(verify_tokens, cache=batched_cache)
            mx.eval(batched_wide)
            model.resolve_speculative_cache(batched_transaction, consumed=2)

        batched_telemetry = batched_replayssm_commit_telemetry()
        self.assertEqual(
            batched_telemetry["schema"],
            "kimi-k3-batched-replayssm-telemetry-v1",
        )
        self.assertEqual(
            batched_telemetry["latest_attestation"]["status"],
            "batched_committed",
        )
        self.assertNotIn(
            "required_target_digest_sha256",
            batched_telemetry["latest_attestation"],
        )
        batched_telemetry["counters"]["batched_commits"] = -1
        batched_telemetry["latest_attestation"]["stacked_state_shape"].append(-1)
        detached_check = batched_replayssm_commit_telemetry()
        self.assertEqual(detached_check["counters"]["batched_commits"], 1)
        self.assertEqual(
            detached_check["latest_attestation"]["stacked_state_shape"],
            [3, 2, 32, 32],
        )

        with mock.patch.dict(
            os.environ,
            {
                **common_env,
                BATCHED_REPLAYSSM_COMMIT_ENV: "1",
                BATCHED_REPLAYSSM_EXPECTED_LAYERS_ENV: "4",
            },
            clear=False,
        ):
            fallback_transaction = model.begin_speculative_cache(fallback_cache, 3)
            fallback_wide = model(verify_tokens, cache=fallback_cache)
            mx.eval(fallback_wide)
            model.resolve_speculative_cache(fallback_transaction, consumed=2)

        stock_arrays = _cache_arrays(stock_cache)
        batched_arrays = _cache_arrays(batched_cache)
        fallback_arrays = _cache_arrays(fallback_cache)
        mx.eval(stock_arrays, batched_arrays, fallback_arrays)
        self.assertTrue(mx.array_equal(stock_wide, batched_wide).item())
        self.assertTrue(mx.array_equal(stock_wide, fallback_wide).item())
        self.assertEqual(len(stock_arrays), len(batched_arrays))
        self.assertEqual(len(stock_arrays), len(fallback_arrays))
        for expected, actual_batched, actual_fallback in zip(
            stock_arrays,
            batched_arrays,
            fallback_arrays,
            strict=True,
        ):
            self.assertTrue(mx.array_equal(expected, actual_batched).item())
            self.assertTrue(mx.array_equal(expected, actual_fallback).item())

        next_token = mx.array([[6]], dtype=mx.uint32)
        stock_next = model(next_token, cache=stock_cache)
        batched_next = model(next_token, cache=batched_cache)
        fallback_next = model(next_token, cache=fallback_cache)
        mx.eval(stock_next, batched_next, fallback_next)
        self.assertTrue(mx.array_equal(stock_next, batched_next).item())
        self.assertTrue(mx.array_equal(stock_next, fallback_next).item())

        self.assertEqual(
            stock_transaction.batched_replayssm_attestation["status"], "disabled"
        )
        attestation = batched_transaction.batched_replayssm_attestation
        self.assertEqual(attestation["status"], "batched_committed")
        self.assertTrue(attestation["used_batched_path"])
        self.assertEqual(attestation["observed_layers"], 3)
        fallback_attestation = fallback_transaction.batched_replayssm_attestation
        self.assertEqual(fallback_attestation["status"], "fallback_committed")
        self.assertFalse(fallback_attestation["used_batched_path"])
        self.assertEqual(
            fallback_attestation["fallback_reason"],
            "unexpected_kda_layer_count",
        )
        counters = batched_replayssm_commit_counters()
        self.assertEqual(counters["attempted_prepares"], 2)
        self.assertEqual(counters["batched_commits"], 1)
        self.assertEqual(counters["fallback_commits"], 1)
        self.assertEqual(counters["layers_batched"], 3)
        telemetry = batched_replayssm_commit_telemetry()
        self.assertEqual(telemetry["counters"], counters)
        self.assertGreater(telemetry["revision"], 0)
        self.assertEqual(
            telemetry["latest_attestation"]["status"], "fallback_committed"
        )
        self.assertNotIn(
            "required_target_digest_sha256", telemetry["latest_attestation"]
        )

    def test_public_counter_snapshot_is_thread_safe(self):
        workers = 8
        increments_per_worker = 250

        def increment_counters():
            for _ in range(increments_per_worker):
                _increment_batched_replayssm_counter("attempted_prepares")

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(increment_counters) for _ in range(workers)]
            for future in futures:
                future.result()

        telemetry = batched_replayssm_commit_telemetry()
        self.assertEqual(
            telemetry["counters"]["attempted_prepares"],
            workers * increments_per_worker,
        )
        self.assertIsNone(telemetry["latest_attestation"])
        self.assertGreaterEqual(
            telemetry["revision"], workers * increments_per_worker
        )

    def test_batched_eval_failure_rolls_back_every_cache(self):
        model = _tiny_model()
        cache = model.make_cache()
        prefill = model(mx.array([[1, 2]], dtype=mx.uint32), cache=cache)
        mx.eval(prefill, _cache_arrays(cache))
        initial_array_states = {
            index: list(entry.cache)
            for index, entry in enumerate(cache)
            if isinstance(entry, ArraysCache)
        }
        initial_kv_states = {
            index: (entry.keys, entry.values, entry.offset)
            for index, entry in enumerate(cache)
            if not isinstance(entry, ArraysCache)
        }
        environment = {
            REPLAYSSM_SPECULATIVE_ENV: "1",
            BATCHED_REPLAYSSM_COMMIT_ENV: "1",
            BATCHED_REPLAYSSM_EXPECTED_LAYERS_ENV: "3",
        }
        with mock.patch.dict(os.environ, environment, clear=False):
            transaction = model.begin_speculative_cache(cache, 3)
            wide = model(mx.array([[3, 4, 5]], dtype=mx.uint32), cache=cache)
            mx.eval(wide)
            with mock.patch(
                "mlx_lm.models.kimi_k3.mx.eval",
                side_effect=RuntimeError("injected batched evaluation failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "injected batched"):
                    model.resolve_speculative_cache(transaction, consumed=2)

        self.assertFalse(transaction.active)
        self.assertEqual(
            transaction.batched_replayssm_attestation["status"],
            "batched_error_rolled_back",
        )
        for index, expected in initial_array_states.items():
            for actual_state, expected_state in zip(
                cache[index].cache, expected, strict=True
            ):
                self.assertIs(actual_state, expected_state)
        for index, (keys, values, offset) in initial_kv_states.items():
            self.assertIs(cache[index].keys, keys)
            self.assertIs(cache[index].values, values)
            self.assertEqual(cache[index].offset, offset)
        counters = batched_replayssm_commit_counters()
        self.assertEqual(counters["batched_errors"], 1)
        self.assertEqual(counters["batched_commits"], 0)
        telemetry = batched_replayssm_commit_telemetry()
        self.assertEqual(
            telemetry["latest_attestation"],
            transaction.batched_replayssm_attestation,
        )

    def test_identity_eval_failure_rolls_back_every_cache(self):
        model = _tiny_model()
        cache = model.make_cache()
        prefill = model(mx.array([[1, 2]], dtype=mx.uint32), cache=cache)
        mx.eval(prefill, _cache_arrays(cache))
        initial_array_states = {
            index: list(entry.cache)
            for index, entry in enumerate(cache)
            if isinstance(entry, ArraysCache)
        }
        initial_kv_states = {
            index: (entry.keys, entry.values, entry.offset)
            for index, entry in enumerate(cache)
            if not isinstance(entry, ArraysCache)
        }
        environment = {
            REPLAYSSM_SPECULATIVE_ENV: "1",
            BATCHED_REPLAYSSM_COMMIT_ENV: "1",
            BATCHED_REPLAYSSM_EXPECTED_LAYERS_ENV: "3",
            FULL_ACCEPT_IDENTITY_COMMIT_ENV: "1",
        }
        with mock.patch.dict(os.environ, environment, clear=False):
            transaction = model.begin_speculative_cache(cache, 3)
            wide = model(mx.array([[3, 4, 5]], dtype=mx.uint32), cache=cache)
            mx.eval(wide)
            with mock.patch(
                "mlx_lm.models.kimi_k3.mx.eval",
                side_effect=RuntimeError("injected identity evaluation failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "injected identity"):
                    model.resolve_speculative_cache(transaction, consumed=3)

        self.assertFalse(transaction.active)
        self.assertEqual(
            transaction.batched_replayssm_attestation["status"],
            "identity_error_rolled_back",
        )
        for index, expected in initial_array_states.items():
            for actual_state, expected_state in zip(
                cache[index].cache, expected, strict=True
            ):
                self.assertIs(actual_state, expected_state)
        for index, (keys, values, offset) in initial_kv_states.items():
            self.assertIs(cache[index].keys, keys)
            self.assertIs(cache[index].values, values)
            self.assertEqual(cache[index].offset, offset)
        counters = batched_replayssm_commit_counters()
        self.assertEqual(counters["identity_prepares"], 1)
        self.assertEqual(counters["identity_errors"], 1)
        self.assertEqual(counters["identity_commits"], 0)
        telemetry = batched_replayssm_commit_telemetry()
        self.assertEqual(
            telemetry["latest_attestation"],
            transaction.batched_replayssm_attestation,
        )

    def test_identity_mid_commit_failure_rolls_back_every_cache(self):
        model = _tiny_model()
        cache = model.make_cache()
        prefill = model(mx.array([[1, 2]], dtype=mx.uint32), cache=cache)
        mx.eval(prefill, _cache_arrays(cache))
        initial_array_states = {
            index: list(entry.cache)
            for index, entry in enumerate(cache)
            if isinstance(entry, ArraysCache)
        }
        initial_kv_states = {
            index: (entry.keys, entry.values, entry.offset)
            for index, entry in enumerate(cache)
            if not isinstance(entry, ArraysCache)
        }
        environment = {
            REPLAYSSM_SPECULATIVE_ENV: "1",
            BATCHED_REPLAYSSM_COMMIT_ENV: "1",
            BATCHED_REPLAYSSM_EXPECTED_LAYERS_ENV: "3",
            FULL_ACCEPT_IDENTITY_COMMIT_ENV: "1",
        }
        original_commit = ArraysCache.commit_speculative
        commit_calls = 0

        def fail_second_commit(layer_cache, states):
            nonlocal commit_calls
            commit_calls += 1
            if commit_calls == 2:
                raise RuntimeError("injected identity mid-commit failure")
            return original_commit(layer_cache, states)

        with mock.patch.dict(os.environ, environment, clear=False):
            transaction = model.begin_speculative_cache(cache, 3)
            wide = model(mx.array([[3, 4, 5]], dtype=mx.uint32), cache=cache)
            mx.eval(wide)
            with mock.patch.object(
                ArraysCache,
                "commit_speculative",
                autospec=True,
                side_effect=fail_second_commit,
            ):
                with self.assertRaisesRegex(RuntimeError, "mid-commit"):
                    model.resolve_speculative_cache(transaction, consumed=3)

        self.assertEqual(commit_calls, 2)
        self.assertFalse(transaction.active)
        self.assertEqual(
            transaction.batched_replayssm_attestation["status"],
            "identity_error_rolled_back",
        )
        for index, expected in initial_array_states.items():
            for actual_state, expected_state in zip(
                cache[index].cache, expected, strict=True
            ):
                self.assertIs(actual_state, expected_state)
        for index, (keys, values, offset) in initial_kv_states.items():
            self.assertIs(cache[index].keys, keys)
            self.assertIs(cache[index].values, values)
            self.assertEqual(cache[index].offset, offset)
        counters = batched_replayssm_commit_counters()
        self.assertEqual(counters["identity_prepares"], 1)
        self.assertEqual(counters["identity_errors"], 1)
        self.assertEqual(counters["identity_commits"], 0)


if __name__ == "__main__":
    unittest.main()
