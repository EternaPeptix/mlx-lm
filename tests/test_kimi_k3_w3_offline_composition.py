# Copyright © 2026 Apple Inc.

from __future__ import annotations

import copy
import os
import unittest
from unittest import mock

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_lm.models import kimi_k3
from mlx_lm.models.cache import ArraysCache, SpeculativeReplayState
from mlx_lm.models.gated_delta import compute_g_safe
from mlx_lm.models.kimi_k3_multibank_moe_front import (
    MULTIBANK_MOE_FRONT_ENV,
    multibank_moe_front_enabled,
)
from mlx_lm.models.kimi_k3_packed_moe_front import (
    AUTHORITATIVE_PACKED_MOE_FRONT_ENV,
    AUTHORITATIVE_PACKED_MOE_FRONT_WIDTH3_ENV,
    PACKED_MOE_FRONT_ENV,
    authoritative_packed_moe_front_enabled,
    authoritative_packed_moe_front_width3_enabled,
    packed_moe_front_enabled,
)
from mlx_lm.models.kimi_k3_w3_prework import (
    K3_W3_PREWORK_HISTORY_ENV,
    k3_w3_prework_history_enabled,
)


def _config():
    kda_layers = (1, 2, 3, 5, 6, 7)
    full_attn_layers = [i for i in range(1, 9) if i not in kda_layers]
    return {
        "model_type": "kimi_k3",
        "vocab_size": 1024,
        "num_hidden_layers": 8,
        "text_config": {
            "model_type": "kimi_linear",
            "vocab_size": 1024,
            "hidden_size": 64,
            "num_hidden_layers": 8,
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
            "attn_res_block_size": 4,
        },
    }


_BASE_ENV = {
    kimi_k3.COMPILED_DECODE_ENV: "0",
    kimi_k3.ASYNC_DECODE_BOUNDARIES_ENV: "laguna8",
    kimi_k3.ASYNC_DECODE_STATE_ENV: "hidden",
    kimi_k3.ASYNC_DECODE_WIDTH3_ENV: "1",
    kimi_k3.PROJECTED_KV_CACHE_ENV: "1",
    kimi_k3.REPLAYSSM_SPECULATIVE_ENV: "1",
    kimi_k3.BATCHED_REPLAYSSM_COMMIT_ENV: "0",
    "MLX_LM_KIMI_K3_FUSED_EXPERTS": "0",
    MULTIBANK_MOE_FRONT_ENV: "0",
    PACKED_MOE_FRONT_ENV: "0",
}


def _selector_env(*, packed: bool, kda: bool):
    return {
        **_BASE_ENV,
        AUTHORITATIVE_PACKED_MOE_FRONT_ENV: "1" if packed else "0",
        AUTHORITATIVE_PACKED_MOE_FRONT_WIDTH3_ENV: "1" if packed else "0",
        K3_W3_PREWORK_HISTORY_ENV: "1" if kda else "0",
    }


def _clear_selector_caches():
    authoritative_packed_moe_front_enabled.cache_clear()
    authoritative_packed_moe_front_width3_enabled.cache_clear()
    packed_moe_front_enabled.cache_clear()
    multibank_moe_front_enabled.cache_clear()
    k3_w3_prework_history_enabled.cache_clear()
    kimi_k3.fused_k3_experts_enabled.cache_clear()


def _make_model():
    mx.random.seed(2026081611)
    args = kimi_k3.ModelArgs.from_dict(_config())
    model = kimi_k3.Model(args)
    model.set_dtype(mx.bfloat16)
    model.eval()
    mx.eval(model.parameters())
    return model


def _warm_cache(model, length=8):
    cache = model.make_cache()
    tokens = mx.arange(length, dtype=mx.int32)[None] % model.args.text_config.vocab_size
    logits = model(tokens, cache=cache)
    mx.eval(logits, [entry.state for entry in cache])
    return cache


def _stock_w3_prework(attention, projected_qkv, state, a_logits):
    batch, width, _ = projected_qkv.shape
    qkv, new_state, history = attention.qkv_conv(
        projected_qkv,
        state,
        None,
        None,
        return_state_history=True,
    )
    projection_dim = attention.projection_dim
    shape = (batch, width, attention.num_heads, attention.head_dim)
    q = qkv[..., :projection_dim].reshape(*shape)
    raw_k = qkv[..., projection_dim : 2 * projection_dim].reshape(*shape)
    v = qkv[..., 2 * projection_dim :].reshape(*shape)
    eps = 1e-6 / attention.head_dim
    q = (attention.scale**2) * mx.fast.rms_norm(q, None, eps)
    k = attention.scale * mx.fast.rms_norm(raw_k, None, eps)
    gk = compute_g_safe(
        attention.A_log.reshape(attention.num_heads, 1),
        a_logits,
        attention.dt_bias.reshape(attention.num_heads, attention.head_dim),
        attention.lower_bound,
    )
    return q, k, raw_k, v, gk, new_state, history


def _exact_array(test, left, right, label):
    test.assertIsInstance(left, mx.array, label)
    test.assertIsInstance(right, mx.array, label)
    test.assertEqual(left.shape, right.shape, f"{label} shape")
    test.assertEqual(left.dtype, right.dtype, f"{label} dtype")
    equal = mx.array_equal(left, right)
    mx.eval(equal)
    test.assertTrue(bool(equal.item()), label)


def _exact_tree(test, left, right, label):
    left_flat = tree_flatten(left)
    right_flat = tree_flatten(right)
    test.assertEqual(
        [name for name, _ in left_flat],
        [name for name, _ in right_flat],
        label,
    )
    for (name, lhs), (_, rhs) in zip(left_flat, right_flat, strict=True):
        if isinstance(lhs, mx.array):
            _exact_array(test, lhs, rhs, f"{label}.{name}")
        else:
            test.assertEqual(lhs, rhs, f"{label}.{name}")


def _exact_speculative_history(test, left, right):
    test.assertEqual(len(left), len(right))
    for layer_index, (lhs, rhs) in enumerate(zip(left, right, strict=True)):
        if not isinstance(lhs, ArraysCache):
            continue
        lhs_history = lhs._speculative_state_history
        rhs_history = rhs._speculative_state_history
        test.assertIsNotNone(lhs_history, f"left history layer {layer_index}")
        test.assertIsNotNone(rhs_history, f"right history layer {layer_index}")
        assert lhs_history is not None and rhs_history is not None
        _exact_array(
            test,
            lhs_history[0],
            rhs_history[0],
            f"conv history layer {layer_index}",
        )
        lhs_replay = lhs_history[1]
        rhs_replay = rhs_history[1]
        test.assertIsInstance(lhs_replay, SpeculativeReplayState)
        test.assertIsInstance(rhs_replay, SpeculativeReplayState)
        test.assertEqual(lhs_replay.width, rhs_replay.width)
        test.assertEqual(lhs_replay.history_axis, rhs_replay.history_axis)
        for raw_index, (lhs_raw, rhs_raw) in enumerate(
            zip(lhs_replay.raw_inputs, rhs_replay.raw_inputs, strict=True)
        ):
            _exact_array(
                test,
                lhs_raw,
                rhs_raw,
                f"replay raw {raw_index} layer {layer_index}",
            )


def _exact_closed_cache(test, left, right):
    test.assertEqual([type(value) for value in left], [type(value) for value in right])
    for layer_index, (lhs, rhs) in enumerate(zip(left, right, strict=True)):
        _exact_tree(test, lhs.state, rhs.state, f"cache layer {layer_index}")
        if isinstance(lhs, ArraysCache):
            test.assertEqual(lhs.speculative_width, 0)
            test.assertEqual(rhs.speculative_width, 0)
            test.assertFalse(lhs.speculative_ready)
            test.assertFalse(rhs.speculative_ready)
        elif isinstance(lhs, kimi_k3.KimiK3ProjectedKVCache):
            test.assertEqual(lhs.offset, rhs.offset)
            test.assertEqual(lhs.projected_valid_offset, rhs.projected_valid_offset)
            test.assertIsNone(lhs._projected_transaction_token)
            test.assertIsNone(rhs._projected_transaction_token)
            if lhs.projected_keys is None or rhs.projected_keys is None:
                test.assertIs(lhs.projected_keys, rhs.projected_keys)
                test.assertIs(lhs.projected_values, rhs.projected_values)
            else:
                valid = lhs.projected_valid_offset
                _exact_array(
                    test,
                    lhs.projected_keys[..., :valid, :],
                    rhs.projected_keys[..., :valid, :],
                    f"projected keys layer {layer_index}",
                )
                _exact_array(
                    test,
                    lhs.projected_values[..., :valid, :],
                    rhs.projected_values[..., :valid, :],
                    f"projected values layer {layer_index}",
                )


class _CompositionAdapters:
    """Route the production-only seams through exact stock arithmetic.

    The focused source suites exercise the real production Metal kernels. This
    adapter lets a small full target prove ordering, ownership, and composition
    without weakening either production selector in model code.
    """

    def __init__(self):
        self.kda_dispatches = 0
        self.packed_dispatches = 0

    def can_use_kda(
        self,
        attention,
        x,
        state,
        *,
        short_conv_type,
        inner_conv_type,
        mask,
        lengths,
        capture_speculative,
    ):
        del short_conv_type, inner_conv_type
        return (
            k3_w3_prework_history_enabled()
            and capture_speculative
            and not attention.training
            and x.ndim == 3
            and x.shape[:2] == (1, 3)
            and state is not None
            and mask is None
            and lengths is None
        )

    def run_kda(self, attention, projected_qkv, state, a_logits):
        self.kda_dispatches += 1
        return _stock_w3_prework(attention, projected_qkv, state, a_logits)

    def run_packed(self, sparse_moe, x):
        if not (
            authoritative_packed_moe_front_enabled()
            and authoritative_packed_moe_front_width3_enabled()
            and x.ndim == 3
            and x.shape[:2] == (1, 3)
        ):
            return None
        self.packed_dispatches += 1
        return (
            sparse_moe.shared_experts.gate_proj(x),
            sparse_moe.shared_experts.up_proj(x),
            sparse_moe.gate(x),
            sparse_moe.routed_expert_down_proj(x),
        )

    def patch(self):
        return (
            mock.patch(
                "mlx_lm.models.kimi_k3.can_use_k3_w3_prework_history",
                side_effect=self.can_use_kda,
            ),
            mock.patch(
                "mlx_lm.models.kimi_k3.maybe_fused_k3_w3_prework_history",
                side_effect=self.run_kda,
            ),
            mock.patch(
                "mlx_lm.models.kimi_k3.maybe_authoritative_packed_k3_moe_front",
                side_effect=self.run_packed,
            ),
        )


@unittest.skipUnless(mx.metal.is_available(), "requires Metal")
class K3W3OfflineCompositionTests(unittest.TestCase):
    def setUp(self):
        self._device = mx.default_device()
        mx.set_default_device(mx.gpu)
        with mock.patch.dict(os.environ, _selector_env(packed=False, kda=False), clear=True):
            _clear_selector_caches()
            self.model = _make_model()
            self.base_cache = _warm_cache(self.model)
        _clear_selector_caches()

    def tearDown(self):
        _clear_selector_caches()
        mx.set_default_device(self._device)

    def _forward(self, cache, *, packed, kda, defer=False):
        adapters = _CompositionAdapters()
        patches = adapters.patch()
        with (
            mock.patch.dict(
                os.environ,
                _selector_env(packed=packed, kda=kda),
                clear=True,
            ),
            patches[0],
            patches[1],
            patches[2],
        ):
            _clear_selector_caches()
            result = self.model.forward_with_aux_hidden_states(
                mx.array([[17, 23, 29]], dtype=mx.int32),
                cache,
                (1, 3),
                defer_async_decode_boundaries=defer,
            )
        _clear_selector_caches()
        return result, adapters

    def test_selectors_individually_and_jointly_are_exact(self):
        reference_cache = copy.deepcopy(self.base_cache)
        reference_transaction = self.model.begin_speculative_cache(reference_cache, 3)
        reference, reference_adapters = self._forward(
            reference_cache,
            packed=False,
            kda=False,
        )
        mx.eval(
            reference.logits,
            reference.aux_hidden_states,
            [entry.state for entry in reference_cache],
        )
        self.model.resolve_speculative_cache(reference_transaction, 3)
        self.assertEqual(reference_adapters.kda_dispatches, 0)
        self.assertEqual(reference_adapters.packed_dispatches, 0)

        for packed, kda in ((True, False), (False, True), (True, True)):
            with self.subTest(packed=packed, kda=kda):
                cache = copy.deepcopy(self.base_cache)
                transaction = self.model.begin_speculative_cache(cache, 3)
                candidate, adapters = self._forward(
                    cache,
                    packed=packed,
                    kda=kda,
                )
                mx.eval(
                    candidate.logits,
                    candidate.aux_hidden_states,
                    [entry.state for entry in cache],
                )
                self.model.resolve_speculative_cache(transaction, 3)

                _exact_array(self, candidate.logits, reference.logits, "logits")
                _exact_array(
                    self,
                    mx.argmax(candidate.logits, axis=-1),
                    mx.argmax(reference.logits, axis=-1),
                    "tokens",
                )
                _exact_tree(
                    self,
                    candidate.aux_hidden_states,
                    reference.aux_hidden_states,
                    "aux hidden",
                )
                _exact_closed_cache(self, cache, reference_cache)
                self.assertEqual(adapters.kda_dispatches > 0, kda)
                self.assertEqual(adapters.packed_dispatches > 0, packed)

    def test_joint_target_history_resolve_123_and_deferred_root_are_exact(self):
        parameter_snapshot = {
            name: value + mx.zeros_like(value)
            for name, value in tree_flatten(self.model.parameters())
            if isinstance(value, mx.array)
        }
        mx.eval(parameter_snapshot)

        for consumed in (1, 2, 3):
            with self.subTest(consumed=consumed):
                control_cache = copy.deepcopy(self.base_cache)
                candidate_cache = copy.deepcopy(self.base_cache)
                control_transaction = self.model.begin_speculative_cache(
                    control_cache, 3
                )
                candidate_transaction = self.model.begin_speculative_cache(
                    candidate_cache, 3
                )
                control, control_adapters = self._forward(
                    control_cache,
                    packed=False,
                    kda=False,
                )
                with mock.patch.object(mx, "async_eval", wraps=mx.async_eval) as submit:
                    candidate, candidate_adapters = self._forward(
                        candidate_cache,
                        packed=True,
                        kda=True,
                        defer=True,
                    )
                submit.assert_not_called()
                self.assertEqual(control_adapters.kda_dispatches, 0)
                self.assertEqual(control_adapters.packed_dispatches, 0)
                self.assertGreater(candidate_adapters.kda_dispatches, 0)
                self.assertGreater(candidate_adapters.packed_dispatches, 0)
                self.assertEqual(len(candidate.deferred_async_decode_states), 1)
                # Boundary 1 is also the first requested target tap. Equality
                # proves capture happens after KDA attention and packed MoE.
                _exact_array(
                    self,
                    candidate.deferred_async_decode_states[0],
                    candidate.aux_hidden_states[0],
                    "post-composed deferred root",
                )
                mx.eval(
                    control.logits,
                    control.aux_hidden_states,
                    [entry.state for entry in control_cache],
                    candidate.logits,
                    candidate.aux_hidden_states,
                    candidate.deferred_async_decode_states,
                    [entry.state for entry in candidate_cache],
                )
                _exact_array(self, candidate.logits, control.logits, "logits")
                _exact_array(
                    self,
                    mx.argmax(candidate.logits, axis=-1),
                    mx.argmax(control.logits, axis=-1),
                    "tokens",
                )
                _exact_tree(
                    self,
                    candidate.aux_hidden_states,
                    control.aux_hidden_states,
                    "aux hidden",
                )
                _exact_speculative_history(self, candidate_cache, control_cache)
                self.model.resolve_speculative_cache(control_transaction, consumed)
                self.model.resolve_speculative_cache(candidate_transaction, consumed)
                _exact_closed_cache(self, candidate_cache, control_cache)

        for name, current in tree_flatten(self.model.parameters()):
            if isinstance(current, mx.array):
                _exact_array(
                    self,
                    current,
                    parameter_snapshot[name],
                    f"source parameter {name}",
                )

    def test_joint_target_cancel_restores_cache_exactly(self):
        candidate_cache = copy.deepcopy(self.base_cache)
        transaction = self.model.begin_speculative_cache(candidate_cache, 3)
        candidate, adapters = self._forward(
            candidate_cache,
            packed=True,
            kda=True,
            defer=True,
        )
        mx.eval(
            candidate.logits,
            candidate.aux_hidden_states,
            candidate.deferred_async_decode_states,
            [entry.state for entry in candidate_cache],
        )
        self.assertGreater(adapters.kda_dispatches, 0)
        self.assertGreater(adapters.packed_dispatches, 0)
        self.model.cancel_speculative_cache(transaction)
        _exact_closed_cache(self, candidate_cache, self.base_cache)

    def test_joint_selectors_fall_back_for_unsupported_width(self):
        inputs = mx.array([[31, 37]], dtype=mx.int32)
        control_cache = copy.deepcopy(self.base_cache)
        candidate_cache = copy.deepcopy(self.base_cache)

        control_adapters = _CompositionAdapters()
        candidate_adapters = _CompositionAdapters()
        control_patches = control_adapters.patch()
        candidate_patches = candidate_adapters.patch()
        with (
            mock.patch.dict(
                os.environ,
                _selector_env(packed=False, kda=False),
                clear=True,
            ),
            control_patches[0],
            control_patches[1],
            control_patches[2],
        ):
            _clear_selector_caches()
            control = self.model(inputs, cache=control_cache)
        with (
            mock.patch.dict(
                os.environ,
                _selector_env(packed=True, kda=True),
                clear=True,
            ),
            candidate_patches[0],
            candidate_patches[1],
            candidate_patches[2],
        ):
            _clear_selector_caches()
            candidate = self.model(inputs, cache=candidate_cache)
        mx.eval(
            control,
            candidate,
            [entry.state for entry in control_cache],
            [entry.state for entry in candidate_cache],
        )
        _exact_array(self, candidate, control, "unsupported width logits")
        _exact_closed_cache(self, candidate_cache, control_cache)
        self.assertEqual(candidate_adapters.kda_dispatches, 0)
        self.assertEqual(candidate_adapters.packed_dispatches, 0)


if __name__ == "__main__":
    unittest.main()
