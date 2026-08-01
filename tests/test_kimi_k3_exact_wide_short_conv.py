from __future__ import annotations

import os
import unittest
from unittest import mock

import mlx.core as mx

from mlx_lm.models.cache import ArraysCache
from mlx_lm.models.gated_delta import gated_delta_kernel
from mlx_lm.models.kimi_k3 import (
    EXACT_WIDE_SHORT_CONV_ENV,
    KimiK3DecoderLayer,
    KimiK3DeltaAttention,
    KimiK3ShortConv,
    ResidualBlocks,
    TextArgs,
    exact_wide_short_conv_enabled,
)
from mlx_lm.models.kimi_linear import ShortConv1d


def _assert_exact(test: unittest.TestCase, actual: mx.array, expected: mx.array):
    mx.eval(actual, expected)
    equal = mx.array_equal(actual, expected)
    max_abs = mx.max(mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32)))
    mx.eval(equal, max_abs)
    test.assertTrue(bool(equal.item()), f"max abs={float(max_abs.item())}")


def _make_cache(conv_state: mx.array, ssm_state: mx.array) -> ArraysCache:
    cache = ArraysCache(size=2)
    cache.cache = [conv_state, ssm_state]
    return cache


class ExactWideShortConvFlagTests(unittest.TestCase):
    def tearDown(self):
        exact_wide_short_conv_enabled.cache_clear()

    def test_flag_is_default_off_and_strictly_boolean(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            exact_wide_short_conv_enabled.cache_clear()
            self.assertFalse(exact_wide_short_conv_enabled())
        for value, expected in (("0", False), ("1", True)):
            with self.subTest(value=value):
                with mock.patch.dict(
                    os.environ,
                    {EXACT_WIDE_SHORT_CONV_ENV: value},
                    clear=True,
                ):
                    exact_wide_short_conv_enabled.cache_clear()
                    self.assertEqual(exact_wide_short_conv_enabled(), expected)
        for value in ("", "true", "yes", "2", " 1"):
            with self.subTest(value=value):
                with mock.patch.dict(
                    os.environ,
                    {EXACT_WIDE_SHORT_CONV_ENV: value},
                    clear=True,
                ):
                    exact_wide_short_conv_enabled.cache_clear()
                    with self.assertRaisesRegex(ValueError, "exactly '0' or '1'"):
                        exact_wide_short_conv_enabled()

    def test_invalid_flag_fails_before_any_convolution_path(self):
        conv = KimiK3ShortConv(channels=4, kernel_size=4)
        x = mx.zeros((1, 1, 4))
        state = mx.zeros((1, 3, 4))
        with mock.patch.dict(
            os.environ,
            {EXACT_WIDE_SHORT_CONV_ENV: "enabled"},
            clear=True,
        ):
            exact_wide_short_conv_enabled.cache_clear()
            with self.assertRaisesRegex(ValueError, EXACT_WIDE_SHORT_CONV_ENV):
                conv(x, state)

    def test_requested_width_two_fails_closed_without_metal(self):
        if mx.metal.is_available():
            self.skipTest("CPU fail-closed contract only")
        conv = KimiK3ShortConv(channels=4, kernel_size=4)
        conv.eval()
        x = mx.zeros((1, 2, 4))
        state = mx.zeros((1, 3, 4))
        with mock.patch.dict(
            os.environ,
            {EXACT_WIDE_SHORT_CONV_ENV: "1"},
            clear=True,
        ):
            exact_wide_short_conv_enabled.cache_clear()
            with self.assertRaisesRegex(ValueError, "requires populated"):
                conv(x, state)


@unittest.skipUnless(mx.metal.is_available(), "requires Metal")
class ExactWideShortConvMetalTests(unittest.TestCase):
    def setUp(self):
        self._device = mx.default_device()
        mx.set_default_device(mx.gpu)
        exact_wide_short_conv_enabled.cache_clear()

    def tearDown(self):
        exact_wide_short_conv_enabled.cache_clear()
        mx.set_default_device(self._device)

    def test_direct_width_two_and_eight_match_history_and_sequential_steps(self):
        for width, channels in ((2, 18432), (8, 512)):
            with self.subTest(width=width, channels=channels):
                mx.random.seed(20260801 + width)
                conv = KimiK3ShortConv(channels=channels, kernel_size=4)
                conv.eval()
                conv.conv.weight = conv.conv.weight.astype(mx.bfloat16)
                x = mx.random.normal((1, width, channels), dtype=mx.bfloat16)
                state = mx.random.normal((1, 3, channels), dtype=mx.bfloat16)

                with mock.patch.dict(
                    os.environ,
                    {EXACT_WIDE_SHORT_CONV_ENV: "1"},
                    clear=False,
                ):
                    exact_wide_short_conv_enabled.cache_clear()
                    direct_y, direct_state = conv(x, state)
                    history_y, history_state, history = conv(
                        x,
                        state,
                        return_state_history=True,
                    )
                    sequential_y = []
                    sequential_states = []
                    sequential_state = state
                    for position in range(width):
                        y, sequential_state = conv(
                            x[:, position : position + 1],
                            sequential_state,
                        )
                        sequential_y.append(y)
                        sequential_states.append(sequential_state)
                    sequential_y = mx.concatenate(sequential_y, axis=1)
                    sequential_history = mx.stack(sequential_states, axis=1)

                _assert_exact(self, direct_y, sequential_y)
                _assert_exact(self, direct_state, sequential_state)
                _assert_exact(self, direct_y, history_y)
                _assert_exact(self, direct_state, history_state)
                _assert_exact(self, history, sequential_history)

    def test_direct_nonhistory_gated_delta_matches_history_and_t1_steps(self):
        mx.random.seed(20260811)
        width = 2
        heads = 2
        dimension = 32
        q = mx.random.normal((1, width, heads, dimension), dtype=mx.bfloat16)
        k = mx.random.normal((1, width, heads, dimension), dtype=mx.bfloat16)
        v = mx.random.normal((1, width, heads, dimension), dtype=mx.bfloat16)
        g = mx.sigmoid(mx.random.normal((1, width, heads, dimension), dtype=mx.float32))
        beta = mx.sigmoid(mx.random.normal((1, width, heads), dtype=mx.float32))
        state = mx.random.normal((1, heads, dimension, dimension), dtype=mx.float32)

        direct_y, direct_state = gated_delta_kernel(q, k, v, g, beta, state)
        history_y, history_state, history = gated_delta_kernel(
            q,
            k,
            v,
            g,
            beta,
            state,
            return_state_history=True,
        )
        sequential_y = []
        sequential_states = []
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
            sequential_states.append(sequential_state)
        sequential_y = mx.concatenate(sequential_y, axis=1)
        sequential_history = mx.stack(sequential_states, axis=2)

        _assert_exact(self, direct_y, sequential_y)
        _assert_exact(self, direct_state, sequential_state)
        _assert_exact(self, direct_y, history_y)
        _assert_exact(self, direct_state, history_state)
        _assert_exact(self, history, sequential_history)

    def test_full_delta_attention_width_two_matches_two_t1_calls(self):
        mx.random.seed(20260821)
        hidden = 128
        heads = 2
        dimension = 32
        attention = KimiK3DeltaAttention(
            TextArgs(
                hidden_size=hidden,
                num_attention_heads=heads,
                num_key_value_heads=heads,
                rms_norm_eps=1e-5,
                linear_attn_config={
                    "kda_layers": [1],
                    "full_attn_layers": [],
                    "num_heads": heads,
                    "head_dim": dimension,
                    "short_conv_kernel_size": 4,
                    "gate_lower_bound": -5.0,
                    "use_full_rank_gate": True,
                },
            ),
            layer_idx=0,
        )
        attention.set_dtype(mx.bfloat16)
        attention.eval()
        x = mx.random.normal((1, 2, hidden), dtype=mx.bfloat16)
        conv_state = mx.random.normal((1, 3, 3 * heads * dimension), dtype=mx.bfloat16)
        ssm_state = mx.random.normal((1, heads, dimension, dimension), dtype=mx.float32)
        mx.eval(attention.parameters(), x, conv_state, ssm_state)
        direct_cache = _make_cache(conv_state, ssm_state)
        sequential_cache = _make_cache(conv_state, ssm_state)

        with mock.patch.dict(
            os.environ,
            {EXACT_WIDE_SHORT_CONV_ENV: "1"},
            clear=False,
        ):
            exact_wide_short_conv_enabled.cache_clear()
            direct_output = attention(x, cache=direct_cache)
            sequential_parts = []
            for position in range(x.shape[1]):
                sequential_parts.append(
                    attention(
                        x[:, position : position + 1],
                        cache=sequential_cache,
                    )
                )
            sequential_output = mx.concatenate(sequential_parts, axis=1)

        _assert_exact(self, direct_output, sequential_output)
        _assert_exact(self, direct_cache[0], sequential_cache[0])
        _assert_exact(self, direct_cache[1], sequential_cache[1])

    def test_full_decoder_layer_width_two_matches_two_t1_calls(self):
        mx.random.seed(20260822)
        hidden = 128
        heads = 2
        dimension = 32
        args = TextArgs(
            hidden_size=hidden,
            num_attention_heads=heads,
            num_key_value_heads=heads,
            intermediate_size=192,
            rms_norm_eps=1e-5,
            hidden_act="situ",
            activation_situ_beta=4.0,
            activation_situ_linear_beta=25.0,
            linear_attn_config={
                "kda_layers": [1],
                "full_attn_layers": [],
                "num_heads": heads,
                "head_dim": dimension,
                "short_conv_kernel_size": 4,
                "gate_lower_bound": -5.0,
                "use_full_rank_gate": True,
            },
            num_experts=8,
            moe_intermediate_size=32,
            num_experts_per_token=2,
            num_shared_experts=1,
            first_k_dense_replace=0,
            routed_expert_hidden_size=32,
            latent_moe_use_norm=True,
            attn_res_block_size=2,
        )
        layer = KimiK3DecoderLayer(args, layer_idx=0)
        layer.set_dtype(mx.bfloat16)
        layer.eval()
        x = mx.random.normal((1, 2, hidden), dtype=mx.bfloat16)
        conv_state = mx.random.normal((1, 3, 3 * heads * dimension), dtype=mx.bfloat16)
        ssm_state = mx.random.normal((1, heads, dimension, dimension), dtype=mx.float32)
        mx.eval(layer.parameters(), x, conv_state, ssm_state)
        direct_cache = _make_cache(conv_state, ssm_state)
        sequential_cache = _make_cache(conv_state, ssm_state)

        with mock.patch.dict(
            os.environ,
            {EXACT_WIDE_SHORT_CONV_ENV: "1"},
            clear=False,
        ):
            exact_wide_short_conv_enabled.cache_clear()
            direct_output, direct_blocks = layer(
                x,
                cache=direct_cache,
                blocks=ResidualBlocks(args.rms_norm_eps),
            )
            sequential_outputs = []
            sequential_raw = []
            sequential_inv_rms = []
            for position in range(x.shape[1]):
                output, blocks = layer(
                    x[:, position : position + 1],
                    cache=sequential_cache,
                    blocks=ResidualBlocks(args.rms_norm_eps),
                )
                sequential_outputs.append(output)
                sequential_raw.append(blocks.raw)
                sequential_inv_rms.append(blocks.inv_rms)
            sequential_output = mx.concatenate(sequential_outputs, axis=1)
            sequential_blocks_raw = mx.concatenate(sequential_raw, axis=2)
            sequential_blocks_inv_rms = mx.concatenate(
                sequential_inv_rms,
                axis=2,
            )

        _assert_exact(self, direct_output, sequential_output)
        _assert_exact(self, direct_cache[0], sequential_cache[0])
        _assert_exact(self, direct_cache[1], sequential_cache[1])
        _assert_exact(self, direct_blocks.raw, sequential_blocks_raw)
        _assert_exact(self, direct_blocks.inv_rms, sequential_blocks_inv_rms)

    def test_unsupported_requested_wide_contract_raises(self):
        conv = KimiK3ShortConv(channels=8, kernel_size=4)
        conv.eval()
        conv.conv.weight = conv.conv.weight.astype(mx.bfloat16)
        x = mx.zeros((1, 2, 8), dtype=mx.bfloat16)
        state = mx.zeros((1, 3, 8), dtype=mx.bfloat16)
        cases = (
            ("missing_state", None, None, None),
            ("mask", state, mx.array([[True, True]]), None),
            ("lengths", state, None, mx.array([2])),
            ("dtype", state.astype(mx.float32), None, None),
        )
        with mock.patch.dict(
            os.environ,
            {EXACT_WIDE_SHORT_CONV_ENV: "1"},
            clear=False,
        ):
            exact_wide_short_conv_enabled.cache_clear()
            for name, state_arg, mask, lengths in cases:
                with self.subTest(name=name):
                    with self.assertRaisesRegex(ValueError, "requires populated"):
                        conv(x, state_arg, mask=mask, lengths=lengths)

    def test_prefill_width_is_unchanged(self):
        mx.random.seed(20260831)
        conv = KimiK3ShortConv(channels=64, kernel_size=4)
        conv.eval()
        conv.conv.weight = conv.conv.weight.astype(mx.bfloat16)
        x = mx.random.normal((1, 128, 64), dtype=mx.bfloat16)
        state = mx.random.normal((1, 3, 64), dtype=mx.bfloat16)
        with mock.patch.dict(
            os.environ,
            {EXACT_WIDE_SHORT_CONV_ENV: "1"},
            clear=False,
        ):
            exact_wide_short_conv_enabled.cache_clear()
            actual_y, actual_state = conv(x, state)
            expected_y, expected_state = ShortConv1d.__call__(
                conv,
                x,
                state,
                None,
                None,
            )
        _assert_exact(self, actual_y, expected_y)
        _assert_exact(self, actual_state, expected_state)


if __name__ == "__main__":
    unittest.main()
