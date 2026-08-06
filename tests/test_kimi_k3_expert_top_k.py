from __future__ import annotations

import os
import unittest
from unittest import mock

import mlx.core as mx

from mlx_lm.models.kimi_k3 import (
    EXPERT_TOP_K_ENV,
    KimiK3ProjectedKVCache,
    KimiK3SparseMoE,
    TextArgs,
    _group_expert_select,
    _selected_expert_top_k,
)


def _args(*, native_top_k: int = 16) -> TextArgs:
    return TextArgs(
        hidden_size=16,
        intermediate_size=32,
        num_experts=32,
        num_experts_per_token=native_top_k,
        num_expert_group=1,
        topk_group=1,
        num_shared_experts=0,
        moe_intermediate_size=16,
        routed_expert_hidden_size=None,
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
    )


def _module(*, native_top_k: int = 16) -> KimiK3SparseMoE:
    module = KimiK3SparseMoE(_args(native_top_k=native_top_k))
    module.set_dtype(mx.bfloat16)
    module.eval()
    mx.eval(module.parameters())
    return module


def _stock_reference(module: KimiK3SparseMoE, x: mx.array, top_k: int):
    scores = module.gate(x)
    indices, weights = _group_expert_select(
        scores,
        module.e_score_correction_bias,
        top_k,
        module.args.num_expert_group,
        module.args.topk_group,
        module.args.routed_scaling_factor,
        module.args.moe_renormalize,
    )
    routed = module.switch_mlp(x, indices)
    return indices, weights, (routed * weights[..., None]).sum(axis=-2)


class TestKimiK3ExpertTopK(unittest.TestCase):
    def tearDown(self):
        os.environ.pop(EXPERT_TOP_K_ENV, None)

    def test_absent_selector_preserves_checkpoint_configuration(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_selected_expert_top_k(16), 16)
            self.assertEqual(_selected_expert_top_k(2), 2)
            self.assertEqual(_module().expert_top_k, 16)

    def test_selector_accepts_only_native_top16_and_experimental_top8(self):
        for value, expected in (("16", 16), ("8", 8)):
            with self.subTest(value=value), mock.patch.dict(
                os.environ, {EXPERT_TOP_K_ENV: value}, clear=True
            ):
                self.assertEqual(_selected_expert_top_k(16), expected)
                self.assertEqual(_module().expert_top_k, expected)

        for value in ("", "0", "7", "08", "top8", "16 "):
            with self.subTest(value=value), mock.patch.dict(
                os.environ, {EXPERT_TOP_K_ENV: value}, clear=True
            ):
                with self.assertRaisesRegex(ValueError, "exactly '16' or '8'"):
                    _selected_expert_top_k(16)

        with mock.patch.dict(
            os.environ, {EXPERT_TOP_K_ENV: "8"}, clear=True
        ), self.assertRaisesRegex(ValueError, "native num_experts_per_token of 16"):
            _module(native_top_k=2)

    def test_explicit_top16_is_bit_exact_to_default_off(self):
        mx.random.seed(20260806)
        with mock.patch.dict(os.environ, {}, clear=True):
            reference_module = _module()
        with mock.patch.dict(os.environ, {EXPERT_TOP_K_ENV: "16"}, clear=True):
            candidate_module = _module()
        candidate_module.update(reference_module.parameters())
        x = mx.random.normal((1, 3, 16), dtype=mx.bfloat16)
        reference = reference_module(x)
        candidate = candidate_module(x)
        mx.eval(reference, candidate)
        self.assertTrue(mx.array_equal(reference, candidate).item())

    def test_top8_matches_stock_k8_for_decode_q3_and_prefill_shapes(self):
        with mock.patch.dict(os.environ, {EXPERT_TOP_K_ENV: "8"}, clear=True):
            mx.random.seed(20260807)
            module = _module()
            for width in (1, 3, 512):
                with self.subTest(width=width):
                    x = mx.random.normal((1, width, 16), dtype=mx.bfloat16)
                    expected_indices, expected_weights, expected = _stock_reference(
                        module, x, 8
                    )
                    actual = module(x)
                    mx.eval(expected_indices, expected_weights, expected, actual)
                    self.assertEqual(expected_indices.shape, (1, width, 8))
                    self.assertEqual(expected_weights.shape, (1, width, 8))
                    self.assertEqual(actual.shape, (1, width, 16))
                    self.assertTrue(mx.array_equal(expected, actual).item())

    def test_top8_q3_does_not_change_projected_cache_transaction(self):
        with mock.patch.dict(os.environ, {EXPERT_TOP_K_ENV: "8"}, clear=True):
            module = _module()
            cache = KimiK3ProjectedKVCache()
            token = object()
            cache.begin_projected_transaction(token, 3)
            output = module(mx.zeros((1, 3, 16), dtype=mx.bfloat16))
            mx.eval(output)
            self.assertEqual(output.shape, (1, 3, 16))
            self.assertTrue(cache.matches_projected_transaction(token, 3))
            self.assertEqual(cache._projected_transaction_width, 3)


if __name__ == "__main__":
    unittest.main()
