from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest import mock

import mlx.core as mx

from mlx_lm.models.kimi_k3_fused_routed_up_add import (
    FUSED_ROUTED_UP_ADD_ENV,
    K3_GROUP_SIZE,
    K3_HIDDEN_SIZE,
    K3_ROUTED_LATENT_SIZE,
    fused_routed_up_add_enabled,
    maybe_fused_k3_routed_up_add,
    supports_fused_routed_up_add,
)
from mlx_lm.models.kimi_k3 import KimiK3DecoderLayer


class _Projection:
    bits = 8
    group_size = K3_GROUP_SIZE
    mode = "affine"

    def __init__(self):
        self.weight = mx.full(
            (K3_HIDDEN_SIZE, K3_ROUTED_LATENT_SIZE // 4),
            0xD3917A5C,
            dtype=mx.uint32,
        )
        scale_shape = (
            K3_HIDDEN_SIZE,
            K3_ROUTED_LATENT_SIZE // K3_GROUP_SIZE,
        )
        self.scales = mx.random.uniform(
            low=0.001,
            high=0.02,
            shape=scale_shape,
        ).astype(mx.bfloat16)
        self.biases = mx.random.uniform(
            low=-0.05,
            high=0.05,
            shape=scale_shape,
        ).astype(mx.bfloat16)

    def get(self, name):
        return getattr(self, name, None)

    def __call__(self, x):
        return mx.quantized_matmul(
            x,
            self.weight,
            scales=self.scales,
            biases=self.biases,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
        )


class _SparseMoE:
    training = False
    latent_size = K3_ROUTED_LATENT_SIZE

    def __init__(self):
        self.routed_expert_up_proj = _Projection()

    def stock(self, routed_latent, shared, residual):
        return residual + (self.routed_expert_up_proj(routed_latent) + shared)


class ContractTest(unittest.TestCase):
    def tearDown(self):
        fused_routed_up_add_enabled.cache_clear()

    def test_default_off(self):
        moe = _SparseMoE()
        routed = mx.zeros(
            (1, 1, K3_ROUTED_LATENT_SIZE),
            dtype=mx.bfloat16,
        )
        shared = mx.zeros(
            (1, 1, K3_HIDDEN_SIZE),
            dtype=mx.bfloat16,
        )
        residual = mx.zeros_like(shared)
        with mock.patch.dict(
            os.environ,
            {FUSED_ROUTED_UP_ADD_ENV: "0"},
        ):
            fused_routed_up_add_enabled.cache_clear()
            self.assertIsNone(
                maybe_fused_k3_routed_up_add(
                    moe,
                    routed,
                    shared,
                    residual,
                )
            )

    def test_training_and_projection_mismatches_fall_back(self):
        moe = _SparseMoE()
        routed = mx.zeros(
            (1, 1, K3_ROUTED_LATENT_SIZE),
            dtype=mx.bfloat16,
        )
        shared = mx.zeros(
            (1, 1, K3_HIDDEN_SIZE),
            dtype=mx.bfloat16,
        )
        residual = mx.zeros_like(shared)
        with mock.patch.dict(
            os.environ,
            {FUSED_ROUTED_UP_ADD_ENV: "1"},
        ):
            fused_routed_up_add_enabled.cache_clear()
            moe.training = True
            self.assertIsNone(
                maybe_fused_k3_routed_up_add(
                    moe,
                    routed,
                    shared,
                    residual,
                )
            )
            moe.training = False
            moe.routed_expert_up_proj.bits = 4
            self.assertIsNone(
                maybe_fused_k3_routed_up_add(
                    moe,
                    routed,
                    shared,
                    residual,
                )
            )

    def test_static_contract_rejects_prefill_before_kernel_lookup(self):
        moe = _SparseMoE()
        projection = (
            moe.routed_expert_up_proj.weight,
            moe.routed_expert_up_proj.scales,
            moe.routed_expert_up_proj.biases,
        )
        routed = mx.zeros(
            (1, 2, K3_ROUTED_LATENT_SIZE),
            dtype=mx.bfloat16,
        )
        shared = mx.zeros(
            (1, 2, K3_HIDDEN_SIZE),
            dtype=mx.bfloat16,
        )
        residual = mx.zeros_like(shared)
        with mock.patch(
            "mlx_lm.models.kimi_k3_fused_routed_up_add._kernel",
            side_effect=AssertionError("kernel lookup must not run"),
        ):
            self.assertFalse(
                supports_fused_routed_up_add(
                    routed,
                    shared,
                    residual,
                    projection,
                )
            )

    def test_decoder_does_not_add_a_consumed_residual_twice(self):
        partial = mx.ones((1, 1, 8), dtype=mx.bfloat16)
        attention = mx.full((1, 1, 8), 2, dtype=mx.bfloat16)
        expected_residual = partial + attention
        sentinel = mx.full((1, 1, 8), 7, dtype=mx.bfloat16)

        class FakeSparseMoE:
            def _call_with_optional_residual(self, mlp_input, residual):
                self.mlp_input = mlp_input
                self.residual = residual
                return sentinel, True

        fake_moe = FakeSparseMoE()
        layer = SimpleNamespace(
            use_attn_res=False,
            mlp=fake_moe,
            post_attention_layernorm=lambda value: value,
        )
        with mock.patch(
            "mlx_lm.models.kimi_k3.KimiK3SparseMoE",
            FakeSparseMoE,
        ):
            actual, blocks = KimiK3DecoderLayer._finish_attention(
                layer,
                partial,
                attention,
                None,
            )
        self.assertIs(actual, sentinel)
        self.assertIsNone(blocks)
        mx.eval(fake_moe.residual, expected_residual)
        self.assertTrue(
            bool(mx.array_equal(fake_moe.residual, expected_residual).item())
        )


@unittest.skipUnless(mx.metal.is_available(), "requires Metal")
class MetalExactnessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_device = mx.default_device()
        mx.set_default_device(mx.gpu)

    @classmethod
    def tearDownClass(cls):
        mx.set_default_device(cls.previous_device)

    def setUp(self):
        os.environ[FUSED_ROUTED_UP_ADD_ENV] = "1"
        fused_routed_up_add_enabled.cache_clear()

    def tearDown(self):
        os.environ.pop(FUSED_ROUTED_UP_ADD_ENV, None)
        fused_routed_up_add_enabled.cache_clear()

    def test_tp2_routed_up_shared_and_residual_are_bit_exact(self):
        mx.random.seed(20260730)
        moe = _SparseMoE()
        for seed in (17, 91, 607):
            with self.subTest(seed=seed):
                mx.random.seed(seed)
                routed = mx.random.normal(
                    (1, 1, K3_ROUTED_LATENT_SIZE),
                    dtype=mx.bfloat16,
                )
                shared = mx.random.normal(
                    (1, 1, K3_HIDDEN_SIZE),
                    dtype=mx.bfloat16,
                )
                residual = mx.random.normal(
                    (1, 1, K3_HIDDEN_SIZE),
                    dtype=mx.bfloat16,
                )
                expected = moe.stock(routed, shared, residual)
                actual = maybe_fused_k3_routed_up_add(
                    moe,
                    routed,
                    shared,
                    residual,
                )
                self.assertIsNotNone(actual)
                mx.eval(expected, actual)
                self.assertTrue(bool(mx.array_equal(expected, actual).item()))


if __name__ == "__main__":
    unittest.main()
