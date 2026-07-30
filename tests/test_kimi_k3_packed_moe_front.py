from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_lm.models.kimi_k3 import KimiK3SparseMoE, TextArgs
from mlx_lm.models.kimi_k3_packed_moe_front import (
    PACKED_MOE_FRONT_ENV,
    PackedK3MoEFront,
    _build_packed_front,
    packed_moe_front_enabled,
)


class _QuantizedProjection:
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        *,
        bits: int = 8,
        group_size: int = 64,
    ):
        weight = mx.random.normal((output_dims, input_dims)).astype(mx.bfloat16)
        self.weight, self.scales, self.biases = mx.quantize(
            weight,
            group_size=group_size,
            bits=bits,
            mode="affine",
        )
        self.bits = bits
        self.group_size = group_size
        self.mode = "affine"

    def get(self, name: str):
        return getattr(self, name, None)

    def __call__(self, x: mx.array) -> mx.array:
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


def _small_sparse_moe(
    *,
    bits: int = 8,
    shared_experts: int = 1,
    latent_size: int | None = 64,
) -> KimiK3SparseMoE:
    args = TextArgs(
        hidden_size=128,
        intermediate_size=256,
        num_experts=8,
        num_experts_per_token=2,
        num_expert_group=1,
        topk_group=1,
        num_shared_experts=shared_experts,
        moe_intermediate_size=64,
        routed_expert_hidden_size=latent_size,
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
    )
    module = KimiK3SparseMoE(args)
    nn.quantize(module, group_size=64, bits=bits, mode="affine")
    module.eval()
    mx.eval(module.parameters())
    return module


class PackedK3MoEFrontTests(unittest.TestCase):
    def tearDown(self):
        packed_moe_front_enabled.cache_clear()

    def test_single_token_rows_are_bit_exact(self):
        mx.random.seed(7)
        projections = (
            _QuantizedProjection(128, 96),
            _QuantizedProjection(128, 80),
            _QuantizedProjection(128, 24),
            _QuantizedProjection(128, 64),
        )
        packed = PackedK3MoEFront(projections)
        x = mx.random.normal((1, 1, 128)).astype(mx.bfloat16)

        expected = tuple(projection(x) for projection in projections)
        actual = packed(x)
        mx.eval(*expected, *actual)

        self.assertEqual(len(actual), 4)
        for want, got in zip(expected, actual, strict=True):
            self.assertEqual(want.shape, got.shape)
            self.assertTrue(bool(mx.array_equal(want, got).item()))

        expected_bytes = sum(
            int(value.nbytes)
            for projection in projections
            for value in (
                projection.weight,
                projection.scales,
                projection.biases,
            )
        )
        self.assertEqual(packed.packed_nbytes, expected_bytes)

    def test_multi_token_call_is_rejected(self):
        projections = tuple(_QuantizedProjection(128, 64) for _ in range(4))
        packed = PackedK3MoEFront(projections)
        x = mx.zeros((1, 2, 128), dtype=mx.bfloat16)
        with self.assertRaisesRegex(ValueError, "decode-only"):
            packed(x)

    def test_full_sparse_moe_output_is_bit_exact(self):
        mx.random.seed(19)
        module = _small_sparse_moe()
        x = mx.random.normal((1, 1, 128)).astype(mx.bfloat16)
        parameter_names_before = tuple(
            name for name, _ in tree_flatten(module.parameters())
        )

        with patch.dict(os.environ, {PACKED_MOE_FRONT_ENV: "0"}):
            packed_moe_front_enabled.cache_clear()
            expected = module(x)
            mx.eval(expected)

        with patch.dict(os.environ, {PACKED_MOE_FRONT_ENV: "1"}):
            packed_moe_front_enabled.cache_clear()
            actual = module(x)
            mx.eval(actual)

        self.assertTrue(bool(mx.array_equal(expected, actual).item()))
        self.assertIsNotNone(getattr(module, "_packed_k3_moe_front", None))
        self.assertEqual(
            parameter_names_before,
            tuple(name for name, _ in tree_flatten(module.parameters())),
        )

    def test_multi_token_integration_stays_on_stock_path(self):
        module = _small_sparse_moe()
        x = mx.random.normal((1, 2, 128)).astype(mx.bfloat16)

        with patch.dict(os.environ, {PACKED_MOE_FRONT_ENV: "0"}):
            packed_moe_front_enabled.cache_clear()
            expected = module(x)
            mx.eval(expected)

        with patch.dict(os.environ, {PACKED_MOE_FRONT_ENV: "1"}):
            packed_moe_front_enabled.cache_clear()
            actual = module(x)
            mx.eval(actual)

        self.assertTrue(bool(mx.array_equal(expected, actual).item()))
        self.assertFalse(hasattr(module, "_packed_k3_moe_front"))

    def test_missing_shared_and_latent_projections_fall_back(self):
        module = _small_sparse_moe(shared_experts=0, latent_size=None)
        x = mx.random.normal((1, 1, 128)).astype(mx.bfloat16)

        with patch.dict(os.environ, {PACKED_MOE_FRONT_ENV: "0"}):
            packed_moe_front_enabled.cache_clear()
            expected = module(x)
            mx.eval(expected)

        with patch.dict(os.environ, {PACKED_MOE_FRONT_ENV: "1"}):
            packed_moe_front_enabled.cache_clear()
            actual = module(x)
            mx.eval(actual)

        self.assertTrue(bool(mx.array_equal(expected, actual).item()))
        self.assertFalse(hasattr(module, "_packed_k3_moe_front"))

    def test_source_mutation_rebuilds_hidden_packed_copy(self):
        module = _small_sparse_moe()
        x = mx.random.normal((1, 1, 128)).astype(mx.bfloat16)

        with patch.dict(os.environ, {PACKED_MOE_FRONT_ENV: "1"}):
            packed_moe_front_enabled.cache_clear()
            first_output = module(x)
            mx.eval(first_output)
            first_packed = module._packed_k3_moe_front

        module.gate.biases = module.gate.biases + mx.ones_like(
            module.gate.biases
        )
        with patch.dict(os.environ, {PACKED_MOE_FRONT_ENV: "0"}):
            packed_moe_front_enabled.cache_clear()
            expected = module(x)
            mx.eval(expected)

        with patch.dict(os.environ, {PACKED_MOE_FRONT_ENV: "1"}):
            packed_moe_front_enabled.cache_clear()
            actual = module(x)
            mx.eval(actual)

        self.assertIsNot(first_packed, module._packed_k3_moe_front)
        self.assertTrue(bool(mx.array_equal(expected, actual).item()))

    def test_unsupported_quantization_fails_closed(self):
        mx.random.seed(23)
        module = _small_sparse_moe(bits=4)
        x = mx.random.normal((1, 1, 128)).astype(mx.bfloat16)

        with patch.dict(os.environ, {PACKED_MOE_FRONT_ENV: "0"}):
            packed_moe_front_enabled.cache_clear()
            expected = module(x)
            mx.eval(expected)

        with patch.dict(os.environ, {PACKED_MOE_FRONT_ENV: "1"}):
            packed_moe_front_enabled.cache_clear()
            actual = module(x)
            mx.eval(actual)

        self.assertTrue(bool(mx.array_equal(expected, actual).item()))
        self.assertIn("8-bit", module._packed_k3_moe_front_reason)

    def test_unchanged_unsupported_layout_is_not_rebuilt_each_token(self):
        module = _small_sparse_moe(bits=4)
        x = mx.random.normal((1, 1, 128)).astype(mx.bfloat16)

        with patch.dict(os.environ, {PACKED_MOE_FRONT_ENV: "1"}):
            packed_moe_front_enabled.cache_clear()
            with patch(
                "mlx_lm.models.kimi_k3_packed_moe_front._build_packed_front",
                wraps=_build_packed_front,
            ) as build:
                first = module(x)
                second = module(x)
                mx.eval(first, second)

        self.assertEqual(build.call_count, 1)


if __name__ == "__main__":
    unittest.main()
