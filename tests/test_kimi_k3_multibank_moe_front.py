from __future__ import annotations

import os
import unittest

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_lm.models.kimi_k3 import KimiK3SparseMoE, TextArgs
from mlx_lm.models.kimi_k3_multibank_moe_front import (
    MULTIBANK_MOE_FRONT_ENV,
    _metal_available,
    maybe_multibank_k3_moe_front,
    multibank_affine8_qmv,
    multibank_moe_front_enabled,
    supports_multibank_affine8_qmv,
)


@mx.compile
def _compiled_multibank(x: mx.array, *flat_banks: mx.array):
    banks = tuple(
        tuple(flat_banks[index : index + 3]) for index in range(0, len(flat_banks), 3)
    )
    return multibank_affine8_qmv(x, banks)


class _QuantizedProjection:
    bits = 8
    group_size = 64
    mode = "affine"

    def __init__(self, input_dims: int, output_dims: int, *, bias: bool = False):
        weight = mx.random.normal((output_dims, input_dims)).astype(mx.bfloat16)
        self.weight, self.scales, self.biases = mx.quantize(
            weight,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
        )
        if bias:
            self.bias = mx.random.normal((output_dims,)).astype(mx.bfloat16)

    def get(self, name: str):
        return getattr(self, name, None)

    def __call__(self, x: mx.array) -> mx.array:
        output = mx.quantized_matmul(
            x,
            self.weight,
            scales=self.scales,
            biases=self.biases,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
        )
        bias = self.get("bias")
        return output if bias is None else output + bias


def _small_sparse_moe() -> KimiK3SparseMoE:
    args = TextArgs(
        hidden_size=512,
        intermediate_size=256,
        num_experts=8,
        num_experts_per_token=2,
        num_expert_group=1,
        topk_group=1,
        num_shared_experts=1,
        moe_intermediate_size=64,
        routed_expert_hidden_size=256,
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
    )
    module = KimiK3SparseMoE(args)
    module.set_dtype(mx.bfloat16)
    nn.quantize(module, group_size=64, bits=8, mode="affine")
    module.eval()
    mx.eval(module.parameters())
    return module


@unittest.skipUnless(_metal_available(), "requires Metal")
class MultiBankMoEFrontTests(unittest.TestCase):
    def setUp(self):
        os.environ[MULTIBANK_MOE_FRONT_ENV] = "1"
        multibank_moe_front_enabled.cache_clear()

    def tearDown(self):
        os.environ.pop(MULTIBANK_MOE_FRONT_ENV, None)
        multibank_moe_front_enabled.cache_clear()

    def test_heterogeneous_banks_are_bit_exact(self):
        mx.random.seed(7)
        projections = tuple(
            _QuantizedProjection(512, output_dims) for output_dims in (96, 80, 24, 64)
        )
        x = mx.random.normal((1, 1, 512)).astype(mx.bfloat16)
        banks = tuple(
            (projection.weight, projection.scales, projection.biases)
            for projection in projections
        )

        expected = tuple(projection(x) for projection in projections)
        actual = multibank_affine8_qmv(x, banks)
        mx.eval(*expected, *actual)

        self.assertTrue(supports_multibank_affine8_qmv(x, banks))
        for want, got in zip(expected, actual, strict=True):
            self.assertEqual(want.shape, got.shape)
            self.assertTrue(bool(mx.array_equal(want, got).item()))

    def test_output_biases_preserve_stock_order(self):
        mx.random.seed(11)
        projections = tuple(
            _QuantizedProjection(512, output_dims, bias=True)
            for output_dims in (64, 48, 16, 32)
        )
        x = mx.random.normal((1, 1, 512)).astype(mx.bfloat16)

        class _Sparse:
            training = False

        sparse = _Sparse()
        sparse.shared_experts = type(
            "_Shared",
            (),
            {
                "gate_proj": projections[0],
                "up_proj": projections[1],
            },
        )()
        sparse.gate = projections[2]
        sparse.routed_expert_down_proj = projections[3]
        expected = tuple(projection(x) for projection in projections)
        actual = maybe_multibank_k3_moe_front(sparse, x)

        self.assertIsNotNone(actual)
        mx.eval(*expected, *actual)
        for want, got in zip(expected, actual, strict=True):
            self.assertTrue(bool(mx.array_equal(want, got).item()))

    def test_compiled_graph_keeps_bank_inputs_dynamic(self):
        mx.random.seed(13)
        first_projections = tuple(
            _QuantizedProjection(512, output_dims) for output_dims in (96, 80, 24, 64)
        )
        mx.random.seed(17)
        second_projections = tuple(
            _QuantizedProjection(512, output_dims) for output_dims in (96, 80, 24, 64)
        )
        x = mx.random.normal((1, 1, 512)).astype(mx.bfloat16)

        def flat(projections):
            return tuple(
                value
                for projection in projections
                for value in (
                    projection.weight,
                    projection.scales,
                    projection.biases,
                )
            )

        first = _compiled_multibank(x, *flat(first_projections))
        second = _compiled_multibank(x, *flat(second_projections))
        expected = tuple(projection(x) for projection in second_projections)
        mx.eval(*first, *second, *expected)

        for old, want, got in zip(first, expected, second, strict=True):
            self.assertTrue(bool(mx.array_equal(want, got).item()))
            self.assertTrue(bool(mx.any(old != got).item()))

    def test_full_sparse_moe_is_bit_exact_without_weight_copy(self):
        mx.random.seed(19)
        module = _small_sparse_moe()
        x = mx.random.normal((1, 1, 512)).astype(mx.bfloat16)
        parameter_items_before = tuple(tree_flatten(module.parameters()))
        source_array_ids = tuple(id(value) for _, value in parameter_items_before)

        os.environ[MULTIBANK_MOE_FRONT_ENV] = "0"
        multibank_moe_front_enabled.cache_clear()
        expected = module(x)
        mx.eval(expected)

        os.environ[MULTIBANK_MOE_FRONT_ENV] = "1"
        multibank_moe_front_enabled.cache_clear()
        actual = module(x)
        mx.eval(actual)

        parameter_items_after = tuple(tree_flatten(module.parameters()))
        self.assertTrue(bool(mx.array_equal(expected, actual).item()))
        self.assertEqual(
            tuple(name for name, _ in parameter_items_before),
            tuple(name for name, _ in parameter_items_after),
        )
        self.assertEqual(
            source_array_ids,
            tuple(id(value) for _, value in parameter_items_after),
        )
        self.assertFalse(hasattr(module, "_packed_k3_moe_front"))
        self.assertFalse(hasattr(module, "_multibank_k3_moe_front_reason"))

    def test_non_decode_and_nonfast_shapes_fall_back(self):
        module = _small_sparse_moe()
        self.assertIsNone(
            maybe_multibank_k3_moe_front(
                module,
                mx.zeros((1, 2, 512), dtype=mx.bfloat16),
            )
        )
        projections = tuple(_QuantizedProjection(128, 32) for _ in range(4))
        x = mx.zeros((1, 1, 128), dtype=mx.bfloat16)
        banks = tuple(
            (projection.weight, projection.scales, projection.biases)
            for projection in projections
        )
        self.assertFalse(supports_multibank_affine8_qmv(x, banks))

    def test_disabled_by_default(self):
        os.environ.pop(MULTIBANK_MOE_FRONT_ENV)
        multibank_moe_front_enabled.cache_clear()
        module = _small_sparse_moe()
        x = mx.zeros((1, 1, 512), dtype=mx.bfloat16)
        self.assertIsNone(maybe_multibank_k3_moe_front(module, x))


if __name__ == "__main__":
    unittest.main()
