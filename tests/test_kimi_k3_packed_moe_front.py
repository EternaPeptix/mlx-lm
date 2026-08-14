from __future__ import annotations

import gc
import json
import os
import tempfile
import unittest
from contextvars import copy_context
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_lm.models import kimi_k3_packed_moe_front as packed_front_module
from mlx_lm.models.kimi_k3 import KimiK3SparseMoE, Model, TextArgs
from mlx_lm.models.kimi_k3_multibank_moe_front import (
    MULTIBANK_MOE_FRONT_ENV,
    multibank_moe_front_enabled,
)
from mlx_lm.models.kimi_k3_packed_moe_front import (
    AUTHORITATIVE_PACKED_MOE_FRONT_ENV,
    AUTHORITATIVE_PACKED_MOE_FRONT_RECEIPT_ENV,
    AUTHORITATIVE_PACKED_MOE_FRONT_RECEIPT_SCHEMA,
    AUTHORITATIVE_PACKED_MOE_FRONT_WIDTH3_ENV,
    PACKED_MOE_FRONT_ENV,
    PACKED_MOE_FRONT_WIDTH8_ENV,
    AuthoritativePackedK3MoEFront,
    PackedK3MoEFront,
    PackedMoEFrontUnsupported,
    _build_authoritative_packed_front,
    _build_packed_front,
    _production_width3_source_layout_supported,
    abort_authoritative_packed_moe_front_receipt,
    authoritative_packed_moe_front_enabled,
    authoritative_packed_moe_front_receipt_enabled,
    authoritative_packed_moe_front_width3_enabled,
    begin_authoritative_packed_moe_front_receipt,
    finish_authoritative_packed_moe_front_receipt,
    invalidate_packed_k3_moe_front,
    maybe_authoritative_packed_k3_moe_front,
    packed_moe_front_enabled,
    packed_moe_front_width8_enabled,
    production_width3_authoritative_front_active,
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


class _PackedProjection(_QuantizedProjection):
    def __init__(self, input_dims: int, output_dims: int, pattern: int):
        self.bits = 8
        self.group_size = 64
        self.mode = "affine"
        self.weight = mx.full(
            (output_dims, input_dims * self.bits // 32),
            pattern,
            dtype=mx.uint32,
        )
        self.scales = mx.full(
            (output_dims, input_dims // self.group_size),
            0.00390625,
            dtype=mx.bfloat16,
        )
        self.biases = mx.full(
            self.scales.shape,
            -0.25,
            dtype=mx.bfloat16,
        )


class _FrontOnlySparse:
    def __init__(self, output_dims=(3072, 3072, 896, 3584)):
        projections = tuple(
            _PackedProjection(7168, size, pattern)
            for pattern, size in enumerate(output_dims, start=1)
        )
        self.shared_experts = SimpleNamespace(
            gate_proj=projections[0],
            up_proj=projections[1],
        )
        self.gate = projections[2]
        self.routed_expert_down_proj = projections[3]
        self.training = False


class _FakeAuthoritativePackedFront:
    _input_dims = 7168
    _output_dims = (3072, 3072, 896, 3584)
    _production_width3_source_layout = True

    def __init__(self, sparse_moe):
        self.modules = (
            sparse_moe.shared_experts.gate_proj,
            sparse_moe.shared_experts.up_proj,
            sparse_moe.gate,
            sparse_moe.routed_expert_down_proj,
        )

    def matches_sources(self, modules):
        return tuple(modules) == self.modules

    def __call__(self, x):
        return tuple(
            mx.zeros((1, 3, size), dtype=x.dtype) for size in self._output_dims
        )


def _small_sparse_moe(
    *,
    bits: int = 8,
    hidden_size: int = 128,
    shared_experts: int = 1,
    latent_size: int | None = 64,
) -> KimiK3SparseMoE:
    args = TextArgs(
        hidden_size=hidden_size,
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


def _front_projections(module: KimiK3SparseMoE):
    return (
        module.shared_experts.gate_proj,
        module.shared_experts.up_proj,
        module.gate,
        module.routed_expert_down_proj,
    )


class PackedK3MoEFrontTests(unittest.TestCase):
    def setUp(self):
        multibank_moe_front_enabled.cache_clear()
        authoritative_packed_moe_front_enabled.cache_clear()
        authoritative_packed_moe_front_width3_enabled.cache_clear()
        packed_moe_front_enabled.cache_clear()
        packed_moe_front_width8_enabled.cache_clear()

    def tearDown(self):
        multibank_moe_front_enabled.cache_clear()
        packed_moe_front_enabled.cache_clear()
        authoritative_packed_moe_front_enabled.cache_clear()
        authoritative_packed_moe_front_width3_enabled.cache_clear()
        packed_moe_front_width8_enabled.cache_clear()

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
        with self.assertRaisesRegex(ValueError, "one decode token"):
            packed(x)

    def test_width_eight_rows_are_bit_exact_when_independently_enabled(self):
        mx.random.seed(11)
        projections = (
            _QuantizedProjection(128, 96),
            _QuantizedProjection(128, 80),
            _QuantizedProjection(128, 24),
            _QuantizedProjection(128, 64),
        )
        packed = PackedK3MoEFront(projections)
        x = mx.random.normal((1, 8, 128)).astype(mx.bfloat16)
        expected = tuple(projection(x) for projection in projections)

        with patch.dict(os.environ, {PACKED_MOE_FRONT_WIDTH8_ENV: "1"}):
            packed_moe_front_width8_enabled.cache_clear()
            actual = packed(x)
            mx.eval(*expected, *actual)

        for want, got in zip(expected, actual, strict=True):
            self.assertTrue(bool(mx.array_equal(want, got).item()))

    def test_width_eight_gate_rejects_other_widths_and_batches(self):
        projections = tuple(_QuantizedProjection(128, 64) for _ in range(4))
        packed = PackedK3MoEFront(projections)
        with patch.dict(os.environ, {PACKED_MOE_FRONT_WIDTH8_ENV: "1"}):
            packed_moe_front_width8_enabled.cache_clear()
            for shape in ((1, 2, 128), (1, 7, 128), (2, 1, 128)):
                with self.subTest(shape=shape):
                    with self.assertRaises(PackedMoEFrontUnsupported):
                        packed(mx.zeros(shape, dtype=mx.bfloat16))

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

        module.gate.biases = module.gate.biases + mx.ones_like(module.gate.biases)
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


class AuthoritativePackedK3MoEFrontTests(unittest.TestCase):
    def setUp(self):
        multibank_moe_front_enabled.cache_clear()
        authoritative_packed_moe_front_enabled.cache_clear()
        authoritative_packed_moe_front_width3_enabled.cache_clear()
        packed_moe_front_enabled.cache_clear()
        packed_moe_front_width8_enabled.cache_clear()

    def tearDown(self):
        multibank_moe_front_enabled.cache_clear()
        packed_moe_front_enabled.cache_clear()
        authoritative_packed_moe_front_enabled.cache_clear()
        authoritative_packed_moe_front_width3_enabled.cache_clear()
        packed_moe_front_width8_enabled.cache_clear()

    def test_width_eight_authoritative_sparse_moe_is_bit_exact(self):
        mx.random.seed(30)
        module = _small_sparse_moe()
        x = mx.random.normal((1, 8, 128)).astype(mx.bfloat16)
        with patch.dict(
            os.environ,
            {
                MULTIBANK_MOE_FRONT_ENV: "0",
                AUTHORITATIVE_PACKED_MOE_FRONT_ENV: "0",
                PACKED_MOE_FRONT_WIDTH8_ENV: "0",
            },
        ):
            authoritative_packed_moe_front_enabled.cache_clear()
            packed_moe_front_width8_enabled.cache_clear()
            expected = module(x)
            mx.eval(expected)

        with patch.dict(
            os.environ,
            {
                MULTIBANK_MOE_FRONT_ENV: "0",
                AUTHORITATIVE_PACKED_MOE_FRONT_ENV: "1",
                PACKED_MOE_FRONT_WIDTH8_ENV: "1",
            },
        ):
            authoritative_packed_moe_front_enabled.cache_clear()
            packed_moe_front_width8_enabled.cache_clear()
            actual = module(x)
            mx.eval(actual)

        self.assertTrue(bool(mx.array_equal(expected, actual).item()))
        self.assertTrue(hasattr(module, "_authoritative_packed_k3_moe_front"))

    def test_width_three_full_pack_is_exact_only_at_production_input_shape(self):
        mx.random.seed(31)
        output_dims = (3072, 3072, 896, 3584)
        projections = tuple(
            _PackedProjection(7168, size, pattern)
            for pattern, size in enumerate(output_dims, start=1)
        )
        packed = AuthoritativePackedK3MoEFront(projections)
        x = mx.random.normal((1, 3, 7168)).astype(mx.bfloat16)
        expected = tuple(projection(x) for projection in projections)

        environment = {AUTHORITATIVE_PACKED_MOE_FRONT_WIDTH3_ENV: "1"}
        with patch.dict(os.environ, environment):
            authoritative_packed_moe_front_width3_enabled.cache_clear()
            actual = packed(x)
            mx.eval(*expected, *actual)

        self.assertEqual(len(actual), 4)
        for want, got in zip(expected, actual, strict=True):
            self.assertTrue(bool(mx.array_equal(want, got).item()))

        with patch.dict(
            os.environ,
            {AUTHORITATIVE_PACKED_MOE_FRONT_WIDTH3_ENV: "0"},
        ):
            authoritative_packed_moe_front_width3_enabled.cache_clear()
            with self.assertRaises(PackedMoEFrontUnsupported):
                packed(x)

        with patch.dict(os.environ, environment):
            authoritative_packed_moe_front_width3_enabled.cache_clear()
            for shape in (
                (1, 2, 7168),
                (1, 3, 128),
                (1, 3, 7167),
                (1, 4, 7168),
                (2, 3, 7168),
            ):
                with self.subTest(shape=shape):
                    with self.assertRaises(PackedMoEFrontUnsupported):
                        packed(mx.zeros(shape, dtype=mx.bfloat16))

        wrong_layout = AuthoritativePackedK3MoEFront(
            tuple(_QuantizedProjection(7168, 64) for _ in range(4))
        )
        with patch.dict(os.environ, environment):
            authoritative_packed_moe_front_width3_enabled.cache_clear()
            with self.assertRaisesRegex(
                PackedMoEFrontUnsupported,
                "exact released TP2 front layout",
            ):
                wrong_layout(x)

    def test_width_three_gate_is_strict_and_default_off(self):
        os.environ.pop(AUTHORITATIVE_PACKED_MOE_FRONT_WIDTH3_ENV, None)
        authoritative_packed_moe_front_width3_enabled.cache_clear()
        self.assertFalse(authoritative_packed_moe_front_width3_enabled())

        with patch.dict(
            os.environ,
            {AUTHORITATIVE_PACKED_MOE_FRONT_WIDTH3_ENV: "true"},
        ):
            authoritative_packed_moe_front_width3_enabled.cache_clear()
            with self.assertRaisesRegex(ValueError, "must be 0 or 1"):
                authoritative_packed_moe_front_width3_enabled()

    def test_width_three_model_eligibility_requires_exact_source_geometry(self):
        x = mx.zeros((1, 3, 7168), dtype=mx.bfloat16)
        environment = {
            AUTHORITATIVE_PACKED_MOE_FRONT_ENV: "1",
            AUTHORITATIVE_PACKED_MOE_FRONT_WIDTH3_ENV: "1",
        }
        exact = _FrontOnlySparse()
        fake = _FakeAuthoritativePackedFront(exact)
        with (
            patch.dict(os.environ, environment),
            patch(
                "mlx_lm.models.kimi_k3_packed_moe_front."
                "_build_authoritative_packed_front",
                return_value=fake,
            ) as build,
        ):
            authoritative_packed_moe_front_enabled.cache_clear()
            authoritative_packed_moe_front_width3_enabled.cache_clear()
            outputs = maybe_authoritative_packed_k3_moe_front(exact, x)

        self.assertEqual(build.call_count, 1)
        self.assertTrue(_production_width3_source_layout_supported(fake.modules))
        self.assertTrue(production_width3_authoritative_front_active(exact, x, outputs))

        for dtype in (mx.float16, mx.float32):
            with self.subTest(dtype=dtype):
                ineligible = mx.zeros((1, 3, 7168), dtype=dtype)
                with (
                    patch.dict(os.environ, environment),
                    patch(
                        "mlx_lm.models.kimi_k3_packed_moe_front."
                        "_build_authoritative_packed_front"
                    ) as dtype_build,
                ):
                    authoritative_packed_moe_front_enabled.cache_clear()
                    authoritative_packed_moe_front_width3_enabled.cache_clear()
                    self.assertIsNone(
                        maybe_authoritative_packed_k3_moe_front(exact, ineligible)
                    )
                dtype_build.assert_not_called()
                self.assertFalse(
                    production_width3_authoritative_front_active(
                        exact,
                        ineligible,
                        outputs,
                    )
                )

        for output_dims in (
            (3072, 3072, 895, 3584),
            (3072, 3072, 896, 3583),
        ):
            with self.subTest(output_dims=output_dims):
                altered = _FrontOnlySparse(output_dims=output_dims)
                with (
                    patch.dict(os.environ, environment),
                    patch(
                        "mlx_lm.models.kimi_k3_packed_moe_front."
                        "_build_authoritative_packed_front"
                    ) as altered_build,
                ):
                    authoritative_packed_moe_front_enabled.cache_clear()
                    authoritative_packed_moe_front_width3_enabled.cache_clear()
                    self.assertIsNone(
                        maybe_authoritative_packed_k3_moe_front(altered, x)
                    )
                altered_build.assert_not_called()

        with_output_bias = _FrontOnlySparse()
        with_output_bias.gate.bias = mx.zeros((896,), dtype=mx.bfloat16)
        with (
            patch.dict(os.environ, environment),
            patch(
                "mlx_lm.models.kimi_k3_packed_moe_front."
                "_build_authoritative_packed_front"
            ) as biased_build,
        ):
            authoritative_packed_moe_front_enabled.cache_clear()
            authoritative_packed_moe_front_width3_enabled.cache_clear()
            self.assertIsNone(
                maybe_authoritative_packed_k3_moe_front(with_output_bias, x)
            )
        biased_build.assert_not_called()

        # If a formerly eligible source changes layout, the next width-three
        # call must release its old authoritative parent before falling back.
        self.assertIs(exact._authoritative_packed_k3_moe_front, fake)
        exact.gate.bits = 4
        with patch.dict(os.environ, environment):
            authoritative_packed_moe_front_enabled.cache_clear()
            authoritative_packed_moe_front_width3_enabled.cache_clear()
            self.assertIsNone(maybe_authoritative_packed_k3_moe_front(exact, x))
        self.assertFalse(hasattr(exact, "_authoritative_packed_k3_moe_front"))

        missing_front = _FrontOnlySparse()
        missing_fake = _FakeAuthoritativePackedFront(missing_front)
        with (
            patch.dict(os.environ, environment),
            patch(
                "mlx_lm.models.kimi_k3_packed_moe_front."
                "_build_authoritative_packed_front",
                return_value=missing_fake,
            ),
        ):
            authoritative_packed_moe_front_enabled.cache_clear()
            authoritative_packed_moe_front_width3_enabled.cache_clear()
            self.assertIsNotNone(
                maybe_authoritative_packed_k3_moe_front(missing_front, x)
            )
            missing_front.shared_experts = None
            self.assertIsNone(maybe_authoritative_packed_k3_moe_front(missing_front, x))
        self.assertFalse(hasattr(missing_front, "_authoritative_packed_k3_moe_front"))

    def test_rows_are_bit_exact_and_original_arrays_are_not_retained(self):
        mx.random.seed(29)
        projections = (
            _QuantizedProjection(128, 96),
            _QuantizedProjection(128, 80),
            _QuantizedProjection(128, 24),
            _QuantizedProjection(128, 64),
        )
        source_arrays = tuple(
            value
            for projection in projections
            for value in (
                projection.weight,
                projection.scales,
                projection.biases,
            )
        )
        source_ids = {id(value) for value in source_arrays}
        x = mx.random.normal((1, 1, 128)).astype(mx.bfloat16)
        expected = tuple(projection(x) for projection in projections)
        mx.eval(*expected)

        packed = AuthoritativePackedK3MoEFront(projections)
        actual = packed(x)
        mx.eval(*actual)

        for want, got in zip(expected, actual, strict=True):
            self.assertTrue(bool(mx.array_equal(want, got).item()))

        installed_arrays = tuple(
            value
            for projection in projections
            for value in (
                projection.weight,
                projection.scales,
                projection.biases,
            )
        )
        self.assertTrue(source_ids.isdisjoint(map(id, installed_arrays)))
        self.assertTrue(
            source_ids.isdisjoint(
                id(value)
                for signature in packed._source_signature
                for value in signature[1:5]
                if isinstance(value, mx.array)
            )
        )
        for signature, projection in zip(
            packed._source_signature,
            projections,
            strict=True,
        ):
            self.assertIs(signature[0], projection)
            self.assertIs(signature[1], projection.weight)
            self.assertIs(signature[2], projection.scales)
            self.assertIs(signature[3], projection.biases)

    def test_full_sparse_moe_output_and_parameter_names_are_bit_exact(self):
        mx.random.seed(31)
        module = _small_sparse_moe()
        x = mx.random.normal((1, 1, 128)).astype(mx.bfloat16)
        names_before = tuple(name for name, _ in tree_flatten(module.parameters()))

        with patch.dict(
            os.environ,
            {
                MULTIBANK_MOE_FRONT_ENV: "0",
                AUTHORITATIVE_PACKED_MOE_FRONT_ENV: "0",
                PACKED_MOE_FRONT_ENV: "0",
            },
        ):
            authoritative_packed_moe_front_enabled.cache_clear()
            packed_moe_front_enabled.cache_clear()
            expected = module(x)
            mx.eval(expected)
        self.assertFalse(hasattr(module, "_authoritative_packed_k3_moe_front"))

        with patch.dict(
            os.environ,
            {
                MULTIBANK_MOE_FRONT_ENV: "0",
                AUTHORITATIVE_PACKED_MOE_FRONT_ENV: "1",
                PACKED_MOE_FRONT_ENV: "0",
            },
        ):
            authoritative_packed_moe_front_enabled.cache_clear()
            packed_moe_front_enabled.cache_clear()
            actual = module(x)
            mx.eval(actual)

        self.assertTrue(bool(mx.array_equal(expected, actual).item()))
        self.assertIsNotNone(
            getattr(module, "_authoritative_packed_k3_moe_front", None)
        )
        self.assertEqual(
            names_before,
            tuple(name for name, _ in tree_flatten(module.parameters())),
        )
        self.assertFalse(
            any(
                "_authoritative_packed" in name
                for name, _ in tree_flatten(module.parameters())
            )
        )

    def test_multi_token_stock_path_uses_installed_views_bit_exactly(self):
        mx.random.seed(37)
        module = _small_sparse_moe()
        decode_x = mx.random.normal((1, 1, 128)).astype(mx.bfloat16)
        prefill_x = mx.random.normal((1, 5, 128)).astype(mx.bfloat16)

        with patch.dict(
            os.environ,
            {
                MULTIBANK_MOE_FRONT_ENV: "0",
                AUTHORITATIVE_PACKED_MOE_FRONT_ENV: "0",
            },
        ):
            authoritative_packed_moe_front_enabled.cache_clear()
            expected = module(prefill_x)
            mx.eval(expected)

        with patch.dict(
            os.environ,
            {
                MULTIBANK_MOE_FRONT_ENV: "0",
                AUTHORITATIVE_PACKED_MOE_FRONT_ENV: "1",
            },
        ):
            authoritative_packed_moe_front_enabled.cache_clear()
            packed_decode = module(decode_x)
            mx.eval(packed_decode)
            actual = module(prefill_x)
            mx.eval(actual)

        self.assertTrue(bool(mx.array_equal(expected, actual).item()))

    def test_first_authoritative_install_inside_compiled_call_is_exact(self):
        mx.random.seed(39)
        module = _small_sparse_moe()
        x = mx.random.normal((1, 1, 128)).astype(mx.bfloat16)
        with patch.dict(
            os.environ,
            {
                MULTIBANK_MOE_FRONT_ENV: "0",
                AUTHORITATIVE_PACKED_MOE_FRONT_ENV: "0",
            },
        ):
            authoritative_packed_moe_front_enabled.cache_clear()
            expected = module(x)
            mx.eval(expected)

        compiled = mx.compile(lambda value: module(value))
        with patch.dict(
            os.environ,
            {
                MULTIBANK_MOE_FRONT_ENV: "0",
                AUTHORITATIVE_PACKED_MOE_FRONT_ENV: "1",
            },
        ):
            authoritative_packed_moe_front_enabled.cache_clear()
            actual = compiled(x)
            mx.eval(actual)

        self.assertTrue(bool(mx.array_equal(expected, actual).item()))
        self.assertTrue(hasattr(module, "_authoritative_packed_k3_moe_front"))

    def test_parameter_update_rebuilds_authoritative_pack(self):
        mx.random.seed(41)
        module = _small_sparse_moe()
        x = mx.random.normal((1, 1, 128)).astype(mx.bfloat16)

        with patch.dict(
            os.environ,
            {
                MULTIBANK_MOE_FRONT_ENV: "0",
                AUTHORITATIVE_PACKED_MOE_FRONT_ENV: "1",
            },
        ):
            authoritative_packed_moe_front_enabled.cache_clear()
            first = module(x)
            mx.eval(first)
            first_packed = module._authoritative_packed_k3_moe_front

        replacement = module.gate.biases + mx.ones_like(module.gate.biases)
        module.update({"gate": {"biases": replacement}})
        with patch.dict(
            os.environ,
            {
                MULTIBANK_MOE_FRONT_ENV: "0",
                AUTHORITATIVE_PACKED_MOE_FRONT_ENV: "0",
            },
        ):
            authoritative_packed_moe_front_enabled.cache_clear()
            expected = module(x)
            mx.eval(expected)

        with patch.dict(
            os.environ,
            {
                MULTIBANK_MOE_FRONT_ENV: "0",
                AUTHORITATIVE_PACKED_MOE_FRONT_ENV: "1",
            },
        ):
            authoritative_packed_moe_front_enabled.cache_clear()
            actual = module(x)
            mx.eval(actual)

        self.assertIsNot(
            first_packed,
            module._authoritative_packed_k3_moe_front,
        )
        self.assertTrue(bool(mx.array_equal(expected, actual).item()))
        self.assertIsNot(module.gate.biases, replacement)

    def test_authoritative_mode_evicts_an_existing_duplicate_cache(self):
        mx.random.seed(42)
        module = _small_sparse_moe()
        x = mx.random.normal((1, 1, 128)).astype(mx.bfloat16)
        with patch.dict(
            os.environ,
            {
                MULTIBANK_MOE_FRONT_ENV: "0",
                AUTHORITATIVE_PACKED_MOE_FRONT_ENV: "0",
                PACKED_MOE_FRONT_ENV: "1",
            },
        ):
            authoritative_packed_moe_front_enabled.cache_clear()
            packed_moe_front_enabled.cache_clear()
            expected = module(x)
            mx.eval(expected)
        self.assertTrue(hasattr(module, "_packed_k3_moe_front"))

        with patch.dict(
            os.environ,
            {
                MULTIBANK_MOE_FRONT_ENV: "0",
                AUTHORITATIVE_PACKED_MOE_FRONT_ENV: "1",
                PACKED_MOE_FRONT_ENV: "1",
            },
        ):
            authoritative_packed_moe_front_enabled.cache_clear()
            packed_moe_front_enabled.cache_clear()
            actual = module(x)
            mx.eval(actual)

        self.assertTrue(bool(mx.array_equal(expected, actual).item()))
        self.assertFalse(hasattr(module, "_packed_k3_moe_front"))
        self.assertTrue(hasattr(module, "_authoritative_packed_k3_moe_front"))

    def test_save_and_load_preserve_checkpoint_names_and_values(self):
        mx.random.seed(43)
        module = _small_sparse_moe()
        x = mx.random.normal((1, 1, 128)).astype(mx.bfloat16)
        with patch.dict(
            os.environ,
            {
                MULTIBANK_MOE_FRONT_ENV: "0",
                AUTHORITATIVE_PACKED_MOE_FRONT_ENV: "1",
            },
        ):
            authoritative_packed_moe_front_enabled.cache_clear()
            output = module(x)
            mx.eval(output)

        expected = dict(tree_flatten(module.parameters()))
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "weights.safetensors")
            module.save_weights(path)
            serialized = mx.load(path)
            self.assertEqual(set(expected), set(serialized))
            self.assertFalse(
                any("_authoritative_packed" in name for name in serialized)
            )
            for name, value in expected.items():
                self.assertTrue(
                    bool(mx.array_equal(value, serialized[name]).item()),
                    name,
                )

            previous = module._authoritative_packed_k3_moe_front
            module.load_weights(path)
            with patch.dict(
                os.environ,
                {
                    MULTIBANK_MOE_FRONT_ENV: "0",
                    AUTHORITATIVE_PACKED_MOE_FRONT_ENV: "1",
                },
            ):
                authoritative_packed_moe_front_enabled.cache_clear()
                reloaded_output = module(x)
                mx.eval(reloaded_output)
            self.assertIsNot(
                previous,
                module._authoritative_packed_k3_moe_front,
            )
            self.assertTrue(bool(mx.array_equal(output, reloaded_output).item()))

            restored = _small_sparse_moe()
            restored.load_weights(path)
            restored_values = dict(tree_flatten(restored.parameters()))
            for name, value in expected.items():
                self.assertTrue(
                    bool(mx.array_equal(value, restored_values[name]).item()),
                    name,
                )

    def test_model_load_invalidates_all_packs_before_parameter_replacement(self):
        mx.random.seed(45)
        module = _small_sparse_moe()
        x = mx.random.normal((1, 1, 128)).astype(mx.bfloat16)
        with patch.dict(
            os.environ,
            {
                MULTIBANK_MOE_FRONT_ENV: "0",
                AUTHORITATIVE_PACKED_MOE_FRONT_ENV: "1",
            },
        ):
            authoritative_packed_moe_front_enabled.cache_clear()
            packed_output = module(x)
            mx.eval(packed_output)
        self.assertTrue(hasattr(module, "_authoritative_packed_k3_moe_front"))
        object.__setattr__(module, "_packed_k3_moe_front", object())

        # Model.pipeline can replace non-local layers with None.  Use that
        # topology here while directly proving invalidation happens before the
        # superclass is allowed to replace any parameter arrays.
        model = Model.__new__(Model)
        layers = [None, SimpleNamespace(mlp=module)]
        object.__setattr__(
            model,
            "language_model",
            SimpleNamespace(model=SimpleNamespace(layers=layers)),
        )
        events = []

        def replace_parameters(loaded_model, file_or_weights, strict=True):
            self.assertIs(loaded_model, model)
            self.assertEqual(file_or_weights, [])
            self.assertFalse(hasattr(module, "_authoritative_packed_k3_moe_front"))
            self.assertFalse(hasattr(module, "_packed_k3_moe_front"))
            events.append(("load", strict))

        def validate_biases(observed_layers):
            self.assertIs(observed_layers, layers)
            events.append(("biases", None))

        with (
            patch.object(nn.Module, "load_weights", new=replace_parameters),
            patch(
                "mlx_lm.models.kimi_k3.elide_validated_k3_biases",
                side_effect=validate_biases,
            ),
        ):
            self.assertIs(model.load_weights([], strict=False), model)

        self.assertEqual(events, [("load", False), ("biases", None)])

    def test_invalidation_detaches_source_views_before_sharding(self):
        mx.random.seed(47)
        module = _small_sparse_moe()
        x = mx.random.normal((1, 1, 128)).astype(mx.bfloat16)
        with patch.dict(
            os.environ,
            {
                MULTIBANK_MOE_FRONT_ENV: "0",
                AUTHORITATIVE_PACKED_MOE_FRONT_ENV: "1",
            },
        ):
            authoritative_packed_moe_front_enabled.cache_clear()
            expected = module(x)
            mx.eval(expected)

        packed = module._authoritative_packed_k3_moe_front
        installed = {
            (id(projection), name): value
            for projection in _front_projections(module)
            for name in ("weight", "scales", "biases")
            if isinstance((value := projection.get(name)), mx.array)
        }
        invalidate_packed_k3_moe_front(module)
        self.assertFalse(hasattr(module, "_authoritative_packed_k3_moe_front"))
        for projection in _front_projections(module):
            for name in ("weight", "scales", "biases"):
                value = projection.get(name)
                previous = installed[(id(projection), name)]
                self.assertIsNot(value, previous)
                self.assertTrue(
                    bool(
                        mx.array_equal(
                            value.view(mx.uint8),
                            previous.view(mx.uint8),
                        ).item()
                    )
                )

        actual = module(x)
        mx.eval(actual)
        self.assertTrue(bool(mx.array_equal(expected, actual).item()))
        del packed

    def test_unsupported_layout_fails_closed_without_replacing_arrays(self):
        mx.random.seed(53)
        module = _small_sparse_moe(bits=4)
        x = mx.random.normal((1, 1, 128)).astype(mx.bfloat16)
        source_ids = tuple(
            id(value)
            for projection in _front_projections(module)
            for value in (
                projection.weight,
                projection.scales,
                projection.biases,
            )
        )

        with patch.dict(
            os.environ,
            {
                MULTIBANK_MOE_FRONT_ENV: "0",
                AUTHORITATIVE_PACKED_MOE_FRONT_ENV: "0",
            },
        ):
            authoritative_packed_moe_front_enabled.cache_clear()
            expected = module(x)
            mx.eval(expected)
        with patch.dict(
            os.environ,
            {
                MULTIBANK_MOE_FRONT_ENV: "0",
                AUTHORITATIVE_PACKED_MOE_FRONT_ENV: "1",
            },
        ):
            authoritative_packed_moe_front_enabled.cache_clear()
            actual = module(x)
            mx.eval(actual)

        self.assertTrue(bool(mx.array_equal(expected, actual).item()))
        self.assertEqual(
            source_ids,
            tuple(
                id(value)
                for projection in _front_projections(module)
                for value in (
                    projection.weight,
                    projection.scales,
                    projection.biases,
                )
            ),
        )
        self.assertIn(
            "8-bit",
            module._authoritative_packed_k3_moe_front_reason,
        )

    def test_stale_pack_that_becomes_unsupported_is_cached_once(self):
        mx.random.seed(59)
        module = _small_sparse_moe()
        x = mx.random.normal((1, 1, 128)).astype(mx.bfloat16)
        with patch.dict(
            os.environ,
            {
                MULTIBANK_MOE_FRONT_ENV: "0",
                AUTHORITATIVE_PACKED_MOE_FRONT_ENV: "1",
            },
        ):
            authoritative_packed_moe_front_enabled.cache_clear()
            initial = module(x)
            mx.eval(initial)

        replacement = _QuantizedProjection(128, 8, bits=4)
        for name in ("weight", "scales", "biases"):
            setattr(module.gate, name, getattr(replacement, name))
        module.gate.bits = replacement.bits
        module.gate.group_size = replacement.group_size
        module.gate.mode = replacement.mode

        with patch.dict(
            os.environ,
            {
                MULTIBANK_MOE_FRONT_ENV: "0",
                AUTHORITATIVE_PACKED_MOE_FRONT_ENV: "1",
            },
        ):
            authoritative_packed_moe_front_enabled.cache_clear()
            with patch(
                "mlx_lm.models.kimi_k3_packed_moe_front."
                "_build_authoritative_packed_front",
                wraps=_build_authoritative_packed_front,
            ) as build:
                first = module(x)
                mx.eval(first)
                second = module(x)
                mx.eval(second)

        self.assertEqual(build.call_count, 1)
        self.assertTrue(bool(mx.array_equal(first, second).item()))
        self.assertIn(
            "quantization layout",
            module._authoritative_packed_k3_moe_front_reason,
        )

    def test_authoritative_storage_has_one_steady_state_copy(self):
        gc.collect()
        mx.clear_cache()
        projections = tuple(_QuantizedProjection(1024, 1024) for _ in range(4))
        mx.eval(
            *(
                value
                for projection in projections
                for value in (
                    projection.weight,
                    projection.scales,
                    projection.biases,
                )
            )
        )
        gc.collect()
        mx.clear_cache()
        source_active = mx.get_active_memory()
        mx.reset_peak_memory()

        packed = AuthoritativePackedK3MoEFront(projections)
        gc.collect()
        mx.clear_cache()
        steady_active = mx.get_active_memory()
        peak_active = mx.get_peak_memory()

        # Allocator granularity and the scalar metadata kernel can add a small
        # fixed amount, but a retained packed duplicate would add ~4.5 MiB.
        tolerance = 64 * 1024
        self.assertLessEqual(steady_active - source_active, tolerance)
        self.assertGreaterEqual(
            peak_active - source_active,
            packed.packed_nbytes - tolerance,
        )


class AuthoritativePackedK3MoEFrontReceiptTests(unittest.TestCase):
    def setUp(self):
        packed_front_module._RECEIPT_STATE.set(None)
        authoritative_packed_moe_front_enabled.cache_clear()
        authoritative_packed_moe_front_width3_enabled.cache_clear()

    def tearDown(self):
        packed_front_module._RECEIPT_STATE.set(None)
        authoritative_packed_moe_front_enabled.cache_clear()
        authoritative_packed_moe_front_width3_enabled.cache_clear()

    @staticmethod
    def _candidate_environment() -> dict[str, str]:
        return {
            AUTHORITATIVE_PACKED_MOE_FRONT_RECEIPT_ENV: "1",
            AUTHORITATIVE_PACKED_MOE_FRONT_ENV: "1",
            AUTHORITATIVE_PACKED_MOE_FRONT_WIDTH3_ENV: "1",
        }

    @staticmethod
    def _control_environment() -> dict[str, str]:
        return {
            AUTHORITATIVE_PACKED_MOE_FRONT_RECEIPT_ENV: "1",
            AUTHORITATIVE_PACKED_MOE_FRONT_ENV: "0",
            AUTHORITATIVE_PACKED_MOE_FRONT_WIDTH3_ENV: "0",
        }

    def test_receipt_selector_is_strict_and_default_off(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(authoritative_packed_moe_front_receipt_enabled())
            with self.assertRaisesRegex(RuntimeError, "receipt capture is disabled"):
                begin_authoritative_packed_moe_front_receipt(7, object())

        with patch.dict(
            os.environ,
            {AUTHORITATIVE_PACKED_MOE_FRONT_RECEIPT_ENV: "true"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "must be 0 or 1"):
                authoritative_packed_moe_front_receipt_enabled()

        for authoritative, width3 in (("0", "1"), ("1", "0")):
            with self.subTest(authoritative=authoritative, width3=width3):
                with (
                    patch.dict(
                        os.environ,
                        {
                            AUTHORITATIVE_PACKED_MOE_FRONT_RECEIPT_ENV: "1",
                            AUTHORITATIVE_PACKED_MOE_FRONT_ENV: authoritative,
                            AUTHORITATIVE_PACKED_MOE_FRONT_WIDTH3_ENV: width3,
                        },
                        clear=True,
                    ),
                    self.assertRaisesRegex(ValueError, "jointly disabled or jointly"),
                ):
                    begin_authoritative_packed_moe_front_receipt(8, object())

    def test_model_scan_reads_installed_matching_parents_at_each_boundary(self):
        modules = (object(), object(), object(), object())

        class FakePacked:
            _input_dims = 7168
            _output_dims = (3072, 3072, 896, 3584)
            _production_width3_source_layout = True

            def __init__(self, matches: bool):
                self.matches = matches

            def matches_sources(self, current):
                return self.matches and tuple(current) == modules

        layers = [
            SimpleNamespace(
                mlp=SimpleNamespace(_authoritative_packed_k3_moe_front=FakePacked(True))
            ),
            SimpleNamespace(
                mlp=SimpleNamespace(
                    _authoritative_packed_k3_moe_front=FakePacked(False)
                )
            ),
            SimpleNamespace(mlp=SimpleNamespace()),
        ]
        model = SimpleNamespace(
            language_model=SimpleNamespace(
                model=SimpleNamespace(layers=layers),
            )
        )
        with (
            patch.object(
                packed_front_module,
                "AuthoritativePackedK3MoEFront",
                FakePacked,
            ),
            patch.object(packed_front_module, "_front_modules", return_value=modules),
            patch.object(
                packed_front_module,
                "_production_width3_source_layout_supported",
                return_value=True,
            ),
        ):
            self.assertEqual(
                packed_front_module._count_model_authoritative_width3_packs(
                    model,
                    expected_layers=3,
                ),
                1,
            )
            layers[1].mlp._authoritative_packed_k3_moe_front.matches = True
            self.assertEqual(
                packed_front_module._count_model_authoritative_width3_packs(
                    model,
                    expected_layers=3,
                ),
                2,
            )

    def test_model_scan_rejects_wrong_wrapper_and_sparse_layer_cardinality(self):
        with self.assertRaisesRegex(
            TypeError,
            "model.language_model.model.layers",
        ):
            packed_front_module._count_model_authoritative_width3_packs(
                object(),
                expected_layers=1,
            )

        model = SimpleNamespace(
            language_model=SimpleNamespace(
                model=SimpleNamespace(layers=(SimpleNamespace(mlp=object()),)),
            )
        )
        with self.assertRaisesRegex(
            ValueError,
            "expected 1 production sparse layers, found 0",
        ):
            packed_front_module._count_model_authoritative_width3_packs(
                model,
                expected_layers=1,
            )

    def test_model_scan_traverses_a_minimal_real_kimi_model_shape(self):
        source = _FrontOnlySparse()
        sparse_moe = KimiK3SparseMoE.__new__(KimiK3SparseMoE)
        nn.Module.__init__(sparse_moe)
        object.__setattr__(sparse_moe, "shared_experts", source.shared_experts)
        object.__setattr__(sparse_moe, "gate", source.gate)
        object.__setattr__(
            sparse_moe,
            "routed_expert_down_proj",
            source.routed_expert_down_proj,
        )
        sparse_moe.eval()
        modules = packed_front_module._front_modules(sparse_moe)
        packed = AuthoritativePackedK3MoEFront(modules)
        object.__setattr__(
            sparse_moe,
            "_authoritative_packed_k3_moe_front",
            packed,
        )
        model = Model.__new__(Model)
        nn.Module.__init__(model)
        object.__setattr__(
            model,
            "language_model",
            SimpleNamespace(
                model=SimpleNamespace(
                    layers=(SimpleNamespace(mlp=sparse_moe),),
                )
            ),
        )

        self.assertEqual(
            packed_front_module._count_model_authoritative_width3_packs(
                model,
                expected_layers=1,
            ),
            1,
        )
        source.gate.bits = 4
        with self.assertRaisesRegex(
            ValueError,
            "expected 1 production sparse layers, found 0",
        ):
            packed_front_module._count_model_authoritative_width3_packs(
                model,
                expected_layers=1,
            )

    def test_exact_receipt_schema_is_scalar_only_and_partitions_calls(self):
        sparse_moe = SimpleNamespace(training=False)
        modules = (object(), object(), object(), object())
        x = mx.zeros((1, 3, 7168), dtype=mx.bfloat16)

        class FakePacked:
            def matches_sources(self, current):
                return tuple(current) == modules

            def __call__(self, inputs):
                return tuple(
                    mx.zeros((1, int(inputs.shape[1]), size), dtype=inputs.dtype)
                    for size in (3072, 3072, 896, 3584)
                )

        with (
            patch.dict(os.environ, self._candidate_environment(), clear=True),
            patch.object(
                packed_front_module,
                "_count_model_authoritative_width3_packs",
                side_effect=(0, 1),
            ),
            patch.object(packed_front_module, "_front_modules", return_value=modules),
            patch.object(
                packed_front_module,
                "_production_width3_source_layout_supported",
                return_value=True,
            ),
            patch.object(
                packed_front_module,
                "_source_signature",
                return_value=((1,),),
            ),
            patch.object(
                packed_front_module,
                "_build_authoritative_packed_front",
                return_value=FakePacked(),
            ),
        ):
            authoritative_packed_moe_front_enabled.cache_clear()
            authoritative_packed_moe_front_width3_enabled.cache_clear()
            sequence, token = begin_authoritative_packed_moe_front_receipt(17, object())
            outputs = maybe_authoritative_packed_k3_moe_front(sparse_moe, x)
            self.assertIsNotNone(outputs)
            self.assertIsNotNone(
                maybe_authoritative_packed_k3_moe_front(
                    sparse_moe,
                    mx.zeros((1, 1, 7168), dtype=mx.bfloat16),
                )
            )
            receipt = finish_authoritative_packed_moe_front_receipt(
                sequence,
                token,
                object(),
            )

        self.assertEqual(
            set(receipt),
            {
                "schema",
                "request_sequence",
                "request_token",
                "expected_layers",
                "finalized",
                "authoritative_gate_enabled",
                "width3_gate_enabled",
                "helper_calls",
                "packed_hits",
                "packed_output_tensors",
                "lazy_installs",
                "gate_disabled_calls",
                "noncontract_calls",
                "unsupported_calls",
                "packed_dispatch_fallback_calls",
                "invalidations",
                "stale_resets",
                "pack_count_before",
                "pack_count_after",
            },
        )
        self.assertEqual(
            receipt["schema"], AUTHORITATIVE_PACKED_MOE_FRONT_RECEIPT_SCHEMA
        )
        self.assertEqual(receipt["helper_calls"], 2)
        self.assertEqual(receipt["expected_layers"], 92)
        self.assertEqual(receipt["packed_hits"], 1)
        self.assertEqual(receipt["packed_output_tensors"], 4)
        self.assertEqual(receipt["lazy_installs"], 1)
        self.assertEqual(receipt["noncontract_calls"], 1)
        self.assertEqual(receipt["pack_count_before"], 0)
        self.assertEqual(receipt["pack_count_after"], 1)
        round_trip = json.loads(json.dumps(receipt, sort_keys=True))
        for name, value in round_trip.items():
            if name == "schema":
                self.assertEqual(value, AUTHORITATIVE_PACKED_MOE_FRONT_RECEIPT_SCHEMA)
            else:
                self.assertIn(type(value), {bool, int})

    def test_control_receipt_counts_only_gate_disabled_calls(self):
        sparse_moe = SimpleNamespace(training=False)
        x = mx.zeros((1, 3, 7168), dtype=mx.bfloat16)
        with (
            patch.dict(os.environ, self._control_environment(), clear=True),
            patch.object(
                packed_front_module,
                "_count_model_authoritative_width3_packs",
                side_effect=(0, 0),
            ),
        ):
            authoritative_packed_moe_front_enabled.cache_clear()
            sequence, token = begin_authoritative_packed_moe_front_receipt(19, object())
            self.assertIsNone(maybe_authoritative_packed_k3_moe_front(sparse_moe, x))
            self.assertIsNone(maybe_authoritative_packed_k3_moe_front(sparse_moe, x))
            receipt = finish_authoritative_packed_moe_front_receipt(
                sequence,
                token,
                object(),
            )

        self.assertEqual(receipt["helper_calls"], 2)
        self.assertEqual(receipt["gate_disabled_calls"], 2)
        for name in (
            "packed_hits",
            "packed_output_tensors",
            "lazy_installs",
            "noncontract_calls",
            "unsupported_calls",
            "packed_dispatch_fallback_calls",
            "pack_count_before",
            "pack_count_after",
        ):
            self.assertEqual(receipt[name], 0)

    def test_receipt_abort_and_context_copy_do_not_leak_counters(self):
        with (
            patch.dict(os.environ, self._control_environment(), clear=True),
            patch.object(
                packed_front_module,
                "_count_model_authoritative_width3_packs",
                return_value=0,
            ),
        ):
            first_context = copy_context()
            second_context = copy_context()
            first = first_context.run(
                begin_authoritative_packed_moe_front_receipt,
                21,
                object(),
            )
            second = second_context.run(
                begin_authoritative_packed_moe_front_receipt,
                22,
                object(),
            )
            first_context.run(
                packed_front_module._increment_receipt,
                helper_calls=1,
                gate_disabled_calls=1,
            )
            first_context.run(
                abort_authoritative_packed_moe_front_receipt,
                *first,
            )
            with self.assertRaisesRegex(RuntimeError, "no packed-front receipt"):
                first_context.run(
                    finish_authoritative_packed_moe_front_receipt,
                    *first,
                    object(),
                )
            second_receipt = second_context.run(
                finish_authoritative_packed_moe_front_receipt,
                *second,
                object(),
            )

        self.assertEqual(second_receipt["request_token"], 22)
        self.assertEqual(second_receipt["helper_calls"], 0)
        self.assertNotEqual(first[0], second[0])

    def test_startup_installs_92_then_later_request_installs_zero(self):
        with (
            patch.dict(os.environ, self._candidate_environment(), clear=True),
            patch.object(
                packed_front_module,
                "_count_model_authoritative_width3_packs",
                side_effect=(0, 92, 92, 92),
            ),
        ):
            first = begin_authoritative_packed_moe_front_receipt(31, object())
            packed_front_module._increment_receipt(
                helper_calls=92,
                packed_hits=92,
                packed_output_tensors=368,
                lazy_installs=92,
            )
            startup = finish_authoritative_packed_moe_front_receipt(*first, object())

            second = begin_authoritative_packed_moe_front_receipt(32, object())
            packed_front_module._increment_receipt(
                helper_calls=92,
                packed_hits=92,
                packed_output_tensors=368,
            )
            later = finish_authoritative_packed_moe_front_receipt(*second, object())

        self.assertEqual(
            (
                startup["pack_count_before"],
                startup["pack_count_after"],
                startup["lazy_installs"],
            ),
            (0, 92, 92),
        )
        self.assertEqual(
            (
                later["pack_count_before"],
                later["pack_count_after"],
                later["lazy_installs"],
            ),
            (92, 92, 0),
        )

    def test_exact_unsupported_and_dispatch_fallback_are_distinct(self):
        sparse_moe = SimpleNamespace(training=False)
        modules = (object(), object(), object(), object())
        x = mx.zeros((1, 3, 7168), dtype=mx.bfloat16)

        class FallbackPacked:
            def __call__(self, _inputs):
                raise PackedMoEFrontUnsupported("synthetic packed fallback")

        with (
            patch.dict(os.environ, self._candidate_environment(), clear=True),
            patch.object(
                packed_front_module,
                "_count_model_authoritative_width3_packs",
                return_value=0,
            ),
            patch.object(packed_front_module, "_front_modules", return_value=modules),
            patch.object(
                packed_front_module,
                "_production_width3_source_layout_supported",
                return_value=True,
            ),
            patch.object(
                packed_front_module,
                "_source_signature",
                return_value=((1,),),
            ),
        ):
            authoritative_packed_moe_front_enabled.cache_clear()
            authoritative_packed_moe_front_width3_enabled.cache_clear()
            unsupported_handle = begin_authoritative_packed_moe_front_receipt(
                41,
                object(),
            )
            with patch.object(
                packed_front_module,
                "_build_authoritative_packed_front",
                side_effect=PackedMoEFrontUnsupported("synthetic source layout"),
            ):
                self.assertIsNone(
                    maybe_authoritative_packed_k3_moe_front(sparse_moe, x)
                )
            unsupported = finish_authoritative_packed_moe_front_receipt(
                *unsupported_handle,
                object(),
            )

            del sparse_moe._authoritative_packed_k3_moe_front
            del sparse_moe._authoritative_packed_k3_moe_front_reason
            del sparse_moe._authoritative_packed_k3_moe_front_source_signature
            fallback_handle = begin_authoritative_packed_moe_front_receipt(
                42,
                object(),
            )
            with patch.object(
                packed_front_module,
                "_build_authoritative_packed_front",
                return_value=FallbackPacked(),
            ):
                self.assertIsNone(
                    maybe_authoritative_packed_k3_moe_front(sparse_moe, x)
                )
            fallback = finish_authoritative_packed_moe_front_receipt(
                *fallback_handle,
                object(),
            )

        self.assertEqual(unsupported["unsupported_calls"], 1)
        self.assertEqual(unsupported["packed_dispatch_fallback_calls"], 0)
        self.assertEqual(fallback["unsupported_calls"], 0)
        self.assertEqual(fallback["packed_dispatch_fallback_calls"], 1)
        self.assertEqual(fallback["lazy_installs"], 1)

    def test_invalidation_and_stale_reset_side_events_are_counted_once(self):
        x = mx.zeros((1, 3, 7168), dtype=mx.bfloat16)
        modules = (object(), object(), object(), object())

        class FakePacked:
            def __call__(self, inputs):
                return tuple(
                    mx.zeros((1, int(inputs.shape[1]), size), dtype=inputs.dtype)
                    for size in (3072, 3072, 896, 3584)
                )

        with (
            patch.dict(os.environ, self._candidate_environment(), clear=True),
            patch.object(
                packed_front_module,
                "_count_model_authoritative_width3_packs",
                return_value=92,
            ),
        ):
            invalidate_handle = begin_authoritative_packed_moe_front_receipt(
                51,
                object(),
            )
            invalidate_packed_k3_moe_front(SimpleNamespace())
            invalidated = finish_authoritative_packed_moe_front_receipt(
                *invalidate_handle,
                object(),
            )

            sparse_moe = SimpleNamespace(
                training=False,
                _authoritative_packed_k3_moe_front=packed_front_module._UNSUPPORTED,
                _authoritative_packed_k3_moe_front_source_signature=((1,),),
            )
            stale_handle = begin_authoritative_packed_moe_front_receipt(52, object())
            with (
                patch.object(
                    packed_front_module,
                    "_front_modules",
                    return_value=modules,
                ),
                patch.object(
                    packed_front_module,
                    "_production_width3_source_layout_supported",
                    return_value=True,
                ),
                patch.object(
                    packed_front_module,
                    "_source_signature",
                    return_value=((2,),),
                ),
                patch.object(
                    packed_front_module,
                    "_build_authoritative_packed_front",
                    return_value=FakePacked(),
                ),
            ):
                outputs = maybe_authoritative_packed_k3_moe_front(sparse_moe, x)
            stale = finish_authoritative_packed_moe_front_receipt(
                *stale_handle,
                object(),
            )

        self.assertEqual(invalidated["invalidations"], 1)
        self.assertEqual(invalidated["stale_resets"], 0)
        self.assertEqual(invalidated["helper_calls"], 0)
        self.assertIsNotNone(outputs)
        self.assertEqual(stale["stale_resets"], 1)
        self.assertEqual(stale["invalidations"], 0)
        self.assertEqual(stale["packed_hits"], 1)
        self.assertEqual(stale["lazy_installs"], 1)


if __name__ == "__main__":
    unittest.main()
