from __future__ import annotations

import gc
import os
import tempfile
import unittest
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_lm.models.kimi_k3 import KimiK3SparseMoE, TextArgs
from mlx_lm.models.kimi_k3_multibank_moe_front import (
    MULTIBANK_MOE_FRONT_ENV,
    multibank_moe_front_enabled,
)
from mlx_lm.models.kimi_k3_packed_moe_front import (
    AUTHORITATIVE_PACKED_MOE_FRONT_ENV,
    PACKED_MOE_FRONT_ENV,
    PACKED_MOE_FRONT_WIDTH8_ENV,
    AuthoritativePackedK3MoEFront,
    PackedK3MoEFront,
    PackedMoEFrontUnsupported,
    _build_authoritative_packed_front,
    _build_packed_front,
    authoritative_packed_moe_front_enabled,
    invalidate_packed_k3_moe_front,
    packed_moe_front_enabled,
    packed_moe_front_width8_enabled,
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
        packed_moe_front_enabled.cache_clear()
        packed_moe_front_width8_enabled.cache_clear()

    def tearDown(self):
        multibank_moe_front_enabled.cache_clear()
        packed_moe_front_enabled.cache_clear()
        authoritative_packed_moe_front_enabled.cache_clear()
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
        packed_moe_front_enabled.cache_clear()
        packed_moe_front_width8_enabled.cache_clear()

    def tearDown(self):
        multibank_moe_front_enabled.cache_clear()
        packed_moe_front_enabled.cache_clear()
        authoritative_packed_moe_front_enabled.cache_clear()
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


if __name__ == "__main__":
    unittest.main()
