from __future__ import annotations

import gc
import os
import unittest
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_lm.models.kimi_k3 import KimiK3DeltaAttention, TextArgs
from mlx_lm.models.kimi_k3_packed_kda_projections import (
    PACKED_KDA_SKINNY_ENV,
    PACKED_KDA_WIDE_ENV,
    AuthoritativePackedK3KDAWide,
    invalidate_packed_k3_kda_wide,
    maybe_authoritative_packed_k3_kda_wide,
    packed_kda_skinny_enabled,
    packed_kda_wide_enabled,
)


class _QuantizedProjection:
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        *,
        bits: int = 6,
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
        self.bits = 6
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


class _Attention:
    def __init__(
        self,
        *,
        full_rank_gate: bool = True,
        bits: tuple[int, int] = (6, 6),
        output_dims: tuple[int, int] = (192, 64),
    ):
        self.training = False
        self.use_full_rank_gate = full_rank_gate
        self.qkv_proj = _QuantizedProjection(
            128,
            output_dims[0],
            bits=bits[0],
        )
        if full_rank_gate:
            self.g_proj = _QuantizedProjection(
                128,
                output_dims[1],
                bits=bits[1],
            )

    @property
    def wide_modules(self):
        return self.qkv_proj, self.g_proj


def _assert_array_equal(
    test: unittest.TestCase,
    actual: mx.array,
    expected: mx.array,
) -> None:
    mx.eval(actual, expected)
    test.assertTrue(
        bool(mx.array_equal(actual, expected).item()),
        f"arrays differ; max abs={mx.max(mx.abs(actual - expected)).item()}",
    )


class AuthoritativePackedK3KDAWideTests(unittest.TestCase):
    def setUp(self):
        self._default_device = mx.default_device()
        mx.set_default_device(mx.cpu)
        packed_kda_skinny_enabled.cache_clear()
        packed_kda_wide_enabled.cache_clear()
        mx.random.seed(20260730)

    def tearDown(self):
        packed_kda_skinny_enabled.cache_clear()
        packed_kda_wide_enabled.cache_clear()
        mx.set_default_device(self._default_device)

    def _maybe(self, attention: _Attention, x: mx.array):
        with patch.dict(os.environ, {PACKED_KDA_WIDE_ENV: "1"}):
            packed_kda_wide_enabled.cache_clear()
            return maybe_authoritative_packed_k3_kda_wide(attention, x)

    def test_pair_is_bit_exact_and_authoritative(self):
        attention = _Attention()
        x = mx.random.normal((1, 1, 128)).astype(mx.bfloat16)
        expected = tuple(module(x) for module in attention.wide_modules)
        original_ids = {
            id(value)
            for module in attention.wide_modules
            for value in (module.weight, module.scales, module.biases)
        }
        mx.eval(*expected)

        actual = self._maybe(attention, x)
        self.assertIsNotNone(actual)
        packed = attention._authoritative_packed_kda_wide
        self.assertIsInstance(packed, AuthoritativePackedK3KDAWide)
        self.assertEqual(packed.output_dims, (192, 64))
        for got, want in zip(actual, expected, strict=True):
            _assert_array_equal(self, got, want)

        installed_ids = {
            id(value)
            for module in attention.wide_modules
            for value in (module.weight, module.scales, module.biases)
        }
        self.assertTrue(original_ids.isdisjoint(installed_ids))
        expected_bytes = sum(
            int(value.nbytes)
            for module in attention.wide_modules
            for value in (module.weight, module.scales, module.biases)
        )
        self.assertEqual(packed.packed_nbytes, expected_bytes)

    def test_default_off_leaves_sources_unmodified(self):
        attention = _Attention()
        x = mx.zeros((1, 1, 128), dtype=mx.bfloat16)
        original_ids = tuple(
            id(value)
            for module in attention.wide_modules
            for value in (module.weight, module.scales, module.biases)
        )
        with patch.dict(os.environ, {PACKED_KDA_WIDE_ENV: "0"}):
            packed_kda_wide_enabled.cache_clear()
            self.assertIsNone(
                maybe_authoritative_packed_k3_kda_wide(attention, x)
            )
        self.assertFalse(hasattr(attention, "_authoritative_packed_kda_wide"))
        self.assertEqual(
            original_ids,
            tuple(
                id(value)
                for module in attention.wide_modules
                for value in (module.weight, module.scales, module.biases)
            ),
        )

    def test_multi_token_prefill_uses_authoritative_source_views(self):
        attention = _Attention()
        decode_x = mx.random.normal((1, 1, 128)).astype(mx.bfloat16)
        prefill_x = mx.random.normal((1, 17, 128)).astype(mx.bfloat16)
        expected = tuple(module(prefill_x) for module in attention.wide_modules)
        mx.eval(*expected)

        self.assertIsNotNone(self._maybe(attention, decode_x))
        self.assertIsNone(self._maybe(attention, prefill_x))
        actual = tuple(module(prefill_x) for module in attention.wide_modules)
        for got, want in zip(actual, expected, strict=True):
            _assert_array_equal(self, got, want)

    def test_unverified_topology_and_quantization_fail_closed(self):
        x = mx.zeros((1, 1, 128), dtype=mx.bfloat16)
        low_rank = _Attention(full_rank_gate=False)
        self.assertIsNone(self._maybe(low_rank, x))
        self.assertFalse(hasattr(low_rank, "_authoritative_packed_kda_wide"))

        mixed_bits = _Attention(bits=(6, 4))
        self.assertIsNone(self._maybe(mixed_bits, x))
        self.assertIn(
            "one quantization layout",
            mixed_bits._authoritative_packed_kda_wide_reason,
        )

        wrong_rows = _Attention(output_dims=(128, 64))
        self.assertIsNone(self._maybe(wrong_rows, x))
        self.assertIn(
            "exactly three times",
            wrong_rows._authoritative_packed_kda_wide_reason,
        )

    def test_source_mutation_rebuilds_pack_exactly(self):
        attention = _Attention()
        x = mx.random.normal((1, 1, 128)).astype(mx.bfloat16)
        self.assertIsNotNone(self._maybe(attention, x))
        first_packed = attention._authoritative_packed_kda_wide

        attention.g_proj.biases = attention.g_proj.biases + mx.ones_like(
            attention.g_proj.biases
        )
        expected = tuple(module(x) for module in attention.wide_modules)
        mx.eval(*expected)
        actual = self._maybe(attention, x)

        self.assertIsNot(first_packed, attention._authoritative_packed_kda_wide)
        for got, want in zip(actual, expected, strict=True):
            _assert_array_equal(self, got, want)

    def test_incompatible_mutation_detaches_views_and_falls_back(self):
        attention = _Attention()
        x = mx.random.normal((1, 1, 128)).astype(mx.bfloat16)
        self.assertIsNotNone(self._maybe(attention, x))

        attention.g_proj = _QuantizedProjection(128, 64, bits=4)
        expected = tuple(module(x) for module in attention.wide_modules)
        mx.eval(*expected)
        self.assertIsNone(self._maybe(attention, x))
        actual = tuple(module(x) for module in attention.wide_modules)
        for got, want in zip(actual, expected, strict=True):
            _assert_array_equal(self, got, want)

    def test_invalidation_detaches_source_views(self):
        attention = _Attention()
        x = mx.random.normal((1, 1, 128)).astype(mx.bfloat16)
        expected = tuple(module(x) for module in attention.wide_modules)
        mx.eval(*expected)
        self.assertIsNotNone(self._maybe(attention, x))

        invalidate_packed_k3_kda_wide(attention)
        self.assertFalse(hasattr(attention, "_authoritative_packed_kda_wide"))
        actual = tuple(module(x) for module in attention.wide_modules)
        for got, want in zip(actual, expected, strict=True):
            _assert_array_equal(self, got, want)

    def test_real_k3_module_selects_qkv_and_full_rank_gate(self):
        args = TextArgs(
            hidden_size=128,
            num_attention_heads=2,
            num_key_value_heads=2,
            linear_attn_config={
                "num_heads": 2,
                "head_dim": 64,
                "short_conv_kernel_size": 4,
                "gate_lower_bound": -5.0,
                "use_full_rank_gate": True,
            },
        )
        attention = KimiK3DeltaAttention(args, layer_idx=1)
        for name in ("qkv_proj", "g_proj"):
            setattr(
                attention,
                name,
                nn.QuantizedLinear.from_linear(
                    getattr(attention, name),
                    group_size=64,
                    bits=6,
                    mode="affine",
                ),
            )
        attention.eval()
        parameter_names = tuple(
            name for name, _ in tree_flatten(attention.parameters())
        )
        x = mx.random.normal((1, 1, 128)).astype(mx.bfloat16)
        expected = (attention.qkv_proj(x), attention.g_proj(x))
        mx.eval(*expected)

        actual = self._maybe(attention, x)
        self.assertIsNotNone(actual)
        self.assertEqual(
            attention._authoritative_packed_kda_wide.output_dims,
            (384, 128),
        )
        self.assertEqual(
            parameter_names,
            tuple(name for name, _ in tree_flatten(attention.parameters())),
        )
        self.assertFalse(
            any(
                "_authoritative_packed" in name
                for name, _ in tree_flatten(attention.parameters())
            )
        )
        for got, want in zip(actual, expected, strict=True):
            _assert_array_equal(self, got, want)

    @unittest.skipUnless(mx.metal.is_available(), "storage accounting requires Metal")
    def test_real_tp2_geometry_has_one_steady_state_copy(self):
        mx.set_default_device(mx.gpu)
        gc.collect()
        mx.clear_cache()
        projections = (
            _PackedProjection(7168, 18432, 0x12345678),
            _PackedProjection(7168, 6144, 0x89ABCDEF),
        )
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

        packed = AuthoritativePackedK3KDAWide(projections)
        gc.collect()
        mx.clear_cache()
        steady_active = mx.get_active_memory()
        peak_active = mx.get_peak_memory()

        tolerance = 2 * 1024 * 1024
        self.assertEqual(packed.packed_nbytes, 143_130_624)
        self.assertLessEqual(steady_active - source_active, tolerance)
        self.assertGreaterEqual(
            peak_active - source_active,
            packed.packed_nbytes - tolerance,
        )

    @unittest.skipUnless(mx.metal.is_available(), "full KDA decode requires Metal")
    def test_compiled_decode_is_exact_with_skinny_and_wide_packs(self):
        mx.set_default_device(mx.gpu)
        args = TextArgs(
            hidden_size=128,
            num_attention_heads=2,
            num_key_value_heads=2,
            linear_attn_config={
                "num_heads": 2,
                "head_dim": 64,
                "short_conv_kernel_size": 4,
                "gate_lower_bound": -5.0,
                "use_full_rank_gate": True,
            },
        )
        attention = KimiK3DeltaAttention(args, layer_idx=1)
        for name in ("qkv_proj", "g_proj", "f_a_proj", "b_proj"):
            setattr(
                attention,
                name,
                nn.QuantizedLinear.from_linear(
                    getattr(attention, name),
                    group_size=64,
                    bits=6,
                    mode="affine",
                ),
            )
        attention.eval()
        mx.eval(attention.parameters())

        x = mx.random.normal((1, 1, 128), dtype=mx.float32)
        conv_state = mx.random.normal((1, 3, 384), dtype=mx.float32)
        ssm_state = mx.random.normal((1, 2, 64, 64), dtype=mx.float32)
        stock = mx.compile(
            lambda value, conv, ssm: attention._decode_core(value, conv, ssm)
        )
        with patch.dict(
            os.environ,
            {
                PACKED_KDA_SKINNY_ENV: "0",
                PACKED_KDA_WIDE_ENV: "0",
            },
        ):
            packed_kda_skinny_enabled.cache_clear()
            packed_kda_wide_enabled.cache_clear()
            expected = stock(x, conv_state, ssm_state)
            mx.eval(*expected)

        candidate = mx.compile(
            lambda value, conv, ssm: attention._decode_core(value, conv, ssm)
        )
        with patch.dict(
            os.environ,
            {
                PACKED_KDA_SKINNY_ENV: "1",
                PACKED_KDA_WIDE_ENV: "1",
            },
        ):
            packed_kda_skinny_enabled.cache_clear()
            packed_kda_wide_enabled.cache_clear()
            actual = candidate(x, conv_state, ssm_state)
            mx.eval(*actual)

        self.assertTrue(hasattr(attention, "_authoritative_packed_kda_skinny"))
        self.assertTrue(hasattr(attention, "_authoritative_packed_kda_wide"))
        for got, want in zip(actual, expected, strict=True):
            _assert_array_equal(self, got, want)


if __name__ == "__main__":
    unittest.main()
