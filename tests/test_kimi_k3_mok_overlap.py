from __future__ import annotations

import os
import unittest
from unittest import mock

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.kimi_k3 import (
    MOK_PREFILL_OVERLAP_ENV,
    MOK_ROUTED_SHARED_OVERLAP_ENV,
    KimiK3SparseMoE,
    TextArgs,
    mok_prefill_overlap_enabled,
    mok_routed_shared_overlap_enabled,
)
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
    maybe_authoritative_packed_k3_moe_front,
    packed_moe_front_enabled,
    production_width3_authoritative_front_active,
)


class _PackedProjection(nn.Module):
    """Released TP2 affine8 geometry without allocating float source weights."""

    def __init__(self, input_dims: int, output_dims: int, pattern: int):
        super().__init__()
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


class _ZeroProjection(nn.Module):
    def __init__(self, output_dims: int):
        super().__init__()
        self.output_dims = output_dims

    def __call__(self, x: mx.array) -> mx.array:
        return mx.zeros((*x.shape[:-1], self.output_dims), dtype=x.dtype)


def _small_sparse_moe(
    *,
    hidden_size: int = 128,
) -> KimiK3SparseMoE:
    args = TextArgs(
        hidden_size=hidden_size,
        intermediate_size=256,
        num_experts=8,
        num_experts_per_token=2,
        num_expert_group=1,
        topk_group=1,
        num_shared_experts=1,
        moe_intermediate_size=64,
        routed_expert_hidden_size=64,
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
    )
    module = KimiK3SparseMoE(args)
    module.eval()
    mx.eval(module.parameters())
    module.sharding_group = object()
    return module


def _run(
    module: KimiK3SparseMoE,
    x: mx.array,
    *,
    overlap: bool,
    prefill_overlap: bool = False,
):
    reduction_widths = []

    def identity_all_sum(value, *, group):
        del group
        reduction_widths.append(value.shape[-1])
        return value

    module.mok_routed_shared_overlap = overlap
    module.mok_prefill_overlap = prefill_overlap
    with (
        mock.patch(
            "mlx_lm.models.kimi_k3.sum_gradients",
            return_value=lambda value: value,
        ),
        mock.patch(
            "mlx_lm.models.kimi_k3.mx.distributed.all_sum",
            side_effect=identity_all_sum,
        ),
    ):
        output = module(x)
        mx.eval(output)
    return output, reduction_widths


def _front_values(module: KimiK3SparseMoE, x: mx.array):
    return (
        module.shared_experts.gate_proj(x),
        module.shared_experts.up_proj(x),
        module.gate(x),
        module.routed_expert_down_proj(x),
    )


class KimiK3MoKOverlapTests(unittest.TestCase):
    def tearDown(self):
        os.environ.pop(AUTHORITATIVE_PACKED_MOE_FRONT_ENV, None)
        os.environ.pop(AUTHORITATIVE_PACKED_MOE_FRONT_WIDTH3_ENV, None)
        os.environ.pop(MULTIBANK_MOE_FRONT_ENV, None)
        os.environ.pop(MOK_PREFILL_OVERLAP_ENV, None)
        os.environ.pop(MOK_ROUTED_SHARED_OVERLAP_ENV, None)
        authoritative_packed_moe_front_enabled.cache_clear()
        authoritative_packed_moe_front_width3_enabled.cache_clear()
        multibank_moe_front_enabled.cache_clear()
        packed_moe_front_enabled.cache_clear()

    def test_environment_is_strict_and_default_off(self):
        os.environ.pop(MOK_ROUTED_SHARED_OVERLAP_ENV, None)
        self.assertFalse(mok_routed_shared_overlap_enabled())

        os.environ[MOK_ROUTED_SHARED_OVERLAP_ENV] = "1"
        self.assertTrue(mok_routed_shared_overlap_enabled())

        os.environ[MOK_ROUTED_SHARED_OVERLAP_ENV] = "true"
        with self.assertRaisesRegex(ValueError, "must be 0 or 1"):
            mok_routed_shared_overlap_enabled()

        os.environ.pop(MOK_PREFILL_OVERLAP_ENV, None)
        self.assertFalse(mok_prefill_overlap_enabled())
        os.environ[MOK_PREFILL_OVERLAP_ENV] = "1"
        self.assertTrue(mok_prefill_overlap_enabled())
        os.environ[MOK_PREFILL_OVERLAP_ENV] = "yes"
        with self.assertRaisesRegex(ValueError, "must be 0 or 1"):
            mok_prefill_overlap_enabled()

    def test_screened_short_widths_split_exactly_into_two_reductions(self):
        for width in (3, 4):
            with self.subTest(width=width):
                mx.random.seed(920 + width)
                module = _small_sparse_moe()
                x = mx.random.normal((1, width, 128), dtype=mx.bfloat16)

                reference, reference_widths = _run(module, x, overlap=False)
                candidate, candidate_widths = _run(module, x, overlap=True)

                self.assertEqual(reference_widths, [192])
                self.assertEqual(candidate_widths, [64, 128])
                self.assertTrue(bool(mx.all(reference == candidate).item()))

    def test_q3_full_pack_is_exact_and_retains_two_reductions(self):
        mx.random.seed(927)
        module = _small_sparse_moe(hidden_size=7168)
        x = mx.random.normal((1, 3, 7168), dtype=mx.bfloat16)

        output_dims = (3072, 3072, 896, 3584)
        projections = tuple(
            _PackedProjection(7168, size, pattern)
            for pattern, size in enumerate(output_dims, start=1)
        )
        module.shared_experts.gate_proj = projections[0]
        module.shared_experts.up_proj = projections[1]
        module.gate = projections[2]
        module.routed_expert_down_proj = projections[3]
        module.shared_experts.down_proj = _ZeroProjection(7168)
        module.routed_expert_up_proj = _ZeroProjection(7168)
        module.routed_expert_norm = None

        environment = {
            MULTIBANK_MOE_FRONT_ENV: "0",
            PACKED_MOE_FRONT_ENV: "0",
            AUTHORITATIVE_PACKED_MOE_FRONT_ENV: "0",
            AUTHORITATIVE_PACKED_MOE_FRONT_WIDTH3_ENV: "0",
        }
        with mock.patch.dict(os.environ, environment):
            multibank_moe_front_enabled.cache_clear()
            packed_moe_front_enabled.cache_clear()
            authoritative_packed_moe_front_enabled.cache_clear()
            authoritative_packed_moe_front_width3_enabled.cache_clear()
            reference_front = _front_values(module, x)
            mx.eval(*reference_front)

        environment[AUTHORITATIVE_PACKED_MOE_FRONT_ENV] = "1"
        environment[AUTHORITATIVE_PACKED_MOE_FRONT_WIDTH3_ENV] = "1"
        with (
            mock.patch.dict(os.environ, environment),
            mock.patch(
                "mlx_lm.models.kimi_k3.maybe_fused_k3_router",
                return_value=(
                    mx.zeros((1, 3, 2), dtype=mx.uint32),
                    mx.zeros((1, 3, 2), dtype=mx.bfloat16),
                ),
            ),
            mock.patch(
                "mlx_lm.models.kimi_k3.maybe_fused_k3_switch_glu_reduce",
                return_value=mx.zeros((1, 3, 3584), dtype=mx.bfloat16),
            ),
        ):
            multibank_moe_front_enabled.cache_clear()
            packed_moe_front_enabled.cache_clear()
            authoritative_packed_moe_front_enabled.cache_clear()
            authoritative_packed_moe_front_width3_enabled.cache_clear()
            packed_front = maybe_authoritative_packed_k3_moe_front(module, x)
            self.assertIsNotNone(packed_front)
            mx.eval(*packed_front)
            self.assertTrue(
                production_width3_authoritative_front_active(
                    module,
                    x,
                    packed_front,
                )
            )
            candidate, candidate_widths = _run(module, x, overlap=True)

        for expected, actual in zip(reference_front, packed_front, strict=True):
            self.assertTrue(bool(mx.array_equal(expected, actual).item()))
        self.assertEqual(candidate_widths, [3584, 7168])
        self.assertTrue(bool(mx.all(candidate == 0).item()))

    def test_other_authoritative_widths_keep_combined_reduction(self):
        for width in (1, 8):
            with self.subTest(width=width):
                mx.random.seed(927 + width)
                module = _small_sparse_moe()
                x = mx.random.normal((1, width, 128), dtype=mx.bfloat16)
                with mock.patch(
                    "mlx_lm.models.kimi_k3.maybe_authoritative_packed_k3_moe_front",
                    return_value=_front_values(module, x),
                ):
                    _, reduction_widths = _run(module, x, overlap=True)
                self.assertEqual(reduction_widths, [192])

    def test_other_q3_optimized_front_keeps_combined_reduction(self):
        mx.random.seed(939)
        module = _small_sparse_moe()
        x = mx.random.normal((1, 3, 128), dtype=mx.bfloat16)
        with mock.patch(
            "mlx_lm.models.kimi_k3.maybe_multibank_k3_moe_front",
            return_value=_front_values(module, x),
        ):
            _, reduction_widths = _run(module, x, overlap=True)
        self.assertEqual(reduction_widths, [192])

    def test_prefill_width_splits_only_with_independent_opt_in(self):
        mx.random.seed(929)
        module = _small_sparse_moe()
        x = mx.random.normal((1, 128, 128), dtype=mx.bfloat16)

        reference, reference_widths = _run(module, x, overlap=True)
        candidate, candidate_widths = _run(
            module,
            x,
            overlap=True,
            prefill_overlap=True,
        )
        self.assertEqual(reference_widths, [192])
        self.assertEqual(candidate_widths, [64, 128])
        self.assertTrue(bool(mx.all(reference == candidate).item()))

    def test_other_short_width_keeps_one_combined_reduction(self):
        mx.random.seed(931)
        module = _small_sparse_moe()
        x = mx.random.normal((1, 5, 128), dtype=mx.bfloat16)

        _, reduction_widths = _run(
            module,
            x,
            overlap=True,
            prefill_overlap=True,
        )
        self.assertEqual(reduction_widths, [192])

    def test_ordinary_decode_keeps_one_combined_reduction(self):
        mx.random.seed(937)
        module = _small_sparse_moe()
        x = mx.random.normal((1, 1, 128), dtype=mx.bfloat16)

        _, reduction_widths = _run(module, x, overlap=True)
        self.assertEqual(reduction_widths, [192])


if __name__ == "__main__":
    unittest.main()
