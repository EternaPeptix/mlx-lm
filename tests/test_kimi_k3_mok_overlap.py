from __future__ import annotations

import os
import unittest
from unittest import mock

import mlx.core as mx

from mlx_lm.models.kimi_k3 import (
    MOK_PREFILL_OVERLAP_ENV,
    MOK_ROUTED_SHARED_OVERLAP_ENV,
    KimiK3SparseMoE,
    TextArgs,
    mok_prefill_overlap_enabled,
    mok_routed_shared_overlap_enabled,
)


def _small_sparse_moe() -> KimiK3SparseMoE:
    args = TextArgs(
        hidden_size=128,
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


class KimiK3MoKOverlapTests(unittest.TestCase):
    def tearDown(self):
        os.environ.pop(MOK_PREFILL_OVERLAP_ENV, None)
        os.environ.pop(MOK_ROUTED_SHARED_OVERLAP_ENV, None)

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
