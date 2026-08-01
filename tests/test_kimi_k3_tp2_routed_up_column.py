from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest import mock

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.kimi_k3_fused_routed_up_add import (
    K3_BITS,
    K3_GROUP_SIZE,
    K3_HIDDEN_SIZE,
    K3_ROUTED_LATENT_SIZE,
    K3_TP2_LOCAL_HIDDEN_SIZE,
)
from mlx_lm.models.kimi_k3_tp2_routed_up_column import (
    TP2_ROUTED_UP_COLUMN_ENV,
    RoutedUpColumnCost,
    _CONFIGURED_ATTR,
    _projection_parameters,
    all_gather_last_dimension,
    configure_k3_tp2_routed_up_column,
    maybe_k3_tp2_routed_up_column,
    ordered_local_routed_up_add,
    tp2_routed_up_column_enabled,
)


class _Group:
    def __init__(self, rank: int, size: int = 2):
        self._rank = rank
        self._size = size

    def rank(self):
        return self._rank

    def size(self):
        return self._size


class _Projection:
    bits = K3_BITS
    group_size = K3_GROUP_SIZE
    mode = "affine"

    def __init__(self, weight, scales, biases):
        self.weight = weight
        self.scales = scales
        self.biases = biases

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


class _ModuleProjection(nn.Module):
    bits = K3_BITS
    group_size = K3_GROUP_SIZE
    mode = "affine"

    def __init__(self, weight, scales, biases):
        super().__init__()
        self.weight = weight
        self.scales = scales
        self.biases = biases

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


class ContractTest(unittest.TestCase):
    def tearDown(self):
        os.environ.pop(TP2_ROUTED_UP_COLUMN_ENV, None)

    def test_default_off_and_malformed_flag_rejected(self):
        self.assertFalse(tp2_routed_up_column_enabled())
        os.environ[TP2_ROUTED_UP_COLUMN_ENV] = "yes"
        with self.assertRaisesRegex(ValueError, "must be exactly"):
            tp2_routed_up_column_enabled()

    def test_affine2_is_rejected_because_deployment_projection_is_affine8(self):
        projection = SimpleNamespace(
            bits=2,
            group_size=K3_GROUP_SIZE,
            mode="affine",
        )
        with self.assertRaisesRegex(ValueError, "affine8/group64"):
            _projection_parameters(projection, output_rows=K3_HIDDEN_SIZE)

    def test_all_gather_rejects_non_tp2(self):
        local = mx.zeros(
            (1, 1, K3_TP2_LOCAL_HIDDEN_SIZE),
            dtype=mx.bfloat16,
        )
        with self.assertRaisesRegex(ValueError, "exactly TP2"):
            all_gather_last_dimension(local, _Group(0, size=1))

    def test_configuration_rejects_unsupported_world_before_weight_access(self):
        os.environ[TP2_ROUTED_UP_COLUMN_ENV] = "1"
        with self.assertRaisesRegex(ValueError, "exactly TP2"):
            configure_k3_tp2_routed_up_column(
                SimpleNamespace(),
                _Group(0, size=4),
            )

    def test_cost_model_and_break_even(self):
        cost = RoutedUpColumnCost(
            full_projection_ms=0.30,
            half_projection_ms=0.17,
            all_gather_ms=0.05,
        )
        self.assertAlmostEqual(cost.break_even_all_gather_ms, 0.13)
        self.assertAlmostEqual(cost.saved_ms_per_layer, 0.08)
        self.assertAlmostEqual(cost.saved_ms_per_token, 7.36)
        self.assertAlmostEqual(
            cost.projected_tokens_per_second(69.893),
            1000.0 / (69.893 - 7.36),
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

    def tearDown(self):
        os.environ.pop(TP2_ROUTED_UP_COLUMN_ENV, None)

    def test_configure_and_runtime_own_only_the_matching_rank_half(self):
        mx.random.seed(20260801)
        full_weight = mx.full(
            (
                K3_HIDDEN_SIZE,
                K3_ROUTED_LATENT_SIZE * K3_BITS // 32,
            ),
            0xD3917A5C,
            dtype=mx.uint32,
        )
        full_scales = mx.random.uniform(
            0.001,
            0.02,
            (K3_HIDDEN_SIZE, K3_ROUTED_LATENT_SIZE // K3_GROUP_SIZE),
        ).astype(mx.bfloat16)
        full_biases = mx.random.uniform(
            -0.05,
            0.05,
            full_scales.shape,
        ).astype(mx.bfloat16)
        full_projection = _Projection(full_weight, full_scales, full_biases)
        moe = SimpleNamespace(
            training=False,
            latent_size=K3_ROUTED_LATENT_SIZE,
            routed_expert_up_proj=_ModuleProjection(
                full_weight,
                full_scales,
                full_biases,
            ),
        )
        group = _Group(rank=1)
        os.environ[TP2_ROUTED_UP_COLUMN_ENV] = "1"
        self.assertTrue(configure_k3_tp2_routed_up_column(moe, group))
        self.assertIs(getattr(moe, _CONFIGURED_ATTR), group)
        local_parameters = _projection_parameters(
            moe.routed_expert_up_proj,
            output_rows=K3_TP2_LOCAL_HIDDEN_SIZE,
        )
        mx.eval(*local_parameters)
        self.assertTrue(
            bool(
                mx.array_equal(
                    local_parameters[0],
                    full_weight[K3_TP2_LOCAL_HIDDEN_SIZE :],
                ).item()
            )
        )

        routed = mx.random.normal(
            (1, 1, K3_ROUTED_LATENT_SIZE), dtype=mx.bfloat16
        )
        shared = mx.random.normal(
            (1, 1, K3_HIDDEN_SIZE), dtype=mx.bfloat16
        )
        residual = mx.random.normal(
            (1, 1, K3_HIDDEN_SIZE), dtype=mx.bfloat16
        )
        expected = residual + (full_projection(routed) + shared)

        def verify_local(local_hidden, actual_group):
            self.assertIs(actual_group, group)
            mx.eval(expected, local_hidden)
            self.assertTrue(
                bool(
                    mx.array_equal(
                        local_hidden,
                        expected[..., K3_TP2_LOCAL_HIDDEN_SIZE :],
                    ).item()
                )
            )
            return expected

        with mock.patch(
            "mlx_lm.models.kimi_k3_tp2_routed_up_column."
            "all_gather_last_dimension",
            side_effect=verify_local,
        ):
            actual = maybe_k3_tp2_routed_up_column(
                moe,
                routed,
                shared,
                residual,
            )
        mx.eval(expected, actual)
        self.assertTrue(bool(mx.array_equal(expected, actual).item()))

        with self.assertRaisesRegex(ValueError, "widths 3584 and 7168"):
            maybe_k3_tp2_routed_up_column(
                moe,
                routed[..., :-1],
                shared,
                residual,
            )

    def test_synthetic_two_rank_output_rows_are_bit_exact(self):
        for seed in (29, 113, 607):
            with self.subTest(seed=seed):
                mx.random.seed(seed)
                full_weight = mx.random.randint(
                    0,
                    2**31,
                    (
                        K3_HIDDEN_SIZE,
                        K3_ROUTED_LATENT_SIZE * K3_BITS // 32,
                    ),
                    dtype=mx.uint32,
                )
                full_scales = mx.random.uniform(
                    low=0.001,
                    high=0.02,
                    shape=(
                        K3_HIDDEN_SIZE,
                        K3_ROUTED_LATENT_SIZE // K3_GROUP_SIZE,
                    ),
                ).astype(mx.bfloat16)
                full_biases = mx.random.uniform(
                    low=-0.05,
                    high=0.05,
                    shape=full_scales.shape,
                ).astype(mx.bfloat16)
                projection = _Projection(full_weight, full_scales, full_biases)
                routed = mx.random.normal(
                    (1, 1, K3_ROUTED_LATENT_SIZE), dtype=mx.bfloat16
                )
                shared = mx.random.normal(
                    (1, 1, K3_HIDDEN_SIZE), dtype=mx.bfloat16
                )
                residual = mx.random.normal(
                    (1, 1, K3_HIDDEN_SIZE), dtype=mx.bfloat16
                )

                expected = residual + (projection(routed) + shared)
                local_outputs = []
                for rank in (0, 1):
                    start = rank * K3_TP2_LOCAL_HIDDEN_SIZE
                    end = start + K3_TP2_LOCAL_HIDDEN_SIZE
                    local_projection = _Projection(
                        full_weight[start:end],
                        full_scales[start:end],
                        full_biases[start:end],
                    )
                    parameters = _projection_parameters(
                        local_projection,
                        output_rows=K3_TP2_LOCAL_HIDDEN_SIZE,
                    )
                    local_outputs.append(
                        ordered_local_routed_up_add(
                            local_projection,
                            routed,
                            shared[..., start:end],
                            residual[..., start:end],
                            parameters,
                        )
                    )

                rank_major = mx.concatenate(
                    [mx.moveaxis(value, -1, 0) for value in local_outputs],
                    axis=0,
                )

                def synthetic_all_gather(_value, *, group):
                    self.assertEqual(group.size(), 2)
                    return rank_major

                actual = all_gather_last_dimension(
                    local_outputs[0],
                    _Group(0),
                    collective=synthetic_all_gather,
                )
                mx.eval(expected, actual)
                self.assertTrue(bool(mx.array_equal(expected, actual).item()))


if __name__ == "__main__":
    unittest.main()
