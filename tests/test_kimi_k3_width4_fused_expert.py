from __future__ import annotations

import os
import unittest
from unittest import mock

import mlx.core as mx

from mlx_lm.models import kimi_k3_fused_expert as fused_expert_adapter
from mlx_lm.models.kimi_k3_derived_bias import (
    DERIVE_AFFINE2_BIAS_ENV,
    derive_affine2_bias_enabled,
    derived_affine2_biases,
)
from mlx_lm.models.kimi_k3_fused_expert import (
    FUSED_DOWN_REDUCE_ENV,
    FUSED_EXPERT_ENV,
    FUSED_EXPERT_WIDTH4_ENV,
    fused_k3_down_reduce_enabled,
    fused_k3_expert_width4_enabled,
    fused_k3_experts_enabled,
    maybe_fused_k3_switch_glu,
    maybe_fused_k3_switch_glu_reduce,
)
from mlx_lm.models.kimi_k3_width4_fused_expert import (
    K3_EXPERTS,
    K3_HIDDEN,
    K3_INTERMEDIATE,
    K3_TOP_K,
    K3_WIDTH4,
    _metal_available,
    supports_width4_down,
    supports_width4_down_reduce_projection,
    supports_width4_native_down_projection,
    supports_width4_switch_situ,
    width4_switch_glu,
    width4_switch_glu_native_down,
    width4_switch_glu_reduce,
)


class _Activation:
    beta = 4.0
    linear_beta = 25.0


class _Projection:
    bits = 2
    group_size = 128
    mode = "affine"

    def __init__(self, input_width: int, output_width: int, salt: int):
        # Expert-dependent broadcast values exercise the full 896-entry address
        # range without large construction temporaries.  Metal's contiguous
        # input contract materializes the broadcast before dispatch.
        expert = mx.arange(K3_EXPERTS, dtype=mx.uint32).reshape(K3_EXPERTS, 1, 1)
        packed = expert * mx.array(0x9E3779B9, dtype=mx.uint32)
        packed = mx.bitwise_xor(packed, mx.array(salt, dtype=mx.uint32))
        self.weight = mx.contiguous(
            mx.broadcast_to(
                packed,
                (K3_EXPERTS, output_width, input_width // 16),
            )
        )
        scale = (
            mx.array(1 / 128, dtype=mx.float32)
            + (expert % 13).astype(mx.float32)
            * mx.array(1 / 4096, dtype=mx.float32)
        ).astype(mx.bfloat16)
        self.scales = mx.contiguous(
            mx.broadcast_to(
                scale,
                (K3_EXPERTS, output_width, input_width // 128),
            )
        )
        self.biases = derived_affine2_biases(self.scales)

    def __contains__(self, name):
        return False

    def __getitem__(self, name):
        return getattr(self, name)

    def get(self, name):
        return getattr(self, name, None)

    def parts(self):
        return self.weight, self.scales, self.biases


class _Switch:
    training = False
    activation = _Activation()

    def __init__(self):
        self.up_proj = _Projection(K3_HIDDEN, K3_INTERMEDIATE, 0x13579BDF)
        self.gate_proj = _Projection(K3_HIDDEN, K3_INTERMEDIATE, 0x2468ACE0)
        self.down_proj = _Projection(K3_INTERMEDIATE, K3_HIDDEN, 0x55AA55AA)


def _qmm(x, indices, projection):
    return mx.gather_qmm(
        x,
        projection.weight,
        projection.scales,
        None,
        rhs_indices=indices,
        transpose=True,
        group_size=128,
        bits=2,
        mode="affine2",
    )


def _stock(switch, x, indices):
    expanded = mx.expand_dims(x, (-2, -3))
    up = _qmm(expanded, indices, switch.up_proj).astype(mx.float32)
    gate = _qmm(expanded, indices, switch.gate_proj).astype(mx.float32)
    activated = (
        4.0
        * mx.tanh(gate / 4.0)
        * mx.sigmoid(gate)
        * (25.0 * mx.tanh(up / 25.0))
    ).astype(mx.bfloat16)
    return _qmm(activated, indices, switch.down_proj).squeeze(-2)


class Width4SelectorTest(unittest.TestCase):
    def tearDown(self):
        os.environ.pop(FUSED_EXPERT_WIDTH4_ENV, None)
        fused_k3_expert_width4_enabled.cache_clear()

    def test_default_off_and_strict_values(self):
        os.environ.pop(FUSED_EXPERT_WIDTH4_ENV, None)
        fused_k3_expert_width4_enabled.cache_clear()
        self.assertFalse(fused_k3_expert_width4_enabled())
        for value, expected in (("0", False), ("1", True)):
            os.environ[FUSED_EXPERT_WIDTH4_ENV] = value
            fused_k3_expert_width4_enabled.cache_clear()
            self.assertEqual(fused_k3_expert_width4_enabled(), expected)

    def test_malformed_selector_fails_closed(self):
        for value in ("", "true", "01", "2", "-1"):
            with self.subTest(value=value):
                os.environ[FUSED_EXPERT_WIDTH4_ENV] = value
                fused_k3_expert_width4_enabled.cache_clear()
                with self.assertRaisesRegex(ValueError, "must be exactly"):
                    fused_k3_expert_width4_enabled()


class _ArrayStub:
    """Shape-only stand-in used to test adapter routing without Metal."""

    def __init__(self, shape):
        self.shape = shape
        self.ndim = len(shape)


class _ProjectionStub:
    bits = 2
    group_size = 128
    mode = "affine"

    def __init__(self, shape):
        self.weight = _ArrayStub(shape)
        self.scales = _ArrayStub(shape)
        self.biases = _ArrayStub(shape)

    def __contains__(self, name):
        return False

    def __getitem__(self, name):
        return getattr(self, name)

    def get(self, name):
        return getattr(self, name, None)


class _SwitchStub:
    training = False
    activation = _Activation()

    def __init__(self):
        self.up_proj = _ProjectionStub((K3_EXPERTS, K3_INTERMEDIATE, 1))
        self.gate_proj = _ProjectionStub((K3_EXPERTS, K3_INTERMEDIATE, 1))
        self.down_proj = _ProjectionStub((K3_EXPERTS, K3_HIDDEN, 1))


class Width4AdapterRoutingTest(unittest.TestCase):
    """Exercise fail-closed model routing without allocating Metal banks."""

    def setUp(self):
        self.switch = _SwitchStub()
        self.x = _ArrayStub((1, K3_WIDTH4, K3_HIDDEN))
        self.indices = _ArrayStub((1, K3_WIDTH4, K3_TOP_K))
        self.router_weights = _ArrayStub((1, K3_WIDTH4, K3_TOP_K))

    def tearDown(self):
        for name in (
            DERIVE_AFFINE2_BIAS_ENV,
            FUSED_DOWN_REDUCE_ENV,
            FUSED_EXPERT_ENV,
            FUSED_EXPERT_WIDTH4_ENV,
        ):
            os.environ.pop(name, None)
        derive_affine2_bias_enabled.cache_clear()
        fused_k3_down_reduce_enabled.cache_clear()
        fused_k3_expert_width4_enabled.cache_clear()
        fused_k3_experts_enabled.cache_clear()

    def _enable_all(self):
        os.environ[DERIVE_AFFINE2_BIAS_ENV] = "1"
        os.environ[FUSED_DOWN_REDUCE_ENV] = "1"
        os.environ[FUSED_EXPERT_ENV] = "1"
        os.environ[FUSED_EXPERT_WIDTH4_ENV] = "1"
        derive_affine2_bias_enabled.cache_clear()
        fused_k3_down_reduce_enabled.cache_clear()
        fused_k3_expert_width4_enabled.cache_clear()
        fused_k3_experts_enabled.cache_clear()

    def _adapter_patches(self):
        return (
            mock.patch.object(
                fused_expert_adapter,
                "supports_width4_switch_situ",
                return_value=True,
            ),
            mock.patch.object(
                fused_expert_adapter,
                "supports_width4_down_reduce_projection",
                return_value=True,
            ),
            mock.patch.object(
                fused_expert_adapter,
                "_all_projections_support_derived_bias",
                return_value=True,
            ),
        )

    def test_selector_chain_is_default_off_and_exact(self):
        sentinel = object()
        front_patch, down_patch, bias_patch = self._adapter_patches()
        with (
            front_patch,
            down_patch,
            bias_patch,
            mock.patch.object(
                fused_expert_adapter,
                "_compiled_width4_switch_glu_reduce_all_derived",
                return_value=sentinel,
            ) as dispatch,
        ):
            # Every selector is independently default-off.
            self.assertIsNone(
                maybe_fused_k3_switch_glu_reduce(
                    self.switch, self.x, self.indices, self.router_weights
                )
            )
            dispatch.assert_not_called()

            self._enable_all()
            os.environ[FUSED_DOWN_REDUCE_ENV] = "true"
            fused_k3_down_reduce_enabled.cache_clear()
            self.assertIsNone(
                maybe_fused_k3_switch_glu_reduce(
                    self.switch, self.x, self.indices, self.router_weights
                )
            )
            dispatch.assert_not_called()

            os.environ[FUSED_DOWN_REDUCE_ENV] = "1"
            os.environ[FUSED_EXPERT_ENV] = "true"
            fused_k3_down_reduce_enabled.cache_clear()
            fused_k3_experts_enabled.cache_clear()
            self.assertIsNone(
                maybe_fused_k3_switch_glu_reduce(
                    self.switch, self.x, self.indices, self.router_weights
                )
            )
            dispatch.assert_not_called()

            os.environ[FUSED_EXPERT_ENV] = "1"
            os.environ[FUSED_EXPERT_WIDTH4_ENV] = "true"
            fused_k3_experts_enabled.cache_clear()
            fused_k3_expert_width4_enabled.cache_clear()
            with self.assertRaisesRegex(ValueError, "must be exactly"):
                maybe_fused_k3_switch_glu_reduce(
                    self.switch, self.x, self.indices, self.router_weights
                )

            os.environ[FUSED_EXPERT_WIDTH4_ENV] = "1"
            os.environ[DERIVE_AFFINE2_BIAS_ENV] = "true"
            fused_k3_expert_width4_enabled.cache_clear()
            derive_affine2_bias_enabled.cache_clear()
            with self.assertRaisesRegex(ValueError, "must be exactly"):
                maybe_fused_k3_switch_glu_reduce(
                    self.switch, self.x, self.indices, self.router_weights
                )
            dispatch.assert_not_called()

    def test_model_adapter_selects_only_the_exact_width4_path(self):
        self._enable_all()
        sentinel = object()
        front_patch, down_patch, bias_patch = self._adapter_patches()
        with (
            front_patch,
            down_patch as down_support,
            bias_patch,
            mock.patch.object(
                fused_expert_adapter,
                "_compiled_width4_switch_glu_reduce_all_derived",
                return_value=sentinel,
            ) as dispatch,
        ):
            result = maybe_fused_k3_switch_glu_reduce(
                self.switch,
                self.x,
                self.indices,
                self.router_weights,
            )
        self.assertIs(result, sentinel)
        down_support.assert_called_once_with(
            self.indices,
            self.router_weights,
            (
                self.switch.down_proj.weight,
                self.switch.down_proj.scales,
                self.switch.down_proj.biases,
            ),
            results_per_threadgroup=4,
            simdgroups_per_threadgroup=8,
        )
        dispatch.assert_called_once()

    def test_neighboring_geometry_falls_back_before_dispatch(self):
        self._enable_all()
        with (
            mock.patch.object(
                fused_expert_adapter,
                "supports_width4_switch_situ",
                return_value=False,
            ),
            mock.patch.object(
                fused_expert_adapter,
                "_compiled_width4_switch_glu_reduce_all_derived",
            ) as dispatch,
        ):
            self.assertIsNone(
                maybe_fused_k3_switch_glu_reduce(
                    self.switch,
                    self.x,
                    _ArrayStub((1, K3_WIDTH4, K3_TOP_K + 1)),
                    _ArrayStub((1, K3_WIDTH4, K3_TOP_K + 1)),
                )
            )
            dispatch.assert_not_called()

    def test_derived_bias_validation_fails_closed_before_dispatch(self):
        self._enable_all()
        front_patch, down_patch, _ = self._adapter_patches()
        with (
            front_patch,
            down_patch,
            mock.patch.object(
                fused_expert_adapter,
                "_all_projections_support_derived_bias",
                return_value=False,
            ),
            mock.patch.object(
                fused_expert_adapter,
                "_compiled_width4_switch_glu_reduce_all_derived",
            ) as dispatch,
        ):
            self.assertIsNone(
                maybe_fused_k3_switch_glu_reduce(
                    self.switch, self.x, self.indices, self.router_weights
                )
            )
            dispatch.assert_not_called()


@unittest.skipUnless(_metal_available(), "requires Metal")
class Width4KernelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mx.random.seed(20260810)
        cls.switch = _Switch()
        cls.x = mx.random.normal(
            (1, K3_WIDTH4, K3_HIDDEN),
            dtype=mx.bfloat16,
        )
        base = mx.array([0, 17, 63, 255, 444, 511, 700, 895], dtype=mx.uint32)
        cls.indices = mx.stack(
            [mx.roll(base, shift) for shift in range(K3_WIDTH4)]
        )[None]
        mx.eval(
            cls.x,
            cls.indices,
            *(
                part
                for projection in (
                    cls.switch.up_proj,
                    cls.switch.gate_proj,
                    cls.switch.down_proj,
                )
                for part in projection.parts()
            ),
        )

    def test_both_production_shapes_are_bit_exact(self):
        reference = _stock(self.switch, self.x, self.indices)
        candidate = width4_switch_glu(
            self.x,
            self.indices,
            self.switch.up_proj.parts(),
            self.switch.gate_proj.parts(),
            self.switch.down_proj.parts(),
            derive_front_bias=True,
            derive_down_bias=True,
        )
        mx.eval(reference, candidate)
        self.assertEqual(candidate.shape, (1, K3_WIDTH4, K3_TOP_K, K3_HIDDEN))
        self.assertTrue(
            bool(
                mx.array_equal(
                    reference.view(mx.uint8),
                    candidate.view(mx.uint8),
                ).item()
            )
        )

        native_down_candidate = width4_switch_glu_native_down(
            self.x,
            self.indices,
            self.switch.up_proj.parts(),
            self.switch.gate_proj.parts(),
            self.switch.down_proj.parts(),
            front_results_per_simdgroup=8,
            derive_front_bias=True,
        )
        mx.eval(native_down_candidate)
        self.assertTrue(
            bool(
                mx.array_equal(
                    reference.view(mx.uint8),
                    native_down_candidate.view(mx.uint8),
                ).item()
            )
        )
        router_weights = mx.random.uniform(
            shape=(1, K3_WIDTH4, K3_TOP_K),
            dtype=mx.bfloat16,
        )
        reduced_reference = (reference * router_weights[..., None]).sum(axis=-2)
        reduced_candidate = width4_switch_glu_reduce(
            self.x,
            self.indices,
            router_weights,
            self.switch.up_proj.parts(),
            self.switch.gate_proj.parts(),
            self.switch.down_proj.parts(),
            derive_front_bias=True,
            derive_down_bias=True,
        )
        mx.eval(reduced_reference, reduced_candidate)
        self.assertTrue(
            bool(
                mx.array_equal(
                    reduced_reference.view(mx.uint8),
                    reduced_candidate.view(mx.uint8),
                ).item()
            )
        )

    def test_neighboring_shapes_bypass(self):
        up = self.switch.up_proj.parts()
        gate = self.switch.gate_proj.parts()
        down = self.switch.down_proj.parts()
        neighbors = (
            (
                mx.zeros((1, 3, K3_HIDDEN), dtype=mx.bfloat16),
                mx.zeros((1, 3, K3_TOP_K), dtype=mx.uint32),
            ),
            (
                mx.zeros((1, 5, K3_HIDDEN), dtype=mx.bfloat16),
                mx.zeros((1, 5, K3_TOP_K), dtype=mx.uint32),
            ),
            (
                self.x,
                mx.zeros((1, K3_WIDTH4, 16), dtype=mx.uint32),
            ),
        )
        for x, indices in neighbors:
            with self.subTest(x_shape=x.shape, indices_shape=indices.shape):
                self.assertFalse(supports_width4_switch_situ(x, indices, up, gate))
        activated = mx.zeros(
            (1, 3, K3_TOP_K, 1, K3_INTERMEDIATE),
            dtype=mx.bfloat16,
        )
        self.assertFalse(supports_width4_down(activated, self.indices[:, :3], down))
        self.assertFalse(
            supports_width4_native_down_projection(self.indices[:, :3], down)
        )
        self.assertFalse(
            supports_width4_down_reduce_projection(
                self.indices[:, :3],
                mx.zeros((1, 3, K3_TOP_K), dtype=mx.bfloat16),
                down,
                results_per_threadgroup=4,
                simdgroups_per_threadgroup=8,
            )
        )
        os.environ[FUSED_EXPERT_ENV] = "1"
        os.environ[FUSED_EXPERT_WIDTH4_ENV] = "1"
        fused_k3_experts_enabled.cache_clear()
        fused_k3_expert_width4_enabled.cache_clear()
        for x, indices in neighbors:
            with self.subTest(adapter_x_shape=x.shape, indices_shape=indices.shape):
                self.assertIsNone(
                    maybe_fused_k3_switch_glu(self.switch, x, indices)
                )

    def tearDown(self):
        os.environ.pop(DERIVE_AFFINE2_BIAS_ENV, None)
        os.environ.pop(FUSED_DOWN_REDUCE_ENV, None)
        os.environ.pop(FUSED_EXPERT_ENV, None)
        os.environ.pop(FUSED_EXPERT_WIDTH4_ENV, None)
        derive_affine2_bias_enabled.cache_clear()
        fused_k3_down_reduce_enabled.cache_clear()
        fused_k3_experts_enabled.cache_clear()
        fused_k3_expert_width4_enabled.cache_clear()

    def test_adapter_is_independently_default_off(self):
        os.environ[DERIVE_AFFINE2_BIAS_ENV] = "1"
        os.environ[FUSED_EXPERT_ENV] = "1"
        os.environ.pop(FUSED_EXPERT_WIDTH4_ENV, None)
        derive_affine2_bias_enabled.cache_clear()
        fused_k3_experts_enabled.cache_clear()
        fused_k3_expert_width4_enabled.cache_clear()
        self.assertIsNone(maybe_fused_k3_switch_glu(self.switch, self.x, self.indices))
        os.environ[FUSED_EXPERT_WIDTH4_ENV] = "1"
        fused_k3_expert_width4_enabled.cache_clear()
        candidate = maybe_fused_k3_switch_glu(self.switch, self.x, self.indices)
        self.assertIsNotNone(candidate)
        reference = _stock(self.switch, self.x, self.indices)
        mx.eval(reference, candidate)
        self.assertTrue(bool(mx.array_equal(reference, candidate).item()))
        os.environ.pop(FUSED_EXPERT_ENV, None)
        fused_k3_experts_enabled.cache_clear()

    def test_adapter_requires_validated_affine2_execution(self):
        os.environ.pop(DERIVE_AFFINE2_BIAS_ENV, None)
        os.environ[FUSED_EXPERT_ENV] = "1"
        os.environ[FUSED_EXPERT_WIDTH4_ENV] = "1"
        derive_affine2_bias_enabled.cache_clear()
        fused_k3_experts_enabled.cache_clear()
        fused_k3_expert_width4_enabled.cache_clear()
        self.assertIsNone(
            maybe_fused_k3_switch_glu(self.switch, self.x, self.indices)
        )

    def test_reduce_adapter_dispatch_is_bit_exact(self):
        os.environ[DERIVE_AFFINE2_BIAS_ENV] = "1"
        os.environ[FUSED_DOWN_REDUCE_ENV] = "1"
        os.environ[FUSED_EXPERT_ENV] = "1"
        os.environ[FUSED_EXPERT_WIDTH4_ENV] = "1"
        derive_affine2_bias_enabled.cache_clear()
        fused_k3_down_reduce_enabled.cache_clear()
        fused_k3_experts_enabled.cache_clear()
        fused_k3_expert_width4_enabled.cache_clear()
        router_weights = mx.random.uniform(
            shape=(1, K3_WIDTH4, K3_TOP_K),
            dtype=mx.bfloat16,
        )
        reference = (
            _stock(self.switch, self.x, self.indices) * router_weights[..., None]
        ).sum(
            axis=-2,
        )
        original = fused_expert_adapter._compiled_width4_switch_glu_reduce_all_derived
        with mock.patch.object(
            fused_expert_adapter,
            "_compiled_width4_switch_glu_reduce_all_derived",
            wraps=original,
        ) as dispatch:
            candidate = maybe_fused_k3_switch_glu_reduce(
                self.switch,
                self.x,
                self.indices,
                router_weights,
            )
        self.assertIsNotNone(candidate)
        dispatch.assert_called_once()
        mx.eval(reference, candidate)
        self.assertEqual(candidate.shape, (1, K3_WIDTH4, K3_HIDDEN))
        self.assertTrue(
            bool(
                mx.array_equal(
                    reference.view(mx.uint8),
                    candidate.view(mx.uint8),
                ).item()
            )
        )


if __name__ == "__main__":
    unittest.main()
