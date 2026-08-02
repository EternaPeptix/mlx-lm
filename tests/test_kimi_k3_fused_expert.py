from __future__ import annotations

import os
import unittest

import mlx.core as mx

from mlx_lm.models.kimi_k3_derived_bias import (
    DERIVE_AFFINE2_BIAS_ENV,
    derive_affine2_bias_enabled,
    derived_affine2_biases,
    projection_has_validated_derived_bias,
)
from mlx_lm.models.kimi_k3_fused_expert import (
    FUSED_DOWN_REDUCE_ENV,
    FUSED_EXPERT_ENV,
    FUSED_EXPERT_WIDTH2_ENV,
    FUSED_EXPERT_WIDTH3_ENV,
    fused_k3_down_reduce_enabled,
    fused_k3_expert_width2_enabled,
    fused_k3_expert_width3_enabled,
    fused_k3_experts_enabled,
    maybe_fused_k3_switch_glu,
    maybe_fused_k3_switch_glu_reduce,
)
from mlx_lm.models.kimi_k3_fused_switch_glu import _metal_available


class _Activation:
    beta = 4.0
    linear_beta = 25.0


class _Projection:
    bits = 2
    group_size = 128
    mode = "affine"

    def __init__(self, weight):
        self.weight, self.scales, self.biases = mx.quantize(
            weight, group_size=128, bits=2, mode="affine"
        )

    @classmethod
    def packed(
        cls,
        *,
        experts: int,
        input_width: int,
        output_width: int,
        packed_value: int,
    ):
        projection = cls.__new__(cls)
        projection.weight = mx.full(
            (experts, output_width, input_width // 16),
            packed_value,
            dtype=mx.uint32,
        )
        projection.scales = mx.full(
            (experts, output_width, input_width // 128),
            0.015625,
            dtype=mx.bfloat16,
        )
        projection.biases = mx.zeros_like(projection.scales)
        return projection

    @classmethod
    def packed_nonuniform(
        cls,
        *,
        experts: int,
        input_width: int,
        output_width: int,
        salt: int,
    ):
        """Build exact packed tensors whose every addressing axis matters."""

        projection = cls.__new__(cls)
        expert = mx.arange(experts, dtype=mx.uint32).reshape(experts, 1, 1)
        output = mx.arange(output_width, dtype=mx.uint32).reshape(
            1, output_width, 1
        )
        packed_k = mx.arange(input_width // 16, dtype=mx.uint32).reshape(
            1, 1, input_width // 16
        )
        projection.weight = mx.bitwise_xor(
            mx.bitwise_xor(
                mx.array(salt, dtype=mx.uint32),
                expert * mx.array(0x9E3779B9, dtype=mx.uint32),
            ),
            mx.bitwise_xor(
                output * mx.array(0x85EBCA6B, dtype=mx.uint32),
                packed_k * mx.array(0xC2B2AE35, dtype=mx.uint32),
            ),
        )

        group_k = mx.arange(input_width // 128, dtype=mx.uint32).reshape(
            1, 1, input_width // 128
        )
        metadata_code = (
            expert * mx.array(17, dtype=mx.uint32)
            + output * mx.array(5, dtype=mx.uint32)
            + group_k * mx.array(3, dtype=mx.uint32)
            + mx.array(salt & 0xFF, dtype=mx.uint32)
        )
        projection.scales = (
            mx.array(1 / 128, dtype=mx.float32)
            + (metadata_code % 13).astype(mx.float32)
            * mx.array(1 / 4096, dtype=mx.float32)
        ).astype(mx.bfloat16)
        projection.biases = (
            ((metadata_code % 17).astype(mx.float32) - 8)
            * mx.array(1 / 4096, dtype=mx.float32)
        ).astype(mx.bfloat16)
        return projection

    def __contains__(self, name):
        return False

    def __getitem__(self, name):
        return getattr(self, name)

    def get(self, name):
        return getattr(self, name, None)

    def __call__(self, x, indices, sorted_indices=False):
        del sorted_indices
        return mx.gather_qmm(
            x,
            self.weight,
            self.scales,
            self.biases,
            rhs_indices=indices,
            transpose=True,
            group_size=128,
            bits=2,
            mode="affine",
        )


class _Switch:
    training = False
    activation = _Activation()

    def __init__(self):
        self.gate_proj = _Projection(mx.random.normal((8, 512, 512), dtype=mx.bfloat16))
        self.up_proj = _Projection(mx.random.normal((8, 512, 512), dtype=mx.bfloat16))
        self.down_proj = _Projection(mx.random.normal((8, 512, 512), dtype=mx.bfloat16))

    @classmethod
    def bounded_tp2_geometry(cls):
        """Keep K3's TP2 dimensions while bounding the inactive expert table."""

        switch = cls.__new__(cls)
        switch.gate_proj = _Projection.packed(
            experts=16,
            input_width=3584,
            output_width=1536,
            packed_value=0x12345678,
        )
        switch.up_proj = _Projection.packed(
            experts=16,
            input_width=3584,
            output_width=1536,
            packed_value=0x76543210,
        )
        switch.down_proj = _Projection.packed(
            experts=16,
            input_width=1536,
            output_width=3584,
            packed_value=0x24681357,
        )
        return switch

    @classmethod
    def bounded_tp2_geometry_nonuniform(cls):
        """Use K3 TP2 dimensions with address-sensitive packed parameters."""

        switch = cls.__new__(cls)
        switch.gate_proj = _Projection.packed_nonuniform(
            experts=16,
            input_width=3584,
            output_width=1536,
            salt=0x13579BDF,
        )
        switch.up_proj = _Projection.packed_nonuniform(
            experts=16,
            input_width=3584,
            output_width=1536,
            salt=0x2468ACE0,
        )
        switch.down_proj = _Projection.packed_nonuniform(
            experts=16,
            input_width=1536,
            output_width=3584,
            salt=0x55AA55AA,
        )
        return switch

    def stock(self, x, indices):
        expanded = mx.expand_dims(x, (-2, -3))
        up = self.up_proj(expanded, indices).astype(mx.float32)
        gate = self.gate_proj(expanded, indices).astype(mx.float32)
        activated = (
            4.0 * mx.tanh(gate / 4.0) * mx.sigmoid(gate) * (25.0 * mx.tanh(up / 25.0))
        ).astype(mx.bfloat16)
        return self.down_proj(activated, indices).squeeze(-2)


def _install_derived_biases(
    switch: _Switch,
    *,
    include_down: bool = False,
) -> None:
    projections = (
        switch.gate_proj,
        switch.up_proj,
    )
    if include_down:
        projections += (switch.down_proj,)
    for projection in projections:
        projection.biases = derived_affine2_biases(projection.scales)


@unittest.skipUnless(_metal_available(), "requires Metal")
class IntegrationTest(unittest.TestCase):
    def setUp(self):
        os.environ[FUSED_EXPERT_ENV] = "1"
        os.environ[FUSED_DOWN_REDUCE_ENV] = "1"
        os.environ.pop(FUSED_EXPERT_WIDTH2_ENV, None)
        os.environ.pop(FUSED_EXPERT_WIDTH3_ENV, None)
        os.environ.pop(DERIVE_AFFINE2_BIAS_ENV, None)
        fused_k3_experts_enabled.cache_clear()
        fused_k3_down_reduce_enabled.cache_clear()
        fused_k3_expert_width2_enabled.cache_clear()
        fused_k3_expert_width3_enabled.cache_clear()
        derive_affine2_bias_enabled.cache_clear()

    def tearDown(self):
        os.environ.pop(FUSED_EXPERT_ENV, None)
        os.environ.pop(FUSED_DOWN_REDUCE_ENV, None)
        os.environ.pop(FUSED_EXPERT_WIDTH2_ENV, None)
        os.environ.pop(FUSED_EXPERT_WIDTH3_ENV, None)
        os.environ.pop(DERIVE_AFFINE2_BIAS_ENV, None)
        fused_k3_experts_enabled.cache_clear()
        fused_k3_down_reduce_enabled.cache_clear()
        fused_k3_expert_width2_enabled.cache_clear()
        fused_k3_expert_width3_enabled.cache_clear()
        derive_affine2_bias_enabled.cache_clear()

    def test_fused_switch_is_bit_exact_on_cached_second_call(self):
        mx.random.seed(19)
        switch = _Switch()
        x = mx.random.normal((1, 1, 512), dtype=mx.bfloat16)
        indices = mx.array([[[1, 3, 5, 7]]], dtype=mx.uint32)
        reference = switch.stock(x, indices)
        first = maybe_fused_k3_switch_glu(switch, x, indices)
        second = maybe_fused_k3_switch_glu(switch, x, indices)
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        mx.eval(reference, first, second)
        self.assertTrue(bool(mx.all(reference == first).item()))
        self.assertTrue(bool(mx.all(reference == second).item()))

    def test_compiled_cache_keeps_weights_dynamic(self):
        mx.random.seed(23)
        first_switch = _Switch()
        mx.random.seed(29)
        second_switch = _Switch()
        x = mx.random.normal((1, 1, 512), dtype=mx.bfloat16)
        indices = mx.array([[[0, 2, 4, 6]]], dtype=mx.uint32)

        first = maybe_fused_k3_switch_glu(first_switch, x, indices)
        second = maybe_fused_k3_switch_glu(second_switch, x, indices)
        reference = second_switch.stock(x, indices)
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        mx.eval(first, second, reference)
        self.assertTrue(bool(mx.all(reference == second).item()))
        self.assertTrue(bool(mx.any(first != second).item()))

    def test_bounded_tp2_geometry_is_bit_exact(self):
        switch = _Switch.bounded_tp2_geometry()
        x = mx.random.normal((1, 1, 3584), dtype=mx.bfloat16)
        indices = mx.arange(16, dtype=mx.uint32).reshape(1, 1, 16)
        reference = switch.stock(x, indices)
        candidate = maybe_fused_k3_switch_glu(switch, x, indices)
        self.assertIsNotNone(candidate)
        mx.eval(reference, candidate)
        self.assertEqual(candidate.shape, (1, 1, 16, 3584))
        self.assertTrue(bool(mx.all(reference == candidate).item()))

    def test_bounded_tp2_down_route_reduce_is_bit_exact(self):
        switch = _Switch.bounded_tp2_geometry()
        x = mx.random.normal((1, 1, 3584), dtype=mx.bfloat16)
        indices = mx.arange(16, dtype=mx.uint32).reshape(1, 1, 16)
        router_weights = mx.random.uniform(
            shape=(1, 1, 16),
            dtype=mx.bfloat16,
        )
        reference = (switch.stock(x, indices) * router_weights[..., None]).sum(axis=-2)
        candidate = maybe_fused_k3_switch_glu_reduce(
            switch,
            x,
            indices,
            router_weights,
        )
        self.assertIsNotNone(candidate)
        mx.eval(reference, candidate)
        self.assertEqual(candidate.shape, (1, 1, 3584))
        self.assertTrue(bool(mx.all(reference == candidate).item()))

    def test_width_two_full_fused_expert_is_bit_exact(self):
        os.environ[FUSED_EXPERT_WIDTH2_ENV] = "1"
        fused_k3_expert_width2_enabled.cache_clear()
        switch = _Switch.bounded_tp2_geometry()
        width = 2
        x = mx.random.normal((1, width, 3584), dtype=mx.bfloat16)
        base = mx.arange(16, dtype=mx.uint32)
        indices = mx.stack([mx.roll(base, shift) for shift in range(width)])[None]
        router_weights = mx.random.uniform(
            shape=(1, width, 16),
            dtype=mx.bfloat16,
        )
        reference = (
            switch.stock(x, indices) * router_weights[..., None]
        ).sum(axis=-2)
        candidate = maybe_fused_k3_switch_glu_reduce(
            switch,
            x,
            indices,
            router_weights,
        )
        self.assertIsNotNone(candidate)
        mx.eval(reference, candidate)
        self.assertEqual(candidate.shape, (1, width, 3584))
        self.assertTrue(bool(mx.all(reference == candidate).item()))

    def test_width_two_full_fused_expert_nonuniform_packing_is_bit_exact(self):
        os.environ[FUSED_EXPERT_WIDTH2_ENV] = "1"
        fused_k3_expert_width2_enabled.cache_clear()
        mx.random.seed(20260801)
        switch = _Switch.bounded_tp2_geometry_nonuniform()
        width = 2
        x = mx.random.normal((1, width, 3584), dtype=mx.bfloat16)
        base = mx.arange(16, dtype=mx.uint32)
        indices = mx.stack(
            [(base * 5 + shift * 3) % 16 for shift in range(width)]
        )[None]
        router_weights = mx.random.uniform(
            shape=(1, width, 16),
            dtype=mx.bfloat16,
        )
        reference = (
            switch.stock(x, indices) * router_weights[..., None]
        ).sum(axis=-2)
        candidate = maybe_fused_k3_switch_glu_reduce(
            switch,
            x,
            indices,
            router_weights,
        )
        self.assertIsNotNone(candidate)
        mx.eval(reference, candidate)
        self.assertEqual(candidate.shape, (1, width, 3584))
        self.assertTrue(bool(mx.array_equal(reference, candidate).item()))

    def test_width_three_fused_front_is_opt_in_and_bit_exact(self):
        mx.random.seed(20260802)
        switch = _Switch.bounded_tp2_geometry_nonuniform()
        _install_derived_biases(switch, include_down=True)
        os.environ[DERIVE_AFFINE2_BIAS_ENV] = "1"
        derive_affine2_bias_enabled.cache_clear()
        width = 3
        x = mx.random.normal((1, width, 3584), dtype=mx.bfloat16)
        base = mx.arange(16, dtype=mx.uint32)
        indices = mx.stack(
            [(base * 5 + shift * 3) % 16 for shift in range(width)]
        )[None]
        reference = switch.stock(x, indices)

        self.assertIsNone(maybe_fused_k3_switch_glu(switch, x, indices))
        os.environ[FUSED_EXPERT_WIDTH3_ENV] = "1"
        fused_k3_expert_width3_enabled.cache_clear()
        candidate = maybe_fused_k3_switch_glu(switch, x, indices)

        self.assertIsNotNone(candidate)
        mx.eval(reference, candidate)
        self.assertEqual(candidate.shape, (1, width, 16, 3584))
        self.assertTrue(bool(mx.array_equal(reference, candidate).item()))

    def test_derived_bias_keeps_mismatched_down_on_stored_path(self):
        os.environ[DERIVE_AFFINE2_BIAS_ENV] = "1"
        derive_affine2_bias_enabled.cache_clear()
        switch = _Switch.bounded_tp2_geometry_nonuniform()
        _install_derived_biases(switch)
        x = mx.random.normal((1, 1, 3584), dtype=mx.bfloat16)
        indices = mx.arange(16, dtype=mx.uint32).reshape(1, 1, 16)
        router_weights = mx.random.uniform(
            shape=(1, 1, 16),
            dtype=mx.bfloat16,
        )
        unreduced_reference = switch.stock(x, indices)
        reference = (unreduced_reference * router_weights[..., None]).sum(axis=-2)
        unreduced_candidate = maybe_fused_k3_switch_glu(switch, x, indices)
        candidate = maybe_fused_k3_switch_glu_reduce(
            switch,
            x,
            indices,
            router_weights,
        )
        self.assertIsNotNone(unreduced_candidate)
        self.assertIsNotNone(candidate)
        mx.eval(unreduced_reference, unreduced_candidate, reference, candidate)
        self.assertTrue(
            bool(mx.array_equal(unreduced_reference, unreduced_candidate).item())
        )
        self.assertTrue(bool(mx.array_equal(reference, candidate).item()))
        self.assertFalse(
            projection_has_validated_derived_bias(
                switch.down_proj,
                switch.down_proj.scales,
                switch.down_proj.biases,
            )
        )

    def test_derived_bias_uses_exact_down_bank_and_is_bit_exact(self):
        os.environ[DERIVE_AFFINE2_BIAS_ENV] = "1"
        derive_affine2_bias_enabled.cache_clear()
        switch = _Switch.bounded_tp2_geometry_nonuniform()
        _install_derived_biases(switch, include_down=True)
        x = mx.random.normal((1, 1, 3584), dtype=mx.bfloat16)
        indices = mx.arange(16, dtype=mx.uint32).reshape(1, 1, 16)
        router_weights = mx.random.uniform(
            shape=(1, 1, 16),
            dtype=mx.bfloat16,
        )
        unreduced_reference = switch.stock(x, indices)
        reference = (unreduced_reference * router_weights[..., None]).sum(axis=-2)
        unreduced_candidate = maybe_fused_k3_switch_glu(switch, x, indices)
        candidate = maybe_fused_k3_switch_glu_reduce(
            switch,
            x,
            indices,
            router_weights,
        )
        self.assertIsNotNone(unreduced_candidate)
        self.assertIsNotNone(candidate)
        mx.eval(unreduced_reference, unreduced_candidate, reference, candidate)
        self.assertTrue(
            bool(mx.array_equal(unreduced_reference, unreduced_candidate).item())
        )
        self.assertTrue(bool(mx.array_equal(reference, candidate).item()))
        self.assertTrue(
            projection_has_validated_derived_bias(
                switch.down_proj,
                switch.down_proj.scales,
                switch.down_proj.biases,
            )
        )

    def test_width_two_has_an_independent_default_off_flag(self):
        switch = _Switch.bounded_tp2_geometry()
        x = mx.zeros((1, 2, 3584), dtype=mx.bfloat16)
        indices = mx.broadcast_to(
            mx.arange(16, dtype=mx.uint32).reshape(1, 1, 16),
            (1, 2, 16),
        )
        router_weights = mx.zeros((1, 2, 16), dtype=mx.bfloat16)
        self.assertIsNone(maybe_fused_k3_switch_glu(switch, x, indices))
        self.assertIsNone(
            maybe_fused_k3_switch_glu_reduce(
                switch,
                x,
                indices,
                router_weights,
            )
        )

    def test_derived_bias_violation_falls_back_before_dispatch(self):
        os.environ[DERIVE_AFFINE2_BIAS_ENV] = "1"
        derive_affine2_bias_enabled.cache_clear()
        switch = _Switch.bounded_tp2_geometry()
        x = mx.zeros((1, 1, 3584), dtype=mx.bfloat16)
        indices = mx.arange(16, dtype=mx.uint32).reshape(1, 1, 16)
        router_weights = mx.zeros((1, 1, 16), dtype=mx.bfloat16)
        self.assertIsNone(maybe_fused_k3_switch_glu(switch, x, indices))
        self.assertIsNone(
            maybe_fused_k3_switch_glu_reduce(
                switch,
                x,
                indices,
                router_weights,
            )
        )

    def test_down_route_reduce_has_an_independent_default_off_flag(self):
        os.environ.pop(FUSED_DOWN_REDUCE_ENV)
        fused_k3_down_reduce_enabled.cache_clear()
        switch = _Switch.bounded_tp2_geometry()
        x = mx.zeros((1, 1, 3584), dtype=mx.bfloat16)
        indices = mx.arange(16, dtype=mx.uint32).reshape(1, 1, 16)
        router_weights = mx.zeros((1, 1, 16), dtype=mx.bfloat16)
        self.assertIsNone(
            maybe_fused_k3_switch_glu_reduce(
                switch,
                x,
                indices,
                router_weights,
            )
        )
        self.assertIsNotNone(maybe_fused_k3_switch_glu(switch, x, indices))

    def test_down_route_reduce_rejects_non_k3_routes_before_dispatch(self):
        switch = _Switch.bounded_tp2_geometry()
        x = mx.zeros((1, 1, 3584), dtype=mx.bfloat16)
        indices = mx.arange(16, dtype=mx.uint32).reshape(1, 1, 16)
        bad_routes = (
            (indices[..., :-1], mx.zeros((1, 1, 15), dtype=mx.bfloat16)),
            (indices, mx.zeros((1, 1, 16), dtype=mx.float32)),
        )
        for routed_indices, router_weights in bad_routes:
            with self.subTest(
                top_k=routed_indices.shape[-1],
                router_dtype=router_weights.dtype,
            ):
                self.assertIsNone(
                    maybe_fused_k3_switch_glu_reduce(
                        switch,
                        x,
                        routed_indices,
                        router_weights,
                    )
                )

        switch.down_proj = _Projection.packed(
            experts=17,
            input_width=1536,
            output_width=3584,
            packed_value=0x24681357,
        )
        self.assertIsNone(
            maybe_fused_k3_switch_glu_reduce(
                switch,
                x,
                indices,
                mx.zeros((1, 1, 16), dtype=mx.bfloat16),
            )
        )

    def test_unsupported_shapes_and_dtypes_fall_back(self):
        switch = _Switch()
        indices = mx.array([[[1, 3, 5, 7]]], dtype=mx.uint32)
        cases = (
            (
                mx.zeros((1, 9, 512), dtype=mx.bfloat16),
                mx.broadcast_to(indices, (1, 9, 4)),
            ),
            (
                mx.zeros((2, 1, 512), dtype=mx.bfloat16),
                mx.broadcast_to(indices, (2, 1, 4)),
            ),
            (mx.zeros((1, 1, 512), dtype=mx.float32), indices),
        )
        for x, routed_indices in cases:
            with self.subTest(shape=x.shape, dtype=x.dtype):
                self.assertIsNone(maybe_fused_k3_switch_glu(switch, x, routed_indices))

    def test_training_falls_back(self):
        switch = _Switch()
        switch.training = True
        x = mx.zeros((1, 1, 512), dtype=mx.bfloat16)
        indices = mx.array([[[1, 3, 5, 7]]], dtype=mx.uint32)
        self.assertIsNone(maybe_fused_k3_switch_glu(switch, x, indices))

    def test_disabled_by_default(self):
        os.environ.pop(FUSED_EXPERT_ENV)
        fused_k3_experts_enabled.cache_clear()
        switch = _Switch()
        x = mx.zeros((1, 1, 512), dtype=mx.bfloat16)
        indices = mx.array([[[1, 3, 5, 7]]], dtype=mx.uint32)
        self.assertIsNone(maybe_fused_k3_switch_glu(switch, x, indices))


if __name__ == "__main__":
    unittest.main()
