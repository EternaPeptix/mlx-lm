from __future__ import annotations

import os
import unittest
from unittest import mock

import mlx.core as mx

from mlx_lm.models.kimi_k3_derived_bias import (
    DERIVE_AFFINE2_BIAS_ENV,
    affine2_bias_relation_is_exact,
    affine2_biases_are_fast_derivable,
    derive_affine2_bias_enabled,
    derived_affine2_biases,
    projection_has_validated_derived_bias,
    validate_k3_biases_for_load,
)


def _bf16_bits(values: list[int]) -> mx.array:
    return mx.array(values, dtype=mx.uint16).view(mx.bfloat16)


class _Projection:
    bits = 2
    group_size = 128
    mode = "affine"

    def __contains__(self, name):
        return False


class _Switch:
    def __init__(self):
        self.gate_proj = _Projection()
        self.up_proj = _Projection()
        self.down_proj = _Projection()


class _SparseMLP:
    def __init__(self):
        self.switch_mlp = _Switch()


class _Layer:
    def __init__(self):
        self.mlp = _SparseMLP()


class DerivedBiasContractTest(unittest.TestCase):
    def tearDown(self):
        os.environ.pop(DERIVE_AFFINE2_BIAS_ENV, None)
        derive_affine2_bias_enabled.cache_clear()

    def test_flag_is_default_off_and_strict(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            derive_affine2_bias_enabled.cache_clear()
            self.assertFalse(derive_affine2_bias_enabled())
        for value, expected in (("0", False), ("1", True)):
            with self.subTest(value=value):
                with mock.patch.dict(
                    os.environ,
                    {DERIVE_AFFINE2_BIAS_ENV: value},
                    clear=True,
                ):
                    derive_affine2_bias_enabled.cache_clear()
                    self.assertEqual(derive_affine2_bias_enabled(), expected)
        for value in ("", "true", "01", "yes", "2"):
            with self.subTest(value=value):
                with mock.patch.dict(
                    os.environ,
                    {DERIVE_AFFINE2_BIAS_ENV: value},
                    clear=True,
                ):
                    derive_affine2_bias_enabled.cache_clear()
                    with self.assertRaisesRegex(ValueError, "exactly '0' or '1'"):
                        derive_affine2_bias_enabled()

    def test_exact_bits_cover_zero_subnormal_and_infinity(self):
        # +0, -0, minimum signed subnormals, +/-infinity, and +/-1.
        scales = _bf16_bits(
            [0x0000, 0x8000, 0x0001, 0x8001, 0x7F80, 0xFF80, 0x3F80, 0xBF80]
        )
        expected_biases = _bf16_bits(
            [0x8000, 0x0000, 0x8002, 0x0002, 0xFF80, 0x7F80, 0xC000, 0x4000]
        )
        actual = derived_affine2_biases(scales)
        mx.eval(actual)
        self.assertTrue(
            bool(
                mx.array_equal(
                    actual.view(mx.uint8), expected_biases.view(mx.uint8)
                ).item()
            )
        )
        self.assertTrue(affine2_bias_relation_is_exact(scales, expected_biases))
        self.assertFalse(affine2_biases_are_fast_derivable(scales, expected_biases))

        corrupt = expected_biases.view(mx.uint16)
        corrupt = mx.concatenate(
            [corrupt[:-1], mx.array([0x4001], dtype=mx.uint16)]
        ).view(mx.bfloat16)
        self.assertFalse(affine2_bias_relation_is_exact(scales, corrupt))

    def test_load_validation_marks_all_present_projections(self):
        os.environ[DERIVE_AFFINE2_BIAS_ENV] = "1"
        derive_affine2_bias_enabled.cache_clear()
        layer = _Layer()
        weights = {}
        scales = _bf16_bits([0x0080, 0x8080, 0x3F80, 0xFF7F])
        biases = derived_affine2_biases(scales)
        for name in ("gate_proj", "up_proj", "down_proj"):
            prefix = f"model.layers.0.mlp.switch_mlp.{name}"
            weights[f"{prefix}.scales"] = scales
            weights[f"{prefix}.biases"] = biases

        self.assertEqual(validate_k3_biases_for_load([layer], weights), 3)
        for name in ("gate_proj", "up_proj", "down_proj"):
            projection = getattr(layer.mlp.switch_mlp, name)
            self.assertTrue(
                projection_has_validated_derived_bias(
                    projection,
                    scales,
                    biases,
                )
            )

    def test_load_validation_fails_closed_on_missing_or_wrong_metadata(self):
        os.environ[DERIVE_AFFINE2_BIAS_ENV] = "1"
        derive_affine2_bias_enabled.cache_clear()
        layer = _Layer()
        prefix = "model.layers.0.mlp.switch_mlp.gate_proj"
        scales = _bf16_bits([0x3F80])
        with self.assertRaisesRegex(ValueError, "both scales and biases"):
            validate_k3_biases_for_load(
                [layer],
                {f"{prefix}.scales": scales},
            )
        with self.assertRaisesRegex(ValueError, "exact BF16"):
            validate_k3_biases_for_load(
                [layer],
                {
                    f"{prefix}.scales": scales,
                    f"{prefix}.biases": _bf16_bits([0x0000]),
                },
            )
        special_scales = _bf16_bits([0x0000, 0x0001, 0x7F80])
        with self.assertRaisesRegex(ValueError, "finite-normal BF16"):
            validate_k3_biases_for_load(
                [layer],
                {
                    f"{prefix}.scales": special_scales,
                    f"{prefix}.biases": derived_affine2_biases(special_scales),
                },
            )

    def test_runtime_guard_falls_back_and_caches_failure(self):
        projection = _Projection()
        scales = _bf16_bits([0x3F80])
        wrong = _bf16_bits([0x0000])
        self.assertFalse(
            projection_has_validated_derived_bias(projection, scales, wrong)
        )
        self.assertFalse(
            projection_has_validated_derived_bias(
                projection,
                scales,
                derived_affine2_biases(scales),
            )
        )


if __name__ == "__main__":
    unittest.main()
