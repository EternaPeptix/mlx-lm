from __future__ import annotations

import os
import unittest
from unittest import mock

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.kimi_k3_derived_bias import (
    DERIVE_AFFINE2_BIAS_ENV,
    ELIDE_AFFINE2_BIAS_ENV,
    affine2_gather_core_available,
    affine2_bias_relation_is_exact,
    affine2_biases_are_fast_derivable,
    derive_affine2_bias_enabled,
    derived_affine2_biases,
    elide_affine2_bias_enabled,
    elide_validated_k3_biases,
    projection_has_validated_derived_bias,
    reelide_sharded_k3_biases,
    validate_k3_biases_for_load,
)
from mlx_lm.models.switch_layers import QuantizedSwitchLinear


def _bf16_bits(values: list[int]) -> mx.array:
    return mx.array(values, dtype=mx.uint16).view(mx.bfloat16)


class _Projection(nn.Module):
    def __init__(self, scales=None, biases=None):
        super().__init__()
        self.bits = 2
        self.group_size = 128
        self.mode = "affine"
        if scales is not None:
            self.scales = scales
        if biases is not None:
            self.biases = biases


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
        os.environ.pop(ELIDE_AFFINE2_BIAS_ENV, None)
        derive_affine2_bias_enabled.cache_clear()
        elide_affine2_bias_enabled.cache_clear()
        affine2_gather_core_available.cache_clear()

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

    def test_elision_flag_is_strict_and_requires_derive(self):
        for value, expected in (("0", False), ("1", True)):
            with self.subTest(value=value):
                with mock.patch.dict(
                    os.environ,
                    {
                        DERIVE_AFFINE2_BIAS_ENV: "1",
                        ELIDE_AFFINE2_BIAS_ENV: value,
                    },
                    clear=True,
                ):
                    derive_affine2_bias_enabled.cache_clear()
                    elide_affine2_bias_enabled.cache_clear()
                    self.assertEqual(elide_affine2_bias_enabled(), expected)
        with mock.patch.dict(
            os.environ,
            {ELIDE_AFFINE2_BIAS_ENV: "1"},
            clear=True,
        ):
            derive_affine2_bias_enabled.cache_clear()
            elide_affine2_bias_enabled.cache_clear()
            with self.assertRaisesRegex(ValueError, "requires"):
                elide_affine2_bias_enabled()
        for value in ("", "true", "01", "yes", "2"):
            with self.subTest(value=value):
                with mock.patch.dict(
                    os.environ,
                    {
                        DERIVE_AFFINE2_BIAS_ENV: "1",
                        ELIDE_AFFINE2_BIAS_ENV: value,
                    },
                    clear=True,
                ):
                    derive_affine2_bias_enabled.cache_clear()
                    elide_affine2_bias_enabled.cache_clear()
                    with self.assertRaisesRegex(ValueError, "exactly '0' or '1'"):
                        elide_affine2_bias_enabled()

    def test_core_probe_evaluates_kernel_and_fails_closed(self):
        sentinel = object()
        with (
            mock.patch.object(mx.metal, "is_available", return_value=True),
            mock.patch.object(mx, "gather_qmm", return_value=sentinel),
            mock.patch.object(
                mx,
                "eval",
                side_effect=RuntimeError("missing affine2 Metal kernel"),
            ) as evaluate,
        ):
            affine2_gather_core_available.cache_clear()
            self.assertFalse(affine2_gather_core_available())
            evaluate.assert_called_once_with(sentinel)

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

    def test_load_validation_marks_all_exact_projections(self):
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

    def test_load_validation_marks_only_mismatched_down_ineligible(self):
        os.environ[DERIVE_AFFINE2_BIAS_ENV] = "1"
        derive_affine2_bias_enabled.cache_clear()
        layer = _Layer()
        weights = {}
        scales = _bf16_bits([0x3F80, 0xBF80])
        biases = derived_affine2_biases(scales)
        for name in ("gate_proj", "up_proj", "down_proj"):
            prefix = f"model.layers.0.mlp.switch_mlp.{name}"
            weights[f"{prefix}.scales"] = scales
            weights[f"{prefix}.biases"] = biases
        down_prefix = "model.layers.0.mlp.switch_mlp.down_proj"
        weights[f"{down_prefix}.biases"] = _bf16_bits([0x0000, 0x4000])

        self.assertEqual(validate_k3_biases_for_load([layer], weights), 2)
        self.assertTrue(
            projection_has_validated_derived_bias(
                layer.mlp.switch_mlp.gate_proj,
                scales,
                biases,
            )
        )
        self.assertTrue(
            projection_has_validated_derived_bias(
                layer.mlp.switch_mlp.up_proj,
                scales,
                biases,
            )
        )
        self.assertFalse(
            projection_has_validated_derived_bias(
                layer.mlp.switch_mlp.down_proj,
                scales,
                biases,
            )
        )

    def test_load_validation_still_requires_complete_down_metadata(self):
        os.environ[DERIVE_AFFINE2_BIAS_ENV] = "1"
        derive_affine2_bias_enabled.cache_clear()
        prefix = "model.layers.0.mlp.switch_mlp.down_proj"
        with self.assertRaisesRegex(ValueError, "both scales and biases"):
            validate_k3_biases_for_load(
                [_Layer()],
                {f"{prefix}.scales": _bf16_bits([0x3F80])},
            )

    def test_load_validation_marks_nonfast_exact_down_ineligible(self):
        os.environ[DERIVE_AFFINE2_BIAS_ENV] = "1"
        derive_affine2_bias_enabled.cache_clear()
        layer = _Layer()
        prefix = "model.layers.0.mlp.switch_mlp.down_proj"
        scales = _bf16_bits([0x0000, 0x0001, 0x7F80])
        biases = derived_affine2_biases(scales)

        self.assertEqual(
            validate_k3_biases_for_load(
                [layer],
                {
                    f"{prefix}.scales": scales,
                    f"{prefix}.biases": biases,
                },
            ),
            0,
        )
        self.assertFalse(
            projection_has_validated_derived_bias(
                layer.mlp.switch_mlp.down_proj,
                _bf16_bits([0x3F80]),
                _bf16_bits([0xC000]),
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

    def test_elision_aliases_only_authorized_banks(self):
        os.environ[DERIVE_AFFINE2_BIAS_ENV] = "1"
        os.environ[ELIDE_AFFINE2_BIAS_ENV] = "1"
        derive_affine2_bias_enabled.cache_clear()
        elide_affine2_bias_enabled.cache_clear()
        layer = _Layer()
        weights = {}
        original = {}
        for name in ("gate_proj", "up_proj", "down_proj"):
            scales = _bf16_bits([0x3F80, 0xBF80])
            biases = derived_affine2_biases(scales)
            if name == "down_proj":
                biases = _bf16_bits([0x0000, 0x4000])
            projection = _Projection(scales, biases)
            setattr(layer.mlp.switch_mlp, name, projection)
            prefix = f"model.layers.0.mlp.switch_mlp.{name}"
            weights[f"{prefix}.scales"] = scales
            weights[f"{prefix}.biases"] = biases
            original[name] = biases

        self.assertEqual(validate_k3_biases_for_load([layer], weights), 2)
        with mock.patch(
            "mlx_lm.models.kimi_k3_derived_bias.affine2_gather_core_available",
            return_value=True,
        ):
            self.assertEqual(elide_validated_k3_biases([layer]), 2)

        for name in ("gate_proj", "up_proj"):
            projection = getattr(layer.mlp.switch_mlp, name)
            self.assertIs(projection.biases, projection.scales)
            self.assertEqual(projection._runtime_quantization_mode, "affine2")
            self.assertTrue(
                projection_has_validated_derived_bias(
                    projection,
                    projection.scales,
                    projection.biases,
                )
            )
        down = layer.mlp.switch_mlp.down_proj
        self.assertIs(down.biases, original["down_proj"])
        self.assertFalse(hasattr(down, "_runtime_quantization_mode"))

    def test_elision_rejects_metadata_replaced_after_validation(self):
        os.environ[DERIVE_AFFINE2_BIAS_ENV] = "1"
        os.environ[ELIDE_AFFINE2_BIAS_ENV] = "1"
        derive_affine2_bias_enabled.cache_clear()
        elide_affine2_bias_enabled.cache_clear()
        layer = _Layer()
        scales = _bf16_bits([0x3F80])
        biases = derived_affine2_biases(scales)
        projection = _Projection(scales, biases)
        layer.mlp.switch_mlp.gate_proj = projection
        prefix = "model.layers.0.mlp.switch_mlp.gate_proj"
        validate_k3_biases_for_load(
            [layer],
            {f"{prefix}.scales": scales, f"{prefix}.biases": biases},
        )
        with mock.patch(
            "mlx_lm.models.kimi_k3_derived_bias.affine2_gather_core_available",
            return_value=True,
        ):
            self.assertEqual(elide_validated_k3_biases([layer]), 1)
            projection.scales = _bf16_bits([0x4000])
            projection.biases = derived_affine2_biases(projection.scales)
            with self.assertRaisesRegex(RuntimeError, "different metadata"):
                elide_validated_k3_biases([layer])

    def test_shard_realias_skips_never_elided_modules(self):
        os.environ[DERIVE_AFFINE2_BIAS_ENV] = "1"
        os.environ[ELIDE_AFFINE2_BIAS_ENV] = "1"
        derive_affine2_bias_enabled.cache_clear()
        elide_affine2_bias_enabled.cache_clear()
        layer = _Layer()
        scales = _bf16_bits([0x3F80, 0xBF80])
        biases = derived_affine2_biases(scales)
        projection = _Projection(scales, biases)
        layer.mlp.switch_mlp.gate_proj = projection
        prefix = "model.layers.0.mlp.switch_mlp.gate_proj"
        validate_k3_biases_for_load(
            [layer],
            {f"{prefix}.scales": scales, f"{prefix}.biases": biases},
        )
        projection.scales = mx.concatenate([scales])
        projection.biases = mx.concatenate([biases])
        with mock.patch(
            "mlx_lm.models.kimi_k3_derived_bias.affine2_gather_core_available",
            return_value=True,
        ):
            self.assertEqual(reelide_sharded_k3_biases([layer]), 0)
        self.assertIsNot(projection.biases, projection.scales)

    def test_shard_realias_preserves_only_prior_elision(self):
        os.environ[DERIVE_AFFINE2_BIAS_ENV] = "1"
        os.environ[ELIDE_AFFINE2_BIAS_ENV] = "1"
        derive_affine2_bias_enabled.cache_clear()
        elide_affine2_bias_enabled.cache_clear()
        layer = _Layer()
        scales = _bf16_bits([0x3F80, 0xBF80])
        biases = derived_affine2_biases(scales)
        projection = _Projection(scales, biases)
        layer.mlp.switch_mlp.gate_proj = projection
        prefix = "model.layers.0.mlp.switch_mlp.gate_proj"
        validate_k3_biases_for_load(
            [layer],
            {f"{prefix}.scales": scales, f"{prefix}.biases": biases},
        )
        with mock.patch(
            "mlx_lm.models.kimi_k3_derived_bias.affine2_gather_core_available",
            return_value=True,
        ):
            self.assertEqual(elide_validated_k3_biases([layer]), 1)
            projection.scales = mx.concatenate([projection.scales])
            projection.biases = mx.concatenate([projection.biases])
            self.assertEqual(reelide_sharded_k3_biases([layer]), 1)
        self.assertIs(projection.biases, projection.scales)

    def test_quantized_switch_omits_elided_bias_from_core_call(self):
        projection = QuantizedSwitchLinear(
            512,
            8,
            1,
            bias=False,
            group_size=128,
            bits=2,
        )
        projection.biases = projection.scales
        projection._runtime_quantization_mode = "affine2"
        x = mx.zeros((1, 1, 512), dtype=mx.bfloat16)
        indices = mx.zeros((1,), dtype=mx.uint32)
        expected = mx.zeros((1, 1, 8), dtype=mx.bfloat16)
        with mock.patch(
            "mlx_lm.models.switch_layers.mx.gather_qmm",
            return_value=expected,
        ) as gather:
            self.assertIs(projection(x, indices), expected)
        args, kwargs = gather.call_args
        self.assertIsNone(args[3])
        self.assertEqual(kwargs["mode"], "affine2")

    @unittest.skipUnless(
        affine2_gather_core_available(),
        "requires MLX affine2 gather support",
    )
    def test_quantized_switch_affine2_is_bit_exact(self):
        projection = QuantizedSwitchLinear(
            512,
            40,
            8,
            bias=False,
            group_size=128,
            bits=2,
        )
        projection.scales = projection.scales.astype(mx.bfloat16)
        stored_biases = derived_affine2_biases(projection.scales)
        projection.biases = stored_biases
        x = mx.random.normal((40, 1, 512)).astype(mx.bfloat16)
        indices = mx.repeat(mx.arange(8), 5)
        expected = projection(x, indices, sorted_indices=True)

        projection.biases = projection.scales
        projection._runtime_quantization_mode = "affine2"
        actual = projection(x, indices, sorted_indices=True)
        mx.eval(expected, actual)
        self.assertTrue(mx.array_equal(expected, actual).item())


if __name__ == "__main__":
    unittest.main()
