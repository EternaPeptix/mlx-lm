# Copyright © 2026 Apple Inc.

import os
import unittest
from unittest import mock

import mlx.core as mx

from mlx_lm.models import kimi_k3_dspark


def _production_arrays(*, context_length=5, dtype=mx.bfloat16):
    mx.random.seed(173)
    return (
        mx.random.normal((1, 64, 7, 64)).astype(dtype),
        mx.random.normal((1, 16, context_length, 64)).astype(dtype),
        mx.random.normal((1, 16, context_length, 64)).astype(dtype),
        mx.random.normal((1, 16, 7, 64)).astype(dtype),
        mx.random.normal((1, 16, 7, 64)).astype(dtype),
    )


def _composite_reference(q, k0, v0, k1, v1, *, scale):
    return mx.fast.scaled_dot_product_attention(
        q,
        mx.concatenate([k0, k1], axis=2),
        mx.concatenate([v0, v1], axis=2),
        scale=scale,
        mask=None,
    )


class TestKimiK3DSparkSegmentedSDPA(unittest.TestCase):
    def test_gate_is_strict_and_default_off(self):
        with mock.patch.dict(os.environ):
            os.environ.pop(kimi_k3_dspark.DSPARK_SEGMENTED_SDPA_ENV, None)
            self.assertFalse(kimi_k3_dspark.kimi_k3_dspark_segmented_sdpa_enabled())

        for value in ("", "true", "yes", "2"):
            with (
                self.subTest(value=value),
                mock.patch.dict(
                    os.environ,
                    {kimi_k3_dspark.DSPARK_SEGMENTED_SDPA_ENV: value},
                ),
                self.assertRaisesRegex(
                    ValueError,
                    kimi_k3_dspark.DSPARK_SEGMENTED_SDPA_ENV,
                ),
            ):
                kimi_k3_dspark.kimi_k3_dspark_segmented_sdpa_enabled()

    def test_disabled_attention_retains_the_accepted_concat_path(self):
        args = kimi_k3_dspark.KimiK3DSparkArgs(
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            vocab_size=16,
            rms_norm_eps=1e-5,
            rope_theta=10_000.0,
            max_position_embeddings=128,
            block_size=2,
            mask_token_id=15,
            target_layer_ids=(0, 1),
            markov_rank=4,
            weight_dtype=mx.float32,
        )
        attention = kimi_k3_dspark.KimiK3DSparkAttention(args)
        target_hidden = mx.arange(24, dtype=mx.float32).reshape(1, 3, 8) / 17
        noise_hidden = mx.arange(16, dtype=mx.float32).reshape(1, 2, 8) / 11
        cache = kimi_k3_dspark.KimiK3DSparkContextCache()
        cache.append(*attention.project_context(target_hidden, offset=0))

        with (
            mock.patch.dict(
                os.environ,
                {kimi_k3_dspark.DSPARK_SEGMENTED_SDPA_ENV: "0"},
            ),
            mock.patch.object(
                kimi_k3_dspark,
                "_kimi_k3_dspark_segmented_sdpa",
                side_effect=AssertionError("segmented path must remain dormant"),
            ) as segmented,
        ):
            output = attention(noise_hidden, block_offset=3, cache=cache)

        mx.eval(output)
        self.assertEqual(tuple(output.shape), (1, 2, 8))
        segmented.assert_not_called()

    def test_enabled_path_passes_disjoint_banks_to_the_feature(self):
        arrays = _production_arrays()
        sentinel = mx.zeros((1, 64, 7, 64), dtype=mx.bfloat16)
        primitive = mock.Mock(return_value=sentinel)
        with (
            mock.patch.object(
                kimi_k3_dspark,
                "_kimi_k3_dspark_segmented_sdpa_primitive",
                return_value=primitive,
            ),
            mock.patch.object(
                kimi_k3_dspark,
                "_kimi_k3_dspark_segmented_sdpa_capabilities_primitive",
                return_value=lambda: ("bounded_memory_metal_v1",),
            ),
        ):
            output = kimi_k3_dspark._kimi_k3_dspark_segmented_sdpa(
                *arrays,
                scale=0.125,
            )

        self.assertIs(output, sentinel)
        primitive.assert_called_once_with(*arrays, scale=0.125)

    def test_fake_primitive_matches_the_real_concat_composite(self):
        arrays = _production_arrays(context_length=11)
        expected = _composite_reference(*arrays, scale=0.125)
        calls = []

        def fake_primitive(q, k0, v0, k1, v1, *, scale):
            calls.append((q, k0, v0, k1, v1, scale))
            return _composite_reference(q, k0, v0, k1, v1, scale=scale)

        with (
            mock.patch.object(
                kimi_k3_dspark,
                "_kimi_k3_dspark_segmented_sdpa_primitive",
                return_value=fake_primitive,
            ),
            mock.patch.object(
                kimi_k3_dspark,
                "_kimi_k3_dspark_segmented_sdpa_capabilities_primitive",
                return_value=lambda: ("bounded_memory_metal_v1",),
            ),
        ):
            actual = kimi_k3_dspark._kimi_k3_dspark_segmented_sdpa(
                *arrays,
                scale=0.125,
            )

        mx.eval(expected, actual)
        self.assertEqual(len(calls), 1)
        self.assertTrue(bool(mx.array_equal(expected, actual).item()))

    def test_enabled_path_fails_closed_without_the_feature(self):
        with (
            mock.patch.object(
                kimi_k3_dspark,
                "_kimi_k3_dspark_segmented_sdpa_primitive",
                return_value=None,
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "mx.fast.segmented_scaled_dot_product_attention",
            ),
        ):
            kimi_k3_dspark._kimi_k3_dspark_segmented_sdpa(
                *_production_arrays(),
                scale=0.125,
            )

    def test_enabled_path_rejects_composite_only_capability(self):
        primitive = mock.Mock()
        with (
            mock.patch.object(
                kimi_k3_dspark,
                "_kimi_k3_dspark_segmented_sdpa_primitive",
                return_value=primitive,
            ),
            mock.patch.object(
                kimi_k3_dspark,
                "_kimi_k3_dspark_segmented_sdpa_capabilities_primitive",
                return_value=lambda: ("composite_v1",),
            ),
            self.assertRaisesRegex(RuntimeError, "bounded_memory_metal_v1"),
        ):
            kimi_k3_dspark._kimi_k3_dspark_segmented_sdpa(
                *_production_arrays(),
                scale=0.125,
            )
        primitive.assert_not_called()

    def test_enabled_path_rejects_invalid_capability_contract(self):
        primitive = mock.Mock()
        with (
            mock.patch.object(
                kimi_k3_dspark,
                "_kimi_k3_dspark_segmented_sdpa_primitive",
                return_value=primitive,
            ),
            mock.patch.object(
                kimi_k3_dspark,
                "_kimi_k3_dspark_segmented_sdpa_capabilities_primitive",
                return_value=lambda: ["bounded_memory_metal_v1"],
            ),
            self.assertRaisesRegex(RuntimeError, "invalid contract"),
        ):
            kimi_k3_dspark._kimi_k3_dspark_segmented_sdpa(
                *_production_arrays(),
                scale=0.125,
            )
        primitive.assert_not_called()

    def test_enabled_path_rejects_nonproduction_geometry_before_dispatch(self):
        production = _production_arrays()
        mismatches = (
            ((*production[:3], production[3][:, :, :6], production[4]), 0.125),
            ((_production_arrays(dtype=mx.float16)), 0.125),
            (production, 0.25),
        )
        for arrays, scale in mismatches:
            with self.subTest(
                shapes=tuple(tuple(array.shape) for array in arrays),
                dtype=arrays[0].dtype,
                scale=scale,
            ):
                primitive = mock.Mock()
                with (
                    mock.patch.object(
                        kimi_k3_dspark,
                        "_kimi_k3_dspark_segmented_sdpa_primitive",
                        return_value=primitive,
                    ),
                    self.assertRaisesRegex(ValueError, "segmented SDPA"),
                ):
                    kimi_k3_dspark._kimi_k3_dspark_segmented_sdpa(
                        *arrays,
                        scale=scale,
                    )
                primitive.assert_not_called()


if __name__ == "__main__":
    unittest.main()
