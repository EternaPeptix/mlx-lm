from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest import mock

import mlx.core as mx

from mlx_lm.models.kimi_k3 import (
    KimiK3DecoderLayer,
    ResidualBlocks,
    _attn_res_mix,
)
from mlx_lm.models.kimi_k3_attnres_rms import (
    FUSED_ATTNRES_RMS_ENV,
    K3_HIDDEN_SIZE,
    K3_MAX_RESIDUAL_BLOCKS,
    _FUSED_ATTNRES_RMS_SOURCE,
    fused_attnres_rms_enabled,
    maybe_fused_attnres_rms,
    supports_fused_attnres_rms_geometry,
)


EPS = 1e-5


def _inputs(
    residual_count: int = 1,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    raw = mx.random.normal(
        (residual_count, 1, 1, K3_HIDDEN_SIZE),
        dtype=mx.bfloat16,
    )
    raw_float = raw.astype(mx.float32)
    inv_rms = mx.rsqrt((raw_float * raw_float).mean(axis=-1) + EPS)
    partial = mx.random.normal(
        (1, 1, K3_HIDDEN_SIZE),
        dtype=mx.bfloat16,
    )
    w_eff = mx.random.normal((K3_HIDDEN_SIZE,), dtype=mx.float32)
    norm_weight = mx.random.normal(
        (K3_HIDDEN_SIZE,),
        dtype=mx.bfloat16,
    )
    return raw, inv_rms, partial, w_eff, norm_weight


class FusedAttnResRMSContractTest(unittest.TestCase):
    def tearDown(self):
        fused_attnres_rms_enabled.cache_clear()

    def test_released_decode_geometry_is_supported(self):
        for residual_count in (1, 4, K3_MAX_RESIDUAL_BLOCKS):
            with self.subTest(residual_count=residual_count):
                self.assertTrue(
                    supports_fused_attnres_rms_geometry(
                        *_inputs(residual_count),
                        EPS,
                    )
                )

    def test_unsupported_shapes_dtypes_and_epsilon_fail_closed(self):
        raw, inv_rms, partial, w_eff, norm_weight = _inputs()
        cases = (
            (
                mx.zeros(
                    (1, 1, 2, K3_HIDDEN_SIZE),
                    dtype=mx.bfloat16,
                ),
                mx.ones((1, 1, 2), dtype=mx.float32),
                mx.zeros(
                    (1, 2, K3_HIDDEN_SIZE),
                    dtype=mx.bfloat16,
                ),
                w_eff,
                norm_weight,
                EPS,
            ),
            (
                mx.zeros(
                    (K3_MAX_RESIDUAL_BLOCKS + 1, 1, 1, K3_HIDDEN_SIZE),
                    dtype=mx.bfloat16,
                ),
                mx.ones(
                    (K3_MAX_RESIDUAL_BLOCKS + 1, 1, 1),
                    dtype=mx.float32,
                ),
                partial,
                w_eff,
                norm_weight,
                EPS,
            ),
            (
                raw.astype(mx.float32),
                inv_rms,
                partial,
                w_eff,
                norm_weight,
                EPS,
            ),
            (
                raw,
                inv_rms.astype(mx.bfloat16),
                partial,
                w_eff,
                norm_weight,
                EPS,
            ),
            (
                raw,
                inv_rms,
                partial,
                w_eff.astype(mx.bfloat16),
                norm_weight,
                EPS,
            ),
            (
                raw,
                inv_rms,
                partial,
                w_eff,
                norm_weight.astype(mx.float32),
                EPS,
            ),
            (raw, inv_rms, partial, w_eff, norm_weight, 0.0),
        )
        for values in cases:
            with self.subTest(
                raw_shape=values[0].shape,
                partial_shape=values[2].shape,
                raw_dtype=values[0].dtype,
                eps=values[-1],
            ):
                self.assertFalse(supports_fused_attnres_rms_geometry(*values))

    def test_feature_is_default_off(self):
        values = _inputs()
        with mock.patch.dict(os.environ, {FUSED_ATTNRES_RMS_ENV: "0"}):
            fused_attnres_rms_enabled.cache_clear()
            self.assertIsNone(maybe_fused_attnres_rms(*values, EPS))

    def test_decoder_adapter_uses_fused_result_only_during_inference(self):
        raw, inv_rms, partial, w_eff, norm_weight = _inputs()
        blocks = ResidualBlocks(EPS)
        blocks.raw = raw
        blocks.inv_rms = inv_rms
        sentinel = mx.zeros_like(partial)

        class UnusedNorm:
            weight = norm_weight

            def __call__(self, value):
                raise AssertionError("stock RMSNorm should not run")

        layer = SimpleNamespace(training=False, eps=EPS)
        with mock.patch(
            "mlx_lm.models.kimi_k3.maybe_fused_attnres_rms",
            return_value=sentinel,
        ) as fused:
            result = KimiK3DecoderLayer._mix_and_norm(
                layer,
                blocks,
                partial,
                w_eff,
                UnusedNorm(),
            )
        self.assertIs(result, sentinel)
        fused.assert_called_once_with(
            raw,
            inv_rms,
            partial,
            w_eff,
            norm_weight,
            EPS,
        )

    def test_source_preserves_bf16_boundary_before_rms_reduction(self):
        boundary = _FUSED_ATTNRES_RMS_SOURCE.index(
            "mixed[d] = static_cast<InT>(value);"
        )
        reduction = _FUSED_ATTNRES_RMS_SOURCE.index("rms_acc += value * value;")
        self.assertLess(boundary, reduction)
        self.assertIn(
            "constexpr int MIX_NSIMD = MIX_THREADS / 32;",
            _FUSED_ATTNRES_RMS_SOURCE,
        )
        self.assertIn("metal::precise::rsqrt", _FUSED_ATTNRES_RMS_SOURCE)


@unittest.skipUnless(mx.metal.is_available(), "requires Metal")
class FusedAttnResRMSExactnessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_device = mx.default_device()
        mx.set_default_device(mx.gpu)

    @classmethod
    def tearDownClass(cls):
        mx.set_default_device(cls.previous_device)

    def tearDown(self):
        fused_attnres_rms_enabled.cache_clear()

    def test_k3_geometry_matches_stock_attnres_then_rms_bit_for_bit(self):
        with mock.patch.dict(os.environ, {FUSED_ATTNRES_RMS_ENV: "1"}):
            fused_attnres_rms_enabled.cache_clear()
            for seed, residual_count in enumerate(
                (1, 2, 4, K3_MAX_RESIDUAL_BLOCKS),
                start=91,
            ):
                with self.subTest(
                    seed=seed,
                    residual_count=residual_count,
                ):
                    mx.random.seed(seed)
                    raw, inv_rms, partial, w_eff, norm_weight = _inputs(
                        residual_count
                    )
                    blocks = ResidualBlocks(EPS)
                    blocks.raw = raw
                    blocks.inv_rms = inv_rms
                    mixed = _attn_res_mix(
                        blocks,
                        partial,
                        w_eff,
                        EPS,
                        use_kernel=True,
                    )
                    reference = mx.fast.rms_norm(mixed, norm_weight, EPS)
                    candidate = maybe_fused_attnres_rms(
                        raw,
                        inv_rms,
                        partial,
                        w_eff,
                        norm_weight,
                        EPS,
                    )
                    self.assertIsNotNone(candidate)
                    mx.eval(reference, candidate)
                    self.assertTrue(
                        bool(mx.array_equal(reference, candidate).item())
                    )


if __name__ == "__main__":
    unittest.main()
