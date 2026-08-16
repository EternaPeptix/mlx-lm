from __future__ import annotations

import os
import unittest
from unittest import mock

import mlx.core as mx
from mlx import nn

from mlx_lm.models.cache import ArraysCache, SpeculativeReplayState
from mlx_lm.models.gated_delta import compute_g_safe
from mlx_lm.models.kimi_k3 import (
    REPLAYSSM_SPECULATIVE_ENV,
    KimiK3DeltaAttention,
    KimiK3ShortConv,
    TextArgs,
)
from mlx_lm.models.kimi_k3_w3_prework import (
    K3_W3_PREWORK_HISTORY_ENV,
    can_use_k3_w3_prework_history,
    fused_k3_w3_prework_history,
    k3_w3_prework_history_enabled,
    maybe_fused_k3_w3_prework_history,
    supports_k3_w3_prework_history,
)


def _assert_exact(
    test: unittest.TestCase,
    actual: mx.array,
    expected: mx.array,
    label: str,
) -> None:
    mx.eval(actual, expected)
    test.assertEqual(actual.shape, expected.shape, f"{label} shape")
    test.assertEqual(actual.dtype, expected.dtype, f"{label} dtype")
    if actual.dtype == mx.bfloat16:
        actual_bits = actual.view(mx.uint16)
        expected_bits = expected.view(mx.uint16)
    elif actual.dtype == mx.float32:
        actual_bits = actual.view(mx.uint32)
        expected_bits = expected.view(mx.uint32)
    else:
        raise AssertionError(f"{label} has unsupported exactness dtype {actual.dtype}")
    equal = mx.array_equal(actual_bits, expected_bits)
    mx.eval(equal)
    if not bool(equal.item()):
        max_abs = mx.max(
            mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32))
        )
        mx.eval(max_abs)
        test.fail(f"{label} raw bits differ; max abs={float(max_abs.item())}")


def _attention(hidden_size: int = 64) -> KimiK3DeltaAttention:
    attention = KimiK3DeltaAttention(
        TextArgs(
            hidden_size=hidden_size,
            num_attention_heads=48,
            num_key_value_heads=48,
            rms_norm_eps=1e-5,
            linear_attn_config={
                "kda_layers": [1],
                "full_attn_layers": [],
                "num_heads": 48,
                "head_dim": 128,
                "short_conv_kernel_size": 4,
                "gate_lower_bound": -5.0,
                "use_full_rank_gate": True,
            },
        ),
        layer_idx=0,
    )
    attention.set_dtype(mx.bfloat16)
    # The released checkpoint keeps these recurrent parameters in FP32 even
    # though affine and convolution weights are BF16/quantized.
    attention.A_log = mx.log(
        mx.random.uniform(low=1.0, high=16.0, shape=(48,)).astype(mx.float32)
    )
    attention.dt_bias = mx.random.normal((48 * 128,)).astype(mx.float32)
    attention.eval()
    return attention


def _stock_prework(
    attention: KimiK3DeltaAttention,
    projected_qkv: mx.array,
    conv_state: mx.array,
    a_logits: mx.array,
) -> tuple[mx.array, ...]:
    qkv, new_state, history = attention.qkv_conv(
        projected_qkv,
        conv_state,
        None,
        None,
        return_state_history=True,
    )
    projection_dim = attention.projection_dim
    q = qkv[..., :projection_dim].reshape(1, 3, 48, 128)
    raw_k = qkv[..., projection_dim : 2 * projection_dim].reshape(1, 3, 48, 128)
    v = qkv[..., 2 * projection_dim :].reshape(1, 3, 48, 128)
    eps = 1e-6 / attention.head_dim
    q = (attention.scale**2) * mx.fast.rms_norm(q, None, eps)
    k = attention.scale * mx.fast.rms_norm(raw_k, None, eps)
    gk = compute_g_safe(
        attention.A_log.reshape(48, 1),
        a_logits,
        attention.dt_bias.reshape(48, 128),
        attention.lower_bound,
    )
    return q, k, raw_k, v, gk, new_state, history


def _make_cache(conv_state: mx.array, ssm_state: mx.array) -> ArraysCache:
    cache = ArraysCache(size=2)
    cache.cache = [conv_state, ssm_state]
    cache.begin_speculative(3)
    return cache


class _InnerConvProxy:
    """Same tensor contract as Conv1d, deliberately different semantics/type."""

    def __init__(self, conv: nn.Conv1d):
        self.weight = conv.weight


class _ShortConvProxy:
    """Same public fields as KimiK3ShortConv, deliberately different type."""

    def __init__(self, conv: KimiK3ShortConv):
        self.conv = conv.conv
        self.kernel_size = conv.kernel_size
        self.training = False


class K3W3PreworkFlagTests(unittest.TestCase):
    def tearDown(self):
        k3_w3_prework_history_enabled.cache_clear()

    def test_flag_is_strict_and_default_off(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            k3_w3_prework_history_enabled.cache_clear()
            self.assertFalse(k3_w3_prework_history_enabled())
        for value, expected in (("0", False), ("1", True)):
            with self.subTest(value=value), mock.patch.dict(
                os.environ,
                {K3_W3_PREWORK_HISTORY_ENV: value},
                clear=True,
            ):
                k3_w3_prework_history_enabled.cache_clear()
                self.assertEqual(k3_w3_prework_history_enabled(), expected)
        for value in ("", "true", "yes", "2", " 1"):
            with self.subTest(value=value), mock.patch.dict(
                os.environ,
                {K3_W3_PREWORK_HISTORY_ENV: value},
                clear=True,
            ):
                k3_w3_prework_history_enabled.cache_clear()
                with self.assertRaisesRegex(ValueError, "exactly '0' or '1'"):
                    k3_w3_prework_history_enabled()


@unittest.skipUnless(mx.metal.is_available(), "requires Metal")
class K3W3PreworkMetalTests(unittest.TestCase):
    def setUp(self):
        self._device = mx.default_device()
        mx.set_default_device(mx.gpu)
        k3_w3_prework_history_enabled.cache_clear()

    def tearDown(self):
        k3_w3_prework_history_enabled.cache_clear()
        mx.set_default_device(self._device)

    def test_production_primitive_is_byte_exact(self):
        cases = (
            (2026081601, 1.0),
            (2026081605, 0.03125),
            (2026081606, 8.0),
            (2026081607, 0.0),
        )
        for seed, amplitude in cases:
            with self.subTest(seed=seed, amplitude=amplitude):
                mx.random.seed(seed)
                attention = _attention()

                def values(shape, amplitude=amplitude):
                    return (
                        mx.random.normal(shape, dtype=mx.bfloat16) * amplitude
                    ).astype(mx.bfloat16)

                projected_qkv = values((1, 3, 3 * 48 * 128))
                conv_state = values((1, 3, 3 * 48 * 128))
                a_logits = values((1, 3, 48, 128))
                mx.eval(
                    attention.qkv_conv.conv.weight,
                    attention.A_log,
                    attention.dt_bias,
                    projected_qkv,
                    conv_state,
                    a_logits,
                )

                stock = _stock_prework(attention, projected_qkv, conv_state, a_logits)
                candidate = fused_k3_w3_prework_history(
                    projected_qkv,
                    conv_state,
                    attention.qkv_conv.conv.weight,
                    a_logits,
                    attention.A_log,
                    attention.dt_bias,
                    num_heads=attention.num_heads,
                    head_dim=attention.head_dim,
                    conv_kernel=attention.conv_kernel,
                    lower_bound=attention.lower_bound,
                )
                mx.eval(*stock, *candidate)

                labels = (
                    "q",
                    "k",
                    "raw_k",
                    "v",
                    "gk",
                    "next_state",
                    "history",
                )
                for label, expected, actual in zip(
                    labels, stock, candidate, strict=True
                ):
                    _assert_exact(self, actual, expected, label)
                for position in range(3):
                    _assert_exact(
                        self,
                        candidate[-1][:, position],
                        stock[-1][:, position],
                        f"history row {position}",
                    )

    def test_unsupported_contracts_return_false(self):
        attention = _attention()
        projected_qkv = mx.zeros((1, 3, 3 * 48 * 128), dtype=mx.bfloat16)
        conv_state = mx.zeros((1, 3, 3 * 48 * 128), dtype=mx.bfloat16)
        a_logits = mx.zeros((1, 3, 48, 128), dtype=mx.bfloat16)
        supported = supports_k3_w3_prework_history(
            projected_qkv,
            conv_state,
            attention.qkv_conv.conv.weight,
            a_logits,
            attention.A_log,
            attention.dt_bias,
            num_heads=48,
            head_dim=128,
            conv_kernel=4,
            lower_bound=-5.0,
        )
        self.assertTrue(supported)

        cases = {
            "width": projected_qkv[:, :2],
            "dtype": projected_qkv.astype(mx.float32),
            "batch": mx.zeros((2, 3, 3 * 48 * 128), dtype=mx.bfloat16),
        }
        for label, value in cases.items():
            with self.subTest(label=label):
                self.assertFalse(
                    supports_k3_w3_prework_history(
                        value,
                        conv_state,
                        attention.qkv_conv.conv.weight,
                        a_logits,
                        attention.A_log,
                        attention.dt_bias,
                        num_heads=48,
                        head_dim=128,
                        conv_kernel=4,
                        lower_bound=-5.0,
                    )
                )

    def test_model_selector_is_production_only_and_default_off(self):
        attention = _attention()
        x = mx.zeros((1, 3, 64), dtype=mx.bfloat16)
        conv_state = mx.zeros((1, 3, 3 * 48 * 128), dtype=mx.bfloat16)
        mx.eval(attention.parameters(), x, conv_state)

        with mock.patch.dict(os.environ, {}, clear=True):
            k3_w3_prework_history_enabled.cache_clear()
            self.assertFalse(
                can_use_k3_w3_prework_history(
                    attention,
                    x,
                    conv_state,
                    short_conv_type=KimiK3ShortConv,
                    inner_conv_type=nn.Conv1d,
                    mask=None,
                    lengths=None,
                    capture_speculative=True,
                )
            )

        with mock.patch.dict(
            os.environ,
            {K3_W3_PREWORK_HISTORY_ENV: "1"},
            clear=False,
        ):
            k3_w3_prework_history_enabled.cache_clear()
            self.assertTrue(
                can_use_k3_w3_prework_history(
                    attention,
                    x,
                    conv_state,
                    short_conv_type=KimiK3ShortConv,
                    inner_conv_type=nn.Conv1d,
                    mask=None,
                    lengths=None,
                    capture_speculative=True,
                )
            )
            unsupported = (
                ("width", x[:, :2], None, None, True),
                ("dtype", x.astype(mx.float32), None, None, True),
                ("mask", x, mx.ones((1, 3), dtype=mx.bool_), None, True),
                ("lengths", x, None, mx.array([3]), True),
                ("not_capture", x, None, None, False),
            )
            for label, value, mask, lengths, capture in unsupported:
                with self.subTest(label=label):
                    self.assertFalse(
                        can_use_k3_w3_prework_history(
                            attention,
                            value,
                            conv_state,
                            short_conv_type=KimiK3ShortConv,
                            inner_conv_type=nn.Conv1d,
                            mask=mask,
                            lengths=lengths,
                            capture_speculative=capture,
                        )
                    )
            attention.train()
            self.assertFalse(
                can_use_k3_w3_prework_history(
                    attention,
                    x,
                    conv_state,
                    short_conv_type=KimiK3ShortConv,
                    inner_conv_type=nn.Conv1d,
                    mask=None,
                    lengths=None,
                    capture_speculative=True,
                )
            )

            for label, mutate in (
                (
                    "scale",
                    lambda candidate: setattr(
                        candidate, "scale", candidate.scale + 1e-6
                    ),
                ),
                (
                    "short_conv_type",
                    lambda candidate: setattr(
                        candidate,
                        "qkv_conv",
                        _ShortConvProxy(candidate.qkv_conv),
                    ),
                ),
                (
                    "inner_conv_type",
                    lambda candidate: setattr(
                        candidate.qkv_conv,
                        "conv",
                        _InnerConvProxy(candidate.qkv_conv.conv),
                    ),
                ),
            ):
                with self.subTest(label=label):
                    candidate = _attention()
                    mutate(candidate)
                    self.assertFalse(
                        can_use_k3_w3_prework_history(
                            candidate,
                            x,
                            conv_state,
                            short_conv_type=KimiK3ShortConv,
                            inner_conv_type=nn.Conv1d,
                            mask=None,
                            lengths=None,
                            capture_speculative=True,
                        )
                    )

    def test_post_projection_contract_drift_fails_closed(self):
        mx.random.seed(2026081608)
        attention = _attention()
        x = mx.random.normal((1, 3, 64), dtype=mx.bfloat16)
        conv_state = mx.random.normal((1, 3, 3 * 48 * 128), dtype=mx.bfloat16)
        ssm_state = mx.random.normal((1, 48, 128, 128), dtype=mx.float32)
        cache = _make_cache(conv_state, ssm_state)
        mx.eval(attention.parameters(), x, conv_state, ssm_state)
        with mock.patch.dict(
            os.environ,
            {
                REPLAYSSM_SPECULATIVE_ENV: "1",
                K3_W3_PREWORK_HISTORY_ENV: "1",
            },
            clear=False,
        ):
            k3_w3_prework_history_enabled.cache_clear()
            with (
                mock.patch(
                    "mlx_lm.models.kimi_k3.maybe_fused_k3_w3_prework_history",
                    return_value=None,
                ) as fused_mock,
                mock.patch(
                    "mlx_lm.models.kimi_k3.record_k3_w3_prework_receipt_decision"
                ) as decision_mock,
                mock.patch(
                    "mlx_lm.models.kimi_k3.record_k3_w3_prework_receipt_outcome"
                ) as outcome_mock,
                self.assertRaisesRegex(RuntimeError, "after selector admission"),
            ):
                attention(x, cache=cache)
            fused_mock.assert_called_once()
            decision_mock.assert_called_once_with(
                x,
                gate_enabled=True,
                admitted=True,
            )
            outcome_mock.assert_called_once_with(success=False)

    def test_receipt_success_is_not_recorded_when_fused_helper_raises(self):
        mx.random.seed(2026081612)
        attention = _attention()
        x = mx.random.normal((1, 3, 64), dtype=mx.bfloat16)
        conv_state = mx.random.normal((1, 3, 3 * 48 * 128), dtype=mx.bfloat16)
        ssm_state = mx.random.normal((1, 48, 128, 128), dtype=mx.float32)
        cache = _make_cache(conv_state, ssm_state)
        mx.eval(attention.parameters(), x, conv_state, ssm_state)
        with mock.patch.dict(
            os.environ,
            {
                REPLAYSSM_SPECULATIVE_ENV: "1",
                K3_W3_PREWORK_HISTORY_ENV: "1",
            },
            clear=False,
        ):
            k3_w3_prework_history_enabled.cache_clear()
            with (
                mock.patch(
                    "mlx_lm.models.kimi_k3.maybe_fused_k3_w3_prework_history",
                    side_effect=RuntimeError("synthetic fused-helper failure"),
                ) as fused_mock,
                mock.patch(
                    "mlx_lm.models.kimi_k3.record_k3_w3_prework_receipt_decision"
                ) as decision_mock,
                mock.patch(
                    "mlx_lm.models.kimi_k3.record_k3_w3_prework_receipt_outcome"
                ) as outcome_mock,
                self.assertRaisesRegex(RuntimeError, "synthetic fused-helper failure"),
            ):
                attention(x, cache=cache)
            fused_mock.assert_called_once()
            decision_mock.assert_called_once_with(
                x,
                gate_enabled=True,
                admitted=True,
            )
            outcome_mock.assert_not_called()

    def test_attention_output_raw_replay_and_prefixes_are_byte_exact(self):
        mx.random.seed(2026081602)
        attention = _attention()
        x = mx.random.normal((1, 3, 64), dtype=mx.bfloat16)
        initial_conv = mx.random.normal((1, 3, 3 * 48 * 128), dtype=mx.bfloat16)
        initial_ssm = mx.random.normal((1, 48, 128, 128), dtype=mx.float32)
        mx.eval(attention.parameters(), x, initial_conv, initial_ssm)

        for consumed in (0, 1, 2, 3):
            with self.subTest(consumed=consumed):
                stock_cache = _make_cache(initial_conv, initial_ssm)
                candidate_cache = _make_cache(initial_conv, initial_ssm)

                with mock.patch.dict(
                    os.environ,
                    {
                        REPLAYSSM_SPECULATIVE_ENV: "1",
                        K3_W3_PREWORK_HISTORY_ENV: "0",
                    },
                    clear=False,
                ):
                    k3_w3_prework_history_enabled.cache_clear()
                    with mock.patch(
                        "mlx_lm.models.kimi_k3.maybe_fused_k3_w3_prework_history",
                        side_effect=AssertionError("default-off path dispatched"),
                    ) as fused_mock:
                        stock_output = attention(x, cache=stock_cache)
                    fused_mock.assert_not_called()
                with mock.patch.dict(
                    os.environ,
                    {
                        REPLAYSSM_SPECULATIVE_ENV: "1",
                        K3_W3_PREWORK_HISTORY_ENV: "1",
                    },
                    clear=False,
                ):
                    k3_w3_prework_history_enabled.cache_clear()
                    with mock.patch(
                        "mlx_lm.models.kimi_k3.maybe_fused_k3_w3_prework_history",
                        wraps=maybe_fused_k3_w3_prework_history,
                    ) as fused_mock:
                        candidate_output = attention(x, cache=candidate_cache)
                    fused_mock.assert_called_once()

                stock_history = stock_cache._speculative_state_history
                candidate_history = candidate_cache._speculative_state_history
                self.assertIsNotNone(stock_history)
                self.assertIsNotNone(candidate_history)
                assert stock_history is not None and candidate_history is not None
                self.assertIsInstance(stock_history[1], SpeculativeReplayState)
                self.assertIsInstance(candidate_history[1], SpeculativeReplayState)

                stock_roots = [stock_output, *stock_cache.cache, stock_history[0]]
                candidate_roots = [
                    candidate_output,
                    *candidate_cache.cache,
                    candidate_history[0],
                ]
                stock_replay = stock_history[1]
                candidate_replay = candidate_history[1]
                assert isinstance(stock_replay, SpeculativeReplayState)
                assert isinstance(candidate_replay, SpeculativeReplayState)
                stock_roots.extend(stock_replay.raw_inputs)
                candidate_roots.extend(candidate_replay.raw_inputs)
                mx.eval(*stock_roots, *candidate_roots)

                labels = (
                    "target_output",
                    "final_conv_state",
                    "final_ssm_state",
                    "conv_history",
                    "replay_v",
                    "replay_raw_k",
                    "replay_gk",
                    "replay_beta",
                )
                for label, expected, actual in zip(
                    labels, stock_roots, candidate_roots, strict=True
                ):
                    _assert_exact(self, actual, expected, label)

                if consumed == 0:
                    stock_cache.cancel_speculative()
                    candidate_cache.cancel_speculative()
                else:
                    stock_cache.resolve_speculative(consumed)
                    candidate_cache.resolve_speculative(consumed)
                mx.eval(*stock_cache.cache, *candidate_cache.cache)
                _assert_exact(
                    self,
                    candidate_cache[0],
                    stock_cache[0],
                    f"committed conv prefix {consumed}",
                )
                _assert_exact(
                    self,
                    candidate_cache[1],
                    stock_cache[1],
                    f"committed SSM prefix {consumed}",
                )

    def test_enabled_candidate_keeps_non_replay_transaction_stock(self):
        mx.random.seed(2026081604)
        attention = _attention()
        x = mx.random.normal((1, 3, 64), dtype=mx.bfloat16)
        conv_state = mx.random.normal((1, 3, 3 * 48 * 128), dtype=mx.bfloat16)
        ssm_state = mx.random.normal((1, 48, 128, 128), dtype=mx.float32)
        cache = _make_cache(conv_state, ssm_state)
        mx.eval(attention.parameters(), x, conv_state, ssm_state)
        with mock.patch.dict(
            os.environ,
            {
                REPLAYSSM_SPECULATIVE_ENV: "0",
                K3_W3_PREWORK_HISTORY_ENV: "1",
            },
            clear=False,
        ):
            k3_w3_prework_history_enabled.cache_clear()
            with mock.patch(
                "mlx_lm.models.kimi_k3.maybe_fused_k3_w3_prework_history",
                side_effect=AssertionError("non-ReplaySSM path dispatched"),
            ) as fused_mock:
                output = attention(x, cache=cache)
            fused_mock.assert_not_called()
        mx.eval(output, *cache.cache)
        self.assertFalse(
            isinstance(cache._speculative_state_history[1], SpeculativeReplayState)
        )


if __name__ == "__main__":
    unittest.main()
