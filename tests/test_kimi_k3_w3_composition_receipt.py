from __future__ import annotations

import json
import os
import unittest
from contextvars import copy_context
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx

from mlx_lm.models import kimi_k3_packed_moe_front as receipt_module
from mlx_lm.models.kimi_k3_packed_moe_front import (
    AUTHORITATIVE_PACKED_MOE_FRONT_ENV,
    AUTHORITATIVE_PACKED_MOE_FRONT_RECEIPT_ENV,
    AUTHORITATIVE_PACKED_MOE_FRONT_WIDTH3_ENV,
    K3_W3_COMPOSITION_RECEIPT_ENV,
    K3_W3_COMPOSITION_RECEIPT_SCHEMA,
    abort_k3_w3_composition_receipt,
    begin_authoritative_packed_moe_front_receipt,
    begin_k3_w3_composition_receipt,
    finish_authoritative_packed_moe_front_receipt,
    finish_k3_w3_composition_receipt,
    k3_w3_composition_receipt_enabled,
    record_k3_w3_prework_receipt_decision,
    record_k3_w3_prework_receipt_outcome,
)


_KDA_ENV = "MLX_LM_KIMI_K3_W3_PREWORK_HISTORY"
_REPLAY_ENV = "MLX_LM_KIMI_K3_REPLAYSSM_SPECULATIVE"
_PROJECTED_KV_ENV = "MLX_LM_KIMI_K3_PROJECTED_KV_CACHE"
_PROJECTED_KV_MAX_ENV = "MLX_LM_KIMI_K3_PROJECTED_KV_CACHE_MAX_TOKENS"
_ASYNC_BOUNDARIES_ENV = "MLX_LM_KIMI_K3_ASYNC_DECODE_BOUNDARIES"
_ASYNC_STATE_ENV = "MLX_LM_KIMI_K3_ASYNC_DECODE_STATE"
_ASYNC_WIDTH3_ENV = "MLX_LM_KIMI_K3_ASYNC_DECODE_WIDTH3"
_NATIVE_ROUTE_ENV = "MLX_METAL_K3_AFFINE8_Q3_TRIPLET"
_NATIVE_RECEIPT_ENV = "MLX_METAL_K3_AFFINE8_Q3_DISPATCH_RECEIPT"


def _environment(*, packed: bool, kda: bool) -> dict[str, str]:
    return {
        K3_W3_COMPOSITION_RECEIPT_ENV: "1",
        AUTHORITATIVE_PACKED_MOE_FRONT_ENV: "1" if packed else "0",
        AUTHORITATIVE_PACKED_MOE_FRONT_WIDTH3_ENV: "1" if packed else "0",
        _KDA_ENV: "1" if kda else "0",
        _REPLAY_ENV: "1",
        _PROJECTED_KV_ENV: "1",
        _PROJECTED_KV_MAX_ENV: "32768",
        _ASYNC_BOUNDARIES_ENV: "laguna8",
        _ASYNC_STATE_ENV: "hidden",
        _ASYNC_WIDTH3_ENV: "1",
        # Native receipt collection stays symmetric.  These selectors prove
        # route identity only; the native counter is joined by EXO.
        _NATIVE_ROUTE_ENV: "1",
        _NATIVE_RECEIPT_ENV: "1",
    }


def _record_packed_call(*, packed: bool, width: int = 3, install: bool = False):
    x = mx.zeros((1, width, 7168), dtype=mx.bfloat16)
    eligible = receipt_module._receipt_helper_started(
        SimpleNamespace(training=False),
        x,
        packed,
    )
    if packed:
        if install:
            receipt_module._increment_receipt(
                lazy_installs=1,
                **{f"packed_width{width}_installs": 1},
            )
        receipt_module._receipt_eligible_outcome(
            eligible,
            "packed_hit",
            output_tensors=4,
        )


class K3W3CompositionReceiptTests(unittest.TestCase):
    def _assert_failed_legacy_restart_clears_context(
        self,
        selector: str,
        error_type: type[Exception],
        error_pattern: str,
    ) -> None:
        environment = {
            AUTHORITATIVE_PACKED_MOE_FRONT_RECEIPT_ENV: "1",
            AUTHORITATIVE_PACKED_MOE_FRONT_ENV: "0",
            AUTHORITATIVE_PACKED_MOE_FRONT_WIDTH3_ENV: "0",
        }
        with (
            patch.dict(os.environ, environment, clear=True),
            patch.object(
                receipt_module,
                "_count_model_authoritative_width3_packs",
                return_value=0,
            ),
        ):
            abandoned = begin_authoritative_packed_moe_front_receipt(
                71,
                object(),
                expected_layers=1,
            )
            os.environ[AUTHORITATIVE_PACKED_MOE_FRONT_RECEIPT_ENV] = selector
            with self.assertRaisesRegex(error_type, error_pattern):
                begin_authoritative_packed_moe_front_receipt(
                    72,
                    object(),
                    expected_layers=1,
                )
            with self.assertRaisesRegex(RuntimeError, "no packed-front receipt"):
                finish_authoritative_packed_moe_front_receipt(
                    *abandoned,
                    object(),
                )

    def _assert_failed_combined_restart_clears_context(
        self,
        selector: str,
        error_type: type[Exception],
        error_pattern: str,
    ) -> None:
        environment = _environment(packed=False, kda=False)
        with (
            patch.dict(os.environ, environment, clear=True),
            patch.object(
                receipt_module,
                "_count_model_authoritative_width3_packs",
                return_value=0,
            ),
            patch.object(
                receipt_module,
                "_count_model_k3_w3_kda_layers",
                return_value=1,
            ),
        ):
            abandoned = begin_k3_w3_composition_receipt(
                73,
                object(),
                expected_sparse_layers=1,
                expected_kda_layers=1,
            )
            os.environ[K3_W3_COMPOSITION_RECEIPT_ENV] = selector
            with self.assertRaisesRegex(error_type, error_pattern):
                begin_k3_w3_composition_receipt(
                    74,
                    object(),
                    expected_sparse_layers=1,
                    expected_kda_layers=1,
                )
            with self.assertRaisesRegex(RuntimeError, "no packed-front receipt"):
                finish_k3_w3_composition_receipt(
                    *abandoned,
                    object(),
                )

    def test_selector_is_strict_and_default_off(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(k3_w3_composition_receipt_enabled())
            with self.assertRaisesRegex(RuntimeError, "capture is disabled"):
                begin_k3_w3_composition_receipt(1, object())
        with patch.dict(
            os.environ,
            {K3_W3_COMPOSITION_RECEIPT_ENV: "true"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "must be 0 or 1"):
                k3_w3_composition_receipt_enabled()

    def test_disabled_legacy_begin_clears_abandoned_context(self):
        self._assert_failed_legacy_restart_clears_context(
            "0",
            RuntimeError,
            "capture is disabled",
        )

    def test_malformed_legacy_begin_clears_abandoned_context(self):
        self._assert_failed_legacy_restart_clears_context(
            "malformed",
            ValueError,
            "must be 0 or 1",
        )

    def test_disabled_combined_begin_clears_abandoned_context(self):
        self._assert_failed_combined_restart_clears_context(
            "0",
            RuntimeError,
            "capture is disabled",
        )

    def test_malformed_combined_begin_clears_abandoned_context(self):
        self._assert_failed_combined_restart_clears_context(
            "malformed",
            ValueError,
            "must be 0 or 1",
        )

    def test_combined_selectors_do_not_change_legacy_receipt_semantics(self):
        environment = {
            AUTHORITATIVE_PACKED_MOE_FRONT_RECEIPT_ENV: "1",
            AUTHORITATIVE_PACKED_MOE_FRONT_ENV: "0",
            AUTHORITATIVE_PACKED_MOE_FRONT_WIDTH3_ENV: "0",
            K3_W3_COMPOSITION_RECEIPT_ENV: "0",
            _KDA_ENV: "not-a-legacy-selector",
            _PROJECTED_KV_MAX_ENV: "+32768",
            _NATIVE_RECEIPT_ENV: "diagnostic-only",
        }
        with (
            patch.dict(os.environ, environment, clear=True),
            patch.object(
                receipt_module,
                "_count_model_authoritative_width3_packs",
                return_value=0,
            ),
        ):
            handle = begin_authoritative_packed_moe_front_receipt(
                90,
                object(),
                expected_layers=1,
            )
            receipt = finish_authoritative_packed_moe_front_receipt(
                *handle,
                object(),
            )
        self.assertEqual(receipt["request_token"], 90)

    def test_candidate_control_and_independent_subsets_partition_exactly(self):
        for packed, kda in ((False, False), (True, False), (False, True), (True, True)):
            with self.subTest(packed=packed, kda=kda):
                with (
                    patch.dict(
                        os.environ,
                        _environment(packed=packed, kda=kda),
                        clear=True,
                    ),
                    patch.object(
                        receipt_module,
                        "_count_model_authoritative_width3_packs",
                        return_value=1 if packed else 0,
                    ),
                    patch.object(
                        receipt_module,
                        "_count_model_k3_w3_kda_layers",
                        return_value=1,
                    ),
                ):
                    handle = begin_k3_w3_composition_receipt(
                        100 + int(packed) * 10 + int(kda),
                        object(),
                        expected_sparse_layers=1,
                        expected_kda_layers=1,
                    )
                    _record_packed_call(packed=packed)
                    x = mx.zeros((1, 3, 64), dtype=mx.bfloat16)
                    record_k3_w3_prework_receipt_decision(
                        x,
                        gate_enabled=kda,
                        admitted=kda,
                    )
                    if kda:
                        record_k3_w3_prework_receipt_outcome(success=True)
                    receipt = finish_k3_w3_composition_receipt(
                        *handle,
                        object(),
                    )

                self.assertEqual(receipt["schema"], K3_W3_COMPOSITION_RECEIPT_SCHEMA)
                self.assertEqual(receipt["helper_calls"], 1)
                self.assertEqual(receipt["packed_width3_hits"], int(packed))
                self.assertEqual(receipt["gate_disabled_calls"], int(not packed))
                self.assertEqual(receipt["kda_helper_calls"], 1)
                self.assertEqual(receipt["kda_success_calls"], int(kda))
                self.assertEqual(receipt["kda_gate_disabled_calls"], int(not kda))
                self.assertEqual(receipt["kda_pending_calls"], 0)
                self.assertEqual(receipt["aborted"], False)
                self.assertEqual(receipt["poisoned"], False)
                round_trip = json.loads(json.dumps(receipt, sort_keys=True))
                for value in round_trip.values():
                    self.assertIn(type(value), {str, int, bool})

    def test_exact_scalar_key_set_is_stable(self):
        with (
            patch.dict(os.environ, _environment(packed=False, kda=False), clear=True),
            patch.object(
                receipt_module,
                "_count_model_authoritative_width3_packs",
                return_value=0,
            ),
            patch.object(
                receipt_module,
                "_count_model_k3_w3_kda_layers",
                return_value=69,
            ),
        ):
            handle = begin_k3_w3_composition_receipt(200, object())
            receipt = finish_k3_w3_composition_receipt(*handle, object())

        self.assertEqual(
            set(receipt),
            {
                "schema",
                "request_sequence",
                "request_token",
                "expected_sparse_layers",
                "expected_kda_layers",
                "finalized",
                "aborted",
                "poisoned",
                "packed_authoritative_enabled",
                "packed_width3_enabled",
                "kda_prework_enabled",
                "replayssm_speculative_enabled",
                "projected_kv_cache_enabled",
                "projected_kv_cache_max_tokens",
                "async_decode_boundaries",
                "async_decode_state",
                "async_decode_width3_enabled",
                "native_q3_triplet_enabled",
                "native_q3_dispatch_receipt_enabled",
                "helper_calls",
                "eligible_width1_calls",
                "eligible_width3_calls",
                "packed_width1_hits",
                "packed_width3_hits",
                "packed_hits",
                "packed_width1_output_tensors",
                "packed_width3_output_tensors",
                "packed_output_tensors",
                "packed_width1_installs",
                "packed_width3_installs",
                "lazy_installs",
                "gate_disabled_calls",
                "noncontract_calls",
                "width1_unsupported_calls",
                "width3_unsupported_calls",
                "unsupported_calls",
                "width1_dispatch_fallback_calls",
                "width3_dispatch_fallback_calls",
                "packed_dispatch_fallback_calls",
                "invalidations",
                "stale_resets",
                "pack_count_before",
                "pack_count_after",
                "kda_helper_calls",
                "kda_gate_disabled_calls",
                "kda_noncontract_calls",
                "kda_admitted_calls",
                "kda_success_calls",
                "kda_fallback_calls",
                "kda_pending_calls",
            },
        )

    def test_startup_first_installer_may_be_width_one_or_width_three(self):
        for width in (1, 3):
            with self.subTest(width=width):
                with (
                    patch.dict(
                        os.environ,
                        _environment(packed=True, kda=False),
                        clear=True,
                    ),
                    patch.object(
                        receipt_module,
                        "_count_model_authoritative_width3_packs",
                        side_effect=(0, 1),
                    ),
                    patch.object(
                        receipt_module,
                        "_count_model_k3_w3_kda_layers",
                        return_value=1,
                    ),
                ):
                    handle = begin_k3_w3_composition_receipt(
                        300 + width,
                        object(),
                        expected_sparse_layers=1,
                        expected_kda_layers=1,
                    )
                    _record_packed_call(packed=True, width=width, install=True)
                    receipt = finish_k3_w3_composition_receipt(
                        *handle,
                        object(),
                    )
                self.assertEqual(receipt["pack_count_before"], 0)
                self.assertEqual(receipt["pack_count_after"], 1)
                self.assertEqual(receipt[f"packed_width{width}_installs"], 1)
                self.assertEqual(receipt[f"packed_width{width}_hits"], 1)

    def test_kda_noncontract_and_admitted_fallback_are_explicit(self):
        with (
            patch.dict(os.environ, _environment(packed=False, kda=True), clear=True),
            patch.object(
                receipt_module,
                "_count_model_authoritative_width3_packs",
                return_value=0,
            ),
            patch.object(
                receipt_module,
                "_count_model_k3_w3_kda_layers",
                return_value=1,
            ),
        ):
            handle = begin_k3_w3_composition_receipt(
                400,
                object(),
                expected_sparse_layers=1,
                expected_kda_layers=1,
            )
            record_k3_w3_prework_receipt_decision(
                mx.zeros((1, 2, 64), dtype=mx.bfloat16),
                gate_enabled=True,
                admitted=False,
            )
            record_k3_w3_prework_receipt_decision(
                mx.zeros((1, 3, 64), dtype=mx.bfloat16),
                gate_enabled=True,
                admitted=True,
            )
            record_k3_w3_prework_receipt_outcome(success=False)
            receipt = finish_k3_w3_composition_receipt(*handle, object())

        self.assertEqual(receipt["kda_helper_calls"], 2)
        self.assertEqual(receipt["kda_noncontract_calls"], 1)
        self.assertEqual(receipt["kda_admitted_calls"], 1)
        self.assertEqual(receipt["kda_success_calls"], 0)
        self.assertEqual(receipt["kda_fallback_calls"], 1)

    def test_unsettled_kda_admission_fails_closed_and_clears_context(self):
        with (
            patch.dict(os.environ, _environment(packed=False, kda=True), clear=True),
            patch.object(
                receipt_module,
                "_count_model_authoritative_width3_packs",
                return_value=0,
            ),
            patch.object(
                receipt_module,
                "_count_model_k3_w3_kda_layers",
                return_value=1,
            ),
        ):
            handle = begin_k3_w3_composition_receipt(
                401,
                object(),
                expected_sparse_layers=1,
                expected_kda_layers=1,
            )
            record_k3_w3_prework_receipt_decision(
                mx.zeros((1, 3, 64), dtype=mx.bfloat16),
                gate_enabled=True,
                admitted=True,
            )
            with self.assertRaisesRegex(RuntimeError, "unsettled KDA"):
                finish_k3_w3_composition_receipt(*handle, object())
            with self.assertRaisesRegex(RuntimeError, "no packed-front receipt"):
                abort_k3_w3_composition_receipt(*handle)

    def test_every_snapshotted_selector_mutation_fails_closed(self):
        mutations = {
            AUTHORITATIVE_PACKED_MOE_FRONT_ENV: "1",
            AUTHORITATIVE_PACKED_MOE_FRONT_WIDTH3_ENV: "1",
            _KDA_ENV: "1",
            _REPLAY_ENV: "0",
            _PROJECTED_KV_ENV: "0",
            _PROJECTED_KV_MAX_ENV: "65536",
            _ASYNC_BOUNDARIES_ENV: "none",
            _ASYNC_STATE_ENV: "residual",
            _ASYNC_WIDTH3_ENV: "0",
            _NATIVE_ROUTE_ENV: "0",
            _NATIVE_RECEIPT_ENV: "0",
        }
        for name, value in mutations.items():
            with self.subTest(name=name):
                environment = _environment(packed=False, kda=False)
                with (
                    patch.dict(os.environ, environment, clear=True),
                    patch.object(
                        receipt_module,
                        "_count_model_authoritative_width3_packs",
                        return_value=0,
                    ),
                    patch.object(
                        receipt_module,
                        "_count_model_k3_w3_kda_layers",
                        return_value=1,
                    ),
                ):
                    handle = begin_k3_w3_composition_receipt(
                        500,
                        object(),
                        expected_sparse_layers=1,
                        expected_kda_layers=1,
                    )
                    os.environ[name] = value
                    with self.assertRaisesRegex(RuntimeError, "selectors changed"):
                        finish_k3_w3_composition_receipt(*handle, object())

    def test_projected_kv_limit_requires_canonical_decimal_spelling(self):
        for invalid in (" 32768", "+32768", "032768", "32768 ", "3_2768"):
            with self.subTest(invalid=invalid):
                environment = _environment(packed=False, kda=False)
                environment[_PROJECTED_KV_MAX_ENV] = invalid
                with patch.dict(os.environ, environment, clear=True):
                    with self.assertRaisesRegex(ValueError, "canonical"):
                        begin_k3_w3_composition_receipt(600, object())

    def test_packed_pair_requires_both_native_diagnostic_selectors(self):
        for missing in (_NATIVE_ROUTE_ENV, _NATIVE_RECEIPT_ENV):
            with self.subTest(missing=missing):
                environment = _environment(packed=True, kda=False)
                environment[missing] = "0"
                with patch.dict(os.environ, environment, clear=True):
                    with self.assertRaisesRegex(ValueError, "both native Q3"):
                        begin_k3_w3_composition_receipt(601, object())

        for authoritative, width3 in (("0", "1"), ("1", "0")):
            with self.subTest(authoritative=authoritative, width3=width3):
                environment = _environment(packed=False, kda=False)
                environment[AUTHORITATIVE_PACKED_MOE_FRONT_ENV] = authoritative
                environment[AUTHORITATIVE_PACKED_MOE_FRONT_WIDTH3_ENV] = width3
                with patch.dict(os.environ, environment, clear=True):
                    with self.assertRaisesRegex(ValueError, "jointly disabled"):
                        begin_k3_w3_composition_receipt(602, object())

    def test_model_kda_scan_is_exact_and_rechecked_at_finish(self):
        def kda_layer():
            return SimpleNamespace(
                is_linear=True,
                self_attn=SimpleNamespace(
                    num_heads=48,
                    head_dim=128,
                    conv_kernel=4,
                    use_full_rank_gate=True,
                    lower_bound=-5.0,
                ),
            )

        layers = [kda_layer() for _ in range(69)]
        model = SimpleNamespace(
            language_model=SimpleNamespace(
                model=SimpleNamespace(layers=layers),
            )
        )
        self.assertEqual(
            receipt_module._count_model_k3_w3_kda_layers(
                model,
                expected_layers=69,
            ),
            69,
        )
        layers[-1].self_attn.head_dim = 64
        with self.assertRaisesRegex(ValueError, "found 68"):
            receipt_module._count_model_k3_w3_kda_layers(
                model,
                expected_layers=69,
            )

        layers[-1].self_attn.head_dim = 128
        with (
            patch.dict(os.environ, _environment(packed=False, kda=False), clear=True),
            patch.object(
                receipt_module,
                "_count_model_authoritative_width3_packs",
                return_value=0,
            ),
        ):
            handle = begin_k3_w3_composition_receipt(
                650,
                model,
                expected_sparse_layers=1,
                expected_kda_layers=69,
            )
            layers[-1].self_attn.head_dim = 64
            with self.assertRaisesRegex(ValueError, "found 68"):
                finish_k3_w3_composition_receipt(*handle, model)

    def test_abort_and_context_copies_do_not_cross_contaminate(self):
        with (
            patch.dict(os.environ, _environment(packed=False, kda=False), clear=True),
            patch.object(
                receipt_module,
                "_count_model_authoritative_width3_packs",
                return_value=0,
            ),
            patch.object(
                receipt_module,
                "_count_model_k3_w3_kda_layers",
                return_value=1,
            ),
        ):
            first_context = copy_context()
            second_context = copy_context()
            first = first_context.run(
                begin_k3_w3_composition_receipt,
                700,
                object(),
                expected_sparse_layers=1,
                expected_kda_layers=1,
            )
            second = second_context.run(
                begin_k3_w3_composition_receipt,
                701,
                object(),
                expected_sparse_layers=1,
                expected_kda_layers=1,
            )
            first_context.run(abort_k3_w3_composition_receipt, *first)
            with self.assertRaisesRegex(RuntimeError, "no packed-front receipt"):
                first_context.run(
                    finish_k3_w3_composition_receipt,
                    *first,
                    object(),
                )
            second_receipt = second_context.run(
                finish_k3_w3_composition_receipt,
                *second,
                object(),
            )
        self.assertEqual(second_receipt["request_token"], 701)
        self.assertNotEqual(first[0], second[0])


if __name__ == "__main__":
    unittest.main()
