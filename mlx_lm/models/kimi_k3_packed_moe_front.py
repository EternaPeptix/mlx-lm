"""Opt-in exact packing for Kimi K3's decode-time MoE front projections.

Kimi K3 evaluates four affine-quantized projections from the same residual
vector before its routed and shared experts:

* shared-expert gate;
* shared-expert up;
* router scores; and
* routed-expert latent down.

At the single-token decode shape, concatenating their already-quantized output
rows lets MLX issue one QMV instead of four while preserving every output bit.
An independent authoritative-only gate admits the exact ``(1, 3, 7168)`` K3
target-verification shape for a full-pack experiment.  This is intentionally
separate from split packing: the full pack evaluates both the shared and routed
front projections in one QMM.  The branches share that front dependency, but
the model keeps their routed and shared reductions independent afterward so
the shared down projection can still overlap routed communication.
No weight is dequantized or requantized.  An independent, default-off gate can
admit the exact width-eight K3 target-verification shape after its QMM kernel
has been validated; every other multi-token call remains on the stock path.
"""

from __future__ import annotations

import os
from contextvars import ContextVar
from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import accumulate
from threading import Lock
from typing import Any, Sequence

import mlx.core as mx
import mlx.nn as nn

PACKED_MOE_FRONT_ENV = "MLX_LM_KIMI_K3_PACKED_MOE_FRONT"
AUTHORITATIVE_PACKED_MOE_FRONT_ENV = "MLX_LM_KIMI_K3_AUTHORITATIVE_PACKED_MOE_FRONT"
AUTHORITATIVE_PACKED_MOE_FRONT_WIDTH3_ENV = (
    "MLX_LM_KIMI_K3_AUTHORITATIVE_PACKED_MOE_FRONT_WIDTH3"
)
AUTHORITATIVE_PACKED_MOE_FRONT_RECEIPT_ENV = (
    "MLX_LM_KIMI_K3_AUTHORITATIVE_PACKED_MOE_FRONT_RECEIPT"
)
K3_W3_COMPOSITION_RECEIPT_ENV = "MLX_LM_KIMI_K3_W3_COMPOSITION_RECEIPT"
PACKED_MOE_FRONT_WIDTH8_ENV = "MLX_LM_KIMI_K3_PACKED_MOE_FRONT_WIDTH8"
AUTHORITATIVE_PACKED_MOE_FRONT_RECEIPT_SCHEMA = (
    "kimi-k3-authoritative-packed-moe-front-receipt/v2"
)
K3_W3_COMPOSITION_RECEIPT_SCHEMA = "kimi-k3-w3-composition-receipt/v1"
_K3_W3_PREWORK_HISTORY_ENV = "MLX_LM_KIMI_K3_W3_PREWORK_HISTORY"
_REPLAYSSM_SPECULATIVE_ENV = "MLX_LM_KIMI_K3_REPLAYSSM_SPECULATIVE"
_PROJECTED_KV_CACHE_ENV = "MLX_LM_KIMI_K3_PROJECTED_KV_CACHE"
_PROJECTED_KV_CACHE_MAX_TOKENS_ENV = "MLX_LM_KIMI_K3_PROJECTED_KV_CACHE_MAX_TOKENS"
_ASYNC_DECODE_BOUNDARIES_ENV = "MLX_LM_KIMI_K3_ASYNC_DECODE_BOUNDARIES"
_ASYNC_DECODE_STATE_ENV = "MLX_LM_KIMI_K3_ASYNC_DECODE_STATE"
_ASYNC_DECODE_WIDTH3_ENV = "MLX_LM_KIMI_K3_ASYNC_DECODE_WIDTH3"
_NATIVE_Q3_TRIPLET_ENV = "MLX_METAL_K3_AFFINE8_Q3_TRIPLET"
_NATIVE_Q3_DISPATCH_RECEIPT_ENV = "MLX_METAL_K3_AFFINE8_Q3_DISPATCH_RECEIPT"
_UNSUPPORTED = object()
_PACKED_ARRAY_NAMES = ("weight", "scales", "biases", "bias")
_WIDTH3_INPUT_DIMS = 7168
_WIDTH3_OUTPUT_DIMS = (3072, 3072, 896, 3584)
_RECEIPT_COUNTER_LIMIT = 1_000_000_000
_RECEIPT_SEQUENCE_LIMIT = 0x7FFFFFFFFFFFFFFF
_RECEIPT_SEQUENCE = 0
_RECEIPT_SEQUENCE_LOCK = Lock()


@dataclass(frozen=True)
class _PackedFrontReceiptState:
    request_sequence: int
    request_token: int
    expected_layers: int
    expected_kda_layers: int
    pack_count_before: int
    composition: bool
    authoritative_gate_enabled: bool
    width3_gate_enabled: bool
    kda_prework_enabled: bool
    replayssm_speculative_enabled: bool
    projected_kv_cache_enabled: bool
    projected_kv_cache_max_tokens: int
    async_decode_boundaries: str
    async_decode_state: str
    async_decode_width3_enabled: bool
    native_q3_triplet_enabled: bool
    native_q3_dispatch_receipt_enabled: bool
    helper_calls: int = 0
    eligible_width1_calls: int = 0
    eligible_width3_calls: int = 0
    packed_width1_hits: int = 0
    packed_width3_hits: int = 0
    packed_hits: int = 0
    packed_width1_output_tensors: int = 0
    packed_width3_output_tensors: int = 0
    packed_output_tensors: int = 0
    packed_width1_installs: int = 0
    packed_width3_installs: int = 0
    lazy_installs: int = 0
    gate_disabled_calls: int = 0
    noncontract_calls: int = 0
    width1_unsupported_calls: int = 0
    width3_unsupported_calls: int = 0
    unsupported_calls: int = 0
    width1_dispatch_fallback_calls: int = 0
    width3_dispatch_fallback_calls: int = 0
    packed_dispatch_fallback_calls: int = 0
    invalidations: int = 0
    stale_resets: int = 0
    kda_helper_calls: int = 0
    kda_gate_disabled_calls: int = 0
    kda_noncontract_calls: int = 0
    kda_admitted_calls: int = 0
    kda_success_calls: int = 0
    kda_fallback_calls: int = 0
    kda_pending_calls: int = 0


_RECEIPT_STATE: ContextVar[_PackedFrontReceiptState | None] = ContextVar(
    "kimi_k3_authoritative_packed_moe_front_receipt",
    default=None,
)
_RECEIPT_COUNTER_FIELDS = frozenset(
    {
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
        "kda_helper_calls",
        "kda_gate_disabled_calls",
        "kda_noncontract_calls",
        "kda_admitted_calls",
        "kda_success_calls",
        "kda_fallback_calls",
        "kda_pending_calls",
    }
)


def authoritative_packed_moe_front_receipt_enabled() -> bool:
    """Parse the strict, default-off request receipt selector."""

    value = os.environ.get(AUTHORITATIVE_PACKED_MOE_FRONT_RECEIPT_ENV, "0")
    if value not in {"0", "1"}:
        raise ValueError(f"{AUTHORITATIVE_PACKED_MOE_FRONT_RECEIPT_ENV} must be 0 or 1")
    return value == "1"


def k3_w3_composition_receipt_enabled() -> bool:
    """Parse the strict, default-off combined diagnostic selector."""

    value = os.environ.get(K3_W3_COMPOSITION_RECEIPT_ENV, "0")
    if value not in {"0", "1"}:
        raise ValueError(f"{K3_W3_COMPOSITION_RECEIPT_ENV} must be 0 or 1")
    return value == "1"


def _strict_receipt_gate(name: str) -> bool:
    value = os.environ.get(name, "0")
    if value not in {"0", "1"}:
        raise ValueError(f"{name} must be 0 or 1 while receipt capture is active")
    return value == "1"


def _strict_receipt_positive_int(name: str, default: str) -> int:
    value = os.environ.get(name, default)
    if not value or any(character not in "0123456789" for character in value):
        raise ValueError(f"{name} must contain canonical ASCII digits")
    parsed = int(value)
    if str(parsed) != value:
        raise ValueError(f"{name} must use its canonical decimal spelling")
    if parsed < 1 or parsed > _RECEIPT_COUNTER_LIMIT:
        raise ValueError(f"{name} must be a bounded positive integer")
    return parsed


def _composition_selector_snapshot() -> dict[str, str | int | bool]:
    boundaries = os.environ.get(_ASYNC_DECODE_BOUNDARIES_ENV, "none")
    state = os.environ.get(_ASYNC_DECODE_STATE_ENV, "hidden")
    if not isinstance(boundaries, str) or not boundaries:
        raise ValueError(f"{_ASYNC_DECODE_BOUNDARIES_ENV} must be nonempty")
    if not isinstance(state, str) or not state:
        raise ValueError(f"{_ASYNC_DECODE_STATE_ENV} must be nonempty")
    return {
        "authoritative_gate_enabled": _strict_receipt_gate(
            AUTHORITATIVE_PACKED_MOE_FRONT_ENV
        ),
        "width3_gate_enabled": _strict_receipt_gate(
            AUTHORITATIVE_PACKED_MOE_FRONT_WIDTH3_ENV
        ),
        "kda_prework_enabled": _strict_receipt_gate(_K3_W3_PREWORK_HISTORY_ENV),
        "replayssm_speculative_enabled": _strict_receipt_gate(
            _REPLAYSSM_SPECULATIVE_ENV
        ),
        "projected_kv_cache_enabled": _strict_receipt_gate(_PROJECTED_KV_CACHE_ENV),
        "projected_kv_cache_max_tokens": _strict_receipt_positive_int(
            _PROJECTED_KV_CACHE_MAX_TOKENS_ENV,
            "32768",
        ),
        "async_decode_boundaries": boundaries,
        "async_decode_state": state,
        "async_decode_width3_enabled": _strict_receipt_gate(_ASYNC_DECODE_WIDTH3_ENV),
        "native_q3_triplet_enabled": _strict_receipt_gate(_NATIVE_Q3_TRIPLET_ENV),
        "native_q3_dispatch_receipt_enabled": _strict_receipt_gate(
            _NATIVE_Q3_DISPATCH_RECEIPT_ENV
        ),
    }


def _validate_composition_selector_snapshot(
    state: _PackedFrontReceiptState,
) -> None:
    if not state.composition:
        return
    current = _composition_selector_snapshot()
    expected = {
        "authoritative_gate_enabled": state.authoritative_gate_enabled,
        "width3_gate_enabled": state.width3_gate_enabled,
        "kda_prework_enabled": state.kda_prework_enabled,
        "replayssm_speculative_enabled": state.replayssm_speculative_enabled,
        "projected_kv_cache_enabled": state.projected_kv_cache_enabled,
        "projected_kv_cache_max_tokens": state.projected_kv_cache_max_tokens,
        "async_decode_boundaries": state.async_decode_boundaries,
        "async_decode_state": state.async_decode_state,
        "async_decode_width3_enabled": state.async_decode_width3_enabled,
        "native_q3_triplet_enabled": state.native_q3_triplet_enabled,
        "native_q3_dispatch_receipt_enabled": (
            state.native_q3_dispatch_receipt_enabled
        ),
    }
    if current != expected:
        raise RuntimeError("K3 W3 composition selectors changed during receipt capture")


def _next_receipt_sequence() -> int:
    global _RECEIPT_SEQUENCE

    with _RECEIPT_SEQUENCE_LOCK:
        if _RECEIPT_SEQUENCE >= _RECEIPT_SEQUENCE_LIMIT:
            raise OverflowError("packed-front receipt sequence exhausted")
        _RECEIPT_SEQUENCE += 1
        return _RECEIPT_SEQUENCE


def _validate_request_binding(request_sequence: int, request_token: int) -> None:
    if (
        type(request_sequence) is not int
        or not 1 <= request_sequence <= _RECEIPT_SEQUENCE_LIMIT
    ):
        raise ValueError("packed-front receipt sequence must fit positive int64")
    if (
        type(request_token) is not int
        or not 0 <= request_token <= _RECEIPT_SEQUENCE_LIMIT
    ):
        raise ValueError("packed-front receipt token must fit nonnegative int64")


def _validate_expected_receipt_layers(expected_layers: int) -> None:
    if (
        type(expected_layers) is not int
        or not 1 <= expected_layers <= _RECEIPT_COUNTER_LIMIT
    ):
        raise ValueError("packed-front receipt expected layers must be a positive int")


def _model_receipt_layers(model: Any) -> Any:
    try:
        layers = model.language_model.model.layers
    except AttributeError as error:
        raise TypeError(
            "K3 receipt requires model.language_model.model.layers"
        ) from error
    try:
        iter(layers)
    except TypeError as error:
        raise TypeError("K3 receipt model layers are not iterable") from error
    return layers


def _count_model_authoritative_width3_packs(
    model: Any,
    *,
    expected_layers: int,
) -> int:
    """Read installed parents after proving the exact sparse-layer traversal."""

    _validate_expected_receipt_layers(expected_layers)
    iterator = iter(_model_receipt_layers(model))

    count = 0
    relevant_layers = 0
    for layer in iterator:
        sparse_moe = getattr(layer, "mlp", None)
        try:
            modules = _front_modules(sparse_moe)
        except (AttributeError, PackedMoEFrontUnsupported):
            continue
        try:
            production_layout = _production_width3_source_layout_supported(modules)
        except (AttributeError, TypeError, ValueError):
            production_layout = False
        if not production_layout:
            continue
        relevant_layers += 1
        if relevant_layers > expected_layers:
            break
        packed = getattr(
            sparse_moe,
            "_authoritative_packed_k3_moe_front",
            None,
        )
        if not isinstance(packed, AuthoritativePackedK3MoEFront):
            continue
        try:
            active = (
                packed._input_dims == _WIDTH3_INPUT_DIMS
                and packed._output_dims == _WIDTH3_OUTPUT_DIMS
                and packed._production_width3_source_layout
                and packed.matches_sources(modules)
            )
        except (AttributeError, PackedMoEFrontUnsupported, TypeError, ValueError):
            active = False
        if active:
            count += 1
            if count > _RECEIPT_COUNTER_LIMIT:
                raise OverflowError(
                    "packed-front receipt pack count exceeded its bound"
                )
    if relevant_layers != expected_layers:
        raise ValueError(
            "packed-front receipt expected "
            f"{expected_layers} production sparse layers, found {relevant_layers}"
        )
    return count


def _count_model_k3_w3_kda_layers(
    model: Any,
    *,
    expected_layers: int,
) -> int:
    """Prove the released rank-local KDA layer geometry without reading tensors."""

    _validate_expected_receipt_layers(expected_layers)
    count = 0
    for layer in iter(_model_receipt_layers(model)):
        if getattr(layer, "is_linear", False) is not True:
            continue
        attention = getattr(layer, "self_attn", None)
        if attention is None:
            continue
        production_geometry = (
            type(getattr(attention, "num_heads", None)) is int
            and attention.num_heads == 48
            and type(getattr(attention, "head_dim", None)) is int
            and attention.head_dim == 128
            and type(getattr(attention, "conv_kernel", None)) is int
            and attention.conv_kernel == 4
            and getattr(attention, "use_full_rank_gate", None) is True
            and type(getattr(attention, "lower_bound", None)) is float
            and attention.lower_bound == -5.0
        )
        if production_geometry:
            count += 1
            if count > expected_layers:
                break
    if count != expected_layers:
        raise ValueError(
            "K3 W3 composition receipt expected "
            f"{expected_layers} production KDA layers, found {count}"
        )
    return count


def _begin_receipt(
    request_token: int,
    model: Any,
    *,
    expected_layers: int,
    expected_kda_layers: int,
    composition: bool,
) -> tuple[int, int]:
    # Clear first so malformed configuration or model traversal cannot revive
    # counters from an abandoned generator in a reused execution context.
    _RECEIPT_STATE.set(None)
    _validate_request_binding(1, request_token)
    _validate_expected_receipt_layers(expected_layers)
    if expected_kda_layers:
        _validate_expected_receipt_layers(expected_kda_layers)

    if composition:
        selector_snapshot = _composition_selector_snapshot()
    else:
        # Preserve the mature packed-only receipt's selector isolation.  An
        # unrelated combined selector must not change its legacy lifecycle.
        selector_snapshot = {
            "authoritative_gate_enabled": _strict_receipt_gate(
                AUTHORITATIVE_PACKED_MOE_FRONT_ENV
            ),
            "width3_gate_enabled": _strict_receipt_gate(
                AUTHORITATIVE_PACKED_MOE_FRONT_WIDTH3_ENV
            ),
            "kda_prework_enabled": False,
            "replayssm_speculative_enabled": False,
            "projected_kv_cache_enabled": False,
            "projected_kv_cache_max_tokens": 32768,
            "async_decode_boundaries": "none",
            "async_decode_state": "hidden",
            "async_decode_width3_enabled": False,
            "native_q3_triplet_enabled": False,
            "native_q3_dispatch_receipt_enabled": False,
        }
    authoritative_gate_enabled = bool(selector_snapshot["authoritative_gate_enabled"])
    width3_gate_enabled = bool(selector_snapshot["width3_gate_enabled"])
    if authoritative_gate_enabled != width3_gate_enabled:
        raise ValueError(
            "packed-front receipt requires authoritative and width3 gates "
            "to be jointly disabled or jointly enabled"
        )
    if (
        composition
        and authoritative_gate_enabled
        and (
            not selector_snapshot["native_q3_triplet_enabled"]
            or not selector_snapshot["native_q3_dispatch_receipt_enabled"]
        )
    ):
        raise ValueError(
            "K3 W3 composition receipt requires both native Q3 route and "
            "dispatch-receipt selectors when width-three packing is enabled"
        )

    pack_count_before = _count_model_authoritative_width3_packs(
        model,
        expected_layers=expected_layers,
    )
    if expected_kda_layers:
        _count_model_k3_w3_kda_layers(
            model,
            expected_layers=expected_kda_layers,
        )
    request_sequence = _next_receipt_sequence()
    state = _PackedFrontReceiptState(
        request_sequence=request_sequence,
        request_token=request_token,
        expected_layers=expected_layers,
        expected_kda_layers=expected_kda_layers,
        pack_count_before=pack_count_before,
        composition=composition,
        authoritative_gate_enabled=authoritative_gate_enabled,
        width3_gate_enabled=width3_gate_enabled,
        kda_prework_enabled=bool(selector_snapshot["kda_prework_enabled"]),
        replayssm_speculative_enabled=bool(
            selector_snapshot["replayssm_speculative_enabled"]
        ),
        projected_kv_cache_enabled=bool(
            selector_snapshot["projected_kv_cache_enabled"]
        ),
        projected_kv_cache_max_tokens=int(
            selector_snapshot["projected_kv_cache_max_tokens"]
        ),
        async_decode_boundaries=str(selector_snapshot["async_decode_boundaries"]),
        async_decode_state=str(selector_snapshot["async_decode_state"]),
        async_decode_width3_enabled=bool(
            selector_snapshot["async_decode_width3_enabled"]
        ),
        native_q3_triplet_enabled=bool(selector_snapshot["native_q3_triplet_enabled"]),
        native_q3_dispatch_receipt_enabled=bool(
            selector_snapshot["native_q3_dispatch_receipt_enabled"]
        ),
    )
    _RECEIPT_STATE.set(state)
    return request_sequence, request_token


def begin_authoritative_packed_moe_front_receipt(
    request_token: int,
    model: Any,
    *,
    expected_layers: int = 92,
) -> tuple[int, int]:
    """Begin one context-local receipt and snapshot installed parents."""

    _RECEIPT_STATE.set(None)
    if not authoritative_packed_moe_front_receipt_enabled():
        raise RuntimeError("packed-front receipt capture is disabled")
    return _begin_receipt(
        request_token,
        model,
        expected_layers=expected_layers,
        expected_kda_layers=0,
        composition=False,
    )


def begin_k3_w3_composition_receipt(
    request_token: int,
    model: Any,
    *,
    expected_sparse_layers: int = 92,
    expected_kda_layers: int = 69,
) -> tuple[int, int]:
    """Begin one request-local combined receipt after exact model traversal."""

    _RECEIPT_STATE.set(None)
    if not k3_w3_composition_receipt_enabled():
        raise RuntimeError("K3 W3 composition receipt capture is disabled")
    return _begin_receipt(
        request_token,
        model,
        expected_layers=expected_sparse_layers,
        expected_kda_layers=expected_kda_layers,
        composition=True,
    )


def _receipt_state_for_binding(
    request_sequence: int,
    request_token: int,
) -> _PackedFrontReceiptState:
    _validate_request_binding(request_sequence, request_token)
    state = _RECEIPT_STATE.get()
    if state is None:
        raise RuntimeError("no packed-front receipt is active")
    if (
        state.request_sequence != request_sequence
        or state.request_token != request_token
    ):
        raise RuntimeError("packed-front receipt request binding does not match")
    return state


def _bounded_receipt_sum(current: int, amount: int) -> int:
    if type(amount) is not int or amount < 0:
        raise ValueError("packed-front receipt increments must be nonnegative ints")
    updated = current + amount
    if updated > _RECEIPT_COUNTER_LIMIT:
        raise OverflowError("packed-front receipt counter exceeded its bound")
    return updated


def _increment_receipt(**increments: int) -> None:
    state = _RECEIPT_STATE.get()
    if state is None:
        return
    changes: dict[str, int] = {}
    for name, amount in increments.items():
        if name not in _RECEIPT_COUNTER_FIELDS:
            raise KeyError(f"unknown packed-front receipt counter: {name}")
        current = getattr(state, name)
        if type(current) is not int:
            raise TypeError(f"packed-front receipt field {name} is not a counter")
        changes[name] = _bounded_receipt_sum(current, amount)
    _RECEIPT_STATE.set(replace(state, **changes))


def _receipt_helper_started(sparse_moe: Any, x: mx.array, gate_enabled: bool) -> int:
    """Account one helper call and return its eligible width, or zero."""

    state = _RECEIPT_STATE.get()
    if state is None:
        return 0
    _validate_composition_selector_snapshot(state)
    current_gate = _strict_receipt_gate(AUTHORITATIVE_PACKED_MOE_FRONT_ENV)
    current_width3_gate = _strict_receipt_gate(
        AUTHORITATIVE_PACKED_MOE_FRONT_WIDTH3_ENV
    )
    if (
        gate_enabled != current_gate
        or current_gate != state.authoritative_gate_enabled
        or current_width3_gate != state.width3_gate_enabled
    ):
        raise RuntimeError("packed-front selectors changed during receipt capture")
    _increment_receipt(helper_calls=1)
    if not current_gate:
        _increment_receipt(gate_disabled_calls=1)
        return 0
    common_contract = (
        not getattr(sparse_moe, "training", True)
        and x.ndim == 3
        and int(x.shape[0]) == 1
        and int(x.shape[2]) == _WIDTH3_INPUT_DIMS
        and x.dtype == mx.bfloat16
    )
    if common_contract:
        width = int(x.shape[1])
        if width == 1:
            _increment_receipt(eligible_width1_calls=1)
            return width
        if width == 3 and current_width3_gate:
            _increment_receipt(eligible_width3_calls=1)
            return width
    _increment_receipt(noncontract_calls=1)
    return 0


def _receipt_eligible_outcome(
    width: int,
    outcome: str,
    *,
    output_tensors: int = 0,
) -> None:
    state = _RECEIPT_STATE.get()
    if state is None:
        return
    if width not in {1, 3}:
        raise ValueError("packed-front receipt eligible width must be one or three")
    if outcome not in {"packed_hit", "unsupported", "packed_dispatch_fallback"}:
        raise KeyError(f"unknown packed-front receipt outcome: {outcome}")
    width_name = f"width{width}"
    if outcome == "packed_hit":
        increments = {
            "packed_hits": 1,
            f"packed_{width_name}_hits": 1,
        }
    elif outcome == "unsupported":
        increments = {
            "unsupported_calls": 1,
            f"{width_name}_unsupported_calls": 1,
        }
    else:
        increments = {
            "packed_dispatch_fallback_calls": 1,
            f"{width_name}_dispatch_fallback_calls": 1,
        }
    if outcome == "packed_hit":
        increments.update(
            {
                "packed_output_tensors": output_tensors,
                f"packed_{width_name}_output_tensors": output_tensors,
            }
        )
    _increment_receipt(**increments)


def record_k3_w3_prework_receipt_decision(
    x: mx.array,
    *,
    gate_enabled: bool,
    admitted: bool,
) -> None:
    """Partition one production T>1 KDA helper decision inside a receipt."""

    state = _RECEIPT_STATE.get()
    if state is None or not state.composition:
        return
    _validate_composition_selector_snapshot(state)
    if type(gate_enabled) is not bool or type(admitted) is not bool:
        raise TypeError("K3 W3 receipt decisions require exact booleans")
    if x.ndim != 3 or int(x.shape[1]) <= 1:
        return
    if gate_enabled != state.kda_prework_enabled:
        raise RuntimeError("K3 W3 prework selector changed during receipt capture")
    _increment_receipt(kda_helper_calls=1)
    if not gate_enabled:
        if admitted:
            raise RuntimeError("disabled K3 W3 prework cannot be admitted")
        _increment_receipt(kda_gate_disabled_calls=1)
    elif not admitted:
        _increment_receipt(kda_noncontract_calls=1)
    else:
        _increment_receipt(kda_admitted_calls=1, kda_pending_calls=1)


def record_k3_w3_prework_receipt_outcome(*, success: bool) -> None:
    """Settle one admitted KDA call after its fused helper has returned."""

    state = _RECEIPT_STATE.get()
    if state is None or not state.composition:
        return
    _validate_composition_selector_snapshot(state)
    if type(success) is not bool:
        raise TypeError("K3 W3 receipt outcomes require an exact boolean")
    if state.kda_pending_calls < 1:
        raise RuntimeError("K3 W3 receipt has no pending KDA admission")
    changes = {
        "kda_pending_calls": state.kda_pending_calls - 1,
        "kda_success_calls": state.kda_success_calls + int(success),
        "kda_fallback_calls": state.kda_fallback_calls + int(not success),
    }
    if any(value > _RECEIPT_COUNTER_LIMIT for value in changes.values()):
        raise OverflowError("K3 W3 receipt counter exceeded its bound")
    _RECEIPT_STATE.set(replace(state, **changes))


def _validate_packed_receipt_partitions(state: _PackedFrontReceiptState) -> None:
    terminal_total = (
        state.gate_disabled_calls
        + state.noncontract_calls
        + state.packed_hits
        + state.unsupported_calls
        + state.packed_dispatch_fallback_calls
    )
    if terminal_total != state.helper_calls:
        raise RuntimeError("packed-front receipt helper partition is incomplete")
    if state.eligible_width1_calls != (
        state.packed_width1_hits
        + state.width1_unsupported_calls
        + state.width1_dispatch_fallback_calls
    ) or state.eligible_width3_calls != (
        state.packed_width3_hits
        + state.width3_unsupported_calls
        + state.width3_dispatch_fallback_calls
    ):
        raise RuntimeError(
            "packed-front receipt eligible-width partition is incomplete"
        )
    if (
        state.packed_hits != state.packed_width1_hits + state.packed_width3_hits
        or state.packed_output_tensors
        != state.packed_width1_output_tensors + state.packed_width3_output_tensors
        or state.lazy_installs
        != state.packed_width1_installs + state.packed_width3_installs
        or state.unsupported_calls
        != state.width1_unsupported_calls + state.width3_unsupported_calls
        or state.packed_dispatch_fallback_calls
        != state.width1_dispatch_fallback_calls + state.width3_dispatch_fallback_calls
    ):
        raise RuntimeError("packed-front receipt aggregate partition is incomplete")


def _validate_kda_receipt_partitions(state: _PackedFrontReceiptState) -> None:
    if state.kda_helper_calls != (
        state.kda_gate_disabled_calls
        + state.kda_noncontract_calls
        + state.kda_admitted_calls
    ):
        raise RuntimeError("K3 W3 receipt KDA helper partition is incomplete")
    if state.kda_admitted_calls != (
        state.kda_success_calls + state.kda_fallback_calls + state.kda_pending_calls
    ):
        raise RuntimeError("K3 W3 receipt KDA admission partition is incomplete")
    if state.kda_pending_calls:
        raise RuntimeError("K3 W3 receipt has unsettled KDA admissions")


def finish_authoritative_packed_moe_front_receipt(
    request_sequence: int,
    request_token: int,
    model: Any,
) -> dict[str, str | int | bool]:
    """Finalize, deactivate, and return one exact scalar-only receipt."""

    state = _receipt_state_for_binding(request_sequence, request_token)
    _RECEIPT_STATE.set(None)
    pack_count_after = _count_model_authoritative_width3_packs(
        model,
        expected_layers=state.expected_layers,
    )
    _validate_packed_receipt_partitions(state)
    return {
        "schema": AUTHORITATIVE_PACKED_MOE_FRONT_RECEIPT_SCHEMA,
        "request_sequence": state.request_sequence,
        "request_token": state.request_token,
        "expected_layers": state.expected_layers,
        "finalized": True,
        "authoritative_gate_enabled": state.authoritative_gate_enabled,
        "width3_gate_enabled": state.width3_gate_enabled,
        "helper_calls": state.helper_calls,
        "eligible_width1_calls": state.eligible_width1_calls,
        "eligible_width3_calls": state.eligible_width3_calls,
        "packed_width1_hits": state.packed_width1_hits,
        "packed_width3_hits": state.packed_width3_hits,
        "packed_hits": state.packed_hits,
        "packed_width1_output_tensors": state.packed_width1_output_tensors,
        "packed_width3_output_tensors": state.packed_width3_output_tensors,
        "packed_output_tensors": state.packed_output_tensors,
        "packed_width1_installs": state.packed_width1_installs,
        "packed_width3_installs": state.packed_width3_installs,
        "lazy_installs": state.lazy_installs,
        "gate_disabled_calls": state.gate_disabled_calls,
        "noncontract_calls": state.noncontract_calls,
        "width1_unsupported_calls": state.width1_unsupported_calls,
        "width3_unsupported_calls": state.width3_unsupported_calls,
        "unsupported_calls": state.unsupported_calls,
        "width1_dispatch_fallback_calls": state.width1_dispatch_fallback_calls,
        "width3_dispatch_fallback_calls": state.width3_dispatch_fallback_calls,
        "packed_dispatch_fallback_calls": state.packed_dispatch_fallback_calls,
        "invalidations": state.invalidations,
        "stale_resets": state.stale_resets,
        "pack_count_before": state.pack_count_before,
        "pack_count_after": pack_count_after,
    }


def abort_authoritative_packed_moe_front_receipt(
    request_sequence: int,
    request_token: int,
) -> None:
    """Deactivate one receipt without publishing positive telemetry."""

    try:
        _receipt_state_for_binding(request_sequence, request_token)
    finally:
        _RECEIPT_STATE.set(None)


def finish_k3_w3_composition_receipt(
    request_sequence: int,
    request_token: int,
    model: Any,
) -> dict[str, str | int | bool]:
    """Finalize one combined receipt, rejecting mutation or partial accounting."""

    state = _receipt_state_for_binding(request_sequence, request_token)
    try:
        if not state.composition:
            raise RuntimeError("active receipt is not a K3 W3 composition receipt")
        _validate_composition_selector_snapshot(state)
        pack_count_after = _count_model_authoritative_width3_packs(
            model,
            expected_layers=state.expected_layers,
        )
        _count_model_k3_w3_kda_layers(
            model,
            expected_layers=state.expected_kda_layers,
        )
        _validate_packed_receipt_partitions(state)
        _validate_kda_receipt_partitions(state)
        return {
            "schema": K3_W3_COMPOSITION_RECEIPT_SCHEMA,
            "request_sequence": state.request_sequence,
            "request_token": state.request_token,
            "expected_sparse_layers": state.expected_layers,
            "expected_kda_layers": state.expected_kda_layers,
            "finalized": True,
            "aborted": False,
            "poisoned": False,
            "packed_authoritative_enabled": state.authoritative_gate_enabled,
            "packed_width3_enabled": state.width3_gate_enabled,
            "kda_prework_enabled": state.kda_prework_enabled,
            "replayssm_speculative_enabled": state.replayssm_speculative_enabled,
            "projected_kv_cache_enabled": state.projected_kv_cache_enabled,
            "projected_kv_cache_max_tokens": (state.projected_kv_cache_max_tokens),
            "async_decode_boundaries": state.async_decode_boundaries,
            "async_decode_state": state.async_decode_state,
            "async_decode_width3_enabled": state.async_decode_width3_enabled,
            "native_q3_triplet_enabled": state.native_q3_triplet_enabled,
            "native_q3_dispatch_receipt_enabled": (
                state.native_q3_dispatch_receipt_enabled
            ),
            "helper_calls": state.helper_calls,
            "eligible_width1_calls": state.eligible_width1_calls,
            "eligible_width3_calls": state.eligible_width3_calls,
            "packed_width1_hits": state.packed_width1_hits,
            "packed_width3_hits": state.packed_width3_hits,
            "packed_hits": state.packed_hits,
            "packed_width1_output_tensors": state.packed_width1_output_tensors,
            "packed_width3_output_tensors": state.packed_width3_output_tensors,
            "packed_output_tensors": state.packed_output_tensors,
            "packed_width1_installs": state.packed_width1_installs,
            "packed_width3_installs": state.packed_width3_installs,
            "lazy_installs": state.lazy_installs,
            "gate_disabled_calls": state.gate_disabled_calls,
            "noncontract_calls": state.noncontract_calls,
            "width1_unsupported_calls": state.width1_unsupported_calls,
            "width3_unsupported_calls": state.width3_unsupported_calls,
            "unsupported_calls": state.unsupported_calls,
            "width1_dispatch_fallback_calls": (state.width1_dispatch_fallback_calls),
            "width3_dispatch_fallback_calls": (state.width3_dispatch_fallback_calls),
            "packed_dispatch_fallback_calls": (state.packed_dispatch_fallback_calls),
            "invalidations": state.invalidations,
            "stale_resets": state.stale_resets,
            "pack_count_before": state.pack_count_before,
            "pack_count_after": pack_count_after,
            "kda_helper_calls": state.kda_helper_calls,
            "kda_gate_disabled_calls": state.kda_gate_disabled_calls,
            "kda_noncontract_calls": state.kda_noncontract_calls,
            "kda_admitted_calls": state.kda_admitted_calls,
            "kda_success_calls": state.kda_success_calls,
            "kda_fallback_calls": state.kda_fallback_calls,
            "kda_pending_calls": state.kda_pending_calls,
        }
    finally:
        _RECEIPT_STATE.set(None)


def abort_k3_w3_composition_receipt(
    request_sequence: int,
    request_token: int,
) -> None:
    """Deactivate one combined receipt without publishing partial telemetry."""

    try:
        state = _receipt_state_for_binding(request_sequence, request_token)
        if not state.composition:
            raise RuntimeError("active receipt is not a K3 W3 composition receipt")
    finally:
        _RECEIPT_STATE.set(None)


class PackedMoEFrontUnsupported(ValueError):
    """Raised when projections cannot be packed without changing semantics."""


@lru_cache(maxsize=1)
def packed_moe_front_enabled() -> bool:
    return os.environ.get(PACKED_MOE_FRONT_ENV, "0") == "1"


@lru_cache(maxsize=1)
def authoritative_packed_moe_front_enabled() -> bool:
    return os.environ.get(AUTHORITATIVE_PACKED_MOE_FRONT_ENV, "0") == "1"


@lru_cache(maxsize=1)
def authoritative_packed_moe_front_width3_enabled() -> bool:
    value = os.environ.get(AUTHORITATIVE_PACKED_MOE_FRONT_WIDTH3_ENV, "0")
    if value not in {"0", "1"}:
        raise ValueError(f"{AUTHORITATIVE_PACKED_MOE_FRONT_WIDTH3_ENV} must be 0 or 1")
    return value == "1"


@lru_cache(maxsize=1)
def packed_moe_front_width8_enabled() -> bool:
    return os.environ.get(PACKED_MOE_FRONT_WIDTH8_ENV, "0") == "1"


def _packed_input_supported(x: mx.array) -> bool:
    if x.ndim != 3 or x.shape[0] != 1:
        return False
    width = int(x.shape[1])
    return width == 1 or (width == 8 and packed_moe_front_width8_enabled())


def _authoritative_packed_input_supported(x: mx.array) -> bool:
    if _RECEIPT_STATE.get() is None and _packed_input_supported(x):
        # Preserve the pre-receipt default-off helper contract. The exact K3
        # diagnostic path below is active only inside a bound receipt.
        return True
    if (
        x.ndim == 3
        and tuple(int(dim) for dim in x.shape) == (1, 1, _WIDTH3_INPUT_DIMS)
        and x.dtype == mx.bfloat16
    ):
        return True
    return (
        x.ndim == 3
        and tuple(int(dim) for dim in x.shape) == (1, 3, _WIDTH3_INPUT_DIMS)
        and x.dtype == mx.bfloat16
        and authoritative_packed_moe_front_width3_enabled()
    )


def _production_width3_source_layout_supported(modules: Sequence[Any]) -> bool:
    """Return whether four sources match the released TP2 affine8 front."""

    if len(modules) != len(_WIDTH3_OUTPUT_DIMS):
        return False
    for module, output_dims in zip(modules, _WIDTH3_OUTPUT_DIMS, strict=True):
        weight = _array_parameter(module, "weight")
        scales = _array_parameter(module, "scales")
        biases = _array_parameter(module, "biases")
        if weight is None or scales is None or biases is None:
            return False
        if _array_parameter(module, "bias") is not None:
            return False
        if (
            int(getattr(module, "group_size", 0)) != 64
            or int(getattr(module, "bits", 0)) != 8
            or str(getattr(module, "mode", "")) != "affine"
            or weight.dtype != mx.uint32
            or scales.dtype != mx.bfloat16
            or biases.dtype != mx.bfloat16
        ):
            return False
        if tuple(int(dim) for dim in weight.shape) != (
            output_dims,
            _WIDTH3_INPUT_DIMS * 8 // 32,
        ):
            return False
        expected_affine_shape = (output_dims, _WIDTH3_INPUT_DIMS // 64)
        if tuple(int(dim) for dim in scales.shape) != expected_affine_shape:
            return False
        if tuple(int(dim) for dim in biases.shape) != expected_affine_shape:
            return False
    return True


def production_width3_authoritative_front_active(
    sparse_moe: Any,
    x: mx.array,
    optimized_front: Any,
) -> bool:
    """Prove the exact width-three full-pack contract used by MOK overlap."""

    if (
        not isinstance(optimized_front, tuple)
        or len(optimized_front) != len(_WIDTH3_OUTPUT_DIMS)
        or x.ndim != 3
        or tuple(int(dim) for dim in x.shape) != (1, 3, _WIDTH3_INPUT_DIMS)
        or x.dtype != mx.bfloat16
        or not authoritative_packed_moe_front_width3_enabled()
    ):
        return False
    for output, output_dims in zip(
        optimized_front,
        _WIDTH3_OUTPUT_DIMS,
        strict=True,
    ):
        if not isinstance(output, mx.array) or tuple(
            int(dim) for dim in output.shape
        ) != (1, 3, output_dims):
            return False
    try:
        modules = _front_modules(sparse_moe)
    except (AttributeError, PackedMoEFrontUnsupported):
        return False
    if not _production_width3_source_layout_supported(modules):
        return False
    packed = getattr(
        sparse_moe,
        "_authoritative_packed_k3_moe_front",
        None,
    )
    try:
        return (
            packed is not None
            and packed._input_dims == _WIDTH3_INPUT_DIMS
            and packed._output_dims == _WIDTH3_OUTPUT_DIMS
            and packed._production_width3_source_layout
            and packed.matches_sources(modules)
        )
    except (AttributeError, TypeError, ValueError):
        return False


def _array_parameter(module: Any, name: str) -> mx.array | None:
    getter = getattr(module, "get", None)
    value = getter(name) if getter is not None else getattr(module, name, None)
    return value if isinstance(value, mx.array) else None


def _all_or_none(
    modules: Sequence[Any],
    name: str,
) -> list[mx.array] | None:
    values = [_array_parameter(module, name) for module in modules]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise PackedMoEFrontUnsupported(
            f"{name!r} must be present on every projection or none"
        )
    return [value for value in values if value is not None]


def _source_signature(modules: Sequence[Any]) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (
            module,
            _array_parameter(module, "weight"),
            _array_parameter(module, "scales"),
            _array_parameter(module, "biases"),
            _array_parameter(module, "bias"),
            int(getattr(module, "group_size", 0)),
            int(getattr(module, "bits", 0)),
            str(getattr(module, "mode", "")),
        )
        for module in modules
    )


def _same_source_signature(
    expected: tuple[tuple[Any, ...], ...],
    actual: tuple[tuple[Any, ...], ...],
) -> bool:
    if len(expected) != len(actual):
        return False
    for expected_projection, actual_projection in zip(expected, actual, strict=True):
        if any(
            expected_projection[index] is not actual_projection[index]
            for index in range(5)
        ):
            return False
        if expected_projection[5:] != actual_projection[5:]:
            return False
    return True


class PackedK3MoEFront(nn.Module):
    """One lossless quantized projection split into the four original rows."""

    def __init__(self, modules: Sequence[Any]):
        super().__init__()
        if len(modules) != 4:
            raise PackedMoEFrontUnsupported(
                f"expected four MoE-front projections, got {len(modules)}"
            )

        configs = []
        weights = []
        output_dims = []
        input_dims = []
        for module in modules:
            weight = _array_parameter(module, "weight")
            scales = _array_parameter(module, "scales")
            biases = _array_parameter(module, "biases")
            if weight is None or scales is None or biases is None:
                raise PackedMoEFrontUnsupported(
                    "every projection must be affine quantized"
                )
            if weight.ndim != 2 or scales.ndim != 2 or biases.ndim != 2:
                raise PackedMoEFrontUnsupported(
                    "packed weights, scales, and biases must be two-dimensional"
                )

            group_size = int(getattr(module, "group_size", 0))
            bits = int(getattr(module, "bits", 0))
            mode = str(getattr(module, "mode", ""))
            if group_size <= 0 or bits <= 0:
                raise PackedMoEFrontUnsupported(
                    "quantization group size and bit width must be positive"
                )
            configs.append((group_size, bits, mode))
            weights.append(weight)
            output_dims.append(int(weight.shape[0]))
            input_dims.append((int(weight.shape[1]) * 32) // bits)

            if int(scales.shape[0]) != int(weight.shape[0]):
                raise PackedMoEFrontUnsupported(
                    "weight and scale output dimensions differ"
                )
            if tuple(scales.shape) != tuple(biases.shape):
                raise PackedMoEFrontUnsupported(
                    "affine scales and biases have different layouts"
                )
            if int(scales.shape[1]) * group_size != input_dims[-1]:
                raise PackedMoEFrontUnsupported(
                    "packed weight and scale input dimensions differ"
                )

        if len(set(configs)) != 1:
            raise PackedMoEFrontUnsupported(
                "all projections must share one quantization layout"
            )
        group_size, bits, mode = configs[0]
        if (group_size, bits, mode) != (64, 8, "affine"):
            raise PackedMoEFrontUnsupported(
                "K3 MoE-front packing requires affine 8-bit/group-64"
            )
        if len(set(input_dims)) != 1:
            raise PackedMoEFrontUnsupported(
                "all projections must consume the same hidden width"
            )

        scales = _all_or_none(modules, "scales")
        quant_biases = _all_or_none(modules, "biases")
        output_biases = _all_or_none(modules, "bias")
        if scales is None or quant_biases is None:
            raise PackedMoEFrontUnsupported(
                "affine projections require scales and quantization biases"
            )

        self.weight = mx.concatenate(weights, axis=0)
        self.scales = mx.concatenate(scales, axis=0)
        self.biases = mx.concatenate(quant_biases, axis=0)
        if output_biases is not None:
            self.bias = mx.concatenate(output_biases, axis=0)

        self.group_size = group_size
        self.bits = bits
        self.mode = mode
        object.__setattr__(self, "_input_dims", input_dims[0])
        object.__setattr__(
            self,
            "_split_indices",
            tuple(accumulate(output_dims))[:-1],
        )
        object.__setattr__(self, "_output_dims", tuple(output_dims))
        object.__setattr__(
            self,
            "_packed_nbytes",
            sum(int(value.nbytes) for value in (self.weight, self.scales, self.biases))
            + (
                int(self["bias"].nbytes)
                if isinstance(self.get("bias"), mx.array)
                else 0
            ),
        )
        object.__setattr__(self, "_source_signature", _source_signature(modules))
        self.freeze()

    @property
    def packed_nbytes(self) -> int:
        return self._packed_nbytes

    def matches_sources(self, modules: Sequence[Any]) -> bool:
        """Return whether every authoritative projection array is unchanged."""

        return _same_source_signature(
            self._source_signature,
            _source_signature(modules),
        )

    def _project(self, x: mx.array) -> tuple[mx.array, ...]:
        if int(x.shape[-1]) != self._input_dims:
            raise PackedMoEFrontUnsupported(
                f"input width {x.shape[-1]} != packed width {self._input_dims}"
            )

        output = mx.quantized_matmul(
            x,
            self["weight"],
            scales=self["scales"],
            biases=self["biases"],
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
        )
        bias = self.get("bias")
        if isinstance(bias, mx.array):
            output = output + bias
        return tuple(mx.split(output, self._split_indices, axis=-1))

    def __call__(self, x: mx.array) -> tuple[mx.array, ...]:
        if not _packed_input_supported(x):
            raise PackedMoEFrontUnsupported(
                "packed K3 MoE front requires one decode token or an enabled "
                "width-eight target block"
            )
        return self._project(x)


class AuthoritativePackedK3MoEFront(PackedK3MoEFront):
    """A packed QMV whose row views are the source modules' parameters.

    ``PackedK3MoEFront`` retains both the original banks and their
    concatenation.  This variant evaluates the concatenations once, replaces
    the original module arrays with zero-copy row views, and then releases the
    original arrays.  The hidden full-width arrays are therefore the only
    backing allocations while parameter traversal keeps the checkpoint's
    original names and shapes.
    """

    def __init__(self, modules: Sequence[Any]):
        modules = tuple(modules)
        production_width3_layout = _production_width3_source_layout_supported(modules)
        super().__init__(modules)
        object.__setattr__(
            self,
            "_production_width3_source_layout",
            production_width3_layout,
        )

        # Force the concatenations to finish while all source arrays are valid.
        # Packing is a one-time operation, and evaluating here bounds transient
        # duplication to the layer currently being installed.
        mx.eval(self.parameters())

        source_views: dict[str, tuple[mx.array, ...]] = {}
        for name in _PACKED_ARRAY_NAMES:
            packed_array = _array_parameter(self, name)
            if packed_array is not None:
                source_views[name] = tuple(
                    mx.split(packed_array, self._split_indices, axis=0)
                )
        mx.eval(*(value for views in source_views.values() for value in views))

        # Do not mutate a source module until every packed array and every view
        # has evaluated successfully.  Constructor failures remain fail closed.
        for index, module in enumerate(modules):
            for name, views in source_views.items():
                setattr(module, name, views[index])

        # super().__init__ captured the pre-pack arrays. Replace that signature
        # so no reference keeps the superseded allocations alive.
        object.__setattr__(self, "_source_signature", _source_signature(modules))

    def __call__(self, x: mx.array) -> tuple[mx.array, ...]:
        if not _authoritative_packed_input_supported(x):
            raise PackedMoEFrontUnsupported(
                "authoritative packed K3 MoE front requires one decode token, "
                "an enabled exact width-three target block, or an enabled "
                "width-eight target block"
            )
        if int(x.shape[1]) == 3 and (
            self._input_dims != _WIDTH3_INPUT_DIMS
            or self._output_dims != _WIDTH3_OUTPUT_DIMS
            or not self._production_width3_source_layout
        ):
            raise PackedMoEFrontUnsupported(
                "width-three authoritative packing requires the exact released "
                "TP2 front layout"
            )
        return self._project(x)

    def detach_source_views(self) -> None:
        """Give installed source views independent, byte-identical storage.

        Normal loading shards weights before the first decode, so this is only
        needed for a later call to ``Model.shard`` or explicit invalidation.
        A byte-wise xor with zero is used because array construction,
        ``copy.copy``, and contiguous conversion may all preserve the same MLX
        backing allocation.
        """

        replacements: list[tuple[Any, str, mx.array]] = []
        for signature in self._source_signature:
            module = signature[0]
            for name, expected in zip(
                _PACKED_ARRAY_NAMES,
                signature[1:5],
                strict=True,
            ):
                current = _array_parameter(module, name)
                if current is expected and current is not None:
                    copied = mx.bitwise_xor(
                        current.view(mx.uint8),
                        mx.array(0, dtype=mx.uint8),
                    ).view(current.dtype)
                    replacements.append((module, name, copied))

        if replacements:
            mx.eval(*(value for _, _, value in replacements))
        for module, name, value in replacements:
            setattr(module, name, value)


def _front_modules(sparse_moe: Any) -> tuple[Any, Any, Any, Any]:
    shared = getattr(sparse_moe, "shared_experts", None)
    routed_down = getattr(sparse_moe, "routed_expert_down_proj", None)
    if shared is None or routed_down is None:
        raise PackedMoEFrontUnsupported(
            "packed path requires shared experts and latent routed experts"
        )
    return (
        shared.gate_proj,
        shared.up_proj,
        sparse_moe.gate,
        routed_down,
    )


def _build_packed_front(sparse_moe: Any) -> PackedK3MoEFront:
    return PackedK3MoEFront(_front_modules(sparse_moe))


def _build_authoritative_packed_front(
    sparse_moe: Any,
) -> AuthoritativePackedK3MoEFront:
    return AuthoritativePackedK3MoEFront(_front_modules(sparse_moe))


def _drop_authoritative_packed_k3_moe_front(sparse_moe: Any) -> None:
    for name in (
        "_authoritative_packed_k3_moe_front",
        "_authoritative_packed_k3_moe_front_reason",
        "_authoritative_packed_k3_moe_front_source_signature",
    ):
        if hasattr(sparse_moe, name):
            object.__delattr__(sparse_moe, name)


def _drop_duplicating_packed_k3_moe_front(sparse_moe: Any) -> None:
    for name in (
        "_packed_k3_moe_front",
        "_packed_k3_moe_front_reason",
        "_packed_k3_moe_front_source_signature",
    ):
        if hasattr(sparse_moe, name):
            object.__delattr__(sparse_moe, name)


def invalidate_packed_k3_moe_front(sparse_moe: Any) -> None:
    """Drop hidden packed state before sharding or other weight mutation."""

    _increment_receipt(invalidations=1)
    authoritative = getattr(
        sparse_moe,
        "_authoritative_packed_k3_moe_front",
        None,
    )
    if isinstance(authoritative, AuthoritativePackedK3MoEFront):
        authoritative.detach_source_views()
    _drop_authoritative_packed_k3_moe_front(sparse_moe)
    _drop_duplicating_packed_k3_moe_front(sparse_moe)


def maybe_authoritative_packed_k3_moe_front(
    sparse_moe: Any,
    x: mx.array,
) -> tuple[mx.array, ...] | None:
    """Return one native packed QMV backed by authoritative source views."""

    gate_enabled = authoritative_packed_moe_front_enabled()
    receipt_eligible_width = _receipt_helper_started(sparse_moe, x, gate_enabled)
    if not gate_enabled:
        return None

    # Runtime experiment toggles must not leave the older duplicate cache
    # resident beside the authoritative allocation.
    _drop_duplicating_packed_k3_moe_front(sparse_moe)

    if getattr(
        sparse_moe, "training", True
    ) or not _authoritative_packed_input_supported(x):
        return None

    try:
        modules = _front_modules(sparse_moe)
    except (AttributeError, PackedMoEFrontUnsupported):
        invalidate_packed_k3_moe_front(sparse_moe)
        if receipt_eligible_width:
            _receipt_eligible_outcome(receipt_eligible_width, "unsupported")
        return None
    exact_receipt_width1 = _RECEIPT_STATE.get() is not None and int(x.shape[1]) == 1
    if (
        int(x.shape[1]) == 3 or exact_receipt_width1
    ) and not _production_width3_source_layout_supported(modules):
        # A previously installed authoritative parent can outlive a later
        # source mutation.  Release it before falling back so an ineligible
        # width-three call never leaves hidden packed storage resident.
        invalidate_packed_k3_moe_front(sparse_moe)
        if receipt_eligible_width:
            _receipt_eligible_outcome(receipt_eligible_width, "unsupported")
        return None

    packed = getattr(
        sparse_moe,
        "_authoritative_packed_k3_moe_front",
        None,
    )
    stale_packed = None
    current_signature = _source_signature(modules)
    if packed is _UNSUPPORTED:
        unsupported_signature = getattr(
            sparse_moe,
            "_authoritative_packed_k3_moe_front_source_signature",
            (),
        )
        if _same_source_signature(unsupported_signature, current_signature):
            if receipt_eligible_width:
                _receipt_eligible_outcome(receipt_eligible_width, "unsupported")
            return None
        _increment_receipt(stale_resets=1)
        _drop_authoritative_packed_k3_moe_front(sparse_moe)
        packed = None
    elif packed is not None:
        try:
            if not packed.matches_sources(modules):
                # Do not detach unchanged row views here. The replacement pack
                # can consume them directly, then atomically replace all four
                # banks and release the old parent allocation.
                stale_packed = packed
                _increment_receipt(stale_resets=1)
                _drop_authoritative_packed_k3_moe_front(sparse_moe)
                packed = None
        except (AttributeError, TypeError, ValueError):
            if isinstance(packed, AuthoritativePackedK3MoEFront):
                stale_packed = packed
            _increment_receipt(stale_resets=1)
            _drop_authoritative_packed_k3_moe_front(sparse_moe)
            packed = None

    if packed is None:
        try:
            packed = _build_authoritative_packed_front(sparse_moe)
        except (
            AttributeError,
            PackedMoEFrontUnsupported,
            TypeError,
            ValueError,
        ) as exc:
            if isinstance(stale_packed, AuthoritativePackedK3MoEFront):
                stale_packed.detach_source_views()
                current_signature = _source_signature(modules)
            object.__setattr__(
                sparse_moe,
                "_authoritative_packed_k3_moe_front_reason",
                str(exc),
            )
            object.__setattr__(
                sparse_moe,
                "_authoritative_packed_k3_moe_front",
                _UNSUPPORTED,
            )
            object.__setattr__(
                sparse_moe,
                "_authoritative_packed_k3_moe_front_source_signature",
                current_signature,
            )
            if receipt_eligible_width:
                _receipt_eligible_outcome(receipt_eligible_width, "unsupported")
            return None
        object.__setattr__(
            sparse_moe,
            "_authoritative_packed_k3_moe_front",
            packed,
        )
        if receipt_eligible_width:
            _increment_receipt(
                lazy_installs=1,
                **{f"packed_width{receipt_eligible_width}_installs": 1},
            )

    try:
        outputs = packed(x)
    except PackedMoEFrontUnsupported:
        if receipt_eligible_width:
            _receipt_eligible_outcome(
                receipt_eligible_width,
                "packed_dispatch_fallback",
            )
        return None
    if receipt_eligible_width:
        _receipt_eligible_outcome(
            receipt_eligible_width,
            "packed_hit",
            output_tensors=len(outputs),
        )
    return outputs


def maybe_packed_k3_moe_front(
    sparse_moe: Any,
    x: mx.array,
) -> tuple[mx.array, ...] | None:
    """Return four exact decode projections, or ``None`` for the stock path."""

    if (
        not packed_moe_front_enabled()
        or authoritative_packed_moe_front_enabled()
        or getattr(sparse_moe, "training", True)
        or not _packed_input_supported(x)
    ):
        return None

    shared = getattr(sparse_moe, "shared_experts", None)
    routed_down = getattr(sparse_moe, "routed_expert_down_proj", None)
    if shared is None or routed_down is None:
        return None
    packed = getattr(sparse_moe, "_packed_k3_moe_front", None)
    modules = (
        shared.gate_proj,
        shared.up_proj,
        sparse_moe.gate,
        routed_down,
    )
    current_signature = _source_signature(modules)
    if packed is _UNSUPPORTED:
        unsupported_signature = getattr(
            sparse_moe,
            "_packed_k3_moe_front_source_signature",
            (),
        )
        if _same_source_signature(unsupported_signature, current_signature):
            return None
        invalidate_packed_k3_moe_front(sparse_moe)
        packed = None
    elif packed is not None:
        try:
            if not packed.matches_sources(modules):
                invalidate_packed_k3_moe_front(sparse_moe)
                packed = None
        except (AttributeError, TypeError, ValueError):
            invalidate_packed_k3_moe_front(sparse_moe)
            packed = None
    if packed is None:
        try:
            packed = _build_packed_front(sparse_moe)
        except (
            AttributeError,
            PackedMoEFrontUnsupported,
            TypeError,
            ValueError,
        ) as exc:
            object.__setattr__(sparse_moe, "_packed_k3_moe_front_reason", str(exc))
            object.__setattr__(sparse_moe, "_packed_k3_moe_front", _UNSUPPORTED)
            object.__setattr__(
                sparse_moe,
                "_packed_k3_moe_front_source_signature",
                current_signature,
            )
            return None
        # Keep the optimization out of the model parameter tree: the original
        # modules remain authoritative for checkpoint save/load and prefill.
        object.__setattr__(sparse_moe, "_packed_k3_moe_front", packed)

    try:
        return packed(x)
    except PackedMoEFrontUnsupported:
        return None
