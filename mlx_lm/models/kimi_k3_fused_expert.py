"""Opt-in adapter for the exact Kimi K3 Metal expert prototype."""

from __future__ import annotations

import os
from collections import Counter
from functools import lru_cache, partial
from threading import Lock
from typing import Any

import mlx.core as mx

from .kimi_k3_derived_bias import (
    DERIVE_AFFINE2_BIAS_ENV,
    derive_affine2_bias_enabled,
    projection_has_validated_derived_bias,
)
from .kimi_k3_fused_down_reduce import (
    fused_down_reduce_decode,
    supports_fused_down_reduce,
    supports_fused_down_reduce_projection,
)
from .kimi_k3_fused_switch_glu import (
    fused_switch_situ_decode,
    supports_fused_switch_situ,
)
from .kimi_k3_tuned_gather_qmv import (
    supports_tuned_gather_qmv,
    tuned_gather_qmv,
)
from .kimi_k3_width4_fused_expert import (
    supports_width4_down_reduce_projection,
    supports_width4_native_down_projection,
    supports_width4_switch_situ,
    width4_switch_glu_native_down,
    width4_switch_glu_reduce,
)

FUSED_EXPERT_ENV = "MLX_LM_KIMI_K3_FUSED_EXPERTS"
FUSED_DOWN_REDUCE_ENV = "MLX_LM_KIMI_K3_FUSED_DOWN_REDUCE"
FUSED_EXPERT_WIDTH2_ENV = "MLX_LM_KIMI_K3_FUSED_EXPERT_WIDTH2"
FUSED_EXPERT_WIDTH3_ENV = "MLX_LM_KIMI_K3_FUSED_EXPERT_WIDTH3"
FUSED_EXPERT_WIDTH4_ENV = "MLX_LM_KIMI_K3_FUSED_EXPERT_WIDTH4_EXACT"
WIDTH4_DISPATCH_RECEIPT_ENV = "MLX_LM_KIMI_K3_WIDTH4_DISPATCH_RECEIPT"

_WIDTH4_RECEIPT_PATHS = ("switch_glu", "switch_glu_reduce")
_WIDTH4_RECEIPT_METRICS = ("attempted", "supported", "dispatched", "fallback")
_WIDTH4_RECEIPT_SELECTOR_ENVS = (
    WIDTH4_DISPATCH_RECEIPT_ENV,
    FUSED_EXPERT_ENV,
    FUSED_DOWN_REDUCE_ENV,
    FUSED_EXPERT_WIDTH4_ENV,
    DERIVE_AFFINE2_BIAS_ENV,
)
_width4_receipt_lock = Lock()
_width4_receipt_totals: Counter[str] = Counter()
_width4_receipt_paths: dict[str, Counter[str]] = {
    path: Counter() for path in _WIDTH4_RECEIPT_PATHS
}
_width4_receipt_fallback_classes: Counter[str] = Counter()
_width4_receipt_selector_states: Counter[
    tuple[str, tuple[tuple[str, str | None], ...]]
] = Counter()


@lru_cache(maxsize=1)
def k3_width4_dispatch_receipt_enabled() -> bool:
    """Parse the independent, default-off width-four receipt selector."""

    value = os.environ.get(WIDTH4_DISPATCH_RECEIPT_ENV, "0")
    if value not in {"0", "1"}:
        raise ValueError(f"{WIDTH4_DISPATCH_RECEIPT_ENV} must be exactly '0' or '1'")
    return value == "1"


def _width4_receipt_selector_state() -> tuple[tuple[str, str | None], ...]:
    """Return raw selector values without changing their evaluation order."""

    return tuple((name, os.environ.get(name)) for name in _WIDTH4_RECEIPT_SELECTOR_ENVS)


def _begin_width4_dispatch_receipt(path: str, x: mx.array) -> bool:
    """Start an aggregate receipt only for the exact width-four model path."""

    if x.ndim != 3 or x.shape[-2] != 4:
        return False
    if not k3_width4_dispatch_receipt_enabled():
        return False
    selector_state = _width4_receipt_selector_state()
    with _width4_receipt_lock:
        _width4_receipt_totals["attempted"] += 1
        _width4_receipt_paths[path]["attempted"] += 1
        _width4_receipt_selector_states[(path, selector_state)] += 1
    return True


def _record_width4_event(
    path: str,
    receipt_active: bool,
    event: str,
    fallback_class: str | None = None,
) -> None:
    if not receipt_active:
        return
    with _width4_receipt_lock:
        _width4_receipt_totals[event] += 1
        _width4_receipt_paths[path][event] += 1
        if fallback_class is not None:
            _width4_receipt_fallback_classes[fallback_class] += 1


def _clear_k3_width4_dispatch_receipt() -> None:
    _width4_receipt_totals.clear()
    for counts in _width4_receipt_paths.values():
        counts.clear()
    _width4_receipt_fallback_classes.clear()
    _width4_receipt_selector_states.clear()


def reset_k3_width4_dispatch_receipt() -> None:
    """Reset the current process's aggregate width-four receipt counters."""

    with _width4_receipt_lock:
        _clear_k3_width4_dispatch_receipt()


def snapshot_k3_width4_dispatch_receipt() -> dict[str, Any]:
    """Snapshot aggregate branch receipts without evaluating or syncing Metal.

    The counters are process-local. ``dispatched`` means the selected compiled
    candidate returned an MLX graph value; normal downstream consumption proves
    device execution without adding a receipt-specific synchronization point.
    """

    enabled = k3_width4_dispatch_receipt_enabled()
    current_selectors = dict(_width4_receipt_selector_state())
    with _width4_receipt_lock:
        snapshot = {
            "schema_version": 1,
            "enabled": enabled,
            "current_selectors": current_selectors,
            "totals": {
                metric: _width4_receipt_totals[metric]
                for metric in _WIDTH4_RECEIPT_METRICS
            },
            "paths": {
                path: {
                    metric: _width4_receipt_paths[path][metric]
                    for metric in _WIDTH4_RECEIPT_METRICS
                }
                for path in _WIDTH4_RECEIPT_PATHS
            },
            "fallback_reason_classes": dict(
                sorted(_width4_receipt_fallback_classes.items())
            ),
            "selector_states": [
                {
                    "path": path,
                    "attempted": count,
                    "selectors": dict(selector_state),
                }
                for (path, selector_state), count in sorted(
                    _width4_receipt_selector_states.items(),
                    key=lambda item: repr(item[0]),
                )
            ],
        }
    return snapshot


@partial(mx.compile, shapeless=False)
def _compiled_fused_switch_situ_decode(
    x: mx.array,
    indices: mx.array,
    up_weight: mx.array,
    up_scales: mx.array,
    up_biases: mx.array,
    gate_weight: mx.array,
    gate_scales: mx.array,
    gate_biases: mx.array,
) -> mx.array:
    """Run the fused gate/up stage with shape-specialized dynamic weights."""

    return fused_switch_situ_decode(
        x,
        indices,
        (up_weight, up_scales, up_biases),
        (gate_weight, gate_scales, gate_biases),
        # The real-weight sweep selected this exact tile on M3 Ultra.
        results_per_simdgroup=2,
        simdgroups=4,
        derive_bias=False,
    )


@partial(mx.compile, shapeless=False)
def _compiled_fused_switch_situ_decode_derived_bias(
    x: mx.array,
    indices: mx.array,
    up_weight: mx.array,
    up_scales: mx.array,
    up_biases: mx.array,
    gate_weight: mx.array,
    gate_scales: mx.array,
    gate_biases: mx.array,
) -> mx.array:
    """Run the fused gate/up stage while deriving validated affine biases."""

    return fused_switch_situ_decode(
        x,
        indices,
        (up_weight, up_scales, up_biases),
        (gate_weight, gate_scales, gate_biases),
        results_per_simdgroup=2,
        simdgroups=4,
        derive_bias=True,
    )


@partial(mx.compile, shapeless=False)
def _compiled_width4_switch_glu_all_derived(
    x: mx.array,
    indices: mx.array,
    up_weight: mx.array,
    up_scales: mx.array,
    up_biases: mx.array,
    gate_weight: mx.array,
    gate_scales: mx.array,
    gate_biases: mx.array,
    down_weight: mx.array,
    down_scales: mx.array,
    down_biases: mx.array,
) -> mx.array:
    return width4_switch_glu_native_down(
        x,
        indices,
        (up_weight, up_scales, up_biases),
        (gate_weight, gate_scales, gate_biases),
        (down_weight, down_scales, down_biases),
        front_results_per_simdgroup=8,
        derive_front_bias=True,
    )


@partial(mx.compile, shapeless=False)
def _compiled_width4_switch_glu_reduce_all_derived(
    x: mx.array,
    indices: mx.array,
    router_weights: mx.array,
    up_weight: mx.array,
    up_scales: mx.array,
    up_biases: mx.array,
    gate_weight: mx.array,
    gate_scales: mx.array,
    gate_biases: mx.array,
    down_weight: mx.array,
    down_scales: mx.array,
    down_biases: mx.array,
) -> mx.array:
    """Run the screened width-four front8/down-results4/SIMDs8 chain."""

    return width4_switch_glu_reduce(
        x,
        indices,
        router_weights,
        (up_weight, up_scales, up_biases),
        (gate_weight, gate_scales, gate_biases),
        (down_weight, down_scales, down_biases),
        front_results_per_simdgroup=8,
        down_results_per_threadgroup=4,
        down_simdgroups_per_threadgroup=8,
        derive_front_bias=True,
        derive_down_bias=True,
    )


@partial(mx.compile, shapeless=False)
def _compiled_tuned_gather_qmv(
    x: mx.array,
    indices: mx.array,
    weight: mx.array,
    scales: mx.array,
    biases: mx.array,
) -> mx.array:
    """Run the down stage with shape-specialized dynamic weights."""

    return tuned_gather_qmv(
        x,
        indices,
        (weight, scales, biases),
        results_per_simdgroup=4,
        simdgroups=2,
        broadcast_x=False,
        derive_bias=False,
    )


@partial(mx.compile, shapeless=False)
def _compiled_tuned_gather_qmv_derived_bias(
    x: mx.array,
    indices: mx.array,
    weight: mx.array,
    scales: mx.array,
    biases: mx.array,
) -> mx.array:
    """Run the tuned down stage while deriving validated affine biases."""

    return tuned_gather_qmv(
        x,
        indices,
        (weight, scales, biases),
        results_per_simdgroup=4,
        simdgroups=2,
        broadcast_x=False,
        derive_bias=True,
    )


@partial(mx.compile, shapeless=False)
def _compiled_fused_down_reduce(
    x: mx.array,
    indices: mx.array,
    router_weights: mx.array,
    weight: mx.array,
    scales: mx.array,
    biases: mx.array,
) -> mx.array:
    """Run K3's fused down-QMV and route reduction with dynamic weights."""

    return fused_down_reduce_decode(
        x,
        indices,
        router_weights,
        (weight, scales, biases),
        results_per_threadgroup=4,
        simdgroups_per_threadgroup=16,
        derive_bias=False,
    )


@partial(mx.compile, shapeless=False)
def _compiled_fused_down_reduce_derived_bias(
    x: mx.array,
    indices: mx.array,
    router_weights: mx.array,
    weight: mx.array,
    scales: mx.array,
    biases: mx.array,
) -> mx.array:
    """Run fused down/reduce while deriving validated affine biases."""

    return fused_down_reduce_decode(
        x,
        indices,
        router_weights,
        (weight, scales, biases),
        results_per_threadgroup=4,
        simdgroups_per_threadgroup=16,
        derive_bias=True,
    )


@lru_cache(maxsize=1)
def fused_k3_experts_enabled() -> bool:
    return os.environ.get(FUSED_EXPERT_ENV, "0") == "1"


@lru_cache(maxsize=1)
def fused_k3_down_reduce_enabled() -> bool:
    return os.environ.get(FUSED_DOWN_REDUCE_ENV, "0") == "1"


@lru_cache(maxsize=1)
def fused_k3_expert_width2_enabled() -> bool:
    return os.environ.get(FUSED_EXPERT_WIDTH2_ENV, "0") == "1"


@lru_cache(maxsize=1)
def fused_k3_expert_width3_enabled() -> bool:
    return os.environ.get(FUSED_EXPERT_WIDTH3_ENV, "0") == "1"


@lru_cache(maxsize=1)
def fused_k3_expert_width4_enabled() -> bool:
    """Parse the independent fail-closed width-four selector."""

    value = os.environ.get(FUSED_EXPERT_WIDTH4_ENV, "0")
    if value not in {"0", "1"}:
        raise ValueError(f"{FUSED_EXPERT_WIDTH4_ENV} must be exactly '0' or '1'")
    return value == "1"


def _fused_expert_width_enabled(x: mx.array) -> bool:
    if x.ndim != 3:
        return True
    if x.shape[-2] == 2:
        return fused_k3_expert_width2_enabled()
    if x.shape[-2] == 3:
        return fused_k3_expert_width3_enabled()
    if x.shape[-2] == 4:
        return fused_k3_expert_width4_enabled()
    return True


def _quantized_projection(module: Any):
    if (
        getattr(module, "bits", None) != 2
        or getattr(module, "group_size", None) != 128
        or getattr(module, "mode", None) != "affine"
    ):
        return None
    if "bias" in module:
        # The custom gate/up kernel does not implement an output bias.
        return None
    weight = module["weight"]
    scales = module["scales"]
    biases = module.get("biases")
    if biases is None:
        return None
    return weight, scales, biases


def _all_projections_support_derived_bias(
    modules_and_parts: tuple[tuple[Any, tuple[mx.array, mx.array, mx.array]], ...],
) -> bool:
    """Require every participating bank to pass the exact metadata contract."""

    return all(
        projection_has_validated_derived_bias(module, parts[1], parts[2])
        for module, parts in modules_and_parts
    )


def maybe_fused_k3_switch_glu(
    switch_mlp: Any,
    x: mx.array,
    indices: mx.array,
) -> mx.array | None:
    """Return an exact fused decode/verification result, or ``None``."""

    if (
        not fused_k3_experts_enabled()
        or not _fused_expert_width_enabled(x)
        or getattr(switch_mlp, "training", True)
    ):
        return None
    activation = getattr(switch_mlp, "activation", None)
    if (
        getattr(activation, "beta", None) != 4.0
        or getattr(activation, "linear_beta", None) != 25.0
    ):
        return None
    up = _quantized_projection(switch_mlp.up_proj)
    gate = _quantized_projection(switch_mlp.gate_proj)
    down = _quantized_projection(switch_mlp.down_proj)
    if up is None or gate is None or down is None:
        return None
    if down[0].shape[0] != up[0].shape[0]:
        return None
    if x.ndim == 3 and x.shape[-2] == 4:
        receipt_path = "switch_glu"
        receipt_active = _begin_width4_dispatch_receipt(receipt_path, x)
        if not supports_width4_switch_situ(x, indices, up, gate):
            _record_width4_event(receipt_path, receipt_active, "fallback", "geometry")
            return None
        derive_bias = derive_affine2_bias_enabled()
        if not derive_bias or not _all_projections_support_derived_bias(
            (
                (switch_mlp.up_proj, up),
                (switch_mlp.gate_proj, gate),
                (switch_mlp.down_proj, down),
            )
        ):
            reason = "selector" if not derive_bias else "metadata"
            _record_width4_event(
                receipt_path,
                receipt_active,
                "fallback",
                reason,
            )
            return None
        if not supports_width4_native_down_projection(indices, down):
            _record_width4_event(
                receipt_path,
                receipt_active,
                "fallback",
                "geometry",
            )
            return None
        _record_width4_event(receipt_path, receipt_active, "supported")
        output = _compiled_width4_switch_glu_all_derived(x, indices, *up, *gate, *down)
        _record_width4_event(receipt_path, receipt_active, "dispatched")
        return output
    if not supports_fused_switch_situ(x, indices, up, gate):
        return None
    derive_bias = derive_affine2_bias_enabled()
    if derive_bias and not _all_projections_support_derived_bias(
        (
            (switch_mlp.up_proj, up),
            (switch_mlp.gate_proj, gate),
        )
    ):
        return None

    fused_front = (
        _compiled_fused_switch_situ_decode_derived_bias
        if derive_bias
        else _compiled_fused_switch_situ_decode
    )
    activated = fused_front(x, indices, *up, *gate)
    if not supports_tuned_gather_qmv(
        activated,
        indices,
        down,
        results_per_simdgroup=4,
        simdgroups=2,
        broadcast_x=False,
    ):
        return None
    derive_down_bias = derive_bias and projection_has_validated_derived_bias(
        switch_mlp.down_proj,
        down[1],
        down[2],
    )
    tuned_down = (
        _compiled_tuned_gather_qmv_derived_bias
        if derive_down_bias
        else _compiled_tuned_gather_qmv
    )
    output = tuned_down(activated, indices, *down)
    return output.squeeze(-2)


def maybe_fused_k3_switch_glu_reduce(
    switch_mlp: Any,
    x: mx.array,
    indices: mx.array,
    router_weights: mx.array,
) -> mx.array | None:
    """Return the fully reduced exact decode/verification result, or stock."""

    if (
        not fused_k3_experts_enabled()
        or not fused_k3_down_reduce_enabled()
        or not _fused_expert_width_enabled(x)
        or getattr(switch_mlp, "training", True)
    ):
        return None
    activation = getattr(switch_mlp, "activation", None)
    if (
        getattr(activation, "beta", None) != 4.0
        or getattr(activation, "linear_beta", None) != 25.0
    ):
        return None
    up = _quantized_projection(switch_mlp.up_proj)
    gate = _quantized_projection(switch_mlp.gate_proj)
    down = _quantized_projection(switch_mlp.down_proj)
    if up is None or gate is None or down is None:
        return None
    if down[0].shape[0] != up[0].shape[0]:
        return None
    if x.ndim == 3 and x.shape[-2] == 4:
        receipt_path = "switch_glu_reduce"
        receipt_active = _begin_width4_dispatch_receipt(receipt_path, x)
        if not supports_width4_switch_situ(x, indices, up, gate):
            _record_width4_event(receipt_path, receipt_active, "fallback", "geometry")
            return None
        if not supports_width4_down_reduce_projection(
            indices,
            router_weights,
            down,
            results_per_threadgroup=4,
            simdgroups_per_threadgroup=8,
        ):
            _record_width4_event(
                receipt_path,
                receipt_active,
                "fallback",
                "geometry",
            )
            return None
        derive_bias = derive_affine2_bias_enabled()
        if not derive_bias or not _all_projections_support_derived_bias(
            (
                (switch_mlp.up_proj, up),
                (switch_mlp.gate_proj, gate),
                (switch_mlp.down_proj, down),
            )
        ):
            reason = "selector" if not derive_bias else "metadata"
            _record_width4_event(
                receipt_path,
                receipt_active,
                "fallback",
                reason,
            )
            return None
        _record_width4_event(receipt_path, receipt_active, "supported")
        output = _compiled_width4_switch_glu_reduce_all_derived(
            x,
            indices,
            router_weights,
            *up,
            *gate,
            *down,
        )
        _record_width4_event(receipt_path, receipt_active, "dispatched")
        return output
    if not supports_fused_switch_situ(x, indices, up, gate):
        return None
    if not supports_fused_down_reduce_projection(
        indices,
        router_weights,
        down,
        results_per_threadgroup=4,
        simdgroups_per_threadgroup=16,
    ):
        return None
    derive_bias = derive_affine2_bias_enabled()
    if derive_bias and not _all_projections_support_derived_bias(
        (
            (switch_mlp.up_proj, up),
            (switch_mlp.gate_proj, gate),
        )
    ):
        return None

    fused_front = (
        _compiled_fused_switch_situ_decode_derived_bias
        if derive_bias
        else _compiled_fused_switch_situ_decode
    )
    activated = fused_front(x, indices, *up, *gate)
    if not supports_fused_down_reduce(
        activated,
        indices,
        router_weights,
        down,
        results_per_threadgroup=4,
        simdgroups_per_threadgroup=16,
    ):
        return None
    derive_down_bias = derive_bias and projection_has_validated_derived_bias(
        switch_mlp.down_proj,
        down[1],
        down[2],
    )
    fused_down = (
        _compiled_fused_down_reduce_derived_bias
        if derive_down_bias
        else _compiled_fused_down_reduce
    )
    return fused_down(
        activated,
        indices,
        router_weights,
        *down,
    )
