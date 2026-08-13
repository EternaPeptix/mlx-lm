"""Opt-in adapter for the exact Kimi K3 Metal expert prototype."""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass
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
_WIDTH4_RECEIPT_METRICS = (
    "attempted",
    "supported",
    "dispatched",
    "fallback",
    "error",
)
_WIDTH4_RECEIPT_SELECTOR_ENVS = (
    WIDTH4_DISPATCH_RECEIPT_ENV,
    FUSED_EXPERT_ENV,
    FUSED_DOWN_REDUCE_ENV,
    FUSED_EXPERT_WIDTH4_ENV,
    DERIVE_AFFINE2_BIAS_ENV,
)


@dataclass(slots=True)
class _Width4ReceiptAttempt:
    path: str
    selectors: tuple[tuple[str, str | None], ...]
    committed: bool = False


@dataclass(frozen=True, slots=True)
class _Width4TerminalRecord:
    path: str
    outcome: str
    supported: bool
    reason_class: str | None
    selectors: tuple[tuple[str, str | None], ...]


_width4_receipt_lock = Lock()
_width4_terminal_records: Counter[_Width4TerminalRecord] = Counter()


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


def _begin_width4_dispatch_receipt(
    path: str,
    x: mx.array,
) -> _Width4ReceiptAttempt | None:
    """Capture metadata for an exact width-four attempt without publishing it."""

    if x.ndim != 3 or x.shape[-2] != 4:
        return None
    if not k3_width4_dispatch_receipt_enabled():
        return None
    return _Width4ReceiptAttempt(path, _width4_receipt_selector_state())


def _commit_width4_dispatch_receipt(
    attempt: _Width4ReceiptAttempt | None,
    outcome: str,
    *,
    supported: bool,
    reason_class: str | None = None,
) -> None:
    """Atomically publish exactly one terminal outcome for an attempt."""

    if attempt is None:
        return
    if outcome not in {"dispatched", "fallback", "error"}:
        raise ValueError(f"invalid width-four receipt outcome: {outcome!r}")
    if outcome == "dispatched" and (not supported or reason_class is not None):
        raise ValueError("dispatched receipt must be supported and have no reason")
    if outcome == "fallback" and (supported or reason_class is None):
        raise ValueError("fallback receipt must be unsupported and have a reason")
    if outcome == "error" and reason_class is None:
        raise ValueError("error receipt must have a reason")
    record = _Width4TerminalRecord(
        path=attempt.path,
        outcome=outcome,
        supported=supported,
        reason_class=reason_class,
        selectors=attempt.selectors,
    )
    with _width4_receipt_lock:
        if attempt.committed:
            raise RuntimeError("width-four receipt attempt is already terminal")
        _width4_terminal_records[record] += 1
        attempt.committed = True


def reset_k3_width4_dispatch_receipt() -> None:
    """Reset the current process's aggregate width-four receipt counters."""

    with _width4_receipt_lock:
        _width4_terminal_records.clear()


def snapshot_k3_width4_dispatch_receipt() -> dict[str, Any]:
    """Snapshot aggregate branch receipts without evaluating or syncing Metal.

    The counters are process-local. ``dispatched`` means the selected compiled
    candidate returned an MLX graph value; normal downstream consumption proves
    device execution without adding a receipt-specific synchronization point.
    """

    with _width4_receipt_lock:
        terminal_records = tuple(_width4_terminal_records.items())

    totals: Counter[str] = Counter()
    paths = {path: Counter() for path in _WIDTH4_RECEIPT_PATHS}
    fallback_classes: Counter[str] = Counter()
    error_classes: Counter[str] = Counter()
    selector_states: Counter[tuple[str, tuple[tuple[str, str | None], ...]]] = Counter()
    for record, count in terminal_records:
        totals["attempted"] += count
        totals[record.outcome] += count
        if record.supported:
            totals["supported"] += count
        paths[record.path]["attempted"] += count
        paths[record.path][record.outcome] += count
        if record.supported:
            paths[record.path]["supported"] += count
        if record.outcome == "fallback":
            fallback_classes[record.reason_class] += count
        elif record.outcome == "error":
            error_classes[record.reason_class] += count
        selector_states[(record.path, record.selectors)] += count

    return {
        "schema_version": 2,
        "enabled": k3_width4_dispatch_receipt_enabled(),
        "current_selectors": dict(_width4_receipt_selector_state()),
        "totals": {metric: totals[metric] for metric in _WIDTH4_RECEIPT_METRICS},
        "paths": {
            path: {metric: paths[path][metric] for metric in _WIDTH4_RECEIPT_METRICS}
            for path in _WIDTH4_RECEIPT_PATHS
        },
        "fallback_reason_classes": dict(sorted(fallback_classes.items())),
        "error_reason_classes": dict(sorted(error_classes.items())),
        "selector_states": [
            {
                "path": path,
                "attempted": count,
                "selectors": dict(selector_state),
            }
            for (path, selector_state), count in sorted(
                selector_states.items(),
                key=lambda item: repr(item[0]),
            )
        ],
        "terminal_records": [
            {
                "path": record.path,
                "outcome": record.outcome,
                "supported": record.supported,
                "reason_class": record.reason_class,
                "selectors": dict(record.selectors),
                "count": count,
            }
            for record, count in sorted(
                terminal_records,
                key=lambda item: repr(item[0]),
            )
        ],
    }


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
        attempt = _begin_width4_dispatch_receipt("switch_glu", x)
        supported = False
        try:
            if not supports_width4_switch_situ(x, indices, up, gate):
                _commit_width4_dispatch_receipt(
                    attempt,
                    "fallback",
                    supported=False,
                    reason_class="geometry",
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
                _commit_width4_dispatch_receipt(
                    attempt,
                    "fallback",
                    supported=False,
                    reason_class=reason,
                )
                return None
            if not supports_width4_native_down_projection(indices, down):
                _commit_width4_dispatch_receipt(
                    attempt,
                    "fallback",
                    supported=False,
                    reason_class="geometry",
                )
                return None
            supported = True
            output = _compiled_width4_switch_glu_all_derived(
                x, indices, *up, *gate, *down
            )
        except Exception as error:
            if attempt is not None and not attempt.committed:
                _commit_width4_dispatch_receipt(
                    attempt,
                    "error",
                    supported=supported,
                    reason_class=type(error).__name__,
                )
            raise
        _commit_width4_dispatch_receipt(
            attempt,
            "dispatched",
            supported=True,
        )
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
        attempt = _begin_width4_dispatch_receipt("switch_glu_reduce", x)
        supported = False
        try:
            if not supports_width4_switch_situ(x, indices, up, gate):
                _commit_width4_dispatch_receipt(
                    attempt,
                    "fallback",
                    supported=False,
                    reason_class="geometry",
                )
                return None
            if not supports_width4_down_reduce_projection(
                indices,
                router_weights,
                down,
                results_per_threadgroup=4,
                simdgroups_per_threadgroup=8,
            ):
                _commit_width4_dispatch_receipt(
                    attempt,
                    "fallback",
                    supported=False,
                    reason_class="geometry",
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
                _commit_width4_dispatch_receipt(
                    attempt,
                    "fallback",
                    supported=False,
                    reason_class=reason,
                )
                return None
            supported = True
            output = _compiled_width4_switch_glu_reduce_all_derived(
                x,
                indices,
                router_weights,
                *up,
                *gate,
                *down,
            )
        except Exception as error:
            if attempt is not None and not attempt.committed:
                _commit_width4_dispatch_receipt(
                    attempt,
                    "error",
                    supported=supported,
                    reason_class=type(error).__name__,
                )
            raise
        _commit_width4_dispatch_receipt(
            attempt,
            "dispatched",
            supported=True,
        )
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
