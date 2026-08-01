"""Opt-in adapter for the exact Kimi K3 Metal expert prototype."""

from __future__ import annotations

import os
from functools import lru_cache, partial
from typing import Any

import mlx.core as mx

from .kimi_k3_derived_bias import (
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

FUSED_EXPERT_ENV = "MLX_LM_KIMI_K3_FUSED_EXPERTS"
FUSED_DOWN_REDUCE_ENV = "MLX_LM_KIMI_K3_FUSED_DOWN_REDUCE"
FUSED_EXPERT_WIDTH2_ENV = "MLX_LM_KIMI_K3_FUSED_EXPERT_WIDTH2"


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


def _fused_expert_width_enabled(x: mx.array) -> bool:
    return x.ndim != 3 or x.shape[-2] != 2 or fused_k3_expert_width2_enabled()


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
