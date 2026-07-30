"""Opt-in adapter for the exact Kimi K3 Metal expert prototype."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import mlx.core as mx

from .kimi_k3_fused_switch_glu import (
    fused_switch_situ_decode,
    supports_fused_switch_situ,
)
from .kimi_k3_tuned_gather_qmv import (
    supports_tuned_gather_qmv,
    tuned_gather_qmv,
)

FUSED_EXPERT_ENV = "MLX_LM_KIMI_K3_FUSED_EXPERTS"


@lru_cache(maxsize=1)
def fused_k3_experts_enabled() -> bool:
    return os.environ.get(FUSED_EXPERT_ENV, "0") == "1"


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


def maybe_fused_k3_switch_glu(
    switch_mlp: Any,
    x: mx.array,
    indices: mx.array,
) -> mx.array | None:
    """Return an exact fused decode result, or ``None`` for the stock path."""

    if not fused_k3_experts_enabled() or getattr(switch_mlp, "training", True):
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
    if not supports_fused_switch_situ(x, indices, up, gate):
        return None

    activated = fused_switch_situ_decode(
        x,
        indices,
        up,
        gate,
        # The real-weight sweep selected this exact tile on M3 Ultra.
        results_per_simdgroup=2,
        simdgroups=4,
    )
    if not supports_tuned_gather_qmv(
        activated,
        indices,
        down,
        results_per_simdgroup=4,
        simdgroups=2,
        broadcast_x=False,
    ):
        return None
    output = tuned_gather_qmv(
        activated,
        indices,
        down,
        results_per_simdgroup=4,
        simdgroups=2,
        broadcast_x=False,
    )
    return output.squeeze(-2)
