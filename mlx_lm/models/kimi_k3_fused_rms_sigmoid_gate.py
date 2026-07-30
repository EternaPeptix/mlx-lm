"""Exact decode-only fusion for Kimi K3's post-KDA norm and output gate."""

from __future__ import annotations

import math
import os
from functools import cache
from typing import Final, Optional

import mlx.core as mx


FUSED_RMS_SIGMOID_GATE_ENV: Final = (
    "MLX_LM_KIMI_K3_FUSED_POST_KDA_RMS_SIGMOID_GATE"
)
K3_HEADS: Final = 96
K3_HEAD_DIM: Final = 128
_K3_THREADS: Final = 128


_K3_FUSED_RMS_SIGMOID_GATE_SOURCE = r"""
    constexpr int D = 128;
    constexpr int N_READS = 4;

    const uint row = threadgroup_position_in_grid.y;
    const uint lane = thread_index_in_simdgroup;
    const uint simd_group = simdgroup_index_in_threadgroup;

    const uint row_offset = row * D;
    float sum_squares = 0.0f;
    for (int i = 0; i < N_READS; ++i) {
      const float value =
          static_cast<float>(x[row_offset + lane * N_READS + i]);
      sum_squares += value * value;
    }

    // Each SIMD group repeats MLX RMSNorm's native N_READS=4 reduction.
    // That preserves the reduction order while four groups independently
    // apply the sigmoid gate to disjoint 32-element quarters of the row.
    sum_squares = simd_sum(sum_squares);
    const float inv_mean = metal::precise::rsqrt(
        sum_squares / static_cast<float>(D) + eps[0]);

    const uint d = simd_group * 32 + lane;
    const uint output_offset = row_offset + d;
    const OutT normalized = static_cast<OutT>(
        static_cast<float>(x[output_offset]) * inv_mean);
    const OutT rms_value =
        static_cast<OutT>(weight[d]) * normalized;

    // Match MLX's stable sigmoid expression and output-dtype rounding.
    const OutT gate_value = static_cast<OutT>(gate[output_offset]);
    const OutT z = static_cast<OutT>(
        static_cast<OutT>(1) /
        (static_cast<OutT>(1) + metal::exp(metal::abs(gate_value))));
    const OutT sigmoid_value =
        gate_value < static_cast<OutT>(0)
            ? z
            : static_cast<OutT>(1) - z;
    y[output_offset] = rms_value * sigmoid_value;
"""


def _metal_available() -> bool:
    try:
        return mx.metal.is_available()
    except RuntimeError:
        return False


_k3_fused_rms_sigmoid_gate_kernel = (
    mx.fast.metal_kernel(
        name="k3_fused_post_kda_rms_sigmoid_gate",
        input_names=["x", "gate", "weight", "eps"],
        output_names=["y"],
        source=_K3_FUSED_RMS_SIGMOID_GATE_SOURCE,
    )
    if _metal_available()
    else None
)

_eps_cache: dict[tuple[float, mx.Dtype], mx.array] = {}


@cache
def fused_rms_sigmoid_gate_enabled() -> bool:
    """Return whether the default-off K3 specialization is enabled."""

    return os.environ.get(FUSED_RMS_SIGMOID_GATE_ENV, "0") == "1"


def supports_fused_rms_sigmoid_gate(
    x: mx.array,
    gate: mx.array,
    weight: mx.array,
    eps: float,
    *,
    training: bool,
) -> bool:
    """Validate the released K3 decode geometry without evaluating tensors."""

    return (
        not training
        and _k3_fused_rms_sigmoid_gate_kernel is not None
        and mx.default_device() == mx.gpu
        and x.shape == gate.shape
        and x.ndim == 4
        and x.shape[-3:] == (1, K3_HEADS, K3_HEAD_DIM)
        and x.dtype in (mx.bfloat16, mx.float16, mx.float32)
        and gate.dtype == x.dtype
        and weight.shape == (K3_HEAD_DIM,)
        and weight.dtype == x.dtype
        and math.isfinite(float(eps))
        and float(eps) > 0.0
    )


def maybe_fused_rms_sigmoid_gate(
    x: mx.array,
    gate: mx.array,
    weight: mx.array,
    eps: float,
    *,
    training: bool,
) -> Optional[mx.array]:
    """Return the exact fused result, or ``None`` for the stock MLX path."""

    if not fused_rms_sigmoid_gate_enabled() or not supports_fused_rms_sigmoid_gate(
        x,
        gate,
        weight,
        eps,
        training=training,
    ):
        return None

    x = mx.contiguous(x)
    gate = mx.contiguous(gate)
    weight = mx.contiguous(weight)

    eps_key = (float(eps), x.dtype)
    eps_array = _eps_cache.get(eps_key)
    if eps_array is None:
        eps_array = _eps_cache.setdefault(
            eps_key,
            mx.array([eps], dtype=mx.float32),
        )

    rows = x.size // K3_HEAD_DIM
    return _k3_fused_rms_sigmoid_gate_kernel(
        inputs=[x, gate, weight, eps_array],
        template=[("OutT", x.dtype)],
        grid=(_K3_THREADS, rows, 1),
        threadgroup=(_K3_THREADS, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )[0]
