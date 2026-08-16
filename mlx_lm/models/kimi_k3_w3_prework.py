"""Exact, width-three Kimi K3 KDA prework/history experiment.

This module deliberately starts after the stock QKV and ``a`` projections.
It does not compose with packed projections, speculative scheduling, cache
ownership, or any DSpark primitive.  The sole candidate kernel replaces the
released TP2 rank-local sequence

``short-conv/history -> Q/K RMSNorm+scale -> bounded g``

while retaining every observable BF16 boundary.  The pre-projection selector
routes every unsupported contract to stock; a contract change after selector
admission fails closed instead of silently running reordered stock work.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import mlx.core as mx

K3_W3_PREWORK_HISTORY_ENV = "MLX_LM_KIMI_K3_W3_PREWORK_HISTORY"

_BATCH = 1
_WIDTH = 3
_HEADS = 48
_HEAD_DIM = 128
_CONV_KERNEL = 4
_LOWER_BOUND = -5.0
_STOCK_SCALE = float(_HEAD_DIM) ** -0.5


@lru_cache(maxsize=1)
def k3_w3_prework_history_enabled() -> bool:
    """Parse the strict, default-off experimental selector."""

    value = os.environ.get(K3_W3_PREWORK_HISTORY_ENV, "0")
    if value not in {"0", "1"}:
        raise ValueError(f"{K3_W3_PREWORK_HISTORY_ENV} must be exactly '0' or '1'")
    return value == "1"


def _metal_available() -> bool:
    metal = getattr(mx, "metal", None)
    return metal is not None and bool(metal.is_available())


_W3_PREWORK_SOURCE = r"""
    constexpr uint VALUES_PER_LANE = 4;

    const uint lane = thread_index_in_simdgroup;
    const uint segment = threadgroup_position_in_grid.y;
    const uint kind = segment / H;
    const uint head = segment - kind * H;
    const uint dim_base = lane * VALUES_PER_LANE;
    const uint channel_base = segment * D;

    threadgroup float local_sums[32];
    threadgroup float local_inv_rms[1];
    T convolved[L][VALUES_PER_LANE];

    for (uint i = 0; i < VALUES_PER_LANE; ++i) {
      const uint dim = dim_base + i;
      const uint channel = channel_base + dim;
      float local_state[KS - 1];
      for (uint j = 0; j < KS - 1; ++j) {
        local_state[j] = static_cast<float>(state[j * C + channel]);
      }

      for (uint t = 0; t < L; ++t) {
        const T projected = projected_qkv[t * C + channel];
        float value = 0.0f;
        for (uint j = 0; j < KS - 1; ++j) {
          value += static_cast<float>(conv_weight[channel * KS + j]) *
              local_state[j];
        }
        value += static_cast<float>(conv_weight[channel * KS + KS - 1]) *
            static_cast<float>(projected);

        // The stock history kernel writes BF16 before RMSNorm or recurrence
        // consumers can observe this value.  Volatile storage prevents the
        // compiler from forwarding the FP32 SiLU result through that boundary.
        volatile T rounded_storage =
            static_cast<T>(value / (1.0f + metal::exp(-value)));
        const T rounded = rounded_storage;
        convolved[t][i] = rounded;

        for (uint j = 0; j < KS - 2; ++j) {
          local_state[j] = local_state[j + 1];
        }
        local_state[KS - 2] = static_cast<float>(projected);

        for (uint j = 0; j < KS - 1; ++j) {
          state_history[((t * (KS - 1) + j) * C) + channel] =
              static_cast<T>(local_state[j]);
        }
      }

      for (uint j = 0; j < KS - 1; ++j) {
        new_state[j * C + channel] = static_cast<T>(local_state[j]);
      }
    }

    for (uint t = 0; t < L; ++t) {
      const uint row_base = (t * H + head) * D + dim_base;

      if (kind < 2) {
        float square_sum = 0.0f;
        for (uint i = 0; i < VALUES_PER_LANE; ++i) {
          const float value = static_cast<float>(convolved[t][i]);
          square_sum += value * value;
        }
        square_sum = simd_sum(square_sum);

        // Preserve rms_single_row's second (cross-SIMD) reduction even though
        // this D=128 specialization has exactly one 32-lane SIMD group.
        local_sums[lane] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) {
          local_sums[0] = square_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        square_sum = simd_sum(local_sums[lane]);
        if (lane == 0) {
          local_inv_rms[0] = metal::precise::rsqrt(
              square_sum / 128.0f + 0.0000000078125f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float inv_rms = local_inv_rms[0];
        const float output_scale =
            kind == 0 ? 0.0078125f : 0.08837890625f;

        for (uint i = 0; i < VALUES_PER_LANE; ++i) {
          volatile T normalized_storage = static_cast<T>(
              static_cast<float>(convolved[t][i]) * inv_rms);
          const T normalized = normalized_storage;
          const T scaled = static_cast<T>(
              static_cast<float>(normalized) * output_scale);
          if (kind == 0) {
            q[row_base + i] = scaled;
          } else {
            k[row_base + i] = scaled;
            raw_k[row_base + i] = convolved[t][i];
          }
        }
      } else {
        for (uint i = 0; i < VALUES_PER_LANE; ++i) {
          v[row_base + i] = convolved[t][i];
        }
      }

      // The Q segment's SIMD group also owns this head's vector gate.  Match
      // compute_g_safe exactly: precise outer exponentials and MLX Sigmoid's
      // abs/branch formulation.  beta deliberately remains outside this kernel.
      if (kind == 0) {
        const float rate = metal::precise::exp(
            static_cast<float>(A_log[head]));
        for (uint i = 0; i < VALUES_PER_LANE; ++i) {
          const uint gate_index = row_base + i;
          const uint parameter_index = head * D + dim_base + i;
          const float gate_x = rate * (
              static_cast<float>(a_logits[gate_index]) +
              static_cast<float>(dt_bias[parameter_index]));
          const float sigmoid_tail =
              1.0f / (1.0f + metal::exp(metal::abs(gate_x)));
          const float sigmoid_value =
              gate_x < 0.0f ? sigmoid_tail : 1.0f - sigmoid_tail;
          gk[gate_index] = metal::precise::exp(
              static_cast<float>(lower_bound[0]) * sigmoid_value);
        }
      }
    }
"""


@lru_cache(maxsize=1)
def _w3_prework_kernel():
    if not _metal_available():
        return None
    return mx.fast.metal_kernel(
        name="k3_w3_projected_prework_history",
        input_names=[
            "projected_qkv",
            "state",
            "conv_weight",
            "a_logits",
            "A_log",
            "dt_bias",
            "lower_bound",
        ],
        output_names=[
            "q",
            "k",
            "raw_k",
            "v",
            "gk",
            "new_state",
            "state_history",
        ],
        source=_W3_PREWORK_SOURCE,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=1)
def _lower_bound_array() -> mx.array:
    return mx.array([_LOWER_BOUND], dtype=mx.float32)


def supports_k3_w3_prework_history(
    projected_qkv: mx.array,
    state: mx.array,
    conv_weight: mx.array,
    a_logits: mx.array,
    A_log: mx.array,
    dt_bias: mx.array,
    *,
    num_heads: int,
    head_dim: int,
    conv_kernel: int,
    lower_bound: float | None,
) -> bool:
    """Return whether tensors match the released TP2 width-three contract."""

    projection_dim = _HEADS * _HEAD_DIM
    channels = 3 * projection_dim
    return (
        _w3_prework_kernel() is not None
        and mx.default_device() == mx.gpu
        and num_heads == _HEADS
        and head_dim == _HEAD_DIM
        and conv_kernel == _CONV_KERNEL
        and lower_bound == _LOWER_BOUND
        and projected_qkv.dtype == mx.bfloat16
        and projected_qkv.shape == (_BATCH, _WIDTH, channels)
        and state.dtype == mx.bfloat16
        and state.shape == (_BATCH, _CONV_KERNEL - 1, channels)
        and conv_weight.dtype == mx.bfloat16
        and conv_weight.shape == (channels, _CONV_KERNEL, 1)
        and a_logits.dtype == mx.bfloat16
        and a_logits.shape == (_BATCH, _WIDTH, _HEADS, _HEAD_DIM)
        and A_log.dtype == mx.float32
        and A_log.shape == (_HEADS,)
        and dt_bias.dtype == mx.float32
        and dt_bias.shape == (projection_dim,)
    )


def fused_k3_w3_prework_history(
    projected_qkv: mx.array,
    state: mx.array,
    conv_weight: mx.array,
    a_logits: mx.array,
    A_log: mx.array,
    dt_bias: mx.array,
    *,
    num_heads: int,
    head_dim: int,
    conv_kernel: int,
    lower_bound: float | None,
) -> tuple[mx.array, ...]:
    """Run the exact post-projection W3 prework/history superkernel."""

    if not supports_k3_w3_prework_history(
        projected_qkv,
        state,
        conv_weight,
        a_logits,
        A_log,
        dt_bias,
        num_heads=num_heads,
        head_dim=head_dim,
        conv_kernel=conv_kernel,
        lower_bound=lower_bound,
    ):
        raise ValueError("unsupported Kimi K3 W3 prework/history contract")

    kernel = _w3_prework_kernel()
    assert kernel is not None
    projection_dim = num_heads * head_dim
    channels = 3 * projection_dim
    vector_shape = (_BATCH, _WIDTH, num_heads, head_dim)
    return tuple(
        kernel(
            inputs=[
                projected_qkv,
                state,
                conv_weight,
                a_logits,
                A_log,
                dt_bias,
                _lower_bound_array(),
            ],
            template=[
                ("T", projected_qkv.dtype),
                ("H", num_heads),
                ("D", head_dim),
                ("C", channels),
                ("KS", conv_kernel),
                ("L", _WIDTH),
            ],
            grid=(32, 3 * num_heads, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[
                vector_shape,
                vector_shape,
                vector_shape,
                vector_shape,
                vector_shape,
                state.shape,
                (_BATCH, _WIDTH, conv_kernel - 1, channels),
            ],
            output_dtypes=[
                mx.bfloat16,
                mx.bfloat16,
                mx.bfloat16,
                mx.bfloat16,
                mx.float32,
                mx.bfloat16,
                mx.bfloat16,
            ],
            stream=mx.gpu,
        )
    )


def can_use_k3_w3_prework_history(
    attention: Any,
    x: mx.array,
    state: mx.array | None,
    *,
    short_conv_type: type,
    inner_conv_type: type,
    mask: mx.array | None,
    lengths: mx.array | None,
    capture_speculative: bool,
) -> bool:
    """Cheap pre-projection selector for the isolated production candidate."""

    if not k3_w3_prework_history_enabled():
        return False
    short_conv = getattr(attention, "qkv_conv", None)
    inner_conv = getattr(short_conv, "conv", None)
    conv_weight = getattr(inner_conv, "weight", None)
    A_log = getattr(attention, "A_log", None)
    dt_bias = getattr(attention, "dt_bias", None)
    projection_dim = _HEADS * _HEAD_DIM
    channels = 3 * projection_dim
    return (
        capture_speculative
        and not getattr(attention, "training", True)
        and bool(getattr(attention, "use_full_rank_gate", False))
        and int(getattr(attention, "num_heads", 0)) == _HEADS
        and int(getattr(attention, "head_dim", 0)) == _HEAD_DIM
        and int(getattr(attention, "projection_dim", 0)) == projection_dim
        and int(getattr(attention, "conv_kernel", 0)) == _CONV_KERNEL
        and type(getattr(attention, "scale", None)) is float
        and attention.scale == _STOCK_SCALE
        and getattr(attention, "lower_bound", None) == _LOWER_BOUND
        and type(short_conv) is short_conv_type
        and not getattr(short_conv, "training", True)
        and int(getattr(short_conv, "kernel_size", 0)) == _CONV_KERNEL
        and type(inner_conv) is inner_conv_type
        and _w3_prework_kernel() is not None
        and mx.default_device() == mx.gpu
        and x.dtype == mx.bfloat16
        and x.ndim == 3
        and x.shape[0] == _BATCH
        and x.shape[1] == _WIDTH
        and isinstance(state, mx.array)
        and state.dtype == mx.bfloat16
        and state.shape == (_BATCH, _CONV_KERNEL - 1, channels)
        and mask is None
        and lengths is None
        and isinstance(conv_weight, mx.array)
        and conv_weight.dtype == mx.bfloat16
        and conv_weight.shape == (channels, _CONV_KERNEL, 1)
        and isinstance(A_log, mx.array)
        and A_log.dtype == mx.float32
        and A_log.shape == (_HEADS,)
        and isinstance(dt_bias, mx.array)
        and dt_bias.dtype == mx.float32
        and dt_bias.shape == (projection_dim,)
    )


def maybe_fused_k3_w3_prework_history(
    attention: Any,
    projected_qkv: mx.array,
    state: mx.array,
    a_logits: mx.array,
) -> tuple[mx.array, ...] | None:
    """Return fused tensors, or ``None`` so the caller uses stock arithmetic."""

    conv_weight = getattr(
        getattr(getattr(attention, "qkv_conv", None), "conv", None),
        "weight",
        None,
    )
    A_log = getattr(attention, "A_log", None)
    dt_bias = getattr(attention, "dt_bias", None)
    if not all(isinstance(value, mx.array) for value in (conv_weight, A_log, dt_bias)):
        return None
    if not supports_k3_w3_prework_history(
        projected_qkv,
        state,
        conv_weight,
        a_logits,
        A_log,
        dt_bias,
        num_heads=int(getattr(attention, "num_heads", 0)),
        head_dim=int(getattr(attention, "head_dim", 0)),
        conv_kernel=int(getattr(attention, "conv_kernel", 0)),
        lower_bound=getattr(attention, "lower_bound", None),
    ):
        return None
    return fused_k3_w3_prework_history(
        projected_qkv,
        state,
        conv_weight,
        a_logits,
        A_log,
        dt_bias,
        num_heads=attention.num_heads,
        head_dim=attention.head_dim,
        conv_kernel=attention.conv_kernel,
        lower_bound=attention.lower_bound,
    )
