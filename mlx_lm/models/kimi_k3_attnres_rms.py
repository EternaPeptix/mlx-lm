"""Exact opt-in Kimi K3 decode fusion for AttnRes followed by RMSNorm.

The stock decode path materializes the BF16 AttnRes mixture and then launches
MLX's RMSNorm kernel.  This prototype preserves that BF16 rounding boundary in
threadgroup memory and performs the native ``rms_looped`` reduction in the
same dispatch.

The implementation is deliberately fail-closed.  It only accepts the released
Kimi K3 decode geometry: one BF16 token, hidden width 7168, and at most eight
stored residual blocks (the 93-layer checkpoint uses a block size of 12).
"""

from __future__ import annotations

import math
import os
from functools import lru_cache
from typing import Final

import mlx.core as mx


FUSED_ATTNRES_RMS_ENV: Final = "MLX_LM_KIMI_K3_FUSED_ATTNRES_RMS"

K3_HIDDEN_SIZE: Final = 7168
K3_MAX_RESIDUAL_BLOCKS: Final = 8
_ATTNRES_THREADS: Final = 512
_RMS_THREADS: Final = 1024
_RMS_N_READS: Final = 4


_FUSED_ATTNRES_RMS_SOURCE = r"""
    constexpr int NACC = K + 2;
    constexpr int MIX_NSIMD = MIX_THREADS / 32;
    constexpr int RMS_SIMD = 32;

    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;

    // Reproduce attnres_mix's 512-thread accumulation and reduction exactly.
    float acc[NACC];
    for (int i = 0; i < NACC; ++i) {
      acc[i] = 0.0f;
    }
    if (tid < MIX_THREADS) {
      for (uint d = tid; d < D; d += MIX_THREADS) {
        const float w = static_cast<float>(w_eff[d]);
        const float pv = static_cast<float>(partial[d]);
        for (int k = 0; k < K; ++k) {
          acc[k] += static_cast<float>(raw[k * D + d]) * w;
        }
        acc[K] += pv * w;
        acc[K + 1] += pv * pv;
      }
    }

    threadgroup float mix_sums[NACC * MIX_NSIMD];
    if (tid < MIX_THREADS) {
      for (int i = 0; i < NACC; ++i) {
        const float value = simd_sum(acc[i]);
        if (lane == 0) {
          mix_sums[i * MIX_NSIMD + sg] = value;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float mix_weights[K + 1];
    if (tid == 0) {
      float totals[NACC];
      for (int i = 0; i < NACC; ++i) {
        totals[i] = 0.0f;
        for (int j = 0; j < MIX_NSIMD; ++j) {
          totals[i] += mix_sums[i * MIX_NSIMD + j];
        }
      }

      const float partial_inv_rms =
          metal::rsqrt(totals[K + 1] / D + eps[0]);
      float logits[K + 1];
      float maximum = -1e30f;
      for (int k = 0; k < K; ++k) {
        logits[k] = totals[k] * inv_rms[k];
        maximum = metal::max(maximum, logits[k]);
      }
      logits[K] = totals[K] * partial_inv_rms;
      maximum = metal::max(maximum, logits[K]);

      float denominator = 0.0f;
      for (int k = 0; k <= K; ++k) {
        logits[k] = metal::exp(logits[k] - maximum);
        denominator += logits[k];
      }
      for (int k = 0; k <= K; ++k) {
        mix_weights[k] = logits[k] / denominator;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // This cast is the stock AttnRes materialization boundary.  RMSNorm must
    // consume the rounded BF16 value, rather than the float accumulator.
    threadgroup InT mixed[D];
    const float partial_weight = mix_weights[K];
    for (uint d = tid; d < D; d += RMS_THREADS) {
      float value = partial_weight * static_cast<float>(partial[d]);
      for (int k = 0; k < K; ++k) {
        value += mix_weights[k] * static_cast<float>(raw[k * D + d]);
      }
      mixed[d] = static_cast<InT>(value);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reproduce MLX rms_looped for D=7168: four adjacent values per thread,
    // a 1024-thread looped reduction, and precise rsqrt.
    float rms_acc = 0.0f;
    for (uint r = 0; r < D; r += RMS_THREADS * RMS_N_READS) {
      const uint base = r + tid * RMS_N_READS;
      for (int i = 0; i < RMS_N_READS; ++i) {
        const uint d = base + i;
        if (d < D) {
          const float value = static_cast<float>(mixed[d]);
          rms_acc += value * value;
        }
      }
    }
    rms_acc = simd_sum(rms_acc);

    threadgroup float rms_sums[RMS_SIMD];
    threadgroup float rms_inv[1];
    if (sg == 0) {
      rms_sums[lane] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
      rms_sums[sg] = rms_acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
      rms_acc = simd_sum(rms_sums[lane]);
      if (lane == 0) {
        rms_inv[0] = metal::precise::rsqrt(rms_acc / D + eps[0]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint r = 0; r < D; r += RMS_THREADS * RMS_N_READS) {
      const uint base = r + tid * RMS_N_READS;
      for (int i = 0; i < RMS_N_READS; ++i) {
        const uint d = base + i;
        if (d < D) {
          // Keep the same intermediate cast and multiply order as rms_looped.
          out[d] = norm_weight[d] *
              static_cast<InT>(
                  static_cast<float>(mixed[d]) * rms_inv[0]);
        }
      }
    }
"""


_fused_attnres_rms_kernel = (
    mx.fast.metal_kernel(
        name="k3_fused_attnres_rms",
        input_names=[
            "raw",
            "inv_rms",
            "partial",
            "w_eff",
            "norm_weight",
            "eps",
        ],
        output_names=["out"],
        source=_FUSED_ATTNRES_RMS_SOURCE,
    )
    if mx.metal.is_available()
    else None
)

_eps_cache: dict[float, mx.array] = {}


@lru_cache(maxsize=1)
def fused_attnres_rms_enabled() -> bool:
    """Return whether the experimental inference-only path was requested."""

    return os.environ.get(FUSED_ATTNRES_RMS_ENV, "0") == "1"


def supports_fused_attnres_rms_geometry(
    raw: mx.array,
    inv_rms: mx.array,
    partial: mx.array,
    w_eff: mx.array,
    norm_weight: mx.array,
    eps: float,
) -> bool:
    """Check the shape and dtype contract without inspecting execution state."""

    if not math.isfinite(eps) or eps <= 0.0:
        return False
    if partial.shape != (1, 1, K3_HIDDEN_SIZE):
        return False
    if raw.ndim != 4 or raw.shape[1:] != partial.shape:
        return False
    residual_count = raw.shape[0]
    if not 1 <= residual_count <= K3_MAX_RESIDUAL_BLOCKS:
        return False
    if inv_rms.shape != raw.shape[:-1]:
        return False
    if w_eff.shape != (K3_HIDDEN_SIZE,):
        return False
    if norm_weight.shape != (K3_HIDDEN_SIZE,):
        return False
    return (
        raw.dtype == mx.bfloat16
        and partial.dtype == mx.bfloat16
        and norm_weight.dtype == mx.bfloat16
        and inv_rms.dtype == mx.float32
        and w_eff.dtype == mx.float32
    )


def supports_fused_attnres_rms(
    raw: mx.array,
    inv_rms: mx.array,
    partial: mx.array,
    w_eff: mx.array,
    norm_weight: mx.array,
    eps: float,
) -> bool:
    """Return whether the exact Metal path can execute for these arrays."""

    return (
        fused_attnres_rms_enabled()
        and _fused_attnres_rms_kernel is not None
        and mx.default_device() == mx.gpu
        and supports_fused_attnres_rms_geometry(
            raw, inv_rms, partial, w_eff, norm_weight, eps
        )
    )


def maybe_fused_attnres_rms(
    raw: mx.array,
    inv_rms: mx.array,
    partial: mx.array,
    w_eff: mx.array,
    norm_weight: mx.array,
    eps: float,
) -> mx.array | None:
    """Return the fused result, or ``None`` to leave the stock path untouched."""

    if not supports_fused_attnres_rms(
        raw, inv_rms, partial, w_eff, norm_weight, eps
    ):
        return None

    eps_value = float(eps)
    eps_array = _eps_cache.get(eps_value)
    if eps_array is None:
        eps_array = _eps_cache.setdefault(
            eps_value, mx.array([eps_value], dtype=mx.float32)
        )

    residual_count = raw.shape[0]
    return _fused_attnres_rms_kernel(
        inputs=[raw, inv_rms, partial, w_eff, norm_weight, eps_array],
        template=[
            ("InT", partial.dtype),
            ("K", residual_count),
            ("D", K3_HIDDEN_SIZE),
            ("MIX_THREADS", _ATTNRES_THREADS),
            ("RMS_THREADS", _RMS_THREADS),
            ("RMS_N_READS", _RMS_N_READS),
        ],
        grid=(_RMS_THREADS, 1, 1),
        threadgroup=(_RMS_THREADS, 1, 1),
        output_shapes=[partial.shape],
        output_dtypes=[partial.dtype],
    )[0]
