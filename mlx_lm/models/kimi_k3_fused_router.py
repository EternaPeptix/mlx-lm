"""Exact router selection for released Kimi K3 decode and Q3 verification.

The stock router materializes a corrected score array and routes it through a
general-purpose ``argpartition`` before gathering and normalizing the original
sigmoid scores.  Kimi K3 always selects 16 of 896 experts from one group, so
one eight-SIMD Metal dispatch can perform the FP32 sigmoid, corrected
selection, and final BF16 weight emission directly.  The sigmoid expression is
copied from MLX's authoritative Metal unary implementation.

Unsupported shapes, dtypes, configurations, devices, and training calls retain
the stock path.
"""

from __future__ import annotations

import os
from functools import lru_cache, partial
from typing import Optional, Tuple

import mlx.core as mx


FUSED_ROUTER_ENV = "MLX_LM_KIMI_K3_FUSED_ROUTER"

_EXPERTS = 896
_TOP_K = 16
_THREADS = 256
_SIMDGROUPS = _THREADS // 32
_EXPERTS_PER_THREAD = (_EXPERTS + _THREADS - 1) // _THREADS


def _metal_available() -> bool:
    metal = getattr(mx, "metal", None)
    return metal is not None and bool(metal.is_available())


_HEADER = r"""
inline float k3_router_sigmoid(float value) {
  const float y =
      1.0f / (1.0f + metal::exp(metal::abs(value)));
  return value < 0.0f ? y : 1.0f - y;
}

inline bool k3_router_better(
    float lhs_score,
    uint lhs_index,
    float rhs_score,
    uint rhs_index) {
  if (lhs_index == UINT_MAX) {
    return false;
  }
  if (rhs_index == UINT_MAX) {
    return true;
  }

  const bool lhs_nan = metal::isnan(lhs_score);
  const bool rhs_nan = metal::isnan(rhs_score);
  if (lhs_nan != rhs_nan) {
    // Match MLX argpartition: NaNs sort behind every numeric score.
    return !lhs_nan;
  }
  if (lhs_nan) {
    // If every remaining score is NaN, MLX selects lower indices first.
    return lhs_index < rhs_index;
  }
  if (lhs_score > rhs_score) {
    return true;
  }
  if (lhs_score < rhs_score) {
    return false;
  }
  return lhs_index < rhs_index;
}
"""


_SOURCE = r"""
const uint thread_id = thread_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const uint simdgroup = simdgroup_index_in_threadgroup;
const uint row = threadgroup_position_in_grid.y;
const device GateT* row_gates = gates + row * EXPERTS;

// Cache the authoritative FP32 sigmoid once per expert.  At 896 experts this
// is a bounded 3.5 KiB of threadgroup storage and avoids an extra dispatch
// without paying for repeated exponentials when a winning lane is rescanned.
threadgroup float row_scores[EXPERTS];
for (uint slot = 0u; slot < EXPERTS_PER_THREAD; ++slot) {
  const uint expert = thread_id + slot * THREADS;
  if (expert < EXPERTS) {
    row_scores[expert] =
        k3_router_sigmoid(static_cast<float>(row_gates[expert]));
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// Each thread owns at most four strided experts.  Keep only its current
// maximum and rescan that thread when (and only when) it supplies a global
// winner.
uint selected_mask = 0u;
float lane_score = 0.0f;
uint lane_index = UINT_MAX;
uint lane_slot = UINT_MAX;
for (uint slot = 0u; slot < EXPERTS_PER_THREAD; ++slot) {
  const uint expert = thread_id + slot * THREADS;
  if (expert >= EXPERTS) {
    continue;
  }
  const float corrected = row_scores[expert] + bias[expert];
  if (k3_router_better(
          corrected, expert, lane_score, lane_index)) {
    lane_score = corrected;
    lane_index = expert;
    lane_slot = slot;
  }
}

uint selected_index = 0u;
threadgroup float simdgroup_scores[SIMDGROUPS];
threadgroup uint simdgroup_indices[SIMDGROUPS];
threadgroup uint global_winner;
threadgroup float selected_scores[TOP_K];
threadgroup float denominator;

for (uint rank = 0u; rank < TOP_K; ++rank) {
  float simd_winner_score = lane_score;
  uint simd_winner_index = lane_index;

  for (ushort delta = 16; delta > 0; delta >>= 1) {
    const float other_score = simd_shuffle_down(simd_winner_score, delta);
    const uint other_index = simd_shuffle_down(simd_winner_index, delta);
    if (lane + delta < 32u &&
        k3_router_better(
            other_score,
            other_index,
            simd_winner_score,
            simd_winner_index)) {
      simd_winner_score = other_score;
      simd_winner_index = other_index;
    }
  }

  if (lane == 0u) {
    simdgroup_scores[simdgroup] = simd_winner_score;
    simdgroup_indices[simdgroup] = simd_winner_index;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simdgroup == 0u) {
    float block_score =
        lane < SIMDGROUPS ? simdgroup_scores[lane] : 0.0f;
    uint block_index =
        lane < SIMDGROUPS ? simdgroup_indices[lane] : UINT_MAX;
    for (ushort delta = 16; delta > 0; delta >>= 1) {
      const float other_score = simd_shuffle_down(block_score, delta);
      const uint other_index = simd_shuffle_down(block_index, delta);
      if (lane + delta < 32u &&
          k3_router_better(
              other_score, other_index, block_score, block_index)) {
        block_score = other_score;
        block_index = other_index;
      }
    }
    if (lane == 0u) {
      global_winner = block_index;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint winner = global_winner;
  const uint winner_thread = winner % THREADS;
  if (thread_id == winner_thread && rank + 1u < TOP_K) {
    selected_mask |= 1u << lane_slot;
    lane_score = 0.0f;
    lane_index = UINT_MAX;
    lane_slot = UINT_MAX;
    for (uint slot = 0u; slot < EXPERTS_PER_THREAD; ++slot) {
      if ((selected_mask & (1u << slot)) != 0u) {
        continue;
      }
      const uint expert = thread_id + slot * THREADS;
      if (expert >= EXPERTS) {
        continue;
      }
      const float corrected = row_scores[expert] + bias[expert];
      if (k3_router_better(
              corrected, expert, lane_score, lane_index)) {
        lane_score = corrected;
        lane_index = expert;
        lane_slot = slot;
      }
    }
  }
  if (thread_id == rank) {
    selected_index = winner;
    selected_scores[rank] = row_scores[winner];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// MLX's small-row reduction handles this 16-element row in one thread and
// folds the values from slot 0 through slot 15.  Preserve that exact
// association: a SIMD reduction can differ by one FP32 ULP, which is enough
// to cross a BF16 rounding midpoint after normalization.
if (thread_id == 0u) {
  float total = 0.0f;
  for (uint rank = 0u; rank < TOP_K; ++rank) {
    total = selected_scores[rank] + total;
  }
  denominator = total + 1.0e-20f;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

if (thread_id < TOP_K) {
  const uint output = row * TOP_K + thread_id;
  indices[output] = selected_index;
  weights[output] =
      static_cast<WeightT>(selected_scores[thread_id] / denominator);
}
"""


@lru_cache(maxsize=None)
def _kernel():
    if not _metal_available():
        return None
    return mx.fast.metal_kernel(
        name="k3_fused_router_top16_v5",
        input_names=["gates", "bias"],
        output_names=["indices", "weights"],
        header=_HEADER,
        source=_SOURCE,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=1)
def fused_k3_router_enabled() -> bool:
    """Return whether the default-off released-K3 specialization is enabled."""

    return os.environ.get(FUSED_ROUTER_ENV, "0") == "1"


def supports_fused_k3_router(
    gates: mx.array,
    bias: Optional[mx.array],
    *,
    top_k: int,
    n_group: int,
    topk_group: int,
    routed_scaling_factor: float,
    renormalize: bool,
) -> bool:
    """Check the deliberately narrow released-K3 decode contract."""

    return (
        _kernel() is not None
        and mx.default_device() == mx.gpu
        and gates.dtype == mx.bfloat16
        and gates.ndim == 3
        and gates.shape[-1] == _EXPERTS
        and (
            gates.shape[-2] == 1
            or (gates.shape[0] == 1 and gates.shape[-2] == 3)
        )
        and gates.size > 0
        and bias is not None
        and bias.dtype == mx.float32
        and bias.shape == (_EXPERTS,)
        and top_k == _TOP_K
        and n_group == 1
        and topk_group == 1
        and routed_scaling_factor == 1.0
        and renormalize
    )


def _fused_k3_router(
    gates: mx.array,
    bias: mx.array,
    output_dtype: mx.Dtype,
) -> Tuple[mx.array, mx.array]:
    kernel = _kernel()
    if kernel is None:
        raise RuntimeError("The fused Kimi K3 router requires Metal")
    output_shape = (*gates.shape[:-1], _TOP_K)
    rows = gates.size // _EXPERTS
    return kernel(
        inputs=[gates, bias],
        template=[
            ("GateT", gates.dtype),
            ("WeightT", output_dtype),
            ("EXPERTS", _EXPERTS),
            ("TOP_K", _TOP_K),
            ("THREADS", _THREADS),
            ("SIMDGROUPS", _SIMDGROUPS),
            ("EXPERTS_PER_THREAD", _EXPERTS_PER_THREAD),
        ],
        grid=(_THREADS, rows, 1),
        threadgroup=(_THREADS, 1, 1),
        output_shapes=[output_shape, output_shape],
        output_dtypes=[mx.uint32, output_dtype],
    )


@partial(mx.compile, shapeless=False)
def _compiled_fused_k3_router(
    gates: mx.array,
    bias: mx.array,
) -> Tuple[mx.array, mx.array]:
    return _fused_k3_router(gates, bias, gates.dtype)


def maybe_fused_k3_router(
    gates: mx.array,
    bias: Optional[mx.array],
    *,
    top_k: int,
    n_group: int,
    topk_group: int,
    routed_scaling_factor: float,
    renormalize: bool,
    training: bool,
) -> Optional[Tuple[mx.array, mx.array]]:
    """Return exact released-K3 routing outputs, or ``None`` for stock."""

    if (
        not fused_k3_router_enabled()
        or training
        or not supports_fused_k3_router(
            gates,
            bias,
            top_k=top_k,
            n_group=n_group,
            topk_group=topk_group,
            routed_scaling_factor=routed_scaling_factor,
            renormalize=renormalize,
        )
    ):
        return None
    return _compiled_fused_k3_router(gates, bias)
