"""Exact router selection for released Kimi K3 decode and Q3 verification.

The stock router materializes a corrected score array and routes it through a
general-purpose ``argpartition`` before gathering and normalizing the original
sigmoid scores.  Released Kimi K3 selects 16 of 896 experts from one group; the
strict K-cut experiment selects 8.  One eight-SIMD Metal dispatch can perform
the FP32 sigmoid, corrected selection, and final BF16 weight emission directly
for either explicitly supported width.  The sigmoid expression is copied from
MLX's authoritative Metal unary implementation.

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
_SUPPORTED_TOP_K = frozenset((8, 16))
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

// Sort each thread's strided candidates once, best first.  The raw score
// stays out of shared memory: every thread reevaluates the deterministic
// sigmoid for its own candidates and for each round's winner, which is
// cheaper than the barrier a shared score table would need.  Extraction
// rounds then only pop the winning thread's head, so the selection order
// is identical to repeated maximum extraction without any rescan.
float lane_scores_list[EXPERTS_PER_THREAD];
uint lane_indices_list[EXPERTS_PER_THREAD];
uint lane_count = 0u;
for (uint slot = 0u; slot < EXPERTS_PER_THREAD; ++slot) {
  const uint expert = thread_id + slot * THREADS;
  if (expert >= EXPERTS) {
    continue;
  }
  const float corrected =
      k3_router_sigmoid(static_cast<float>(row_gates[expert])) +
      bias[expert];
  uint pos = 0u;
  while (pos < lane_count &&
         k3_router_better(
             lane_scores_list[pos],
             lane_indices_list[pos],
             corrected,
             expert)) {
    ++pos;
  }
  for (uint shift = lane_count; shift > pos; --shift) {
    lane_scores_list[shift] = lane_scores_list[shift - 1];
    lane_indices_list[shift] = lane_indices_list[shift - 1];
  }
  lane_scores_list[pos] = corrected;
  lane_indices_list[pos] = expert;
  ++lane_count;
}
uint lane_head = 0u;

uint selected_index = 0u;
float my_scores[TOP_K];
// Double-buffering lets the next round overwrite the other bank while
// stragglers still read this one, so one barrier suffices per round.
threadgroup float simdgroup_scores[2][SIMDGROUPS];
threadgroup uint simdgroup_indices[2][SIMDGROUPS];

for (uint rank = 0u; rank < TOP_K; ++rank) {
  const uint bank = rank & 1u;
  const float head_score =
      lane_head < lane_count ? lane_scores_list[lane_head] : 0.0f;
  const uint head_index =
      lane_head < lane_count ? lane_indices_list[lane_head] : UINT_MAX;

  float simd_winner_score = head_score;
  uint simd_winner_index = head_index;

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
    simdgroup_scores[bank][simdgroup] = simd_winner_score;
    simdgroup_indices[bank][simdgroup] = simd_winner_index;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Every SIMD group replays the block reduction, so the winner is broadcast
  // without a second shared-memory round trip or a trailing barrier.  Only
  // SIMDGROUPS lanes hold candidates and the comparison is a strict total
  // order, so the shorter butterfly finds the same unique winner.
  float block_score =
      lane < SIMDGROUPS ? simdgroup_scores[bank][lane] : 0.0f;
  uint block_index =
      lane < SIMDGROUPS ? simdgroup_indices[bank][lane] : UINT_MAX;
  for (ushort delta = SIMDGROUPS / 2; delta > 0; delta >>= 1) {
    const float other_score = simd_shuffle_down(block_score, delta);
    const uint other_index = simd_shuffle_down(block_index, delta);
    if (lane + delta < 32u &&
        k3_router_better(
            other_score, other_index, block_score, block_index)) {
      block_score = other_score;
      block_index = other_index;
    }
  }

  // The shuffle_down butterfly completes on lane 0 only.
  const uint winner = simd_broadcast_first(block_index);
  my_scores[rank] =
      k3_router_sigmoid(static_cast<float>(row_gates[winner]));
  if (winner == head_index) {
    ++lane_head;
  }
  if (thread_id == rank) {
    selected_index = winner;
  }
}

// MLX's small-row reduction handles each supported row in one thread and
// folds the values from slot 0 through slot TOP_K-1.  Preserve that exact
// association: a SIMD reduction can differ by one FP32 ULP, which is enough
// to cross a BF16 rounding midpoint after normalization.  Every thread
// recorded the same per-round scores, so it folds the denominator locally
// with no shared table or barrier.
float denominator = 0.0f;
for (uint rank = 0u; rank < TOP_K; ++rank) {
  denominator = my_scores[rank] + denominator;
}
denominator += 1.0e-20f;

if (thread_id < TOP_K) {
  const uint output = row * TOP_K + thread_id;
  indices[output] = selected_index;
  weights[output] =
      static_cast<WeightT>(my_scores[thread_id] / denominator);
}
"""


@lru_cache(maxsize=None)
def _kernel(top_k: int):
    if top_k not in _SUPPORTED_TOP_K:
        return None
    if not _metal_available():
        return None
    return mx.fast.metal_kernel(
        name=f"k3_fused_router_top{top_k}_v6",
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
        _kernel(top_k) is not None
        and mx.default_device() == mx.gpu
        and gates.dtype == mx.bfloat16
        and gates.ndim == 3
        and gates.shape[-1] == _EXPERTS
        and (
            gates.shape[-2] == 1
            or (
                gates.shape[0] == 1
                and (gates.shape[-2] == 3 or (top_k == 8 and gates.shape[-2] <= 4096))
            )
        )
        and gates.size > 0
        and bias is not None
        and bias.dtype == mx.float32
        and bias.shape == (_EXPERTS,)
        and top_k in _SUPPORTED_TOP_K
        and n_group == 1
        and topk_group == 1
        and routed_scaling_factor == 1.0
        and renormalize
    )


def _fused_k3_router(
    gates: mx.array,
    bias: mx.array,
    output_dtype: mx.Dtype,
    top_k: int,
) -> Tuple[mx.array, mx.array]:
    kernel = _kernel(top_k)
    if kernel is None:
        raise RuntimeError("The fused Kimi K3 router requires Metal")
    output_shape = (*gates.shape[:-1], top_k)
    rows = gates.size // _EXPERTS
    return kernel(
        inputs=[gates, bias],
        template=[
            ("GateT", gates.dtype),
            ("WeightT", output_dtype),
            ("EXPERTS", _EXPERTS),
            ("TOP_K", top_k),
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
def _compiled_fused_k3_router_top16(
    gates: mx.array,
    bias: mx.array,
) -> Tuple[mx.array, mx.array]:
    return _fused_k3_router(gates, bias, gates.dtype, 16)


@partial(mx.compile, shapeless=False)
def _compiled_fused_k3_router_top8(
    gates: mx.array,
    bias: mx.array,
) -> Tuple[mx.array, mx.array]:
    return _fused_k3_router(gates, bias, gates.dtype, 8)


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
    compiled = (
        _compiled_fused_k3_router_top16
        if top_k == 16
        else _compiled_fused_k3_router_top8
    )
    return compiled(gates, bias)
