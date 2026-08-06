"""Exact, opt-in Kimi K3 prefill down-QMM route combination."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import mlx.core as mx

PREFILL_ROUTE_COMBINE_ENV = "MLX_LM_KIMI_K3_PREFILL_ROUTE_COMBINE"
_EXPERTS = 896
_SUPPORTED_TOP_K = frozenset((8, 16))
_INPUT_DIMS = 3584
_INTERMEDIATE_DIMS = 1536
_MIN_TOKENS = 512
_MAX_TOKENS_EXCLUSIVE = 8192

_COMBINE_SOURCE = r"""
constexpr uint COLS = 3584;
constexpr uint WIDTH = 4;
constexpr uint COL_TILES = COLS / WIDTH;

const uint gid = thread_position_in_grid.x;
const uint token = gid / COL_TILES;
const uint col_base = (gid % COL_TILES) * WIDTH;

uint sorted_rows[TOP_K];
T route_weights[TOP_K];
#pragma clang loop unroll(full)
for (uint slot = 0; slot < TOP_K; ++slot) {
  const uint route = token * TOP_K + slot;
  sorted_rows[slot] = inverse[route];
  route_weights[slot] = weights[route];
}

#pragma clang loop unroll(full)
for (uint lane = 0; lane < WIDTH; ++lane) {
  T partials[8];
#pragma clang loop unroll(full)
  for (uint row = 0; row < 8; ++row) {
    T partial = static_cast<T>(0.0f);
#pragma clang loop unroll(full)
    for (uint slot = row; slot < TOP_K; slot += 8) {
      const T product = static_cast<T>(
          sorted_routes[sorted_rows[slot] * COLS + col_base + lane] *
          route_weights[slot]);
      partial = static_cast<T>(product + partial);
    }
    partials[row] = partial;
  }
  T total = partials[0];
#pragma clang loop unroll(full)
  for (uint row = 1; row < 8; ++row) {
    total = static_cast<T>(partials[row] + total);
  }
  combined[token * COLS + col_base + lane] = total;
}
"""


@lru_cache(maxsize=1)
def prefill_route_combine_enabled() -> bool:
    """Parse the strict default-off optimization switch."""

    value = os.environ.get(PREFILL_ROUTE_COMBINE_ENV, "0")
    if value not in {"0", "1"}:
        raise ValueError(f"{PREFILL_ROUTE_COMBINE_ENV} must be 0 or 1")
    return value == "1"


@lru_cache(maxsize=1)
def _combine_kernel():
    return mx.fast.metal_kernel(
        name="k3_sorted_route_weight_combine_prefill",
        input_names=["sorted_routes", "inverse", "weights"],
        output_names=["combined"],
        source=_COMBINE_SOURCE,
        ensure_row_contiguous=True,
    )


def fused_sorted_route_combine(
    sorted_routes: mx.array,
    inverse: mx.array,
    weights: mx.array,
) -> mx.array:
    """Restore token order, apply BF16 route weights, and reduce routes.

    The reduction deliberately matches MLX's fixed BF16 size-8/16 tree: eight
    partials followed by an ordered fold of partials one through seven into
    partial zero.
    """

    if weights.ndim < 1 or weights.shape[-1] not in _SUPPORTED_TOP_K:
        raise ValueError("invalid K3 BF16 sorted-route combine contract")
    top_k = weights.shape[-1]
    tokens = inverse.size // top_k
    if (
        sorted_routes.dtype != mx.bfloat16
        or weights.dtype != mx.bfloat16
        or inverse.size != weights.size
        or sorted_routes.size != inverse.size * _INPUT_DIMS
    ):
        raise ValueError("invalid K3 BF16 sorted-route combine contract")
    return _combine_kernel()(
        inputs=[sorted_routes, inverse, weights],
        template=[("T", sorted_routes.dtype), ("TOP_K", top_k)],
        grid=(tokens * (_INPUT_DIMS // 4), 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(1, tokens, _INPUT_DIMS)],
        output_dtypes=[sorted_routes.dtype],
        stream=mx.gpu,
    )[0]


def _supports_projection(
    projection: Any,
    *,
    input_dims: int,
    output_dims: int,
) -> bool:
    return (
        getattr(projection, "bits", None) == 2
        and getattr(projection, "group_size", None) == 128
        and getattr(
            projection,
            "_runtime_quantization_mode",
            getattr(projection, "mode", None),
        )
        == "affine2"
        and getattr(projection, "num_experts", None) == _EXPERTS
        and getattr(projection, "input_dims", None) == input_dims
        and getattr(projection, "output_dims", None) == output_dims
        and "bias" not in projection
    )


def _supports_prefill_route_combine(
    switch_mlp: Any,
    x: mx.array,
    indices: mx.array,
    weights: mx.array,
) -> bool:
    if (
        not prefill_route_combine_enabled()
        or getattr(switch_mlp, "training", True)
        or x.ndim != 3
        or x.shape[0] != 1
        or x.shape[-1] != _INPUT_DIMS
        or not (_MIN_TOKENS <= x.shape[1] < _MAX_TOKENS_EXCLUSIVE)
        or indices.shape[:-1] != x.shape[:-1]
        or indices.shape[-1] not in _SUPPORTED_TOP_K
        or weights.shape != indices.shape
        or x.dtype != mx.bfloat16
        or weights.dtype != mx.bfloat16
    ):
        return False
    activation = getattr(switch_mlp, "activation", None)
    if (
        getattr(activation, "beta", None) != 4.0
        or getattr(activation, "linear_beta", None) != 25.0
    ):
        return False
    return (
        _supports_projection(
            switch_mlp.up_proj,
            input_dims=_INPUT_DIMS,
            output_dims=_INTERMEDIATE_DIMS,
        )
        and _supports_projection(
            switch_mlp.gate_proj,
            input_dims=_INPUT_DIMS,
            output_dims=_INTERMEDIATE_DIMS,
        )
        and _supports_projection(
            switch_mlp.down_proj,
            input_dims=_INTERMEDIATE_DIMS,
            output_dims=_INPUT_DIMS,
        )
    )


def maybe_fused_k3_prefill_switch_glu_reduce(
    switch_mlp: Any,
    x: mx.array,
    indices: mx.array,
    weights: mx.array,
) -> mx.array | None:
    """Return an exact reduced prefill result, or fail closed to stock."""

    if not _supports_prefill_route_combine(switch_mlp, x, indices, weights):
        return None

    flat_indices = indices.flatten()
    order = mx.argsort(flat_indices)
    inverse = mx.argsort(order)
    sorted_indices = flat_indices[order]
    expanded = mx.expand_dims(x, (-2, -3))
    sorted_x = expanded.flatten(0, -3)[order // indices.shape[-1]]

    up = switch_mlp.up_proj(sorted_x, sorted_indices, sorted_indices=True)
    gate = switch_mlp.gate_proj(sorted_x, sorted_indices, sorted_indices=True)
    activated = switch_mlp.activation(up, gate)
    sorted_routes = switch_mlp.down_proj(
        activated,
        sorted_indices,
        sorted_indices=True,
    )
    return fused_sorted_route_combine(sorted_routes, inverse, weights)
