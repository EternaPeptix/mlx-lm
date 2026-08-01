"""Exact, default-off TP2 output-row sharding for Kimi K3 routed-up.

Kimi K3's ordinary tensor-parallel plan leaves the latent-to-hidden routed
projection replicated.  The UVMAX checkpoint stores this particular
3,584-to-7,168 projection as affine 8-bit/group-64 (despite the checkpoint's
"2bit" name).  This prototype gives each TP2 rank 3,584 output rows, performs
the native FP32-accumulating quantized projection, rounds to BF16, applies the
matching BF16 shared and residual slices in their original order, and then
all-gathers the two halves into the replicated hidden representation.

The flag is consumed while ``Model.shard`` mutates ownership.  Once a module
has been converted, disabling the flag or calling it with an unsupported
world, shape, dtype, training state, or projection is an error: a half-owned
projection cannot safely fall back to the full-width path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Optional

import mlx.core as mx
from mlx.nn.layers.distributed import shard_linear

from .kimi_k3_fused_routed_up_add import (
    K3_BITS,
    K3_GROUP_SIZE,
    K3_HIDDEN_SIZE,
    K3_ROUTED_LATENT_SIZE,
    K3_TP2_LOCAL_HIDDEN_SIZE,
    fused_routed_up_add,
    supports_fused_routed_up_add,
)


TP2_ROUTED_UP_COLUMN_ENV = "MLX_LM_KIMI_K3_TP2_ROUTED_UP_COLUMN"
K3_SPARSE_LAYER_COUNT = 92
K3_TP_WORLD_SIZE = 2
_CONFIGURED_ATTR = "_kimi_k3_tp2_routed_up_column_group"


def tp2_routed_up_column_enabled() -> bool:
    """Parse the immutable fail-closed feature flag."""

    value = os.environ.get(TP2_ROUTED_UP_COLUMN_ENV, "0")
    if value not in {"0", "1"}:
        raise ValueError(f"{TP2_ROUTED_UP_COLUMN_ENV} must be exactly '0' or '1'")
    return value == "1"


def _group_identity(group: Any) -> tuple[int, int]:
    try:
        size = int(group.size())
        rank = int(group.rank())
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("Kimi K3 routed-up column sharding requires an MLX group") from error
    if size != K3_TP_WORLD_SIZE:
        raise ValueError(
            "Kimi K3 routed-up column sharding requires exactly TP2, "
            f"got TP{size}"
        )
    if rank not in (0, 1):
        raise ValueError(f"invalid Kimi K3 TP2 rank {rank}")
    return size, rank


def _array_parameter(module: Any, name: str) -> Optional[mx.array]:
    getter = getattr(module, "get", None)
    value = getter(name) if getter is not None else getattr(module, name, None)
    return value if isinstance(value, mx.array) else None


def _projection_parameters(
    projection: Any,
    *,
    output_rows: int,
) -> tuple[mx.array, mx.array, mx.array]:
    """Validate the authoritative affine8/group64 projection contract."""

    if (
        getattr(projection, "bits", None) != K3_BITS
        or getattr(projection, "group_size", None) != K3_GROUP_SIZE
        or getattr(projection, "mode", None) != "affine"
        or _array_parameter(projection, "bias") is not None
    ):
        raise ValueError(
            "Kimi K3 routed-up column sharding requires the authoritative "
            "bias-free affine8/group64 projection"
        )
    weight = _array_parameter(projection, "weight")
    scales = _array_parameter(projection, "scales")
    biases = _array_parameter(projection, "biases")
    expected_weight = (
        output_rows,
        K3_ROUTED_LATENT_SIZE * K3_BITS // 32,
    )
    expected_scales = (
        output_rows,
        K3_ROUTED_LATENT_SIZE // K3_GROUP_SIZE,
    )
    if (
        weight is None
        or scales is None
        or biases is None
        or weight.shape != expected_weight
        or scales.shape != expected_scales
        or biases.shape != expected_scales
        or weight.dtype != mx.uint32
        or scales.dtype != mx.bfloat16
        or biases.dtype != mx.bfloat16
    ):
        raise ValueError(
            "Kimi K3 routed-up projection parameters do not match the exact "
            f"{output_rows}x{K3_ROUTED_LATENT_SIZE} affine8/group64 contract"
        )
    return weight, scales, biases


def configure_k3_tp2_routed_up_column(sparse_moe: Any, group: Any) -> bool:
    """Output-shard one loaded full-width routed-up projection in place."""

    if not tp2_routed_up_column_enabled():
        return False
    _group_identity(group)
    if getattr(sparse_moe, "latent_size", None) != K3_ROUTED_LATENT_SIZE:
        raise ValueError("Kimi K3 routed-up column sharding has wrong latent width")
    if getattr(sparse_moe, _CONFIGURED_ATTR, None) is not None:
        raise ValueError("Kimi K3 routed-up column sharding was configured twice")

    projection = getattr(sparse_moe, "routed_expert_up_proj", None)
    _projection_parameters(projection, output_rows=K3_HIDDEN_SIZE)
    local_projection = shard_linear(
        projection,
        "all-to-sharded",
        group=group,
    )
    _projection_parameters(
        local_projection,
        output_rows=K3_TP2_LOCAL_HIDDEN_SIZE,
    )
    sparse_moe.routed_expert_up_proj = local_projection
    setattr(sparse_moe, _CONFIGURED_ATTR, group)
    return True


def _validate_runtime_inputs(
    sparse_moe: Any,
    routed_latent: mx.array,
    shared: Optional[mx.array],
    residual: Optional[mx.array],
) -> tuple[Any, int, tuple[mx.array, mx.array, mx.array]]:
    group = getattr(sparse_moe, _CONFIGURED_ATTR, None)
    if group is None:
        raise ValueError("Kimi K3 routed-up column sharding is not configured")
    _, rank = _group_identity(group)
    if not tp2_routed_up_column_enabled():
        raise RuntimeError(
            f"{TP2_ROUTED_UP_COLUMN_ENV} cannot be disabled after weights are sharded"
        )
    if getattr(sparse_moe, "training", True):
        raise ValueError("Kimi K3 routed-up column sharding is inference-only")
    if shared is None or residual is None:
        raise ValueError(
            "Kimi K3 routed-up column sharding requires shared and residual branches"
        )
    if (
        not isinstance(routed_latent, mx.array)
        or not isinstance(shared, mx.array)
        or not isinstance(residual, mx.array)
        or routed_latent.ndim != 3
        or routed_latent.shape[-1] != K3_ROUTED_LATENT_SIZE
        or routed_latent.shape[:-1] != shared.shape[:-1]
        or shared.shape != residual.shape
        or shared.shape[-1] != K3_HIDDEN_SIZE
        or routed_latent.dtype != mx.bfloat16
        or shared.dtype != mx.bfloat16
        or residual.dtype != mx.bfloat16
        or any(int(dim) <= 0 for dim in routed_latent.shape[:-1])
    ):
        raise ValueError(
            "Kimi K3 routed-up column inputs require matching positive [B,T,*] "
            "BF16 tensors with widths 3584 and 7168"
        )
    projection = getattr(sparse_moe, "routed_expert_up_proj", None)
    parameters = _projection_parameters(
        projection,
        output_rows=K3_TP2_LOCAL_HIDDEN_SIZE,
    )
    return group, rank, parameters


def ordered_local_routed_up_add(
    projection: Any,
    routed_latent: mx.array,
    shared_slice: mx.array,
    residual_slice: mx.array,
    parameters: tuple[mx.array, mx.array, mx.array],
) -> mx.array:
    """Compute one output-row half with stock FP32/BF16 boundaries."""

    if routed_latent.shape[:-1] != shared_slice.shape[:-1] or (
        shared_slice.shape != residual_slice.shape
        or shared_slice.shape[-1] != K3_TP2_LOCAL_HIDDEN_SIZE
    ):
        raise ValueError("Kimi K3 TP2 local routed-up slices have incompatible shapes")

    # The accepted Metal writer is exact and removes the two local elementwise
    # dispatches for decode.  Multi-token prefill retains the native QMM and
    # ordered BF16 adds, so column ownership is correct at every context size.
    if supports_fused_routed_up_add(
        routed_latent,
        shared_slice,
        residual_slice,
        parameters,
    ):
        return fused_routed_up_add(
            routed_latent,
            shared_slice,
            residual_slice,
            parameters,
        )
    routed = projection(routed_latent)
    moe_output = routed + shared_slice
    return residual_slice + moe_output


def all_gather_last_dimension(
    local_hidden: mx.array,
    group: Any,
    *,
    collective: Optional[Callable[..., mx.array]] = None,
) -> mx.array:
    """All-gather a final-axis shard using MLX's leading-axis collective."""

    _group_identity(group)
    if (
        not isinstance(local_hidden, mx.array)
        or local_hidden.ndim != 3
        or local_hidden.shape[-1] != K3_TP2_LOCAL_HIDDEN_SIZE
        or local_hidden.dtype != mx.bfloat16
    ):
        raise ValueError("Kimi K3 TP2 all-gather requires a BF16 hidden half")
    rank_major = mx.contiguous(mx.moveaxis(local_hidden, -1, 0))
    gather = collective or mx.distributed.all_gather
    gathered = gather(rank_major, group=group)
    expected = (
        K3_HIDDEN_SIZE,
        local_hidden.shape[0],
        local_hidden.shape[1],
    )
    if not isinstance(gathered, mx.array) or gathered.shape != expected:
        raise RuntimeError(
            "Kimi K3 TP2 routed-up all-gather returned shape "
            f"{getattr(gathered, 'shape', None)}, expected {expected}"
        )
    return mx.contiguous(mx.moveaxis(gathered, 0, -1))


def maybe_k3_tp2_routed_up_column(
    sparse_moe: Any,
    routed_latent: mx.array,
    shared: Optional[mx.array],
    residual: Optional[mx.array],
) -> Optional[mx.array]:
    """Return replicated exact hidden state, or ``None`` when unconfigured."""

    if getattr(sparse_moe, _CONFIGURED_ATTR, None) is None:
        return None
    group, rank, parameters = _validate_runtime_inputs(
        sparse_moe,
        routed_latent,
        shared,
        residual,
    )
    assert shared is not None and residual is not None
    start = rank * K3_TP2_LOCAL_HIDDEN_SIZE
    end = start + K3_TP2_LOCAL_HIDDEN_SIZE
    local_hidden = ordered_local_routed_up_add(
        sparse_moe.routed_expert_up_proj,
        routed_latent,
        shared[..., start:end],
        residual[..., start:end],
        parameters,
    )
    return all_gather_last_dimension(local_hidden, group)


@dataclass(frozen=True)
class RoutedUpColumnCost:
    full_projection_ms: float
    half_projection_ms: float
    all_gather_ms: float
    sparse_layers: int = K3_SPARSE_LAYER_COUNT

    def __post_init__(self) -> None:
        values = (
            self.full_projection_ms,
            self.half_projection_ms,
            self.all_gather_ms,
        )
        if any(type(value) not in (int, float) or value < 0 for value in values):
            raise ValueError("Kimi K3 routed-up costs must be non-negative numbers")
        if type(self.sparse_layers) is not int or self.sparse_layers <= 0:
            raise ValueError("Kimi K3 sparse layer count must be positive")

    @property
    def break_even_all_gather_ms(self) -> float:
        return self.full_projection_ms - self.half_projection_ms

    @property
    def saved_ms_per_layer(self) -> float:
        return self.break_even_all_gather_ms - self.all_gather_ms

    @property
    def saved_ms_per_token(self) -> float:
        return self.sparse_layers * self.saved_ms_per_layer

    def projected_tokens_per_second(self, baseline_ms_per_token: float) -> float:
        if type(baseline_ms_per_token) not in (int, float) or baseline_ms_per_token <= 0:
            raise ValueError("baseline token latency must be positive")
        candidate_ms = baseline_ms_per_token - self.saved_ms_per_token
        if candidate_ms <= 0:
            raise ValueError("cost model produced a non-positive token latency")
        return 1000.0 / candidate_ms
