# Copyright © 2026 Apple Inc.

import os
import re
from dataclasses import dataclass
from functools import lru_cache, partial
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace, shard_linear, sum_gradients

from .base import (
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from .cache import ArraysCache, BatchKVCache, KVCache, SpeculativeReplayState
from .gated_delta import (
    compute_g,
    compute_g_safe,
    gated_delta_kernel,
    gated_delta_update,
)
from .kimi_k3_attnres_rms import maybe_fused_attnres_rms
from .kimi_k3_fused_expert import (
    fused_k3_experts_enabled,
    maybe_fused_k3_switch_glu,
    maybe_fused_k3_switch_glu_reduce,
)
from .kimi_k3_fused_routed_up_add import maybe_fused_k3_routed_up_add
from .kimi_k3_fused_router import maybe_fused_k3_router
from .kimi_k3_multibank_moe_front import maybe_multibank_k3_moe_front
from .kimi_k3_packed_kda_projections import (
    invalidate_packed_k3_kda_skinny,
    invalidate_packed_k3_kda_wide,
    maybe_authoritative_packed_k3_kda_skinny,
    maybe_authoritative_packed_k3_kda_wide,
)
from .kimi_k3_packed_moe_front import (
    invalidate_packed_k3_moe_front,
    maybe_authoritative_packed_k3_moe_front,
    maybe_packed_k3_moe_front,
)
from .kimi_linear import ShortConv1d
from .mla import MultiLinear
from .switch_layers import SwitchGLU

COMPILED_DECODE_ENV = "MLX_LM_KIMI_K3_COMPILED_DECODE"
# Segment 0 is the KDA prefix, 1..N-1 are MLA-to-MLA transitions, and
# segment N is the final MLA/output tail. The production K3 topology has N=24.
COMPILED_DECODE_SEGMENTS_ENV = "MLX_LM_KIMI_K3_COMPILED_DECODE_SEGMENTS"
ASYNC_DECODE_BOUNDARIES_ENV = "MLX_LM_KIMI_K3_ASYNC_DECODE_BOUNDARIES"
ASYNC_DECODE_STATE_ENV = "MLX_LM_KIMI_K3_ASYNC_DECODE_STATE"
REPLAYSSM_SPECULATIVE_ENV = "MLX_LM_KIMI_K3_REPLAYSSM_SPECULATIVE"
EXACT_WIDE_SHORT_CONV_ENV = "MLX_LM_KIMI_K3_EXACT_WIDE_SHORT_CONV"
_EXACT_WIDE_SHORT_CONV_MAX_WIDTH = 8


def replayssm_speculative_enabled() -> bool:
    """Parse the fail-closed Kimi K3 ReplaySSM opt-in."""

    value = os.environ.get(REPLAYSSM_SPECULATIVE_ENV, "0")
    if value not in {"0", "1"}:
        raise ValueError(f"{REPLAYSSM_SPECULATIVE_ENV} must be 0 or 1")
    return value == "1"


def _validate_aux_hidden_state_layer_ids(
    layer_ids: Optional[Tuple[int, ...]],
    num_layers: int,
) -> Tuple[int, ...]:
    if layer_ids is None:
        return ()
    if not layer_ids:
        raise ValueError("auxiliary hidden-state layer ids cannot be empty")
    if any(type(layer_id) is not int for layer_id in layer_ids):
        raise TypeError("auxiliary hidden-state layer ids must be integers")
    if tuple(sorted(set(layer_ids))) != layer_ids:
        raise ValueError(
            "auxiliary hidden-state layer ids must be unique and increasing"
        )
    if layer_ids[0] < 0 or layer_ids[-1] >= num_layers:
        raise ValueError("auxiliary hidden-state layer id is outside the target")
    return layer_ids


@lru_cache(maxsize=1)
def exact_wide_short_conv_enabled() -> bool:
    """Return whether exact recurrent short-convolution was requested.

    This flag guards a correctness path, so malformed values fail closed
    instead of silently selecting the generic wide ``Conv1d`` fallback.
    """

    value = os.environ.get(EXACT_WIDE_SHORT_CONV_ENV, "0")
    if value not in {"0", "1"}:
        raise ValueError(f"{EXACT_WIDE_SHORT_CONV_ENV} must be exactly '0' or '1'")
    return value == "1"


def _parse_compiled_decode_segments(
    selector: str,
    segment_count: int,
) -> FrozenSet[int]:
    """Parse an immutable, inclusive compiled-decode segment selector."""
    if segment_count < 0:
        raise ValueError("Compiled decode segment count cannot be negative")

    selector = selector.strip().lower()
    if selector == "all":
        return frozenset(range(segment_count))
    if selector == "none":
        return frozenset()
    if not selector:
        raise ValueError(
            f"{COMPILED_DECODE_SEGMENTS_ENV} must be 'all', 'none', "
            "or a comma-separated list of indices and inclusive ranges"
        )

    segments = set()
    for item in selector.split(","):
        item = item.strip()
        if not item:
            raise ValueError(
                f"Invalid empty item in {COMPILED_DECODE_SEGMENTS_ENV}={selector!r}"
            )

        bounds = [part.strip() for part in item.split("-")]
        if len(bounds) == 1 and bounds[0].isdigit():
            start = end = int(bounds[0])
        elif len(bounds) == 2 and bounds[0].isdigit() and bounds[1].isdigit():
            start, end = (int(bound) for bound in bounds)
            if start > end:
                raise ValueError(
                    f"Reversed range {item!r} in {COMPILED_DECODE_SEGMENTS_ENV}"
                )
        else:
            raise ValueError(f"Invalid item {item!r} in {COMPILED_DECODE_SEGMENTS_ENV}")

        if start < 0 or end >= segment_count:
            valid = f"0-{segment_count - 1}" if segment_count else "none"
            raise ValueError(f"Segment {item!r} is outside the valid range {valid}")
        segments.update(range(start, end + 1))

    return frozenset(segments)


def _parse_async_decode_boundaries(
    selector: str,
    layer_count: int,
) -> FrozenSet[int]:
    """Parse eager-decode layer boundaries that should be submitted early."""

    if layer_count < 0:
        raise ValueError("Kimi K3 layer count cannot be negative")

    selector = selector.strip().lower()
    if selector == "none":
        return frozenset()
    if selector == "laguna8":
        # Reproduce the exact scheduling ladder that proved useful for Laguna:
        # one early submission followed by eight-layer stages.
        return frozenset(
            boundary
            for boundary in (1, *range(7, layer_count - 1, 8))
            if boundary < layer_count - 1
        )
    if selector == "block8":
        return frozenset(range(7, layer_count - 1, 8))
    if selector == "all":
        return frozenset(range(max(layer_count - 1, 0)))
    if not selector:
        raise ValueError(
            f"{ASYNC_DECODE_BOUNDARIES_ENV} must be 'none', 'laguna8', "
            "'block8', 'all', or a comma-separated list of layer indices "
            "and inclusive ranges"
        )

    boundaries = set()
    for item in selector.split(","):
        item = item.strip()
        if not item:
            raise ValueError(
                f"Invalid empty item in "
                f"{ASYNC_DECODE_BOUNDARIES_ENV}={selector!r}"
            )

        bounds = [part.strip() for part in item.split("-")]
        if len(bounds) == 1 and bounds[0].isdigit():
            start = end = int(bounds[0])
        elif len(bounds) == 2 and bounds[0].isdigit() and bounds[1].isdigit():
            start, end = (int(bound) for bound in bounds)
            if start > end:
                raise ValueError(
                    f"Reversed range {item!r} in "
                    f"{ASYNC_DECODE_BOUNDARIES_ENV}"
                )
        else:
            raise ValueError(
                f"Invalid item {item!r} in {ASYNC_DECODE_BOUNDARIES_ENV}"
            )

        # The final layer is already submitted by the generation boundary and
        # therefore cannot be an early scheduling boundary.
        if start < 0 or end >= layer_count - 1:
            valid = f"0-{layer_count - 2}" if layer_count > 1 else "none"
            raise ValueError(
                f"Layer boundary {item!r} is outside the valid range {valid}"
            )
        boundaries.update(range(start, end + 1))

    return frozenset(boundaries)


@mx.compile
def _group_expert_select(
    gates: mx.array,
    bias: Optional[mx.array],
    top_k: int,
    n_group: int,
    topk_group: int,
    routed_scaling_factor: float,
    renormalize: bool,
) -> Tuple[mx.array, mx.array]:
    in_type = gates.dtype
    scores = mx.sigmoid(gates.astype(mx.float32))
    orig_scores = scores
    if bias is not None:
        scores = scores + bias.astype(scores.dtype)

    if n_group > 1:
        scores = mx.unflatten(scores, axis=-1, shape=(n_group, -1))
        group_scores = mx.topk(scores, 2, axis=-1).sum(axis=-1, keepdims=True)
        k = n_group - topk_group
        group_idx = mx.argpartition(group_scores, kth=k - 1, axis=-2)[..., :k, :]
        scores = mx.put_along_axis(
            scores,
            mx.stop_gradient(group_idx),
            mx.array(0.0, dtype=scores.dtype),
            axis=-2,
        )
        scores = mx.flatten(scores, -2, -1)

    inds = mx.argpartition(-scores, kth=top_k - 1, axis=-1)[..., :top_k]
    scores = mx.take_along_axis(orig_scores, inds, axis=-1)

    if top_k > 1 and renormalize:
        denominator = scores.sum(axis=-1, keepdims=True) + 1e-20
        scores = scores / denominator

    return inds, (scores * routed_scaling_factor).astype(in_type)


@dataclass
class TextArgs(BaseModelArgs):
    model_type: str = "kimi_linear"
    vocab_size: int = 163840
    hidden_size: int = 7168
    num_hidden_layers: int = 93
    num_attention_heads: int = 96
    num_key_value_heads: int = 96
    intermediate_size: int = 33792
    rms_norm_eps: float = 1e-5
    max_position_embeddings: int = 1048576
    linear_attn_config: Optional[Dict[str, Any]] = None
    hidden_act: str = "situ"
    activation_situ_beta: Optional[float] = None
    activation_situ_linear_beta: Optional[float] = None
    attn_res_block_size: Optional[int] = None
    q_lora_rank: Optional[int] = None
    kv_lora_rank: Optional[int] = None
    qk_nope_head_dim: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None
    v_head_dim: Optional[int] = None
    mla_use_nope: bool = True
    mla_use_output_gate: bool = False
    num_experts: Optional[int] = None
    num_experts_per_token: int = 16
    num_shared_experts: int = 0
    moe_intermediate_size: Optional[int] = None
    moe_router_activation_func: str = "sigmoid"
    moe_renormalize: bool = True
    routed_scaling_factor: float = 1.0
    first_k_dense_replace: int = 0
    moe_layer_freq: int = 1
    use_grouped_topk: bool = True
    num_expert_group: int = 1
    topk_group: int = 1
    routed_expert_hidden_size: Optional[int] = None
    latent_moe_use_norm: bool = False
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if self.hidden_act != "situ":
            raise ValueError(f"Unsupported activation '{self.hidden_act}'")
        if self.moe_router_activation_func != "sigmoid":
            raise ValueError(
                f"Unsupported MoE router activation '{self.moe_router_activation_func}'"
            )


@dataclass
class ModelArgs(BaseModelArgs):
    text_config: Union[TextArgs, dict]
    model_type: str = "kimi_k3"

    @classmethod
    def from_dict(cls, params):
        if "text_config" not in params:
            return cls(model_type=params["model_type"], text_config=params)
        return super().from_dict(params)

    def __post_init__(self):
        if isinstance(self.text_config, dict):
            self.text_config = TextArgs.from_dict(self.text_config)


@partial(mx.compile, shapeless=True)
def _situ(x, gate, beta, linear_beta):
    dtype = x.dtype
    gate = gate.astype(mx.float32)
    x = x.astype(mx.float32)
    a = beta * mx.tanh(gate / beta) * mx.sigmoid(gate)
    if linear_beta is not None:
        x = linear_beta * mx.tanh(x / linear_beta)
    return (a * x).astype(dtype)


class SiTU(nn.Module):
    def __init__(self, beta: float = 1.0, linear_beta: Optional[float] = None):
        super().__init__()
        self.beta = beta
        self.linear_beta = linear_beta

    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        return _situ(x, gate, self.beta, self.linear_beta)


class KimiK3MLP(nn.Module):
    def __init__(self, args: TextArgs, intermediate_size: Optional[int] = None):
        super().__init__()
        dim = args.hidden_size
        hidden = intermediate_size or args.intermediate_size
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)
        self.beta = args.activation_situ_beta or 1.0
        self.linear_beta = args.activation_situ_linear_beta

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(
            _situ(self.up_proj(x), self.gate_proj(x), self.beta, self.linear_beta)
        )


class ResidualBlocks:
    def __init__(self, eps: float):
        self.eps = eps
        self.raw = None
        self.inv_rms = None

    def append(self, x: mx.array):
        xf = x.astype(mx.float32)
        n = mx.rsqrt((xf * xf).mean(axis=-1) + self.eps)[None]
        r = x[None]
        if self.raw is None:
            self.raw = r
            self.inv_rms = n
        else:
            self.raw = mx.concatenate([self.raw, r])
            self.inv_rms = mx.concatenate([self.inv_rms, n])


@mx.compile
def _attn_res_combine(raw, inv_rms, partial_sum, w_eff, eps):
    pf = partial_sum.astype(mx.float32)
    p_logit = (pf @ w_eff) * mx.rsqrt((pf * pf).mean(axis=-1) + eps)
    logits = mx.concatenate([(raw.astype(mx.float32) @ w_eff) * inv_rms, p_logit[None]])
    p = mx.softmax(logits, axis=0, precise=True)
    out = (p[:-1, ..., None] * raw).sum(axis=0) + p[-1, ..., None] * partial_sum
    return out.astype(partial_sum.dtype)


_ATTN_RES_SOURCE = """
    constexpr int NACC = K + 2;
    constexpr int NSIMD = THREADS / 32;

    auto n = threadgroup_position_in_grid.y;
    auto tid = thread_position_in_threadgroup.x;
    auto lane = thread_index_in_simdgroup;
    auto sg = simdgroup_index_in_threadgroup;

    auto partial_ = partial + n * D;

    float acc[NACC];
    for (int i = 0; i < NACC; ++i) {
      acc[i] = 0.0f;
    }
    for (uint d = tid; d < D; d += THREADS) {
      float w = static_cast<float>(w_eff[d]);
      float pv = static_cast<float>(partial_[d]);
      for (int k = 0; k < K; ++k) {
        acc[k] += static_cast<float>(raw[(k * N + n) * D + d]) * w;
      }
      acc[K] += pv * w;
      acc[K + 1] += pv * pv;
    }

    threadgroup float shm[NACC * NSIMD];
    for (int i = 0; i < NACC; ++i) {
      float s = simd_sum(acc[i]);
      if (lane == 0) {
        shm[i * NSIMD + sg] = s;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float weights[K + 1];
    if (tid == 0) {
      float tot[NACC];
      for (int i = 0; i < NACC; ++i) {
        tot[i] = 0.0f;
        for (int j = 0; j < NSIMD; ++j) {
          tot[i] += shm[i * NSIMD + j];
        }
      }
      float rinv = metal::rsqrt(tot[K + 1] / D + eps[0]);
      float logits[K + 1];
      float m = -1e30f;
      for (int k = 0; k < K; ++k) {
        logits[k] = tot[k] * inv_rms[k * N + n];
        m = metal::max(m, logits[k]);
      }
      logits[K] = tot[K] * rinv;
      m = metal::max(m, logits[K]);
      float denom = 0.0f;
      for (int k = 0; k <= K; ++k) {
        logits[k] = metal::exp(logits[k] - m);
        denom += logits[k];
      }
      for (int k = 0; k <= K; ++k) {
        weights[k] = logits[k] / denom;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float wp = weights[K];
    auto out_ = out + n * D;
    for (uint d = tid; d < D; d += THREADS) {
      float o = wp * static_cast<float>(partial_[d]);
      for (int k = 0; k < K; ++k) {
        o += weights[k] * static_cast<float>(raw[(k * N + n) * D + d]);
      }
      out_[d] = static_cast<InT>(o);
    }
"""

_attn_res_kernel = (
    mx.fast.metal_kernel(
        name="attnres_mix",
        input_names=["raw", "inv_rms", "partial", "w_eff", "eps", "N"],
        output_names=["out"],
        source=_ATTN_RES_SOURCE,
    )
    if mx.metal.is_available()
    else None
)

_ATTN_RES_THREADS = 512
_attn_res_eps_cache: Dict[float, mx.array] = {}


def _attn_res_mix(
    blocks: ResidualBlocks,
    partial_sum: mx.array,
    w_eff: mx.array,
    eps: float,
    use_kernel: bool = True,
) -> mx.array:
    if blocks.raw is None:
        return partial_sum
    if use_kernel and _attn_res_kernel is not None and mx.default_device() == mx.gpu:
        raw = blocks.raw
        K, D = raw.shape[0], raw.shape[-1]
        N = raw.size // (K * D)
        eps_arr = _attn_res_eps_cache.get(eps)
        if eps_arr is None:
            eps_arr = _attn_res_eps_cache.setdefault(
                eps, mx.array([eps], dtype=mx.float32)
            )
        return _attn_res_kernel(
            inputs=[raw, blocks.inv_rms, partial_sum, w_eff, eps_arr, N],
            template=[
                ("InT", partial_sum.dtype),
                ("K", K),
                ("D", D),
                ("THREADS", _ATTN_RES_THREADS),
            ],
            grid=(_ATTN_RES_THREADS, N, 1),
            threadgroup=(_ATTN_RES_THREADS, 1, 1),
            output_shapes=[partial_sum.shape],
            output_dtypes=[partial_sum.dtype],
        )[0]
    return _attn_res_combine(blocks.raw, blocks.inv_rms, partial_sum, w_eff, eps)


_SHORT_CONV_SOURCE = """
    auto c = thread_position_in_grid.x;
    auto b = thread_position_in_grid.y;

    const device T* s = state + b * (KS - 1) * C;
    float v = 0.0f;
    for (int j = 0; j < KS - 1; ++j) {
      v += static_cast<float>(w[c * KS + j]) * static_cast<float>(s[j * C + c]);
    }
    v += static_cast<float>(w[c * KS + KS - 1]) * static_cast<float>(x[b * C + c]);
    y[b * C + c] = static_cast<T>(v / (1.0f + metal::exp(-v)));

    device T* ns = new_state + b * (KS - 1) * C;
    for (int j = 0; j < KS - 2; ++j) {
      ns[j * C + c] = s[(j + 1) * C + c];
    }
    ns[(KS - 2) * C + c] = x[b * C + c];
"""

_short_conv_kernel = (
    mx.fast.metal_kernel(
        name="k3_short_conv_step",
        input_names=["x", "state", "w"],
        output_names=["y", "new_state"],
        source=_SHORT_CONV_SOURCE,
    )
    if mx.metal.is_available()
    else None
)

_SHORT_CONV_WIDE_SOURCE = """
    auto c = thread_position_in_grid.x;
    auto b = thread_position_in_grid.y;

    const device T* initial_state = state + b * (KS - 1) * C;
    float local_state[KS - 1];
    for (int j = 0; j < KS - 1; ++j) {
      local_state[j] = static_cast<float>(initial_state[j * C + c]);
    }

    for (int t = 0; t < L; ++t) {
      auto input = static_cast<float>(x[(b * L + t) * C + c]);
      float v = 0.0f;
      for (int j = 0; j < KS - 1; ++j) {
        v += static_cast<float>(w[c * KS + j]) * local_state[j];
      }
      v += static_cast<float>(w[c * KS + KS - 1]) * input;
      y[(b * L + t) * C + c] =
          static_cast<T>(v / (1.0f + metal::exp(-v)));

      for (int j = 0; j < KS - 2; ++j) {
        local_state[j] = local_state[j + 1];
      }
      local_state[KS - 2] = input;
    }

    device T* final_state = new_state + b * (KS - 1) * C;
    for (int j = 0; j < KS - 1; ++j) {
      final_state[j * C + c] = static_cast<T>(local_state[j]);
    }
"""

_short_conv_wide_kernel = (
    mx.fast.metal_kernel(
        name="k3_short_conv_exact_wide",
        input_names=["x", "state", "w"],
        output_names=["y", "new_state"],
        source=_SHORT_CONV_WIDE_SOURCE,
    )
    if mx.metal.is_available()
    else None
)

_SHORT_CONV_HISTORY_SOURCE = """
    auto c = thread_position_in_grid.x;
    auto b = thread_position_in_grid.y;

    const device T* initial_state = state + b * (KS - 1) * C;
    float local_state[KS - 1];
    for (int j = 0; j < KS - 1; ++j) {
      local_state[j] = static_cast<float>(initial_state[j * C + c]);
    }

    for (int t = 0; t < L; ++t) {
      auto input = static_cast<float>(x[(b * L + t) * C + c]);
      float v = 0.0f;
      for (int j = 0; j < KS - 1; ++j) {
        v += static_cast<float>(w[c * KS + j]) * local_state[j];
      }
      v += static_cast<float>(w[c * KS + KS - 1]) * input;
      y[(b * L + t) * C + c] =
          static_cast<T>(v / (1.0f + metal::exp(-v)));

      for (int j = 0; j < KS - 2; ++j) {
        local_state[j] = local_state[j + 1];
      }
      local_state[KS - 2] = input;

      device T* checkpoint =
          state_history + ((b * L + t) * (KS - 1)) * C;
      for (int j = 0; j < KS - 1; ++j) {
        checkpoint[j * C + c] = static_cast<T>(local_state[j]);
      }
    }

    device T* final_state = new_state + b * (KS - 1) * C;
    for (int j = 0; j < KS - 1; ++j) {
      final_state[j * C + c] = static_cast<T>(local_state[j]);
    }
"""

_short_conv_history_kernel = (
    mx.fast.metal_kernel(
        name="k3_short_conv_history",
        input_names=["x", "state", "w"],
        output_names=["y", "new_state", "state_history"],
        source=_SHORT_CONV_HISTORY_SOURCE,
    )
    if mx.metal.is_available()
    else None
)


class KimiK3ShortConv(ShortConv1d):
    def __call__(
        self,
        x,
        state,
        mask=None,
        lengths=None,
        return_state_history=False,
    ):
        exact_wide = exact_wide_short_conv_enabled()
        if return_state_history:
            if (
                _short_conv_history_kernel is None
                or self.training
                or x.shape[1] <= 1
                or state is None
                or mask is not None
                or lengths is not None
                or x.dtype != state.dtype
                or x.dtype != self.conv.weight.dtype
                or mx.default_device() != mx.gpu
            ):
                raise ValueError(
                    "Kimi K3 speculative short-conv checkpoints are unsupported"
                )
            B, L, C = x.shape
            return _short_conv_history_kernel(
                inputs=[x, state, self.conv.weight],
                template=[
                    ("T", x.dtype),
                    ("C", C),
                    ("KS", self.kernel_size),
                    ("L", L),
                ],
                grid=(C, B, 1),
                threadgroup=(min(1024, C), 1, 1),
                output_shapes=[
                    x.shape,
                    state.shape,
                    (B, L, self.kernel_size - 1, C),
                ],
                output_dtypes=[x.dtype, x.dtype, x.dtype],
            )
        if exact_wide and 1 < x.shape[1] <= _EXACT_WIDE_SHORT_CONV_MAX_WIDTH:
            if (
                _short_conv_wide_kernel is None
                or self.training
                or state is None
                or mask is not None
                or lengths is not None
                or x.dtype != state.dtype
                or x.dtype != self.conv.weight.dtype
                or mx.default_device() != mx.gpu
            ):
                raise ValueError(
                    "Exact Kimi K3 wide short-convolution requires populated, "
                    "unpadded, matching-dtype Metal state and width in [2, 8]"
                )
            B, L, C = x.shape
            return _short_conv_wide_kernel(
                inputs=[x, state, self.conv.weight],
                template=[
                    ("T", x.dtype),
                    ("C", C),
                    ("KS", self.kernel_size),
                    ("L", L),
                ],
                grid=(C, B, 1),
                threadgroup=(min(1024, C), 1, 1),
                output_shapes=[x.shape, state.shape],
                output_dtypes=[x.dtype, x.dtype],
            )
        if (
            _short_conv_kernel is None
            or self.training
            or x.shape[1] != 1
            or state is None
            or mask is not None
            or lengths is not None
            or x.dtype != state.dtype
            or x.dtype != self.conv.weight.dtype
            or mx.default_device() != mx.gpu
        ):
            return super().__call__(x, state, mask, lengths)
        B, _, C = x.shape
        return _short_conv_kernel(
            inputs=[x, state, self.conv.weight],
            template=[("T", x.dtype), ("C", C), ("KS", self.kernel_size)],
            grid=(C, B, 1),
            threadgroup=(min(1024, C), 1, 1),
            output_shapes=[x.shape, state.shape],
            output_dtypes=[x.dtype, x.dtype],
        )


class KimiK3DeltaAttention(nn.Module):
    def __init__(self, args: TextArgs, layer_idx: int):
        super().__init__()
        cfg = args.linear_attn_config

        self.layer_idx = layer_idx
        self.num_heads = cfg["num_heads"]
        self.head_dim = cfg["head_dim"]
        self.conv_kernel = cfg["short_conv_kernel_size"]
        self.projection_dim = self.num_heads * self.head_dim
        self.scale = float(self.head_dim) ** -0.5
        self.lower_bound = cfg.get("gate_lower_bound", None)
        self.use_full_rank_gate = cfg.get("use_full_rank_gate", False)

        hidden = args.hidden_size
        self.qkv_proj = nn.Linear(hidden, 3 * self.projection_dim, bias=False)
        self.qkv_conv = KimiK3ShortConv(3 * self.projection_dim, self.conv_kernel)

        self.f_a_proj = nn.Linear(hidden, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, self.projection_dim, bias=False)
        self.b_proj = nn.Linear(hidden, self.num_heads, bias=False)

        if self.use_full_rank_gate:
            self.g_proj = nn.Linear(hidden, self.projection_dim, bias=False)
        else:
            self.g_a_proj = nn.Linear(hidden, self.head_dim, bias=False)
            self.g_b_proj = nn.Linear(self.head_dim, self.projection_dim, bias=False)

        self.A_log = mx.log(
            mx.random.uniform(low=1.0, high=16.0, shape=(self.num_heads,))
        )
        self.dt_bias = mx.zeros((self.projection_dim,))

        self.o_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.o_proj = nn.Linear(self.projection_dim, hidden, bias=False)
        self._step = None

    def _decode_core(self, x, conv_state, ssm_state):
        B = x.shape[0]
        P = self.projection_dim
        packed_wide = maybe_authoritative_packed_k3_kda_wide(self, x)
        if packed_wide is None:
            projected_qkv = self.qkv_proj(x)
            gate = None
        else:
            projected_qkv, gate = packed_wide
        qkv, conv_state = self.qkv_conv(
            projected_qkv,
            conv_state,
            None,
            None,
        )

        q = qkv[..., :P].reshape(B, 1, self.num_heads, self.head_dim)
        k = qkv[..., P : 2 * P].reshape(B, 1, self.num_heads, self.head_dim)
        v = qkv[..., 2 * P :].reshape(B, 1, self.num_heads, self.head_dim)

        eps = 1e-6 / self.head_dim
        q = (self.scale**2) * mx.fast.rms_norm(q, None, eps)
        k = self.scale * mx.fast.rms_norm(k, None, eps)

        packed_skinny = maybe_authoritative_packed_k3_kda_skinny(self, x)
        if packed_skinny is None:
            f_a = self.f_a_proj(x)
            g_a = None
            b_logits = self.b_proj(x)
        elif self.use_full_rank_gate:
            f_a, b_logits = packed_skinny
            g_a = None
        else:
            f_a, g_a, b_logits = packed_skinny

        a_logits = self.f_b_proj(f_a).reshape(
            B, 1, self.num_heads, self.head_dim
        )
        b_logits = b_logits.reshape(B, 1, self.num_heads)

        out, ssm_state = gated_delta_update(
            q,
            k,
            v,
            a_logits,
            b_logits,
            self.A_log.reshape(self.num_heads, 1),
            self.dt_bias.reshape(self.num_heads, self.head_dim),
            state=ssm_state,
            mask=None,
            use_kernel=True,
            lower_bound=self.lower_bound,
        )

        if self.use_full_rank_gate:
            if gate is None:
                gate = self.g_proj(x)
        else:
            if g_a is None:
                g_a = self.g_a_proj(x)
            gate = self.g_b_proj(g_a)
        gate = gate.reshape(B, 1, self.num_heads, self.head_dim)
        out = (
            self.o_norm(out.reshape(B, 1, self.num_heads, self.head_dim))
            * mx.sigmoid(gate)
        ).reshape(B, 1, -1)
        return self.o_proj(out), conv_state, ssm_state

    def _replay_speculative_ssm(
        self,
        initial_state: mx.array,
        raw_inputs: Tuple[mx.array, ...],
        consumed: int,
    ) -> mx.array:
        """Fold an accepted raw-input prefix through the baseline recurrence."""

        if len(raw_inputs) != 4:
            raise ValueError("Kimi K3 ReplaySSM requires (v, k, gk, beta)")
        raw_v, raw_k, gk, beta = raw_inputs
        if consumed < 1 or consumed > raw_v.shape[1]:
            raise ValueError("Kimi K3 ReplaySSM consumed width is invalid")

        raw_v = raw_v[:, :consumed]
        raw_k = raw_k[:, :consumed]
        gk = gk[:, :consumed]
        beta = beta[:, :consumed]

        # Keep the same division-form RMS normalization used by the verifier.
        # MLX's ``gk`` is already the post-exp multiplicative decay consumed by
        # gated_delta_kernel; re-log/re-exp would break exact rollback.
        eps = 1e-6 / self.head_dim
        k = self.scale * mx.fast.rms_norm(raw_k, None, eps)
        q = mx.zeros_like(k)
        _, state = gated_delta_kernel(
            q,
            k,
            raw_v,
            gk,
            beta,
            initial_state,
        )
        return state

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, T, _ = x.shape
        dtype = x.dtype
        P = self.projection_dim

        if cache is not None:
            conv_state, ssm_state = cache
            lengths = cache.lengths
        else:
            conv_state = None
            ssm_state = None
            lengths = None

        speculative_width = (
            int(getattr(cache, "speculative_width", 0)) if cache is not None else 0
        )
        capture_speculative = speculative_width != 0
        if capture_speculative and (
            speculative_width != T
            or B != 1
            or T <= 1
            or T > 8
            or self.training
            or mask is not None
            or lengths is not None
            or conv_state is None
            or ssm_state is None
            or not mx.metal.is_available()
            or mx.default_device() != mx.gpu
        ):
            raise ValueError(
                "Kimi K3 speculative checkpoints require a populated, unpadded "
                "batch-one Metal decode with width in [2, 8]"
            )

        if (
            T == 1
            and not self.training
            and mask is None
            and lengths is None
            and cache is not None
            and mx.metal.is_available()
        ):
            if conv_state is None:
                conv_state = mx.zeros((B, self.conv_kernel - 1, 3 * P), dtype=dtype)
            if ssm_state is None:
                ssm_state = mx.zeros(
                    (B, self.num_heads, self.head_dim, self.head_dim),
                    dtype=mx.float32,
                )
            if self._step is None:
                self._step = mx.compile(self._decode_core)
            y, conv_state, ssm_state = self._step(x, conv_state, ssm_state)
            cache[0] = conv_state
            cache[1] = ssm_state
            cache.advance(1)
            return y

        if conv_state is None:
            conv_state = mx.zeros((B, self.conv_kernel - 1, 3 * P), dtype=dtype)

        projected_qkv = self.qkv_proj(x)
        if capture_speculative:
            qkv, conv_state, conv_state_history = self.qkv_conv(
                projected_qkv,
                conv_state,
                mask,
                lengths,
                return_state_history=True,
            )
        else:
            qkv, conv_state = self.qkv_conv(
                projected_qkv,
                conv_state,
                mask,
                lengths,
            )

        if cache is not None:
            cache[0] = conv_state

        q = qkv[..., :P].reshape(B, T, self.num_heads, self.head_dim)
        raw_k = qkv[..., P : 2 * P].reshape(
            B, T, self.num_heads, self.head_dim
        )
        v = qkv[..., 2 * P :].reshape(B, T, self.num_heads, self.head_dim)

        inv_scale = self.scale
        eps = 1e-6 / self.head_dim
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, eps)
        k = inv_scale * mx.fast.rms_norm(raw_k, None, eps)

        a_logits = self.f_b_proj(self.f_a_proj(x)).reshape(
            B, T, self.num_heads, self.head_dim
        )
        b_logits = self.b_proj(x).reshape(B, T, self.num_heads)

        replay_speculative = capture_speculative and replayssm_speculative_enabled()
        if replay_speculative:
            beta = mx.sigmoid(b_logits)
            if self.lower_bound is None:
                gk = compute_g(
                    self.A_log.reshape(self.num_heads, 1),
                    a_logits,
                    self.dt_bias.reshape(self.num_heads, self.head_dim),
                )
            else:
                gk = compute_g_safe(
                    self.A_log.reshape(self.num_heads, 1),
                    a_logits,
                    self.dt_bias.reshape(self.num_heads, self.head_dim),
                    self.lower_bound,
                )
            out, ssm_state = gated_delta_kernel(
                q,
                k,
                v,
                gk,
                beta,
                ssm_state,
                mask,
            )
        else:
            gated_delta_result = gated_delta_update(
                q,
                k,
                v,
                a_logits,
                b_logits,
                self.A_log.reshape(self.num_heads, 1),
                self.dt_bias.reshape(self.num_heads, self.head_dim),
                state=ssm_state,
                mask=mask,
                use_kernel=not self.training,
                lower_bound=self.lower_bound,
                return_state_history=capture_speculative,
            )
            if capture_speculative:
                out, ssm_state, ssm_state_history = gated_delta_result
            else:
                out, ssm_state = gated_delta_result

        if cache is not None:
            cache[1] = ssm_state
            cache.advance(T)
            if capture_speculative:
                ssm_history: Union[mx.array, SpeculativeReplayState]
                if replay_speculative:
                    ssm_history = SpeculativeReplayState(
                        width=T,
                        raw_inputs=(v, raw_k, gk, beta),
                        history_axis=1,
                        state_shape=tuple(ssm_state.shape),
                        state_dtype=ssm_state.dtype,
                        replay=self._replay_speculative_ssm,
                    )
                else:
                    ssm_history = ssm_state_history.transpose(2, 0, 1, 3, 4)
                cache.capture_speculative(
                    [
                        conv_state_history.transpose(1, 0, 2, 3),
                        ssm_history,
                    ]
                )

        if self.use_full_rank_gate:
            gate = self.g_proj(x)
        else:
            gate = self.g_b_proj(self.g_a_proj(x))
        gate = gate.reshape(B, T, self.num_heads, self.head_dim)
        out = (
            self.o_norm(out.reshape(B, T, self.num_heads, self.head_dim))
            * mx.sigmoid(gate)
        ).reshape(B, T, -1)
        return self.o_proj(out)


class KimiK3MLAAttention(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        if not args.mla_use_nope:
            raise ValueError("Only NoPE MLA is supported (mla_use_nope=True)")
        self.num_heads = args.num_attention_heads
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.scale = self.q_head_dim**-0.5
        self.use_gate = args.mla_use_output_gate

        hidden = args.hidden_size
        if self.q_lora_rank is not None:
            self.q_a_proj = nn.Linear(hidden, self.q_lora_rank, bias=False)
            self.q_a_layernorm = nn.RMSNorm(self.q_lora_rank, eps=1e-6)
            self.q_b_proj = nn.Linear(
                self.q_lora_rank, self.num_heads * self.q_head_dim, bias=False
            )
        else:
            self.q_proj = nn.Linear(
                hidden, self.num_heads * self.q_head_dim, bias=False
            )
        self.kv_a_proj_with_mqa = nn.Linear(
            hidden, self.kv_lora_rank + self.qk_rope_head_dim, bias=False
        )
        self.kv_a_layernorm = nn.RMSNorm(self.kv_lora_rank, eps=1e-6)
        self.embed_q = MultiLinear(
            self.qk_nope_head_dim, self.kv_lora_rank, self.num_heads
        )
        self.unembed_out = MultiLinear(
            self.kv_lora_rank, self.v_head_dim, self.num_heads
        )
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, hidden, bias=False)
        if self.use_gate:
            self.g_proj = nn.Linear(
                hidden, self.num_heads * self.v_head_dim, bias=False
            )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        if self.q_lora_rank is not None:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        else:
            q = self.q_proj(x)
        q = q.reshape(B, L, self.num_heads, self.q_head_dim).transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)

        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)
        k_pe = k_pe.reshape(B, L, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
        kv_latent = self.kv_a_layernorm(compressed_kv)

        kv_latent = mx.expand_dims(kv_latent, axis=1)

        if cache is not None:
            kv_latent, k_pe = cache.update_and_fetch(kv_latent, k_pe)

        pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
        if mask is not None:
            pe_scores = mx.where(
                mask,
                pe_scores,
                mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype),
            )

        if L == 1:
            q_nope = self.embed_q(q_nope)
            k = v = kv_latent
        else:
            k = self.embed_q(kv_latent, transpose=False)
            v = self.unembed_out(kv_latent)

        output = scaled_dot_product_attention(
            q_nope, k, v, cache=cache, scale=self.scale, mask=pe_scores
        )

        if L == 1:
            output = self.unembed_out(output)

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        if self.use_gate:
            output = output * mx.sigmoid(self.g_proj(x))
        return self.o_proj(output)


class KimiK3SparseMoE(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.args = args
        hidden = args.hidden_size
        experts = args.num_experts
        self.latent_size = args.routed_expert_hidden_size

        expert_dim = self.latent_size or hidden
        self.gate = nn.Linear(hidden, experts, bias=False)
        self.switch_mlp = SwitchGLU(
            expert_dim,
            args.moe_intermediate_size,
            experts,
            activation=SiTU(
                args.activation_situ_beta or 1.0, args.activation_situ_linear_beta
            ),
        )
        self.e_score_correction_bias = mx.zeros((experts,), dtype=mx.float32)

        if self.latent_size is not None:
            self.routed_expert_down_proj = nn.Linear(
                hidden, self.latent_size, bias=False
            )
            self.routed_expert_up_proj = nn.Linear(self.latent_size, hidden, bias=False)
            if args.latent_moe_use_norm:
                self.routed_expert_norm = nn.RMSNorm(
                    self.latent_size, eps=args.rms_norm_eps
                )
            else:
                self.routed_expert_norm = None
        else:
            self.routed_expert_norm = None

        if args.num_shared_experts:
            shared_hidden = args.moe_intermediate_size * args.num_shared_experts
            self.shared_experts = KimiK3MLP(args, intermediate_size=shared_hidden)
        else:
            self.shared_experts = None

        self.sharding_group = None

    def _call_with_optional_residual(
        self,
        x: mx.array,
        residual: Optional[mx.array],
    ) -> Tuple[mx.array, bool]:
        """Evaluate the MoE and optionally consume its decoder residual."""

        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)

        optimized_front = maybe_multibank_k3_moe_front(self, x)
        if optimized_front is None:
            optimized_front = maybe_authoritative_packed_k3_moe_front(self, x)
        if optimized_front is None:
            optimized_front = maybe_packed_k3_moe_front(self, x)
        if optimized_front is None:
            scores = self.gate(x)
            y = self.routed_expert_down_proj(x) if self.latent_size is not None else x
        else:
            shared_gate, shared_up, scores, y = optimized_front
        routed = maybe_fused_k3_router(
            scores,
            self.e_score_correction_bias,
            top_k=self.args.num_experts_per_token,
            n_group=self.args.num_expert_group,
            topk_group=self.args.topk_group,
            routed_scaling_factor=self.args.routed_scaling_factor,
            renormalize=self.args.moe_renormalize,
            training=getattr(self, "training", True),
        )
        if routed is None:
            inds, weights = _group_expert_select(
                scores,
                self.e_score_correction_bias,
                self.args.num_experts_per_token,
                self.args.num_expert_group,
                self.args.topk_group,
                self.args.routed_scaling_factor,
                self.args.moe_renormalize,
            )
        else:
            inds, weights = routed
        fused_reduced_y = maybe_fused_k3_switch_glu_reduce(
            self.switch_mlp,
            y,
            inds,
            weights,
        )
        if fused_reduced_y is None:
            fused_y = maybe_fused_k3_switch_glu(self.switch_mlp, y, inds)
            y = self.switch_mlp(y, inds) if fused_y is None else fused_y
            y = (y * weights[..., None]).sum(axis=-2)
        else:
            y = fused_reduced_y
        if self.shared_experts is None:
            shared = None
        elif optimized_front is None:
            shared = self.shared_experts(x)
        else:
            shared = self.shared_experts.down_proj(
                _situ(
                    shared_up,
                    shared_gate,
                    self.shared_experts.beta,
                    self.shared_experts.linear_beta,
                )
            )
        if self.sharding_group is not None:
            if shared is not None:
                split = y.shape[-1]
                combined = mx.distributed.all_sum(
                    mx.concatenate([y, shared], axis=-1), group=self.sharding_group
                )
                y, shared = mx.split(combined, [split], axis=-1)
            else:
                y = mx.distributed.all_sum(y, group=self.sharding_group)
        if self.routed_expert_norm is not None:
            y = self.routed_expert_norm(y)
        residual_consumed = False
        if self.latent_size is not None:
            fused_up_add = (
                maybe_fused_k3_routed_up_add(
                    self,
                    y,
                    shared,
                    residual,
                )
                if shared is not None and residual is not None
                else None
            )
            if fused_up_add is None:
                y = self.routed_expert_up_proj(y)
            else:
                y = fused_up_add
                shared = None
                residual_consumed = True
        if shared is not None:
            y = y + shared
        return y, residual_consumed

    def __call__(self, x: mx.array) -> mx.array:
        return self._call_with_optional_residual(x, None)[0]


class KimiK3DecoderLayer(nn.Module):
    def __init__(self, args: TextArgs, layer_idx: int):
        super().__init__()
        self.eps = args.rms_norm_eps
        kda_layers = args.linear_attn_config["kda_layers"]
        self.is_linear = (layer_idx + 1) in kda_layers

        if self.is_linear:
            self.self_attn = KimiK3DeltaAttention(args, layer_idx)
        else:
            self.self_attn = KimiK3MLAAttention(args)

        if (
            (args.num_experts or 0) > 0
            and layer_idx >= args.first_k_dense_replace
            and layer_idx % args.moe_layer_freq == 0
        ):
            self.mlp = KimiK3SparseMoE(args)
        else:
            self.mlp = KimiK3MLP(args)

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

        self.use_attn_res = args.attn_res_block_size is not None
        if self.use_attn_res:
            self.is_block_start = layer_idx % args.attn_res_block_size == 0
            self.self_attention_res_proj = nn.Linear(args.hidden_size, 1, bias=False)
            self.self_attention_res_norm = nn.RMSNorm(
                args.hidden_size, eps=args.rms_norm_eps
            )
            self.mlp_res_proj = nn.Linear(args.hidden_size, 1, bias=False)
            self.mlp_res_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            self._attn_res_w_eff = None
            self._mlp_res_w_eff = None

    def _ensure_attn_res_weights(self):
        if self.training or self._attn_res_w_eff is None:
            self._attn_res_w_eff = self.self_attention_res_norm.weight.astype(
                mx.float32
            ) * self.self_attention_res_proj.weight.reshape(-1)
            self._mlp_res_w_eff = self.mlp_res_norm.weight.astype(
                mx.float32
            ) * self.mlp_res_proj.weight.reshape(-1)

    def _mix_and_norm(
        self,
        blocks: ResidualBlocks,
        partial_sum: mx.array,
        w_eff: mx.array,
        norm: nn.RMSNorm,
    ) -> mx.array:
        if (
            not self.training
            and blocks.raw is not None
            and blocks.inv_rms is not None
        ):
            fused = maybe_fused_attnres_rms(
                blocks.raw,
                blocks.inv_rms,
                partial_sum,
                w_eff,
                norm.weight,
                self.eps,
            )
            if fused is not None:
                return fused
        return norm(
            _attn_res_mix(
                blocks,
                partial_sum,
                w_eff,
                self.eps,
                not self.training,
            )
        )

    def _prepare_attention(
        self,
        x: mx.array,
        blocks: Optional[ResidualBlocks],
    ) -> Tuple[mx.array, Optional[mx.array], Optional[ResidualBlocks]]:
        if not self.use_attn_res:
            return self.input_layernorm(x), x, blocks

        self._ensure_attn_res_weights()
        partial_sum = x
        attention_input = self._mix_and_norm(
            blocks,
            partial_sum,
            self._attn_res_w_eff,
            self.input_layernorm,
        )
        if self.is_block_start:
            blocks.append(partial_sum)
            partial_sum = None
        return attention_input, partial_sum, blocks

    def _finish_attention(
        self,
        partial_sum: Optional[mx.array],
        y: mx.array,
        blocks: Optional[ResidualBlocks],
    ) -> Tuple[mx.array, Optional[ResidualBlocks]]:
        if not self.use_attn_res:
            h = partial_sum + y
            mlp_input = self.post_attention_layernorm(h)
            if isinstance(self.mlp, KimiK3SparseMoE):
                mlp_output, residual_consumed = (
                    self.mlp._call_with_optional_residual(mlp_input, h)
                )
            else:
                mlp_output = self.mlp(mlp_input)
                residual_consumed = False
            return (
                mlp_output if residual_consumed else h + mlp_output
            ), blocks

        partial_sum = y if partial_sum is None else partial_sum + y
        mlp_input = self._mix_and_norm(
            blocks,
            partial_sum,
            self._mlp_res_w_eff,
            self.post_attention_layernorm,
        )
        if isinstance(self.mlp, KimiK3SparseMoE):
            mlp_output, residual_consumed = (
                self.mlp._call_with_optional_residual(mlp_input, partial_sum)
            )
        else:
            mlp_output = self.mlp(mlp_input)
            residual_consumed = False
        partial_sum = (
            mlp_output
            if residual_consumed
            else partial_sum + mlp_output
        )
        return partial_sum, blocks

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        blocks: Optional[ResidualBlocks] = None,
    ) -> Tuple[mx.array, Optional[ResidualBlocks]]:
        attention_input, partial_sum, blocks = self._prepare_attention(x, blocks)
        y = self.self_attn(attention_input, mask, cache)
        return self._finish_attention(partial_sum, y, blocks)


@dataclass(frozen=True)
class _CompiledDecodeTransition:
    step: Optional[Callable[..., Any]]
    kda_indices: Tuple[int, ...]


@dataclass(frozen=True)
class _CompiledDecodeSchedule:
    prefix: Optional[Callable[..., Any]]
    prefix_kda_indices: Tuple[int, ...]
    mla_indices: Tuple[int, ...]
    transitions: Tuple[_CompiledDecodeTransition, ...]
    tail: Optional[Callable[..., mx.array]]


def _decode_kda_group(
    h: mx.array,
    blocks: ResidualBlocks,
    layers: Tuple[KimiK3DecoderLayer, ...],
    states: Tuple[mx.array, ...],
) -> Tuple[mx.array, ResidualBlocks, Tuple[mx.array, ...]]:
    if len(states) != 2 * len(layers):
        raise ValueError("Each KDA layer requires convolution and recurrent state")

    updated_states = []
    for i, layer in enumerate(layers):
        attention_input, partial_sum, blocks = layer._prepare_attention(h, blocks)
        y, conv_state, ssm_state = layer.self_attn._decode_core(
            attention_input,
            states[2 * i],
            states[2 * i + 1],
        )
        h, blocks = layer._finish_attention(partial_sum, y, blocks)
        updated_states.extend((conv_state, ssm_state))
    return h, blocks, tuple(updated_states)


def _compile_decode_prefix(
    kda_layers: Tuple[KimiK3DecoderLayer, ...],
    next_mla_layer: KimiK3DecoderLayer,
    eps: float,
):
    def prefix(h, states):
        blocks = ResidualBlocks(eps)
        h, blocks, updated_states = _decode_kda_group(
            h, blocks, kda_layers, states
        )
        attention_input, partial_sum, blocks = next_mla_layer._prepare_attention(
            h, blocks
        )
        assert partial_sum is not None
        assert blocks.raw is not None and blocks.inv_rms is not None
        return (
            attention_input,
            partial_sum,
            blocks.raw,
            blocks.inv_rms,
            updated_states,
        )

    return mx.compile(prefix, shapeless=False)


def _compile_decode_transition(
    current_mla_layer: KimiK3DecoderLayer,
    kda_layers: Tuple[KimiK3DecoderLayer, ...],
    next_mla_layer: KimiK3DecoderLayer,
    eps: float,
):
    if not kda_layers:

        def adjacent_transition(mla_output, partial_sum, raw, inv_rms):
            blocks = ResidualBlocks(eps)
            blocks.raw = raw
            blocks.inv_rms = inv_rms
            h, blocks = current_mla_layer._finish_attention(
                partial_sum, mla_output, blocks
            )
            attention_input, partial_sum, blocks = (
                next_mla_layer._prepare_attention(h, blocks)
            )
            assert partial_sum is not None
            return attention_input, partial_sum, blocks.raw, blocks.inv_rms

        return mx.compile(adjacent_transition, shapeless=False)

    def transition(mla_output, partial_sum, raw, inv_rms, states):
        blocks = ResidualBlocks(eps)
        blocks.raw = raw
        blocks.inv_rms = inv_rms
        h, blocks = current_mla_layer._finish_attention(
            partial_sum, mla_output, blocks
        )
        h, blocks, updated_states = _decode_kda_group(
            h, blocks, kda_layers, states
        )
        attention_input, partial_sum, blocks = next_mla_layer._prepare_attention(
            h, blocks
        )
        assert partial_sum is not None
        return (
            attention_input,
            partial_sum,
            blocks.raw,
            blocks.inv_rms,
            updated_states,
        )

    return mx.compile(transition, shapeless=False)


def _compile_decode_tail(
    final_mla_layer: KimiK3DecoderLayer,
    output_res_w_eff: mx.array,
    norm: nn.RMSNorm,
    eps: float,
):
    def tail(mla_output, partial_sum, raw, inv_rms):
        blocks = ResidualBlocks(eps)
        blocks.raw = raw
        blocks.inv_rms = inv_rms
        h, blocks = final_mla_layer._finish_attention(
            partial_sum, mla_output, blocks
        )
        h = _attn_res_mix(
            blocks,
            h,
            output_res_w_eff,
            eps,
            True,
        )
        return norm(h)

    return mx.compile(tail, shapeless=False)


class KimiK3TextModel(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            KimiK3DecoderLayer(args, i) for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.use_attn_res = args.attn_res_block_size is not None
        if self.use_attn_res:
            self.output_attn_res_proj = nn.Linear(args.hidden_size, 1, bias=False)
            self.output_attn_res_norm = nn.RMSNorm(
                args.hidden_size, eps=args.rms_norm_eps
            )
            self._output_res_w_eff = None

        self.pipeline_rank = 0
        self.pipeline_size = 1
        self.start_idx = 0
        self.end_idx = len(self.layers)
        self.num_layers = len(self.layers)
        self.in_blocks = 0
        self._set_cache_indices()
        self._compiled_decode_enabled = (
            os.environ.get(COMPILED_DECODE_ENV, "0") == "1"
        )
        segment_count = (
            sum(not layer.is_linear for layer in self.layers) + 1
            if any(not layer.is_linear for layer in self.layers)
            else 0
        )
        self._compiled_decode_segments = (
            _parse_compiled_decode_segments(
                os.environ.get(COMPILED_DECODE_SEGMENTS_ENV, "all"),
                segment_count,
            )
            if self._compiled_decode_enabled
            else frozenset(range(segment_count))
        )
        self._compiled_decode_schedule = None
        self._async_decode_boundaries = _parse_async_decode_boundaries(
            os.environ.get(ASYNC_DECODE_BOUNDARIES_ENV, "none"),
            len(self.layers),
        )
        async_decode_state = os.environ.get(ASYNC_DECODE_STATE_ENV)
        if not self._async_decode_boundaries and async_decode_state is not None:
            raise ValueError(
                f"{ASYNC_DECODE_STATE_ENV} requires "
                f"{ASYNC_DECODE_BOUNDARIES_ENV}"
            )
        self._async_decode_state = async_decode_state or "residual"
        if self._async_decode_state not in {"hidden", "residual"}:
            raise ValueError(
                f"{ASYNC_DECODE_STATE_ENV} must be 'hidden' or 'residual'"
            )
        if self._compiled_decode_enabled and self._async_decode_boundaries:
            raise ValueError(
                f"{ASYNC_DECODE_BOUNDARIES_ENV} cannot be combined with "
                f"{COMPILED_DECODE_ENV}"
            )

    def _set_cache_indices(self, layers=None):
        if layers is None:
            layers = self.layers[self.start_idx : self.end_idx]
        self.ssm_idx = None
        self.attn_idx = None
        for i, layer in enumerate(layers):
            if layer.is_linear:
                if self.ssm_idx is None:
                    self.ssm_idx = i
            elif self.attn_idx is None:
                self.attn_idx = i
            if self.ssm_idx is not None and self.attn_idx is not None:
                break

    def _invalidate_compiled_decode(self):
        self._compiled_decode_schedule = None

    def _async_decode_boundary_eligible(
        self,
        h: mx.array,
        cache: List[Any],
        ssm_mask: Optional[mx.array],
        active_layers: List[KimiK3DecoderLayer],
    ) -> bool:
        if (
            not self._async_decode_boundaries
            or self._compiled_decode_enabled
            or self.training
            or not mx.metal.is_available()
            or mx.default_device() != mx.gpu
            or h.ndim != 3
            or h.shape[0] != 1
            or h.shape[1] != 1
            or ssm_mask is not None
            or self.pipeline_size != 1
            or self.start_idx != 0
            or self.end_idx != len(self.layers)
            or len(active_layers) != self.args.num_hidden_layers
            or len(cache) != len(active_layers)
        ):
            return False

        for layer, layer_cache in zip(active_layers, cache, strict=True):
            if layer_cache is None:
                return False
            if layer.is_linear:
                if (
                    getattr(layer_cache, "lengths", None) is not None
                    or layer_cache[0] is None
                    or layer_cache[1] is None
                ):
                    return False
            elif (
                getattr(layer_cache, "keys", None) is None
                or getattr(layer_cache, "values", None) is None
            ):
                return False
        return True

    def _submit_async_decode_boundary(
        self,
        h: mx.array,
        blocks: Optional[ResidualBlocks],
    ) -> None:
        if (
            self._async_decode_state == "hidden"
            or blocks is None
            or blocks.raw is None
        ):
            mx.async_eval(h)
        else:
            mx.async_eval(h, blocks.raw, blocks.inv_rms)

    def _compiled_decode_eligible(
        self,
        h: mx.array,
        cache: List[Any],
        ssm_mask: Optional[mx.array],
        active_layers: List[KimiK3DecoderLayer],
    ) -> bool:
        if (
            not self._compiled_decode_enabled
            or not self._compiled_decode_segments
            or self.training
            or fused_k3_experts_enabled()
            or not mx.metal.is_available()
            or mx.default_device() != mx.gpu
            or h.ndim != 3
            or h.shape[0] != 1
            or h.shape[1] != 1
            or ssm_mask is not None
            or self.pipeline_size != 1
            or self.start_idx != 0
            or self.end_idx != len(self.layers)
            or len(active_layers) != self.args.num_hidden_layers
            or len(cache) != len(active_layers)
            or not self.use_attn_res
            or self._output_res_w_eff is None
            or self.args.rms_norm_eps not in _attn_res_eps_cache
        ):
            return False

        if (
            not active_layers
            or not active_layers[0].is_linear
            or not active_layers[0].is_block_start
            or active_layers[-1].is_linear
        ):
            return False

        for layer, layer_cache in zip(active_layers, cache, strict=True):
            if (
                layer.training
                or not layer.use_attn_res
                or layer._attn_res_w_eff is None
                or layer._mlp_res_w_eff is None
            ):
                return False

            if layer.is_linear:
                attn = layer.self_attn
                if (
                    type(attn) is not KimiK3DeltaAttention
                    or type(layer_cache) is not ArraysCache
                    or len(layer_cache.cache) != 2
                    or layer_cache.lengths is not None
                    or layer_cache.left_padding is not None
                ):
                    return False
                conv_state, recurrent_state = layer_cache
                if (
                    conv_state is None
                    or recurrent_state is None
                    or conv_state.shape
                    != (1, attn.conv_kernel - 1, 3 * attn.projection_dim)
                    or recurrent_state.shape
                    != (1, attn.num_heads, attn.head_dim, attn.head_dim)
                    or conv_state.dtype != h.dtype
                    or recurrent_state.dtype != mx.float32
                ):
                    return False
            else:
                attn = layer.self_attn
                if (
                    type(attn) is not KimiK3MLAAttention
                    or layer.is_block_start
                    or type(layer_cache) not in (KVCache, BatchKVCache)
                    or layer_cache.keys is None
                    or layer_cache.values is None
                    or layer_cache.keys.ndim != 4
                    or layer_cache.values.ndim != 4
                    or layer_cache.keys.shape[0] != 1
                    or layer_cache.values.shape[0] != 1
                    or layer_cache.keys.shape[1] != 1
                    or layer_cache.values.shape[1] != 1
                    or layer_cache.keys.shape[2] != layer_cache.values.shape[2]
                    or layer_cache.keys.shape[3] != attn.kv_lora_rank
                    or layer_cache.values.shape[3] != attn.qk_rope_head_dim
                    or layer_cache.keys.dtype != h.dtype
                    or layer_cache.values.dtype != h.dtype
                ):
                    return False
                if type(layer_cache) is KVCache:
                    if (
                        layer_cache.offset <= 0
                        or layer_cache.offset > layer_cache.keys.shape[2]
                    ):
                        return False
                elif (
                    layer_cache._idx <= 0
                    or layer_cache._idx > layer_cache.keys.shape[2]
                    or layer_cache.offset.shape != (1,)
                    or layer_cache.left_padding.shape != (1,)
                    or layer_cache._right_padding is not None
                ):
                    return False

        return True

    def _build_compiled_decode_schedule(
        self,
        active_layers: List[KimiK3DecoderLayer],
    ) -> _CompiledDecodeSchedule:
        mla_indices = tuple(
            i for i, layer in enumerate(active_layers) if not layer.is_linear
        )
        first_mla = mla_indices[0]
        prefix_indices = tuple(range(first_mla))
        prefix = (
            _compile_decode_prefix(
                tuple(active_layers[i] for i in prefix_indices),
                active_layers[first_mla],
                self.args.rms_norm_eps,
            )
            if 0 in self._compiled_decode_segments
            else None
        )

        transitions = []
        for segment_idx, (current_mla, next_mla) in enumerate(
            zip(mla_indices, mla_indices[1:]),
            start=1,
        ):
            kda_indices = tuple(range(current_mla + 1, next_mla))
            transitions.append(
                _CompiledDecodeTransition(
                    step=(
                        _compile_decode_transition(
                            active_layers[current_mla],
                            tuple(active_layers[i] for i in kda_indices),
                            active_layers[next_mla],
                            self.args.rms_norm_eps,
                        )
                        if segment_idx in self._compiled_decode_segments
                        else None
                    ),
                    kda_indices=kda_indices,
                )
            )

        tail_segment_idx = len(mla_indices)
        tail = (
            _compile_decode_tail(
                active_layers[mla_indices[-1]],
                self._output_res_w_eff,
                self.norm,
                self.args.rms_norm_eps,
            )
            if tail_segment_idx in self._compiled_decode_segments
            else None
        )
        return _CompiledDecodeSchedule(
            prefix=prefix,
            prefix_kda_indices=prefix_indices,
            mla_indices=mla_indices,
            transitions=tuple(transitions),
            tail=tail,
        )

    @staticmethod
    def _read_kda_states(
        cache: List[Any],
        indices: Tuple[int, ...],
    ) -> Tuple[mx.array, ...]:
        states = []
        for idx in indices:
            states.extend((cache[idx][0], cache[idx][1]))
        return tuple(states)

    @staticmethod
    def _write_kda_states(
        cache: List[Any],
        indices: Tuple[int, ...],
        states: Tuple[mx.array, ...],
    ):
        if len(states) != 2 * len(indices):
            raise ValueError("Compiled KDA state output does not match its layer group")
        for i, idx in enumerate(indices):
            layer_cache = cache[idx]
            layer_cache[0] = states[2 * i]
            layer_cache[1] = states[2 * i + 1]
            layer_cache.advance(1)

    def _run_eager_decode_prefix(
        self,
        h: mx.array,
        cache: List[Any],
        active_layers: List[KimiK3DecoderLayer],
        schedule: _CompiledDecodeSchedule,
    ) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
        blocks = ResidualBlocks(self.args.rms_norm_eps)
        for idx in schedule.prefix_kda_indices:
            h, blocks = active_layers[idx](
                h,
                mask=None,
                cache=cache[idx],
                blocks=blocks,
            )
        attention_input, partial_sum, blocks = active_layers[
            schedule.mla_indices[0]
        ]._prepare_attention(h, blocks)
        assert partial_sum is not None
        assert blocks.raw is not None and blocks.inv_rms is not None
        return attention_input, partial_sum, blocks.raw, blocks.inv_rms

    def _run_eager_decode_transition(
        self,
        mla_output: mx.array,
        partial_sum: mx.array,
        raw: mx.array,
        inv_rms: mx.array,
        cache: List[Any],
        active_layers: List[KimiK3DecoderLayer],
        current_mla_idx: int,
        next_mla_idx: int,
        transition: _CompiledDecodeTransition,
    ) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
        blocks = ResidualBlocks(self.args.rms_norm_eps)
        blocks.raw = raw
        blocks.inv_rms = inv_rms
        h, blocks = active_layers[current_mla_idx]._finish_attention(
            partial_sum,
            mla_output,
            blocks,
        )
        for idx in transition.kda_indices:
            h, blocks = active_layers[idx](
                h,
                mask=None,
                cache=cache[idx],
                blocks=blocks,
            )
        attention_input, partial_sum, blocks = active_layers[
            next_mla_idx
        ]._prepare_attention(h, blocks)
        assert partial_sum is not None
        assert blocks.raw is not None and blocks.inv_rms is not None
        return attention_input, partial_sum, blocks.raw, blocks.inv_rms

    def _run_eager_decode_tail(
        self,
        mla_output: mx.array,
        partial_sum: mx.array,
        raw: mx.array,
        inv_rms: mx.array,
        final_mla_layer: KimiK3DecoderLayer,
    ) -> mx.array:
        blocks = ResidualBlocks(self.args.rms_norm_eps)
        blocks.raw = raw
        blocks.inv_rms = inv_rms
        h, blocks = final_mla_layer._finish_attention(
            partial_sum,
            mla_output,
            blocks,
        )
        h = _attn_res_mix(
            blocks,
            h,
            self._output_res_w_eff,
            self.args.rms_norm_eps,
            True,
        )
        return self.norm(h)

    def _run_compiled_decode(
        self,
        h: mx.array,
        cache: List[Any],
        attn_mask: Optional[mx.array],
        active_layers: List[KimiK3DecoderLayer],
    ) -> mx.array:
        schedule = self._compiled_decode_schedule
        if schedule is None:
            schedule = self._build_compiled_decode_schedule(active_layers)
            self._compiled_decode_schedule = schedule

        if schedule.prefix is None:
            attention_input, partial_sum, raw, inv_rms = self._run_eager_decode_prefix(
                h,
                cache,
                active_layers,
                schedule,
            )
        else:
            (
                attention_input,
                partial_sum,
                raw,
                inv_rms,
                updated_states,
            ) = schedule.prefix(
                h,
                self._read_kda_states(cache, schedule.prefix_kda_indices),
            )
            self._write_kda_states(
                cache,
                schedule.prefix_kda_indices,
                updated_states,
            )

        for i, mla_idx in enumerate(schedule.mla_indices):
            mla_output = active_layers[mla_idx].self_attn(
                attention_input,
                mask=attn_mask,
                cache=cache[mla_idx],
            )
            if i == len(schedule.transitions):
                if schedule.tail is None:
                    return self._run_eager_decode_tail(
                        mla_output,
                        partial_sum,
                        raw,
                        inv_rms,
                        active_layers[mla_idx],
                    )
                return schedule.tail(mla_output, partial_sum, raw, inv_rms)

            transition = schedule.transitions[i]
            if transition.step is None:
                attention_input, partial_sum, raw, inv_rms = (
                    self._run_eager_decode_transition(
                        mla_output,
                        partial_sum,
                        raw,
                        inv_rms,
                        cache,
                        active_layers,
                        mla_idx,
                        schedule.mla_indices[i + 1],
                        transition,
                    )
                )
            elif transition.kda_indices:
                (
                    attention_input,
                    partial_sum,
                    raw,
                    inv_rms,
                    updated_states,
                ) = transition.step(
                    mla_output,
                    partial_sum,
                    raw,
                    inv_rms,
                    self._read_kda_states(cache, transition.kda_indices),
                )
                self._write_kda_states(
                    cache,
                    transition.kda_indices,
                    updated_states,
                )
            else:
                attention_input, partial_sum, raw, inv_rms = transition.step(
                    mla_output,
                    partial_sum,
                    raw,
                    inv_rms,
                )

        raise RuntimeError("Compiled Kimi K3 decode schedule has no MLA tail")

    def pipeline(self, group):
        self._invalidate_compiled_decode()
        self.pipeline_rank = group.rank()
        self.pipeline_size = group.size()
        base, extra = divmod(len(self.layers), self.pipeline_size)
        seg = self.pipeline_size - self.pipeline_rank - 1
        self.start_idx = seg * base + min(seg, extra)
        self.end_idx = self.start_idx + base + (1 if seg < extra else 0)
        self.num_layers = self.end_idx - self.start_idx
        self.layers = self.layers[: self.end_idx]
        self.layers[: self.start_idx] = [None] * self.start_idx
        if self.use_attn_res:
            block_size = self.args.attn_res_block_size
            self.in_blocks = (self.start_idx + block_size - 1) // block_size
        self._set_cache_indices()

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]] = None,
        *,
        aux_hidden_state_layer_ids: Optional[Tuple[int, ...]] = None,
    ) -> Union[mx.array, Tuple[mx.array, Tuple[mx.array, ...]]]:
        capture_layer_ids = _validate_aux_hidden_state_layer_ids(
            aux_hidden_state_layer_ids,
            len(self.layers),
        )
        if capture_layer_ids and self.pipeline_size != 1:
            raise ValueError(
                "Kimi K3 auxiliary hidden capture requires pipeline_size == 1"
            )
        h = self.embed_tokens(inputs)
        boundary_dtype = h.dtype
        active_layers = self.layers[self.start_idx : self.end_idx]
        self._set_cache_indices(active_layers)
        if cache is None:
            cache = [None] * len(active_layers)

        ssm_mask = (
            create_ssm_mask(h, cache[self.ssm_idx])
            if self.ssm_idx is not None
            else None
        )
        if self.attn_idx is not None:
            attn_mask = create_attention_mask(
                h, cache[self.attn_idx], return_array=True
            )
        else:
            attn_mask = None

        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size

        blocks = ResidualBlocks(self.args.rms_norm_eps) if self.use_attn_res else None

        if not capture_layer_ids and self._compiled_decode_eligible(
            h, cache, ssm_mask, active_layers
        ):
            return self._run_compiled_decode(h, cache, attn_mask, active_layers)

        if pipeline_rank < pipeline_size - 1:
            src = pipeline_rank + 1
            if blocks is not None:
                packed = mx.distributed.recv(
                    (self.in_blocks + 1, *h.shape), h.dtype, src
                )
                h = packed[-1]
                blocks.raw = packed[:-1]
                xf = blocks.raw.astype(mx.float32)
                blocks.inv_rms = mx.rsqrt(
                    (xf * xf).mean(axis=-1) + self.args.rms_norm_eps
                )
            else:
                h = mx.distributed.recv_like(h, src)

        submit_async_boundaries = self._async_decode_boundary_eligible(
            h,
            cache,
            ssm_mask,
            active_layers,
        )
        aux_hidden_states = []
        for layer_idx, (layer, layer_cache) in enumerate(
            zip(active_layers, cache, strict=True),
            start=self.start_idx,
        ):
            mask = ssm_mask if layer.is_linear else attn_mask
            h, blocks = layer(h, mask=mask, cache=layer_cache, blocks=blocks)
            if layer_idx in capture_layer_ids:
                # DSpark/DFlash target ids use the Hugging Face convention:
                # capture the residual stream after target layer ``layer_idx``.
                aux_hidden_states.append(h)
            if (
                submit_async_boundaries
                and layer_idx in self._async_decode_boundaries
            ):
                self._submit_async_decode_boundary(h, blocks)

        if pipeline_rank != 0:
            dst = pipeline_rank - 1
            if blocks is not None:
                packed = mx.concatenate([blocks.raw, h[None]]).astype(boundary_dtype)
                packed = mx.distributed.send(packed, dst)
                h = packed[-1]
            else:
                h = mx.distributed.send(h.astype(boundary_dtype), dst)
            if cache[-1] is not None:
                if hasattr(cache[-1], "keys"):
                    cache[-1].keys = mx.depends(cache[-1].keys, h)
                else:
                    cache[-1][0] = mx.depends(cache[-1][0], h)
        elif blocks is not None:
            if self.training or self._output_res_w_eff is None:
                self._output_res_w_eff = self.output_attn_res_norm.weight.astype(
                    mx.float32
                ) * self.output_attn_res_proj.weight.reshape(-1)
            h = _attn_res_mix(
                blocks,
                h,
                self._output_res_w_eff,
                self.args.rms_norm_eps,
                not self.training,
            )

        if pipeline_size > 1:
            h = mx.distributed.all_gather(h.astype(boundary_dtype))[: h.shape[0]]

        h = self.norm(h)
        if capture_layer_ids:
            if len(aux_hidden_states) != len(capture_layer_ids):
                raise RuntimeError("Kimi K3 auxiliary hidden capture is incomplete")
            return h, tuple(aux_hidden_states)
        return h


@dataclass
class KimiK3SpeculativeCacheTransaction:
    """Fail-closed rollback state for one Kimi K3 target verification."""

    owner_id: int
    cache: List[Any]
    width: int
    array_states: List[Tuple[int, ArraysCache, List[Any]]]
    kv_states: List[Tuple[int, KVCache, Any, Any, int]]
    active: bool = True

    def release(self):
        """Release rollback-only references after commit or cancellation."""

        self.array_states = []
        self.kv_states = []


@dataclass(frozen=True)
class KimiK3TargetForward:
    """Target logits plus ordered post-layer hidden taps for a draft model."""

    logits: mx.array
    aux_hidden_states: Tuple[mx.array, ...]


class LanguageModel(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.args = args
        self.model = KimiK3TextModel(args)
        if args.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]] = None,
    ) -> mx.array:
        out = self.model(inputs, cache)
        if isinstance(out, tuple):
            raise RuntimeError("unexpected auxiliary target output")
        if self.lm_head is None:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    def forward_with_aux_hidden_states(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]],
        layer_ids: Tuple[int, ...],
    ) -> KimiK3TargetForward:
        result = self.model(
            inputs,
            cache,
            aux_hidden_state_layer_ids=layer_ids,
        )
        if not isinstance(result, tuple):
            raise RuntimeError("Kimi K3 target did not return auxiliary states")
        out, aux_hidden_states = result
        logits = (
            self.model.embed_tokens.as_linear(out)
            if self.lm_head is None
            else self.lm_head(out)
        )
        return KimiK3TargetForward(
            logits=logits,
            aux_hidden_states=aux_hidden_states,
        )

    @property
    def layers(self):
        return self.model.layers[self.model.start_idx : self.model.end_idx]

    def make_cache(self):
        caches: List[Any] = []
        for layer in self.layers:
            if layer.is_linear:
                caches.append(ArraysCache(size=2))
            else:
                caches.append(KVCache())
        return caches

    def begin_speculative_cache(
        self,
        cache: List[Any],
        width: int,
    ) -> KimiK3SpeculativeCacheTransaction:
        if self.model.pipeline_size != 1:
            raise ValueError(
                "Kimi K3 speculative cache requires pipeline_size == 1"
            )
        if any(isinstance(layer_cache, BatchKVCache) for layer_cache in cache):
            raise ValueError(
                "Kimi K3 speculative cache does not support BatchKVCache"
            )
        layers = list(self.layers)
        if len(cache) != len(layers):
            raise ValueError("Kimi K3 speculative cache does not match the model")
        if width <= 1 or width > 8:
            raise ValueError("Kimi K3 speculative width must be in [2, 8]")

        array_states: List[Tuple[int, ArraysCache, List[Any]]] = []
        kv_states: List[Tuple[int, KVCache, Any, Any, int]] = []
        kv_offsets = set()
        for index, (layer, layer_cache) in enumerate(
            zip(layers, cache, strict=True)
        ):
            if layer.is_linear:
                if not isinstance(layer_cache, ArraysCache):
                    raise ValueError(
                        f"Kimi K3 layer {index} requires an ArraysCache"
                    )
                layer_cache.validate_begin_speculative(width)
                array_states.append((index, layer_cache, list(layer_cache.cache)))
            else:
                if not isinstance(layer_cache, KVCache):
                    raise ValueError(f"Kimi K3 layer {index} requires a KVCache")
                if layer_cache.keys is None or layer_cache.values is None:
                    raise ValueError(
                        "Kimi K3 speculative cache requires populated MLA state"
                    )
                kv_offsets.add(int(layer_cache.offset))
                kv_states.append(
                    (
                        index,
                        layer_cache,
                        layer_cache.keys,
                        layer_cache.values,
                        int(layer_cache.offset),
                    )
                )
        if not array_states:
            raise ValueError("Kimi K3 speculative cache contains no KDA state")
        if len(kv_offsets) > 1:
            raise ValueError("Kimi K3 speculative MLA cache offsets disagree")

        try:
            for _, layer_cache, _ in array_states:
                layer_cache.begin_speculative(width)
        except BaseException:
            for index, layer_cache, states in array_states:
                cache[index] = layer_cache
                layer_cache.restore_speculative(states)
            raise

        return KimiK3SpeculativeCacheTransaction(
            owner_id=id(self),
            cache=cache,
            width=width,
            array_states=array_states,
            kv_states=kv_states,
        )

    def _validate_speculative_transaction(
        self,
        transaction: KimiK3SpeculativeCacheTransaction,
    ):
        if not isinstance(transaction, KimiK3SpeculativeCacheTransaction):
            raise TypeError("invalid Kimi K3 speculative cache transaction")
        if transaction.owner_id != id(self):
            raise ValueError("Kimi K3 speculative transaction belongs to another model")
        if not transaction.active:
            raise ValueError("Kimi K3 speculative transaction is no longer active")
        if len(transaction.cache) != len(self.layers):
            raise ValueError("Kimi K3 speculative cache changed during the transaction")
        for index, layer_cache, _ in transaction.array_states:
            if transaction.cache[index] is not layer_cache:
                raise ValueError(
                    "Kimi K3 speculative KDA cache was replaced during the transaction"
                )
        for index, layer_cache, _, _, _ in transaction.kv_states:
            if transaction.cache[index] is not layer_cache:
                raise ValueError(
                    "Kimi K3 speculative MLA cache was replaced during the transaction"
                )

    def resolve_speculative_cache(
        self,
        transaction: KimiK3SpeculativeCacheTransaction,
        consumed: int,
    ):
        try:
            self._validate_speculative_transaction(transaction)
            width = transaction.width
            if consumed < 1 or consumed > width:
                raise ValueError("invalid Kimi K3 speculative cache resolution")

            prepared = []
            for _, layer_cache, _ in transaction.array_states:
                if (
                    layer_cache.speculative_width != width
                    or not layer_cache.speculative_ready
                ):
                    raise ValueError(
                        "Kimi K3 speculative KDA checkpoints are incomplete"
                    )
                prepared.append(
                    (layer_cache, layer_cache.prepare_speculative(consumed))
                )

            for _, layer_cache, _, _, initial_offset in transaction.kv_states:
                if layer_cache.offset != initial_offset + width:
                    raise ValueError(
                        "Kimi K3 speculative MLA cache did not advance by its width"
                    )

            # Evaluate every cache-side output before changing any logical state.
            # This includes full acceptance and the MLA backing arrays, neither
            # of which is guaranteed to detach when only logits are evaluated.
            mx.eval(
                [state for _, states in prepared for state in states],
                [
                    layer_cache.state
                    for _, layer_cache, _, _, _ in transaction.kv_states
                ],
            )

            for layer_cache, states in prepared:
                layer_cache.validate_speculative_commit(states)
            for layer_cache, states in prepared:
                layer_cache.commit_speculative(states)
            for _, layer_cache, _, _, initial_offset in transaction.kv_states:
                layer_cache.offset = initial_offset + consumed
        except BaseException:
            if (
                isinstance(transaction, KimiK3SpeculativeCacheTransaction)
                and transaction.owner_id == id(self)
                and transaction.active
            ):
                self.cancel_speculative_cache(transaction)
            raise

        transaction.active = False
        transaction.release()

    def cancel_speculative_cache(
        self,
        transaction: KimiK3SpeculativeCacheTransaction,
    ):
        if not isinstance(transaction, KimiK3SpeculativeCacheTransaction):
            raise TypeError("invalid Kimi K3 speculative cache transaction")
        if transaction.owner_id != id(self):
            raise ValueError("Kimi K3 speculative transaction belongs to another model")
        if not transaction.active:
            return

        for index, layer_cache, states in transaction.array_states:
            transaction.cache[index] = layer_cache
            layer_cache.restore_speculative(states)
        for (
            index,
            layer_cache,
            keys,
            values,
            initial_offset,
        ) in transaction.kv_states:
            transaction.cache[index] = layer_cache
            layer_cache.keys = keys
            layer_cache.values = values
            layer_cache.offset = initial_offset
        transaction.active = False
        transaction.release()

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        args = self.args
        weights = {
            k: v for k, v in weights.items() if not k.startswith(("model.mtp", "mtp"))
        }
        layer_re = re.compile(r"model\.layers\.(\d+)\.")
        weights = {
            k: v
            for k, v in weights.items()
            if not (m := layer_re.match(k)) or int(m.group(1)) < args.num_hidden_layers
        }

        if args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        res_renames = []
        for src in ("self_attention_res", "mlp_res", "output_attn_res"):
            res_renames.append((f"{src}.proj_weight", f"{src}_proj.weight"))
            res_renames.append((f"{src}.norm_weight", f"{src}_norm.weight"))
        for k in list(weights):
            for pat, dst in res_renames:
                if k.endswith(pat):
                    weights[k[: -len(pat)] + dst] = weights.pop(k)
                    break

        for layer_idx, layer in enumerate(self.model.layers):
            lp = f"model.layers.{layer_idx}"

            if isinstance(layer.mlp, KimiK3SparseMoE):
                src_prefix = f"{lp}.block_sparse_moe"
                dst_prefix = f"{lp}.mlp"
                for src, dst in [
                    ("w1", "gate_proj"),
                    ("w2", "down_proj"),
                    ("w3", "up_proj"),
                ]:
                    if f"{src_prefix}.experts.0.{src}.weight_packed" in weights:
                        packed = mx.stack(
                            [
                                weights.pop(
                                    f"{src_prefix}.experts.{i}.{src}.weight_packed"
                                )
                                for i in range(args.num_experts)
                            ]
                        )
                        scales = mx.stack(
                            [
                                weights.pop(
                                    f"{src_prefix}.experts.{i}.{src}.weight_scale"
                                )
                                for i in range(args.num_experts)
                            ]
                        )
                        weights[f"{dst_prefix}.switch_mlp.{dst}.weight"] = packed.view(
                            mx.uint32
                        )
                        weights[f"{dst_prefix}.switch_mlp.{dst}.scales"] = scales
                    else:
                        for suffix in ("weight", "scales", "biases"):
                            if f"{src_prefix}.experts.0.{src}.{suffix}" in weights:
                                weights[f"{dst_prefix}.switch_mlp.{dst}.{suffix}"] = (
                                    mx.stack(
                                        [
                                            weights.pop(
                                                f"{src_prefix}.experts.{i}.{src}.{suffix}"
                                            )
                                            for i in range(args.num_experts)
                                        ]
                                    )
                                )

                for name in (
                    "shared_experts.gate_proj",
                    "shared_experts.up_proj",
                    "shared_experts.down_proj",
                    "routed_expert_down_proj",
                    "routed_expert_up_proj",
                    "routed_expert_norm",
                    "gate",
                    "switch_mlp.gate_proj",
                    "switch_mlp.up_proj",
                    "switch_mlp.down_proj",
                ):
                    for suffix in ("weight", "scales", "biases"):
                        src_key = f"{src_prefix}.{name}.{suffix}"
                        if src_key in weights:
                            weights[f"{dst_prefix}.{name}.{suffix}"] = weights.pop(
                                src_key
                            )

                for bias_key in (
                    f"{src_prefix}.gate.e_score_correction_bias",
                    f"{src_prefix}.e_score_correction_bias",
                ):
                    if bias_key in weights:
                        weights[f"{dst_prefix}.e_score_correction_bias"] = weights.pop(
                            bias_key
                        )

            attn = getattr(layer, "self_attn", None)
            ap = f"{lp}.self_attn"
            if isinstance(attn, KimiK3DeltaAttention):
                for src_name, dst_name in (
                    ("q_conv1d", "q_conv"),
                    ("k_conv1d", "k_conv"),
                    ("v_conv1d", "v_conv"),
                ):
                    src_key = f"{ap}.{src_name}.weight"
                    if src_key in weights:
                        w = weights.pop(src_key)
                        if w.ndim == 3:
                            w = w.moveaxis(2, 1)
                        weights[f"{ap}.{dst_name}.conv.weight"] = w
                for name in ("dt_bias", "A_log"):
                    key = f"{ap}.{name}"
                    if key in weights and weights[key].ndim > 1:
                        weights[key] = mx.reshape(weights[key], (-1,))
                a_log_key = f"{ap}.A_log"
                num_heads = args.linear_attn_config["num_heads"]
                if a_log_key in weights and weights[a_log_key].shape[0] > num_heads:
                    weights[a_log_key] = weights[a_log_key][:num_heads]

                if f"{ap}.qkv_proj.weight" not in weights:
                    for suffix in ("weight", "scales", "biases"):
                        parts = [f"{ap}.{p}_proj.{suffix}" for p in "qkv"]
                        if all(p in weights for p in parts):
                            weights[f"{ap}.qkv_proj.{suffix}"] = mx.concatenate(
                                [weights.pop(p) for p in parts], axis=0
                            )
                conv_parts = [f"{ap}.{p}_conv.conv.weight" for p in "qkv"]
                if f"{ap}.qkv_conv.conv.weight" not in weights and all(
                    p in weights for p in conv_parts
                ):
                    weights[f"{ap}.qkv_conv.conv.weight"] = mx.concatenate(
                        [weights.pop(p) for p in conv_parts], axis=0
                    )

            kv_b_key = f"{ap}.kv_b_proj.weight"
            if kv_b_key in weights:
                qk_nope = args.qk_nope_head_dim
                v_head = args.v_head_dim
                head_dim = qk_nope + v_head
                num_heads = args.num_attention_heads

                quantized = f"{ap}.kv_b_proj.scales" in weights
                v = weights.pop(kv_b_key)

                if quantized:
                    dims = args.kv_lora_rank
                    scales = weights.pop(f"{ap}.kv_b_proj.scales")
                    biases = weights.pop(f"{ap}.kv_b_proj.biases")
                    bits = (v.shape[-1] * 32) // dims
                    group_size = dims // scales.shape[-1]
                    v = mx.dequantize(
                        v, scales, biases, bits=bits, group_size=group_size
                    )

                v = v.reshape(num_heads, head_dim, -1)
                wk = mx.contiguous(v[:, :qk_nope, :].swapaxes(-1, -2))
                wv = mx.contiguous(v[:, qk_nope:, :])

                if quantized:
                    wk, wk_s, wk_b = mx.quantize(wk, bits=bits, group_size=group_size)
                    wv, wv_s, wv_b = mx.quantize(wv, bits=bits, group_size=group_size)
                    weights[f"{ap}.embed_q.scales"] = wk_s
                    weights[f"{ap}.embed_q.biases"] = wk_b
                    weights[f"{ap}.unembed_out.scales"] = wv_s
                    weights[f"{ap}.unembed_out.biases"] = wv_b

                weights[f"{ap}.embed_q.weight"] = wk
                weights[f"{ap}.unembed_out.weight"] = wv

        return weights

    @property
    def cast_predicate(self):
        def predicate(path: str):
            if "e_score_correction_bias" in path:
                return False
            if path.endswith("A_log") or path.endswith("dt_bias"):
                return False
            return True

        return predicate

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if path.endswith("mlp.gate"):
                return {"group_size": 64, "bits": 8}
            if path.endswith("res_proj"):
                return False
            return True

        return predicate


class VocabParallelHead(nn.Module):
    """Row-shard an untied LM head and reconstruct full-vocabulary logits.

    This preserves the standard model contract for sampling and logprobs
    while avoiding replicated projection work.  The vocabulary axis is moved
    to the front because MLX ``all_gather`` concatenates its leading axis.
    """

    def __init__(self, lm_head: nn.Module, group: mx.distributed.Group):
        super().__init__()
        self.group = group
        self.local_head = shard_linear(
            lm_head,
            "all-to-sharded",
            group=group,
        )

    def __call__(self, x: mx.array) -> mx.array:
        local_logits = self.local_head(x)
        vocab_first = mx.contiguous(mx.moveaxis(local_logits, -1, 0))
        full_vocab_first = mx.distributed.all_gather(
            vocab_first,
            group=self.group,
        )
        return mx.contiguous(mx.moveaxis(full_vocab_first, 0, -1))


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.language_model = LanguageModel(args.text_config)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]] = None,
    ) -> mx.array:
        return self.language_model(inputs, cache)

    def forward_with_aux_hidden_states(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]],
        layer_ids: Tuple[int, ...],
    ) -> KimiK3TargetForward:
        return self.language_model.forward_with_aux_hidden_states(
            inputs,
            cache,
            layer_ids,
        )

    @property
    def model(self):
        return self.language_model.model

    @property
    def layers(self):
        return self.language_model.layers

    def make_cache(self):
        return self.language_model.make_cache()

    def begin_speculative_cache(
        self,
        cache: List[Any],
        width: int,
    ) -> KimiK3SpeculativeCacheTransaction:
        return self.language_model.begin_speculative_cache(cache, width)

    def resolve_speculative_cache(
        self,
        transaction: KimiK3SpeculativeCacheTransaction,
        consumed: int,
    ):
        return self.language_model.resolve_speculative_cache(transaction, consumed)

    def cancel_speculative_cache(
        self,
        transaction: KimiK3SpeculativeCacheTransaction,
    ):
        return self.language_model.cancel_speculative_cache(transaction)

    def shard_vocab_head(
        self,
        group: Optional[mx.distributed.Group] = None,
    ) -> None:
        """Shard the loaded untied LM head while preserving full logits."""

        group = group or mx.distributed.init()
        lm_head = self.language_model.lm_head
        if (
            group.size() == 1
            or lm_head is None
            or isinstance(lm_head, VocabParallelHead)
        ):
            return
        self.language_model.lm_head = VocabParallelHead(lm_head, group)

    def shard(self, group: Optional[mx.distributed.Group] = None):
        group = group or mx.distributed.init()
        N = group.size()
        if N == 1:
            return
        self.model._invalidate_compiled_decode()
        rank = group.rank()

        for layer in self.layers:
            attn = layer.self_attn

            if layer.is_linear:
                invalidate_packed_k3_kda_wide(attn)
                invalidate_packed_k3_kda_skinny(attn)
                D = attn.head_dim
                P = attn.projection_dim
                num_heads = attn.num_heads // N
                sh = rank * num_heads
                eh = sh + num_heads

                attn.qkv_proj = shard_linear(
                    attn.qkv_proj, "all-to-sharded", segments=3, group=group
                )
                attn.f_b_proj = shard_linear(
                    attn.f_b_proj, "all-to-sharded", group=group
                )
                if attn.use_full_rank_gate:
                    attn.g_proj = shard_linear(
                        attn.g_proj, "all-to-sharded", group=group
                    )
                else:
                    attn.g_b_proj = shard_linear(
                        attn.g_b_proj, "all-to-sharded", group=group
                    )
                attn.b_proj = shard_linear(attn.b_proj, "all-to-sharded", group=group)
                attn.o_proj = shard_linear(attn.o_proj, "sharded-to-all", group=group)

                w = attn.qkv_conv.conv.weight
                attn.qkv_conv.conv.weight = mx.concatenate(
                    [w[seg * P + sh * D : seg * P + eh * D] for seg in range(3)],
                    axis=0,
                )
                attn.qkv_conv.conv.groups = 3 * num_heads * D

                attn.A_log = attn.A_log.reshape(-1)[sh:eh]
                attn.dt_bias = attn.dt_bias.reshape(-1)[sh * D : eh * D]
                attn.num_heads = num_heads
                attn.projection_dim = num_heads * D
            else:
                if attn.q_lora_rank is not None:
                    attn.q_b_proj = shard_linear(
                        attn.q_b_proj, "all-to-sharded", group=group
                    )
                else:
                    attn.q_proj = shard_linear(
                        attn.q_proj, "all-to-sharded", group=group
                    )
                if attn.use_gate:
                    attn.g_proj = shard_linear(
                        attn.g_proj, "all-to-sharded", group=group
                    )
                attn.o_proj = shard_linear(attn.o_proj, "sharded-to-all", group=group)

                attn.num_heads //= N
                num_heads = attn.num_heads
                sh = rank * num_heads
                eh = sh + num_heads

                def shard_heads(w):
                    return w[sh:eh]

                attn.embed_q.apply(shard_heads)
                attn.unembed_out.apply(shard_heads)

            if isinstance(layer.mlp, KimiK3SparseMoE):
                invalidate_packed_k3_moe_front(layer.mlp)
                layer.mlp.sharding_group = group
                shard_inplace(
                    layer.mlp.switch_mlp.gate_proj, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.mlp.switch_mlp.up_proj, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.mlp.switch_mlp.down_proj, "sharded-to-all", group=group
                )
                if layer.mlp.shared_experts is not None:
                    shard_inplace(
                        layer.mlp.shared_experts.gate_proj,
                        "all-to-sharded",
                        group=group,
                    )
                    shard_inplace(
                        layer.mlp.shared_experts.up_proj,
                        "all-to-sharded",
                        group=group,
                    )
                    shard_inplace(
                        layer.mlp.shared_experts.down_proj,
                        "sharded-to-all",
                        group=group,
                    )
            else:
                layer.mlp.gate_proj = shard_linear(
                    layer.mlp.gate_proj, "all-to-sharded", group=group
                )
                layer.mlp.up_proj = shard_linear(
                    layer.mlp.up_proj, "all-to-sharded", group=group
                )
                layer.mlp.down_proj = shard_linear(
                    layer.mlp.down_proj, "sharded-to-all", group=group
                )

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        prefix = "language_model."
        weights = {
            k[len(prefix) :] if k.startswith(prefix) else k: v
            for k, v in weights.items()
            if not k.startswith(
                (
                    "vision_tower",
                    "vision_model",
                    "multi_modal_projector",
                    "mm_projector",
                )
            )
        }
        weights = self.language_model.sanitize(weights)
        return {f"{prefix}{k}": v for k, v in weights.items()}

    @property
    def cast_predicate(self):
        return self.language_model.cast_predicate

    @property
    def quant_predicate(self):
        return self.language_model.quant_predicate
