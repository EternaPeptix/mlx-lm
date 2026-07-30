"""Exact Kimi K3 TP2 routed-up, shared-add, and residual-add fusion.

After Kimi K3's routed latent and shared branch are all-reduced, stock
MLX-LM launches an affine-8 routed-up QMV, materializes its BF16 output, then
adds the BF16 shared branch and decoder residual in two separate dispatches.
This default-off Metal path preserves the native QMV accumulation and output
rounding, then performs both ordered additions in the QMV output writer.

The adapter is intentionally decode-only and restricted to the released K3
TP2 geometry.  Unsupported platforms, shapes, dtypes, quantization, training,
and output-bias projections fail closed to the stock path.
"""

from __future__ import annotations

import os
from functools import lru_cache, partial
from typing import Any

import mlx.core as mx


FUSED_ROUTED_UP_ADD_ENV = "MLX_LM_KIMI_K3_FUSED_ROUTED_UP_ADD"

K3_ROUTED_LATENT_SIZE = 3584
K3_HIDDEN_SIZE = 7168
K3_GROUP_SIZE = 64
K3_BITS = 8


def _metal_available() -> bool:
    metal = getattr(mx, "metal", None)
    return metal is not None and bool(metal.is_available())


@lru_cache(maxsize=1)
def fused_routed_up_add_enabled() -> bool:
    return os.environ.get(FUSED_ROUTED_UP_ADD_ENV, "0") == "1"


_HEADER = r"""
template <typename T>
inline float k3_routed_up_load_x_8bit(
    const device T* x,
    thread float* x_thread) {
  float sum = 0.0f;
  for (int i = 0; i < 8; ++i) {
    // Match MLX quantized.h::load_vector<T, float, 8, 8>.
    sum += x[i];
    x_thread[i] = x[i];
  }
  return sum;
}

inline float k3_routed_up_qdot_8bit(
    const device uint8_t* weight,
    const thread float* x_thread,
    float scale,
    float bias,
    float sum) {
  float accum = 0.0f;
  for (int i = 0; i < 8; ++i) {
    // Match MLX quantized.h::qdot<float, 8, 8>.
    accum += x_thread[i] * weight[i];
  }
  return scale * accum + sum * bias;
}
"""


_SOURCE = r"""
constexpr uint VALUES_PER_THREAD = 8;
constexpr uint SIMD_SIZE = 32;
constexpr uint BLOCK_SIZE = VALUES_PER_THREAD * SIMD_SIZE;
constexpr uint GROUP_SIZE = 64;
constexpr uint RESULTS = 4;
constexpr uint SIMDS = 2;
constexpr uint OUTPUTS_PER_THREADGROUP = RESULTS * SIMDS;

const uint output_base =
    threadgroup_position_in_grid.y * OUTPUTS_PER_THREADGROUP +
    simdgroup_index_in_threadgroup * RESULTS;
const uint lane = thread_index_in_simdgroup;

const uint input_width = scales_shape[scales_ndim - 1] * GROUP_SIZE;
const uint output_width = scales_shape[scales_ndim - 2];
const uint scale_width = input_width / GROUP_SIZE;

const device T* x_ptr = routed_latent + lane * VALUES_PER_THREAD;
const device uint8_t* weight_ptr =
    reinterpret_cast<const device uint8_t*>(weight) +
    output_base * input_width +
    lane * VALUES_PER_THREAD;
const device T* scale_ptr =
    scales + output_base * scale_width + lane / 8;
const device T* bias_ptr =
    biases + output_base * scale_width + lane / 8;

thread float x_thread[VALUES_PER_THREAD];
thread float result[RESULTS] = {0.0f};

for (uint k = 0; k < input_width; k += BLOCK_SIZE) {
  float sum = k3_routed_up_load_x_8bit<T>(x_ptr, x_thread);
  for (uint row = 0; row < RESULTS; ++row) {
    result[row] += k3_routed_up_qdot_8bit(
        weight_ptr + row * input_width,
        x_thread,
        static_cast<float>(scale_ptr[row * scale_width]),
        static_cast<float>(bias_ptr[row * scale_width]),
        sum);
  }
  x_ptr += BLOCK_SIZE;
  weight_ptr += BLOCK_SIZE;
  scale_ptr += BLOCK_SIZE / GROUP_SIZE;
  bias_ptr += BLOCK_SIZE / GROUP_SIZE;
}

for (uint row = 0; row < RESULTS; ++row) {
  float value = simd_sum(result[row]);
  if (lane == 0) {
    const uint output_index = output_base + row;
    // Preserve the native affine-QMV FP32 -> BF16 boundary before the stock
    // BF16 `routed + shared`, then `residual + MoE` additions.
    T routed = static_cast<T>(value);
    T moe_output = static_cast<T>(routed + shared[output_index]);
    output[output_index] =
        static_cast<T>(residual[output_index] + moe_output);
  }
}
"""


@lru_cache(maxsize=1)
def _kernel():
    if not _metal_available():
        return None
    return mx.fast.metal_kernel(
        name="k3_fused_affine8_routed_up_shared_add",
        input_names=[
            "routed_latent",
            "shared",
            "residual",
            "weight",
            "scales",
            "biases",
        ],
        output_names=["output"],
        header=_HEADER,
        source=_SOURCE,
        ensure_row_contiguous=True,
    )


def _array_parameter(module: Any, name: str) -> mx.array | None:
    getter = getattr(module, "get", None)
    value = getter(name) if getter is not None else getattr(module, name, None)
    return value if isinstance(value, mx.array) else None


def _quantized_projection(module: Any):
    if (
        getattr(module, "bits", None) != K3_BITS
        or getattr(module, "group_size", None) != K3_GROUP_SIZE
        or getattr(module, "mode", None) != "affine"
        or _array_parameter(module, "bias") is not None
    ):
        return None
    weight = _array_parameter(module, "weight")
    scales = _array_parameter(module, "scales")
    biases = _array_parameter(module, "biases")
    if weight is None or scales is None or biases is None:
        return None
    return weight, scales, biases


def supports_fused_routed_up_add(
    routed_latent: mx.array,
    shared: mx.array,
    residual: mx.array,
    projection: tuple[mx.array, mx.array, mx.array],
) -> bool:
    """Return whether values satisfy the exact K3 TP2 decode contract."""

    weight, scales, biases = projection
    static_contract = (
        routed_latent.shape == (1, 1, K3_ROUTED_LATENT_SIZE)
        and shared.shape == (1, 1, K3_HIDDEN_SIZE)
        and residual.shape == shared.shape
        and routed_latent.dtype == mx.bfloat16
        and shared.dtype == mx.bfloat16
        and residual.dtype == mx.bfloat16
        and weight.shape
        == (
            K3_HIDDEN_SIZE,
            K3_ROUTED_LATENT_SIZE * K3_BITS // 32,
        )
        and scales.shape
        == (
            K3_HIDDEN_SIZE,
            K3_ROUTED_LATENT_SIZE // K3_GROUP_SIZE,
        )
        and biases.shape == scales.shape
        and weight.dtype == mx.uint32
        and scales.dtype == mx.bfloat16
        and biases.dtype == mx.bfloat16
    )
    return (
        static_contract
        and mx.default_device() == mx.gpu
        and _kernel() is not None
    )


def fused_routed_up_add(
    routed_latent: mx.array,
    shared: mx.array,
    residual: mx.array,
    projection: tuple[mx.array, mx.array, mx.array],
) -> mx.array:
    """Run routed-up QMV and both ordered BF16 adds in one dispatch."""

    if not supports_fused_routed_up_add(
        routed_latent,
        shared,
        residual,
        projection,
    ):
        raise ValueError("unsupported fused K3 routed-up/add contract")
    kernel = _kernel()
    assert kernel is not None
    weight, scales, biases = projection
    return kernel(
        inputs=[
            routed_latent,
            shared,
            residual,
            weight,
            scales,
            biases,
        ],
        template=[("T", routed_latent.dtype)],
        grid=(32, (K3_HIDDEN_SIZE // 8) * 2, 1),
        threadgroup=(32, 2, 1),
        output_shapes=[shared.shape],
        output_dtypes=[routed_latent.dtype],
        stream=mx.gpu,
    )[0]


@partial(mx.compile, shapeless=False)
def _compiled_fused_routed_up_add(
    routed_latent: mx.array,
    shared: mx.array,
    residual: mx.array,
    weight: mx.array,
    scales: mx.array,
    biases: mx.array,
) -> mx.array:
    return fused_routed_up_add(
        routed_latent,
        shared,
        residual,
        (weight, scales, biases),
    )


def maybe_fused_k3_routed_up_add(
    sparse_moe: Any,
    routed_latent: mx.array,
    shared: mx.array,
    residual: mx.array,
) -> mx.array | None:
    """Return an exact fused result, or ``None`` for the stock path."""

    if (
        not fused_routed_up_add_enabled()
        or getattr(sparse_moe, "training", True)
        or getattr(sparse_moe, "latent_size", None) != K3_ROUTED_LATENT_SIZE
    ):
        return None
    projection = _quantized_projection(
        getattr(sparse_moe, "routed_expert_up_proj", None)
    )
    if projection is None or not supports_fused_routed_up_add(
        routed_latent,
        shared,
        residual,
        projection,
    ):
        return None
    return _compiled_fused_routed_up_add(
        routed_latent,
        shared,
        residual,
        *projection,
    )
