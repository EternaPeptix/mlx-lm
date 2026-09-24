"""Exact Kimi K3 down-QMV, router weighting, and expert reduction.

This Metal prototype keeps the sixteen selected-expert down
projection rows in threadgroup memory.  It then reproduces MLX's BF16 router
multiply and fixed slot-order reduction before emitting the routed branch.
The stock ``[top_k, latent]`` down-projection result is therefore never
materialized.

The contract is intentionally limited to Kimi K3's TP2 geometry:

* sixteen selected experts;
* affine 2-bit weights with group size 128;
* BF16 activations, scales, biases, and router weights;
* a 1536-wide rank-local expert input; and
* a 3584-wide routed latent output; and
* one decode token or an exact width-two target-verification block.

Unsupported platforms, shapes, or dtypes must use the stock MLX-LM path.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx

K3_TOP_K = 16
K3_DOWN_INPUT_WIDTH = 1536
K3_DOWN_OUTPUT_WIDTH = 3584
K3_GROUP_SIZE = 128


def _metal_available() -> bool:
    metal = getattr(mx, "metal", None)
    return metal is not None and bool(metal.is_available())


_HEADER = r"""
template <typename T>
inline float k3_down_reduce_load_x_2bit(
    const device T* x,
    thread float* x_thread) {
  float sum = 0.0f;
  for (int i = 0; i < 16; i += 4) {
    // Keep the same expression types and association as MLX's load_vector.
    sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
    x_thread[i] = x[i];
    x_thread[i + 1] = x[i + 1] / 4.0f;
    x_thread[i + 2] = x[i + 2] / 16.0f;
    x_thread[i + 3] = x[i + 3] / 64.0f;
  }
  return sum;
}

inline float k3_down_reduce_qdot_2bit(
    const device uint8_t* w,
    const thread float* x_thread,
    float scale,
    float bias,
    float sum) {
  float accum = 0.0f;
  for (int i = 0; i < 4; ++i) {
    uint8_t packed = w[i];
    accum +=
        x_thread[4 * i] * (packed & 0x03) +
        x_thread[4 * i + 1] * (packed & 0x0c) +
        x_thread[4 * i + 2] * (packed & 0x30) +
        x_thread[4 * i + 3] * (packed & 0xc0);
  }
  return scale * accum + sum * bias;
}

template <typename T>
inline float k3_down_reduce_derive_affine2_bias(T scale) {
  return -2.0f * static_cast<float>(scale);
}
"""


_SOURCE = r"""
constexpr uint VALUES_PER_THREAD = 16;
constexpr uint SIMD_SIZE = 32;
constexpr uint BLOCK_SIZE = VALUES_PER_THREAD * SIMD_SIZE;
constexpr uint GROUP_SIZE = 128;
constexpr uint EXPERTS_PER_TOKEN = 16;

const uint tile = threadgroup_position_in_grid.x;
const uint token_index = threadgroup_position_in_grid.y;
const uint simd_slot = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const uint output_base = tile * RESULTS;

const uint input_width = scales_shape[scales_ndim - 1] * GROUP_SIZE;
const uint output_width = scales_shape[scales_ndim - 2];
const uint packed_input_width = input_width / 4;
const uint scale_width = input_width / GROUP_SIZE;
const device uint32_t* token_indices =
    indices + token_index * EXPERTS_PER_TOKEN;
const device T* token_router_weights =
    router_weights + token_index * EXPERTS_PER_TOKEN;
const device T* token_x =
    x + token_index * EXPERTS_PER_TOKEN * input_width;
device T* token_routed = routed + token_index * output_width;

thread float x_thread[VALUES_PER_THREAD];
thread float result[RESULTS];
threadgroup T expert_outputs[EXPERTS_PER_TOKEN * RESULTS];

// A SIMD group handles one or more expert slots. This keeps the complete
// top-16 exchange inside one threadgroup while allowing the launch occupancy
// to be tuned independently of K3's fixed route count.
for (uint expert_slot = simd_slot;
     expert_slot < EXPERTS_PER_TOKEN;
     expert_slot += SIMDS) {
  const uint expert = token_indices[expert_slot];
  const device T* x_ptr =
      token_x + expert_slot * input_width + lane * VALUES_PER_THREAD;
  const device uint8_t* weight_ptr =
      reinterpret_cast<const device uint8_t*>(weight) +
      (expert * output_width + output_base) * packed_input_width +
      lane * 4;
  const device T* scale_ptr =
      scales + (expert * output_width + output_base) * scale_width +
      lane / 8;
  const device T* bias_ptr =
      biases + (expert * output_width + output_base) * scale_width +
      lane / 8;

  for (uint row = 0; row < RESULTS; ++row) {
    result[row] = 0.0f;
  }

  // K3's rank-local down input is 1536 wide, so this preserves the native
  // affine-2bit QMV's three FP32 accumulation passes of 512 values each.
  for (uint k = 0; k < input_width; k += BLOCK_SIZE) {
    float sum = k3_down_reduce_load_x_2bit<T>(x_ptr, x_thread);
    for (uint row = 0; row < RESULTS; ++row) {
      T scale_value = scale_ptr[row * scale_width];
      float scale = static_cast<float>(scale_value);
      float bias;
      if constexpr (DERIVE_BIAS) {
        bias = k3_down_reduce_derive_affine2_bias<T>(scale_value);
      } else {
        bias = static_cast<float>(bias_ptr[row * scale_width]);
      }
      result[row] += k3_down_reduce_qdot_2bit(
          weight_ptr + row * packed_input_width,
          x_thread,
          scale,
          bias,
          sum);
    }
    x_ptr += BLOCK_SIZE;
    weight_ptr += BLOCK_SIZE / 4;
    scale_ptr += BLOCK_SIZE / GROUP_SIZE;
    bias_ptr += BLOCK_SIZE / GROUP_SIZE;
  }

  for (uint row = 0; row < RESULTS; ++row) {
    float value = simd_sum(result[row]);
    if (lane == 0) {
      // Preserve gather_qmm's FP32-accumulator-to-BF16 output boundary.
      expert_outputs[expert_slot * RESULTS + row] =
          static_cast<T>(value);
    }
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

if (simd_slot == 0 && lane < RESULTS) {
  // Match `(expert_outputs * weights[..., None]).sum(axis=-2)`:
  // route weights and products are BF16. MLX's small strided reduction uses
  // eight rows here: row r accumulates slots r and r + 8 from BF16 zero,
  // then row 0 folds rows 1 through 7 in order.
  T partials[8];
  for (uint row = 0; row < 8; ++row) {
    T partial = static_cast<T>(0.0f);
    for (uint slot = row; slot < EXPERTS_PER_TOKEN; slot += 8) {
      T route_weight = static_cast<T>(token_router_weights[slot]);
      T product = static_cast<T>(
          expert_outputs[slot * RESULTS + lane] * route_weight);
      partial = static_cast<T>(product + partial);
    }
    partials[row] = partial;
  }
  T total = partials[0];
  for (uint row = 1; row < 8; ++row) {
    total = static_cast<T>(partials[row] + total);
  }
  token_routed[output_base + lane] = total;
}
"""


@lru_cache(maxsize=None)
def _kernel():
    if not _metal_available():
        return None
    return mx.fast.metal_kernel(
        name="k3_affine2_down_route_reduce_decode",
        input_names=[
            "x",
            "indices",
            "router_weights",
            "weight",
            "scales",
            "biases",
        ],
        output_names=["routed"],
        header=_HEADER,
        source=_SOURCE,
        ensure_row_contiguous=True,
    )


def supports_fused_down_reduce_projection(
    indices: mx.array,
    router_weights: mx.array,
    projection: tuple[mx.array, mx.array, mx.array],
    *,
    results_per_threadgroup: int,
    simdgroups_per_threadgroup: int,
) -> bool:
    """Check the static K3 projection and routing contract before gate/up."""

    if results_per_threadgroup not in (2, 4, 8):
        return False
    if (
        simdgroups_per_threadgroup not in (4, 8, 16)
        or K3_TOP_K % simdgroups_per_threadgroup
    ):
        return False
    weight, scales, biases = projection
    if (
        indices.ndim != 3
        or indices.shape[0] != 1
        or indices.shape[1] not in (1, 2)
        or indices.shape[2] != K3_TOP_K
        or indices.dtype != mx.uint32
        or router_weights.shape != indices.shape
        or router_weights.dtype != mx.bfloat16
        or weight.ndim != 3
        or scales.ndim != 3
        or biases.ndim != 3
        or weight.shape[:2] != scales.shape[:2]
        or scales.shape != biases.shape
    ):
        return False
    input_width = scales.shape[-1] * K3_GROUP_SIZE
    output_width = scales.shape[-2]
    return (
        weight.shape[0] >= K3_TOP_K
        and input_width == K3_DOWN_INPUT_WIDTH
        and output_width == K3_DOWN_OUTPUT_WIDTH
        and output_width % results_per_threadgroup == 0
        and weight.dtype == mx.uint32
        and scales.dtype == mx.bfloat16
        and biases.dtype == mx.bfloat16
        and weight.shape[-1] * 16 == input_width
        and _kernel() is not None
    )


def supports_fused_down_reduce(
    activated: mx.array,
    indices: mx.array,
    router_weights: mx.array,
    projection: tuple[mx.array, mx.array, mx.array],
    *,
    results_per_threadgroup: int,
    simdgroups_per_threadgroup: int,
) -> bool:
    """Return whether all inputs satisfy the fail-closed decode contract."""

    if indices.ndim != 3:
        return False
    return (
        activated.dtype == mx.bfloat16
        and activated.shape
        == (
            1,
            indices.shape[1],
            K3_TOP_K,
            1,
            K3_DOWN_INPUT_WIDTH,
        )
        and supports_fused_down_reduce_projection(
            indices,
            router_weights,
            projection,
            results_per_threadgroup=results_per_threadgroup,
            simdgroups_per_threadgroup=simdgroups_per_threadgroup,
        )
    )


def fused_down_reduce_decode(
    activated: mx.array,
    indices: mx.array,
    router_weights: mx.array,
    projection: tuple[mx.array, mx.array, mx.array],
    *,
    results_per_threadgroup: int = 4,
    simdgroups_per_threadgroup: int = 8,
    derive_bias: bool = False,
) -> mx.array:
    """Project, BF16-route, and reduce K3's selected routed experts."""

    if not isinstance(derive_bias, bool):
        raise TypeError("derive_bias must be a bool")
    if not supports_fused_down_reduce(
        activated,
        indices,
        router_weights,
        projection,
        results_per_threadgroup=results_per_threadgroup,
        simdgroups_per_threadgroup=simdgroups_per_threadgroup,
    ):
        raise ValueError("unsupported fused K3 down/reduce inputs")
    kernel = _kernel()
    assert kernel is not None
    weight, scales, biases = projection
    return kernel(
        inputs=[
            activated,
            indices,
            router_weights,
            weight,
            scales,
            biases,
        ],
        template=[
            ("T", activated.dtype),
            ("RESULTS", results_per_threadgroup),
            ("SIMDS", simdgroups_per_threadgroup),
            ("DERIVE_BIAS", derive_bias),
        ],
        grid=(
            (K3_DOWN_OUTPUT_WIDTH // results_per_threadgroup)
            * 32
            * simdgroups_per_threadgroup,
            indices.shape[1],
            1,
        ),
        threadgroup=(32 * simdgroups_per_threadgroup, 1, 1),
        output_shapes=[(1, indices.shape[1], K3_DOWN_OUTPUT_WIDTH)],
        output_dtypes=[activated.dtype],
        stream=mx.gpu,
    )[0]
