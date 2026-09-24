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
  if constexpr (sizeof(T) == 2) {
    // Two 16-byte reads carry the same sixteen values the scalar walk loads;
    // callers always hand in freshly allocated, 16-byte-aligned activations.
    const device uint4* x_octets =
        reinterpret_cast<const device uint4*>(x);
    for (int i = 0; i < 16; i += 8) {
      const uint4 octet = x_octets[i / 8];
      const thread T values[8] = {
          as_type<T>(ushort(octet.x)),
          as_type<T>(ushort(octet.x >> 16)),
          as_type<T>(ushort(octet.y)),
          as_type<T>(ushort(octet.y >> 16)),
          as_type<T>(ushort(octet.z)),
          as_type<T>(ushort(octet.z >> 16)),
          as_type<T>(ushort(octet.w)),
          as_type<T>(ushort(octet.w >> 16)),
      };
      // Keep the same expression types and association as MLX's load_vector.
      sum += values[0] + values[1] + values[2] + values[3];
      x_thread[i] = values[0];
      x_thread[i + 1] = values[1] / 4.0f;
      x_thread[i + 2] = values[2] / 16.0f;
      x_thread[i + 3] = values[3] / 64.0f;
      sum += values[4] + values[5] + values[6] + values[7];
      x_thread[i + 4] = values[4];
      x_thread[i + 5] = values[5] / 4.0f;
      x_thread[i + 6] = values[6] / 16.0f;
      x_thread[i + 7] = values[7] / 64.0f;
    }
  } else {
    for (int i = 0; i < 16; i += 4) {
      // Keep the same expression types and association as MLX's load_vector.
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 4.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 64.0f;
    }
  }
  return sum;
}

inline float k3_down_reduce_qdot_2bit(
    const uint packed_word,
    const thread float* x_thread,
    float scale,
    float bias,
    float sum) {
  // The caller issues the one 32-bit read per row up front so a block's
  // loads can all be in flight; every weight pointer is four-byte aligned.
  float accum = 0.0f;
  for (int i = 0; i < 4; ++i) {
    const uint8_t packed = uint8_t((packed_word >> (8 * i)) & 0xffu);
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

  // Register double-buffering: the next block's DRAM loads are issued before
  // the current block's dot products, so they stay in flight during the FMAs.
  // Loads are pure, so the arithmetic itself is unchanged.
  uint weight_words[RESULTS];
  T scale_values[RESULTS];
  T bias_values[RESULTS];
  for (uint row = 0; row < RESULTS; ++row) {
    weight_words[row] = *reinterpret_cast<const device uint*>(
        weight_ptr + row * packed_input_width);
    scale_values[row] = scale_ptr[row * scale_width];
    if constexpr (!DERIVE_BIAS) {
      bias_values[row] = bias_ptr[row * scale_width];
    }
  }

  // K3's rank-local down input is 1536 wide, so this preserves the native
  // affine-2bit QMV's three FP32 accumulation passes of 512 values each.
  for (uint k = 0; k < input_width; k += BLOCK_SIZE) {
    const bool has_next = k + BLOCK_SIZE < input_width;
    uint next_words[RESULTS];
    T next_scales[RESULTS];
    T next_biases[RESULTS];
    if (has_next) {
      weight_ptr += BLOCK_SIZE / 4;
      scale_ptr += BLOCK_SIZE / GROUP_SIZE;
      bias_ptr += BLOCK_SIZE / GROUP_SIZE;
      for (uint row = 0; row < RESULTS; ++row) {
        next_words[row] = *reinterpret_cast<const device uint*>(
            weight_ptr + row * packed_input_width);
        next_scales[row] = scale_ptr[row * scale_width];
        if constexpr (!DERIVE_BIAS) {
          next_biases[row] = bias_ptr[row * scale_width];
        }
      }
    }
    float sum = k3_down_reduce_load_x_2bit<T>(x_ptr, x_thread);
    for (uint row = 0; row < RESULTS; ++row) {
      const float scale = static_cast<float>(scale_values[row]);
      const float bias =
          DERIVE_BIAS
              ? k3_down_reduce_derive_affine2_bias<T>(scale_values[row])
              : static_cast<float>(bias_values[row]);
      result[row] += k3_down_reduce_qdot_2bit(
          weight_words[row],
          x_thread,
          scale,
          bias,
          sum);
    }
    if (has_next) {
      for (uint row = 0; row < RESULTS; ++row) {
        weight_words[row] = next_words[row];
        scale_values[row] = next_scales[row];
        if constexpr (!DERIVE_BIAS) {
          bias_values[row] = next_biases[row];
        }
      }
    }
    x_ptr += BLOCK_SIZE;
  }

  thread float result_sums[RESULTS];
  // A vector simd_sum folds each component through the same tree as a
  // scalar call, so the per-row associations are unchanged.
  for (uint row = 0; row + 4 <= RESULTS; row += 4) {
    const float4 sums = simd_sum(float4(
        result[row], result[row + 1], result[row + 2], result[row + 3]));
    result_sums[row] = sums.x;
    result_sums[row + 1] = sums.y;
    result_sums[row + 2] = sums.z;
    result_sums[row + 3] = sums.w;
  }
  if constexpr (RESULTS % 4 >= 2) {
    const uint row = (RESULTS / 4) * 4;
    const float2 sums = simd_sum(float2(result[row], result[row + 1]));
    result_sums[row] = sums.x;
    result_sums[row + 1] = sums.y;
  }
  if constexpr (RESULTS % 2 == 1) {
    result_sums[RESULTS - 1] = simd_sum(result[RESULTS - 1]);
  }
  // Run the RESULTS output casts on parallel lanes instead of serially.
  if (lane < RESULTS) {
    // Preserve gather_qmm's FP32-accumulator-to-BF16 output boundary.
    expert_outputs[expert_slot * RESULTS + lane] =
        static_cast<T>(result_sums[lane]);
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
