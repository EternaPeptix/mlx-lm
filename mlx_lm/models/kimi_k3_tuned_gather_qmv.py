"""Decode-only affine-2bit gather-QMV tuning probe for Kimi K3.

The native MLX kernel assigns four output rows to each SIMD group.  This
prototype makes that tile size (and the number of SIMD groups per threadgroup)
explicit so K3's exact 3584x1536 expert geometry can be benchmarked without
changing weight bits or arithmetic order.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx


def _metal_available() -> bool:
    metal = getattr(mx, "metal", None)
    return metal is not None and bool(metal.is_available())


_HEADER = r"""
template <typename T>
inline float k3_tuned_load_x_2bit(
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
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 4.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 64.0f;
    }
  }
  return sum;
}

inline float k3_tuned_qdot_2bit(
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
inline float k3_tuned_derive_affine2_bias(T scale) {
  return -2.0f * static_cast<float>(scale);
}
"""


_SOURCE = r"""
constexpr int VALUES_PER_THREAD = 16;
constexpr int SIMD_SIZE = 32;
constexpr int BLOCK_SIZE = VALUES_PER_THREAD * SIMD_SIZE;
constexpr int GROUP_SIZE = 128;
constexpr int OUTPUTS_PER_THREADGROUP = RESULTS * SIMDS;

const uint expert_slot = threadgroup_position_in_grid.z;
const uint expert = indices[expert_slot];
const uint output_base =
    threadgroup_position_in_grid.y * OUTPUTS_PER_THREADGROUP +
    simdgroup_index_in_threadgroup * RESULTS;
const uint lane = thread_index_in_simdgroup;

const uint input_width = scales_shape[scales_ndim - 1] * GROUP_SIZE;
const uint output_width = scales_shape[scales_ndim - 2];
const uint packed_input_width = input_width / 4;
const uint scale_width = input_width / GROUP_SIZE;

const device T* x_ptr =
    x + (BROADCAST_X ? 0 : expert_slot * input_width) +
    lane * VALUES_PER_THREAD;
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

thread float x_thread[VALUES_PER_THREAD];
thread float result[RESULTS] = {0.0f};

// Register double-buffering: the next block's DRAM loads are issued before
// the current block's dot products, so they stay in flight during the FMAs.
// Loads are pure, so the arithmetic itself is unchanged.
uint weight_words[RESULTS];
T scale_values[RESULTS];
T bias_values[RESULTS];
for (int row = 0; row < RESULTS; ++row) {
  weight_words[row] = *reinterpret_cast<const device uint*>(
      weight_ptr + row * packed_input_width);
  scale_values[row] = scale_ptr[row * scale_width];
  if constexpr (!DERIVE_BIAS) {
    bias_values[row] = bias_ptr[row * scale_width];
  }
}

for (uint k = 0; k < input_width; k += BLOCK_SIZE) {
  const bool has_next = k + BLOCK_SIZE < input_width;
  uint next_words[RESULTS];
  T next_scales[RESULTS];
  T next_biases[RESULTS];
  if (has_next) {
    weight_ptr += BLOCK_SIZE / 4;
    scale_ptr += BLOCK_SIZE / GROUP_SIZE;
    bias_ptr += BLOCK_SIZE / GROUP_SIZE;
    for (int row = 0; row < RESULTS; ++row) {
      next_words[row] = *reinterpret_cast<const device uint*>(
          weight_ptr + row * packed_input_width);
      next_scales[row] = scale_ptr[row * scale_width];
      if constexpr (!DERIVE_BIAS) {
        next_biases[row] = bias_ptr[row * scale_width];
      }
    }
  }
  float sum = k3_tuned_load_x_2bit<T>(x_ptr, x_thread);
  for (int row = 0; row < RESULTS; ++row) {
    const float scale = static_cast<float>(scale_values[row]);
    const float bias =
        DERIVE_BIAS
            ? k3_tuned_derive_affine2_bias<T>(scale_values[row])
            : static_cast<float>(bias_values[row]);
    result[row] += k3_tuned_qdot_2bit(
        weight_words[row],
        x_thread,
        scale,
        bias,
        sum);
  }
  if (has_next) {
    for (int row = 0; row < RESULTS; ++row) {
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
// A vector simd_sum folds each component through the same tree as a scalar
// call, so the per-row associations are unchanged.
for (int row = 0; row + 4 <= RESULTS; row += 4) {
  const float4 sums = simd_sum(
      float4(result[row], result[row + 1], result[row + 2], result[row + 3]));
  result_sums[row] = sums.x;
  result_sums[row + 1] = sums.y;
  result_sums[row + 2] = sums.z;
  result_sums[row + 3] = sums.w;
}
if constexpr (RESULTS % 4 >= 2) {
  const int row = (RESULTS / 4) * 4;
  const float2 sums = simd_sum(float2(result[row], result[row + 1]));
  result_sums[row] = sums.x;
  result_sums[row + 1] = sums.y;
}
if constexpr (RESULTS % 2 == 1) {
  result_sums[RESULTS - 1] = simd_sum(result[RESULTS - 1]);
}
// Run the RESULTS output casts on parallel lanes instead of serially.
if (lane < RESULTS) {
  y[expert_slot * output_width + output_base + lane] =
      static_cast<T>(result_sums[lane]);
}
"""


@lru_cache(maxsize=None)
def _kernel():
    if not _metal_available():
        return None
    return mx.fast.metal_kernel(
        name="k3_tuned_affine2_gather_qmv",
        input_names=["x", "indices", "weight", "scales", "biases"],
        output_names=["y"],
        header=_HEADER,
        source=_SOURCE,
        ensure_row_contiguous=True,
    )


def supports_tuned_gather_qmv(
    x: mx.array,
    indices: mx.array,
    projection: tuple[mx.array, mx.array, mx.array],
    *,
    results_per_simdgroup: int,
    simdgroups: int,
    broadcast_x: bool,
) -> bool:
    if results_per_simdgroup not in (2, 4, 8, 16):
        return False
    if simdgroups not in (1, 2, 4):
        return False
    weight, scales, biases = projection
    if (
        weight.ndim != 3
        or scales.ndim != 3
        or biases.ndim != 3
        or weight.shape[:2] != scales.shape[:2]
        or scales.shape != biases.shape
        or indices.ndim != 3
        or indices.size == 0
        or indices.shape[-1] > weight.shape[0]
    ):
        return False
    input_width = scales.shape[-1] * 128
    output_width = scales.shape[-2]
    tile = results_per_simdgroup * simdgroups
    if (
        x.dtype != mx.bfloat16
        or indices.dtype != mx.uint32
        or weight.dtype != mx.uint32
        or scales.dtype != mx.bfloat16
        or biases.dtype != mx.bfloat16
        or weight.shape[-1] * 16 != input_width
        or input_width % 512 != 0
        or output_width % tile != 0
    ):
        return False
    x_rows = x.size // input_width
    return (
        x.size == x_rows * input_width
        and (
            (broadcast_x and x_rows == 1)
            or (not broadcast_x and x_rows == indices.size)
        )
        and _kernel() is not None
    )


def tuned_gather_qmv(
    x: mx.array,
    indices: mx.array,
    projection: tuple[mx.array, mx.array, mx.array],
    *,
    results_per_simdgroup: int,
    simdgroups: int,
    broadcast_x: bool,
    derive_bias: bool = False,
) -> mx.array:
    """Run an exact-arithmetic tiled QMV for one-token expert routing."""

    if not isinstance(derive_bias, bool):
        raise TypeError("derive_bias must be a bool")
    if not supports_tuned_gather_qmv(
        x,
        indices,
        projection,
        results_per_simdgroup=results_per_simdgroup,
        simdgroups=simdgroups,
        broadcast_x=broadcast_x,
    ):
        raise ValueError("unsupported tuned gather-QMV contract")
    _, scales, _ = projection
    output_width = scales.shape[-2]
    tile = results_per_simdgroup * simdgroups
    kernel = _kernel()
    assert kernel is not None
    weight, scales, biases = projection
    return kernel(
        inputs=[x, indices, weight, scales, biases],
        template=[
            ("T", x.dtype),
            ("RESULTS", results_per_simdgroup),
            ("SIMDS", simdgroups),
            ("BROADCAST_X", broadcast_x),
            ("DERIVE_BIAS", derive_bias),
        ],
        grid=(
            32,
            (output_width // tile) * simdgroups,
            indices.size,
        ),
        threadgroup=(32, simdgroups, 1),
        output_shapes=[(*indices.shape, 1, output_width)],
        output_dtypes=[x.dtype],
        stream=mx.gpu,
    )[0]
