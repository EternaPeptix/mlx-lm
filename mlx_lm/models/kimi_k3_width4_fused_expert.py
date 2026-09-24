"""Exact, production-geometry Kimi K3 width-four expert kernels.

The retained fused expert kernel maps every routed row independently.  That
mapping was screened previously at wider verification lengths and was not
promoted.  This module instead makes width four part of the Metal launch
geometry: each of four SIMD groups owns one verification token while the
threadgroup shares an output tile and expert-slot coordinate.

The path is deliberately narrower than the general SwitchGLU implementation.
It accepts only the released Kimi K3 TP2 geometry used by production:

* batch one, verification width four, top-k eight;
* 896 experts;
* BF16 activations and affine 2-bit/group-128 parameters;
* rank-local 3,584 -> 1,536 gate/up and 1,536 -> 3,584 down projections; and
* no parameter or output bias beyond the quantizer's affine metadata.

Unsupported inputs must remain on MLX-LM's stock path.  The public adapter has
an additional default-off environment selector.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx

from .kimi_k3_derived_bias import affine2_gather_core_available

K3_WIDTH4 = 4
K3_TOP_K = 8
K3_EXPERTS = 896
K3_HIDDEN = 3584
K3_INTERMEDIATE = 1536
K3_GROUP_SIZE = 128


def _metal_available() -> bool:
    metal = getattr(mx, "metal", None)
    return metal is not None and bool(metal.is_available())


_HEADER = r"""
template <typename T>
inline float k3_w4_load_x_2bit(
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
      // Preserve MLX gather-QMV's expression types and association.
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
      // Preserve MLX gather-QMV's expression types and association.
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 4.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 64.0f;
    }
  }
  return sum;
}

inline float k3_w4_qdot_2bit(
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

inline float k3_w4_sigmoid(float value) {
  float z = 1.0f / (1.0f + metal::exp(metal::abs(value)));
  return value < 0.0f ? z : 1.0f - z;
}

template <typename T>
inline float k3_w4_derive_affine2_bias(T scale) {
  return -2.0f * static_cast<float>(scale);
}
"""


_FRONT_SOURCE = r"""
constexpr uint WIDTH = 4;
constexpr uint TOP_K = 8;
constexpr uint VALUES_PER_THREAD = 16;
constexpr uint SIMD_SIZE = 32;
constexpr uint BLOCK_SIZE = VALUES_PER_THREAD * SIMD_SIZE;
constexpr uint GROUP_SIZE = 128;
constexpr float BETA = 4.0f;
constexpr float LINEAR_BETA = 25.0f;

// Width four is encoded in the threadgroup itself.  One SIMD group owns one
// token; z remains the score-ordered expert slot.  This is not the previously
// screened flattened routed-row mapping.
const uint token_index = simdgroup_index_in_threadgroup;
const uint expert_slot = threadgroup_position_in_grid.z;
const uint flattened_slot = token_index * TOP_K + expert_slot;
const uint expert = indices[flattened_slot];
const uint output_base = threadgroup_position_in_grid.y * RESULTS;
const uint lane = thread_index_in_simdgroup;

const uint input_width = up_scales_shape[up_scales_ndim - 1] * GROUP_SIZE;
const uint output_width = up_scales_shape[up_scales_ndim - 2];
const uint packed_input_width = input_width / 4;
const uint scale_width = input_width / GROUP_SIZE;

const device T* x_ptr =
    x + token_index * input_width + lane * VALUES_PER_THREAD;
const device uint8_t* up_ptr =
    reinterpret_cast<const device uint8_t*>(up_weight) +
    (expert * output_width + output_base) * packed_input_width + lane * 4;
const device uint8_t* gate_ptr =
    reinterpret_cast<const device uint8_t*>(gate_weight) +
    (expert * output_width + output_base) * packed_input_width + lane * 4;
const device T* up_scale_ptr =
    up_scales + (expert * output_width + output_base) * scale_width + lane / 8;
const device T* up_bias_ptr =
    up_biases + (expert * output_width + output_base) * scale_width + lane / 8;
const device T* gate_scale_ptr =
    gate_scales + (expert * output_width + output_base) * scale_width + lane / 8;
const device T* gate_bias_ptr =
    gate_biases + (expert * output_width + output_base) * scale_width + lane / 8;

thread float x_thread[VALUES_PER_THREAD];
thread float up_acc[RESULTS] = {0.0f};
thread float gate_acc[RESULTS] = {0.0f};

// Register double-buffering: the next block's DRAM loads are issued before
// the current block's dot products, so they stay in flight during the FMAs.
// Loads are pure, so the arithmetic itself is unchanged.
uint up_words[RESULTS];
uint gate_words[RESULTS];
T up_scale_values[RESULTS];
T gate_scale_values[RESULTS];
T up_bias_values[RESULTS];
T gate_bias_values[RESULTS];
for (uint row = 0; row < RESULTS; ++row) {
  up_words[row] = *reinterpret_cast<const device uint*>(
      up_ptr + row * packed_input_width);
  gate_words[row] = *reinterpret_cast<const device uint*>(
      gate_ptr + row * packed_input_width);
  up_scale_values[row] = up_scale_ptr[row * scale_width];
  gate_scale_values[row] = gate_scale_ptr[row * scale_width];
  if constexpr (!DERIVE_BIAS) {
    up_bias_values[row] = up_bias_ptr[row * scale_width];
    gate_bias_values[row] = gate_bias_ptr[row * scale_width];
  }
}

for (uint k = 0; k < input_width; k += BLOCK_SIZE) {
  const bool has_next = k + BLOCK_SIZE < input_width;
  uint next_up_words[RESULTS];
  uint next_gate_words[RESULTS];
  T next_up_scales[RESULTS];
  T next_gate_scales[RESULTS];
  T next_up_biases[RESULTS];
  T next_gate_biases[RESULTS];
  if (has_next) {
    up_ptr += BLOCK_SIZE / 4;
    gate_ptr += BLOCK_SIZE / 4;
    up_scale_ptr += BLOCK_SIZE / GROUP_SIZE;
    up_bias_ptr += BLOCK_SIZE / GROUP_SIZE;
    gate_scale_ptr += BLOCK_SIZE / GROUP_SIZE;
    gate_bias_ptr += BLOCK_SIZE / GROUP_SIZE;
    for (uint row = 0; row < RESULTS; ++row) {
      next_up_words[row] = *reinterpret_cast<const device uint*>(
          up_ptr + row * packed_input_width);
      next_gate_words[row] = *reinterpret_cast<const device uint*>(
          gate_ptr + row * packed_input_width);
      next_up_scales[row] = up_scale_ptr[row * scale_width];
      next_gate_scales[row] = gate_scale_ptr[row * scale_width];
      if constexpr (!DERIVE_BIAS) {
        next_up_biases[row] = up_bias_ptr[row * scale_width];
        next_gate_biases[row] = gate_bias_ptr[row * scale_width];
      }
    }
  }
  float sum = k3_w4_load_x_2bit<T>(x_ptr, x_thread);
  for (uint row = 0; row < RESULTS; ++row) {
    const float up_scale = static_cast<float>(up_scale_values[row]);
    const float gate_scale = static_cast<float>(gate_scale_values[row]);
    const float up_bias =
        DERIVE_BIAS
            ? k3_w4_derive_affine2_bias<T>(up_scale_values[row])
            : static_cast<float>(up_bias_values[row]);
    const float gate_bias =
        DERIVE_BIAS
            ? k3_w4_derive_affine2_bias<T>(gate_scale_values[row])
            : static_cast<float>(gate_bias_values[row]);
    up_acc[row] += k3_w4_qdot_2bit(
        up_words[row],
        x_thread,
        up_scale,
        up_bias,
        sum);
    gate_acc[row] += k3_w4_qdot_2bit(
        gate_words[row],
        x_thread,
        gate_scale,
        gate_bias,
        sum);
  }
  if (has_next) {
    for (uint row = 0; row < RESULTS; ++row) {
      up_words[row] = next_up_words[row];
      gate_words[row] = next_gate_words[row];
      up_scale_values[row] = next_up_scales[row];
      gate_scale_values[row] = next_gate_scales[row];
      if constexpr (!DERIVE_BIAS) {
        up_bias_values[row] = next_up_biases[row];
        gate_bias_values[row] = next_gate_biases[row];
      }
    }
  }
  x_ptr += BLOCK_SIZE;
}

thread float up_sums[RESULTS];
thread float gate_sums[RESULTS];
// A vector simd_sum folds each component through the same tree as a scalar
// call, so the per-row associations are unchanged.
for (uint row = 0; row + 4 <= RESULTS; row += 4) {
  const float4 up = simd_sum(
      float4(up_acc[row], up_acc[row + 1], up_acc[row + 2], up_acc[row + 3]));
  const float4 gate = simd_sum(float4(
      gate_acc[row], gate_acc[row + 1], gate_acc[row + 2], gate_acc[row + 3]));
  up_sums[row] = up.x;
  up_sums[row + 1] = up.y;
  up_sums[row + 2] = up.z;
  up_sums[row + 3] = up.w;
  gate_sums[row] = gate.x;
  gate_sums[row + 1] = gate.y;
  gate_sums[row + 2] = gate.z;
  gate_sums[row + 3] = gate.w;
}
if constexpr (RESULTS % 4 >= 2) {
  const uint row = (RESULTS / 4) * 4;
  const float2 up = simd_sum(float2(up_acc[row], up_acc[row + 1]));
  const float2 gate = simd_sum(float2(gate_acc[row], gate_acc[row + 1]));
  up_sums[row] = up.x;
  up_sums[row + 1] = up.y;
  gate_sums[row] = gate.x;
  gate_sums[row + 1] = gate.y;
}
if constexpr (RESULTS % 2 == 1) {
  up_sums[RESULTS - 1] = simd_sum(up_acc[RESULTS - 1]);
  gate_sums[RESULTS - 1] = simd_sum(gate_acc[RESULTS - 1]);
}
// Run the RESULTS SiTU epilogues on parallel lanes instead of serially.
if (lane < RESULTS) {
  const uint row = lane;
  // Preserve both gather-QMV BF16 boundaries before SiTU's FP32 arithmetic.
  T up_rounded = static_cast<T>(up_sums[row]);
  T gate_rounded = static_cast<T>(gate_sums[row]);
  float up_value = static_cast<float>(up_rounded);
  float gate_value = static_cast<float>(gate_rounded);
  float activation =
      BETA * metal::precise::tanh(gate_value / BETA) *
      k3_w4_sigmoid(gate_value);
  up_value = LINEAR_BETA * metal::precise::tanh(up_value / LINEAR_BETA);
  y[flattened_slot * output_width + output_base + row] =
      static_cast<T>(activation * up_value);
}
"""


_DOWN_SOURCE = r"""
constexpr uint WIDTH = 4;
constexpr uint TOP_K = 8;
constexpr uint VALUES_PER_THREAD = 16;
constexpr uint SIMD_SIZE = 32;
constexpr uint BLOCK_SIZE = VALUES_PER_THREAD * SIMD_SIZE;
constexpr uint GROUP_SIZE = 128;

const uint token_index = simdgroup_index_in_threadgroup;
const uint expert_slot = threadgroup_position_in_grid.z;
const uint flattened_slot = token_index * TOP_K + expert_slot;
const uint expert = indices[flattened_slot];
const uint output_base = threadgroup_position_in_grid.y * RESULTS;
const uint lane = thread_index_in_simdgroup;

const uint input_width = scales_shape[scales_ndim - 1] * GROUP_SIZE;
const uint output_width = scales_shape[scales_ndim - 2];
const uint packed_input_width = input_width / 4;
const uint scale_width = input_width / GROUP_SIZE;

const device T* x_ptr =
    x + flattened_slot * input_width + lane * VALUES_PER_THREAD;
const device uint8_t* weight_ptr =
    reinterpret_cast<const device uint8_t*>(weight) +
    (expert * output_width + output_base) * packed_input_width + lane * 4;
const device T* scale_ptr =
    scales + (expert * output_width + output_base) * scale_width + lane / 8;
const device T* bias_ptr =
    biases + (expert * output_width + output_base) * scale_width + lane / 8;

thread float x_thread[VALUES_PER_THREAD];
thread float accum[RESULTS] = {0.0f};

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
  float sum = k3_w4_load_x_2bit<T>(x_ptr, x_thread);
  for (uint row = 0; row < RESULTS; ++row) {
    const float scale = static_cast<float>(scale_values[row]);
    const float bias =
        DERIVE_BIAS
            ? k3_w4_derive_affine2_bias<T>(scale_values[row])
            : static_cast<float>(bias_values[row]);
    accum[row] += k3_w4_qdot_2bit(
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

thread float accum_sums[RESULTS];
// A vector simd_sum folds each component through the same tree as a scalar
// call, so the per-row associations are unchanged.
for (uint row = 0; row + 4 <= RESULTS; row += 4) {
  const float4 sums = simd_sum(
      float4(accum[row], accum[row + 1], accum[row + 2], accum[row + 3]));
  accum_sums[row] = sums.x;
  accum_sums[row + 1] = sums.y;
  accum_sums[row + 2] = sums.z;
  accum_sums[row + 3] = sums.w;
}
if constexpr (RESULTS % 4 >= 2) {
  const uint row = (RESULTS / 4) * 4;
  const float2 sums = simd_sum(float2(accum[row], accum[row + 1]));
  accum_sums[row] = sums.x;
  accum_sums[row + 1] = sums.y;
}
if constexpr (RESULTS % 2 == 1) {
  accum_sums[RESULTS - 1] = simd_sum(accum[RESULTS - 1]);
}
// Run the RESULTS output casts on parallel lanes instead of serially.
if (lane < RESULTS) {
  y[flattened_slot * output_width + output_base + lane] =
      static_cast<T>(accum_sums[lane]);
}
"""


_DOWN_REDUCE_SOURCE = r"""
constexpr uint TOP_K = 8;
constexpr uint VALUES_PER_THREAD = 16;
constexpr uint SIMD_SIZE = 32;
constexpr uint BLOCK_SIZE = VALUES_PER_THREAD * SIMD_SIZE;
constexpr uint GROUP_SIZE = 128;

const uint tile = threadgroup_position_in_grid.x;
const uint token_index = threadgroup_position_in_grid.y;
const uint simd_slot = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const uint output_base = tile * RESULTS;

const uint input_width = scales_shape[scales_ndim - 1] * GROUP_SIZE;
const uint output_width = scales_shape[scales_ndim - 2];
const uint packed_input_width = input_width / 4;
const uint scale_width = input_width / GROUP_SIZE;
const device uint32_t* token_indices = indices + token_index * TOP_K;
const device T* token_router_weights = router_weights + token_index * TOP_K;
const device T* token_x = x + token_index * TOP_K * input_width;
device T* token_routed = routed + token_index * output_width;

thread float x_thread[VALUES_PER_THREAD];
thread float result[RESULTS];
threadgroup T expert_outputs[TOP_K * RESULTS];

for (uint expert_slot = simd_slot; expert_slot < TOP_K; expert_slot += SIMDS) {
  const uint expert = token_indices[expert_slot];
  const device T* x_ptr =
      token_x + expert_slot * input_width + lane * VALUES_PER_THREAD;
  const device uint8_t* weight_ptr =
      reinterpret_cast<const device uint8_t*>(weight) +
      (expert * output_width + output_base) * packed_input_width + lane * 4;
  const device T* scale_ptr =
      scales + (expert * output_width + output_base) * scale_width + lane / 8;
  const device T* bias_ptr =
      biases + (expert * output_width + output_base) * scale_width + lane / 8;

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
    float sum = k3_w4_load_x_2bit<T>(x_ptr, x_thread);
    for (uint row = 0; row < RESULTS; ++row) {
      const float scale = static_cast<float>(scale_values[row]);
      const float bias =
          DERIVE_BIAS
              ? k3_w4_derive_affine2_bias<T>(scale_values[row])
              : static_cast<float>(bias_values[row]);
      result[row] += k3_w4_qdot_2bit(
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
    expert_outputs[expert_slot * RESULTS + lane] =
        static_cast<T>(result_sums[lane]);
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

if (simd_slot == 0 && lane < RESULTS) {
  // Match MLX's small BF16 reduction exactly.  At top-8 each strided row owns
  // one product, then row zero folds rows one through seven in order.
  T partials[8];
  for (uint row = 0; row < 8; ++row) {
    T route_weight = static_cast<T>(token_router_weights[row]);
    partials[row] = static_cast<T>(
        expert_outputs[row * RESULTS + lane] * route_weight);
  }
  T total = partials[0];
  for (uint row = 1; row < 8; ++row) {
    total = static_cast<T>(partials[row] + total);
  }
  token_routed[output_base + lane] = total;
}
"""


@lru_cache(maxsize=None)
def _front_kernel():
    if not _metal_available():
        return None
    return mx.fast.metal_kernel(
        name="k3_width4_top8_affine2_switch_situ",
        input_names=[
            "x",
            "indices",
            "up_weight",
            "up_scales",
            "up_biases",
            "gate_weight",
            "gate_scales",
            "gate_biases",
        ],
        output_names=["y"],
        header=_HEADER,
        source=_FRONT_SOURCE,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=None)
def _down_kernel():
    if not _metal_available():
        return None
    return mx.fast.metal_kernel(
        name="k3_width4_top8_affine2_down",
        input_names=["x", "indices", "weight", "scales", "biases"],
        output_names=["y"],
        header=_HEADER,
        source=_DOWN_SOURCE,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=None)
def _down_reduce_kernel():
    if not _metal_available():
        return None
    return mx.fast.metal_kernel(
        name="k3_width4_top8_affine2_down_route_reduce",
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
        source=_DOWN_REDUCE_SOURCE,
        ensure_row_contiguous=True,
    )


def _projection_has_geometry(
    projection: tuple[mx.array, mx.array, mx.array],
    *,
    input_width: int,
    output_width: int,
) -> bool:
    if len(projection) != 3:
        return False
    weight, scales, biases = projection
    return (
        weight.ndim == 3
        and scales.ndim == 3
        and biases.ndim == 3
        and weight.shape
        == (K3_EXPERTS, output_width, input_width // 16)
        and scales.shape
        == (K3_EXPERTS, output_width, input_width // K3_GROUP_SIZE)
        and biases.shape == scales.shape
        and weight.dtype == mx.uint32
        and scales.dtype == mx.bfloat16
        and biases.dtype == mx.bfloat16
    )


def supports_width4_switch_situ(
    x: mx.array,
    indices: mx.array,
    up: tuple[mx.array, mx.array, mx.array],
    gate: tuple[mx.array, mx.array, mx.array],
) -> bool:
    """Require the exact released-K3 TP2 gate/up geometry."""

    return (
        x.shape == (1, K3_WIDTH4, K3_HIDDEN)
        and x.dtype == mx.bfloat16
        and indices.shape == (1, K3_WIDTH4, K3_TOP_K)
        and indices.dtype == mx.uint32
        and _projection_has_geometry(
            up,
            input_width=K3_HIDDEN,
            output_width=K3_INTERMEDIATE,
        )
        and _projection_has_geometry(
            gate,
            input_width=K3_HIDDEN,
            output_width=K3_INTERMEDIATE,
        )
        and _front_kernel() is not None
    )


def supports_width4_down(
    activated: mx.array,
    indices: mx.array,
    down: tuple[mx.array, mx.array, mx.array],
) -> bool:
    """Require the exact released-K3 TP2 down-projection geometry."""

    return (
        activated.shape
        == (1, K3_WIDTH4, K3_TOP_K, 1, K3_INTERMEDIATE)
        and activated.dtype == mx.bfloat16
        and indices.shape == (1, K3_WIDTH4, K3_TOP_K)
        and indices.dtype == mx.uint32
        and _projection_has_geometry(
            down,
            input_width=K3_INTERMEDIATE,
            output_width=K3_HIDDEN,
        )
        and _down_kernel() is not None
    )


def supports_width4_down_projection(
    indices: mx.array,
    down: tuple[mx.array, mx.array, mx.array],
) -> bool:
    """Validate the static down bank before materializing the front result."""

    return (
        indices.shape == (1, K3_WIDTH4, K3_TOP_K)
        and indices.dtype == mx.uint32
        and _projection_has_geometry(
            down,
            input_width=K3_INTERMEDIATE,
            output_width=K3_HIDDEN,
        )
        and _down_kernel() is not None
    )


def supports_width4_native_down_projection(
    indices: mx.array,
    down: tuple[mx.array, mx.array, mx.array],
) -> bool:
    """Validate the exact down bank for MLX's affine2 gather core."""

    return (
        indices.shape == (1, K3_WIDTH4, K3_TOP_K)
        and indices.dtype == mx.uint32
        and _projection_has_geometry(
            down,
            input_width=K3_INTERMEDIATE,
            output_width=K3_HIDDEN,
        )
        and affine2_gather_core_available()
    )


def supports_width4_down_reduce(
    activated: mx.array,
    indices: mx.array,
    router_weights: mx.array,
    down: tuple[mx.array, mx.array, mx.array],
    *,
    results_per_threadgroup: int,
    simdgroups_per_threadgroup: int,
) -> bool:
    """Validate the exact top-8 width-four fused down/reduce contract."""

    return (
        activated.shape
        == (1, K3_WIDTH4, K3_TOP_K, 1, K3_INTERMEDIATE)
        and activated.dtype == mx.bfloat16
        and supports_width4_down_reduce_projection(
            indices,
            router_weights,
            down,
            results_per_threadgroup=results_per_threadgroup,
            simdgroups_per_threadgroup=simdgroups_per_threadgroup,
        )
    )


def supports_width4_down_reduce_projection(
    indices: mx.array,
    router_weights: mx.array,
    down: tuple[mx.array, mx.array, mx.array],
    *,
    results_per_threadgroup: int,
    simdgroups_per_threadgroup: int,
) -> bool:
    """Validate width-four routes and down bank before running gate/up.

    The model adapter uses this projection-level predicate before dispatching
    the fused front.  Unsupported routing geometry therefore falls back to
    the stock MLX-LM graph without constructing an intermediate activation or
    allowing ``width4_down_reduce`` to raise.
    """

    return (
        results_per_threadgroup in (2, 4, 8, 16)
        and simdgroups_per_threadgroup in (4, 8)
        and K3_TOP_K % simdgroups_per_threadgroup == 0
        and indices.shape == (1, K3_WIDTH4, K3_TOP_K)
        and indices.dtype == mx.uint32
        and router_weights.shape == indices.shape
        and router_weights.dtype == mx.bfloat16
        and _projection_has_geometry(
            down,
            input_width=K3_INTERMEDIATE,
            output_width=K3_HIDDEN,
        )
        and _down_reduce_kernel() is not None
    )


def width4_switch_situ(
    x: mx.array,
    indices: mx.array,
    up: tuple[mx.array, mx.array, mx.array],
    gate: tuple[mx.array, mx.array, mx.array],
    *,
    results_per_simdgroup: int = 4,
    derive_bias: bool = False,
) -> mx.array:
    """Compute width-four gate/up/SiTU with one token per SIMD group."""

    if not supports_width4_switch_situ(x, indices, up, gate):
        raise ValueError("unsupported Kimi K3 width-four gate/up geometry")
    if results_per_simdgroup not in (2, 4, 8, 16):
        raise ValueError("results_per_simdgroup must be 2, 4, 8, or 16")
    if not isinstance(derive_bias, bool):
        raise TypeError("derive_bias must be a bool")
    kernel = _front_kernel()
    assert kernel is not None
    return kernel(
        inputs=[x, indices, *up, *gate],
        template=[
            ("T", x.dtype),
            ("RESULTS", results_per_simdgroup),
            ("DERIVE_BIAS", derive_bias),
        ],
        grid=(
            32,
            (K3_INTERMEDIATE // results_per_simdgroup) * K3_WIDTH4,
            K3_TOP_K,
        ),
        threadgroup=(32, K3_WIDTH4, 1),
        output_shapes=[(1, K3_WIDTH4, K3_TOP_K, 1, K3_INTERMEDIATE)],
        output_dtypes=[x.dtype],
        stream=mx.gpu,
    )[0]


def width4_down(
    activated: mx.array,
    indices: mx.array,
    down: tuple[mx.array, mx.array, mx.array],
    *,
    results_per_simdgroup: int = 8,
    derive_bias: bool = False,
) -> mx.array:
    """Compute the exact width-four routed down projection."""

    if not supports_width4_down(activated, indices, down):
        raise ValueError("unsupported Kimi K3 width-four down geometry")
    if results_per_simdgroup not in (2, 4, 8, 16):
        raise ValueError("results_per_simdgroup must be 2, 4, 8, or 16")
    if not isinstance(derive_bias, bool):
        raise TypeError("derive_bias must be a bool")
    kernel = _down_kernel()
    assert kernel is not None
    return kernel(
        inputs=[activated, indices, *down],
        template=[
            ("T", activated.dtype),
            ("RESULTS", results_per_simdgroup),
            ("DERIVE_BIAS", derive_bias),
        ],
        grid=(
            32,
            (K3_HIDDEN // results_per_simdgroup) * K3_WIDTH4,
            K3_TOP_K,
        ),
        threadgroup=(32, K3_WIDTH4, 1),
        output_shapes=[(1, K3_WIDTH4, K3_TOP_K, 1, K3_HIDDEN)],
        output_dtypes=[activated.dtype],
        stream=mx.gpu,
    )[0]


def width4_down_reduce(
    activated: mx.array,
    indices: mx.array,
    router_weights: mx.array,
    down: tuple[mx.array, mx.array, mx.array],
    *,
    results_per_threadgroup: int = 4,
    simdgroups_per_threadgroup: int = 8,
    derive_bias: bool = False,
) -> mx.array:
    """Fuse the width-four down projection, router multiply, and reduction."""

    if not isinstance(derive_bias, bool):
        raise TypeError("derive_bias must be a bool")
    if not supports_width4_down_reduce(
        activated,
        indices,
        router_weights,
        down,
        results_per_threadgroup=results_per_threadgroup,
        simdgroups_per_threadgroup=simdgroups_per_threadgroup,
    ):
        raise ValueError("unsupported Kimi K3 width-four down/reduce geometry")
    kernel = _down_reduce_kernel()
    assert kernel is not None
    return kernel(
        inputs=[activated, indices, router_weights, *down],
        template=[
            ("T", activated.dtype),
            ("RESULTS", results_per_threadgroup),
            ("SIMDS", simdgroups_per_threadgroup),
            ("DERIVE_BIAS", derive_bias),
        ],
        grid=(
            (K3_HIDDEN // results_per_threadgroup)
            * 32
            * simdgroups_per_threadgroup,
            K3_WIDTH4,
            1,
        ),
        threadgroup=(32 * simdgroups_per_threadgroup, 1, 1),
        output_shapes=[(1, K3_WIDTH4, K3_HIDDEN)],
        output_dtypes=[activated.dtype],
        stream=mx.gpu,
    )[0]


def width4_switch_glu(
    x: mx.array,
    indices: mx.array,
    up: tuple[mx.array, mx.array, mx.array],
    gate: tuple[mx.array, mx.array, mx.array],
    down: tuple[mx.array, mx.array, mx.array],
    *,
    front_results_per_simdgroup: int = 4,
    down_results_per_simdgroup: int = 8,
    derive_front_bias: bool = False,
    derive_down_bias: bool = False,
) -> mx.array:
    """Run both production affine-2 expert shapes at verification width four."""

    activated = width4_switch_situ(
        x,
        indices,
        up,
        gate,
        results_per_simdgroup=front_results_per_simdgroup,
        derive_bias=derive_front_bias,
    )
    output = width4_down(
        activated,
        indices,
        down,
        results_per_simdgroup=down_results_per_simdgroup,
        derive_bias=derive_down_bias,
    )
    return output.squeeze(-2)


def width4_switch_glu_native_down(
    x: mx.array,
    indices: mx.array,
    up: tuple[mx.array, mx.array, mx.array],
    gate: tuple[mx.array, mx.array, mx.array],
    down: tuple[mx.array, mx.array, mx.array],
    *,
    front_results_per_simdgroup: int = 8,
    derive_front_bias: bool = True,
) -> mx.array:
    """Run the width-four front and retain MLX's exact affine2 down core.

    The caller must first validate that the down projection's stored BF16
    biases are exactly derivable from its scales.  The production adapter does
    that before entering this function; ``mode='affine2'`` then avoids loading
    the redundant bias tensor without changing the output bits.
    """

    if not isinstance(derive_front_bias, bool):
        raise TypeError("derive_front_bias must be a bool")
    if not supports_width4_native_down_projection(indices, down):
        raise ValueError("unsupported Kimi K3 width-four native down geometry")
    activated = width4_switch_situ(
        x,
        indices,
        up,
        gate,
        results_per_simdgroup=front_results_per_simdgroup,
        derive_bias=derive_front_bias,
    )
    weight, scales, _ = down
    output = mx.gather_qmm(
        activated,
        weight,
        scales,
        None,
        rhs_indices=indices,
        transpose=True,
        group_size=K3_GROUP_SIZE,
        bits=2,
        mode="affine2",
    )
    return output.squeeze(-2)


def width4_switch_glu_reduce(
    x: mx.array,
    indices: mx.array,
    router_weights: mx.array,
    up: tuple[mx.array, mx.array, mx.array],
    gate: tuple[mx.array, mx.array, mx.array],
    down: tuple[mx.array, mx.array, mx.array],
    *,
    front_results_per_simdgroup: int = 4,
    down_results_per_threadgroup: int = 4,
    down_simdgroups_per_threadgroup: int = 8,
    derive_front_bias: bool = False,
    derive_down_bias: bool = False,
) -> mx.array:
    """Run the exact width-four expert chain without materializing outputs."""

    activated = width4_switch_situ(
        x,
        indices,
        up,
        gate,
        results_per_simdgroup=front_results_per_simdgroup,
        derive_bias=derive_front_bias,
    )
    return width4_down_reduce(
        activated,
        indices,
        router_weights,
        down,
        results_per_threadgroup=down_results_per_threadgroup,
        simdgroups_per_threadgroup=down_simdgroups_per_threadgroup,
        derive_bias=derive_down_bias,
    )
