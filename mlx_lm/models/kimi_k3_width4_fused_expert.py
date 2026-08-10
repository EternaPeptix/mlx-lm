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
  for (int i = 0; i < 16; i += 4) {
    // Preserve MLX gather-QMV's expression types and association.
    sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
    x_thread[i] = x[i];
    x_thread[i + 1] = x[i + 1] / 4.0f;
    x_thread[i + 2] = x[i + 2] / 16.0f;
    x_thread[i + 3] = x[i + 3] / 64.0f;
  }
  return sum;
}

inline float k3_w4_qdot_2bit(
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

for (uint k = 0; k < input_width; k += BLOCK_SIZE) {
  float sum = k3_w4_load_x_2bit<T>(x_ptr, x_thread);
  for (uint row = 0; row < RESULTS; ++row) {
    const device uint8_t* up_row = up_ptr + row * packed_input_width;
    const device uint8_t* gate_row = gate_ptr + row * packed_input_width;
    const device T* up_scale_row = up_scale_ptr + row * scale_width;
    const device T* up_bias_row = up_bias_ptr + row * scale_width;
    const device T* gate_scale_row = gate_scale_ptr + row * scale_width;
    const device T* gate_bias_row = gate_bias_ptr + row * scale_width;
    T up_scale_value = up_scale_row[0];
    T gate_scale_value = gate_scale_row[0];
    float up_bias;
    float gate_bias;
    if constexpr (DERIVE_BIAS) {
      up_bias = k3_w4_derive_affine2_bias<T>(up_scale_value);
      gate_bias = k3_w4_derive_affine2_bias<T>(gate_scale_value);
    } else {
      up_bias = static_cast<float>(up_bias_row[0]);
      gate_bias = static_cast<float>(gate_bias_row[0]);
    }
    up_acc[row] += k3_w4_qdot_2bit(
        up_row,
        x_thread,
        static_cast<float>(up_scale_value),
        up_bias,
        sum);
    gate_acc[row] += k3_w4_qdot_2bit(
        gate_row,
        x_thread,
        static_cast<float>(gate_scale_value),
        gate_bias,
        sum);
  }
  x_ptr += BLOCK_SIZE;
  up_ptr += BLOCK_SIZE / 4;
  gate_ptr += BLOCK_SIZE / 4;
  up_scale_ptr += BLOCK_SIZE / GROUP_SIZE;
  up_bias_ptr += BLOCK_SIZE / GROUP_SIZE;
  gate_scale_ptr += BLOCK_SIZE / GROUP_SIZE;
  gate_bias_ptr += BLOCK_SIZE / GROUP_SIZE;
}

for (uint row = 0; row < RESULTS; ++row) {
  float up_sum = simd_sum(up_acc[row]);
  float gate_sum = simd_sum(gate_acc[row]);
  if (lane == 0) {
    // Preserve both gather-QMV BF16 boundaries before SiTU's FP32 arithmetic.
    T up_rounded = static_cast<T>(up_sum);
    T gate_rounded = static_cast<T>(gate_sum);
    float up_value = static_cast<float>(up_rounded);
    float gate_value = static_cast<float>(gate_rounded);
    float activation =
        BETA * metal::precise::tanh(gate_value / BETA) *
        k3_w4_sigmoid(gate_value);
    up_value = LINEAR_BETA * metal::precise::tanh(up_value / LINEAR_BETA);
    y[flattened_slot * output_width + output_base + row] =
        static_cast<T>(activation * up_value);
  }
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

for (uint k = 0; k < input_width; k += BLOCK_SIZE) {
  float sum = k3_w4_load_x_2bit<T>(x_ptr, x_thread);
  for (uint row = 0; row < RESULTS; ++row) {
    T scale_value = scale_ptr[row * scale_width];
    float bias;
    if constexpr (DERIVE_BIAS) {
      bias = k3_w4_derive_affine2_bias<T>(scale_value);
    } else {
      bias = static_cast<float>(bias_ptr[row * scale_width]);
    }
    accum[row] += k3_w4_qdot_2bit(
        weight_ptr + row * packed_input_width,
        x_thread,
        static_cast<float>(scale_value),
        bias,
        sum);
  }
  x_ptr += BLOCK_SIZE;
  weight_ptr += BLOCK_SIZE / 4;
  scale_ptr += BLOCK_SIZE / GROUP_SIZE;
  bias_ptr += BLOCK_SIZE / GROUP_SIZE;
}

for (uint row = 0; row < RESULTS; ++row) {
  float value = simd_sum(accum[row]);
  if (lane == 0) {
    y[flattened_slot * output_width + output_base + row] =
        static_cast<T>(value);
  }
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
  for (uint k = 0; k < input_width; k += BLOCK_SIZE) {
    float sum = k3_w4_load_x_2bit<T>(x_ptr, x_thread);
    for (uint row = 0; row < RESULTS; ++row) {
      T scale_value = scale_ptr[row * scale_width];
      float bias;
      if constexpr (DERIVE_BIAS) {
        bias = k3_w4_derive_affine2_bias<T>(scale_value);
      } else {
        bias = static_cast<float>(bias_ptr[row * scale_width]);
      }
      result[row] += k3_w4_qdot_2bit(
          weight_ptr + row * packed_input_width,
          x_thread,
          static_cast<float>(scale_value),
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
      expert_outputs[expert_slot * RESULTS + row] = static_cast<T>(value);
    }
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
        results_per_threadgroup in (2, 4, 8, 16)
        and simdgroups_per_threadgroup in (4, 8)
        and K3_TOP_K % simdgroups_per_threadgroup == 0
        and activated.shape
        == (1, K3_WIDTH4, K3_TOP_K, 1, K3_INTERMEDIATE)
        and activated.dtype == mx.bfloat16
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
