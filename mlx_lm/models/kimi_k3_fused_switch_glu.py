"""Exact-weight fused 2-bit SwitchGLU input projection for Kimi K3 decode.

This is an inference prototype for the Metal decode case only:

* affine 2-bit weights with group size 128;
* one token, top-k expert routing;
* input width divisible by 512 and rank-local intermediate width by 8;
* BF16 activations; and
* Kimi K3's SiTU(beta=4, linear_beta=25).

It fuses the independent gate/up gather-QMVs and SiTU into one dispatch.
Weights are neither dequantized nor requantized, so the model parameters are
preserved exactly. Unsupported shapes should retain the stock MLX-LM path.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx


def _metal_available() -> bool:
    metal = getattr(mx, "metal", None)
    return metal is not None and bool(metal.is_available())


_HEADER = r"""
template <typename T>
inline float k3_load_x_2bit(
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

inline float k3_qdot_2bit(
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

inline float k3_sigmoid(float value) {
  float z = 1.0f / (1.0f + metal::exp(metal::abs(value)));
  return value < 0.0f ? z : 1.0f - z;
}

template <typename T>
inline float k3_derive_affine2_bias(T scale) {
  // Full-array validation admits finite-normal BF16 scales only. Multiplying
  // such a value by a power of two is exact before the existing FP32 QMV.
  return -2.0f * static_cast<float>(scale);
}
"""


_SOURCE = r"""
constexpr int VALUES_PER_THREAD = 16;
constexpr int SIMD_SIZE = 32;
constexpr int BLOCK_SIZE = VALUES_PER_THREAD * SIMD_SIZE;
constexpr int OUTPUTS_PER_THREADGROUP =
    RESULTS * SIMDS;
constexpr int GROUP_SIZE = 128;
constexpr float BETA = 4.0f;
constexpr float LINEAR_BETA = 25.0f;

const uint expert_slot = threadgroup_position_in_grid.z;
const uint expert = indices[expert_slot];
const uint output_base =
    threadgroup_position_in_grid.y * OUTPUTS_PER_THREADGROUP +
    simdgroup_index_in_threadgroup * RESULTS;
const uint lane = thread_index_in_simdgroup;

const uint input_width = x_shape[x_ndim - 1];
const uint output_width = up_scales_shape[up_scales_ndim - 2];
const uint packed_input_width = input_width / 4;
const uint scale_width = input_width / GROUP_SIZE;

const device T* x_ptr = x + lane * VALUES_PER_THREAD;
const device uint8_t* up_ptr =
    reinterpret_cast<const device uint8_t*>(up_weight) +
    (expert * output_width + output_base) * packed_input_width +
    lane * 4;
const device uint8_t* gate_ptr =
    reinterpret_cast<const device uint8_t*>(gate_weight) +
    (expert * output_width + output_base) * packed_input_width +
    lane * 4;
const device T* up_scale_ptr =
    up_scales + (expert * output_width + output_base) * scale_width +
    lane / 8;
const device T* up_bias_ptr =
    up_biases + (expert * output_width + output_base) * scale_width +
    lane / 8;
const device T* gate_scale_ptr =
    gate_scales + (expert * output_width + output_base) * scale_width +
    lane / 8;
const device T* gate_bias_ptr =
    gate_biases + (expert * output_width + output_base) * scale_width +
    lane / 8;

thread float x_thread[VALUES_PER_THREAD];
thread float up_acc[RESULTS] = {0.0f};
thread float gate_acc[RESULTS] = {0.0f};

for (uint k = 0; k < input_width; k += BLOCK_SIZE) {
  float sum = k3_load_x_2bit<T>(x_ptr, x_thread);
  for (int row = 0; row < RESULTS; ++row) {
    const device uint8_t* up_row = up_ptr + row * packed_input_width;
    const device uint8_t* gate_row = gate_ptr + row * packed_input_width;
    const device T* up_scale_row = up_scale_ptr + row * scale_width;
    const device T* up_bias_row = up_bias_ptr + row * scale_width;
    const device T* gate_scale_row = gate_scale_ptr + row * scale_width;
    const device T* gate_bias_row = gate_bias_ptr + row * scale_width;
    T up_scale_value = up_scale_row[0];
    T gate_scale_value = gate_scale_row[0];
    float up_scale = static_cast<float>(up_scale_value);
    float gate_scale = static_cast<float>(gate_scale_value);
    float up_bias;
    float gate_bias;
    if constexpr (DERIVE_BIAS) {
      up_bias = k3_derive_affine2_bias<T>(up_scale_value);
      gate_bias = k3_derive_affine2_bias<T>(gate_scale_value);
    } else {
      up_bias = static_cast<float>(up_bias_row[0]);
      gate_bias = static_cast<float>(gate_bias_row[0]);
    }
    up_acc[row] += k3_qdot_2bit(
        up_row,
        x_thread,
        up_scale,
        up_bias,
        sum);
    gate_acc[row] += k3_qdot_2bit(
        gate_row,
        x_thread,
        gate_scale,
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

for (int row = 0; row < RESULTS; ++row) {
  float up_sum = simd_sum(up_acc[row]);
  float gate_sum = simd_sum(gate_acc[row]);
  if (lane == 0) {
    // Match the native gather_qmv output rounding before SiTU casts to FP32.
    T up_rounded = static_cast<T>(up_sum);
    T gate_rounded = static_cast<T>(gate_sum);
    float up_value = static_cast<float>(up_rounded);
    float gate_value = static_cast<float>(gate_rounded);
    float activation =
        BETA * metal::precise::tanh(gate_value / BETA) *
        k3_sigmoid(gate_value);
    up_value =
        LINEAR_BETA * metal::precise::tanh(up_value / LINEAR_BETA);
    const uint output = output_base + row;
    y[expert_slot * output_width + output] =
        static_cast<T>(activation * up_value);
  }
}
"""


@lru_cache(maxsize=None)
def _kernel():
    if not _metal_available():
        return None
    return mx.fast.metal_kernel(
        name="k3_fused_2bit_switch_situ_decode",
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
        source=_SOURCE,
        ensure_row_contiguous=True,
    )


def supports_fused_switch_situ(
    x: mx.array,
    indices: mx.array,
    up: tuple[mx.array, mx.array, mx.array],
    gate: tuple[mx.array, mx.array, mx.array],
) -> bool:
    """Return whether inputs satisfy the prototype's fail-closed contract."""

    if x.dtype != mx.bfloat16 or _kernel() is None:
        return False
    if x.ndim != 3 or x.shape[0] != 1 or x.shape[-2] != 1 or indices.ndim != 3:
        return False
    if indices.shape[:-1] != x.shape[:-1] or indices.dtype != mx.uint32:
        return False
    if x.shape[-1] % 512 != 0:
        return False
    if any(part.ndim != 3 for part in (*up, *gate)):
        return False
    if any(a.shape != b.shape for a, b in zip(up, gate, strict=True)):
        return False
    if up[0].shape[:2] != up[1].shape[:2] or up[1].shape != up[2].shape:
        return False
    output_width = up[1].shape[-2]
    expert_count = up[0].shape[0]
    return (
        expert_count > 0
        and 0 < indices.shape[-1] <= expert_count
        and output_width % 8 == 0
        and up[0].dtype == mx.uint32
        and gate[0].dtype == mx.uint32
        and all(part.dtype == mx.bfloat16 for part in (*up[1:], *gate[1:]))
        and up[0].shape[-1] * 16 == x.shape[-1]
        and up[1].shape[-1] * 128 == x.shape[-1]
    )


def fused_switch_situ_decode(
    x: mx.array,
    indices: mx.array,
    up: tuple[mx.array, mx.array, mx.array],
    gate: tuple[mx.array, mx.array, mx.array],
    *,
    results_per_simdgroup: int = 4,
    simdgroups: int = 2,
    derive_bias: bool = False,
) -> mx.array:
    """Compute K3's gate/up gather-QMVs and SiTU in one Metal dispatch."""

    if not supports_fused_switch_situ(x, indices, up, gate):
        raise ValueError("unsupported fused K3 SwitchGLU decode inputs")
    if results_per_simdgroup not in (2, 4, 8, 16):
        raise ValueError("results_per_simdgroup must be 2, 4, 8, or 16")
    if simdgroups not in (1, 2, 4):
        raise ValueError("simdgroups must be 1, 2, or 4")
    if not isinstance(derive_bias, bool):
        raise TypeError("derive_bias must be a bool")
    kernel = _kernel()
    assert kernel is not None
    output_width = up[1].shape[-2]
    top_k = indices.shape[-1]
    tile = results_per_simdgroup * simdgroups
    if output_width % tile:
        raise ValueError("output width must be divisible by the fused tile")
    outputs = kernel(
        inputs=[x, indices, *up, *gate],
        template=[
            ("T", x.dtype),
            ("RESULTS", results_per_simdgroup),
            ("SIMDS", simdgroups),
            ("DERIVE_BIAS", derive_bias),
        ],
        grid=(32, (output_width // tile) * simdgroups, top_k),
        threadgroup=(32, simdgroups, 1),
        output_shapes=[(*indices.shape, 1, output_width)],
        output_dtypes=[x.dtype],
        stream=mx.gpu,
    )
    return outputs[0]
