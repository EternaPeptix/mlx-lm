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
  for (int i = 0; i < 16; i += 4) {
    sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
    x_thread[i] = x[i];
    x_thread[i + 1] = x[i + 1] / 4.0f;
    x_thread[i + 2] = x[i + 2] / 16.0f;
    x_thread[i + 3] = x[i + 3] / 64.0f;
  }
  return sum;
}

inline float k3_tuned_qdot_2bit(
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

for (uint k = 0; k < input_width; k += BLOCK_SIZE) {
  float sum = k3_tuned_load_x_2bit<T>(x_ptr, x_thread);
  for (int row = 0; row < RESULTS; ++row) {
    result[row] += k3_tuned_qdot_2bit(
        weight_ptr + row * packed_input_width,
        x_thread,
        static_cast<float>(scale_ptr[row * scale_width]),
        static_cast<float>(bias_ptr[row * scale_width]),
        sum);
  }
  x_ptr += BLOCK_SIZE;
  weight_ptr += BLOCK_SIZE / 4;
  scale_ptr += BLOCK_SIZE / GROUP_SIZE;
  bias_ptr += BLOCK_SIZE / GROUP_SIZE;
}

for (int row = 0; row < RESULTS; ++row) {
  float value = simd_sum(result[row]);
  if (lane == 0) {
    y[expert_slot * output_width + output_base + row] =
        static_cast<T>(value);
  }
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
) -> mx.array:
    """Run an exact-arithmetic tiled QMV for one-token expert routing."""

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
