"""No-copy multi-bank QMV prototype for Kimi K3's decode-time MoE front.

Kimi K3 evaluates four affine-quantized projections from the same residual
vector before its routed and shared experts:

* shared-expert gate;
* shared-expert up;
* router scores; and
* routed-expert latent down.

The existing packed prototype concatenates those banks to issue one native
QMV, but retains a second copy of roughly 81 MB of quantized weights per sparse
layer.  This module binds the four authoritative banks directly to one Metal
dispatch and returns four independent outputs.  Every output row retains the
native MLX affine-8 QMV K loop, per-lane accumulation order, SIMD reduction,
and final cast.  Unsupported shapes remain on the stock MLX-LM path.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Sequence

import mlx.core as mx


MULTIBANK_MOE_FRONT_ENV = "MLX_LM_KIMI_K3_MULTIBANK_MOE_FRONT"


class MultiBankMoEFrontUnsupported(ValueError):
    """Raised when the exact no-copy decode contract is not satisfied."""


@lru_cache(maxsize=1)
def multibank_moe_front_enabled() -> bool:
    return os.environ.get(MULTIBANK_MOE_FRONT_ENV, "0") == "1"


def _metal_available() -> bool:
    metal = getattr(mx, "metal", None)
    return metal is not None and bool(metal.is_available())


def _array_parameter(module: Any, name: str) -> mx.array | None:
    getter = getattr(module, "get", None)
    value = getter(name) if getter is not None else getattr(module, name, None)
    return value if isinstance(value, mx.array) else None


def _quantized_bank(
    module: Any,
) -> tuple[mx.array, mx.array, mx.array, mx.array | None]:
    if (
        int(getattr(module, "group_size", 0)) != 64
        or int(getattr(module, "bits", 0)) != 8
        or str(getattr(module, "mode", "")) != "affine"
    ):
        raise MultiBankMoEFrontUnsupported(
            "multi-bank K3 MoE front requires affine 8-bit/group-64 projections"
        )
    weight = _array_parameter(module, "weight")
    scales = _array_parameter(module, "scales")
    biases = _array_parameter(module, "biases")
    if weight is None or scales is None or biases is None:
        raise MultiBankMoEFrontUnsupported(
            "every projection must have quantized weight, scales, and biases"
        )
    return weight, scales, biases, _array_parameter(module, "bias")


_HEADER = r"""
template <typename T>
inline float k3_multibank_load_x_8bit(
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

inline float k3_multibank_qdot_8bit(
    const device uint8_t* w,
    const thread float* x_thread,
    float scale,
    float bias,
    float sum) {
  float accum = 0.0f;
  for (int i = 0; i < 8; ++i) {
    // Match MLX quantized.h::qdot<float, 8, 8>.
    accum += x_thread[i] * w[i];
  }
  return scale * accum + sum * bias;
}

template <typename T>
inline void k3_multibank_qmv_bank(
    const device uint32_t* weight,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    uint bank_tile,
    uint simdgroup_index,
    uint lane,
    uint input_width,
    uint output_offset) {
  constexpr uint VALUES_PER_THREAD = 8;
  constexpr uint SIMD_SIZE = 32;
  constexpr uint BLOCK_SIZE = VALUES_PER_THREAD * SIMD_SIZE;
  constexpr uint GROUP_SIZE = 64;
  constexpr uint RESULTS_PER_SIMDGROUP = 4;
  constexpr uint SIMDGROUPS = 2;
  constexpr uint OUTPUTS_PER_THREADGROUP =
      RESULTS_PER_SIMDGROUP * SIMDGROUPS;

  const uint output_base =
      bank_tile * OUTPUTS_PER_THREADGROUP +
      simdgroup_index * RESULTS_PER_SIMDGROUP;
  const uint scale_width = input_width / GROUP_SIZE;

  const device T* x_ptr = x + lane * VALUES_PER_THREAD;
  const device uint8_t* weight_ptr =
      reinterpret_cast<const device uint8_t*>(weight) +
      output_base * input_width +
      lane * VALUES_PER_THREAD;
  const device T* scale_ptr =
      scales + output_base * scale_width + lane / 8;
  const device T* bias_ptr =
      biases + output_base * scale_width + lane / 8;

  thread float x_thread[VALUES_PER_THREAD];
  thread float result[RESULTS_PER_SIMDGROUP] = {0.0f};

  // This is the affine_qmv_fast loop from MLX.  Each bank accumulates and
  // rounds independently; no value from another projection enters the loop.
  for (uint k = 0; k < input_width; k += BLOCK_SIZE) {
    float sum = k3_multibank_load_x_8bit<T>(x_ptr, x_thread);
    for (uint row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
      result[row] += k3_multibank_qdot_8bit(
          weight_ptr + row * input_width,
          x_thread,
          scale_ptr[row * scale_width],
          bias_ptr[row * scale_width],
          sum);
    }
    x_ptr += BLOCK_SIZE;
    weight_ptr += BLOCK_SIZE;
    scale_ptr += BLOCK_SIZE / GROUP_SIZE;
    bias_ptr += BLOCK_SIZE / GROUP_SIZE;
  }

  for (uint row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
    result[row] = simd_sum(result[row]);
    if (lane == 0) {
      y[output_offset + output_base + row] = static_cast<T>(result[row]);
    }
  }
}
"""


_SOURCE = r"""
constexpr uint TILES_0 = N0 / 8;
constexpr uint TILES_1 = N1 / 8;
constexpr uint TILES_2 = N2 / 8;

uint tile = threadgroup_position_in_grid.y;
const uint simdgroup_index = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;

if (tile < TILES_0) {
  k3_multibank_qmv_bank<T>(
      weight0, scales0, biases0, x, y, tile, simdgroup_index, lane, K, 0);
} else if ((tile -= TILES_0) < TILES_1) {
  k3_multibank_qmv_bank<T>(
      weight1, scales1, biases1, x, y, tile, simdgroup_index, lane, K, N0);
} else if ((tile -= TILES_1) < TILES_2) {
  k3_multibank_qmv_bank<T>(
      weight2,
      scales2,
      biases2,
      x,
      y,
      tile,
      simdgroup_index,
      lane,
      K,
      N0 + N1);
} else {
  tile -= TILES_2;
  k3_multibank_qmv_bank<T>(
      weight3,
      scales3,
      biases3,
      x,
      y,
      tile,
      simdgroup_index,
      lane,
      K,
      N0 + N1 + N2);
}
"""


@lru_cache(maxsize=1)
def _kernel():
    if not _metal_available():
        return None
    return mx.fast.metal_kernel(
        name="k3_affine8_multibank_qmv_decode",
        input_names=[
            "x",
            "weight0",
            "scales0",
            "biases0",
            "weight1",
            "scales1",
            "biases1",
            "weight2",
            "scales2",
            "biases2",
            "weight3",
            "scales3",
            "biases3",
        ],
        output_names=["y"],
        header=_HEADER,
        source=_SOURCE,
        ensure_row_contiguous=True,
    )


def supports_multibank_affine8_qmv(
    x: mx.array,
    banks: Sequence[tuple[mx.array, mx.array, mx.array]],
) -> bool:
    """Return whether stock MLX selects the matching affine-QMV-fast kernel."""

    if (
        _kernel() is None
        or x.dtype != mx.bfloat16
        or x.ndim != 3
        or x.shape[0] != 1
        or x.shape[1] != 1
        or len(banks) != 4
    ):
        return False

    input_width = int(x.shape[-1])
    if input_width <= 0 or input_width % 512:
        return False

    for weight, scales, biases in banks:
        if (
            weight.ndim != 2
            or scales.ndim != 2
            or biases.ndim != 2
            or weight.dtype != mx.uint32
            or scales.dtype != mx.bfloat16
            or biases.dtype != mx.bfloat16
            or scales.shape != biases.shape
            or int(weight.shape[0]) != int(scales.shape[0])
            or int(weight.shape[1]) * 4 != input_width
            or int(scales.shape[1]) * 64 != input_width
            or int(weight.shape[0]) <= 0
            or int(weight.shape[0]) % 8
        ):
            return False
    return True


def multibank_affine8_qmv(
    x: mx.array,
    banks: Sequence[tuple[mx.array, mx.array, mx.array]],
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Evaluate four independent quantized MVs in one no-copy dispatch."""

    if not supports_multibank_affine8_qmv(x, banks):
        raise MultiBankMoEFrontUnsupported(
            "unsupported no-copy multi-bank affine-8 QMV contract"
        )
    kernel = _kernel()
    assert kernel is not None
    output_widths = tuple(int(bank[0].shape[0]) for bank in banks)
    total_tiles = sum(width // 8 for width in output_widths)
    flat_inputs = [x]
    for bank in banks:
        flat_inputs.extend(bank)
    outputs = kernel(
        inputs=flat_inputs,
        template=[
            ("T", x.dtype),
            ("K", int(x.shape[-1])),
            *((f"N{index}", width) for index, width in enumerate(output_widths)),
        ],
        grid=(32, total_tiles * 2, 1),
        threadgroup=(32, 2, 1),
        output_shapes=[(*x.shape[:-1], sum(output_widths))],
        output_dtypes=[x.dtype],
        stream=mx.gpu,
    )
    split_indices = []
    running_width = 0
    for output_width in output_widths[:-1]:
        running_width += output_width
        split_indices.append(running_width)
    return tuple(mx.split(outputs[0], split_indices, axis=-1))


def _front_modules(sparse_moe: Any) -> tuple[Any, Any, Any, Any]:
    shared = getattr(sparse_moe, "shared_experts", None)
    routed_down = getattr(sparse_moe, "routed_expert_down_proj", None)
    if shared is None or routed_down is None:
        raise MultiBankMoEFrontUnsupported(
            "multi-bank path requires shared experts and latent routed experts"
        )
    return (
        shared.gate_proj,
        shared.up_proj,
        sparse_moe.gate,
        routed_down,
    )


def maybe_multibank_k3_moe_front(
    sparse_moe: Any,
    x: mx.array,
) -> tuple[mx.array, mx.array, mx.array, mx.array] | None:
    """Return four exact no-copy decode projections, or ``None``."""

    if (
        not multibank_moe_front_enabled()
        or getattr(sparse_moe, "training", True)
        or x.ndim != 3
        or x.shape[0] * x.shape[1] != 1
    ):
        return None
    try:
        projections = tuple(
            _quantized_bank(module) for module in _front_modules(sparse_moe)
        )
        banks = tuple(parts[:3] for parts in projections)
        outputs = multibank_affine8_qmv(x, banks)
        with_output_bias = []
        for output, parts in zip(outputs, projections, strict=True):
            output_bias = parts[3]
            with_output_bias.append(
                output if output_bias is None else output + output_bias
            )
        return tuple(with_output_bias)  # type: ignore[return-value]
    except (
        AttributeError,
        MultiBankMoEFrontUnsupported,
        TypeError,
        ValueError,
    ) as exc:
        object.__setattr__(
            sparse_moe,
            "_multibank_k3_moe_front_reason",
            str(exc),
        )
        return None
