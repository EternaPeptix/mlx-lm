"""Bit-exact split-K decode kernels for Kimi K3's affine 2-bit routed experts.

Experimental alternative to ``kimi_k3_fused_switch_glu.fused_switch_situ_decode``
and ``kimi_k3_tuned_gather_qmv.tuned_gather_qmv`` for one decode token with
derived affine-2 biases (``bias = -2 * scale``).

Each SIMD-group owns one 512-wide input block instead of walking the whole row.
Per-lane block partials are staged in threadgroup memory and summed in the
original block order before ``simd_sum``, so every output is bit-identical to
the row-walking kernels (verified on real K3 shapes, top-8 routing).

Measured on M3 Ultra, top-8, 92 layers: 9.00 ms/token for gate/up + down versus
9.85 ms for the production kernels in isolation. In the full TP2 model the
difference was within noise (8.46 s vs 8.45 s decode per 128 tokens), so this
module is not wired into the model by default.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx

_HEADER = r"""
template <typename T>
inline float k3s_load_x_2bit(const device T* x, thread float* x_thread) {
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
inline float k3s_qdot_2bit_u32(uint w, const thread float* x_thread, float scale, float bias, float sum) {
  float accum = 0.0f;
  for (int i = 0; i < 4; ++i) {
    uint8_t packed = uint8_t((w >> (8 * i)) & 0xffu);
    accum += x_thread[4 * i] * (packed & 0x03) + x_thread[4 * i + 1] * (packed & 0x0c) +
             x_thread[4 * i + 2] * (packed & 0x30) + x_thread[4 * i + 3] * (packed & 0xc0);
  }
  return scale * accum + sum * bias;
}
inline float k3s_sigmoid(float value) {
  float z = 1.0f / (1.0f + metal::exp(metal::abs(value)));
  return value < 0.0f ? z : 1.0f - z;
}
"""

_GATE_UP_SOURCE = r"""
constexpr int VPT = 16; constexpr int BS = 512; constexpr int GS = 128;
constexpr int NB = IN_W / BS;
constexpr float BETA = 4.0f; constexpr float LINEAR_BETA = 25.0f;
threadgroup float pu[ROWS][NB][32];
threadgroup float pg[ROWS][NB][32];
const uint epk = indices_shape[indices_ndim - 1];
const uint slot = threadgroup_position_in_grid.z;
const uint tok = slot / epk;
const uint expert = indices[slot];
const uint ob = threadgroup_position_in_grid.y * ROWS;
const uint lane = thread_index_in_simdgroup;
const uint b = simdgroup_index_in_threadgroup;
const uint out_w = up_scales_shape[up_scales_ndim - 2];
constexpr uint PW = IN_W / 4; constexpr uint SW = IN_W / GS;
const device uint8_t* upb = (const device uint8_t*)up_weight + (expert * out_w + ob) * PW + b * (BS / 4) + lane * 4;
const device uint8_t* gpb = (const device uint8_t*)gate_weight + (expert * out_w + ob) * PW + b * (BS / 4) + lane * 4;
const device T* ups = up_scales + (expert * out_w + ob) * SW + b * (BS / GS) + lane / 8;
const device T* gss = gate_scales + (expert * out_w + ob) * SW + b * (BS / GS) + lane / 8;
thread float xt[VPT];
float sum = k3s_load_x_2bit<T>(x + tok * IN_W + b * BS + lane * VPT, xt);
for (int r = 0; r < ROWS; ++r) {
  T su = ups[r * SW]; T sg = gss[r * SW];
  pu[r][b][lane] = k3s_qdot_2bit_u32(*(const device uint*)(upb + r * PW), xt, float(su), -2.0f * float(su), sum);
  pg[r][b][lane] = k3s_qdot_2bit_u32(*(const device uint*)(gpb + r * PW), xt, float(sg), -2.0f * float(sg), sum);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (b < ROWS) {
  const int r = b;
  float ua = 0.0f; float ga = 0.0f;
  for (int j = 0; j < NB; ++j) { ua += pu[r][j][lane]; ga += pg[r][j][lane]; }
  float us = simd_sum(ua); float gsum = simd_sum(ga);
  if (lane == 0) {
    float uv = float(static_cast<T>(us)); float gv = float(static_cast<T>(gsum));
    float act = BETA * metal::precise::tanh(gv / BETA) * k3s_sigmoid(gv);
    uv = LINEAR_BETA * metal::precise::tanh(uv / LINEAR_BETA);
    y[slot * out_w + ob + r] = static_cast<T>(act * uv);
  }
}
"""

_DOWN_SOURCE = r"""
constexpr int VPT = 16; constexpr int BS = 512; constexpr int GS = 128;
constexpr int NB = IN_W / BS;
threadgroup float pp[ROWS][NB][32];
const uint slot = threadgroup_position_in_grid.z;
const uint expert = indices[slot];
const uint lane = thread_index_in_simdgroup;
const uint b = simdgroup_index_in_threadgroup;
const uint ob = threadgroup_position_in_grid.y * ROWS;
const uint out_w = scales_shape[scales_ndim - 2];
constexpr uint PW = IN_W / 4; constexpr uint SW = IN_W / GS;
const device uint8_t* wb = (const device uint8_t*)weight + (expert * out_w + ob) * PW + b * (BS / 4) + lane * 4;
const device T* sp = scales + (expert * out_w + ob) * SW + b * (BS / GS) + lane / 8;
thread float xt[VPT];
float sum = k3s_load_x_2bit<T>(x + slot * IN_W + b * BS + lane * VPT, xt);
for (int r = 0; r < ROWS; ++r) {
  T s = sp[r * SW];
  pp[r][b][lane] = k3s_qdot_2bit_u32(*(const device uint*)(wb + r * PW), xt, float(s), -2.0f * float(s), sum);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (int r = b; r < ROWS; r += NB) {
  float acc = 0.0f;
  for (int j = 0; j < NB; ++j) acc += pp[r][j][lane];
  float v = simd_sum(acc);
  if (lane == 0) y[slot * out_w + ob + r] = static_cast<T>(v);
}
"""

GATE_UP_ROWS = 2
DOWN_ROWS = 4


def _metal_available() -> bool:
    metal = getattr(mx, "metal", None)
    return metal is not None and bool(metal.is_available())


@lru_cache(maxsize=None)
def _gate_up_kernel():
    if not _metal_available():
        return None
    return mx.fast.metal_kernel(
        name="k3_splitk_switch_situ_decode",
        input_names=["x", "indices", "up_weight", "up_scales", "gate_weight", "gate_scales"],
        output_names=["y"],
        header=_HEADER,
        source=_GATE_UP_SOURCE,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=None)
def _down_kernel():
    if not _metal_available():
        return None
    return mx.fast.metal_kernel(
        name="k3_splitk_gather_qmv_down",
        input_names=["x", "indices", "weight", "scales"],
        output_names=["y"],
        header=_HEADER,
        source=_DOWN_SOURCE,
        ensure_row_contiguous=True,
    )


def splitk_switch_situ_decode(x, indices, up, gate):
    """Fused gate/up gather-QMV + SiTU for one token (derived affine-2 biases).

    ``x``: (1, 1, D) bfloat16; ``indices``: (1, 1, K) uint32;
    ``up``/``gate``: (weight uint32 [E, I, D/16], scales bf16 [E, I, D/128], biases).
    Returns (1, 1, K, 1, I).
    """
    input_width = x.shape[-1]
    output_width = up[1].shape[-2]
    if input_width % 512 or output_width % GATE_UP_ROWS or x.shape[-2] != 1:
        raise ValueError("unsupported split-K gate/up shape")
    blocks = input_width // 512
    return _gate_up_kernel()(
        inputs=[x, indices, up[0], up[1], gate[0], gate[1]],
        template=[("T", x.dtype), ("ROWS", GATE_UP_ROWS), ("IN_W", input_width)],
        grid=(32, (output_width // GATE_UP_ROWS) * blocks, indices.size),
        threadgroup=(32, blocks, 1),
        output_shapes=[(*indices.shape, 1, output_width)],
        output_dtypes=[x.dtype],
        stream=mx.gpu,
    )[0]


def splitk_gather_qmv_down(x, indices, down):
    """Per-expert down projection for one token (derived affine-2 biases).

    ``x``: activations with one row per routed slot, width I;
    ``down``: (weight uint32 [E, D, I/16], scales bf16 [E, D, I/128], biases).
    Returns (*indices.shape, 1, D).
    """
    weight, scales, _ = down
    input_width = scales.shape[-1] * 128
    output_width = scales.shape[-2]
    if input_width % 512 or output_width % DOWN_ROWS or indices.shape[-2] != 1:
        raise ValueError("unsupported split-K down shape")
    blocks = input_width // 512
    return _down_kernel()(
        inputs=[x, indices, weight, scales],
        template=[("T", x.dtype), ("ROWS", DOWN_ROWS), ("IN_W", input_width)],
        grid=(32, (output_width // DOWN_ROWS) * blocks, indices.size),
        threadgroup=(32, blocks, 1),
        output_shapes=[(*indices.shape, 1, output_width)],
        output_dtypes=[x.dtype],
        stream=mx.gpu,
    )[0]
