#!/usr/bin/env python3
"""Default-off exact one-token Kimi K3 Q/K RMS normalization fusion."""

from __future__ import annotations

import os
from functools import lru_cache
from threading import Lock
from typing import Any

import mlx.core as mx


SELECTOR = "MLX_LM_KIMI_K3_FUSED_QK_RMS_DECODE"
PACKED_WIDE_SELECTOR = "MLX_LM_KIMI_K3_PACKED_KDA_WIDE"
M1_PARENT_SELECTOR = "MLX_LM_KIMI_K3_MOK_ROUTED_SHARED_OVERLAP"
M1_WIDTH1_SELECTOR = "MLX_LM_KIMI_K3_MOK_ROUTED_SHARED_OVERLAP_WIDTH1"
M1_RECEIPT_SELECTOR = (
    "MLX_LM_KIMI_K3_MOK_ROUTED_SHARED_OVERLAP_WIDTH1_RECEIPT"
)
ROWS = 48
HEAD_DIM = 128
_NREADS = 4
_RECEIPT_LOCK = Lock()
_RECEIPT = {
    "attempted": 0,
    "dispatched": 0,
    "unsupported": 0,
    "errors": 0,
}
_ADMITTED_LAYERS: set[int] = set()

_SOURCE = r"""
    uint row = threadgroup_position_in_grid.y;
    uint lid = thread_position_in_threadgroup.x;
    bool use_k = row >= ROWS;
    uint local_row = use_k ? row - ROWS : row;
    const device T* input = (use_k ? k : q) + local_row * D;
    device T* output = (use_k ? out_k : out_q) + local_row * D;

    float acc = 0.0f;
    for (uint i = 0; i < NREADS; ++i) {
      float value = static_cast<float>(input[lid * NREADS + i]);
      acc += value * value;
    }
    acc = simd_sum(acc);
    float inv = metal::precise::rsqrt(acc / D + eps[0]);
    float scale = scales[use_k ? 1 : 0];
    for (uint i = 0; i < NREADS; ++i) {
      uint index = lid * NREADS + i;
      // Stock mx.fast.rms_norm stores BF16 before the separate BF16 scalar
      // multiply. Preserve both boundaries exactly.
      T rms = static_cast<T>(static_cast<float>(input[index]) * inv);
      output[index] = static_cast<T>(static_cast<float>(rms) * scale);
    }
"""

_KERNEL = (
    mx.fast.metal_kernel(
        name="k3_fused_qk_rms_decode",
        input_names=["q", "k", "eps", "scales"],
        output_names=["out_q", "out_k"],
        source=_SOURCE,
    )
    if mx.metal.is_available()
    else None
)
_CONSTANTS: dict[float, tuple[mx.array, mx.array]] = {}


@lru_cache(maxsize=1)
def fused_qk_rms_enabled() -> bool:
    value = os.environ.get(SELECTOR, "0")
    if value not in {"0", "1"}:
        raise ValueError(f"{SELECTOR} must be exactly '0' or '1'")
    enabled = value == "1"
    if enabled:
        required = {
            PACKED_WIDE_SELECTOR: "1",
            M1_PARENT_SELECTOR: "1",
            M1_WIDTH1_SELECTOR: "1",
            M1_RECEIPT_SELECTOR: "1",
        }
        drift = {
            name: os.environ.get(name, "0")
            for name, expected in required.items()
            if os.environ.get(name, "0") != expected
        }
        if drift:
            raise ValueError(
                "fused KDA Q/K RMS requires packed-wide, M1 overlap, and "
                f"bilateral M1 receipt: {sorted(drift)}"
            )
    return enabled


def reset_fused_qk_rms_receipt() -> None:
    with _RECEIPT_LOCK:
        _RECEIPT.update({name: 0 for name in _RECEIPT})
        _ADMITTED_LAYERS.clear()


def fused_qk_rms_receipt() -> dict[str, Any]:
    """Return metadata only; never evaluate or synchronize an MLX array."""

    with _RECEIPT_LOCK:
        counters = dict(_RECEIPT)
        admitted_layer_ids = sorted(_ADMITTED_LAYERS)
    return {
        "schema": "k3-c1-target-only-fused-kda-qk-rms-receipt/v1",
        "enabled": fused_qk_rms_enabled(),
        "counters": counters,
        "admitted_layer_ids": admitted_layer_ids,
        "poisoned": counters["unsupported"] > 0 or counters["errors"] > 0,
    }


def _constants(scale: float) -> tuple[mx.array, mx.array]:
    value = _CONSTANTS.get(float(scale))
    if value is None:
        value = (
            mx.array([1e-6 / HEAD_DIM], dtype=mx.float32),
            # Stock BF16 x Python-float scalar promotion represents both
            # scalars at BF16 before the multiply.
            mx.array([scale**2, scale], dtype=mx.bfloat16),
        )
        _CONSTANTS[float(scale)] = value
    return value


def maybe_fused_kda_qk_rms(
    q: mx.array,
    k: mx.array,
    scale: float,
    *,
    training: bool = False,
    layer_idx: int | None = None,
) -> tuple[mx.array, mx.array] | None:
    if not fused_qk_rms_enabled():
        return None
    if type(layer_idx) is not int or layer_idx < 0:
        with _RECEIPT_LOCK:
            _RECEIPT["unsupported"] += 1
        raise RuntimeError("fused KDA Q/K RMS requires an exact layer identity")
    with _RECEIPT_LOCK:
        _RECEIPT["attempted"] += 1
    expected = (1, 1, ROWS, HEAD_DIM)
    if (
        _KERNEL is None
        or training
        or mx.default_device() != mx.gpu
        or q.shape != expected
        or k.shape != expected
        or q.dtype != mx.bfloat16
        or k.dtype != mx.bfloat16
    ):
        with _RECEIPT_LOCK:
            _RECEIPT["unsupported"] += 1
        raise RuntimeError("requested fused KDA Q/K RMS contract is unsupported")
    try:
        eps, scales = _constants(scale)
        output = tuple(
            _KERNEL(
                inputs=[q, k, eps, scales],
                template=[
                    ("T", mx.bfloat16),
                    ("ROWS", ROWS),
                    ("D", HEAD_DIM),
                    ("NREADS", _NREADS),
                ],
                grid=(32, ROWS * 2, 1),
                threadgroup=(32, 1, 1),
                output_shapes=[q.shape, k.shape],
                output_dtypes=[q.dtype, k.dtype],
                stream=mx.gpu,
            )
        )
    except BaseException:
        with _RECEIPT_LOCK:
            _RECEIPT["errors"] += 1
        raise
    with _RECEIPT_LOCK:
        _RECEIPT["dispatched"] += 1
        _ADMITTED_LAYERS.add(layer_idx)
    return output
