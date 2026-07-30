"""Opt-in exact packing for Kimi K3's decode-time MoE front projections.

Kimi K3 evaluates four affine-quantized projections from the same residual
vector before its routed and shared experts:

* shared-expert gate;
* shared-expert up;
* router scores; and
* routed-expert latent down.

At the single-token decode shape, concatenating their already-quantized output
rows lets MLX issue one QMV instead of four while preserving every output bit.
No weight is dequantized or requantized.  Multi-token calls remain on the stock
path because a wider QMM may choose a different accumulation tiling.
"""

from __future__ import annotations

import os
from functools import lru_cache
from itertools import accumulate
from typing import Any, Sequence

import mlx.core as mx
import mlx.nn as nn


PACKED_MOE_FRONT_ENV = "MLX_LM_KIMI_K3_PACKED_MOE_FRONT"
_UNSUPPORTED = object()


class PackedMoEFrontUnsupported(ValueError):
    """Raised when projections cannot be packed without changing semantics."""


@lru_cache(maxsize=1)
def packed_moe_front_enabled() -> bool:
    return os.environ.get(PACKED_MOE_FRONT_ENV, "0") == "1"


def _array_parameter(module: Any, name: str) -> mx.array | None:
    getter = getattr(module, "get", None)
    value = getter(name) if getter is not None else getattr(module, name, None)
    return value if isinstance(value, mx.array) else None


def _all_or_none(
    modules: Sequence[Any],
    name: str,
) -> list[mx.array] | None:
    values = [_array_parameter(module, name) for module in modules]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise PackedMoEFrontUnsupported(
            f"{name!r} must be present on every projection or none"
        )
    return [value for value in values if value is not None]


class PackedK3MoEFront(nn.Module):
    """One lossless quantized projection split into the four original rows."""

    def __init__(self, modules: Sequence[Any]):
        super().__init__()
        if len(modules) != 4:
            raise PackedMoEFrontUnsupported(
                f"expected four MoE-front projections, got {len(modules)}"
            )

        configs = []
        weights = []
        output_dims = []
        input_dims = []
        for module in modules:
            weight = _array_parameter(module, "weight")
            scales = _array_parameter(module, "scales")
            biases = _array_parameter(module, "biases")
            if weight is None or scales is None or biases is None:
                raise PackedMoEFrontUnsupported(
                    "every projection must be affine quantized"
                )
            if weight.ndim != 2 or scales.ndim != 2 or biases.ndim != 2:
                raise PackedMoEFrontUnsupported(
                    "packed weights, scales, and biases must be two-dimensional"
                )

            group_size = int(getattr(module, "group_size", 0))
            bits = int(getattr(module, "bits", 0))
            mode = str(getattr(module, "mode", ""))
            if group_size <= 0 or bits <= 0:
                raise PackedMoEFrontUnsupported(
                    "quantization group size and bit width must be positive"
                )
            configs.append((group_size, bits, mode))
            weights.append(weight)
            output_dims.append(int(weight.shape[0]))
            input_dims.append((int(weight.shape[1]) * 32) // bits)

            if int(scales.shape[0]) != int(weight.shape[0]):
                raise PackedMoEFrontUnsupported(
                    "weight and scale output dimensions differ"
                )
            if tuple(scales.shape) != tuple(biases.shape):
                raise PackedMoEFrontUnsupported(
                    "affine scales and biases have different layouts"
                )
            if int(scales.shape[1]) * group_size != input_dims[-1]:
                raise PackedMoEFrontUnsupported(
                    "packed weight and scale input dimensions differ"
                )

        if len(set(configs)) != 1:
            raise PackedMoEFrontUnsupported(
                "all projections must share one quantization layout"
            )
        group_size, bits, mode = configs[0]
        if (group_size, bits, mode) != (64, 8, "affine"):
            raise PackedMoEFrontUnsupported(
                "K3 MoE-front packing requires affine 8-bit/group-64"
            )
        if len(set(input_dims)) != 1:
            raise PackedMoEFrontUnsupported(
                "all projections must consume the same hidden width"
            )

        scales = _all_or_none(modules, "scales")
        quant_biases = _all_or_none(modules, "biases")
        output_biases = _all_or_none(modules, "bias")
        if scales is None or quant_biases is None:
            raise PackedMoEFrontUnsupported(
                "affine projections require scales and quantization biases"
            )

        self.weight = mx.concatenate(weights, axis=0)
        self.scales = mx.concatenate(scales, axis=0)
        self.biases = mx.concatenate(quant_biases, axis=0)
        if output_biases is not None:
            self.bias = mx.concatenate(output_biases, axis=0)

        self.group_size = group_size
        self.bits = bits
        self.mode = mode
        object.__setattr__(self, "_input_dims", input_dims[0])
        object.__setattr__(
            self,
            "_split_indices",
            tuple(accumulate(output_dims))[:-1],
        )
        object.__setattr__(
            self,
            "_packed_nbytes",
            sum(int(value.nbytes) for value in (self.weight, self.scales, self.biases))
            + (
                int(self["bias"].nbytes)
                if isinstance(self.get("bias"), mx.array)
                else 0
            ),
        )
        self.freeze()

    @property
    def packed_nbytes(self) -> int:
        return self._packed_nbytes

    def __call__(self, x: mx.array) -> tuple[mx.array, ...]:
        if x.ndim != 3 or x.shape[0] * x.shape[1] != 1:
            raise PackedMoEFrontUnsupported(
                "packed K3 MoE front is decode-only (one total token)"
            )
        if int(x.shape[-1]) != self._input_dims:
            raise PackedMoEFrontUnsupported(
                f"input width {x.shape[-1]} != packed width {self._input_dims}"
            )

        output = mx.quantized_matmul(
            x,
            self["weight"],
            scales=self["scales"],
            biases=self["biases"],
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
        )
        bias = self.get("bias")
        if isinstance(bias, mx.array):
            output = output + bias
        return tuple(mx.split(output, self._split_indices, axis=-1))


def _build_packed_front(sparse_moe: Any) -> PackedK3MoEFront:
    shared = getattr(sparse_moe, "shared_experts", None)
    routed_down = getattr(sparse_moe, "routed_expert_down_proj", None)
    if shared is None or routed_down is None:
        raise PackedMoEFrontUnsupported(
            "packed path requires shared experts and latent routed experts"
        )
    return PackedK3MoEFront(
        (
            shared.gate_proj,
            shared.up_proj,
            sparse_moe.gate,
            routed_down,
        )
    )


def maybe_packed_k3_moe_front(
    sparse_moe: Any,
    x: mx.array,
) -> tuple[mx.array, ...] | None:
    """Return four exact decode projections, or ``None`` for the stock path."""

    if (
        not packed_moe_front_enabled()
        or getattr(sparse_moe, "training", True)
        or x.ndim != 3
        or x.shape[0] * x.shape[1] != 1
    ):
        return None

    packed = getattr(sparse_moe, "_packed_k3_moe_front", None)
    if packed is _UNSUPPORTED:
        return None
    if packed is None:
        try:
            packed = _build_packed_front(sparse_moe)
        except (
            AttributeError,
            PackedMoEFrontUnsupported,
            TypeError,
            ValueError,
        ) as exc:
            object.__setattr__(sparse_moe, "_packed_k3_moe_front_reason", str(exc))
            object.__setattr__(sparse_moe, "_packed_k3_moe_front", _UNSUPPORTED)
            return None
        # Keep the optimization out of the model parameter tree: the original
        # modules remain authoritative for checkpoint save/load and prefill.
        object.__setattr__(sparse_moe, "_packed_k3_moe_front", packed)

    try:
        return packed(x)
    except PackedMoEFrontUnsupported:
        return None
