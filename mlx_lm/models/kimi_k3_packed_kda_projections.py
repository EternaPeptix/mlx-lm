"""Exact decode-only packing for Kimi K3's same-input KDA projections.

The released Kimi K3 checkpoint uses a full-rank output gate.  Its KDA decode
evaluates two useful groups of affine-quantized projections from the same
hidden-state vector:

* skinny ``f_a_proj`` plus head-sharded ``b_proj``; and
* rank-local ``qkv_proj`` plus the full-rank output ``g_proj``.

Each default-off pack concatenates already-quantized rows and replaces two
single-token QMV launches with one.  Older low-rank-gate KDA configurations
can use only the skinny pack, where ``g_a_proj`` joins ``f_a_proj`` and
``b_proj``.

Both packed allocations are authoritative: their original modules are
repointed to row views after concatenation has evaluated successfully.  This
keeps the original parameter names and multi-token behavior without retaining
duplicate weights.  Both optimizations fail closed for unverified shapes or
quantization layouts.
"""

from __future__ import annotations

import os
from functools import lru_cache
from itertools import accumulate
from typing import Any, Sequence

import mlx.core as mx
import mlx.nn as nn


PACKED_KDA_SKINNY_ENV = "MLX_LM_KIMI_K3_PACKED_KDA_SKINNY"
PACKED_KDA_WIDE_ENV = "MLX_LM_KIMI_K3_PACKED_KDA_WIDE"
_ARRAY_NAMES = ("weight", "scales", "biases", "bias")
_UNSUPPORTED = object()


class PackedKDASkinnyUnsupported(ValueError):
    """Raised when KDA projections cannot be packed exactly."""


class PackedKDAWideUnsupported(PackedKDASkinnyUnsupported):
    """Raised when KDA's wide QKV/gate projections cannot be packed exactly."""


@lru_cache(maxsize=1)
def packed_kda_skinny_enabled() -> bool:
    return os.environ.get(PACKED_KDA_SKINNY_ENV, "0") == "1"


@lru_cache(maxsize=1)
def packed_kda_wide_enabled() -> bool:
    return os.environ.get(PACKED_KDA_WIDE_ENV, "0") == "1"


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
        raise PackedKDASkinnyUnsupported(
            f"{name!r} must be present on every projection or none"
        )
    return [value for value in values if value is not None]


def _source_signature(modules: Sequence[Any]) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (
            module,
            _array_parameter(module, "weight"),
            _array_parameter(module, "scales"),
            _array_parameter(module, "biases"),
            _array_parameter(module, "bias"),
            int(getattr(module, "group_size", 0)),
            int(getattr(module, "bits", 0)),
            str(getattr(module, "mode", "")),
        )
        for module in modules
    )


def _same_source_signature(
    expected: tuple[tuple[Any, ...], ...],
    actual: tuple[tuple[Any, ...], ...],
) -> bool:
    if len(expected) != len(actual):
        return False
    for expected_projection, actual_projection in zip(expected, actual, strict=True):
        if any(
            expected_projection[index] is not actual_projection[index]
            for index in range(5)
        ):
            return False
        if expected_projection[5:] != actual_projection[5:]:
            return False
    return True


class AuthoritativePackedK3KDASkinny(nn.Module):
    """One QMV backed by the original projections' authoritative row storage."""

    def __init__(self, modules: Sequence[Any]):
        super().__init__()
        modules = tuple(modules)
        if len(modules) not in (2, 3):
            raise PackedKDASkinnyUnsupported(
                f"expected two or three KDA projections, got {len(modules)}"
            )

        configs: list[tuple[int, int, str]] = []
        weights: list[mx.array] = []
        output_dims: list[int] = []
        input_dims: list[int] = []
        for module in modules:
            weight = _array_parameter(module, "weight")
            scales = _array_parameter(module, "scales")
            biases = _array_parameter(module, "biases")
            if weight is None or scales is None or biases is None:
                raise PackedKDASkinnyUnsupported(
                    "every packed KDA projection must be affine quantized"
                )
            if weight.ndim != 2 or scales.ndim != 2 or biases.ndim != 2:
                raise PackedKDASkinnyUnsupported(
                    "packed weights, scales, and biases must be two-dimensional"
                )

            group_size = int(getattr(module, "group_size", 0))
            bits = int(getattr(module, "bits", 0))
            mode = str(getattr(module, "mode", ""))
            configs.append((group_size, bits, mode))
            weights.append(weight)
            output_dims.append(int(weight.shape[0]))
            input_dims.append((int(weight.shape[1]) * 32) // bits)

            if int(scales.shape[0]) != int(weight.shape[0]):
                raise PackedKDASkinnyUnsupported(
                    "weight and scale output dimensions differ"
                )
            if tuple(scales.shape) != tuple(biases.shape):
                raise PackedKDASkinnyUnsupported(
                    "affine scales and biases have different layouts"
                )
            if int(scales.shape[1]) * group_size != input_dims[-1]:
                raise PackedKDASkinnyUnsupported(
                    "packed weight and scale input dimensions differ"
                )

        if len(set(configs)) != 1:
            raise PackedKDASkinnyUnsupported(
                "all packed KDA projections must share one quantization layout"
            )
        group_size, bits, mode = configs[0]
        if (group_size, bits, mode) != (64, 6, "affine"):
            raise PackedKDASkinnyUnsupported(
                "K3 KDA packing requires affine 6-bit/group-64 weights"
            )
        if len(set(input_dims)) != 1:
            raise PackedKDASkinnyUnsupported(
                "all packed KDA projections must consume the same hidden width"
            )

        scales = _all_or_none(modules, "scales")
        quant_biases = _all_or_none(modules, "biases")
        output_biases = _all_or_none(modules, "bias")
        if scales is None or quant_biases is None:
            raise PackedKDASkinnyUnsupported(
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

        split_indices = tuple(accumulate(output_dims))[:-1]
        object.__setattr__(self, "_input_dims", input_dims[0])
        object.__setattr__(self, "_output_dims", tuple(output_dims))
        object.__setattr__(self, "_split_indices", split_indices)
        object.__setattr__(
            self,
            "_packed_nbytes",
            sum(
                int(value.nbytes)
                for name in _ARRAY_NAMES
                if isinstance((value := _array_parameter(self, name)), mx.array)
            ),
        )
        self.freeze()

        # Complete every concatenation and every row view before mutating any
        # source module. Constructor failures therefore remain fail closed.
        mx.eval(self.parameters())
        source_views: dict[str, tuple[mx.array, ...]] = {}
        for name in _ARRAY_NAMES:
            packed_array = _array_parameter(self, name)
            if packed_array is not None:
                source_views[name] = tuple(
                    mx.split(packed_array, split_indices, axis=0)
                )
        mx.eval(*(value for views in source_views.values() for value in views))

        for index, module in enumerate(modules):
            for name, views in source_views.items():
                setattr(module, name, views[index])
        object.__setattr__(self, "_source_signature", _source_signature(modules))

    @property
    def output_dims(self) -> tuple[int, ...]:
        return self._output_dims

    @property
    def packed_nbytes(self) -> int:
        return self._packed_nbytes

    def matches_sources(self, modules: Sequence[Any]) -> bool:
        return _same_source_signature(
            self._source_signature,
            _source_signature(modules),
        )

    def detach_source_views(self) -> None:
        """Copy installed row views before the parent allocation is discarded."""

        replacements: list[tuple[Any, str, mx.array]] = []
        for signature in self._source_signature:
            module = signature[0]
            for name, expected in zip(_ARRAY_NAMES, signature[1:5], strict=True):
                current = _array_parameter(module, name)
                if current is expected and current is not None:
                    copied = mx.bitwise_xor(
                        current.view(mx.uint8),
                        mx.array(0, dtype=mx.uint8),
                    ).view(current.dtype)
                    replacements.append((module, name, copied))

        if replacements:
            mx.eval(*(value for _, _, value in replacements))
        for module, name, value in replacements:
            setattr(module, name, value)

    def __call__(self, x: mx.array) -> tuple[mx.array, ...]:
        if x.ndim != 3 or x.shape[0] * x.shape[1] != 1:
            raise PackedKDASkinnyUnsupported(
                "packed K3 KDA projections are decode-only (one total token)"
            )
        if int(x.shape[-1]) != self._input_dims:
            raise PackedKDASkinnyUnsupported(
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


class AuthoritativePackedK3KDAWide(AuthoritativePackedK3KDASkinny):
    """One exact QMV for K3's rank-local QKV and full-rank gate rows."""

    def __init__(self, modules: Sequence[Any]):
        modules = tuple(modules)
        if len(modules) != 2:
            raise PackedKDAWideUnsupported(
                f"expected QKV and gate projections, got {len(modules)}"
            )
        qkv_weight = _array_parameter(modules[0], "weight")
        gate_weight = _array_parameter(modules[1], "weight")
        if qkv_weight is None or gate_weight is None:
            raise PackedKDAWideUnsupported(
                "QKV and gate projections must expose array weights"
            )
        if qkv_weight.ndim != 2 or gate_weight.ndim != 2:
            raise PackedKDAWideUnsupported(
                "QKV and gate packed weights must be two-dimensional"
            )
        if int(qkv_weight.shape[0]) != 3 * int(gate_weight.shape[0]):
            raise PackedKDAWideUnsupported(
                "rank-local QKV rows must be exactly three times gate rows"
            )
        super().__init__(modules)


def _skinny_modules(attention: Any) -> tuple[Any, ...]:
    required = ["f_a_proj"]
    if not bool(getattr(attention, "use_full_rank_gate", False)):
        required.append("g_a_proj")
    required.append("b_proj")
    missing = [name for name in required if not hasattr(attention, name)]
    if missing:
        raise PackedKDASkinnyUnsupported(
            f"KDA attention is missing projections: {missing}"
        )
    return tuple(getattr(attention, name) for name in required)


def _wide_modules(attention: Any) -> tuple[Any, Any]:
    if not bool(getattr(attention, "use_full_rank_gate", False)):
        raise PackedKDAWideUnsupported(
            "wide KDA packing requires Kimi K3's full-rank gate"
        )
    required = ("qkv_proj", "g_proj")
    missing = [name for name in required if not hasattr(attention, name)]
    if missing:
        raise PackedKDAWideUnsupported(
            f"KDA attention is missing projections: {missing}"
        )
    return tuple(getattr(attention, name) for name in required)


def _drop_packed_kda_skinny(attention: Any) -> None:
    for name in (
        "_authoritative_packed_kda_skinny",
        "_authoritative_packed_kda_skinny_reason",
        "_authoritative_packed_kda_skinny_source_signature",
    ):
        if hasattr(attention, name):
            object.__delattr__(attention, name)


def invalidate_packed_k3_kda_skinny(attention: Any) -> None:
    """Drop hidden packed state before sharding or explicit weight mutation."""

    packed = getattr(attention, "_authoritative_packed_kda_skinny", None)
    if isinstance(packed, AuthoritativePackedK3KDASkinny):
        packed.detach_source_views()
    _drop_packed_kda_skinny(attention)


def _drop_packed_kda_wide(attention: Any) -> None:
    for name in (
        "_authoritative_packed_kda_wide",
        "_authoritative_packed_kda_wide_reason",
        "_authoritative_packed_kda_wide_source_signature",
    ):
        if hasattr(attention, name):
            object.__delattr__(attention, name)


def invalidate_packed_k3_kda_wide(attention: Any) -> None:
    """Drop the authoritative wide pack before sharding or weight mutation."""

    packed = getattr(attention, "_authoritative_packed_kda_wide", None)
    if isinstance(packed, AuthoritativePackedK3KDAWide):
        packed.detach_source_views()
    _drop_packed_kda_wide(attention)


def maybe_authoritative_packed_k3_kda_skinny(
    attention: Any,
    x: mx.array,
) -> tuple[mx.array, ...] | None:
    """Return exact packed skinny projections, or ``None`` for the stock path."""

    if (
        not packed_kda_skinny_enabled()
        or getattr(attention, "training", True)
        or x.ndim != 3
        or x.shape[0] * x.shape[1] != 1
    ):
        return None

    try:
        modules = _skinny_modules(attention)
    except (AttributeError, PackedKDASkinnyUnsupported):
        return None

    packed = getattr(attention, "_authoritative_packed_kda_skinny", None)
    stale_packed = None
    current_signature = _source_signature(modules)
    if packed is _UNSUPPORTED:
        unsupported_signature = getattr(
            attention,
            "_authoritative_packed_kda_skinny_source_signature",
            (),
        )
        if _same_source_signature(unsupported_signature, current_signature):
            return None
        _drop_packed_kda_skinny(attention)
        packed = None
    elif packed is not None:
        try:
            if not packed.matches_sources(modules):
                stale_packed = packed
                _drop_packed_kda_skinny(attention)
                packed = None
        except (AttributeError, TypeError, ValueError):
            if isinstance(packed, AuthoritativePackedK3KDASkinny):
                stale_packed = packed
            _drop_packed_kda_skinny(attention)
            packed = None

    if packed is None:
        try:
            packed = AuthoritativePackedK3KDASkinny(modules)
        except (
            AttributeError,
            PackedKDASkinnyUnsupported,
            TypeError,
            ValueError,
        ) as exc:
            if isinstance(stale_packed, AuthoritativePackedK3KDASkinny):
                stale_packed.detach_source_views()
                current_signature = _source_signature(modules)
            object.__setattr__(
                attention,
                "_authoritative_packed_kda_skinny_reason",
                str(exc),
            )
            object.__setattr__(
                attention,
                "_authoritative_packed_kda_skinny",
                _UNSUPPORTED,
            )
            object.__setattr__(
                attention,
                "_authoritative_packed_kda_skinny_source_signature",
                current_signature,
            )
            return None
        object.__setattr__(
            attention,
            "_authoritative_packed_kda_skinny",
            packed,
        )

    try:
        return packed(x)
    except PackedKDASkinnyUnsupported:
        return None


def maybe_authoritative_packed_k3_kda_wide(
    attention: Any,
    x: mx.array,
) -> tuple[mx.array, mx.array] | None:
    """Return exact packed QKV/gate projections, or the stock fallback."""

    if (
        not packed_kda_wide_enabled()
        or getattr(attention, "training", True)
        or x.ndim != 3
        or x.shape[0] * x.shape[1] != 1
    ):
        return None

    try:
        modules = _wide_modules(attention)
    except (AttributeError, PackedKDAWideUnsupported):
        return None

    packed = getattr(attention, "_authoritative_packed_kda_wide", None)
    stale_packed = None
    current_signature = _source_signature(modules)
    if packed is _UNSUPPORTED:
        unsupported_signature = getattr(
            attention,
            "_authoritative_packed_kda_wide_source_signature",
            (),
        )
        if _same_source_signature(unsupported_signature, current_signature):
            return None
        _drop_packed_kda_wide(attention)
        packed = None
    elif packed is not None:
        try:
            if not packed.matches_sources(modules):
                stale_packed = packed
                _drop_packed_kda_wide(attention)
                packed = None
        except (AttributeError, TypeError, ValueError):
            if isinstance(packed, AuthoritativePackedK3KDAWide):
                stale_packed = packed
            _drop_packed_kda_wide(attention)
            packed = None

    if packed is None:
        try:
            packed = AuthoritativePackedK3KDAWide(modules)
        except (
            AttributeError,
            PackedKDASkinnyUnsupported,
            TypeError,
            ValueError,
        ) as exc:
            if isinstance(stale_packed, AuthoritativePackedK3KDAWide):
                stale_packed.detach_source_views()
                current_signature = _source_signature(modules)
            object.__setattr__(
                attention,
                "_authoritative_packed_kda_wide_reason",
                str(exc),
            )
            object.__setattr__(
                attention,
                "_authoritative_packed_kda_wide",
                _UNSUPPORTED,
            )
            object.__setattr__(
                attention,
                "_authoritative_packed_kda_wide_source_signature",
                current_signature,
            )
            return None
        object.__setattr__(
            attention,
            "_authoritative_packed_kda_wide",
            packed,
        )

    try:
        qkv, gate = packed(x)
        return qkv, gate
    except PackedKDASkinnyUnsupported:
        return None
