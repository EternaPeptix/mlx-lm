"""Fail-closed contract for Kimi K3's derived affine-2bit biases.

The K3 UVMAX routed expert banks store BF16 affine metadata with an exact
``bias == -2 * scale`` relationship.  Decode kernels may skip the bias load
only after the complete incoming metadata has passed a bitwise check.  The
raw bias tensors remain part of the model so stock, prefill, and fallback
paths are unchanged.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Mapping, Sequence

import mlx.core as mx

DERIVE_AFFINE2_BIAS_ENV = "MLX_LM_KIMI_K3_DERIVE_AFFINE2_BIAS"
_VALIDATED_ATTR = "_k3_affine2_derived_bias_validated"
_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


@lru_cache(maxsize=1)
def derive_affine2_bias_enabled() -> bool:
    """Parse the K3-only opt-in; values other than exact ``0``/``1`` fail."""

    value = os.environ.get(DERIVE_AFFINE2_BIAS_ENV, "0")
    if value not in {"0", "1"}:
        raise ValueError(f"{DERIVE_AFFINE2_BIAS_ENV} must be exactly '0' or '1'")
    return value == "1"


def derived_affine2_biases(scales: mx.array) -> mx.array:
    """Derive exact BF16 bits without flushing signed subnormal values."""

    if scales.dtype != mx.bfloat16:
        raise TypeError("K3 derived affine-2bit scales must be BF16")
    raw = scales.view(mx.uint16).astype(mx.uint32)
    magnitude = raw & mx.array(0x7FFF, dtype=mx.uint32)
    exponent = magnitude & mx.array(0x7F80, dtype=mx.uint32)
    doubled_magnitude = mx.where(
        exponent == 0,
        magnitude << 1,
        mx.where(
            exponent < mx.array(0x7F00, dtype=mx.uint32),
            magnitude + mx.array(0x0080, dtype=mx.uint32),
            mx.where(
                exponent == mx.array(0x7F00, dtype=mx.uint32),
                mx.array(0x7F80, dtype=mx.uint32),
                magnitude,
            ),
        ),
    )
    flipped_sign = (raw ^ mx.array(0x8000, dtype=mx.uint32)) & mx.array(
        0x8000, dtype=mx.uint32
    )
    return (flipped_sign | doubled_magnitude).astype(mx.uint16).view(mx.bfloat16)


def affine2_bias_relation_is_exact(
    scales: mx.array,
    biases: mx.array,
) -> bool:
    """Check shape, dtype, and every BF16 bit, including signed zero."""

    if (
        scales.dtype != mx.bfloat16
        or biases.dtype != mx.bfloat16
        or scales.shape != biases.shape
    ):
        return False
    scale_bits = scales.view(mx.uint16).astype(mx.uint32)
    exponent = scale_bits & mx.array(0x7F80, dtype=mx.uint32)
    fraction = scale_bits & mx.array(0x007F, dtype=mx.uint32)
    no_nan = (exponent != mx.array(0x7F80, dtype=mx.uint32)) | (fraction == 0)
    expected_bits = derived_affine2_biases(scales).view(mx.uint16)
    return bool(mx.all(no_nan & (expected_bits == biases.view(mx.uint16))).item())


def affine2_biases_are_fast_derivable(
    scales: mx.array,
    biases: mx.array,
) -> bool:
    """Require exact metadata plus finite-normal scales for the fast kernel.

    Metal may flush BF16 subnormal values when they are promoted for the
    floating-point multiply.  Zeros and infinities are also excluded so the
    optimized kernel has one uniform arithmetic contract.  These values do
    not occur in the audited K3 UVMAX banks; alternate metadata falls back.
    """

    if (
        scales.dtype != mx.bfloat16
        or biases.dtype != mx.bfloat16
        or scales.shape != biases.shape
    ):
        return False
    bits = scales.view(mx.uint16).astype(mx.uint32)
    exponent = bits & mx.array(0x7F80, dtype=mx.uint32)
    expected_bits = derived_affine2_biases(scales).view(mx.uint16)
    return bool(
        mx.all(
            (exponent != 0)
            & (exponent != mx.array(0x7F80, dtype=mx.uint32))
            & (expected_bits == biases.view(mx.uint16))
        ).item()
    )


def validate_affine2_bias_relation(
    scales: mx.array,
    biases: mx.array,
    *,
    label: str,
) -> None:
    """Raise before derived-bias execution when one metadata bit differs."""

    if not affine2_bias_relation_is_exact(scales, biases):
        raise ValueError(f"{label} violates the exact BF16 bias == -2 * scale contract")


def projection_has_validated_derived_bias(
    projection: Any,
    scales: mx.array,
    biases: mx.array,
) -> bool:
    """Validate an ad-hoc projection once, falling back safely on failure.

    Production K3 loads are validated in :func:`validate_k3_biases_for_load`
    and arrive with the positive marker.  This runtime guard covers manually
    constructed modules and alternate loaders without risking derived-bias
    execution on unchecked metadata.
    """

    status = getattr(projection, _VALIDATED_ATTR, None)
    if status is not None:
        return status is True
    valid = affine2_biases_are_fast_derivable(scales, biases)
    setattr(projection, _VALIDATED_ATTR, valid)
    return valid


def _target_projection(module: Any) -> bool:
    """Return whether a loaded module is the exact K3 affine2/group128 type."""

    return (
        getattr(module, "bits", None) == 2
        and getattr(module, "group_size", None) == 128
        and getattr(module, "mode", None) == "affine"
        and "bias" not in module
    )


def validate_k3_biases_for_load(
    layers: Sequence[Any],
    weights: Mapping[str, mx.array],
) -> int:
    """Validate and mark every routed expert projection present in a load.

    A metadata-only model construction legitimately supplies no expert
    tensors and returns zero.  Once any member of a projection is present,
    both metadata arrays are mandatory and a mismatch fails the load.  The
    check is intentionally full-array and bitwise; sampled validation is not
    sufficient authority to skip a runtime metadata load.
    """

    if not derive_affine2_bias_enabled():
        return 0

    validated = 0
    for layer_index, layer in enumerate(layers):
        switch_mlp = getattr(getattr(layer, "mlp", None), "switch_mlp", None)
        if switch_mlp is None:
            continue
        for projection_name in _PROJECTIONS:
            projection = getattr(switch_mlp, projection_name)
            prefix = f"model.layers.{layer_index}.mlp.switch_mlp.{projection_name}"
            scales = weights.get(f"{prefix}.scales")
            biases = weights.get(f"{prefix}.biases")
            if scales is None and biases is None:
                continue
            if scales is None or biases is None:
                raise ValueError(
                    f"{prefix} must provide both scales and biases for derived-bias "
                    "validation"
                )
            if not _target_projection(projection):
                # Stock loading quantizes structure after sanitize.  The
                # incoming K3 keys are still validated here, while the runtime
                # guard will mark the replacement QuantizedSwitchLinear.
                if any(
                    hasattr(projection, name) for name in ("bits", "group_size", "mode")
                ):
                    raise ValueError(
                        f"{prefix} is not affine 2-bit/group-128 without output bias"
                    )
            if not affine2_biases_are_fast_derivable(scales, biases):
                validate_affine2_bias_relation(scales, biases, label=prefix)
                raise ValueError(
                    f"{prefix} contains a zero, subnormal, infinity, or NaN scale; "
                    "derived-bias decode requires finite-normal BF16 scales"
                )
            if _target_projection(projection):
                setattr(projection, _VALIDATED_ATTR, True)
            validated += 1
    return validated


__all__ = [
    "DERIVE_AFFINE2_BIAS_ENV",
    "affine2_bias_relation_is_exact",
    "affine2_biases_are_fast_derivable",
    "derive_affine2_bias_enabled",
    "derived_affine2_biases",
    "projection_has_validated_derived_bias",
    "validate_affine2_bias_relation",
    "validate_k3_biases_for_load",
]
