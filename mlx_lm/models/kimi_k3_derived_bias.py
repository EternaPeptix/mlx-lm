"""Fail-closed contract for Kimi K3's derived affine-2bit biases.

The K3 UVMAX routed expert banks store BF16 affine metadata with an exact
``bias == -2 * scale`` relationship.  Kernels may skip the bias load only
after the complete incoming metadata has passed a bitwise check.  A separate
opt-in may then alias each validated bias parameter to its scale parameter and
use MLX's inference-only ``affine2`` gather mode, reclaiming the raw allocation
without changing non-exact fallback banks.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Mapping, Sequence

import mlx.core as mx

DERIVE_AFFINE2_BIAS_ENV = "MLX_LM_KIMI_K3_DERIVE_AFFINE2_BIAS"
ELIDE_AFFINE2_BIAS_ENV = "MLX_LM_KIMI_K3_ELIDE_AFFINE2_BIAS"
_VALIDATED_ATTR = "_k3_affine2_derived_bias_validated"
_ELIDED_ATTR = "_k3_affine2_bias_elided"
_ELISION_AUTH_ATTR = "_k3_affine2_elision_authorizations"
_RUNTIME_MODE_ATTR = "_runtime_quantization_mode"
_STRICT_PROJECTIONS = ("gate_proj", "up_proj")
_SELECTIVE_PROJECTIONS = ("down_proj",)


@lru_cache(maxsize=1)
def derive_affine2_bias_enabled() -> bool:
    """Parse the K3-only opt-in; values other than exact ``0``/``1`` fail."""

    value = os.environ.get(DERIVE_AFFINE2_BIAS_ENV, "0")
    if value not in {"0", "1"}:
        raise ValueError(f"{DERIVE_AFFINE2_BIAS_ENV} must be exactly '0' or '1'")
    return value == "1"


@lru_cache(maxsize=1)
def elide_affine2_bias_enabled() -> bool:
    """Parse the allocation-elision opt-in and require derived execution."""

    value = os.environ.get(ELIDE_AFFINE2_BIAS_ENV, "0")
    if value not in {"0", "1"}:
        raise ValueError(f"{ELIDE_AFFINE2_BIAS_ENV} must be exactly '0' or '1'")
    if value == "1" and not derive_affine2_bias_enabled():
        raise ValueError(
            f"{ELIDE_AFFINE2_BIAS_ENV}=1 requires {DERIVE_AFFINE2_BIAS_ENV}=1"
        )
    return value == "1"


@lru_cache(maxsize=1)
def affine2_gather_core_available() -> bool:
    """Evaluate a tiny graph so parser-only or missing-kernel builds fail."""

    if not mx.metal.is_available():
        return False
    x = mx.zeros((1, 1, 512), dtype=mx.bfloat16)
    weight = mx.zeros((1, 8, 32), dtype=mx.uint32)
    scales = mx.ones((1, 8, 4), dtype=mx.bfloat16)
    indices = mx.zeros((1,), dtype=mx.uint32)
    try:
        out = mx.gather_qmm(
            x,
            weight,
            scales,
            rhs_indices=indices,
            group_size=128,
            bits=2,
            mode="affine2",
        )
        mx.eval(out)
    except (ValueError, RuntimeError):
        return False
    return True


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
    """Validate and mark routed expert projections present in a load.

    A metadata-only model construction legitimately supplies no expert
    tensors and returns zero.  Once any member of a projection is present,
    both metadata arrays are mandatory.  The check is intentionally full-array
    and bitwise; sampled validation is not sufficient authority to skip a
    runtime metadata load.  Gate/up failures remain fatal because their
    derived kernels are requested as a pair.  A down-projection mismatch is
    marked ineligible instead: its stored bias remains authoritative and only
    that module uses the incumbent down path.
    """

    if not derive_affine2_bias_enabled():
        return 0

    validated = 0
    for layer_index, layer in enumerate(layers):
        switch_mlp = getattr(getattr(layer, "mlp", None), "switch_mlp", None)
        if switch_mlp is None:
            continue
        authorizations = set()
        for projection_name in (*_STRICT_PROJECTIONS, *_SELECTIVE_PROJECTIONS):
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
            fast_derivable = affine2_biases_are_fast_derivable(scales, biases)
            if not fast_derivable and projection_name in _SELECTIVE_PROJECTIONS:
                if _target_projection(projection):
                    setattr(projection, _VALIDATED_ATTR, False)
                continue
            if not fast_derivable:
                validate_affine2_bias_relation(scales, biases, label=prefix)
                raise ValueError(
                    f"{prefix} contains a zero, subnormal, infinity, or NaN scale; "
                    "derived-bias decode requires finite-normal BF16 scales"
                )
            if _target_projection(projection):
                setattr(projection, _VALIDATED_ATTR, True)
            # Keep only object identities, never a second reference to the raw
            # bias tensor.  Standard loading assigns these exact arrays into
            # the quantized module after sanitize returns.
            authorizations.add((projection_name, id(scales), id(biases)))
            validated += 1
        setattr(switch_mlp, _ELISION_AUTH_ATTR, frozenset(authorizations))
    return validated


def _elide_validated_k3_biases(
    layers: Sequence[Any],
    *,
    after_shard: bool,
) -> int:
    """Implement load-time elision or post-shard re-aliasing.

    The authorization is produced only by the full-array validation above and
    is bound to the exact incoming MLX array objects.  Load-time finalization
    always checks those identities, including checkpoint reloads.  Post-shard
    re-entry may bypass the identities only for modules that were already
    validated and elided before the internal transform.
    """

    if not elide_affine2_bias_enabled():
        return 0
    if not affine2_gather_core_available():
        raise RuntimeError(
            "K3 affine2 bias elision requires an MLX core with Metal "
            "gather_qmm(mode='affine2') support"
        )

    elided = 0
    for layer_index, layer in enumerate(layers):
        switch_mlp = getattr(getattr(layer, "mlp", None), "switch_mlp", None)
        if switch_mlp is None:
            continue
        authorizations = getattr(switch_mlp, _ELISION_AUTH_ATTR, frozenset())
        for projection_name, scales_id, biases_id in authorizations:
            projection = getattr(switch_mlp, projection_name)
            prefix = f"model.layers.{layer_index}.mlp.switch_mlp.{projection_name}"
            already_elided = getattr(projection, _ELIDED_ATTR, None) is True
            if after_shard and not already_elided:
                continue
            if not _target_projection(projection):
                raise RuntimeError(
                    f"{prefix} changed after affine2 validation and cannot elide bias"
                )
            scales = projection.get("scales")
            biases = projection.get("biases")
            if (
                not isinstance(scales, mx.array)
                or not isinstance(biases, mx.array)
                or scales.dtype != mx.bfloat16
                or biases.dtype != mx.bfloat16
                or scales.shape != biases.shape
            ):
                raise RuntimeError(
                    f"{prefix} no longer has matching BF16 affine metadata"
                )
            if not after_shard and (
                id(scales) != scales_id or id(biases) != biases_id
            ):
                raise RuntimeError(
                    f"{prefix} loaded different metadata than the validated arrays"
                )
            if after_shard and (
                getattr(projection, _VALIDATED_ATTR, None) is not True
                or getattr(projection, _RUNTIME_MODE_ATTR, None) != "affine2"
            ):
                raise RuntimeError(
                    f"{prefix} lost its validated affine2 state during sharding"
                )

            # Preserve the six-input affine primitive ABI with one allocation:
            # MLX core binds scales into the bias slot as an affine2 scale alias.
            projection.biases = projection.scales
            setattr(projection, _VALIDATED_ATTR, True)
            setattr(projection, _ELIDED_ATTR, True)
            setattr(projection, _RUNTIME_MODE_ATTR, "affine2")
            elided += 1
    return elided


def elide_validated_k3_biases(layers: Sequence[Any]) -> int:
    """Alias only the exact arrays authorized by the current checkpoint load."""

    return _elide_validated_k3_biases(layers, after_shard=False)


def reelide_sharded_k3_biases(layers: Sequence[Any]) -> int:
    """Re-establish aliases only for modules elided before an internal shard."""

    return _elide_validated_k3_biases(layers, after_shard=True)


__all__ = [
    "DERIVE_AFFINE2_BIAS_ENV",
    "ELIDE_AFFINE2_BIAS_ENV",
    "affine2_gather_core_available",
    "affine2_bias_relation_is_exact",
    "affine2_biases_are_fast_derivable",
    "derive_affine2_bias_enabled",
    "derived_affine2_biases",
    "elide_affine2_bias_enabled",
    "elide_validated_k3_biases",
    "projection_has_validated_derived_bias",
    "reelide_sharded_k3_biases",
    "validate_affine2_bias_relation",
    "validate_k3_biases_for_load",
]
