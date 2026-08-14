"""Opt-in exact packing for Kimi K3's decode-time MoE front projections.

Kimi K3 evaluates four affine-quantized projections from the same residual
vector before its routed and shared experts:

* shared-expert gate;
* shared-expert up;
* router scores; and
* routed-expert latent down.

At the single-token decode shape, concatenating their already-quantized output
rows lets MLX issue one QMV instead of four while preserving every output bit.
An independent authoritative-only gate admits the exact ``(1, 3, 7168)`` K3
target-verification shape for a full-pack experiment.  This is intentionally
separate from split packing: the full pack evaluates both the shared and routed
front projections in one QMM.  The branches share that front dependency, but
the model keeps their routed and shared reductions independent afterward so
the shared down projection can still overlap routed communication.
No weight is dequantized or requantized.  An independent, default-off gate can
admit the exact width-eight K3 target-verification shape after its QMM kernel
has been validated; every other multi-token call remains on the stock path.
"""

from __future__ import annotations

import os
from contextvars import ContextVar
from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import accumulate
from threading import Lock
from typing import Any, Sequence

import mlx.core as mx
import mlx.nn as nn

PACKED_MOE_FRONT_ENV = "MLX_LM_KIMI_K3_PACKED_MOE_FRONT"
AUTHORITATIVE_PACKED_MOE_FRONT_ENV = "MLX_LM_KIMI_K3_AUTHORITATIVE_PACKED_MOE_FRONT"
AUTHORITATIVE_PACKED_MOE_FRONT_WIDTH3_ENV = (
    "MLX_LM_KIMI_K3_AUTHORITATIVE_PACKED_MOE_FRONT_WIDTH3"
)
AUTHORITATIVE_PACKED_MOE_FRONT_RECEIPT_ENV = (
    "MLX_LM_KIMI_K3_AUTHORITATIVE_PACKED_MOE_FRONT_RECEIPT"
)
PACKED_MOE_FRONT_WIDTH8_ENV = "MLX_LM_KIMI_K3_PACKED_MOE_FRONT_WIDTH8"
AUTHORITATIVE_PACKED_MOE_FRONT_RECEIPT_SCHEMA = (
    "kimi-k3-authoritative-packed-moe-front-receipt/v1"
)
_UNSUPPORTED = object()
_PACKED_ARRAY_NAMES = ("weight", "scales", "biases", "bias")
_WIDTH3_INPUT_DIMS = 7168
_WIDTH3_OUTPUT_DIMS = (3072, 3072, 896, 3584)
_RECEIPT_COUNTER_LIMIT = 1_000_000_000
_RECEIPT_SEQUENCE_LIMIT = 0x7FFFFFFFFFFFFFFF
_RECEIPT_SEQUENCE = 0
_RECEIPT_SEQUENCE_LOCK = Lock()


@dataclass(frozen=True)
class _PackedFrontReceiptState:
    request_sequence: int
    request_token: int
    pack_count_before: int
    authoritative_gate_enabled: bool
    width3_gate_enabled: bool
    helper_calls: int = 0
    packed_hits: int = 0
    packed_output_tensors: int = 0
    lazy_installs: int = 0
    gate_disabled_calls: int = 0
    noncontract_calls: int = 0
    unsupported_calls: int = 0
    packed_dispatch_fallback_calls: int = 0
    invalidations: int = 0
    stale_resets: int = 0


_RECEIPT_STATE: ContextVar[_PackedFrontReceiptState | None] = ContextVar(
    "kimi_k3_authoritative_packed_moe_front_receipt",
    default=None,
)


def authoritative_packed_moe_front_receipt_enabled() -> bool:
    """Parse the strict, default-off request receipt selector."""

    value = os.environ.get(AUTHORITATIVE_PACKED_MOE_FRONT_RECEIPT_ENV, "0")
    if value not in {"0", "1"}:
        raise ValueError(f"{AUTHORITATIVE_PACKED_MOE_FRONT_RECEIPT_ENV} must be 0 or 1")
    return value == "1"


def _strict_receipt_gate(name: str) -> bool:
    value = os.environ.get(name, "0")
    if value not in {"0", "1"}:
        raise ValueError(f"{name} must be 0 or 1 while receipt capture is active")
    return value == "1"


def _next_receipt_sequence() -> int:
    global _RECEIPT_SEQUENCE

    with _RECEIPT_SEQUENCE_LOCK:
        if _RECEIPT_SEQUENCE >= _RECEIPT_SEQUENCE_LIMIT:
            raise OverflowError("packed-front receipt sequence exhausted")
        _RECEIPT_SEQUENCE += 1
        return _RECEIPT_SEQUENCE


def _validate_request_binding(request_sequence: int, request_token: int) -> None:
    if (
        type(request_sequence) is not int
        or not 1 <= request_sequence <= _RECEIPT_SEQUENCE_LIMIT
    ):
        raise ValueError("packed-front receipt sequence must fit positive int64")
    if (
        type(request_token) is not int
        or not 0 <= request_token <= _RECEIPT_SEQUENCE_LIMIT
    ):
        raise ValueError("packed-front receipt token must fit nonnegative int64")


def _count_model_authoritative_width3_packs(model: Any) -> int:
    """Read the exact currently installed production-width3 parent count."""

    language_model = getattr(model, "language_model", None)
    inner_model = getattr(language_model, "model", None)
    layers = getattr(inner_model, "layers", ())
    try:
        iterator = iter(layers)
    except TypeError as error:
        raise TypeError("packed-front receipt model layers are not iterable") from error

    count = 0
    for layer in iterator:
        sparse_moe = getattr(layer, "mlp", None)
        packed = getattr(
            sparse_moe,
            "_authoritative_packed_k3_moe_front",
            None,
        )
        if not isinstance(packed, AuthoritativePackedK3MoEFront):
            continue
        try:
            modules = _front_modules(sparse_moe)
            active = (
                packed._input_dims == _WIDTH3_INPUT_DIMS
                and packed._output_dims == _WIDTH3_OUTPUT_DIMS
                and packed._production_width3_source_layout
                and _production_width3_source_layout_supported(modules)
                and packed.matches_sources(modules)
            )
        except (AttributeError, PackedMoEFrontUnsupported, TypeError, ValueError):
            active = False
        if active:
            count += 1
            if count > _RECEIPT_COUNTER_LIMIT:
                raise OverflowError(
                    "packed-front receipt pack count exceeded its bound"
                )
    return count


def begin_authoritative_packed_moe_front_receipt(
    request_token: int,
    model: Any,
) -> tuple[int, int]:
    """Begin one context-local receipt and snapshot installed parents."""

    # Clear first so malformed configuration or model traversal cannot revive
    # counters from an abandoned generator in a reused execution context.
    _RECEIPT_STATE.set(None)
    if not authoritative_packed_moe_front_receipt_enabled():
        raise RuntimeError("packed-front receipt capture is disabled")
    _validate_request_binding(1, request_token)
    request_sequence = _next_receipt_sequence()
    authoritative_gate_enabled = _strict_receipt_gate(
        AUTHORITATIVE_PACKED_MOE_FRONT_ENV
    )
    width3_gate_enabled = _strict_receipt_gate(
        AUTHORITATIVE_PACKED_MOE_FRONT_WIDTH3_ENV
    )
    if authoritative_gate_enabled != width3_gate_enabled:
        raise ValueError(
            "packed-front receipt requires authoritative and width3 gates "
            "to be jointly disabled or jointly enabled"
        )
    state = _PackedFrontReceiptState(
        request_sequence=request_sequence,
        request_token=request_token,
        pack_count_before=_count_model_authoritative_width3_packs(model),
        authoritative_gate_enabled=authoritative_gate_enabled,
        width3_gate_enabled=width3_gate_enabled,
    )
    _RECEIPT_STATE.set(state)
    return request_sequence, request_token


def _receipt_state_for_binding(
    request_sequence: int,
    request_token: int,
) -> _PackedFrontReceiptState:
    _validate_request_binding(request_sequence, request_token)
    state = _RECEIPT_STATE.get()
    if state is None:
        raise RuntimeError("no packed-front receipt is active")
    if (
        state.request_sequence != request_sequence
        or state.request_token != request_token
    ):
        raise RuntimeError("packed-front receipt request binding does not match")
    return state


def _bounded_receipt_sum(current: int, amount: int) -> int:
    if type(amount) is not int or amount < 0:
        raise ValueError("packed-front receipt increments must be nonnegative ints")
    updated = current + amount
    if updated > _RECEIPT_COUNTER_LIMIT:
        raise OverflowError("packed-front receipt counter exceeded its bound")
    return updated


def _increment_receipt(**increments: int) -> None:
    state = _RECEIPT_STATE.get()
    if state is None:
        return
    changes: dict[str, int] = {}
    for name, amount in increments.items():
        if name not in _PackedFrontReceiptState.__dataclass_fields__:
            raise KeyError(f"unknown packed-front receipt counter: {name}")
        current = getattr(state, name)
        if type(current) is not int:
            raise TypeError(f"packed-front receipt field {name} is not a counter")
        changes[name] = _bounded_receipt_sum(current, amount)
    _RECEIPT_STATE.set(replace(state, **changes))


def _receipt_helper_started(sparse_moe: Any, x: mx.array, gate_enabled: bool) -> bool:
    """Account one helper call and return whether it is exact width-three."""

    state = _RECEIPT_STATE.get()
    if state is None:
        return False
    current_gate = _strict_receipt_gate(AUTHORITATIVE_PACKED_MOE_FRONT_ENV)
    current_width3_gate = _strict_receipt_gate(
        AUTHORITATIVE_PACKED_MOE_FRONT_WIDTH3_ENV
    )
    if (
        gate_enabled != current_gate
        or current_gate != state.authoritative_gate_enabled
        or current_width3_gate != state.width3_gate_enabled
    ):
        raise RuntimeError("packed-front selectors changed during receipt capture")
    _increment_receipt(helper_calls=1)
    if not current_gate:
        _increment_receipt(gate_disabled_calls=1)
        return False
    exact_width3 = (
        not getattr(sparse_moe, "training", True)
        and current_width3_gate
        and x.ndim == 3
        and tuple(int(dim) for dim in x.shape) == (1, 3, _WIDTH3_INPUT_DIMS)
        and x.dtype == mx.bfloat16
    )
    if not exact_width3:
        _increment_receipt(noncontract_calls=1)
    return exact_width3


def _receipt_width3_outcome(
    outcome: str,
    *,
    output_tensors: int = 0,
) -> None:
    state = _RECEIPT_STATE.get()
    if state is None:
        return
    fields = {
        "packed_hit": "packed_hits",
        "unsupported": "unsupported_calls",
        "packed_dispatch_fallback": "packed_dispatch_fallback_calls",
    }
    field = fields.get(outcome)
    if field is None:
        raise KeyError(f"unknown packed-front receipt outcome: {outcome}")
    increments = {field: 1}
    if outcome == "packed_hit":
        increments["packed_output_tensors"] = output_tensors
    _increment_receipt(**increments)


def finish_authoritative_packed_moe_front_receipt(
    request_sequence: int,
    request_token: int,
    model: Any,
) -> dict[str, str | int | bool]:
    """Finalize, deactivate, and return one exact scalar-only receipt."""

    state = _receipt_state_for_binding(request_sequence, request_token)
    _RECEIPT_STATE.set(None)
    pack_count_after = _count_model_authoritative_width3_packs(model)
    terminal_total = (
        state.gate_disabled_calls
        + state.noncontract_calls
        + state.packed_hits
        + state.unsupported_calls
        + state.packed_dispatch_fallback_calls
    )
    if terminal_total != state.helper_calls:
        raise RuntimeError("packed-front receipt helper partition is incomplete")
    return {
        "schema": AUTHORITATIVE_PACKED_MOE_FRONT_RECEIPT_SCHEMA,
        "request_sequence": state.request_sequence,
        "request_token": state.request_token,
        "finalized": True,
        "authoritative_gate_enabled": state.authoritative_gate_enabled,
        "width3_gate_enabled": state.width3_gate_enabled,
        "helper_calls": state.helper_calls,
        "packed_hits": state.packed_hits,
        "packed_output_tensors": state.packed_output_tensors,
        "lazy_installs": state.lazy_installs,
        "gate_disabled_calls": state.gate_disabled_calls,
        "noncontract_calls": state.noncontract_calls,
        "unsupported_calls": state.unsupported_calls,
        "packed_dispatch_fallback_calls": state.packed_dispatch_fallback_calls,
        "invalidations": state.invalidations,
        "stale_resets": state.stale_resets,
        "pack_count_before": state.pack_count_before,
        "pack_count_after": pack_count_after,
    }


def abort_authoritative_packed_moe_front_receipt(
    request_sequence: int,
    request_token: int,
) -> None:
    """Deactivate one receipt without publishing positive telemetry."""

    try:
        _receipt_state_for_binding(request_sequence, request_token)
    finally:
        _RECEIPT_STATE.set(None)


class PackedMoEFrontUnsupported(ValueError):
    """Raised when projections cannot be packed without changing semantics."""


@lru_cache(maxsize=1)
def packed_moe_front_enabled() -> bool:
    return os.environ.get(PACKED_MOE_FRONT_ENV, "0") == "1"


@lru_cache(maxsize=1)
def authoritative_packed_moe_front_enabled() -> bool:
    return os.environ.get(AUTHORITATIVE_PACKED_MOE_FRONT_ENV, "0") == "1"


@lru_cache(maxsize=1)
def authoritative_packed_moe_front_width3_enabled() -> bool:
    value = os.environ.get(AUTHORITATIVE_PACKED_MOE_FRONT_WIDTH3_ENV, "0")
    if value not in {"0", "1"}:
        raise ValueError(f"{AUTHORITATIVE_PACKED_MOE_FRONT_WIDTH3_ENV} must be 0 or 1")
    return value == "1"


@lru_cache(maxsize=1)
def packed_moe_front_width8_enabled() -> bool:
    return os.environ.get(PACKED_MOE_FRONT_WIDTH8_ENV, "0") == "1"


def _packed_input_supported(x: mx.array) -> bool:
    if x.ndim != 3 or x.shape[0] != 1:
        return False
    width = int(x.shape[1])
    return width == 1 or (width == 8 and packed_moe_front_width8_enabled())


def _authoritative_packed_input_supported(x: mx.array) -> bool:
    if _packed_input_supported(x):
        return True
    return (
        x.ndim == 3
        and tuple(int(dim) for dim in x.shape) == (1, 3, 7168)
        and x.dtype == mx.bfloat16
        and authoritative_packed_moe_front_width3_enabled()
    )


def _production_width3_source_layout_supported(modules: Sequence[Any]) -> bool:
    """Return whether four sources match the released TP2 affine8 front."""

    if len(modules) != len(_WIDTH3_OUTPUT_DIMS):
        return False
    for module, output_dims in zip(modules, _WIDTH3_OUTPUT_DIMS, strict=True):
        weight = _array_parameter(module, "weight")
        scales = _array_parameter(module, "scales")
        biases = _array_parameter(module, "biases")
        if weight is None or scales is None or biases is None:
            return False
        if _array_parameter(module, "bias") is not None:
            return False
        if (
            int(getattr(module, "group_size", 0)) != 64
            or int(getattr(module, "bits", 0)) != 8
            or str(getattr(module, "mode", "")) != "affine"
            or weight.dtype != mx.uint32
            or scales.dtype != mx.bfloat16
            or biases.dtype != mx.bfloat16
        ):
            return False
        if tuple(int(dim) for dim in weight.shape) != (
            output_dims,
            _WIDTH3_INPUT_DIMS * 8 // 32,
        ):
            return False
        expected_affine_shape = (output_dims, _WIDTH3_INPUT_DIMS // 64)
        if tuple(int(dim) for dim in scales.shape) != expected_affine_shape:
            return False
        if tuple(int(dim) for dim in biases.shape) != expected_affine_shape:
            return False
    return True


def production_width3_authoritative_front_active(
    sparse_moe: Any,
    x: mx.array,
    optimized_front: Any,
) -> bool:
    """Prove the exact width-three full-pack contract used by MOK overlap."""

    if (
        not isinstance(optimized_front, tuple)
        or len(optimized_front) != len(_WIDTH3_OUTPUT_DIMS)
        or x.ndim != 3
        or tuple(int(dim) for dim in x.shape) != (1, 3, _WIDTH3_INPUT_DIMS)
        or x.dtype != mx.bfloat16
        or not authoritative_packed_moe_front_width3_enabled()
    ):
        return False
    for output, output_dims in zip(
        optimized_front,
        _WIDTH3_OUTPUT_DIMS,
        strict=True,
    ):
        if not isinstance(output, mx.array) or tuple(
            int(dim) for dim in output.shape
        ) != (1, 3, output_dims):
            return False
    try:
        modules = _front_modules(sparse_moe)
    except (AttributeError, PackedMoEFrontUnsupported):
        return False
    if not _production_width3_source_layout_supported(modules):
        return False
    packed = getattr(
        sparse_moe,
        "_authoritative_packed_k3_moe_front",
        None,
    )
    try:
        return (
            packed is not None
            and packed._input_dims == _WIDTH3_INPUT_DIMS
            and packed._output_dims == _WIDTH3_OUTPUT_DIMS
            and packed._production_width3_source_layout
            and packed.matches_sources(modules)
        )
    except (AttributeError, TypeError, ValueError):
        return False


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
        object.__setattr__(self, "_output_dims", tuple(output_dims))
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
        object.__setattr__(self, "_source_signature", _source_signature(modules))
        self.freeze()

    @property
    def packed_nbytes(self) -> int:
        return self._packed_nbytes

    def matches_sources(self, modules: Sequence[Any]) -> bool:
        """Return whether every authoritative projection array is unchanged."""

        return _same_source_signature(
            self._source_signature,
            _source_signature(modules),
        )

    def _project(self, x: mx.array) -> tuple[mx.array, ...]:
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

    def __call__(self, x: mx.array) -> tuple[mx.array, ...]:
        if not _packed_input_supported(x):
            raise PackedMoEFrontUnsupported(
                "packed K3 MoE front requires one decode token or an enabled "
                "width-eight target block"
            )
        return self._project(x)


class AuthoritativePackedK3MoEFront(PackedK3MoEFront):
    """A packed QMV whose row views are the source modules' parameters.

    ``PackedK3MoEFront`` retains both the original banks and their
    concatenation.  This variant evaluates the concatenations once, replaces
    the original module arrays with zero-copy row views, and then releases the
    original arrays.  The hidden full-width arrays are therefore the only
    backing allocations while parameter traversal keeps the checkpoint's
    original names and shapes.
    """

    def __init__(self, modules: Sequence[Any]):
        modules = tuple(modules)
        production_width3_layout = _production_width3_source_layout_supported(modules)
        super().__init__(modules)
        object.__setattr__(
            self,
            "_production_width3_source_layout",
            production_width3_layout,
        )

        # Force the concatenations to finish while all source arrays are valid.
        # Packing is a one-time operation, and evaluating here bounds transient
        # duplication to the layer currently being installed.
        mx.eval(self.parameters())

        source_views: dict[str, tuple[mx.array, ...]] = {}
        for name in _PACKED_ARRAY_NAMES:
            packed_array = _array_parameter(self, name)
            if packed_array is not None:
                source_views[name] = tuple(
                    mx.split(packed_array, self._split_indices, axis=0)
                )
        mx.eval(*(value for views in source_views.values() for value in views))

        # Do not mutate a source module until every packed array and every view
        # has evaluated successfully.  Constructor failures remain fail closed.
        for index, module in enumerate(modules):
            for name, views in source_views.items():
                setattr(module, name, views[index])

        # super().__init__ captured the pre-pack arrays. Replace that signature
        # so no reference keeps the superseded allocations alive.
        object.__setattr__(self, "_source_signature", _source_signature(modules))

    def __call__(self, x: mx.array) -> tuple[mx.array, ...]:
        if not _authoritative_packed_input_supported(x):
            raise PackedMoEFrontUnsupported(
                "authoritative packed K3 MoE front requires one decode token, "
                "an enabled exact width-three target block, or an enabled "
                "width-eight target block"
            )
        if int(x.shape[1]) == 3 and (
            self._input_dims != _WIDTH3_INPUT_DIMS
            or self._output_dims != _WIDTH3_OUTPUT_DIMS
            or not self._production_width3_source_layout
        ):
            raise PackedMoEFrontUnsupported(
                "width-three authoritative packing requires the exact released "
                "TP2 front layout"
            )
        return self._project(x)

    def detach_source_views(self) -> None:
        """Give installed source views independent, byte-identical storage.

        Normal loading shards weights before the first decode, so this is only
        needed for a later call to ``Model.shard`` or explicit invalidation.
        A byte-wise xor with zero is used because array construction,
        ``copy.copy``, and contiguous conversion may all preserve the same MLX
        backing allocation.
        """

        replacements: list[tuple[Any, str, mx.array]] = []
        for signature in self._source_signature:
            module = signature[0]
            for name, expected in zip(
                _PACKED_ARRAY_NAMES,
                signature[1:5],
                strict=True,
            ):
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


def _front_modules(sparse_moe: Any) -> tuple[Any, Any, Any, Any]:
    shared = getattr(sparse_moe, "shared_experts", None)
    routed_down = getattr(sparse_moe, "routed_expert_down_proj", None)
    if shared is None or routed_down is None:
        raise PackedMoEFrontUnsupported(
            "packed path requires shared experts and latent routed experts"
        )
    return (
        shared.gate_proj,
        shared.up_proj,
        sparse_moe.gate,
        routed_down,
    )


def _build_packed_front(sparse_moe: Any) -> PackedK3MoEFront:
    return PackedK3MoEFront(_front_modules(sparse_moe))


def _build_authoritative_packed_front(
    sparse_moe: Any,
) -> AuthoritativePackedK3MoEFront:
    return AuthoritativePackedK3MoEFront(_front_modules(sparse_moe))


def _drop_authoritative_packed_k3_moe_front(sparse_moe: Any) -> None:
    for name in (
        "_authoritative_packed_k3_moe_front",
        "_authoritative_packed_k3_moe_front_reason",
        "_authoritative_packed_k3_moe_front_source_signature",
    ):
        if hasattr(sparse_moe, name):
            object.__delattr__(sparse_moe, name)


def _drop_duplicating_packed_k3_moe_front(sparse_moe: Any) -> None:
    for name in (
        "_packed_k3_moe_front",
        "_packed_k3_moe_front_reason",
        "_packed_k3_moe_front_source_signature",
    ):
        if hasattr(sparse_moe, name):
            object.__delattr__(sparse_moe, name)


def invalidate_packed_k3_moe_front(sparse_moe: Any) -> None:
    """Drop hidden packed state before sharding or other weight mutation."""

    _increment_receipt(invalidations=1)
    authoritative = getattr(
        sparse_moe,
        "_authoritative_packed_k3_moe_front",
        None,
    )
    if isinstance(authoritative, AuthoritativePackedK3MoEFront):
        authoritative.detach_source_views()
    _drop_authoritative_packed_k3_moe_front(sparse_moe)
    _drop_duplicating_packed_k3_moe_front(sparse_moe)


def maybe_authoritative_packed_k3_moe_front(
    sparse_moe: Any,
    x: mx.array,
) -> tuple[mx.array, ...] | None:
    """Return one native packed QMV backed by authoritative source views."""

    gate_enabled = authoritative_packed_moe_front_enabled()
    receipt_width3_call = _receipt_helper_started(sparse_moe, x, gate_enabled)
    if not gate_enabled:
        return None

    # Runtime experiment toggles must not leave the older duplicate cache
    # resident beside the authoritative allocation.
    _drop_duplicating_packed_k3_moe_front(sparse_moe)

    if getattr(
        sparse_moe, "training", True
    ) or not _authoritative_packed_input_supported(x):
        return None

    try:
        modules = _front_modules(sparse_moe)
    except (AttributeError, PackedMoEFrontUnsupported):
        invalidate_packed_k3_moe_front(sparse_moe)
        if receipt_width3_call:
            _receipt_width3_outcome("unsupported")
        return None
    if int(x.shape[1]) == 3 and not _production_width3_source_layout_supported(modules):
        # A previously installed authoritative parent can outlive a later
        # source mutation.  Release it before falling back so an ineligible
        # width-three call never leaves hidden packed storage resident.
        invalidate_packed_k3_moe_front(sparse_moe)
        if receipt_width3_call:
            _receipt_width3_outcome("unsupported")
        return None

    packed = getattr(
        sparse_moe,
        "_authoritative_packed_k3_moe_front",
        None,
    )
    stale_packed = None
    current_signature = _source_signature(modules)
    if packed is _UNSUPPORTED:
        unsupported_signature = getattr(
            sparse_moe,
            "_authoritative_packed_k3_moe_front_source_signature",
            (),
        )
        if _same_source_signature(unsupported_signature, current_signature):
            if receipt_width3_call:
                _receipt_width3_outcome("unsupported")
            return None
        _increment_receipt(stale_resets=1)
        _drop_authoritative_packed_k3_moe_front(sparse_moe)
        packed = None
    elif packed is not None:
        try:
            if not packed.matches_sources(modules):
                # Do not detach unchanged row views here. The replacement pack
                # can consume them directly, then atomically replace all four
                # banks and release the old parent allocation.
                stale_packed = packed
                _increment_receipt(stale_resets=1)
                _drop_authoritative_packed_k3_moe_front(sparse_moe)
                packed = None
        except (AttributeError, TypeError, ValueError):
            if isinstance(packed, AuthoritativePackedK3MoEFront):
                stale_packed = packed
            _increment_receipt(stale_resets=1)
            _drop_authoritative_packed_k3_moe_front(sparse_moe)
            packed = None

    if packed is None:
        try:
            packed = _build_authoritative_packed_front(sparse_moe)
        except (
            AttributeError,
            PackedMoEFrontUnsupported,
            TypeError,
            ValueError,
        ) as exc:
            if isinstance(stale_packed, AuthoritativePackedK3MoEFront):
                stale_packed.detach_source_views()
                current_signature = _source_signature(modules)
            object.__setattr__(
                sparse_moe,
                "_authoritative_packed_k3_moe_front_reason",
                str(exc),
            )
            object.__setattr__(
                sparse_moe,
                "_authoritative_packed_k3_moe_front",
                _UNSUPPORTED,
            )
            object.__setattr__(
                sparse_moe,
                "_authoritative_packed_k3_moe_front_source_signature",
                current_signature,
            )
            if receipt_width3_call:
                _receipt_width3_outcome("unsupported")
            return None
        object.__setattr__(
            sparse_moe,
            "_authoritative_packed_k3_moe_front",
            packed,
        )
        if receipt_width3_call:
            _increment_receipt(lazy_installs=1)

    try:
        outputs = packed(x)
    except PackedMoEFrontUnsupported:
        if receipt_width3_call:
            _receipt_width3_outcome("packed_dispatch_fallback")
        return None
    if receipt_width3_call:
        _receipt_width3_outcome("packed_hit", output_tensors=len(outputs))
    return outputs


def maybe_packed_k3_moe_front(
    sparse_moe: Any,
    x: mx.array,
) -> tuple[mx.array, ...] | None:
    """Return four exact decode projections, or ``None`` for the stock path."""

    if (
        not packed_moe_front_enabled()
        or authoritative_packed_moe_front_enabled()
        or getattr(sparse_moe, "training", True)
        or not _packed_input_supported(x)
    ):
        return None

    shared = getattr(sparse_moe, "shared_experts", None)
    routed_down = getattr(sparse_moe, "routed_expert_down_proj", None)
    if shared is None or routed_down is None:
        return None
    packed = getattr(sparse_moe, "_packed_k3_moe_front", None)
    modules = (
        shared.gate_proj,
        shared.up_proj,
        sparse_moe.gate,
        routed_down,
    )
    current_signature = _source_signature(modules)
    if packed is _UNSUPPORTED:
        unsupported_signature = getattr(
            sparse_moe,
            "_packed_k3_moe_front_source_signature",
            (),
        )
        if _same_source_signature(unsupported_signature, current_signature):
            return None
        invalidate_packed_k3_moe_front(sparse_moe)
        packed = None
    elif packed is not None:
        try:
            if not packed.matches_sources(modules):
                invalidate_packed_k3_moe_front(sparse_moe)
                packed = None
        except (AttributeError, TypeError, ValueError):
            invalidate_packed_k3_moe_front(sparse_moe)
            packed = None
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
            object.__setattr__(
                sparse_moe,
                "_packed_k3_moe_front_source_signature",
                current_signature,
            )
            return None
        # Keep the optimization out of the model parameter tree: the original
        # modules remain authoritative for checkpoint save/load and prefill.
        object.__setattr__(sparse_moe, "_packed_k3_moe_front", packed)

    try:
        return packed(x)
    except PackedMoEFrontUnsupported:
        return None
