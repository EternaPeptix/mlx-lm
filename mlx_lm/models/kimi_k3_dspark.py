"""Exact, fail-closed MLX primitives for the RadixArk Kimi K3 DSpark draft.

The checkpoint contains only the five-layer Qwen3/DFlash backbone, target
feature projection, Markov head, and confidence head.  Token embeddings and
the vocabulary head are borrowed by reference from the Kimi K3 target.
"""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx import nn

from .activations import swiglu
from .rope_utils import initialize_rope

RADIXARK_KIMI_K3_DSPARK_MODEL = "RadixArk/Kimi-K3-DSpark"
RADIXARK_KIMI_K3_DSPARK_REVISION = "eb03982e58d4fb79bcfc099e902158f562e2e27b"
RADIXARK_KIMI_K3_DSPARK_CONFIG_SHA256 = (
    "6aed20890d95cd69cf2ec006d1f30506fbd4f3091d44ca8e8b93e9fc7d50928f"
)
RADIXARK_KIMI_K3_DSPARK_WEIGHTS_SHA256 = (
    "29df0e8eafb81909f785df55cb352b90d6a1500c609b1d60526c1a62b4d42495"
)
RADIXARK_KIMI_K3_DSPARK_CONFIG_BYTES = 1_288
RADIXARK_KIMI_K3_DSPARK_WEIGHTS_BYTES = 4_498_585_858
RADIXARK_KIMI_K3_DSPARK_SNAPSHOT_FILES = 6
RADIXARK_KIMI_K3_DSPARK_SNAPSHOT_BYTES = 4_498_617_103
RADIXARK_KIMI_K3_DSPARK_WEIGHT_TENSORS = 62
RADIXARK_KIMI_K3_DSPARK_PARAMETERS = 2_249_289_601
RADIXARK_KIMI_K3_DSPARK_TARGET_LAYERS = (7, 23, 51, 67, 83)
RADIXARK_KIMI_K3_DSPARK_BLOCK_SIZE = 7
KIMI_K3_DSPARK_MODEL_VERIFY_WIDTH = RADIXARK_KIMI_K3_DSPARK_BLOCK_SIZE + 1
KIMI_K3_DSPARK_SCREENING_VERIFY_WIDTH = 3
# Compatibility name retained for callers of the first contract prototype.
KIMI_K3_DSPARK_INITIAL_VERIFY_WIDTH = KIMI_K3_DSPARK_MODEL_VERIFY_WIDTH

DSPARK_PROPOSER_ENV = "MLX_LM_KIMI_K3_DSPARK_PROPOSER"
DSPARK_STACKED_CONTEXT_KV_ENV = "MLX_LM_KIMI_K3_DSPARK_STACKED_CONTEXT_KV"
DSPARK_SEGMENTED_SDPA_ENV = "MLX_LM_KIMI_K3_DSPARK_SEGMENTED_SDPA"
DSPARK_SEGMENTED_SDPA_REQUIRED_CAPABILITY = "bounded_memory_metal_v1"

_DSPARK_SEGMENTED_BATCH_SIZE = 1
_DSPARK_SEGMENTED_QUERY_HEADS = 64
_DSPARK_SEGMENTED_KV_HEADS = 16
_DSPARK_SEGMENTED_HEAD_DIM = 64
_DSPARK_SEGMENTED_BLOCK_SIZE = RADIXARK_KIMI_K3_DSPARK_BLOCK_SIZE
_DSPARK_SEGMENTED_MAX_POSITIONS = 1_048_576
_DSPARK_SEGMENTED_SCALE = _DSPARK_SEGMENTED_HEAD_DIM**-0.5


def _strict_environment_flag(name: str) -> bool:
    value = os.environ.get(name, "0")
    if value not in {"0", "1"}:
        raise ValueError(f"{name} must be 0 or 1")
    return value == "1"


def kimi_k3_dspark_proposer_enabled() -> bool:
    """Return the strict, default-off proposer feature state."""

    return _strict_environment_flag(DSPARK_PROPOSER_ENV)


def kimi_k3_dspark_stacked_context_kv_enabled() -> bool:
    """Return the strict, default-off stacked context-KV feature state."""

    return _strict_environment_flag(DSPARK_STACKED_CONTEXT_KV_ENV)


def kimi_k3_dspark_segmented_sdpa_enabled() -> bool:
    """Return the strict, default-off two-bank DSpark attention state."""

    return _strict_environment_flag(DSPARK_SEGMENTED_SDPA_ENV)


def _kimi_k3_dspark_segmented_sdpa_primitive():
    """Feature-detect the optional MLX two-bank attention primitive."""

    return getattr(mx.fast, "segmented_scaled_dot_product_attention", None)


def _kimi_k3_dspark_segmented_sdpa_capabilities_primitive():
    """Feature-detect the matching MLX capability-reporting primitive."""

    return getattr(
        mx.fast,
        "segmented_scaled_dot_product_attention_capabilities",
        None,
    )


def kimi_k3_dspark_segmented_sdpa_capabilities() -> tuple[str, ...]:
    """Return a validated MLX segmented-attention capability tuple."""

    capabilities_primitive = _kimi_k3_dspark_segmented_sdpa_capabilities_primitive()
    if capabilities_primitive is None:
        return ()
    capabilities = capabilities_primitive()
    if not isinstance(capabilities, tuple) or any(
        not isinstance(capability, str) for capability in capabilities
    ):
        raise RuntimeError(
            "mx.fast.segmented_scaled_dot_product_attention_capabilities "
            "returned an invalid contract"
        )
    return capabilities


def require_kimi_k3_dspark_segmented_sdpa():
    """Return the primitive only when MLX advertises its bounded Metal path."""

    primitive = _kimi_k3_dspark_segmented_sdpa_primitive()
    if primitive is None:
        raise RuntimeError(
            f"{DSPARK_SEGMENTED_SDPA_ENV}=1 requires "
            "mx.fast.segmented_scaled_dot_product_attention"
        )
    capabilities = kimi_k3_dspark_segmented_sdpa_capabilities()
    if DSPARK_SEGMENTED_SDPA_REQUIRED_CAPABILITY not in capabilities:
        raise RuntimeError(
            f"{DSPARK_SEGMENTED_SDPA_ENV}=1 requires MLX capability "
            f"{DSPARK_SEGMENTED_SDPA_REQUIRED_CAPABILITY!r}; "
            f"advertised capabilities are {capabilities!r}"
        )
    return primitive


def _validate_kimi_k3_dspark_segmented_sdpa(
    queries: mx.array,
    context_keys: mx.array,
    context_values: mx.array,
    noise_keys: mx.array,
    noise_values: mx.array,
    *,
    scale: float,
) -> None:
    """Require the exact released K3 DSpark attention specialization.

    The first primitive deliberately supports only the geometry exercised by
    the pinned production checkpoint.  Keeping this model-side contract exact
    prevents an opt-in from silently selecting a numerically or semantically
    different path for tests, training, batching, or a future checkpoint.
    """

    arrays = (
        queries,
        context_keys,
        context_values,
        noise_keys,
        noise_values,
    )
    if any(not isinstance(array, mx.array) or array.ndim != 4 for array in arrays):
        raise ValueError(
            "Kimi K3 DSpark segmented SDPA requires five rank-four MLX arrays"
        )
    if any(array.dtype != mx.bfloat16 for array in arrays):
        raise ValueError("Kimi K3 DSpark segmented SDPA requires BF16 inputs")
    if type(scale) is not float or scale != _DSPARK_SEGMENTED_SCALE:
        raise ValueError(
            "Kimi K3 DSpark segmented SDPA scale does not match head dimension 64"
        )

    batch = _DSPARK_SEGMENTED_BATCH_SIZE
    query_heads = _DSPARK_SEGMENTED_QUERY_HEADS
    kv_heads = _DSPARK_SEGMENTED_KV_HEADS
    block = _DSPARK_SEGMENTED_BLOCK_SIZE
    head_dim = _DSPARK_SEGMENTED_HEAD_DIM
    context_length = int(context_keys.shape[2])
    expected_queries = (batch, query_heads, block, head_dim)
    expected_context = (batch, kv_heads, context_length, head_dim)
    expected_noise = (batch, kv_heads, block, head_dim)
    if (
        tuple(queries.shape) != expected_queries
        or context_length <= 0
        or context_length + block > _DSPARK_SEGMENTED_MAX_POSITIONS
        or tuple(context_keys.shape) != expected_context
        or tuple(context_values.shape) != expected_context
        or tuple(noise_keys.shape) != expected_noise
        or tuple(noise_values.shape) != expected_noise
    ):
        raise ValueError(
            "Kimi K3 DSpark segmented SDPA inputs do not match the released "
            "batch-one 64x16-head, width-seven, head-dimension-64 geometry"
        )


def _kimi_k3_dspark_segmented_sdpa(
    queries: mx.array,
    context_keys: mx.array,
    context_values: mx.array,
    noise_keys: mx.array,
    noise_values: mx.array,
    *,
    scale: float,
) -> mx.array:
    """Call the optional two-bank primitive without an unsafe fallback."""

    _validate_kimi_k3_dspark_segmented_sdpa(
        queries,
        context_keys,
        context_values,
        noise_keys,
        noise_values,
        scale=scale,
    )
    primitive = require_kimi_k3_dspark_segmented_sdpa()
    return primitive(
        queries,
        context_keys,
        context_values,
        noise_keys,
        noise_values,
        scale=scale,
    )


@dataclass(frozen=True)
class KimiK3DSparkContract:
    """Validated checkpoint, proposal-width, and placement contract."""

    target_layer_ids: tuple[int, ...]
    verify_width: int
    checkpoint_block_size: int
    parameter_count: int
    placement: str
    owns_embedding: bool
    owns_lm_head: bool
    screening_override: bool

    @property
    def num_draft_tokens(self) -> int:
        return self.verify_width - 1

    @property
    def maximum_verify_width(self) -> int:
        return self.checkpoint_block_size + 1

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        verify_width: int = KIMI_K3_DSPARK_MODEL_VERIFY_WIDTH,
        placement: str = "replicated",
        screening_override: bool = False,
    ) -> KimiK3DSparkContract:
        """Validate the pinned public 2.249B checkpoint before allocation."""

        if config.get("architectures") != ["DSparkDraftModel"]:
            raise ValueError("Kimi K3 DSpark architecture contract does not match")

        expected_scalars = {
            "model_type": "qwen3",
            "block_size": RADIXARK_KIMI_K3_DSPARK_BLOCK_SIZE,
            "hidden_size": 7168,
            "intermediate_size": 14336,
            "num_hidden_layers": 5,
            "num_attention_heads": 64,
            "num_key_value_heads": 16,
            "head_dim": 64,
            "num_target_layers": 93,
            "vocab_size": 163840,
            "markov_rank": 256,
            "markov_head_type": "vanilla",
            "dtype": "bfloat16",
            "hidden_act": "silu",
            "rms_norm_eps": 1e-5,
            "max_position_embeddings": 1_048_576,
            "attention_bias": False,
            "attention_dropout": 0.0,
            "enable_confidence_head": True,
            "confidence_head_with_markov": True,
            "tie_word_embeddings": False,
        }
        for name, expected in expected_scalars.items():
            actual = config.get(name)
            if type(actual) is not type(expected) or actual != expected:
                raise ValueError(
                    f"Kimi K3 DSpark {name} must be {expected!r}, got {actual!r}"
                )

        if config.get("layer_types") != ["full_attention"] * 5:
            raise ValueError("Kimi K3 DSpark layer_types contract does not match")
        if config.get("rope_parameters") != {
            "rope_theta": 10_000.0,
            "rope_type": "default",
        }:
            raise ValueError("Kimi K3 DSpark RoPE contract does not match")

        dflash = config.get("dflash_config")
        if not isinstance(dflash, Mapping):
            raise TypeError("Kimi K3 DSpark dflash_config must be a mapping")
        target_layer_ids = tuple(dflash.get("target_layer_ids", ()))
        if target_layer_ids != RADIXARK_KIMI_K3_DSPARK_TARGET_LAYERS:
            raise ValueError("Kimi K3 DSpark target hidden taps do not match")
        mask_token_id = dflash.get("mask_token_id")
        if type(mask_token_id) is not int or mask_token_id != 163824:
            raise ValueError("Kimi K3 DSpark mask token does not match")

        block_size = int(config["block_size"])
        native_width = block_size + 1
        if type(verify_width) is not int or not 2 <= verify_width <= native_width:
            raise ValueError(
                f"Kimi K3 DSpark verify width must be in [2, {native_width}]"
            )
        if type(screening_override) is not bool:
            raise TypeError("Kimi K3 DSpark screening_override must be bool")
        if verify_width != native_width and (
            not screening_override
            or verify_width != KIMI_K3_DSPARK_SCREENING_VERIFY_WIDTH
        ):
            raise ValueError(
                "non-native Kimi K3 DSpark verification requires the explicit "
                "width-three screening override"
            )
        if placement != "replicated":
            raise ValueError(
                "Kimi K3 DSpark must be replicated on every target TP rank"
            )

        return cls(
            target_layer_ids=target_layer_ids,
            verify_width=verify_width,
            checkpoint_block_size=block_size,
            parameter_count=RADIXARK_KIMI_K3_DSPARK_PARAMETERS,
            placement=placement,
            owns_embedding=False,
            owns_lm_head=False,
            screening_override=screening_override,
        )


@dataclass(frozen=True)
class KimiK3DSparkArgs:
    """The exact GQA drafter geometry, also parameterizable for unit tests."""

    hidden_size: int = 7168
    intermediate_size: int = 14336
    num_hidden_layers: int = 5
    num_attention_heads: int = 64
    num_key_value_heads: int = 16
    head_dim: int = 64
    vocab_size: int = 163840
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10_000.0
    max_position_embeddings: int = 1_048_576
    block_size: int = 7
    mask_token_id: int = 163824
    target_layer_ids: tuple[int, ...] = RADIXARK_KIMI_K3_DSPARK_TARGET_LAYERS
    markov_rank: int = 256
    enable_confidence_head: bool = True
    confidence_head_with_markov: bool = True
    weight_dtype: Any = mx.bfloat16

    def __post_init__(self):
        positive = (
            self.hidden_size,
            self.intermediate_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.vocab_size,
            self.block_size,
            self.markov_rank,
        )
        if any(type(value) is not int or value <= 0 for value in positive):
            raise ValueError("Kimi K3 DSpark dimensions must be positive integers")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("Kimi K3 DSpark query heads must divide by KV heads")
        if not self.target_layer_ids:
            raise ValueError("Kimi K3 DSpark requires target hidden taps")
        if self.mask_token_id < 0 or self.mask_token_id >= self.vocab_size:
            raise ValueError("Kimi K3 DSpark mask token is outside the vocabulary")
        if not self.enable_confidence_head or not self.confidence_head_with_markov:
            raise ValueError("this exact slice requires the released confidence head")

    @property
    def native_verify_width(self) -> int:
        return self.block_size + 1

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> KimiK3DSparkArgs:
        contract = KimiK3DSparkContract.from_config(config)
        return cls(
            hidden_size=int(config["hidden_size"]),
            intermediate_size=int(config["intermediate_size"]),
            num_hidden_layers=int(config["num_hidden_layers"]),
            num_attention_heads=int(config["num_attention_heads"]),
            num_key_value_heads=int(config["num_key_value_heads"]),
            head_dim=int(config["head_dim"]),
            vocab_size=int(config["vocab_size"]),
            rms_norm_eps=float(config["rms_norm_eps"]),
            rope_theta=float(config["rope_parameters"]["rope_theta"]),
            max_position_embeddings=int(config["max_position_embeddings"]),
            block_size=contract.checkpoint_block_size,
            mask_token_id=int(config["dflash_config"]["mask_token_id"]),
            target_layer_ids=contract.target_layer_ids,
            markov_rank=int(config["markov_rank"]),
            enable_confidence_head=bool(config["enable_confidence_head"]),
            confidence_head_with_markov=bool(config["confidence_head_with_markov"]),
            weight_dtype=mx.bfloat16,
        )


def kimi_k3_dspark_expected_weight_shapes(
    args: KimiK3DSparkArgs,
) -> dict[str, tuple[int, ...]]:
    """Return the complete checkpoint key/shape inventory."""

    h = args.hidden_size
    q = args.num_attention_heads * args.head_dim
    kv = args.num_key_value_heads * args.head_dim
    shapes = {
        "fc.weight": (h, len(args.target_layer_ids) * h),
        "hidden_norm.weight": (h,),
        "norm.weight": (h,),
    }
    for index in range(args.num_hidden_layers):
        prefix = f"layers.{index}"
        shapes.update(
            {
                f"{prefix}.input_layernorm.weight": (h,),
                f"{prefix}.self_attn.q_proj.weight": (q, h),
                f"{prefix}.self_attn.k_proj.weight": (kv, h),
                f"{prefix}.self_attn.v_proj.weight": (kv, h),
                f"{prefix}.self_attn.o_proj.weight": (h, q),
                f"{prefix}.self_attn.q_norm.weight": (args.head_dim,),
                f"{prefix}.self_attn.k_norm.weight": (args.head_dim,),
                f"{prefix}.post_attention_layernorm.weight": (h,),
                f"{prefix}.mlp.gate_proj.weight": (args.intermediate_size, h),
                f"{prefix}.mlp.up_proj.weight": (args.intermediate_size, h),
                f"{prefix}.mlp.down_proj.weight": (h, args.intermediate_size),
            }
        )
    shapes.update(
        {
            "markov_head.markov_w1.weight": (args.vocab_size, args.markov_rank),
            "markov_head.markov_w2.weight": (args.vocab_size, args.markov_rank),
            "confidence_head.proj.weight": (
                1,
                args.hidden_size + args.markov_rank,
            ),
            "confidence_head.proj.bias": (1,),
        }
    )
    return shapes


def kimi_k3_dspark_parameter_count(args: KimiK3DSparkArgs) -> int:
    return sum(
        _shape_size(shape)
        for shape in kimi_k3_dspark_expected_weight_shapes(args).values()
    )


def _shape_size(shape: Sequence[int]) -> int:
    size = 1
    for dimension in shape:
        size *= int(dimension)
    return size


def attest_kimi_k3_dspark_weights(
    weights: Mapping[str, mx.array],
    args: KimiK3DSparkArgs,
) -> int:
    """Fail closed on any checkpoint key, shape, dtype, or element mismatch."""

    expected = kimi_k3_dspark_expected_weight_shapes(args)
    actual_keys = set(weights)
    expected_keys = set(expected)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    if missing or unexpected:
        raise ValueError(
            "Kimi K3 DSpark tensor keys do not match: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )

    count = 0
    for name, shape in expected.items():
        value = weights[name]
        if tuple(value.shape) != shape:
            raise ValueError(
                f"Kimi K3 DSpark tensor {name} must have shape {shape}, "
                f"got {tuple(value.shape)}"
            )
        if value.dtype != args.weight_dtype:
            raise ValueError(
                f"Kimi K3 DSpark tensor {name} must have dtype "
                f"{args.weight_dtype}, got {value.dtype}"
            )
        count += int(value.size)

    expected_count = kimi_k3_dspark_parameter_count(args)
    if count != expected_count:
        raise ValueError(
            f"Kimi K3 DSpark parameter count must be {expected_count}, got {count}"
        )
    return count


class KimiK3DSparkContextCache:
    """Append-only projected target-context K/V for one draft layer.

    The occupied prefix is stored in a capacity buffer.  Small appends write
    into existing storage instead of concatenating the complete context on
    every token.  Capacity grows geometrically at first, then in bounded
    increments so a long-context allocation does not reserve an unbounded
    fraction of unused memory.

    ``capacity_hint`` is useful when the prompt length is known up front.  It
    avoids every intermediate history copy while retaining the same logical
    K/V shape exposed through :attr:`keys` and :attr:`values`.
    """

    step: int = 256
    max_growth: int = 65_536

    __slots__: tuple[str, ...] = (
        "_allocation_count",
        "_capacity_hint",
        "_copied_tokens",
        "_keys",
        "_max_growth",
        "_offset",
        "_step",
        "_values",
    )

    def __init__(
        self,
        *,
        capacity_hint: int = 0,
        step: int | None = None,
        max_growth: int | None = None,
    ):
        step = self.step if step is None else step
        max_growth = self.max_growth if max_growth is None else max_growth
        if type(capacity_hint) is not int or capacity_hint < 0:
            raise ValueError("Kimi K3 DSpark context capacity hint is invalid")
        if type(step) is not int or step <= 0:
            raise ValueError("Kimi K3 DSpark context capacity step is invalid")
        if type(max_growth) is not int or max_growth < step:
            raise ValueError("Kimi K3 DSpark context maximum growth is invalid")
        self._keys: mx.array | None = None
        self._values: mx.array | None = None
        self._offset: int = 0
        self._capacity_hint: int = capacity_hint
        self._step: int = step
        self._max_growth: int = max_growth
        self._allocation_count: int = 0
        self._copied_tokens: int = 0

    @property
    def keys(self) -> mx.array | None:
        if self._keys is None:
            return None
        return self._keys[..., : self._offset, :]

    @property
    def values(self) -> mx.array | None:
        if self._values is None:
            return None
        return self._values[..., : self._offset, :]

    @property
    def length(self) -> int:
        return self._offset

    @property
    def capacity(self) -> int:
        return 0 if self._keys is None else int(self._keys.shape[2])

    @property
    def allocation_count(self) -> int:
        """Number of backing-buffer allocations, including the first one."""

        return self._allocation_count

    @property
    def copied_tokens(self) -> int:
        """Logical history positions copied while growing the buffer."""

        return self._copied_tokens

    def _round_capacity(self, capacity: int) -> int:
        return ((capacity + self._step - 1) // self._step) * self._step

    def _planned_capacity(self, current: int, required: int) -> int:
        if required <= current:
            return current
        if current == 0:
            return self._round_capacity(max(required, self._capacity_hint))
        growth = min(self._max_growth, max(self._step, current))
        return self._round_capacity(max(required, current + growth))

    def _next_capacity(self, required: int) -> int:
        return self._planned_capacity(self.capacity, required)

    @staticmethod
    def _padding_like(value: mx.array, length: int) -> mx.array:
        shape = (*value.shape[:2], length, value.shape[3])
        return mx.zeros(shape, dtype=value.dtype)

    def _allocate_with_append(
        self,
        keys: mx.array,
        values: mx.array,
        required: int,
    ) -> None:
        new_capacity = self._next_capacity(required)
        padding = new_capacity - required
        key_parts = [keys]
        value_parts = [values]
        if self._keys is not None:
            assert self._values is not None
            key_parts.insert(0, self._keys[..., : self._offset, :])
            value_parts.insert(0, self._values[..., : self._offset, :])
            self._copied_tokens += self._offset
        if padding:
            key_parts.append(self._padding_like(keys, padding))
            value_parts.append(self._padding_like(values, padding))
        self._keys = (
            key_parts[0] if len(key_parts) == 1 else mx.concatenate(key_parts, axis=2)
        )
        self._values = (
            value_parts[0]
            if len(value_parts) == 1
            else mx.concatenate(value_parts, axis=2)
        )
        self._allocation_count += 1

    def append(self, keys: mx.array, values: mx.array) -> None:
        if keys.shape != values.shape or keys.ndim != 4:
            raise ValueError("Kimi K3 DSpark context K/V shape does not match")
        if keys.dtype != values.dtype:
            raise ValueError("Kimi K3 DSpark context K/V dtype does not match")
        if self._keys is None and keys.shape[2] == 0:
            self._keys = keys
            self._values = values
            return
        if self._keys is not None and (
            self._values is None
            or keys.shape[:2] != self._keys.shape[:2]
            or keys.shape[3:] != self._keys.shape[3:]
            or keys.dtype != self._keys.dtype
        ):
            raise ValueError("Kimi K3 DSpark context append is incompatible")
        append_length = int(keys.shape[2])
        if append_length == 0:
            return
        previous = self._offset
        required = previous + append_length
        if required > self.capacity:
            self._allocate_with_append(keys, values, required)
        else:
            assert self._keys is not None and self._values is not None
            self._keys[..., previous:required, :] = keys
            self._values[..., previous:required, :] = values
        self._offset = required


class KimiK3DSparkAttention(nn.Module):
    """Non-causal dual-source GQA used by the released DFlash backbone."""

    def __init__(self, args: KimiK3DSparkArgs):
        super().__init__()
        h = args.hidden_size
        self.num_heads = args.num_attention_heads
        self.num_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = args.head_dim**-0.5
        self.rms_norm_eps = args.rms_norm_eps
        self.q_proj = nn.Linear(h, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(h, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(h, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, h, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.rope = initialize_rope(
            self.head_dim,
            base=args.rope_theta,
            traditional=False,
            scaling_config={"rope_type": "default"},
            max_position_embeddings=args.max_position_embeddings,
        )

    def project_context(
        self,
        hidden: mx.array,
        offset: int,
    ) -> tuple[mx.array, mx.array]:
        batch, length, _ = hidden.shape
        keys = self.k_proj(hidden).reshape(
            batch, length, self.num_kv_heads, self.head_dim
        )
        values = self.v_proj(hidden).reshape(
            batch, length, self.num_kv_heads, self.head_dim
        )
        keys = self.k_norm(keys).transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)
        return self.rope(keys, offset=offset), values

    def __call__(
        self,
        hidden: mx.array,
        block_offset: int,
        cache: KimiK3DSparkContextCache,
    ) -> mx.array:
        context_keys = cache.keys
        context_values = cache.values
        if context_keys is None or context_values is None:
            raise ValueError("Kimi K3 DSpark context cache is empty")
        batch, length, _ = hidden.shape
        queries = self.q_proj(hidden).reshape(
            batch, length, self.num_heads, self.head_dim
        )
        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        queries = self.rope(queries, offset=block_offset)

        noise_keys, noise_values = self.project_context(hidden, block_offset)
        if kimi_k3_dspark_segmented_sdpa_enabled():
            output = _kimi_k3_dspark_segmented_sdpa(
                queries,
                context_keys,
                context_values,
                noise_keys,
                noise_values,
                scale=self.scale,
            )
        else:
            # This is the accepted implementation.  Keep it bit-for-bit
            # reachable whenever the experimental two-bank path is disabled.
            keys = mx.concatenate([context_keys, noise_keys], axis=2)
            values = mx.concatenate([context_values, noise_values], axis=2)
            output = mx.fast.scaled_dot_product_attention(
                queries,
                keys,
                values,
                scale=self.scale,
                mask=None,
            )
        output = output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.o_proj(output)


class KimiK3DSparkMLP(nn.Module):
    def __init__(self, args: KimiK3DSparkArgs):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

    def __call__(self, hidden: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(hidden), self.up_proj(hidden)))


class KimiK3DSparkDecoderLayer(nn.Module):
    def __init__(self, args: KimiK3DSparkArgs):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.self_attn = KimiK3DSparkAttention(args)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.mlp = KimiK3DSparkMLP(args)

    def __call__(
        self,
        hidden: mx.array,
        block_offset: int,
        cache: KimiK3DSparkContextCache,
    ) -> mx.array:
        hidden = hidden + self.self_attn(
            self.input_layernorm(hidden), block_offset, cache
        )
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class KimiK3DSparkMarkovHead(nn.Module):
    def __init__(self, args: KimiK3DSparkArgs):
        super().__init__()
        self.markov_w1 = nn.Embedding(args.vocab_size, args.markov_rank)
        self.markov_w2 = nn.Linear(args.markov_rank, args.vocab_size, bias=False)

    def previous_embeddings(self, token_ids: mx.array) -> mx.array:
        return self.markov_w1(token_ids)

    def bias(self, token_ids: mx.array) -> mx.array:
        return self.markov_w2(self.previous_embeddings(token_ids))


class KimiK3DSparkConfidenceHead(nn.Module):
    def __init__(self, args: KimiK3DSparkArgs):
        super().__init__()
        self.proj = nn.Linear(args.hidden_size + args.markov_rank, 1)

    def __call__(self, features: mx.array) -> mx.array:
        return self.proj(features).squeeze(-1)


class KimiK3DSparkModel(nn.Module):
    """Five-layer drafter with externally owned target embedding and head."""

    def __init__(self, args: KimiK3DSparkArgs):
        super().__init__()
        self.args = args
        self.fc = nn.Linear(
            len(args.target_layer_ids) * args.hidden_size,
            args.hidden_size,
            bias=False,
        )
        self.hidden_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.layers = [
            KimiK3DSparkDecoderLayer(args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.markov_head = KimiK3DSparkMarkovHead(args)
        self.confidence_head = KimiK3DSparkConfidenceHead(args)
        # Leading underscores keep borrowed modules and derived packed arrays out
        # of MLX's owned-parameter traversal.  They are references, not copies.
        self._target_embedding = None
        self._target_vocab_head = None
        self._tied_target_head = False
        self._stacked_context_kv = None

    @property
    def target_embedding(self):
        return self._target_embedding

    @property
    def target_vocab_head(self):
        return self._target_vocab_head

    def bind_target_modules(self, embedding: nn.Module, vocab_head=None):
        weight = getattr(embedding, "weight", None)
        if weight is None or tuple(weight.shape) != (
            self.args.vocab_size,
            self.args.hidden_size,
        ):
            raise ValueError("Kimi K3 DSpark target embedding shape does not match")
        if vocab_head is not None:
            head_weight = getattr(vocab_head, "weight", None)
            if head_weight is not None and tuple(head_weight.shape) != (
                self.args.vocab_size,
                self.args.hidden_size,
            ):
                raise ValueError("Kimi K3 DSpark target vocabulary head does not match")
            if not callable(vocab_head):
                raise TypeError("Kimi K3 DSpark target vocabulary head is not callable")
        elif not hasattr(embedding, "as_linear"):
            raise ValueError("Kimi K3 DSpark tied target head is unavailable")
        self._target_embedding = embedding
        self._target_vocab_head = vocab_head
        self._tied_target_head = vocab_head is None
        return self

    def bind_target(self, target_model):
        language_model = getattr(target_model, "language_model", target_model)
        inner = getattr(language_model, "model", None)
        if inner is None or not hasattr(inner, "embed_tokens"):
            inner = getattr(target_model, "model", None)
        if inner is None or not hasattr(inner, "embed_tokens"):
            raise ValueError("Kimi K3 DSpark cannot find the target embedding")
        vocab_head = getattr(language_model, "lm_head", None)
        tied = bool(
            getattr(
                getattr(language_model, "args", None),
                "tie_word_embeddings",
                False,
            )
        )
        if vocab_head is None and not tied:
            raise ValueError("Kimi K3 DSpark cannot find the untied target head")
        return self.bind_target_modules(inner.embed_tokens, vocab_head)

    def make_context_cache(
        self,
        *,
        capacity_hint: int = 0,
    ) -> list[KimiK3DSparkContextCache]:
        return [
            KimiK3DSparkContextCache(capacity_hint=capacity_hint) for _ in self.layers
        ]

    def _validate_context_cache(
        self,
        context_cache: Sequence[KimiK3DSparkContextCache],
        expected_offset: int,
    ) -> None:
        if len(context_cache) != len(self.layers):
            raise ValueError("Kimi K3 DSpark context cache count does not match")
        lengths = {cache.length for cache in context_cache}
        if lengths != {expected_offset}:
            raise ValueError(
                "Kimi K3 DSpark context cache offsets do not match the request"
            )

    def project_target_hidden(
        self,
        aux_hidden_states: Sequence[mx.array],
    ) -> mx.array:
        if len(aux_hidden_states) != len(self.args.target_layer_ids):
            raise ValueError("Kimi K3 DSpark target hidden tap count does not match")
        reference_shape = tuple(aux_hidden_states[0].shape)
        if len(reference_shape) != 3 or reference_shape[-1] != self.args.hidden_size:
            raise ValueError("Kimi K3 DSpark target hidden shape does not match")
        for hidden in aux_hidden_states:
            if tuple(hidden.shape) != reference_shape:
                raise ValueError("Kimi K3 DSpark target hidden shapes disagree")
        return self.hidden_norm(self.fc(mx.concatenate(aux_hidden_states, axis=-1)))

    def _stacked_context_parameters(self) -> tuple[mx.array, mx.array]:
        if self._stacked_context_kv is None:
            projection_weights = []
            norm_weights = []
            for layer in self.layers:
                attention = layer.self_attn
                projection_weights.extend(
                    [attention.k_proj.weight, attention.v_proj.weight]
                )
                norm_weights.append(attention.k_norm.weight)
            self._stacked_context_kv = (
                mx.concatenate(projection_weights, axis=0),
                mx.stack(norm_weights),
            )
        return self._stacked_context_kv

    def _append_target_context_stacked(
        self,
        projected: mx.array,
        context_offset: int,
        context_cache: Sequence[KimiK3DSparkContextCache],
    ) -> None:
        batch, length, _ = projected.shape
        layer_count = len(self.layers)
        kv_heads = self.args.num_key_value_heads
        head_dim = self.args.head_dim
        projection_weight, norm_weight = self._stacked_context_parameters()
        kv = mx.matmul(projected, projection_weight.swapaxes(-1, -2)).reshape(
            batch,
            length,
            layer_count,
            2,
            kv_heads,
            head_dim,
        )
        keys = kv[:, :, :, 0].transpose(0, 2, 3, 1, 4)
        values = kv[:, :, :, 1].transpose(0, 2, 3, 1, 4)
        keys_dtype = keys.dtype
        keys = keys.astype(mx.float32)
        variance = mx.mean(mx.square(keys), axis=-1, keepdims=True)
        keys = keys * mx.rsqrt(variance + self.args.rms_norm_eps)
        keys = keys * norm_weight[None, :, None, None, :].astype(mx.float32)
        keys = keys.astype(keys_dtype)
        keys = (
            self.layers[0]
            .self_attn.rope(
                keys.reshape(batch * layer_count, kv_heads, length, head_dim),
                offset=context_offset,
            )
            .reshape(batch, layer_count, kv_heads, length, head_dim)
        )
        for index, cache in enumerate(context_cache):
            cache.append(keys[:, index], values[:, index])

    def append_target_context(
        self,
        aux_hidden_states: Sequence[mx.array],
        context_offset: int,
        context_cache: Sequence[KimiK3DSparkContextCache],
        *,
        use_stacked_context_kv: bool | None = None,
    ) -> None:
        if type(context_offset) is not int or context_offset < 0:
            raise ValueError("Kimi K3 DSpark context offset is invalid")
        self._validate_context_cache(context_cache, context_offset)
        projected = self.project_target_hidden(aux_hidden_states)
        use_stacked = (
            kimi_k3_dspark_stacked_context_kv_enabled()
            if use_stacked_context_kv is None
            else use_stacked_context_kv
        )
        if type(use_stacked) is not bool:
            raise TypeError("Kimi K3 DSpark stacked context selector must be bool")
        if use_stacked:
            self._append_target_context_stacked(
                projected, context_offset, context_cache
            )
            return
        for layer, cache in zip(self.layers, context_cache, strict=True):
            keys, values = layer.self_attn.project_context(projected, context_offset)
            cache.append(keys, values)

    def _embed(self, token_ids: mx.array) -> mx.array:
        if self._target_embedding is None:
            raise ValueError("Kimi K3 DSpark target modules are not bound")
        return self._target_embedding(token_ids)

    def _target_logits(self, hidden: mx.array) -> mx.array:
        if self._target_embedding is None:
            raise ValueError("Kimi K3 DSpark target modules are not bound")
        if self._tied_target_head:
            return self._target_embedding.as_linear(hidden)
        return self._target_vocab_head(hidden)

    def forward_block(
        self,
        token_ids: mx.array,
        context_cache: Sequence[KimiK3DSparkContextCache],
    ) -> mx.array:
        if token_ids.ndim != 2 or tuple(token_ids.shape) != (
            1,
            self.args.block_size,
        ):
            raise ValueError(
                "Kimi K3 DSpark proposal requires a batch-one, model-native "
                f"width-{self.args.block_size} draft block"
            )
        block_offset = context_cache[0].length if context_cache else -1
        if block_offset <= 0:
            raise ValueError("Kimi K3 DSpark proposal requires populated context")
        self._validate_context_cache(context_cache, block_offset)
        hidden = self._embed(token_ids)
        for layer, cache in zip(self.layers, context_cache, strict=True):
            hidden = layer(hidden, block_offset, cache)
        return self.norm(hidden)

    def base_logits(
        self,
        block_hidden: mx.array,
        proposal_count: int,
    ) -> mx.array:
        if block_hidden.ndim != 3 or block_hidden.shape[1] != self.args.block_size:
            raise ValueError("Kimi K3 DSpark block hidden width is invalid")
        if (
            type(proposal_count) is not int
            or proposal_count < 1
            or proposal_count > self.args.block_size
        ):
            raise ValueError("Kimi K3 DSpark proposal count is invalid")
        # The anchor occupies draft input position zero, whose output predicts
        # proposal zero.  All block positions attend bidirectionally, so even a
        # shorter screening head must run the full model-native draft block.
        return self._target_logits(block_hidden[:, :proposal_count])

    def sample_greedy_markov(
        self,
        base_logits: mx.array,
        anchor_token: int,
    ) -> tuple[mx.array, mx.array]:
        if base_logits.ndim != 3 or base_logits.shape[0] != 1:
            raise ValueError("Kimi K3 DSpark Markov sampling requires batch one")
        previous = mx.array([anchor_token], dtype=mx.int32)
        sampled = []
        corrected = []
        for position in range(base_logits.shape[1]):
            step_logits = base_logits[:, position] + self.markov_head.bias(previous)
            token = mx.argmax(step_logits, axis=-1).astype(mx.int32)
            corrected.append(step_logits[:, None])
            sampled.append(token[:, None])
            previous = token
        return mx.concatenate(sampled, axis=1), mx.concatenate(corrected, axis=1)

    def confidence_logits(
        self,
        block_hidden: mx.array,
        anchor_token: int,
        sampled_tokens: mx.array,
    ) -> mx.array:
        if (
            block_hidden.ndim != 3
            or sampled_tokens.ndim != 2
            or tuple(block_hidden.shape[:2]) != tuple(sampled_tokens.shape)
        ):
            raise ValueError("Kimi K3 DSpark confidence inputs do not align")
        previous = mx.concatenate(
            [
                mx.array([[anchor_token]], dtype=mx.int32),
                sampled_tokens[:, :-1],
            ],
            axis=1,
        )
        markov = self.markov_head.previous_embeddings(previous)
        features = mx.concatenate([block_hidden, markov], axis=-1)
        return self.confidence_head(features)

    def load_owned_weights(self, weights: Mapping[str, mx.array]) -> None:
        attest_kimi_k3_dspark_weights(weights, self.args)
        self.load_weights(list(weights.items()), strict=True)
        self._stacked_context_kv = None


@dataclass(frozen=True)
class KimiK3DSparkProposal:
    tokens: mx.array
    base_logits: mx.array
    corrected_logits: mx.array
    confidence_logits: mx.array
    verify_width: int
    draft_block_width: int
    mode: str


class KimiK3DSparkProposer:
    """Isolated greedy proposer; target verification remains the caller's job."""

    def __init__(
        self,
        drafter: KimiK3DSparkModel,
        *,
        verify_width: int | None = None,
        screening_override: bool = False,
    ):
        self.drafter = drafter
        native_width = drafter.args.native_verify_width
        self.verify_width = native_width if verify_width is None else verify_width
        if (
            type(self.verify_width) is not int
            or not 2 <= self.verify_width <= native_width
        ):
            raise ValueError(
                f"Kimi K3 DSpark verify width must be in [2, {native_width}]"
            )
        if self.verify_width == native_width:
            if screening_override:
                raise ValueError(
                    "native Kimi K3 DSpark width is not a screening override"
                )
            self.mode = "model_native"
        elif (
            self.verify_width == KIMI_K3_DSPARK_SCREENING_VERIFY_WIDTH
            and screening_override
        ):
            self.mode = "width3_screening_override"
            warnings.warn(
                "Kimi K3 DSpark width-three screening overrides the checkpoint's "
                "model-native gamma=7 / verify-width=8 contract",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            raise ValueError(
                "non-native Kimi K3 DSpark verification requires the explicit "
                "width-three screening override"
            )

    @property
    def proposal_count(self) -> int:
        return self.verify_width - 1

    def make_context_cache(
        self,
        *,
        capacity_hint: int = 0,
    ) -> list[KimiK3DSparkContextCache]:
        return self.drafter.make_context_cache(capacity_hint=capacity_hint)

    def append_target_context(
        self,
        aux_hidden_states: Sequence[mx.array],
        context_offset: int,
        context_cache: Sequence[KimiK3DSparkContextCache],
        *,
        use_stacked_context_kv: bool | None = None,
    ) -> None:
        self.drafter.append_target_context(
            aux_hidden_states,
            context_offset,
            context_cache,
            use_stacked_context_kv=use_stacked_context_kv,
        )

    def propose(
        self,
        anchor_token: int,
        context_cache: Sequence[KimiK3DSparkContextCache],
    ) -> KimiK3DSparkProposal:
        if not kimi_k3_dspark_proposer_enabled():
            raise RuntimeError(
                f"Kimi K3 DSpark proposer is disabled; set {DSPARK_PROPOSER_ENV}=1"
            )
        if (
            type(anchor_token) is not int
            or not 0 <= anchor_token < self.drafter.args.vocab_size
        ):
            raise ValueError("Kimi K3 DSpark anchor token is invalid")
        token_ids = mx.array(
            [
                [anchor_token]
                + [self.drafter.args.mask_token_id] * (self.drafter.args.block_size - 1)
            ],
            dtype=mx.int32,
        )
        hidden = self.drafter.forward_block(token_ids, context_cache)
        head_hidden = hidden[:, : self.proposal_count]
        base_logits = self.drafter.base_logits(hidden, self.proposal_count)
        tokens, corrected_logits = self.drafter.sample_greedy_markov(
            base_logits, anchor_token
        )
        confidence_logits = self.drafter.confidence_logits(
            head_hidden, anchor_token, tokens
        )
        return KimiK3DSparkProposal(
            tokens=tokens,
            base_logits=base_logits,
            corrected_logits=corrected_logits,
            confidence_logits=confidence_logits,
            verify_width=self.verify_width,
            draft_block_width=self.drafter.args.block_size,
            mode=self.mode,
        )


@dataclass(frozen=True)
class KimiK3DSparkCheckpointFile:
    filename: str
    size: int
    sha256: str


RADIXARK_KIMI_K3_DSPARK_RUNTIME_FILES = (
    KimiK3DSparkCheckpointFile(
        "config.json",
        RADIXARK_KIMI_K3_DSPARK_CONFIG_BYTES,
        RADIXARK_KIMI_K3_DSPARK_CONFIG_SHA256,
    ),
    KimiK3DSparkCheckpointFile(
        "model.safetensors",
        RADIXARK_KIMI_K3_DSPARK_WEIGHTS_BYTES,
        RADIXARK_KIMI_K3_DSPARK_WEIGHTS_SHA256,
    ),
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def attest_kimi_k3_dspark_file(
    path: Path,
    manifest: KimiK3DSparkCheckpointFile,
    *,
    verify_sha256: bool = True,
) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Kimi K3 DSpark file is missing or unsafe: {path}")
    size = path.stat().st_size
    if size != manifest.size:
        raise ValueError(
            f"Kimi K3 DSpark {manifest.filename} must be {manifest.size} bytes, "
            f"got {size}"
        )
    if verify_sha256:
        actual = _sha256_file(path)
        if actual != manifest.sha256:
            raise ValueError(
                f"Kimi K3 DSpark {manifest.filename} SHA256 does not match"
            )


def attest_kimi_k3_dspark_directory(
    checkpoint_dir: str | Path,
    *,
    verify_weights_sha256: bool = True,
) -> dict[str, Any]:
    """Attest the two runtime files without executing checkpoint Python."""

    directory = Path(checkpoint_dir)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Kimi K3 DSpark checkpoint directory is missing or unsafe")
    safetensors = sorted(path.name for path in directory.glob("*.safetensors"))
    if safetensors != ["model.safetensors"]:
        raise ValueError("Kimi K3 DSpark requires exactly model.safetensors")
    for manifest in RADIXARK_KIMI_K3_DSPARK_RUNTIME_FILES:
        attest_kimi_k3_dspark_file(
            directory / manifest.filename,
            manifest,
            verify_sha256=(manifest.filename == "config.json" or verify_weights_sha256),
        )
    with (directory / "config.json").open() as stream:
        config = json.load(stream)
    KimiK3DSparkContract.from_config(config)
    return config


def load_kimi_k3_dspark(
    checkpoint_dir: str | Path,
    target_model,
    *,
    verify_weights_sha256: bool = True,
) -> KimiK3DSparkModel:
    """Load the exact pinned BF16 drafter and bind borrowed target modules."""

    if not kimi_k3_dspark_proposer_enabled():
        raise RuntimeError(
            f"Kimi K3 DSpark loader is disabled; set {DSPARK_PROPOSER_ENV}=1"
        )
    directory = Path(checkpoint_dir)
    config = attest_kimi_k3_dspark_directory(
        directory,
        verify_weights_sha256=verify_weights_sha256,
    )
    args = KimiK3DSparkArgs.from_config(config)
    drafter = KimiK3DSparkModel(args)
    weights = mx.load(str(directory / "model.safetensors"))
    drafter.load_owned_weights(weights)
    drafter.eval()
    mx.eval(drafter.parameters())
    return drafter.bind_target(target_model)
