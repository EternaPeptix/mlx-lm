#!/usr/bin/env python3
"""Offline, fail-closed smoke test for RadixArk/Kimi-K3-DSpark.

The default mode reads ``config.json`` and only the safetensors header.  It
does not import MLX, instantiate a model, or read tensor payloads.  ``--load``
is the explicit heavyweight path; it imports MLX and evaluates the complete
BF16 drafter through ``load_kimi_k3_dspark``.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence


MODEL_ID = "RadixArk/Kimi-K3-DSpark"
PINNED_REVISION = "eb03982e58d4fb79bcfc099e902158f562e2e27b"
CONFIG_BYTES = 1_288
CONFIG_SHA256 = "6aed20890d95cd69cf2ec006d1f30506fbd4f3091d44ca8e8b93e9fc7d50928f"
MODEL_BYTES = 4_498_585_858
MODEL_SHA256 = "29df0e8eafb81909f785df55cb352b90d6a1500c609b1d60526c1a62b4d42495"
WEIGHT_TENSORS = 62
PARAMETERS = 2_249_289_601
TARGET_LAYERS = (7, 23, 51, 67, 83)
BLOCK_SIZE = 7
_MAX_SAFETENSORS_HEADER_BYTES = 16 * 1024 * 1024


class SmokeError(ValueError):
    """A checkpoint or invocation failed the smoke-test contract."""


def _shape_size(shape: Sequence[int]) -> int:
    return math.prod(shape)


def production_weight_shapes() -> dict[str, tuple[int, ...]]:
    """Return the complete released checkpoint inventory without importing MLX."""

    hidden = 7168
    intermediate = 14336
    query = 64 * 64
    key_value = 16 * 64
    shapes: dict[str, tuple[int, ...]] = {
        "fc.weight": (hidden, len(TARGET_LAYERS) * hidden),
        "hidden_norm.weight": (hidden,),
        "norm.weight": (hidden,),
    }
    for index in range(5):
        prefix = f"layers.{index}"
        shapes.update(
            {
                f"{prefix}.input_layernorm.weight": (hidden,),
                f"{prefix}.self_attn.q_proj.weight": (query, hidden),
                f"{prefix}.self_attn.k_proj.weight": (key_value, hidden),
                f"{prefix}.self_attn.v_proj.weight": (key_value, hidden),
                f"{prefix}.self_attn.o_proj.weight": (hidden, query),
                f"{prefix}.self_attn.q_norm.weight": (64,),
                f"{prefix}.self_attn.k_norm.weight": (64,),
                f"{prefix}.post_attention_layernorm.weight": (hidden,),
                f"{prefix}.mlp.gate_proj.weight": (intermediate, hidden),
                f"{prefix}.mlp.up_proj.weight": (intermediate, hidden),
                f"{prefix}.mlp.down_proj.weight": (hidden, intermediate),
            }
        )
    shapes.update(
        {
            "markov_head.markov_w1.weight": (163840, 256),
            "markov_head.markov_w2.weight": (163840, 256),
            "confidence_head.proj.weight": (1, 7168 + 256),
            "confidence_head.proj.bias": (1,),
        }
    )
    return shapes


def _require_exact_scalar(
    config: Mapping[str, Any], name: str, expected: Any
) -> None:
    actual = config.get(name)
    if type(actual) is not type(expected) or actual != expected:
        raise SmokeError(
            f"Kimi K3 DSpark {name} must be {expected!r}, got {actual!r}"
        )


def validate_production_config(config: Mapping[str, Any]) -> None:
    """Validate the exact public config using only Python scalar objects."""

    if config.get("architectures") != ["DSparkDraftModel"]:
        raise SmokeError("Kimi K3 DSpark architecture contract does not match")
    expected_scalars = {
        "model_type": "qwen3",
        "block_size": BLOCK_SIZE,
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
        _require_exact_scalar(config, name, expected)
    if config.get("layer_types") != ["full_attention"] * 5:
        raise SmokeError("Kimi K3 DSpark layer_types contract does not match")
    if config.get("rope_parameters") != {
        "rope_theta": 10_000.0,
        "rope_type": "default",
    }:
        raise SmokeError("Kimi K3 DSpark RoPE contract does not match")
    dflash = config.get("dflash_config")
    if not isinstance(dflash, Mapping):
        raise SmokeError("Kimi K3 DSpark dflash_config must be a mapping")
    if tuple(dflash.get("target_layer_ids", ())) != TARGET_LAYERS:
        raise SmokeError("Kimi K3 DSpark target hidden taps do not match")
    _require_exact_scalar(dflash, "mask_token_id", 163824)


@dataclass(frozen=True)
class CheckpointContract:
    model_id: str
    revision: str
    config_bytes: int
    config_sha256: str
    model_bytes: int
    model_sha256: str
    expected_shapes: Mapping[str, tuple[int, ...]]
    expected_dtype: str
    expected_elements: int
    config_validator: Callable[[Mapping[str, Any]], None]


PRODUCTION_CONTRACT = CheckpointContract(
    model_id=MODEL_ID,
    revision=PINNED_REVISION,
    config_bytes=CONFIG_BYTES,
    config_sha256=CONFIG_SHA256,
    model_bytes=MODEL_BYTES,
    model_sha256=MODEL_SHA256,
    expected_shapes=production_weight_shapes(),
    expected_dtype="BF16",
    expected_elements=PARAMETERS,
    config_validator=validate_production_config,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicate_json_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise SmokeError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json_bytes(payload: bytes, description: str) -> Any:
    try:
        return json.loads(
            payload.decode("utf-8"), object_pairs_hook=_reject_duplicate_json_keys
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SmokeError(f"invalid {description}: {error}") from error


def _attest_regular_file(path: Path, description: str) -> int:
    if path.is_symlink() or not path.is_file():
        raise SmokeError(f"{description} is missing or unsafe: {path}")
    return path.stat().st_size


def _local_checkpoint_directory(checkpoint_dir: str | Path) -> Path:
    raw = os.fspath(checkpoint_dir)
    if "://" in raw:
        raise SmokeError(
            "checkpoint must be a local directory; downloads are forbidden"
        )
    path = Path(raw).expanduser()
    if path.is_symlink() or not path.is_dir():
        raise SmokeError(
            f"checkpoint must be an existing non-symlink directory: {path}"
        )
    return path.resolve(strict=True)


def _revision_evidence(
    directory: Path,
    revision: str,
    contract: CheckpointContract,
) -> str:
    if revision != contract.revision:
        raise SmokeError(
            f"revision must be pinned to {contract.revision}, got {revision}"
        )
    parts = directory.parts
    snapshot_indices = [
        index for index, part in enumerate(parts) if part == "snapshots"
    ]
    for index in snapshot_indices:
        if index + 1 >= len(parts):
            raise SmokeError("snapshot path does not contain a revision")
        observed = parts[index + 1]
        if observed != contract.revision:
            raise SmokeError(
                f"snapshot path revision must be {contract.revision}, got {observed}"
            )
        return "huggingface_snapshot_path_and_content_attestation"
    return "explicit_revision_and_content_attestation"


def inspect_safetensors_metadata(
    path: Path,
    expected_shapes: Mapping[str, tuple[int, ...]],
    *,
    expected_dtype: str,
    expected_elements: int,
) -> dict[str, Any]:
    """Validate a safetensors inventory without reading any tensor payload."""

    file_size = _attest_regular_file(path, "model.safetensors")
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise SmokeError("model.safetensors is missing its 8-byte header length")
        header_size = int.from_bytes(prefix, "little", signed=False)
        if not 2 <= header_size <= _MAX_SAFETENSORS_HEADER_BYTES:
            raise SmokeError(f"unsafe safetensors header size: {header_size}")
        if 8 + header_size > file_size:
            raise SmokeError("safetensors header extends beyond the file")
        header_bytes = stream.read(header_size)
        if len(header_bytes) != header_size:
            raise SmokeError("safetensors header is truncated")

    header = _read_json_bytes(header_bytes, "safetensors header JSON")
    if not isinstance(header, Mapping):
        raise SmokeError("safetensors header must be a JSON object")
    tensor_entries = {
        key: value for key, value in header.items() if key != "__metadata__"
    }
    actual_keys = set(tensor_entries)
    expected_keys = set(expected_shapes)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    if missing or unexpected:
        raise SmokeError(
            "safetensors keys do not match: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )

    inventory: dict[str, dict[str, Any]] = {}
    intervals: list[tuple[int, int, str]] = []
    total_elements = 0
    for name, expected_shape in expected_shapes.items():
        entry = tensor_entries[name]
        if not isinstance(entry, Mapping):
            raise SmokeError(f"safetensors tensor {name} metadata must be an object")
        dtype = entry.get("dtype")
        if dtype != expected_dtype:
            raise SmokeError(
                f"safetensors tensor {name} must have dtype {expected_dtype}, "
                f"got {dtype}"
            )
        shape = entry.get("shape")
        if (
            not isinstance(shape, list)
            or any(type(dimension) is not int or dimension < 0 for dimension in shape)
        ):
            raise SmokeError(f"safetensors tensor {name} has an invalid shape")
        actual_shape = tuple(shape)
        if actual_shape != tuple(expected_shape):
            raise SmokeError(
                f"safetensors tensor {name} must have shape {tuple(expected_shape)}, "
                f"got {actual_shape}"
            )
        offsets = entry.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(offset) is not int or offset < 0 for offset in offsets)
            or offsets[1] < offsets[0]
        ):
            raise SmokeError(f"safetensors tensor {name} has invalid data offsets")
        elements = _shape_size(actual_shape)
        expected_bytes = elements * 2  # exact BF16 inventory
        if offsets[1] - offsets[0] != expected_bytes:
            raise SmokeError(
                f"safetensors tensor {name} payload must be {expected_bytes} bytes"
            )
        total_elements += elements
        intervals.append((offsets[0], offsets[1], name))
        inventory[name] = {
            "shape": list(actual_shape),
            "dtype": dtype,
            "elements": elements,
            "data_offsets": offsets,
        }

    if total_elements != expected_elements:
        raise SmokeError(
            f"safetensors element count must be {expected_elements}, "
            f"got {total_elements}"
        )
    cursor = 0
    for start, end, name in sorted(intervals):
        if start != cursor:
            raise SmokeError(
                f"safetensors payload is non-contiguous before {name}: "
                f"expected offset {cursor}, got {start}"
            )
        cursor = end
    payload_bytes = file_size - 8 - header_size
    if cursor != payload_bytes:
        raise SmokeError(
            f"safetensors payload coverage must be {payload_bytes} bytes, got {cursor}"
        )

    return {
        "header_bytes": header_size,
        "payload_bytes": payload_bytes,
        "tensor_count": len(inventory),
        "element_count": total_elements,
        "dtype": expected_dtype,
        "tensors": inventory,
    }


def inspect_checkpoint(
    checkpoint_dir: str | Path,
    *,
    revision: str,
    verify_model_sha256: bool,
    contract: CheckpointContract = PRODUCTION_CONTRACT,
) -> dict[str, Any]:
    """Run the allocation-free local checkpoint attestation."""

    directory = _local_checkpoint_directory(checkpoint_dir)
    revision_evidence = _revision_evidence(directory, revision, contract)
    safetensors = sorted(path.name for path in directory.glob("*.safetensors"))
    if safetensors != ["model.safetensors"]:
        raise SmokeError("checkpoint requires exactly model.safetensors")

    config_path = directory / "config.json"
    config_size = _attest_regular_file(config_path, "config.json")
    if config_size != contract.config_bytes:
        raise SmokeError(
            f"config.json must be {contract.config_bytes} bytes, got {config_size}"
        )
    config_sha256 = _sha256_file(config_path)
    if config_sha256 != contract.config_sha256:
        raise SmokeError("config.json SHA256 does not match the pinned revision")
    config = _read_json_bytes(config_path.read_bytes(), "config.json")
    if not isinstance(config, Mapping):
        raise SmokeError("config.json must be a JSON object")
    contract.config_validator(config)

    model_path = directory / "model.safetensors"
    model_size = _attest_regular_file(model_path, "model.safetensors")
    if model_size != contract.model_bytes:
        raise SmokeError(
            f"model.safetensors must be {contract.model_bytes} bytes, got {model_size}"
        )
    model_sha256 = None
    if verify_model_sha256:
        model_sha256 = _sha256_file(model_path)
        if model_sha256 != contract.model_sha256:
            raise SmokeError(
                "model.safetensors SHA256 does not match the pinned revision"
            )

    metadata = inspect_safetensors_metadata(
        model_path,
        contract.expected_shapes,
        expected_dtype=contract.expected_dtype,
        expected_elements=contract.expected_elements,
    )
    return {
        "model_id": contract.model_id,
        "revision": contract.revision,
        "revision_evidence": revision_evidence,
        "checkpoint_dir": str(directory),
        "config": {
            "bytes": config_size,
            "sha256": config_sha256,
            "sha256_verified": True,
        },
        "model_file": {
            "bytes": model_size,
            "expected_sha256": contract.model_sha256,
            "sha256": model_sha256,
            "sha256_verified": verify_model_sha256,
        },
        "safetensors": metadata,
    }


class _ShapeOnlyWeight:
    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape


class _BorrowedModuleStub:
    """Satisfy loader binding without allocating target embedding/head tensors."""

    def __init__(self):
        self.weight = _ShapeOnlyWeight((163840, 7168))

    def __call__(self, _):
        raise RuntimeError(
            "target weights are intentionally absent in checkpoint smoke"
        )


def _target_stub():
    return SimpleNamespace(
        language_model=SimpleNamespace(
            model=SimpleNamespace(embed_tokens=_BorrowedModuleStub()),
            lm_head=_BorrowedModuleStub(),
            args=SimpleNamespace(tie_word_embeddings=False),
        )
    )


def _load_runtime():
    """Import the heavyweight runtime only for the explicit load path."""

    repository_root = str(Path(__file__).resolve().parents[1])
    if repository_root not in sys.path:
        sys.path.insert(0, repository_root)
    import mlx.core as mx
    from mlx.utils import tree_flatten

    from mlx_lm.models import kimi_k3_dspark as dspark

    return mx, tree_flatten, dspark


def _validate_runtime_contract(dspark) -> None:
    expected = {
        "RADIXARK_KIMI_K3_DSPARK_MODEL": MODEL_ID,
        "RADIXARK_KIMI_K3_DSPARK_REVISION": PINNED_REVISION,
        "RADIXARK_KIMI_K3_DSPARK_CONFIG_SHA256": CONFIG_SHA256,
        "RADIXARK_KIMI_K3_DSPARK_WEIGHTS_SHA256": MODEL_SHA256,
        "RADIXARK_KIMI_K3_DSPARK_CONFIG_BYTES": CONFIG_BYTES,
        "RADIXARK_KIMI_K3_DSPARK_WEIGHTS_BYTES": MODEL_BYTES,
        "RADIXARK_KIMI_K3_DSPARK_WEIGHT_TENSORS": WEIGHT_TENSORS,
        "RADIXARK_KIMI_K3_DSPARK_PARAMETERS": PARAMETERS,
    }
    for name, value in expected.items():
        if getattr(dspark, name, None) != value:
            raise SmokeError(f"runtime DSpark contract drifted at {name}")
    runtime_shapes = dspark.kimi_k3_dspark_expected_weight_shapes(
        dspark.KimiK3DSparkArgs()
    )
    if runtime_shapes != production_weight_shapes():
        raise SmokeError("runtime DSpark weight inventory drifted")


def _synthetic_values(mx, shape: tuple[int, ...], salt: int):
    total = math.prod(shape)
    values = mx.arange(total, dtype=mx.int32)
    values = ((values * (2 * salt + 1) + 17 * salt) % 257).astype(mx.float32)
    return ((values - 128.0) / 256.0).astype(mx.bfloat16).reshape(shape)


def _run_synthetic_backbone(drafter, mx, *, gamma: int) -> dict[str, Any]:
    if gamma not in (2, 7):
        raise SmokeError("synthetic gamma must be 7 or explicit screening gamma 2")
    args = drafter.args
    exact_geometry = (
        args.hidden_size == 7168
        and args.intermediate_size == 14336
        and args.num_hidden_layers == 5
        and args.num_attention_heads == 64
        and args.num_key_value_heads == 16
        and args.head_dim == 64
        and args.block_size == BLOCK_SIZE
        and tuple(args.target_layer_ids) == TARGET_LAYERS
        and args.weight_dtype == mx.bfloat16
    )
    if not exact_geometry:
        raise SmokeError("loaded drafter does not have exact K3 DSpark BF16 geometry")

    context_width = 1
    mx.reset_peak_memory()
    baseline_active = int(mx.get_active_memory())
    total_started = time.perf_counter()
    context_cache = drafter.make_context_cache()
    target_taps = tuple(
        _synthetic_values(mx, (1, context_width, args.hidden_size), index + 1)
        for index in range(len(args.target_layer_ids))
    )
    drafter.append_target_context(
        target_taps,
        0,
        context_cache,
        use_stacked_context_kv=False,
    )
    hidden = _synthetic_values(mx, (1, BLOCK_SIZE, args.hidden_size), 97)
    mx.eval(
        hidden,
        *[cache.keys for cache in context_cache],
        *[cache.values for cache in context_cache],
    )
    mx.synchronize()
    setup_seconds = time.perf_counter() - total_started

    backbone_started = time.perf_counter()
    for layer, cache in zip(drafter.layers, context_cache, strict=True):
        hidden = layer(hidden, context_width, cache)
    full_output = drafter.norm(hidden)
    selected_output = full_output[:, :gamma]
    mx.eval(selected_output)
    mx.synchronize()
    backbone_seconds = time.perf_counter() - backbone_started
    total_seconds = time.perf_counter() - total_started
    if selected_output.dtype != mx.bfloat16:
        raise SmokeError(
            f"synthetic backbone output must be BF16, got {selected_output.dtype}"
        )

    # BF16 values round-trip exactly through float32; this avoids depending on
    # NumPy's optional native bfloat16 support.
    import numpy as np

    digest_values = np.asarray(
        selected_output.astype(mx.float32), dtype=np.float32
    ).astype("<f4", copy=False)
    digest_payload = digest_values.tobytes(order="C")
    digest = hashlib.sha256(digest_payload).hexdigest()
    result = {
        "mode": "gamma7_native" if gamma == 7 else "gamma2_screening",
        "screening_override": gamma == 2,
        "execution_width": BLOCK_SIZE,
        "selected_width": gamma,
        "context_width": context_width,
        "target_tap_count": len(target_taps),
        "target_tap_shape": [1, context_width, args.hidden_size],
        "noise_embedding_shape": [1, BLOCK_SIZE, args.hidden_size],
        "full_output_shape": list(full_output.shape),
        "output_shape": list(selected_output.shape),
        "output_dtype": "BF16",
        "output_sha256_float32_le": digest,
        "setup_seconds": setup_seconds,
        "backbone_seconds": backbone_seconds,
        "total_seconds": total_seconds,
        "active_memory_bytes": int(mx.get_active_memory()),
        "active_memory_delta_bytes": int(mx.get_active_memory()) - baseline_active,
        "peak_memory_bytes": int(mx.get_peak_memory()),
        "peak_memory_delta_bytes": int(mx.get_peak_memory()) - baseline_active,
    }
    del (
        selected_output,
        full_output,
        hidden,
        target_taps,
        context_cache,
        digest_values,
        digest_payload,
    )
    return result


def run_load_smoke(
    checkpoint_dir: Path,
    *,
    verify_model_sha256: bool,
    synthetic_forward: bool,
    synthetic_gamma: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Load, optionally execute, and always release the full drafter."""

    mx, tree_flatten, dspark = _load_runtime()
    _validate_runtime_contract(dspark)
    drafter = None
    weights = None
    synthetic = None
    load_report: dict[str, Any] = {}
    previous_gate = os.environ.get(dspark.DSPARK_PROPOSER_ENV)
    try:
        gc.collect()
        mx.clear_cache()
        mx.synchronize()
        active_before = int(mx.get_active_memory())
        mx.reset_peak_memory()
        os.environ[dspark.DSPARK_PROPOSER_ENV] = "1"
        started = time.perf_counter()
        drafter = dspark.load_kimi_k3_dspark(
            checkpoint_dir,
            _target_stub(),
            verify_weights_sha256=verify_model_sha256,
        )
        mx.eval(drafter.parameters())
        mx.synchronize()
        load_seconds = time.perf_counter() - started
        weights = dict(tree_flatten(drafter.parameters()))
        element_count = dspark.attest_kimi_k3_dspark_weights(
            weights, drafter.args
        )
        mx.eval(*weights.values())
        mx.synchronize()
        active_loaded = int(mx.get_active_memory())
        peak_loaded = int(mx.get_peak_memory())
        load_report = {
            "seconds": load_seconds,
            "tensor_count": len(weights),
            "element_count": element_count,
            "parameter_dtype": "BF16",
            "active_memory_before_bytes": active_before,
            "active_memory_loaded_bytes": active_loaded,
            "active_memory_delta_bytes": active_loaded - active_before,
            "peak_memory_bytes": peak_loaded,
            "peak_memory_delta_bytes": peak_loaded - active_before,
        }
        if synthetic_forward:
            synthetic = _run_synthetic_backbone(
                drafter, mx, gamma=synthetic_gamma
            )
    finally:
        if previous_gate is None:
            os.environ.pop(dspark.DSPARK_PROPOSER_ENV, None)
        else:
            os.environ[dspark.DSPARK_PROPOSER_ENV] = previous_gate
        weights = None
        drafter = None
        gc.collect()
        mx.clear_cache()
        mx.synchronize()
        active_after_release = int(mx.get_active_memory())

    load_report["release"] = {
        "active_memory_after_bytes": active_after_release,
        "active_memory_retained_bytes": active_after_release
        - load_report["active_memory_before_bytes"],
        "active_memory_returned_to_baseline": active_after_release
        <= load_report["active_memory_before_bytes"],
        "python_model_references_dropped": True,
        "cache_cleared": True,
    }
    if not load_report["release"]["active_memory_returned_to_baseline"]:
        raise SmokeError(
            "Kimi K3 DSpark load smoke retained active memory after release"
        )
    return load_report, synthetic


def run_checkpoint_smoke(
    checkpoint_dir: str | Path,
    *,
    revision: str = PINNED_REVISION,
    verify_model_sha256: bool = False,
    load: bool = False,
    synthetic_forward: bool = False,
    synthetic_gamma: int = 7,
    contract: CheckpointContract = PRODUCTION_CONTRACT,
) -> dict[str, Any]:
    if synthetic_forward and not load:
        raise SmokeError("--synthetic-forward requires explicit --load")
    metadata = inspect_checkpoint(
        checkpoint_dir,
        revision=revision,
        verify_model_sha256=verify_model_sha256,
        contract=contract,
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "ok": True,
        "mode": "load" if load else "metadata",
        "offline": True,
        "metadata_only_allocates_tensors": False,
        "checkpoint": metadata,
    }
    if load:
        if contract is not PRODUCTION_CONTRACT:
            raise SmokeError(
                "load mode only accepts the production checkpoint contract"
            )
        load_report, synthetic = run_load_smoke(
            Path(metadata["checkpoint_dir"]),
            verify_model_sha256=verify_model_sha256,
            synthetic_forward=synthetic_forward,
            synthetic_gamma=synthetic_gamma,
        )
        result["load"] = load_report
        if synthetic is not None:
            result["synthetic_forward"] = synthetic
    return result


def _write_json(payload: Mapping[str, Any], destination: str, *, pretty: bool) -> None:
    serialized = json.dumps(
        payload,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
        sort_keys=True,
    )
    if destination == "-":
        print(serialized)
        return
    path = Path(destination).expanduser()
    if path.exists() and path.is_symlink():
        raise SmokeError(f"refusing to write JSON through symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline RadixArk/Kimi-K3-DSpark checkpoint smoke test.\n"
            "Metadata-only/no tensor allocation is the default."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  python3 benchmarks/kimi_k3_dspark_checkpoint_smoke.py /local/checkpoint
  python3 benchmarks/kimi_k3_dspark_checkpoint_smoke.py /local/checkpoint \\
    --verify-model-sha256 --json-output smoke.json --pretty
  python3 benchmarks/kimi_k3_dspark_checkpoint_smoke.py /local/checkpoint \\
    --load --synthetic-forward
  python3 benchmarks/kimi_k3_dspark_checkpoint_smoke.py /local/checkpoint \\
    --load --synthetic-forward --synthetic-gamma 2

Gamma 2 is a screening view of the first two outputs; the exact seven-position
bidirectional backbone still executes in full. No mode downloads files.
""",
    )
    parser.add_argument("checkpoint_dir", help="existing local checkpoint directory")
    parser.add_argument("--revision", default=PINNED_REVISION)
    parser.add_argument(
        "--verify-model-sha256",
        action="store_true",
        help="stream and hash the 4.5 GB model file (off by default)",
    )
    parser.add_argument(
        "--load",
        action="store_true",
        help="explicitly allocate and evaluate the complete BF16 drafter",
    )
    parser.add_argument(
        "--synthetic-forward",
        action="store_true",
        help="after --load, run the backbone with synthetic taps/embeddings",
    )
    parser.add_argument(
        "--synthetic-gamma",
        type=int,
        choices=(7, 2),
        default=7,
        help="native gamma 7 (default) or explicit gamma 2 screening",
    )
    parser.add_argument(
        "--json-output",
        default="-",
        metavar="PATH",
        help="JSON destination, or - for stdout (default)",
    )
    parser.add_argument("--pretty", action="store_true", help="indent JSON output")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = run_checkpoint_smoke(
            args.checkpoint_dir,
            revision=args.revision,
            verify_model_sha256=args.verify_model_sha256,
            load=args.load,
            synthetic_forward=args.synthetic_forward,
            synthetic_gamma=args.synthetic_gamma,
        )
        _write_json(result, args.json_output, pretty=args.pretty)
        return 0
    except Exception as error:
        failure = {
            "schema_version": 1,
            "ok": False,
            "mode": "load" if args.load else "metadata",
            "offline": True,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        try:
            _write_json(failure, args.json_output, pretty=args.pretty)
        except Exception as output_error:
            print(json.dumps(failure, sort_keys=True), file=sys.stderr)
            print(f"JSON output failure: {output_error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
