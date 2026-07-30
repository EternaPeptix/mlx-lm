#!/usr/bin/env python3

"""Create a file-backed MXFP4 checkpoint for CUDA sorted GatherQMM."""

import argparse
import json
import os
import shutil
from pathlib import Path

import mlx.core as mx

from .models.switch_layers import (
    SORTED_QMM_PACKED_CONFIG_KEY,
    SORTED_QMM_PACKED_FORMAT_VERSION,
    SORTED_QMM_PACKED_LAYOUT,
    _pack_mxfp4_scales_for_sorted_qmm,
    _pack_mxfp4_weight_for_sorted_qmm,
)


def _eligible_pair(weights: dict[str, mx.array], weight_key: str) -> bool:
    if ".switch_mlp." not in weight_key or not weight_key.endswith(".weight"):
        return False
    scales_key = weight_key.removesuffix(".weight") + ".scales"
    if scales_key not in weights:
        return False
    weight = weights[weight_key]
    scales = weights[scales_key]
    return (
        weight.dtype == mx.uint32
        and scales.dtype == mx.uint8
        and weight.ndim == 3
        and scales.ndim == 3
        and weight.shape[-2] % 256 == 0
        and weight.shape[-1] % 8 == 0
        and scales.shape[-2] % 256 == 0
        and scales.shape[-1] % 2 == 0
    )


def pack_safetensor(source: Path, destination: Path) -> list[str]:
    """Pack eligible tensors in one shard and atomically save it."""
    weights = mx.load(str(source))
    packed_keys = []
    for weight_key in sorted(weights):
        if not _eligible_pair(weights, weight_key):
            continue
        scales_key = weight_key.removesuffix(".weight") + ".scales"
        packed_weight = _pack_mxfp4_weight_for_sorted_qmm(
            weights[weight_key]
        )
        packed_scales = _pack_mxfp4_scales_for_sorted_qmm(
            weights[scales_key]
        )
        # Bound each CUDA graph to one projection. Deferring all projections
        # until save time creates a graph large enough to exceed CUDA graph
        # node limits on GB10.
        mx.eval(packed_weight, packed_scales)
        weights[weight_key] = packed_weight
        weights[scales_key] = packed_scales
        packed_keys.extend((weight_key, scales_key))

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        destination.stem + ".partial.safetensors"
    )
    mx.save_safetensors(
        str(temporary),
        weights,
        metadata={"format": "mlx"},
    )
    os.replace(temporary, destination)
    del weights
    mx.clear_cache()
    return packed_keys


def pack_checkpoint(source: Path, destination: Path) -> dict:
    """Convert a checkpoint directory without publishing a partial result."""
    source = source.resolve()
    destination = destination.resolve()
    if source == destination:
        raise ValueError("source and destination must be different directories")
    if not (source / "config.json").is_file():
        raise FileNotFoundError(f"missing config.json in {source}")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"destination is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    for path in source.iterdir():
        if path.name == "config.json" or path.suffix == ".safetensors":
            continue
        if path.is_file():
            shutil.copy2(path, destination / path.name)

    packed_keys = []
    shard_names = sorted(path.name for path in source.glob("model*.safetensors"))
    if not shard_names:
        raise FileNotFoundError(f"no model safetensors in {source}")
    for index, name in enumerate(shard_names, start=1):
        print(f"[{index}/{len(shard_names)}] packing {name}", flush=True)
        packed_keys.extend(
            pack_safetensor(source / name, destination / name)
        )

    if not packed_keys:
        raise ValueError("checkpoint contains no eligible MXFP4 switch weights")

    with open(source / "config.json") as file:
        config = json.load(file)
    config[SORTED_QMM_PACKED_CONFIG_KEY] = {
        "format_version": SORTED_QMM_PACKED_FORMAT_VERSION,
        "layout": SORTED_QMM_PACKED_LAYOUT,
        "packed_tensor_count": len(packed_keys),
        "source": str(source),
    }
    temporary_config = destination / "config.json.partial"
    with open(temporary_config, "w") as file:
        json.dump(config, file, indent=2)
        file.write("\n")
    os.replace(temporary_config, destination / "config.json")
    return config[SORTED_QMM_PACKED_CONFIG_KEY]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    metadata = pack_checkpoint(args.source, args.destination)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
