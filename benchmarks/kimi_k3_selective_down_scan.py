#!/usr/bin/env python3
"""Read-only exact scan of Kimi K3 routed down-projection affine metadata."""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path

import numpy as np

PREFIX = "language_model.model.layers."
SUFFIXES = (
    ".mlp.switch_mlp.down_proj.scales",
    ".mlp.switch_mlp.down_proj.biases",
)


def read_header(path: Path) -> tuple[int, dict[str, object]]:
    with path.open("rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise ValueError(f"short safetensors prefix: {path}")
        (size,) = struct.unpack("<Q", raw)
        if size > 256 * 1024 * 1024:
            raise ValueError(f"implausible header size {size}: {path}")
        payload = handle.read(size)
        if len(payload) != size:
            raise ValueError(f"short safetensors header: {path}")
    return 8 + size, json.loads(payload)


def catalog(root: Path) -> dict[tuple[int, str], dict[str, object]]:
    result: dict[tuple[int, str], dict[str, object]] = {}
    for path in sorted(root.glob("*.safetensors")):
        data_start, header = read_header(path)
        for name, descriptor in header.items():
            if not name.startswith(PREFIX):
                continue
            kind = None
            for suffix, candidate in zip(SUFFIXES, ("scales", "biases")):
                if name.endswith(suffix):
                    kind = candidate
                    layer_text = name[len(PREFIX) : -len(suffix)]
                    break
            if kind is None or not layer_text.isdigit():
                continue
            if descriptor["dtype"] != "BF16":
                raise ValueError(f"unexpected dtype for {name}: {descriptor['dtype']}")
            result[(int(layer_text), kind)] = {
                "path": path,
                "name": name,
                "data_start": data_start,
                "offsets": tuple(int(v) for v in descriptor["data_offsets"]),
                "shape": tuple(int(v) for v in descriptor["shape"]),
            }
    return result


def read_u16(entry: dict[str, object], start: int, count: int) -> np.ndarray:
    offsets = entry["offsets"]
    with entry["path"].open("rb") as handle:
        handle.seek(entry["data_start"] + offsets[0] + start * 2)
        payload = handle.read(count * 2)
    if len(payload) != count * 2:
        raise ValueError(f"short metadata read: {entry['name']}")
    return np.frombuffer(payload, dtype="<u2")


def expected_bias_bits(scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw = scales.astype(np.uint32, copy=False)
    magnitude = raw & np.uint32(0x7FFF)
    exponent = magnitude & np.uint32(0x7F80)
    doubled = np.where(
        exponent == 0,
        magnitude << np.uint32(1),
        np.where(
            exponent < np.uint32(0x7F00),
            magnitude + np.uint32(0x0080),
            np.where(
                exponent == np.uint32(0x7F00),
                np.uint32(0x7F80),
                magnitude,
            ),
        ),
    )
    flipped_sign = (raw ^ np.uint32(0x8000)) & np.uint32(0x8000)
    expected = (flipped_sign | doubled).astype(np.uint16)
    fast = (exponent != 0) & (exponent != np.uint32(0x7F80))
    return expected, fast


def bf16_float(bits: int) -> float:
    return float(np.array([bits << 16], dtype=np.uint32).view(np.float32)[0])


def coordinate(shape: tuple[int, ...], flat_index: int) -> dict[str, int]:
    if len(shape) != 3:
        return {"flat_index": flat_index}
    experts, rows, groups = shape
    per_expert = rows * groups
    expert, within = divmod(flat_index, per_expert)
    row, group = divmod(within, groups)
    if expert >= experts:
        raise ValueError("flat index exceeds tensor shape")
    return {
        "flat_index": flat_index,
        "expert": expert,
        "row": row,
        "group": group,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--chunk-pairs", type=int, default=4_194_304)
    parser.add_argument("--max-details", type=int, default=32)
    args = parser.parse_args()

    entries = catalog(args.root)
    present_layers = sorted(layer for layer, kind in entries if kind == "scales")
    if args.layers == "all":
        layers = present_layers
    else:
        layers = [int(value) for value in args.layers.split(",") if value]

    records = []
    bytes_read = 0
    for layer in layers:
        scales_entry = entries[(layer, "scales")]
        biases_entry = entries[(layer, "biases")]
        if scales_entry["shape"] != biases_entry["shape"]:
            raise ValueError(f"shape mismatch at layer {layer}")
        shape = scales_entry["shape"]
        count = math.prod(shape)
        relation_mismatches = 0
        non_fast_scales = 0
        details = []
        for start in range(0, count, args.chunk_pairs):
            chunk_count = min(args.chunk_pairs, count - start)
            scales = read_u16(scales_entry, start, chunk_count)
            biases = read_u16(biases_entry, start, chunk_count)
            bytes_read += chunk_count * 4
            expected, fast = expected_bias_bits(scales)
            mismatch = expected != biases
            relation_mismatches += int(np.count_nonzero(mismatch))
            non_fast_scales += int(np.count_nonzero(~fast))
            room = args.max_details - len(details)
            if room > 0 and np.any(mismatch | ~fast):
                local_indices = np.flatnonzero(mismatch | ~fast)[:room]
                for local in local_indices:
                    flat = start + int(local)
                    scale_bits = int(scales[local])
                    bias_bits = int(biases[local])
                    expected_bits = int(expected[local])
                    details.append(
                        {
                            **coordinate(shape, flat),
                            "scale_bits": f"0x{scale_bits:04x}",
                            "scale": bf16_float(scale_bits),
                            "bias_bits": f"0x{bias_bits:04x}",
                            "bias": bf16_float(bias_bits),
                            "expected_bias_bits": f"0x{expected_bits:04x}",
                            "expected_bias": bf16_float(expected_bits),
                            "relation_mismatch": bool(mismatch[local]),
                            "fast_scale": bool(fast[local]),
                        }
                    )
        records.append(
            {
                "layer": layer,
                "shape": shape,
                "pairs": count,
                "relation_mismatches": relation_mismatches,
                "non_fast_scales": non_fast_scales,
                "fast_derivable": relation_mismatches == 0 and non_fast_scales == 0,
                "details": details,
            }
        )

    print(
        json.dumps(
            {
                "root": str(args.root),
                "layers_scanned": len(records),
                "metadata_bytes_read": bytes_read,
                "invalid_modules": [
                    record for record in records if not record["fast_derivable"]
                ],
                "valid_modules": sum(record["fast_derivable"] for record in records),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
