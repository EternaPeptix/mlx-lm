#!/usr/bin/env python3
"""Benchmark Kimi K3's stock, packed, and no-copy MoE-front QMV paths."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx

from mlx_lm.models.kimi_k3_multibank_moe_front import multibank_affine8_qmv
from mlx_lm.models.kimi_k3_packed_moe_front import PackedK3MoEFront


class _Projection:
    bits = 8
    group_size = 64
    mode = "affine"

    def __init__(self, input_width: int, output_width: int, pattern: int):
        self.weight = mx.full(
            (output_width, input_width // 4),
            pattern,
            dtype=mx.uint32,
        )
        self.scales = mx.full(
            (output_width, input_width // self.group_size),
            0.00390625,
            dtype=mx.bfloat16,
        )
        self.biases = mx.full(
            self.scales.shape,
            -0.25,
            dtype=mx.bfloat16,
        )

    def get(self, name: str):
        return getattr(self, name, None)

    def __call__(self, x: mx.array) -> mx.array:
        return mx.quantized_matmul(
            x,
            self.weight,
            scales=self.scales,
            biases=self.biases,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
        )


def _time_ms(call, iterations: int) -> float:
    started = time.perf_counter()
    for _ in range(iterations):
        mx.eval(*call())
    mx.synchronize()
    return (time.perf_counter() - started) * 1000.0 / iterations


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-width", type=int, default=7168)
    parser.add_argument(
        "--output-widths",
        type=int,
        nargs=4,
        default=(3072, 3072, 896, 3584),
        metavar=("SHARED_GATE", "SHARED_UP", "ROUTER", "LATENT"),
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not mx.metal.is_available():
        raise SystemExit("Metal is required")
    if args.input_width % 512:
        raise SystemExit("--input-width must be divisible by 512")
    if any(width <= 0 or width % 8 for width in args.output_widths):
        raise SystemExit("every output width must be positive and divisible by 8")

    patterns = (0x12345678, 0x89ABCDEF, 0x24681357, 0xA5A5A5A5)
    projections = tuple(
        _Projection(args.input_width, width, pattern)
        for width, pattern in zip(args.output_widths, patterns, strict=True)
    )
    banks = tuple(
        (projection.weight, projection.scales, projection.biases)
        for projection in projections
    )
    x = mx.random.normal((1, 1, args.input_width)).astype(mx.bfloat16)
    mx.eval(
        x,
        *(value for bank in banks for value in bank),
    )
    source_nbytes = sum(int(value.nbytes) for bank in banks for value in bank)
    source_active_bytes = mx.get_active_memory()

    def stock_call():
        return tuple(projection(x) for projection in projections)

    def multibank_call():
        return multibank_affine8_qmv(x, banks)

    reference = stock_call()
    candidate = multibank_call()
    mx.eval(*reference, *candidate)
    if not all(
        bool(mx.array_equal(want, got).item())
        for want, got in zip(reference, candidate, strict=True)
    ):
        raise SystemExit("no-copy multi-bank output differs from stock QMV")
    multibank_active_bytes = mx.get_active_memory()

    packed = PackedK3MoEFront(projections)
    packed_reference = packed(x)
    mx.eval(*packed_reference)
    if not all(
        bool(mx.array_equal(want, got).item())
        for want, got in zip(reference, packed_reference, strict=True)
    ):
        raise SystemExit("packed output differs from stock QMV")
    packed_active_bytes = mx.get_active_memory()

    for _ in range(args.warmup):
        mx.eval(*stock_call())
        mx.eval(*multibank_call())
        mx.eval(*packed(x))
    mx.synchronize()

    samples = {"stock_ms": [], "multibank_ms": [], "packed_ms": []}
    calls = (
        ("stock_ms", stock_call),
        ("multibank_ms", multibank_call),
        ("packed_ms", lambda: packed(x)),
    )
    for trial in range(args.trials):
        for name, call in calls[trial % len(calls) :] + calls[: trial % len(calls)]:
            samples[name].append(_time_ms(call, args.iterations))

    medians = {name: statistics.median(values) for name, values in samples.items()}
    stock_ms = medians["stock_ms"]
    multibank_ms = medians["multibank_ms"]
    result = {
        "shape": {
            "input_width": args.input_width,
            "output_widths": list(args.output_widths),
        },
        "benchmark": {
            "warmup": args.warmup,
            "iterations": args.iterations,
            "trials": args.trials,
            "samples_ms": samples,
            "medians_ms": medians,
            "multibank_speedup_vs_stock": stock_ms / multibank_ms,
            "multibank_saved_ms_per_sparse_layer": stock_ms - multibank_ms,
            "projected_saved_ms_per_92_layers": 92 * (stock_ms - multibank_ms),
        },
        "correctness": {
            "stock_vs_multibank_bit_exact": True,
            "stock_vs_packed_bit_exact": True,
        },
        "memory": {
            "authoritative_bank_bytes": source_nbytes,
            "multibank_persistent_duplicate_weight_bytes": 0,
            "packed_persistent_duplicate_weight_bytes": packed.packed_nbytes,
            "active_bytes_after_sources": source_active_bytes,
            "active_bytes_after_multibank": multibank_active_bytes,
            "active_bytes_after_packed": packed_active_bytes,
        },
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
