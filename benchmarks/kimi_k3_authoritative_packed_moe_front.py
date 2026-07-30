#!/usr/bin/env python3
"""Benchmark duplicate and authoritative Kimi K3 MoE-front packing."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx

from mlx_lm.models.kimi_k3_packed_moe_front import (
    AuthoritativePackedK3MoEFront,
    PackedK3MoEFront,
)


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


def _projections(
    input_width: int,
    output_widths: tuple[int, int, int, int],
) -> tuple[_Projection, ...]:
    patterns = (0x12345678, 0x89ABCDEF, 0x24681357, 0xA5A5A5A5)
    return tuple(
        _Projection(input_width, width, pattern)
        for width, pattern in zip(output_widths, patterns, strict=True)
    )


def _projection_arrays(projections: tuple[_Projection, ...]):
    return tuple(
        value
        for projection in projections
        for value in (
            projection.weight,
            projection.scales,
            projection.biases,
        )
    )


def _measure_storage(
    input_width: int,
    output_widths: tuple[int, int, int, int],
    *,
    authoritative: bool,
) -> dict[str, int | float]:
    gc.collect()
    mx.clear_cache()
    baseline = mx.get_active_memory()
    projections = _projections(input_width, output_widths)
    source_arrays = _projection_arrays(projections)
    mx.eval(*source_arrays)
    del source_arrays
    gc.collect()
    mx.clear_cache()
    source_active = mx.get_active_memory()
    source_nbytes = sum(int(value.nbytes) for value in _projection_arrays(projections))
    mx.reset_peak_memory()

    install_started = time.perf_counter()
    front = (
        AuthoritativePackedK3MoEFront(projections)
        if authoritative
        else PackedK3MoEFront(projections)
    )
    mx.eval(front.parameters())
    mx.synchronize()
    install_ms = (time.perf_counter() - install_started) * 1000.0
    gc.collect()
    mx.clear_cache()
    steady_active = mx.get_active_memory()
    peak_active = mx.get_peak_memory()
    packed_nbytes = front.packed_nbytes

    result = {
        "baseline_active_bytes": baseline,
        "source_active_bytes": source_active,
        "source_nbytes": source_nbytes,
        "packed_nbytes": packed_nbytes,
        "install_ms": install_ms,
        "steady_active_bytes": steady_active,
        "steady_extra_bytes": steady_active - source_active,
        "peak_active_bytes": peak_active,
        "transient_peak_extra_bytes": peak_active - source_active,
    }
    del front, projections
    gc.collect()
    mx.clear_cache()
    return result


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
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--trials", type=int, default=11)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not mx.metal.is_available():
        raise SystemExit("Metal is required")
    if args.input_width % 64:
        raise SystemExit("--input-width must be divisible by 64")
    if any(width <= 0 for width in args.output_widths):
        raise SystemExit("every output width must be positive")
    output_widths = tuple(args.output_widths)

    duplicate_memory = _measure_storage(
        args.input_width,
        output_widths,
        authoritative=False,
    )
    authoritative_memory = _measure_storage(
        args.input_width,
        output_widths,
        authoritative=True,
    )
    tolerance = 64 * 1024
    if (
        authoritative_memory["steady_extra_bytes"] > tolerance
        or duplicate_memory["steady_extra_bytes"]
        < duplicate_memory["packed_nbytes"] - tolerance
    ):
        raise SystemExit("packed storage residency invariant failed")

    projections = _projections(args.input_width, output_widths)
    duplicate = PackedK3MoEFront(projections)
    authoritative = AuthoritativePackedK3MoEFront(projections)
    x = mx.random.normal((1, 1, args.input_width)).astype(mx.bfloat16)
    mx.eval(x, duplicate.parameters(), authoritative.parameters())

    def stock_call():
        return tuple(projection(x) for projection in projections)

    def duplicate_call():
        return duplicate(x)

    def authoritative_call():
        return authoritative(x)

    reference = stock_call()
    duplicate_output = duplicate_call()
    authoritative_output = authoritative_call()
    mx.eval(*reference, *duplicate_output, *authoritative_output)
    duplicate_exact = all(
        bool(mx.array_equal(want, got).item())
        for want, got in zip(reference, duplicate_output, strict=True)
    )
    authoritative_exact = all(
        bool(mx.array_equal(want, got).item())
        for want, got in zip(reference, authoritative_output, strict=True)
    )
    if not duplicate_exact or not authoritative_exact:
        raise SystemExit("packed output differs from stock QMV")

    calls = (
        ("stock_ms", stock_call),
        ("duplicate_packed_ms", duplicate_call),
        ("authoritative_packed_ms", authoritative_call),
    )
    for _ in range(args.warmup):
        for _, call in calls:
            mx.eval(*call())
    mx.synchronize()

    samples = {name: [] for name, _ in calls}
    for trial in range(args.trials):
        rotated = calls[trial % len(calls) :] + calls[: trial % len(calls)]
        for name, call in rotated:
            samples[name].append(_time_ms(call, args.iterations))

    medians = {name: statistics.median(values) for name, values in samples.items()}
    stock_ms = medians["stock_ms"]
    authoritative_ms = medians["authoritative_packed_ms"]
    result = {
        "shape": {
            "input_width": args.input_width,
            "output_widths": list(output_widths),
        },
        "benchmark": {
            "warmup": args.warmup,
            "iterations": args.iterations,
            "trials": args.trials,
            "samples_ms": samples,
            "medians_ms": medians,
            "authoritative_speedup_vs_stock": stock_ms / authoritative_ms,
            "authoritative_saved_ms_per_sparse_layer": (stock_ms - authoritative_ms),
            "projected_saved_ms_per_92_layers": (92 * (stock_ms - authoritative_ms)),
        },
        "correctness": {
            "stock_vs_duplicate_bit_exact": duplicate_exact,
            "stock_vs_authoritative_bit_exact": authoritative_exact,
        },
        "memory": {
            "duplicating_packed": duplicate_memory,
            "authoritative_packed": authoritative_memory,
        },
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
