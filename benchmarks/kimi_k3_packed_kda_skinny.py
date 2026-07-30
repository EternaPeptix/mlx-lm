#!/usr/bin/env python3
"""Benchmark exact K3 TP2 skinny-projection packing at decode geometry."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlx_lm.models.kimi_k3_packed_kda_projections import (
    AuthoritativePackedK3KDASkinny,
)


class _Projection:
    bits = 6
    group_size = 64
    mode = "affine"

    def __init__(self, input_width: int, output_width: int, pattern: int):
        self.weight = mx.full(
            (output_width, input_width * self.bits // 32),
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


def _make_projections(
    input_width: int,
    output_widths: tuple[int, int],
) -> tuple[_Projection, _Projection]:
    return (
        _Projection(input_width, output_widths[0], 0x12345678),
        _Projection(input_width, output_widths[1], 0x89ABCDEF),
    )


def _measure_storage(
    input_width: int,
    output_widths: tuple[int, int],
) -> dict[str, int | float]:
    gc.collect()
    mx.clear_cache()
    baseline = mx.get_active_memory()
    projections = _make_projections(input_width, output_widths)
    mx.eval(*_projection_arrays(projections))
    gc.collect()
    mx.clear_cache()
    source_active = mx.get_active_memory()
    source_nbytes = sum(
        int(value.nbytes) for value in _projection_arrays(projections)
    )
    mx.reset_peak_memory()

    install_started = time.perf_counter()
    packed = AuthoritativePackedK3KDASkinny(projections)
    mx.synchronize()
    install_ms = (time.perf_counter() - install_started) * 1000.0
    gc.collect()
    mx.clear_cache()
    steady_active = mx.get_active_memory()
    peak_active = mx.get_peak_memory()

    result = {
        "baseline_active_bytes": baseline,
        "source_active_bytes": source_active,
        "source_nbytes": source_nbytes,
        "packed_nbytes": packed.packed_nbytes,
        "install_ms": install_ms,
        "steady_active_bytes": steady_active,
        "steady_extra_bytes": steady_active - source_active,
        "peak_active_bytes": peak_active,
        "transient_peak_extra_bytes": peak_active - source_active,
    }
    del packed, projections
    gc.collect()
    mx.clear_cache()
    return result


def _time_ms(call, iterations: int) -> float:
    started = time.perf_counter()
    for _ in range(iterations):
        mx.eval(*call())
    mx.synchronize()
    return (time.perf_counter() - started) * 1000.0 / iterations


def _paired_samples(
    stock,
    packed,
    *,
    warmup: int,
    iterations: int,
    trials: int,
) -> dict:
    for _ in range(warmup):
        mx.eval(*stock(), *packed())
    mx.synchronize()

    stock_samples = []
    packed_samples = []
    paired_speedups = []
    for trial in range(trials):
        arms = (
            (("stock", stock), ("packed", packed))
            if trial % 2 == 0
            else (("packed", packed), ("stock", stock))
        )
        pair = {}
        for name, call in arms:
            pair[name] = _time_ms(call, iterations)
        stock_samples.append(pair["stock"])
        packed_samples.append(pair["packed"])
        paired_speedups.append(pair["stock"] / pair["packed"])

    return {
        "stock_samples_ms": stock_samples,
        "packed_samples_ms": packed_samples,
        "paired_speedups": paired_speedups,
        "stock_median_ms": statistics.median(stock_samples),
        "packed_median_ms": statistics.median(packed_samples),
        "median_paired_speedup": statistics.median(paired_speedups),
    }


def _bit_exact(actual, expected) -> bool:
    mx.eval(*actual, *expected)
    return all(
        bool(mx.array_equal(got, want).item())
        for got, want in zip(actual, expected, strict=True)
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-width", type=int, default=7168)
    parser.add_argument("--fa-width", type=int, default=128)
    parser.add_argument("--b-width", type=int, default=48)
    parser.add_argument("--fb-width", type=int, default=6144)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--trials", type=int, default=9)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not mx.metal.is_available():
        raise SystemExit("Metal is required")
    if args.input_width % 64 or args.input_width * 6 % 32:
        raise SystemExit("--input-width must satisfy group-64 affine-6 packing")
    output_widths = (args.fa_width, args.b_width)
    if any(width <= 0 for width in output_widths) or args.fb_width <= 0:
        raise SystemExit("output widths must be positive")

    memory = _measure_storage(args.input_width, output_widths)
    tolerance = 64 * 1024
    if memory["steady_extra_bytes"] > tolerance:
        raise SystemExit("authoritative storage retained a packed duplicate")
    if memory["transient_peak_extra_bytes"] < memory["packed_nbytes"] - tolerance:
        raise SystemExit("storage instrumentation did not observe pack installation")

    projections = _make_projections(args.input_width, output_widths)
    packed = AuthoritativePackedK3KDASkinny(projections)
    x = mx.random.normal((1, 1, args.input_width)).astype(mx.bfloat16)
    mx.eval(x, packed.parameters())

    def stock_call():
        return tuple(projection(x) for projection in projections)

    def packed_call():
        return packed(x)

    compiled_stock = mx.compile(
        lambda value: tuple(projection(value) for projection in projections)
    )
    compiled_packed = mx.compile(lambda value: packed(value))
    f_b = _Projection(args.fa_width, args.fb_width, 0x24681357)

    def compiled_stock_call():
        return compiled_stock(x)

    def compiled_packed_call():
        return compiled_packed(x)

    def stock_consumer_call():
        f_a = projections[0](x)
        return f_b(f_a), projections[1](x)

    def packed_consumer_call():
        f_a, b = packed(x)
        return f_b(f_a), b

    compiled_stock_consumer = mx.compile(
        lambda value: (f_b(projections[0](value)), projections[1](value))
    )
    compiled_packed_consumer = mx.compile(
        lambda value: (lambda parts: (f_b(parts[0]), parts[1]))(packed(value))
    )

    def compiled_stock_consumer_call():
        return compiled_stock_consumer(x)

    def compiled_packed_consumer_call():
        return compiled_packed_consumer(x)

    stock_output = stock_call()
    packed_output = packed_call()
    compiled_stock_output = compiled_stock_call()
    compiled_packed_output = compiled_packed_call()
    stock_consumer_output = stock_consumer_call()
    packed_consumer_output = packed_consumer_call()
    compiled_stock_consumer_output = compiled_stock_consumer_call()
    compiled_packed_consumer_output = compiled_packed_consumer_call()
    eager_exact = _bit_exact(packed_output, stock_output)
    compiled_exact = _bit_exact(compiled_packed_output, compiled_stock_output)
    cross_exact = _bit_exact(compiled_packed_output, stock_output)
    consumer_eager_exact = _bit_exact(
        packed_consumer_output,
        stock_consumer_output,
    )
    consumer_compiled_exact = _bit_exact(
        compiled_packed_consumer_output,
        compiled_stock_consumer_output,
    )
    if (
        not eager_exact
        or not compiled_exact
        or not cross_exact
        or not consumer_eager_exact
        or not consumer_compiled_exact
    ):
        raise SystemExit("packed skinny output differs from stock QMV")

    eager = _paired_samples(
        stock_call,
        packed_call,
        warmup=args.warmup,
        iterations=args.iterations,
        trials=args.trials,
    )
    compiled = _paired_samples(
        compiled_stock_call,
        compiled_packed_call,
        warmup=args.warmup,
        iterations=args.iterations,
        trials=args.trials,
    )
    compiled_consumer = _paired_samples(
        compiled_stock_consumer_call,
        compiled_packed_consumer_call,
        warmup=args.warmup,
        iterations=args.iterations,
        trials=args.trials,
    )
    saved_ms = (
        compiled_consumer["stock_median_ms"]
        - compiled_consumer["packed_median_ms"]
    )
    result = {
        "shape": {
            "input_width": args.input_width,
            "output_widths": list(output_widths),
            "f_b_output_width": args.fb_width,
            "quantization": "affine-6/group-64",
        },
        "correctness": {
            "eager_bit_exact": eager_exact,
            "compiled_bit_exact": compiled_exact,
            "compiled_vs_eager_bit_exact": cross_exact,
            "consumer_eager_bit_exact": consumer_eager_exact,
            "consumer_compiled_bit_exact": consumer_compiled_exact,
        },
        "memory": memory,
        "benchmark": {
            "warmup": args.warmup,
            "iterations": args.iterations,
            "trials": args.trials,
            "eager": eager,
            "compiled": compiled,
            "compiled_with_f_b_consumer": compiled_consumer,
            "compiled_saved_ms_per_kda_layer": saved_ms,
            "projected_saved_ms_per_69_kda_layers": 69 * saved_ms,
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
