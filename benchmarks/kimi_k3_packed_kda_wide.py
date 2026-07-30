#!/usr/bin/env python3
"""Benchmark authoritative K3 QKV/gate packing after skinny packing."""

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
    AuthoritativePackedK3KDAWide,
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


def _arrays(projections):
    return tuple(
        value
        for projection in projections
        for value in (
            projection.weight,
            projection.scales,
            projection.biases,
        )
    )


def _measure_storage(input_width: int, qkv_width: int, gate_width: int) -> dict:
    gc.collect()
    mx.clear_cache()
    baseline = mx.get_active_memory()
    projections = (
        _Projection(input_width, qkv_width, 0x12345678),
        _Projection(input_width, gate_width, 0x89ABCDEF),
    )
    mx.eval(*_arrays(projections))
    gc.collect()
    mx.clear_cache()
    source_active = mx.get_active_memory()
    source_nbytes = sum(int(value.nbytes) for value in _arrays(projections))
    mx.reset_peak_memory()

    started = time.perf_counter()
    packed = AuthoritativePackedK3KDAWide(projections)
    mx.synchronize()
    install_ms = 1000.0 * (time.perf_counter() - started)
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
    return 1000.0 * (time.perf_counter() - started) / iterations


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
    parser.add_argument("--qkv-width", type=int, default=18432)
    parser.add_argument("--gate-width", type=int, default=6144)
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
    if args.qkv_width != 3 * args.gate_width:
        raise SystemExit("--qkv-width must be three times --gate-width")

    memory = _measure_storage(
        args.input_width,
        args.qkv_width,
        args.gate_width,
    )
    tolerance = 2 * 1024 * 1024
    if memory["steady_extra_bytes"] > tolerance:
        raise SystemExit("authoritative wide pack retained a duplicate")
    if memory["transient_peak_extra_bytes"] < memory["packed_nbytes"] - tolerance:
        raise SystemExit("storage instrumentation missed pack installation")

    qkv = _Projection(args.input_width, args.qkv_width, 0x12345678)
    gate = _Projection(args.input_width, args.gate_width, 0x89ABCDEF)
    fa = _Projection(args.input_width, args.fa_width, 0x24681357)
    beta = _Projection(args.input_width, args.b_width, 0x13572468)
    fb = _Projection(args.fa_width, args.fb_width, 0x10293847)
    skinny = AuthoritativePackedK3KDASkinny((fa, beta))
    wide = AuthoritativePackedK3KDAWide((qkv, gate))
    x = mx.random.normal((1, 1, args.input_width)).astype(mx.bfloat16)
    mx.eval(x, skinny.parameters(), wide.parameters(), fb.weight, fb.scales, fb.biases)

    def stock_region(value):
        skinny_fa, skinny_beta = skinny(value)
        return qkv(value), gate(value), fb(skinny_fa), skinny_beta

    def packed_region(value):
        packed_qkv, packed_gate = wide(value)
        skinny_fa, skinny_beta = skinny(value)
        return packed_qkv, packed_gate, fb(skinny_fa), skinny_beta

    def stock_call():
        return stock_region(x)

    def packed_call():
        return packed_region(x)

    compiled_stock = mx.compile(stock_region)
    compiled_packed = mx.compile(packed_region)

    def compiled_stock_call():
        return compiled_stock(x)

    def compiled_packed_call():
        return compiled_packed(x)

    eager_exact = _bit_exact(packed_call(), stock_call())
    compiled_exact = _bit_exact(compiled_packed_call(), compiled_stock_call())
    cross_exact = _bit_exact(compiled_packed_call(), stock_call())
    if not eager_exact or not compiled_exact or not cross_exact:
        raise SystemExit("wide pack differs from the skinny-only reference")

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
    saved_ms = compiled["stock_median_ms"] - compiled["packed_median_ms"]
    result = {
        "shape": {
            "input_width": args.input_width,
            "qkv_width": args.qkv_width,
            "gate_width": args.gate_width,
            "fa_width": args.fa_width,
            "b_width": args.b_width,
            "fb_width": args.fb_width,
            "quantization": "affine-6/group-64",
        },
        "comparison": "skinny-only versus skinny-plus-wide",
        "correctness": {
            "eager_bit_exact": eager_exact,
            "compiled_bit_exact": compiled_exact,
            "compiled_vs_eager_bit_exact": cross_exact,
        },
        "memory": memory,
        "benchmark": {
            "warmup": args.warmup,
            "iterations": args.iterations,
            "trials": args.trials,
            "eager": eager,
            "compiled_consumer_region": compiled,
            "compiled_saved_ms_per_kda_layer": saved_ms,
            "projected_saved_ms_per_69_kda_layers": 69 * saved_ms,
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
