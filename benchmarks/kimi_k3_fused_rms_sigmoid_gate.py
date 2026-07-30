"""Matched compiled microbenchmark for Kimi K3 post-KDA norm/gate fusion."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import mlx.core as mx
import numpy as np

from mlx_lm.models.kimi_k3_fused_rms_sigmoid_gate import (
    FUSED_RMS_SIGMOID_GATE_ENV,
    fused_rms_sigmoid_gate_enabled,
    maybe_fused_rms_sigmoid_gate,
)


_EPS = 1e-5


@mx.compile
def _compiled_reference(x, gate, weight):
    return mx.fast.rms_norm(x, weight, _EPS) * mx.sigmoid(gate)


@mx.compile
def _compiled_fused(x, gate, weight):
    return maybe_fused_rms_sigmoid_gate(
        x,
        gate,
        weight,
        _EPS,
        training=False,
    )


def _one_sample(op, x, gate, weight, depth):
    y = x
    start = time.perf_counter_ns()
    for _ in range(depth):
        y = op(y, gate, weight)
    mx.eval(y)
    mx.synchronize()
    return (time.perf_counter_ns() - start) / depth / 1e6


def _summarize(samples):
    ordered = sorted(samples)
    return {
        "median_ms": statistics.median(samples),
        "minimum_ms": ordered[0],
        "p90_ms": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=69)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--warmup-seconds", type=float, default=1.0)
    parser.add_argument("--trials", type=int, default=201)
    args = parser.parse_args()

    if not mx.metal.is_available():
        raise SystemExit("This benchmark requires an Apple Metal GPU")

    os.environ[FUSED_RMS_SIGMOID_GATE_ENV] = "1"
    fused_rms_sigmoid_gate_enabled.cache_clear()

    mx.random.seed(20260730)
    shape = (1, 1, 96, 128)
    x = mx.random.normal(shape).astype(mx.bfloat16)
    gate = (3.0 * mx.random.normal(shape)).astype(mx.bfloat16)
    weight = mx.random.uniform(low=0.25, high=1.75, shape=(128,)).astype(
        mx.bfloat16
    )
    mx.eval(x, gate, weight)

    expected = _compiled_reference(x, gate, weight)
    actual = _compiled_fused(x, gate, weight)
    mx.eval(expected, actual)
    expected_np = np.asarray(expected.astype(mx.float32))
    actual_np = np.asarray(actual.astype(mx.float32))

    ops = {
        "reference": _compiled_reference,
        "fused": _compiled_fused,
    }
    for _ in range(args.warmup):
        for op in ops.values():
            _one_sample(op, x, gate, weight, args.depth)
    deadline = time.perf_counter() + args.warmup_seconds
    while time.perf_counter() < deadline:
        for op in ops.values():
            _one_sample(op, x, gate, weight, args.depth)

    samples = {name: [] for name in ops}
    for trial in range(args.trials):
        order = ("reference", "fused") if trial % 2 == 0 else ("fused", "reference")
        for name in order:
            samples[name].append(
                _one_sample(ops[name], x, gate, weight, args.depth)
            )

    paired_savings = [
        reference - fused
        for reference, fused in zip(samples["reference"], samples["fused"])
    ]
    paired_speedups = [
        reference / fused
        for reference, fused in zip(samples["reference"], samples["fused"])
    ]
    saved_per_layer = statistics.median(paired_savings)
    report = {
        "device": mx.device_info(),
        "mlx_version": mx.__version__,
        "shape": shape,
        "dtype": "bfloat16",
        "eps": _EPS,
        "serial_depth": args.depth,
        "trials": args.trials,
        "execution": "alternating paired mx.compile paths",
        "correctness": {
            "max_abs_error": float(np.abs(expected_np - actual_np).max()),
            "mean_abs_error": float(np.abs(expected_np - actual_np).mean()),
            "exact_fraction": float((expected_np == actual_np).mean()),
        },
        "reference": _summarize(samples["reference"]),
        "fused": _summarize(samples["fused"]),
        "median_paired_speedup": statistics.median(paired_speedups),
        "median_paired_saved_ms_per_layer": saved_per_layer,
        "projected_saved_ms_per_token": saved_per_layer * args.depth,
    }
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
