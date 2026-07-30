"""Microbenchmark Kimi K3's stock and fused selected-expert down tails.

Run on an Apple Silicon host with MLX:

    python benchmarks/kimi_k3_fused_down_reduce.py --iterations 200

The benchmark uses the exact TP2 rank-local K3 geometry but only sixteen
expert banks, because a token can read no more than those sixteen banks.
"""

from __future__ import annotations

import argparse
import gc
import statistics
import time
from collections.abc import Callable

import mlx.core as mx

from mlx_lm.models.kimi_k3_fused_down_reduce import (
    K3_DOWN_INPUT_WIDTH,
    K3_DOWN_OUTPUT_WIDTH,
    K3_TOP_K,
    fused_down_reduce_decode,
)
from mlx_lm.models.kimi_k3_tuned_gather_qmv import tuned_gather_qmv

MEBIBYTE = 1024 * 1024


def packed_projection() -> tuple[mx.array, mx.array, mx.array]:
    weight = mx.full(
        (K3_TOP_K, K3_DOWN_OUTPUT_WIDTH, K3_DOWN_INPUT_WIDTH // 16),
        0x24681357,
        dtype=mx.uint32,
    )
    scales = mx.full(
        (K3_TOP_K, K3_DOWN_OUTPUT_WIDTH, K3_DOWN_INPUT_WIDTH // 128),
        0.015625,
        dtype=mx.bfloat16,
    )
    return weight, scales, mx.zeros_like(scales)


def synchronize() -> None:
    mx.synchronize()


def interleaved_median_ms(
    operations: dict[str, Callable[[], mx.array]],
    *,
    warmup: int,
    iterations: int,
    trials: int,
) -> dict[str, float]:
    for operation in operations.values():
        for _ in range(warmup):
            mx.eval(operation())
    synchronize()

    samples = {name: [] for name in operations}
    ordered = list(operations.items())
    for trial in range(trials):
        offset = trial % len(ordered)
        rotated = ordered[offset:] + ordered[:offset]
        for name, operation in rotated:
            started = time.perf_counter_ns()
            for _ in range(iterations):
                mx.eval(operation())
            synchronize()
            elapsed_ms = (time.perf_counter_ns() - started) / iterations / 1e6
            samples[name].append(elapsed_ms)
    return {
        name: statistics.median(measurements)
        for name, measurements in samples.items()
    }


def peak_delta_bytes(operation: Callable[[], mx.array]) -> tuple[int, int]:
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()
    baseline = mx.get_active_memory()
    output = operation()
    mx.eval(output)
    synchronize()
    return mx.get_peak_memory() - baseline, mx.get_active_memory() - baseline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--trials", type=int, default=9)
    parser.add_argument(
        "--tiles",
        type=int,
        choices=(2, 4, 8),
        nargs="+",
        default=(2, 4, 8),
        help="output rows computed by each selected-expert SIMD group",
    )
    parser.add_argument(
        "--simdgroups",
        type=int,
        choices=(4, 8, 16),
        nargs="+",
        default=(4, 8, 16),
        help="SIMD groups per threadgroup",
    )
    args = parser.parse_args()

    mx.random.seed(47)
    projection = packed_projection()
    activated = mx.random.normal(
        (1, 1, K3_TOP_K, 1, K3_DOWN_INPUT_WIDTH),
        dtype=mx.bfloat16,
    )
    indices = mx.arange(K3_TOP_K, dtype=mx.uint32).reshape(1, 1, K3_TOP_K)
    router_weights = mx.random.uniform(
        shape=(1, 1, K3_TOP_K),
        dtype=mx.bfloat16,
    )
    mx.eval(activated, indices, router_weights, *projection)

    def stock() -> mx.array:
        expert_outputs = mx.gather_qmm(
            activated,
            *projection,
            rhs_indices=indices,
            transpose=True,
            group_size=128,
            bits=2,
            mode="affine",
        ).squeeze(-2)
        return (expert_outputs * router_weights[..., None]).sum(axis=-2)

    def tuned() -> mx.array:
        expert_outputs = tuned_gather_qmv(
            activated,
            indices,
            projection,
            results_per_simdgroup=4,
            simdgroups=2,
            broadcast_x=False,
        ).squeeze(-2)
        return (expert_outputs * router_weights[..., None]).sum(axis=-2)

    def fused(
        results_per_threadgroup: int,
        simdgroups_per_threadgroup: int,
    ) -> mx.array:
        return fused_down_reduce_decode(
            activated,
            indices,
            router_weights,
            projection,
            results_per_threadgroup=results_per_threadgroup,
            simdgroups_per_threadgroup=simdgroups_per_threadgroup,
        )

    reference = stock()
    current_candidate = tuned()
    configurations = [
        (tile, simdgroups)
        for tile in args.tiles
        for simdgroups in args.simdgroups
    ]
    candidates = [fused(*configuration) for configuration in configurations]
    mx.eval(reference, current_candidate, *candidates)
    if not bool(mx.all(reference == current_candidate).item()):
        raise RuntimeError("current tuned down path is not bit exact")
    for configuration, candidate in zip(configurations, candidates, strict=True):
        if not bool(mx.all(reference == candidate).item()):
            raise RuntimeError(f"configuration {configuration} is not bit exact")

    operations = {"stock": stock, "tuned": tuned}
    for tile, simdgroups in configurations:
        name = f"R{tile}/S{simdgroups}"
        operations[name] = lambda tile=tile, simdgroups=simdgroups: fused(
            tile,
            simdgroups,
        )
    timings = interleaved_median_ms(
        operations,
        warmup=args.warmup,
        iterations=args.iterations,
        trials=args.trials,
    )
    stock_ms = timings["stock"]
    tuned_ms = timings["tuned"]
    fused_times = {
        configuration: timings[f"R{configuration[0]}/S{configuration[1]}"]
        for configuration in configurations
    }
    best_configuration = min(fused_times, key=fused_times.__getitem__)
    fused_ms = fused_times[best_configuration]
    tuned_peak, tuned_active = peak_delta_bytes(tuned)
    fused_peak, fused_active = peak_delta_bytes(
        lambda: fused(*best_configuration)
    )
    materialized_bytes = K3_TOP_K * K3_DOWN_OUTPUT_WIDTH * 2

    print("Kimi K3 TP2 selected-expert down + route reduction")
    print(f"native MLX median:  {stock_ms:.6f} ms")
    print(f"current tuned path: {tuned_ms:.6f} ms")
    for (tile, simdgroups), elapsed_ms in fused_times.items():
        print(
            f"fused R{tile}/S{simdgroups}: {elapsed_ms:.6f} ms "
            f"({tuned_ms / elapsed_ms:.4f}x vs tuned, "
            f"{tuned_ms - elapsed_ms:.6f} ms saved)"
        )
    print(
        "best launch:  "
        f"R{best_configuration[0]}/S{best_configuration[1]}"
    )
    print(f"tuned peak delta:  {tuned_peak / MEBIBYTE:.3f} MiB")
    print(f"fused peak delta:  {fused_peak / MEBIBYTE:.3f} MiB")
    print(f"tuned active delta:{tuned_active / MEBIBYTE:9.3f} MiB")
    print(f"fused active delta:{fused_active / MEBIBYTE:9.3f} MiB")
    print(
        "eliminated expert-row tensor: "
        f"{materialized_bytes / MEBIBYTE:.3f} MiB/layer"
    )
    print(
        "92-layer isolated projection: "
        f"{92 * (tuned_ms - fused_ms):.3f} ms/token"
    )


if __name__ == "__main__":
    main()
