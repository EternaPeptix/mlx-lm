"""Benchmark exact Kimi K3 AttnRes + RMSNorm decode fusion.

Run on Apple Silicon with the opt-in MLX-LM checkout on ``PYTHONPATH``:

    python benchmarks/kimi_k3_attnres_rms.py \
        --warmup 20 --iterations 300 --trials 15

The released 93-layer model uses ``attn_res_block_size=12``.  A decode token
therefore crosses 185 fuseable AttnRes→RMSNorm boundaries: 24 at each stored
residual count 1 through 7, and 17 at residual count 8.
"""

from __future__ import annotations

import argparse
import gc
import os
import statistics
import time
from collections.abc import Callable
from functools import partial

import mlx.core as mx

from mlx_lm.models.kimi_k3 import ResidualBlocks, _attn_res_mix
from mlx_lm.models.kimi_k3_attnres_rms import (
    FUSED_ATTNRES_RMS_ENV,
    K3_HIDDEN_SIZE,
    fused_attnres_rms_enabled,
    maybe_fused_attnres_rms,
)


EPS = 1e-5
MEBIBYTE = 1024 * 1024
BOUNDARIES_BY_RESIDUAL_COUNT = {
    **{residual_count: 24 for residual_count in range(1, 8)},
    8: 17,
}


def inputs_for(
    residual_count: int,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    raw = mx.random.normal(
        (residual_count, 1, 1, K3_HIDDEN_SIZE),
        dtype=mx.bfloat16,
    )
    raw_float = raw.astype(mx.float32)
    inv_rms = mx.rsqrt((raw_float * raw_float).mean(axis=-1) + EPS)
    partial_sum = mx.random.normal(
        (1, 1, K3_HIDDEN_SIZE),
        dtype=mx.bfloat16,
    )
    w_eff = mx.random.normal((K3_HIDDEN_SIZE,), dtype=mx.float32)
    norm_weight = mx.random.normal(
        (K3_HIDDEN_SIZE,),
        dtype=mx.bfloat16,
    )
    mx.eval(raw, inv_rms, partial_sum, w_eff, norm_weight)
    return raw, inv_rms, partial_sum, w_eff, norm_weight


@partial(mx.compile, shapeless=False)
def stock(
    raw: mx.array,
    inv_rms: mx.array,
    partial_sum: mx.array,
    w_eff: mx.array,
    norm_weight: mx.array,
) -> mx.array:
    blocks = ResidualBlocks(EPS)
    blocks.raw = raw
    blocks.inv_rms = inv_rms
    mixed = _attn_res_mix(
        blocks,
        partial_sum,
        w_eff,
        EPS,
        use_kernel=True,
    )
    return mx.fast.rms_norm(mixed, norm_weight, EPS)


@partial(mx.compile, shapeless=False)
def fused(
    raw: mx.array,
    inv_rms: mx.array,
    partial_sum: mx.array,
    w_eff: mx.array,
    norm_weight: mx.array,
) -> mx.array:
    result = maybe_fused_attnres_rms(
        raw,
        inv_rms,
        partial_sum,
        w_eff,
        norm_weight,
        EPS,
    )
    if result is None:
        raise RuntimeError("fused AttnRes + RMSNorm geometry was rejected")
    return result


def interleaved_medians_ms(
    operations: dict[str, Callable[[], mx.array]],
    *,
    warmup: int,
    iterations: int,
    trials: int,
) -> dict[str, float]:
    for operation in operations.values():
        for _ in range(warmup):
            mx.eval(operation())
    mx.synchronize()

    samples = {name: [] for name in operations}
    ordered = list(operations.items())
    for trial in range(trials):
        offset = trial % len(ordered)
        rotated = ordered[offset:] + ordered[:offset]
        for name, operation in rotated:
            started = time.perf_counter_ns()
            for _ in range(iterations):
                mx.eval(operation())
            mx.synchronize()
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
    mx.synchronize()
    return mx.get_peak_memory() - baseline, mx.get_active_memory() - baseline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--trials", type=int, default=15)
    args = parser.parse_args()

    os.environ[FUSED_ATTNRES_RMS_ENV] = "1"
    fused_attnres_rms_enabled.cache_clear()
    mx.set_default_device(mx.gpu)
    mx.random.seed(109)

    stock_ms = {}
    fused_ms = {}
    values_by_count = {}
    for residual_count in BOUNDARIES_BY_RESIDUAL_COUNT:
        values = inputs_for(residual_count)
        values_by_count[residual_count] = values
        reference = stock(*values)
        candidate = fused(*values)
        mx.eval(reference, candidate)
        if not bool(mx.array_equal(reference, candidate).item()):
            raise RuntimeError(
                f"residual count {residual_count} is not bit exact"
            )

        timings = interleaved_medians_ms(
            {
                "stock": lambda values=values: stock(*values),
                "fused": lambda values=values: fused(*values),
            },
            warmup=args.warmup,
            iterations=args.iterations,
            trials=args.trials,
        )
        stock_ms[residual_count] = timings["stock"]
        fused_ms[residual_count] = timings["fused"]

    peak_values = values_by_count[max(BOUNDARIES_BY_RESIDUAL_COUNT)]
    stock_peak, stock_active = peak_delta_bytes(lambda: stock(*peak_values))
    fused_peak, fused_active = peak_delta_bytes(lambda: fused(*peak_values))

    stock_projection = sum(
        BOUNDARIES_BY_RESIDUAL_COUNT[count] * stock_ms[count]
        for count in BOUNDARIES_BY_RESIDUAL_COUNT
    )
    fused_projection = sum(
        BOUNDARIES_BY_RESIDUAL_COUNT[count] * fused_ms[count]
        for count in BOUNDARIES_BY_RESIDUAL_COUNT
    )

    print("Kimi K3 BF16 T=1 AttnRes + RMSNorm")
    print("K  boundaries  stock_ms  fused_ms  speedup  saved_ms")
    for count, boundaries in BOUNDARIES_BY_RESIDUAL_COUNT.items():
        print(
            f"{count:<2} {boundaries:>10}  "
            f"{stock_ms[count]:>8.6f}  {fused_ms[count]:>8.6f}  "
            f"{stock_ms[count] / fused_ms[count]:>7.4f}x  "
            f"{stock_ms[count] - fused_ms[count]:>8.6f}"
        )
    print(f"stock 93-layer projection: {stock_projection:.6f} ms/token")
    print(f"fused 93-layer projection: {fused_projection:.6f} ms/token")
    print(
        "isolated projected saving: "
        f"{stock_projection - fused_projection:.6f} ms/token"
    )
    print(
        f"isolated projected speedup: {stock_projection / fused_projection:.4f}x"
    )
    print(f"stock peak delta:  {stock_peak / MEBIBYTE:.6f} MiB")
    print(f"fused peak delta:  {fused_peak / MEBIBYTE:.6f} MiB")
    print(f"stock active delta:{stock_active / MEBIBYTE:10.6f} MiB")
    print(f"fused active delta:{fused_active / MEBIBYTE:10.6f} MiB")
    eliminated_bytes = 2 * K3_HIDDEN_SIZE * 2
    eliminated_mebibytes = (
        eliminated_bytes * sum(BOUNDARIES_BY_RESIDUAL_COUNT.values()) / MEBIBYTE
    )
    print(
        "eliminated global traffic: "
        f"{eliminated_bytes} bytes/boundary, "
        f"{eliminated_mebibytes:.3f} MiB/token/rank"
    )
    print(
        "eliminated dispatches: "
        f"{sum(BOUNDARIES_BY_RESIDUAL_COUNT.values())}/token/rank"
    )


if __name__ == "__main__":
    main()
