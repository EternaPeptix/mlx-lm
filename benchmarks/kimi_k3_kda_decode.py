"""Amortized decode-tile screen for Kimi K3's recurrent KDA kernel.

The benchmark builds a dependent lazy chain before each timed region and
evaluates the whole chain with one final synchronization.  This removes the
roughly 0.2 ms per-call synchronization noise that dominates a T=1 kernel.
It remains a post-projection microbenchmark, not an end-to-end model result.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import statistics
import time
from collections.abc import Callable

import mlx.core as mx

from mlx_lm.models.gated_delta import (
    experimental_kda_row_prefill_kernel,
    gated_delta_kernel,
)

Kernel = Callable[[], tuple[mx.array, mx.array]]


def _make_inputs(heads: int):
    batch, tokens, dim = 1, 1, 128
    shape = (batch, tokens, heads, dim)
    mx.random.seed(1729)
    q = (mx.random.normal(shape) / dim**0.5).astype(mx.bfloat16)
    k = (mx.random.normal(shape) / dim**0.5).astype(mx.bfloat16)
    v = (0.05 * mx.random.normal(shape)).astype(mx.bfloat16)
    gate = mx.full(shape, 0.98, dtype=mx.float32)
    beta = mx.full(shape[:3], 0.4, dtype=mx.bfloat16)
    state = mx.zeros((batch, heads, dim, dim), dtype=mx.float32)
    mx.eval(q, k, v, gate, beta, state)
    return q, k, v, gate, beta, state


def _chain(
    inputs: tuple[mx.array, ...],
    *,
    rows: int | None,
    length: int,
) -> tuple[mx.array, mx.array]:
    q, k, v, gate, beta, state = inputs
    output = q
    for _ in range(length):
        if rows is None:
            output, state = gated_delta_kernel(q, k, v, gate, beta, state)
        else:
            output, state = experimental_kda_row_prefill_kernel(
                q,
                k,
                v,
                gate,
                beta,
                state,
                rows_per_simd=rows,
            )
    return output, state


def _run_once(inputs, *, rows: int | None, chain_length: int) -> float:
    # Exclude Python graph construction, matching a compiled/deferred model
    # graph more closely than synchronizing every individual T=1 invocation.
    output, state = _chain(inputs, rows=rows, length=chain_length)
    mx.synchronize()
    start = time.perf_counter_ns()
    mx.eval(output, state)
    mx.synchronize()
    return (time.perf_counter_ns() - start) / 1_000_000 / chain_length


def _bit_exact(a: mx.array, b: mx.array) -> bool:
    return bool(mx.all(a == b).item())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heads", type=int, default=48)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--chain-length", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--order-seed", type=int, default=314159)
    args = parser.parse_args()

    if not mx.metal.is_available():
        print(json.dumps({"status": "skipped", "reason": "Metal unavailable"}))
        return 0
    if args.heads <= 0 or args.chain_length <= 0 or args.repeats <= 0:
        parser.error("heads, chain-length, and repeats must be positive")
    if any(rows not in (1, 2, 4, 8) for rows in args.rows):
        parser.error("rows must contain only 1, 2, 4, or 8")

    inputs = _make_inputs(args.heads)
    variants: dict[str, int | None] = {"current": None}
    variants.update({f"row{rows}": rows for rows in args.rows})

    for rows in variants.values():
        for _ in range(args.warmup):
            _run_once(inputs, rows=rows, chain_length=args.chain_length)

    samples = {name: [] for name in variants}
    order = list(variants)
    rng = random.Random(args.order_seed)
    for _ in range(args.repeats):
        rng.shuffle(order)
        for name in order:
            samples[name].append(
                _run_once(
                    inputs,
                    rows=variants[name],
                    chain_length=args.chain_length,
                )
            )

    reference = _chain(inputs, rows=None, length=args.chain_length)
    mx.eval(*reference)
    results = {}
    reference_median = statistics.median(samples["current"])
    for name, rows in variants.items():
        candidate = _chain(inputs, rows=rows, length=args.chain_length)
        mx.eval(*candidate)
        ordered = sorted(samples[name])
        median = statistics.median(ordered)
        results[name] = {
            "median_ms_per_kernel": median,
            "min_ms_per_kernel": ordered[0],
            "max_ms_per_kernel": ordered[-1],
            "p10_ms_per_kernel": ordered[len(ordered) // 10],
            "p90_ms_per_kernel": ordered[9 * len(ordered) // 10],
            "speedup": reference_median / median,
            "output_bit_exact": _bit_exact(reference[0], candidate[0]),
            "state_bit_exact": _bit_exact(reference[1], candidate[1]),
        }

    print(
        json.dumps(
            {
                "status": "ok",
                "hardware": platform.machine(),
                "heads": args.heads,
                "head_dim": 128,
                "chain_length": args.chain_length,
                "warmup": args.warmup,
                "repeats": args.repeats,
                "timing_boundary": "one dependent lazy chain and final sync",
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
