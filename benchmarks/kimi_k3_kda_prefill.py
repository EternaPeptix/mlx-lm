"""Microbenchmark the exact Kimi K3 KDA row-tiled Metal prototype.

This measures only the post-projection KDA recurrence. It does not measure the
model's projections, MoE layers, EXO transport, tokenization, or sampling.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time

import mlx.core as mx

from mlx_lm.models.gated_delta import (
    experimental_kda_row_prefill_kernel,
    gated_delta_kernel,
)


def _make_inputs(tokens: int, heads: int):
    batch, dim = 1, 128
    shape = (batch, tokens, heads, dim)
    mx.random.seed(1729 + tokens)
    q = (mx.random.normal(shape) / dim**0.5).astype(mx.bfloat16)
    k = (mx.random.normal(shape) / dim**0.5).astype(mx.bfloat16)
    v = (0.05 * mx.random.normal(shape)).astype(mx.bfloat16)
    gate = mx.full(shape, 0.98, dtype=mx.float32)
    beta = mx.full(shape[:3], 0.4, dtype=mx.bfloat16)
    state = mx.zeros((batch, heads, dim, dim), dtype=mx.float32)
    mx.eval(q, k, v, gate, beta, state)
    return q, k, v, gate, beta, state


def _measure(fn, *, warmup: int, repeats: int):
    for _ in range(warmup):
        result = fn()
        mx.eval(*result)
        mx.synchronize()

    samples = []
    for _ in range(repeats):
        mx.synchronize()
        start = time.perf_counter()
        result = fn()
        mx.eval(*result)
        mx.synchronize()
        samples.append(1_000.0 * (time.perf_counter() - start))
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def _max_abs(a, b) -> float:
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())


def _flashkda_workspace_bytes(heads: int, window_chunks: int = 32) -> int:
    """Peak bytes for the audited bounded-window FlashKDA K1/K2 prototype."""

    chunk, dim = 16, 128
    persistent_elements = (
        heads * window_chunks * (3 * chunk * dim + 2 * chunk * chunk + chunk + dim)
    )
    transient_elements = heads * window_chunks * (chunk * dim + chunk * chunk)
    return persistent_elements * 2 + transient_elements * 4


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=[128, 512, 2048])
    parser.add_argument("--heads", type=int, default=96)
    parser.add_argument("--rows", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=11)
    args = parser.parse_args()

    if not mx.metal.is_available():
        print(json.dumps({"status": "skipped", "reason": "Metal unavailable"}))
        return 0

    report = {
        "status": "ok",
        "hardware": platform.machine(),
        "heads": args.heads,
        "head_dim": 128,
        "dtype": "bfloat16 inputs / float32 state",
        "extra_workspace": {
            "current_recurrent_bytes": 0,
            "row_tiled_bytes": 0,
            "flashkda_window_512_tokens_bytes": _flashkda_workspace_bytes(args.heads),
        },
        "cases": [],
    }

    for tokens in args.tokens:
        inputs = _make_inputs(tokens, args.heads)
        current_fn = lambda: gated_delta_kernel(*inputs)
        current = current_fn()
        mx.eval(*current)
        case = {
            "tokens": tokens,
            "current": _measure(
                current_fn,
                warmup=args.warmup,
                repeats=args.repeats,
            ),
            "row_tiled": {},
        }

        for rows in args.rows:
            candidate_fn = lambda rows=rows: experimental_kda_row_prefill_kernel(
                *inputs,
                rows_per_simd=rows,
            )
            candidate = candidate_fn()
            mx.eval(*candidate)
            timing = _measure(
                candidate_fn,
                warmup=args.warmup,
                repeats=args.repeats,
            )
            timing["speedup"] = case["current"]["median_ms"] / timing["median_ms"]
            timing["output_max_abs_error"] = _max_abs(current[0], candidate[0])
            timing["state_max_abs_error"] = _max_abs(current[1], candidate[1])
            timing["output_bit_exact"] = bool(mx.all(current[0] == candidate[0]).item())
            timing["state_bit_exact"] = bool(mx.all(current[1] == candidate[1]).item())
            case["row_tiled"][str(rows)] = timing
        report["cases"].append(case)

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
