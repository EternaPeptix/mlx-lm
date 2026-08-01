"""Single-device compute screen for exact K3 TP2 routed-up column sharding."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import mlx.core as mx

from mlx_lm.models.kimi_k3_fused_routed_up_add import (
    K3_BITS,
    K3_GROUP_SIZE,
    K3_HIDDEN_SIZE,
    K3_ROUTED_LATENT_SIZE,
    K3_TP2_LOCAL_HIDDEN_SIZE,
    fused_routed_up_add,
)
from mlx_lm.models.kimi_k3_tp2_routed_up_column import RoutedUpColumnCost


def _projection(rows: int):
    weight = mx.full(
        (rows, K3_ROUTED_LATENT_SIZE * K3_BITS // 32),
        0xD3917A5C,
        dtype=mx.uint32,
    )
    scale_shape = (rows, K3_ROUTED_LATENT_SIZE // K3_GROUP_SIZE)
    scales = mx.random.uniform(0.001, 0.02, scale_shape).astype(mx.bfloat16)
    biases = mx.random.uniform(-0.05, 0.05, scale_shape).astype(mx.bfloat16)
    return weight, scales, biases


def _dependent_chain(calls, initial: mx.array, iterations: int) -> mx.array:
    value = initial
    for index in range(iterations):
        value = calls[index % len(calls)](value)
    return value


def _time_ms(calls, initial: mx.array, iterations: int) -> float:
    result = _dependent_chain(calls, initial, iterations)
    mx.synchronize()
    started = time.perf_counter_ns()
    mx.eval(result)
    mx.synchronize()
    return (time.perf_counter_ns() - started) / iterations / 1e6


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=800)
    parser.add_argument("--warmup", type=int, default=150)
    parser.add_argument("--trials", type=int, default=21)
    parser.add_argument(
        "--banks",
        type=int,
        default=8,
        help="distinct projection banks cycled to exceed the system cache",
    )
    parser.add_argument(
        "--all-gather-us",
        type=float,
        default=53.230,
        help="provisional JACCL latency; hardware all-gather remains a gate",
    )
    parser.add_argument("--baseline-ms-per-token", type=float, default=69.893)
    args = parser.parse_args()
    if min(args.iterations, args.warmup, args.trials, args.banks) <= 0:
        raise ValueError("benchmark counts must be positive")
    if args.all_gather_us < 0:
        raise ValueError("all-gather latency cannot be negative")

    mx.set_default_device(mx.gpu)
    mx.random.seed(20260801)
    routed = mx.random.normal(
        (1, 1, K3_ROUTED_LATENT_SIZE), dtype=mx.bfloat16
    )
    shared = mx.random.normal((1, 1, K3_HIDDEN_SIZE), dtype=mx.bfloat16)
    residual = mx.random.normal((1, 1, K3_HIDDEN_SIZE), dtype=mx.bfloat16)
    full_parameter_banks = [_projection(K3_HIDDEN_SIZE) for _ in range(args.banks)]
    half_parameter_banks = [
        tuple(
            mx.contiguous(value[:K3_TP2_LOCAL_HIDDEN_SIZE])
            for value in parameters
        )
        for parameters in full_parameter_banks
    ]
    mx.eval(
        routed,
        shared,
        residual,
        *[value for bank in full_parameter_banks for value in bank],
        *[value for bank in half_parameter_banks for value in bank],
    )

    def make_full(parameters):
        def full(value):
            output = fused_routed_up_add(
                value,
                shared,
                residual,
                parameters,
            )
            return output[..., :K3_ROUTED_LATENT_SIZE]

        return full

    def make_half(parameters):
        def half(value):
            return fused_routed_up_add(
                value,
                shared[..., :K3_TP2_LOCAL_HIDDEN_SIZE],
                residual[..., :K3_TP2_LOCAL_HIDDEN_SIZE],
                parameters,
            )

        return half

    full_calls = [make_full(parameters) for parameters in full_parameter_banks]
    half_calls = [make_half(parameters) for parameters in half_parameter_banks]

    expected_half = full_calls[0](routed)
    actual_half = half_calls[0](routed)
    mx.eval(expected_half, actual_half)
    if not bool(mx.array_equal(expected_half, actual_half).item()):
        raise RuntimeError("half projection does not match the full output slice")
    warm_full = _dependent_chain(full_calls, routed, args.warmup)
    warm_half = _dependent_chain(half_calls, routed, args.warmup)
    mx.eval(warm_full, warm_half)
    mx.synchronize()

    samples = {"full_ms": [], "half_ms": []}
    paired_deltas = []
    for trial in range(args.trials):
        order = (("full_ms", full_calls), ("half_ms", half_calls))
        if trial % 2:
            order = tuple(reversed(order))
        pair = {}
        for name, calls in order:
            pair[name] = _time_ms(calls, routed, args.iterations)
            samples[name].append(pair[name])
        paired_deltas.append(pair["full_ms"] - pair["half_ms"])

    medians = {name: statistics.median(values) for name, values in samples.items()}
    cost = RoutedUpColumnCost(
        full_projection_ms=medians["full_ms"],
        half_projection_ms=medians["half_ms"],
        all_gather_ms=args.all_gather_us / 1000.0,
    )
    result = {
        "geometry": {
            "routed_latent": K3_ROUTED_LATENT_SIZE,
            "full_hidden": K3_HIDDEN_SIZE,
            "local_hidden": K3_TP2_LOCAL_HIDDEN_SIZE,
            "quantization": "affine8/group64",
            "dtype_boundary": "FP32 accumulation -> BF16 -> BF16 adds",
            "sparse_layers": cost.sparse_layers,
        },
        "protocol": vars(args),
        "exact_half_slice": True,
        "samples_ms": samples,
        "medians_ms": medians,
        "median_paired_compute_saving_ms": statistics.median(paired_deltas),
        "paired_wins": sum(value > 0 for value in paired_deltas),
        "break_even_all_gather_us": 1000.0 * cost.break_even_all_gather_ms,
        "provisional_all_gather_us": args.all_gather_us,
        "projected_saved_ms_per_token": cost.saved_ms_per_token,
        "projected_tokens_per_second": cost.projected_tokens_per_second(
            args.baseline_ms_per_token
        ),
        "hardware_gate": (
            "measure dependency-serialized 14336-byte BF16 JACCL all-gather "
            "p50/p95 and full-model output identity on both M3 Ultra ranks"
        ),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
