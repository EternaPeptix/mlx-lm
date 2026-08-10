#!/usr/bin/env python3
"""Fresh-process M3 screen for the exact K3 width-four expert kernels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_lm.models.kimi_k3_derived_bias import (
    affine2_gather_core_available,
    derived_affine2_biases,
)
from mlx_lm.models.kimi_k3_width4_fused_expert import (
    K3_EXPERTS,
    K3_HIDDEN,
    K3_INTERMEDIATE,
    K3_TOP_K,
    K3_WIDTH4,
    width4_down,
    width4_switch_glu,
    width4_switch_glu_reduce,
    width4_switch_situ,
)


@dataclass(frozen=True)
class Projection:
    weight: mx.array
    scales: mx.array
    biases: mx.array

    def parts(self):
        return self.weight, self.scales, self.biases


@dataclass(frozen=True)
class Bank:
    up: Projection
    gate: Projection
    down: Projection


def _projection(input_width: int, output_width: int, salt: int) -> Projection:
    expert = mx.arange(K3_EXPERTS, dtype=mx.uint32).reshape(K3_EXPERTS, 1, 1)
    packed = mx.bitwise_xor(
        expert * mx.array(0x9E3779B9, dtype=mx.uint32),
        mx.array(salt, dtype=mx.uint32),
    )
    weight = mx.contiguous(
        mx.broadcast_to(
            packed,
            (K3_EXPERTS, output_width, input_width // 16),
        )
    )
    scale = (
        mx.array(1 / 128, dtype=mx.float32)
        + (expert % 13).astype(mx.float32)
        * mx.array(1 / 4096, dtype=mx.float32)
    ).astype(mx.bfloat16)
    scales = mx.contiguous(
        mx.broadcast_to(
            scale,
            (K3_EXPERTS, output_width, input_width // 128),
        )
    )
    return Projection(weight, scales, derived_affine2_biases(scales))


def _bank(index: int) -> Bank:
    offset = index * 0x01010101
    return Bank(
        up=_projection(K3_HIDDEN, K3_INTERMEDIATE, 0x13579BDF ^ offset),
        gate=_projection(K3_HIDDEN, K3_INTERMEDIATE, 0x2468ACE0 ^ offset),
        down=_projection(K3_INTERMEDIATE, K3_HIDDEN, 0x55AA55AA ^ offset),
    )


def _qmm(x, indices, projection):
    return mx.gather_qmm(
        x,
        projection.weight,
        projection.scales,
        None,
        rhs_indices=indices,
        transpose=True,
        group_size=128,
        bits=2,
        mode="affine2",
    )


def _measure(
    operations,
    *,
    warmup: int,
    iterations: int,
    trials: int,
    divisor: int,
    scrub,
):
    for operation in operations.values():
        for _ in range(warmup):
            scrub()
            mx.eval(*operation())
    mx.synchronize()
    samples = {name: [] for name in operations}
    for trial in range(trials):
        names = tuple(operations)
        if trial % 2:
            names = tuple(reversed(names))
        for name in names:
            scrub()
            started = time.perf_counter_ns()
            for _ in range(iterations):
                mx.eval(*operations[name]())
            mx.synchronize()
            elapsed = (time.perf_counter_ns() - started) / 1e6
            samples[name].append(elapsed / iterations / divisor)
    return samples


def _summary(control, candidate):
    paired = [
        left - right for left, right in zip(control, candidate, strict=True)
    ]
    control_median = statistics.median(control)
    candidate_median = statistics.median(candidate)
    return {
        "control_median_ms": control_median,
        "candidate_median_ms": candidate_median,
        "independent_median_saving_ms": control_median - candidate_median,
        "speedup": control_median / candidate_median,
        "paired_median_saving_ms": statistics.median(paired),
        "paired_wins": sum(value > 0 for value in paired),
        "trials": len(paired),
        "control_samples_ms": control,
        "candidate_samples_ms": candidate,
    }


def _digest(value: mx.array) -> str:
    return hashlib.sha256(np.asarray(value.view(mx.uint8)).tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--banks", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--front-results", type=int, default=4)
    parser.add_argument("--down-results", type=int, default=8)
    parser.add_argument("--down-simds", type=int, default=8)
    parser.add_argument("--fused-reduce", action="store_true")
    parser.add_argument("--native-down", action="store_true")
    parser.add_argument("--cache-scrub-mib", type=int, default=384)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.banks <= 4:
        raise ValueError("banks must be in [1, 4]")
    if args.front_results not in (2, 4, 8, 16):
        raise ValueError("front-results must be 2, 4, 8, or 16")
    if args.down_results not in (2, 4, 8, 16):
        raise ValueError("down-results must be 2, 4, 8, or 16")
    if args.down_simds not in (4, 8):
        raise ValueError("down-simds must be 4 or 8")
    if args.fused_reduce and args.native_down:
        raise ValueError("fused-reduce and native-down are mutually exclusive")
    if not 64 <= args.cache_scrub_mib <= 1024:
        raise ValueError("cache-scrub-mib must be in [64, 1024]")
    if not affine2_gather_core_available():
        raise RuntimeError("benchmark requires the production affine2 MLX core")

    mx.random.seed(20260810)
    banks = tuple(_bank(index) for index in range(args.banks))
    hidden = mx.random.normal((1, K3_WIDTH4, K3_HIDDEN), dtype=mx.bfloat16)
    base = mx.array([0, 17, 63, 255, 444, 511, 700, 895], dtype=mx.uint32)
    indices = mx.stack([mx.roll(base, shift) for shift in range(K3_WIDTH4)])[None]
    router_weights = mx.random.uniform(
        shape=(1, K3_WIDTH4, K3_TOP_K),
        dtype=mx.bfloat16,
    )
    cache_scrub = mx.random.uniform(
        shape=(args.cache_scrub_mib * 1024 * 1024 // 4,),
        dtype=mx.float32,
    )
    mx.eval(
        hidden,
        indices,
        router_weights,
        cache_scrub,
        *(
            part
            for bank in banks
            for projection in (bank.up, bank.gate, bank.down)
            for part in projection.parts()
        ),
    )

    def scrub():
        # Construct a fresh reduction each time so every measured arm starts
        # after reading a pool substantially larger than M3 Max's system cache.
        mx.eval(mx.sum(cache_scrub))
        mx.synchronize()

    @partial(mx.compile, shapeless=False)
    def stock_front(value, candidate_indices, *parts):
        up = Projection(*parts[:3])
        gate = Projection(*parts[3:])
        expanded = mx.expand_dims(value, (-2, -3))
        up_value = _qmm(expanded, candidate_indices, up).astype(mx.float32)
        gate_value = _qmm(expanded, candidate_indices, gate).astype(mx.float32)
        return (
            4.0
            * mx.tanh(gate_value / 4.0)
            * mx.sigmoid(gate_value)
            * (25.0 * mx.tanh(up_value / 25.0))
        ).astype(mx.bfloat16)

    @partial(mx.compile, shapeless=False)
    def candidate_front(value, candidate_indices, *parts):
        return width4_switch_situ(
            value,
            candidate_indices,
            tuple(parts[:3]),
            tuple(parts[3:]),
            results_per_simdgroup=args.front_results,
            derive_bias=True,
        )

    @partial(mx.compile, shapeless=False)
    def stock_down(value, candidate_indices, weight, scales, biases):
        del biases
        projection = Projection(weight, scales, scales)
        return _qmm(value, candidate_indices, projection)

    @partial(mx.compile, shapeless=False)
    def candidate_down(value, candidate_indices, weight, scales, biases):
        return width4_down(
            value,
            candidate_indices,
            (weight, scales, biases),
            results_per_simdgroup=args.down_results,
            derive_bias=True,
        )

    @partial(mx.compile, shapeless=False)
    def stock_full(value, candidate_indices, route_weights, *parts):
        up = Projection(*parts[:3])
        gate = Projection(*parts[3:6])
        down = Projection(*parts[6:])
        expanded = mx.expand_dims(value, (-2, -3))
        up_value = _qmm(expanded, candidate_indices, up).astype(mx.float32)
        gate_value = _qmm(expanded, candidate_indices, gate).astype(mx.float32)
        activated = (
            4.0
            * mx.tanh(gate_value / 4.0)
            * mx.sigmoid(gate_value)
            * (25.0 * mx.tanh(up_value / 25.0))
        ).astype(mx.bfloat16)
        expert = _qmm(activated, candidate_indices, down).squeeze(-2)
        return (expert * route_weights[..., None]).sum(axis=-2)

    @partial(mx.compile, shapeless=False)
    def candidate_full(value, candidate_indices, route_weights, *parts):
        if args.native_down:
            activated = width4_switch_situ(
                value,
                candidate_indices,
                tuple(parts[:3]),
                tuple(parts[3:6]),
                results_per_simdgroup=args.front_results,
                derive_bias=True,
            )
            down = Projection(*parts[6:])
            expert = _qmm(activated, candidate_indices, down).squeeze(-2)
            return (expert * route_weights[..., None]).sum(axis=-2)
        if args.fused_reduce:
            return width4_switch_glu_reduce(
                value,
                candidate_indices,
                route_weights,
                tuple(parts[:3]),
                tuple(parts[3:6]),
                tuple(parts[6:]),
                front_results_per_simdgroup=args.front_results,
                down_results_per_threadgroup=args.down_results,
                down_simdgroups_per_threadgroup=args.down_simds,
                derive_front_bias=True,
                derive_down_bias=True,
            )
        expert = width4_switch_glu(
            value,
            candidate_indices,
            tuple(parts[:3]),
            tuple(parts[3:6]),
            tuple(parts[6:]),
            front_results_per_simdgroup=args.front_results,
            down_results_per_simdgroup=args.down_results,
            derive_front_bias=True,
            derive_down_bias=True,
        )
        return (expert * route_weights[..., None]).sum(axis=-2)

    front_parts = tuple((*bank.up.parts(), *bank.gate.parts()) for bank in banks)
    front_stock = [stock_front(hidden, indices, *parts) for parts in front_parts]
    front_candidate = [
        candidate_front(hidden, indices, *parts) for parts in front_parts
    ]
    mx.eval(*front_stock, *front_candidate)
    front_exact = all(
        bool(mx.array_equal(left.view(mx.uint8), right.view(mx.uint8)).item())
        for left, right in zip(front_stock, front_candidate, strict=True)
    )
    if not front_exact:
        raise RuntimeError("width-four front is not bit exact")

    down_stock = [
        stock_down(front, indices, *bank.down.parts())
        for front, bank in zip(front_stock, banks, strict=True)
    ]
    down_candidate = [
        candidate_down(front, indices, *bank.down.parts())
        for front, bank in zip(front_stock, banks, strict=True)
    ]
    mx.eval(*down_stock, *down_candidate)
    down_exact = all(
        bool(mx.array_equal(left.view(mx.uint8), right.view(mx.uint8)).item())
        for left, right in zip(down_stock, down_candidate, strict=True)
    )
    if not down_exact:
        raise RuntimeError("width-four down is not bit exact")

    def full_control():
        value = hidden
        outputs = []
        for bank in banks:
            parts = (*bank.up.parts(), *bank.gate.parts(), *bank.down.parts())
            value = stock_full(value, indices, router_weights, *parts)
            outputs.append(value)
        return outputs

    def full_candidate_operation():
        value = hidden
        outputs = []
        for bank in banks:
            parts = (*bank.up.parts(), *bank.gate.parts(), *bank.down.parts())
            value = candidate_full(value, indices, router_weights, *parts)
            outputs.append(value)
        return outputs

    full_stock = full_control()[-1]
    full_candidate = full_candidate_operation()[-1]
    mx.eval(full_stock, full_candidate)
    full_exact = bool(
        mx.array_equal(
            full_stock.view(mx.uint8),
            full_candidate.view(mx.uint8),
        ).item()
    )
    if not full_exact:
        raise RuntimeError("width-four full chain is not bit exact")

    front_timings = _measure(
        {
            "control": lambda: [
                stock_front(hidden, indices, *parts) for parts in front_parts
            ],
            "candidate": lambda: [
                candidate_front(hidden, indices, *parts) for parts in front_parts
            ],
        },
        warmup=args.warmup,
        iterations=args.iterations,
        trials=args.trials,
        divisor=args.banks,
        scrub=scrub,
    )
    down_timings = _measure(
        {
            "control": lambda: [
                stock_down(front, indices, *bank.down.parts())
                for front, bank in zip(front_stock, banks, strict=True)
            ],
            "candidate": lambda: [
                candidate_down(front, indices, *bank.down.parts())
                for front, bank in zip(front_stock, banks, strict=True)
            ],
        },
        warmup=args.warmup,
        iterations=args.iterations,
        trials=args.trials,
        divisor=args.banks,
        scrub=scrub,
    )
    full_timings = _measure(
        {"control": full_control, "candidate": full_candidate_operation},
        warmup=args.warmup,
        iterations=args.iterations,
        trials=args.trials,
        divisor=args.banks,
        scrub=scrub,
    )

    result = {
        "schema": "k3-width4-expert-m3-screen/v1",
        "pid": os.getpid(),
        "device": mx.device_info(),
        "geometry": {
            "width": K3_WIDTH4,
            "top_k": K3_TOP_K,
            "experts": K3_EXPERTS,
            "hidden": K3_HIDDEN,
            "intermediate": K3_INTERMEDIATE,
            "banks": args.banks,
            "front_results_per_simdgroup": args.front_results,
            "down_results_per_simdgroup": args.down_results,
            "down_simdgroups_per_threadgroup": args.down_simds,
            "fused_down_route_reduce": args.fused_reduce,
            "native_down_after_custom_front": args.native_down,
        },
        "method": {
            "production_affine2_core": True,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "trials": args.trials,
            "alternating_order": True,
            "distinct_weight_banks": True,
            "cache_scrub_mib_before_each_arm": args.cache_scrub_mib,
            "dependent_full_chain": True,
        },
        "exact": {
            "front": front_exact,
            "down": down_exact,
            "full_chain": full_exact,
            "output_sha256": _digest(full_candidate),
        },
        "front_per_layer": _summary(
            front_timings["control"], front_timings["candidate"]
        ),
        "down_per_layer": _summary(
            down_timings["control"], down_timings["candidate"]
        ),
        "full_chain_per_layer": _summary(
            full_timings["control"], full_timings["candidate"]
        ),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.json_output is not None:
        args.json_output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
