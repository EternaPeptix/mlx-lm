#!/usr/bin/env python3
"""Paired real-geometry screen for width-three K3 fused experts.

This mirrors the deployed non-reduced expert path: fused gate/up/SiTU, tuned
down projection, and the ordinary router-weight reduction.  The stock arm uses
the three native gather-QMM projections and the same reduction.  Multiple
distinct banks limit same-layer cache reuse while keeping the screen small.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from functools import partial

import mlx.core as mx

from mlx_lm.models.kimi_k3_derived_bias import (
    DERIVE_AFFINE2_BIAS_ENV,
    derive_affine2_bias_enabled,
)
from mlx_lm.models.kimi_k3_fused_expert import (
    FUSED_EXPERT_ENV,
    FUSED_EXPERT_WIDTH3_ENV,
    fused_k3_expert_width3_enabled,
    fused_k3_experts_enabled,
)
from mlx_lm.models.kimi_k3_fused_switch_glu import fused_switch_situ_decode
from mlx_lm.models.kimi_k3_tuned_gather_qmv import tuned_gather_qmv
from tests.test_kimi_k3_fused_expert import (
    _Switch,
    _install_derived_biases,
)

WIDTH = 3
TOP_K = 16
HIDDEN = 3584


def _measure(
    operations,
    *,
    banks: int,
    warmup: int,
    iterations: int,
    trials: int,
):
    for operation in operations.values():
        for _ in range(warmup):
            mx.eval(operation())
    mx.synchronize()

    samples = {name: [] for name in operations}
    for trial in range(trials):
        names = tuple(operations)
        if trial % 2:
            names = tuple(reversed(names))
        for name in names:
            started = time.perf_counter_ns()
            for _ in range(iterations):
                mx.eval(operations[name]())
            mx.synchronize()
            elapsed_ms = (time.perf_counter_ns() - started) / 1e6
            samples[name].append(elapsed_ms / iterations / banks)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--banks", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--results-per-simdgroup", type=int, default=2)
    parser.add_argument("--simdgroups", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.banks <= 16:
        raise ValueError("banks must be in [1, 16]")

    os.environ[FUSED_EXPERT_ENV] = "1"
    os.environ[FUSED_EXPERT_WIDTH3_ENV] = "1"
    os.environ[DERIVE_AFFINE2_BIAS_ENV] = "1"
    fused_k3_experts_enabled.cache_clear()
    fused_k3_expert_width3_enabled.cache_clear()
    derive_affine2_bias_enabled.cache_clear()

    mx.random.seed(20260802)
    switches = tuple(
        _Switch.bounded_tp2_geometry_nonuniform() for _ in range(args.banks)
    )
    for switch in switches:
        _install_derived_biases(switch, include_down=True)
    hidden = mx.random.normal((1, WIDTH, HIDDEN), dtype=mx.bfloat16)
    base = mx.arange(TOP_K, dtype=mx.uint32)
    indices = mx.stack(
        [(base * 5 + shift * 3) % TOP_K for shift in range(WIDTH)]
    )[None]
    router_weights = mx.random.uniform(
        shape=(1, WIDTH, TOP_K), dtype=mx.bfloat16
    )

    @partial(mx.compile, shapeless=False)
    def candidate_experts(
        value,
        candidate_indices,
        up_weight,
        up_scales,
        up_biases,
        gate_weight,
        gate_scales,
        gate_biases,
        down_weight,
        down_scales,
        down_biases,
    ):
        activated = fused_switch_situ_decode(
            value,
            candidate_indices,
            (up_weight, up_scales, up_biases),
            (gate_weight, gate_scales, gate_biases),
            results_per_simdgroup=args.results_per_simdgroup,
            simdgroups=args.simdgroups,
            derive_bias=True,
        )
        return tuned_gather_qmv(
            activated,
            candidate_indices,
            (down_weight, down_scales, down_biases),
            results_per_simdgroup=4,
            simdgroups=2,
            broadcast_x=False,
            derive_bias=True,
        ).squeeze(-2)

    def stock():
        value = hidden
        for switch in switches:
            value = (
                switch.stock(value, indices) * router_weights[..., None]
            ).sum(axis=-2)
        return value

    def fused():
        value = hidden
        for switch in switches:
            expert_outputs = candidate_experts(
                value,
                indices,
                switch.up_proj.weight,
                switch.up_proj.scales,
                switch.up_proj.biases,
                switch.gate_proj.weight,
                switch.gate_proj.scales,
                switch.gate_proj.biases,
                switch.down_proj.weight,
                switch.down_proj.scales,
                switch.down_proj.biases,
            )
            value = (expert_outputs * router_weights[..., None]).sum(axis=-2)
        return value

    expected = stock()
    actual = fused()
    mx.eval(expected, actual)
    exact = bool(mx.array_equal(expected.view(mx.uint8), actual.view(mx.uint8)).item())
    if not exact:
        raise RuntimeError("width-three fused chain is not bit exact")

    samples = _measure(
        {"stock": stock, "fused": fused},
        banks=args.banks,
        warmup=args.warmup,
        iterations=args.iterations,
        trials=args.trials,
    )
    stock_ms = statistics.median(samples["stock"])
    fused_ms = statistics.median(samples["fused"])
    paired_savings = [
        stock_sample - fused_sample
        for stock_sample, fused_sample in zip(
            samples["stock"], samples["fused"], strict=True
        )
    ]
    result = {
        "width": WIDTH,
        "results_per_simdgroup": args.results_per_simdgroup,
        "simdgroups": args.simdgroups,
        "banks": args.banks,
        "exact": exact,
        "stock_ms_per_layer": stock_ms,
        "fused_ms_per_layer": fused_ms,
        "speedup": stock_ms / fused_ms,
        "median_paired_saving_ms_per_layer": statistics.median(paired_savings),
        "paired_wins": sum(value > 0 for value in paired_savings),
        "trials": args.trials,
        "stock_samples_ms_per_layer": samples["stock"],
        "fused_samples_ms_per_layer": samples["fused"],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
