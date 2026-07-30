"""Shape-realistic benchmark for K3 TP2 routed-up/shared-add fusion."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import mlx.core as mx

from mlx_lm.models.kimi_k3_fused_routed_up_add import (
    FUSED_ROUTED_UP_ADD_ENV,
    K3_GROUP_SIZE,
    K3_HIDDEN_SIZE,
    K3_ROUTED_LATENT_SIZE,
    fused_routed_up_add_enabled,
    maybe_fused_k3_routed_up_add,
)


class Projection:
    bits = 8
    group_size = K3_GROUP_SIZE
    mode = "affine"

    def __init__(self):
        self.weight = mx.full(
            (K3_HIDDEN_SIZE, K3_ROUTED_LATENT_SIZE // 4),
            0xD3917A5C,
            dtype=mx.uint32,
        )
        scale_shape = (
            K3_HIDDEN_SIZE,
            K3_ROUTED_LATENT_SIZE // K3_GROUP_SIZE,
        )
        self.scales = mx.random.uniform(
            low=0.001,
            high=0.02,
            shape=scale_shape,
        ).astype(mx.bfloat16)
        self.biases = mx.random.uniform(
            low=-0.05,
            high=0.05,
            shape=scale_shape,
        ).astype(mx.bfloat16)

    def get(self, name):
        return getattr(self, name, None)

    def __call__(self, x):
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


class SparseMoE:
    training = False
    latent_size = K3_ROUTED_LATENT_SIZE

    def __init__(self):
        self.routed_expert_up_proj = Projection()

    def stock(self, routed, shared, residual):
        return residual + (self.routed_expert_up_proj(routed) + shared)


def time_ms(call, iterations: int) -> float:
    mx.synchronize()
    started = time.perf_counter_ns()
    for _ in range(iterations):
        mx.eval(call())
    mx.synchronize()
    return (time.perf_counter_ns() - started) / iterations / 1e6


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--trials", type=int, default=15)
    args = parser.parse_args()

    mx.random.seed(20260730)
    mx.set_default_device(mx.gpu)
    moe = SparseMoE()
    routed = mx.random.normal(
        (1, 1, K3_ROUTED_LATENT_SIZE),
        dtype=mx.bfloat16,
    )
    shared = mx.random.normal(
        (1, 1, K3_HIDDEN_SIZE),
        dtype=mx.bfloat16,
    )
    residual = mx.random.normal(
        (1, 1, K3_HIDDEN_SIZE),
        dtype=mx.bfloat16,
    )
    mx.eval(
        routed,
        shared,
        residual,
        moe.routed_expert_up_proj.weight,
        moe.routed_expert_up_proj.scales,
        moe.routed_expert_up_proj.biases,
    )

    os.environ[FUSED_ROUTED_UP_ADD_ENV] = "1"
    fused_routed_up_add_enabled.cache_clear()

    def stock():
        return moe.stock(routed, shared, residual)

    def fused():
        result = maybe_fused_k3_routed_up_add(
            moe,
            routed,
            shared,
            residual,
        )
        assert result is not None
        return result

    expected = stock()
    actual = fused()
    mx.eval(expected, actual)
    if not bool(mx.array_equal(expected, actual).item()):
        differing = int(mx.sum(expected != actual).item())
        raise RuntimeError(f"candidate is not exact: {differing} values differ")

    for _ in range(args.warmup):
        mx.eval(stock(), fused())
    mx.synchronize()

    samples = {"stock_ms": [], "fused_ms": []}
    paired_speedups = []
    paired_deltas_ms = []
    for trial in range(args.trials):
        pair = {}
        order = (
            (("stock_ms", stock), ("fused_ms", fused))
            if trial % 2 == 0
            else (("fused_ms", fused), ("stock_ms", stock))
        )
        for name, call in order:
            pair[name] = time_ms(call, args.iterations)
            samples[name].append(pair[name])
        paired_speedups.append(pair["stock_ms"] / pair["fused_ms"])
        paired_deltas_ms.append(pair["stock_ms"] - pair["fused_ms"])

    medians = {
        name: statistics.median(values)
        for name, values in samples.items()
    }
    saved_per_layer = medians["stock_ms"] - medians["fused_ms"]
    result = {
        "geometry": {
            "routed_latent_size": K3_ROUTED_LATENT_SIZE,
            "hidden_size": K3_HIDDEN_SIZE,
            "sparse_layers": 92,
            "dtype": "bfloat16",
            "projection": "affine-8/group-64",
        },
        "protocol": vars(args),
        "exact": True,
        "samples_ms": samples,
        "medians_ms": medians,
        "median_paired_speedup": statistics.median(paired_speedups),
        "median_paired_delta_ms": statistics.median(paired_deltas_ms),
        "paired_wins": sum(delta > 0 for delta in paired_deltas_ms),
        "speedup_from_independent_medians": (
            medians["stock_ms"] / medians["fused_ms"]
        ),
        "saved_ms_per_layer": saved_per_layer,
        "projected_saved_ms_per_token": 92 * saved_per_layer,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
