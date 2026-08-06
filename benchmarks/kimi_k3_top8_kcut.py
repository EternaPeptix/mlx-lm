#!/usr/bin/env python3
"""Local M3 Max screen for the lossy released-K3 top-16 to top-8 K-cut.

The screen uses Kimi K3's exact released TP2 expert dimensions, 896 expert
banks, affine 2-bit/group-128 projections, BF16 activations and route weights,
and the same route-dynamic kernels used by the projected-KV deployment.  The
packed values are deterministic synthetic tensors because the full UVMAX
checkpoint is not present on this host; the tensor geometry and bytes touched
per selected route match the checkpoint contract.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import mlx.core as mx

from mlx_lm.models.kimi_k3 import SiTU, _group_expert_select
from mlx_lm.models.kimi_k3_fused_expert import (
    FUSED_DOWN_REDUCE_ENV,
    FUSED_EXPERT_ENV,
    FUSED_EXPERT_WIDTH3_ENV,
    fused_k3_down_reduce_enabled,
    fused_k3_expert_width3_enabled,
    fused_k3_experts_enabled,
    maybe_fused_k3_switch_glu,
    maybe_fused_k3_switch_glu_reduce,
)
from mlx_lm.models.kimi_k3_fused_router import (
    FUSED_ROUTER_ENV,
    fused_k3_router_enabled,
    maybe_fused_k3_router,
)
from mlx_lm.models.kimi_k3_prefill_route_combine import (
    PREFILL_ROUTE_COMBINE_ENV,
    maybe_fused_k3_prefill_switch_glu_reduce,
    prefill_route_combine_enabled,
)

EXPERTS = 896
HIDDEN = 3584
INTERMEDIATE = 1536
GROUP_SIZE = 128


class Projection:
    bits = 2
    group_size = GROUP_SIZE
    mode = "affine"
    _runtime_quantization_mode = "affine2"

    def __init__(
        self,
        *,
        input_dims: int,
        output_dims: int,
        packed_value: int,
        scale_salt: int,
    ):
        self.num_experts = EXPERTS
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.weight = mx.full(
            (EXPERTS, output_dims, input_dims // 16),
            packed_value,
            dtype=mx.uint32,
        )
        expert = mx.arange(EXPERTS, dtype=mx.uint32).reshape(EXPERTS, 1, 1)
        scale_code = (expert * 5 + scale_salt) % 13
        base = mx.array(1 / 128, dtype=mx.float32)
        step = mx.array(1 / 4096, dtype=mx.float32)
        self.scales = mx.broadcast_to(
            (base + scale_code.astype(mx.float32) * step).astype(mx.bfloat16),
            (EXPERTS, output_dims, input_dims // GROUP_SIZE),
        )
        self.biases = (-2 * self.scales).astype(mx.bfloat16)

    def __contains__(self, name: str) -> bool:
        return False

    def __getitem__(self, name: str):
        return getattr(self, name)

    def get(self, name: str, default=None):
        return getattr(self, name, default)

    def __call__(
        self,
        x: mx.array,
        indices: mx.array,
        sorted_indices: bool = False,
    ) -> mx.array:
        del sorted_indices
        return mx.gather_qmm(
            x,
            self.weight,
            self.scales,
            self.biases,
            rhs_indices=indices,
            transpose=True,
            group_size=GROUP_SIZE,
            bits=2,
            mode="affine",
        )


class Switch:
    training = False

    def __init__(self):
        self.activation = SiTU(beta=4.0, linear_beta=25.0)
        self.gate_proj = Projection(
            input_dims=HIDDEN,
            output_dims=INTERMEDIATE,
            packed_value=0xE4E4E4E4,
            scale_salt=1,
        )
        self.up_proj = Projection(
            input_dims=HIDDEN,
            output_dims=INTERMEDIATE,
            packed_value=0x39393939,
            scale_salt=3,
        )
        self.down_proj = Projection(
            input_dims=INTERMEDIATE,
            output_dims=HIDDEN,
            packed_value=0x93939393,
            scale_salt=7,
        )


def configure_fast_paths() -> None:
    os.environ[FUSED_ROUTER_ENV] = "1"
    os.environ[FUSED_EXPERT_ENV] = "1"
    os.environ[FUSED_EXPERT_WIDTH3_ENV] = "1"
    os.environ[FUSED_DOWN_REDUCE_ENV] = "0"
    os.environ[PREFILL_ROUTE_COMBINE_ENV] = "1"
    fused_k3_router_enabled.cache_clear()
    fused_k3_experts_enabled.cache_clear()
    fused_k3_expert_width3_enabled.cache_clear()
    fused_k3_down_reduce_enabled.cache_clear()
    prefill_route_combine_enabled.cache_clear()


def route(scores: mx.array, bias: mx.array, top_k: int):
    selected = maybe_fused_k3_router(
        scores,
        bias,
        top_k=top_k,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        renormalize=True,
        training=False,
    )
    if selected is not None:
        return selected
    return _group_expert_select(scores, bias, top_k, 1, 1, 1.0, True)


def candidate_chain(
    switch: Switch,
    x: mx.array,
    scores: mx.array,
    bias: mx.array,
    top_k: int,
) -> mx.array:
    indices, weights = route(scores, bias, top_k)
    return expert_chain_from_routes(switch, x, indices, weights)


def expert_chain_from_routes(
    switch: Switch,
    x: mx.array,
    indices: mx.array,
    weights: mx.array,
) -> mx.array:
    reduced = maybe_fused_k3_switch_glu_reduce(
        switch,
        x,
        indices,
        weights,
    )
    if reduced is None:
        reduced = maybe_fused_k3_prefill_switch_glu_reduce(
            switch,
            x,
            indices,
            weights,
        )
    if reduced is not None:
        return reduced
    routed = maybe_fused_k3_switch_glu(switch, x, indices)
    if routed is None:
        expanded = mx.expand_dims(x, (-2, -3))
        up = switch.up_proj(expanded, indices)
        gate = switch.gate_proj(expanded, indices)
        routed = switch.down_proj(
            switch.activation(up, gate),
            indices,
        ).squeeze(-2)
    return (routed * weights[..., None]).sum(axis=-2)


def stock_chain(
    switch: Switch,
    x: mx.array,
    scores: mx.array,
    bias: mx.array,
    top_k: int,
) -> mx.array:
    indices, weights = _group_expert_select(
        scores,
        bias,
        top_k,
        1,
        1,
        1.0,
        True,
    )
    expanded = mx.expand_dims(x, (-2, -3))
    up = switch.up_proj(expanded, indices)
    gate = switch.gate_proj(expanded, indices)
    routed = switch.down_proj(
        switch.activation(up, gate),
        indices,
    ).squeeze(-2)
    return (routed * weights[..., None]).sum(axis=-2)


def timed(output_factory) -> float:
    mx.synchronize()
    started = time.perf_counter_ns()
    mx.eval(output_factory())
    mx.synchronize()
    return (time.perf_counter_ns() - started) / 1e6


def benchmark_case(
    switch: Switch,
    width: int,
    *,
    warmup: int,
    trials: int,
) -> dict:
    mx.random.seed(20260806 + width)
    x = mx.random.normal((1, width, HIDDEN), dtype=mx.bfloat16)
    scores = mx.random.normal((1, width, EXPERTS), dtype=mx.bfloat16)
    bias = 0.25 * mx.random.normal((EXPERTS,), dtype=mx.float32)
    mx.eval(x, scores, bias)

    for top_k in (16, 8):
        for _ in range(warmup):
            mx.eval(candidate_chain(switch, x, scores, bias, top_k))

    samples = {16: [], 8: []}
    for trial in range(trials):
        order = (16, 8) if trial % 2 == 0 else (8, 16)
        for top_k in order:
            samples[top_k].append(
                timed(
                    lambda top_k=top_k: candidate_chain(switch, x, scores, bias, top_k)
                )
            )

    correctness = None
    if width in (3, 512, 4096):
        expected = stock_chain(switch, x, scores, bias, 8)
        actual = candidate_chain(switch, x, scores, bias, 8)
        mx.eval(expected, actual)
        correctness = bool(mx.array_equal(expected, actual).item())

    native_median = statistics.median(samples[16])
    k8_median = statistics.median(samples[8])
    paired_speedups = [
        native / candidate
        for native, candidate in zip(samples[16], samples[8], strict=True)
    ]
    result = {
        "width": width,
        "warmup": warmup,
        "trials": trials,
        "stock_k8_exact": correctness,
        "native_top16": {
            "median_ms": native_median,
            "samples_ms": samples[16],
        },
        "experimental_top8": {
            "median_ms": k8_median,
            "samples_ms": samples[8],
        },
        "speedup_top16_over_top8": native_median / k8_median,
        "paired_speedups_top16_over_top8": paired_speedups,
        "median_paired_speedup_top16_over_top8": statistics.median(paired_speedups),
    }
    if width == 3:
        routes = {top_k: route(scores, bias, top_k) for top_k in (16, 8)}
        mx.eval(*(value for pair in routes.values() for value in pair))
        route_samples = {16: [], 8: []}
        expert_samples = {16: [], 8: []}
        for top_k in (16, 8):
            indices, weights = routes[top_k]
            for _ in range(warmup):
                mx.eval(expert_chain_from_routes(switch, x, indices, weights))
        for trial in range(trials):
            order = (16, 8) if trial % 2 == 0 else (8, 16)
            for top_k in order:
                indices, weights = routes[top_k]
                route_samples[top_k].append(
                    timed(lambda top_k=top_k: route(scores, bias, top_k))
                )
                expert_samples[top_k].append(
                    timed(
                        lambda indices=indices, weights=weights: (
                            expert_chain_from_routes(switch, x, indices, weights)
                        )
                    )
                )
        route_medians = {
            top_k: statistics.median(route_samples[top_k]) for top_k in (16, 8)
        }
        expert_medians = {
            top_k: statistics.median(expert_samples[top_k]) for top_k in (16, 8)
        }
        route_paired = [
            native / candidate
            for native, candidate in zip(
                route_samples[16], route_samples[8], strict=True
            )
        ]
        expert_paired = [
            native / candidate
            for native, candidate in zip(
                expert_samples[16], expert_samples[8], strict=True
            )
        ]
        result["q3_breakdown"] = {
            "router": {
                "native_top16_median_ms": route_medians[16],
                "experimental_top8_median_ms": route_medians[8],
                "speedup": route_medians[16] / route_medians[8],
                "median_paired_speedup": statistics.median(route_paired),
                "paired_speedups": route_paired,
                "native_samples_ms": route_samples[16],
                "experimental_samples_ms": route_samples[8],
            },
            "expert_after_route": {
                "native_top16_median_ms": expert_medians[16],
                "experimental_top8_median_ms": expert_medians[8],
                "speedup": expert_medians[16] / expert_medians[8],
                "median_paired_speedup": statistics.median(expert_paired),
                "paired_speedups": expert_paired,
                "native_samples_ms": expert_samples[16],
                "experimental_samples_ms": expert_samples[8],
            },
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--q3-trials", type=int, default=15)
    parser.add_argument("--prefill-trials", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_fast_paths()
    mx.random.seed(20260806)
    switch = Switch()
    mx.eval(
        switch.gate_proj.weight,
        switch.gate_proj.scales,
        switch.gate_proj.biases,
        switch.up_proj.weight,
        switch.up_proj.scales,
        switch.up_proj.biases,
        switch.down_proj.weight,
        switch.down_proj.scales,
        switch.down_proj.biases,
    )
    mx.reset_peak_memory()

    cases = [
        benchmark_case(switch, 3, warmup=3, trials=args.q3_trials),
        benchmark_case(switch, 512, warmup=2, trials=args.prefill_trials),
        benchmark_case(switch, 4096, warmup=1, trials=args.prefill_trials),
    ]
    result = {
        "schema": "kimi-k3-top8-kcut-local-screen/v1",
        "device": mx.device_info(),
        "geometry": {
            "experts": EXPERTS,
            "native_top_k": 16,
            "experimental_top_k": 8,
            "hidden": HIDDEN,
            "tp2_intermediate": INTERMEDIATE,
            "quantization": "affine2-group128",
            "activation_dtype": "bfloat16",
            "weights": "deterministic synthetic packed values",
        },
        "fast_paths": {
            "fused_router": True,
            "fused_route_dynamic_experts": True,
            "fused_down_reduce": False,
            "prefill_route_combine": True,
        },
        "cases": cases,
        "representative_gate": {
            "minimum_speedup": 1.2,
            "passes_all_shapes": all(
                case["median_paired_speedup_top16_over_top8"] >= 1.2 for case in cases
            ),
        },
        "peak_memory_bytes_during_cases": int(mx.get_peak_memory()),
    }
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.json_out.write_text(serialized)
    print(serialized, end="")


if __name__ == "__main__":
    main()
