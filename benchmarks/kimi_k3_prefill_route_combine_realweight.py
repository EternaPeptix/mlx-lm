#!/usr/bin/env python3
"""Benchmark the complete K3 SwitchGLU prefill boundary with real weights."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path

import mlx.core as mx

from mlx_lm.models.kimi_k3 import SiTU
from mlx_lm.models.kimi_k3_prefill_route_combine import (
    PREFILL_ROUTE_COMBINE_ENV,
    maybe_fused_k3_prefill_switch_glu_reduce,
    prefill_route_combine_enabled,
)


TOP_K = 16
EXPERTS = 896
INPUT_DIMS = 3584
INTERMEDIATE_DIMS = 1536


class _RealAffine2Projection:
    def __init__(self, weight, scales, input_dims: int, output_dims: int):
        self.weight = weight
        self.scales = scales
        self.bits = 2
        self.group_size = 128
        self.mode = "affine"
        self._runtime_quantization_mode = "affine2"
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.num_experts = EXPERTS

    def __contains__(self, name: str) -> bool:
        return False

    def __call__(self, x, indices, sorted_indices=False):
        return mx.gather_qmm(
            x,
            self.weight,
            self.scales,
            rhs_indices=indices,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self._runtime_quantization_mode,
            sorted_indices=sorted_indices,
        )


class _RealSwitchGLU:
    def __init__(self, gate, up, down):
        self.gate_proj = gate
        self.up_proj = up
        self.down_proj = down
        self.activation = SiTU(4.0, 25.0)
        self.training = False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--route-jsonl", type=Path, required=True)
    parser.add_argument("--record", type=int, default=6)
    parser.add_argument("--tokens", type=int, nargs="+", default=(512, 2048, 4096))
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if any(tokens < 512 or tokens > 4096 for tokens in args.tokens):
        parser.error("tokens must be in [512, 4096]")
    if min(args.warmup, args.trials, args.iterations) <= 0:
        parser.error("warmup, trials, and iterations must be positive")
    return args


def _record(path: Path, index: int) -> dict:
    with path.open(encoding="utf-8") as handle:
        for current, line in enumerate(handle):
            if current == index:
                return json.loads(line)
    raise ValueError(f"route record {index} not found")


def _scale_counts(counts: list[int], target: int) -> list[int]:
    exact = [count * target / sum(counts) for count in counts]
    scaled = [math.floor(value) for value in exact]
    order = sorted(
        range(len(counts)),
        key=lambda index: (exact[index] - scaled[index], counts[index]),
        reverse=True,
    )
    for index in order[: target - sum(scaled)]:
        scaled[index] += 1
    return scaled


def _load_tensor(checkpoint: Path, index: dict, name: str, cache: dict):
    shard = index["weight_map"][name]
    if shard not in cache:
        cache[shard] = mx.load(str(checkpoint / shard))
    return cache[shard][name]


def _switch(checkpoint: Path, layer: int):
    index = json.loads(
        (checkpoint / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    cache = {}
    projections = {}
    shapes = {
        "gate_proj": (INPUT_DIMS, INTERMEDIATE_DIMS),
        "up_proj": (INPUT_DIMS, INTERMEDIATE_DIMS),
        "down_proj": (INTERMEDIATE_DIMS, INPUT_DIMS),
    }
    names = []
    for projection, (input_dims, output_dims) in shapes.items():
        prefix = (
            f"language_model.model.layers.{layer}.mlp.switch_mlp.{projection}"
        )
        weight_name = prefix + ".weight"
        scales_name = prefix + ".scales"
        weight = _load_tensor(checkpoint, index, weight_name, cache)
        scales = _load_tensor(checkpoint, index, scales_name, cache)
        projections[projection] = _RealAffine2Projection(
            weight, scales, input_dims, output_dims
        )
        names.extend((weight_name, scales_name))
    mx.eval(
        *(projection.weight for projection in projections.values()),
        *(projection.scales for projection in projections.values()),
    )
    return _RealSwitchGLU(
        projections["gate_proj"],
        projections["up_proj"],
        projections["down_proj"],
    ), names


def _inputs(counts: list[int], tokens: int):
    routes = tokens * TOP_K
    scaled = _scale_counts(counts, routes)
    sorted_experts = mx.concatenate(
        [
            mx.full((count,), expert, dtype=mx.uint32)
            for expert, count in enumerate(scaled)
            if count
        ]
    )
    permutation = (mx.arange(routes, dtype=mx.uint32) * 17 + 13) % routes
    indices = sorted_experts[permutation].reshape(1, tokens, TOP_K)
    mx.random.seed(0x4B334000 + tokens)
    x = mx.random.uniform(
        low=-1.0, high=1.0, shape=(1, tokens, INPUT_DIMS)
    ).astype(mx.bfloat16)
    weights = mx.random.uniform(
        low=0.0, high=1.0, shape=(1, tokens, TOP_K)
    ).astype(mx.bfloat16)
    weights = (
        weights / weights.astype(mx.float32).sum(axis=-1, keepdims=True)
    ).astype(mx.bfloat16)
    mx.eval(x, indices, weights)
    return x, indices, weights, scaled


def _control(switch, x, indices, weights):
    flat_indices = indices.flatten()
    order = mx.argsort(flat_indices)
    inverse = mx.argsort(order)
    expanded = mx.expand_dims(x, (-2, -3))
    sorted_x = expanded.flatten(0, -3)[order // TOP_K]
    sorted_indices = flat_indices[order]
    up = switch.up_proj(sorted_x, sorted_indices, sorted_indices=True)
    gate = switch.gate_proj(sorted_x, sorted_indices, sorted_indices=True)
    activated = switch.activation(up, gate)
    sorted_routes = switch.down_proj(
        activated, sorted_indices, sorted_indices=True
    )
    routes = sorted_routes[inverse]
    routes = mx.unflatten(routes, 0, indices.shape).squeeze(-2)
    return (routes * weights[..., None]).sum(axis=-2)


def _elapsed(call, iterations: int) -> float:
    mx.synchronize()
    start = time.perf_counter_ns()
    for _ in range(iterations):
        mx.eval(call())
    mx.synchronize()
    return (time.perf_counter_ns() - start) / 1e6 / iterations


def _paired(control, candidate, trials: int, iterations: int) -> dict:
    control_samples = []
    candidate_samples = []
    ratios = []
    for trial in range(trials):
        if trial % 2:
            candidate_ms = _elapsed(candidate, iterations)
            before = _elapsed(control, iterations)
            after = _elapsed(control, iterations)
        else:
            before = _elapsed(control, iterations)
            candidate_ms = _elapsed(candidate, iterations)
            after = _elapsed(control, iterations)
        control_ms = (before + after) / 2
        control_samples.append(control_ms)
        candidate_samples.append(candidate_ms)
        ratios.append(control_ms / candidate_ms)
    control_median = statistics.median(control_samples)
    candidate_median = statistics.median(candidate_samples)
    return {
        "control_median_ms": control_median,
        "candidate_median_ms": candidate_median,
        "ratio_of_medians": control_median / candidate_median,
        "median_paired_speedup": statistics.median(ratios),
        "positive_pairs": sum(ratio > 1 for ratio in ratios),
        "trials": trials,
        "control_samples_ms": control_samples,
        "candidate_samples_ms": candidate_samples,
        "paired_speedups": ratios,
    }


def _case(switch, counts, tokens: int, warmup: int, trials: int, iterations: int):
    x, indices, weights, scaled = _inputs(counts, tokens)
    control = lambda: _control(switch, x, indices, weights)
    candidate = lambda: maybe_fused_k3_prefill_switch_glu_reduce(
        switch, x, indices, weights
    )
    expected = control()
    actual = candidate()
    if actual is None:
        raise RuntimeError("candidate failed its real K3 prefill guard")
    mx.eval(expected, actual)
    exact = bool(mx.array_equal(expected, actual).item())
    if not exact:
        raise RuntimeError(f"complete SwitchGLU mismatch at {tokens} tokens")
    for _ in range(warmup):
        mx.eval(control(), candidate())
    mx.synchronize()
    return {
        "tokens": tokens,
        "routes": tokens * TOP_K,
        "active_experts": sum(count > 0 for count in scaled),
        "max_expert_count": max(scaled),
        "exact": exact,
        "complete_switch_glu": _paired(control, candidate, trials, iterations),
    }


def main() -> int:
    args = _parse_args()
    os.environ[PREFILL_ROUTE_COMBINE_ENV] = "1"
    os.environ["MLX_METAL_K3_AFFINE2_EXPERT_TASKS"] = "1"
    prefill_route_combine_enabled.cache_clear()
    route = _record(args.route_jsonl, args.record)
    switch, weight_names = _switch(args.checkpoint, int(route["layer_index"]))
    cases = [
        _case(
            switch,
            [int(value) for value in route["expert_counts"]],
            tokens,
            args.warmup,
            args.trials,
            args.iterations,
        )
        for tokens in args.tokens
    ]
    gate_passed = all(
        case["exact"]
        and case["complete_switch_glu"]["ratio_of_medians"] >= 1.01
        and case["complete_switch_glu"]["positive_pairs"]
        >= (args.trials + 1) // 2
        for case in cases
    )
    report = {
        "schema": "k3-prefill-route-combine-realweight-switch-glu/v1",
        "device": mx.device_info(),
        "checkpoint": str(args.checkpoint),
        "weight_names": weight_names,
        "route_profile": {
            "path": str(args.route_jsonl),
            "record": args.record,
            "layer_index": route["layer_index"],
            "expert_counts_sha256": route["expert_counts_sha256"],
        },
        "protocol": {
            "real_weights": True,
            "complete_switch_glu_boundary": True,
            "warmup": args.warmup,
            "trials": args.trials,
            "iterations": args.iterations,
            "gate": "bit exact and >=1.01x complete SwitchGLU with majority wins",
        },
        "gate_passed": gate_passed,
        "cases": cases,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if gate_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
