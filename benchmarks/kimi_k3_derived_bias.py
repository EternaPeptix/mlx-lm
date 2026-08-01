#!/usr/bin/env python3
"""Paired real-shape screen for K3 decode-time affine-bias derivation.

The incumbent and candidate use identical raw tensors.  The candidate merely
skips each BF16 bias load after the exact ``bias == -2 * scale`` contract has
been established.  The benchmark covers all accepted TP2 decode consumers:
fused routed gate/up, tuned down, fused down/reduce, and their complete chain.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

from mlx_lm.models.kimi_k3_derived_bias import derived_affine2_biases
from mlx_lm.models.kimi_k3_fused_down_reduce import fused_down_reduce_decode
from mlx_lm.models.kimi_k3_fused_switch_glu import fused_switch_situ_decode
from mlx_lm.models.kimi_k3_tuned_gather_qmv import tuned_gather_qmv

HIDDEN = 3584
INTERMEDIATE = 1536
TOP_K = 16
GROUP_SIZE = 128
MLX_BASE = "2cfb83040011c273377a25df8ed16def80c6646c"
MLX_LM_BASE = "bf378e33831e745715a88418a44ce20ab1075b9b"


@dataclass(frozen=True)
class Projection:
    weight: mx.array
    scales: mx.array
    biases: mx.array

    def parts(self) -> tuple[mx.array, mx.array, mx.array]:
        return self.weight, self.scales, self.biases


@dataclass(frozen=True)
class ExpertBank:
    up: Projection
    gate: Projection
    down: Projection


def _projection(
    *,
    experts: int,
    input_width: int,
    output_width: int,
    salt: int,
) -> Projection:
    expert = mx.arange(experts, dtype=mx.uint32).reshape(experts, 1, 1)
    output = mx.arange(output_width, dtype=mx.uint32).reshape(1, output_width, 1)
    packed_k = mx.arange(input_width // 16, dtype=mx.uint32).reshape(
        1, 1, input_width // 16
    )
    weight = mx.bitwise_xor(
        mx.bitwise_xor(
            mx.array(salt, dtype=mx.uint32),
            expert * mx.array(0x9E3779B9, dtype=mx.uint32),
        ),
        mx.bitwise_xor(
            output * mx.array(0x85EBCA6B, dtype=mx.uint32),
            packed_k * mx.array(0xC2B2AE35, dtype=mx.uint32),
        ),
    )
    # Exact BF16 scale values observed in the K3 UVMAX expert banks.
    codebook = mx.array(
        [
            -0.046875,
            -0.03125,
            -0.0234375,
            -0.015625,
            -0.01171875,
            -0.0078125,
            -0.005859375,
            -0.00390625,
            -0.001953125,
            0.001953125,
            0.00390625,
            0.005859375,
            0.0078125,
            0.01171875,
            0.015625,
            0.0234375,
            0.03125,
            0.046875,
        ],
        dtype=mx.bfloat16,
    )
    group_k = mx.arange(input_width // GROUP_SIZE, dtype=mx.uint32).reshape(
        1, 1, input_width // GROUP_SIZE
    )
    codes = (
        expert * mx.array(17, dtype=mx.uint32)
        + output * mx.array(5, dtype=mx.uint32)
        + group_k * mx.array(3, dtype=mx.uint32)
        + mx.array(salt & 0xFF, dtype=mx.uint32)
    ) % codebook.size
    scales = mx.contiguous(codebook[codes])
    biases = mx.contiguous(derived_affine2_biases(scales))
    return Projection(mx.contiguous(weight), scales, biases)


def _bank(index: int) -> ExpertBank:
    salt = 0x10203040 + index * 0x01010101
    return ExpertBank(
        up=_projection(
            experts=TOP_K,
            input_width=HIDDEN,
            output_width=INTERMEDIATE,
            salt=salt ^ 0x13579BDF,
        ),
        gate=_projection(
            experts=TOP_K,
            input_width=HIDDEN,
            output_width=INTERMEDIATE,
            salt=salt ^ 0x2468ACE0,
        ),
        down=_projection(
            experts=TOP_K,
            input_width=INTERMEDIATE,
            output_width=HIDDEN,
            salt=salt ^ 0x55AA55AA,
        ),
    )


def _front(
    hidden: mx.array,
    indices: mx.array,
    bank: ExpertBank,
    *,
    derive_bias: bool,
) -> mx.array:
    return fused_switch_situ_decode(
        hidden,
        indices,
        bank.up.parts(),
        bank.gate.parts(),
        results_per_simdgroup=2,
        simdgroups=4,
        derive_bias=derive_bias,
    )


def _tuned_down(
    activated: mx.array,
    indices: mx.array,
    bank: ExpertBank,
    *,
    derive_bias: bool,
) -> mx.array:
    return tuned_gather_qmv(
        activated,
        indices,
        bank.down.parts(),
        results_per_simdgroup=4,
        simdgroups=2,
        broadcast_x=False,
        derive_bias=derive_bias,
    )


def _fused_down(
    activated: mx.array,
    indices: mx.array,
    router_weights: mx.array,
    bank: ExpertBank,
    *,
    derive_bias: bool,
) -> mx.array:
    return fused_down_reduce_decode(
        activated,
        indices,
        router_weights,
        bank.down.parts(),
        results_per_threadgroup=4,
        simdgroups_per_threadgroup=16,
        derive_bias=derive_bias,
    )


def _full_chain(
    hidden: mx.array,
    indices: mx.array,
    router_weights: mx.array,
    banks: Sequence[ExpertBank],
    *,
    derive_bias: bool,
) -> mx.array:
    for bank in banks:
        activated = _front(hidden, indices, bank, derive_bias=derive_bias)
        hidden = _fused_down(
            activated,
            indices,
            router_weights,
            bank,
            derive_bias=derive_bias,
        )
    return hidden


def _eval_result(value: mx.array | Sequence[mx.array]) -> None:
    if isinstance(value, mx.array):
        mx.eval(value)
    else:
        mx.eval(*value)


def _bit_exact(left: mx.array, right: mx.array) -> bool:
    return bool(mx.array_equal(left.view(mx.uint8), right.view(mx.uint8)).item())


def _measure_pair(
    incumbent: Callable[[], mx.array | Sequence[mx.array]],
    candidate: Callable[[], mx.array | Sequence[mx.array]],
    *,
    warmup: int,
    iterations: int,
    trials: int,
    divisor: int,
) -> dict[str, object]:
    operations = (("incumbent", incumbent), ("derived_bias", candidate))
    for _, operation in operations:
        for _ in range(warmup):
            _eval_result(operation())
    mx.synchronize()

    samples = {name: [] for name, _ in operations}
    paired_speedups = []
    paired_wins = 0
    for trial in range(trials):
        ordered = operations if trial % 2 == 0 else tuple(reversed(operations))
        round_times = {}
        for name, operation in ordered:
            started = time.perf_counter_ns()
            for _ in range(iterations):
                _eval_result(operation())
            mx.synchronize()
            elapsed = (time.perf_counter_ns() - started) / 1e6 / divisor / iterations
            samples[name].append(elapsed)
            round_times[name] = elapsed
        speedup = round_times["incumbent"] / round_times["derived_bias"]
        paired_speedups.append(speedup)
        paired_wins += speedup > 1.0

    incumbent_median = statistics.median(samples["incumbent"])
    candidate_median = statistics.median(samples["derived_bias"])
    return {
        "incumbent_median_ms": incumbent_median,
        "derived_bias_median_ms": candidate_median,
        "median_ratio": incumbent_median / candidate_median,
        "paired_geomean_speedup": math.exp(
            statistics.fmean(math.log(value) for value in paired_speedups)
        ),
        "paired_median_speedup": statistics.median(paired_speedups),
        "paired_wins": paired_wins,
        "iterations_per_trial": iterations,
        "trials": trials,
        "incumbent_samples_ms": samples["incumbent"],
        "derived_bias_samples_ms": samples["derived_bias"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--banks", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--trials", type=int, default=31)
    parser.add_argument("--minimum-full-speedup", type=float, default=1.0)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.banks <= 8:
        raise ValueError("banks must be in [1, 8]")
    if args.warmup < 1 or args.iterations < 1 or args.trials < 3:
        raise ValueError(
            "warmup and iterations must be positive and trials at least three"
        )

    mx.random.seed(20260801)
    banks = tuple(_bank(index) for index in range(args.banks))
    hidden = mx.random.normal((1, 1, HIDDEN), dtype=mx.bfloat16)
    activated = mx.random.normal((1, 1, TOP_K, 1, INTERMEDIATE), dtype=mx.bfloat16)
    indices = mx.arange(TOP_K, dtype=mx.uint32).reshape(1, 1, TOP_K)
    router_weights = mx.random.uniform(shape=(1, 1, TOP_K), dtype=mx.bfloat16)
    mx.eval(
        hidden,
        activated,
        indices,
        router_weights,
        *(
            part
            for bank in banks
            for projection in (bank.up, bank.gate, bank.down)
            for part in projection.parts()
        ),
    )

    exactness = {}
    for name, incumbent, candidate in (
        (
            "fused_front",
            lambda: _front(hidden, indices, banks[0], derive_bias=False),
            lambda: _front(hidden, indices, banks[0], derive_bias=True),
        ),
        (
            "tuned_down",
            lambda: _tuned_down(activated, indices, banks[0], derive_bias=False),
            lambda: _tuned_down(activated, indices, banks[0], derive_bias=True),
        ),
        (
            "fused_down_reduce",
            lambda: _fused_down(
                activated,
                indices,
                router_weights,
                banks[0],
                derive_bias=False,
            ),
            lambda: _fused_down(
                activated,
                indices,
                router_weights,
                banks[0],
                derive_bias=True,
            ),
        ),
        (
            "full_fused_chain",
            lambda: _full_chain(
                hidden,
                indices,
                router_weights,
                banks,
                derive_bias=False,
            ),
            lambda: _full_chain(
                hidden,
                indices,
                router_weights,
                banks,
                derive_bias=True,
            ),
        ),
    ):
        expected = incumbent()
        actual = candidate()
        mx.eval(expected, actual)
        exactness[name] = _bit_exact(expected, actual)
        if not exactness[name]:
            raise RuntimeError(f"{name} derived-bias output is not bit exact")

    pairs = {
        "fused_front": (
            lambda: [
                _front(hidden, indices, bank, derive_bias=False) for bank in banks
            ],
            lambda: [_front(hidden, indices, bank, derive_bias=True) for bank in banks],
        ),
        "tuned_down": (
            lambda: [
                _tuned_down(activated, indices, bank, derive_bias=False)
                for bank in banks
            ],
            lambda: [
                _tuned_down(activated, indices, bank, derive_bias=True)
                for bank in banks
            ],
        ),
        "fused_down_reduce": (
            lambda: [
                _fused_down(
                    activated,
                    indices,
                    router_weights,
                    bank,
                    derive_bias=False,
                )
                for bank in banks
            ],
            lambda: [
                _fused_down(
                    activated,
                    indices,
                    router_weights,
                    bank,
                    derive_bias=True,
                )
                for bank in banks
            ],
        ),
        "full_fused_chain": (
            lambda: _full_chain(
                hidden,
                indices,
                router_weights,
                banks,
                derive_bias=False,
            ),
            lambda: _full_chain(
                hidden,
                indices,
                router_weights,
                banks,
                derive_bias=True,
            ),
        ),
    }
    timings = {
        name: _measure_pair(
            incumbent,
            candidate,
            warmup=args.warmup,
            iterations=args.iterations,
            trials=args.trials,
            divisor=args.banks,
        )
        for name, (incumbent, candidate) in pairs.items()
    }
    result = {
        "bases": {"mlx": MLX_BASE, "mlx_lm": MLX_LM_BASE},
        "device": mx.device_info(),
        "geometry": {
            "hidden": HIDDEN,
            "intermediate": INTERMEDIATE,
            "top_k": TOP_K,
            "group_size": GROUP_SIZE,
            "bits": 2,
            "banks": args.banks,
        },
        "exactness": exactness,
        "timings": timings,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.json_output is not None:
        args.json_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    full_speedup = timings["full_fused_chain"]["paired_geomean_speedup"]
    if full_speedup <= args.minimum_full_speedup:
        raise RuntimeError(
            "derived-bias candidate rejected: full fused chain speedup "
            f"{full_speedup:.6f}x does not exceed "
            f"{args.minimum_full_speedup:.6f}x"
        )


if __name__ == "__main__":
    main()
