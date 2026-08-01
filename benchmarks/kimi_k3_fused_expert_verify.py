"""Screen exact fused K3 experts at speculative-verification widths.

The benchmark rotates through distinct rank-local expert banks and chains each
layer's routed output into the next layer.  It compares the actual sorted
SwitchGLU graph used for multi-token inference with the custom fused gate/up/
SiTU and fused down/route-reduction kernels.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

from mlx_lm.models.kimi_k3_fused_down_reduce import fused_down_reduce_decode
from mlx_lm.models.kimi_k3_fused_switch_glu import fused_switch_situ_decode
from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort

HIDDEN = 3584
INTERMEDIATE = 1536
TOP_K = 16
GROUP_SIZE = 128
BITS = 2


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
    experts: int,
    input_width: int,
    output_width: int,
    packed_value: int,
) -> Projection:
    weight = mx.full(
        (experts, output_width, input_width // 16),
        packed_value,
        dtype=mx.uint32,
    )
    scales = mx.full(
        (experts, output_width, input_width // GROUP_SIZE),
        0.015625,
        dtype=mx.bfloat16,
    )
    return Projection(weight, scales, mx.zeros_like(scales))


def _bank(index: int, experts: int) -> ExpertBank:
    seed = 0x10203040 + index * 0x01010101
    return ExpertBank(
        up=_projection(experts, HIDDEN, INTERMEDIATE, seed ^ 0x13579BDF),
        gate=_projection(experts, HIDDEN, INTERMEDIATE, seed ^ 0x2468ACE0),
        down=_projection(experts, INTERMEDIATE, HIDDEN, seed ^ 0x55AA55AA),
    )


def _qmm(x: mx.array, indices: mx.array, projection: Projection) -> mx.array:
    return mx.gather_qmm(
        x,
        *projection.parts(),
        rhs_indices=indices,
        transpose=True,
        group_size=GROUP_SIZE,
        bits=BITS,
        mode="affine",
        sorted_indices=True,
    )


def _stock_layer(
    hidden: mx.array,
    indices: mx.array,
    router_weights: mx.array,
    bank: ExpertBank,
) -> mx.array:
    expanded = mx.expand_dims(hidden, (-2, -3))
    routed_input, sorted_indices, inv_order = _gather_sort(expanded, indices)
    up = _qmm(routed_input, sorted_indices, bank.up).astype(mx.float32)
    gate = _qmm(routed_input, sorted_indices, bank.gate).astype(mx.float32)
    activated = (
        4.0
        * mx.tanh(gate / 4.0)
        * mx.sigmoid(gate)
        * (25.0 * mx.tanh(up / 25.0))
    ).astype(mx.bfloat16)
    expert_outputs = _qmm(activated, sorted_indices, bank.down)
    expert_outputs = _scatter_unsort(expert_outputs, inv_order, indices.shape)
    expert_outputs = expert_outputs.squeeze(-2)
    return (expert_outputs * router_weights[..., None]).sum(axis=-2)


def _fused_layer(
    hidden: mx.array,
    indices: mx.array,
    router_weights: mx.array,
    bank: ExpertBank,
) -> mx.array:
    activated = fused_switch_situ_decode(
        hidden,
        indices,
        bank.up.parts(),
        bank.gate.parts(),
        results_per_simdgroup=2,
        simdgroups=4,
    )
    return fused_down_reduce_decode(
        activated,
        indices,
        router_weights,
        bank.down.parts(),
        results_per_threadgroup=4,
        simdgroups_per_threadgroup=16,
    )


def _chain(
    layer: Callable[[mx.array, mx.array, mx.array, ExpertBank], mx.array],
    hidden: mx.array,
    indices: mx.array,
    router_weights: mx.array,
    banks: tuple[ExpertBank, ...],
) -> mx.array:
    for bank in banks:
        hidden = layer(hidden, indices, router_weights, bank)
    return hidden


def _measure_pair(
    operations: tuple[tuple[str, Callable[[], mx.array]], ...],
    *,
    warmup: int,
    trials: int,
) -> dict[str, list[float]]:
    for _, operation in operations:
        for _ in range(warmup):
            mx.eval(operation())
    mx.synchronize()
    samples = {name: [] for name, _ in operations}
    for trial in range(trials):
        offset = trial % len(operations)
        ordered = operations[offset:] + operations[:offset]
        for name, operation in ordered:
            started = time.perf_counter_ns()
            mx.eval(operation())
            mx.synchronize()
            samples[name].append((time.perf_counter_ns() - started) / 1e6)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--widths", type=int, nargs="+", default=(1, 2))
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--banks", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--trials", type=int, default=11)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    if not args.widths or any(width not in (1, 2) for width in args.widths):
        raise ValueError("widths must be 1 or 2")
    if args.experts < TOP_K:
        raise ValueError("experts must be at least top-k")
    if not 1 <= args.banks <= 5:
        raise ValueError("banks must be in [1, 5]")

    mx.random.seed(83)
    banks = tuple(_bank(index, args.experts) for index in range(args.banks))
    mx.eval(
        *(
            part
            for bank in banks
            for projection in (bank.up, bank.gate, bank.down)
            for part in projection.parts()
        )
    )

    records = []
    for width in args.widths:
        hidden = mx.random.normal((1, width, HIDDEN), dtype=mx.bfloat16)
        base = mx.arange(TOP_K, dtype=mx.uint32)
        indices = mx.stack(
            [(base + shift * 17) % args.experts for shift in range(width)]
        )[None]
        router_weights = mx.random.uniform(
            shape=(1, width, TOP_K),
            dtype=mx.bfloat16,
        )
        mx.eval(hidden, indices, router_weights)

        def stock() -> mx.array:
            return _chain(_stock_layer, hidden, indices, router_weights, banks)

        def fused() -> mx.array:
            return _chain(_fused_layer, hidden, indices, router_weights, banks)
        expected = stock()
        actual = fused()
        mx.eval(expected, actual)
        if not bool(mx.array_equal(expected, actual).item()):
            raise RuntimeError(f"width {width} fused output is not bit exact")

        operations = (("stock", stock), ("fused", fused))
        if width % 2 == 0:
            operations = tuple(reversed(operations))
        samples = _measure_pair(
            operations,
            warmup=args.warmup,
            trials=args.trials,
        )
        stock_ms = statistics.median(samples["stock"]) / len(banks)
        fused_ms = statistics.median(samples["fused"]) / len(banks)
        paired_savings = [
            (stock - fused) / len(banks)
            for stock, fused in zip(samples["stock"], samples["fused"], strict=True)
        ]
        paired_median_saving = statistics.median(paired_savings)
        paired_wins = sum(saving > 0 for saving in paired_savings)
        records.append(
            {
                "width": width,
                "stock_ms_per_layer": stock_ms,
                "fused_ms_per_layer": fused_ms,
                "speedup": stock_ms / fused_ms,
                "median_paired_saving_ms_per_layer": paired_median_saving,
                "paired_wins": paired_wins,
                "trials": args.trials,
                "exact": True,
                "stock_samples_ms": samples["stock"],
                "fused_samples_ms": samples["fused"],
            }
        )
        print(
            f"width={width} stock={stock_ms:.6f} ms/layer "
            f"fused={fused_ms:.6f} ms/layer "
            f"speedup={stock_ms / fused_ms:.4f}x "
            f"saving={stock_ms - fused_ms:.6f} ms/layer "
            f"paired_median={paired_median_saving:.6f} ms/layer "
            f"wins={paired_wins}/{args.trials} exact=yes"
        )

    if args.json_output is not None:
        args.json_output.write_text(
            json.dumps(
                {
                    "experts": args.experts,
                    "banks": args.banks,
                    "warmup": args.warmup,
                    "trials": args.trials,
                    "records": records,
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
