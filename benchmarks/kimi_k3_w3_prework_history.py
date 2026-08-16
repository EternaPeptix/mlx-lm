"""Frozen local Metal screen for the K3 W3 prework/history candidate."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import statistics
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_lm.models.gated_delta import compute_g_safe
from mlx_lm.models.kimi_k3 import KimiK3ShortConv
from mlx_lm.models.kimi_k3_w3_prework import fused_k3_w3_prework_history

BASELINE_REQUEST_MS = 5684.585742
BASELINE_TOKENS = 128
FULL_W3_CALLS = 45
KDA_LAYERS = 69
HEADS = 48
HEAD_DIM = 128
PROJECTION_DIM = HEADS * HEAD_DIM
CHANNELS = 3 * PROJECTION_DIM
WIDTH = 3
CONV_KERNEL = 4
LOWER_BOUND = -5.0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(value: mx.array) -> str:
    mx.eval(value)
    if value.dtype == mx.bfloat16:
        host = np.asarray(value.view(mx.uint16))
    elif value.dtype == mx.float32:
        host = np.asarray(value.view(mx.uint32))
    else:
        host = np.asarray(value)
    return hashlib.sha256(host.tobytes(order="C")).hexdigest()


def _exact(actual: mx.array, expected: mx.array) -> bool:
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return False
    if actual.dtype == mx.bfloat16:
        actual = actual.view(mx.uint16)
        expected = expected.view(mx.uint16)
    elif actual.dtype == mx.float32:
        actual = actual.view(mx.uint32)
        expected = expected.view(mx.uint32)
    else:
        raise TypeError(f"unsupported exactness dtype {actual.dtype}")
    equal = mx.array_equal(actual, expected)
    mx.eval(equal)
    return bool(equal.item())


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values))


def _append_event(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
        handle.flush()


def _existing_attempt_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    attempt_ids = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                attempt_ids.add(str(json.loads(line)["attempt_id"]))
    return attempt_ids


def _make_inputs(seed: int):
    mx.random.seed(seed)
    projected_qkv = mx.random.normal((1, WIDTH, CHANNELS), dtype=mx.bfloat16)
    initial_state = mx.random.normal((1, CONV_KERNEL - 1, CHANNELS), dtype=mx.bfloat16)
    conv_weight = mx.random.normal((CHANNELS, CONV_KERNEL, 1), dtype=mx.bfloat16)
    a_logits = mx.random.normal((1, WIDTH, HEADS, HEAD_DIM), dtype=mx.bfloat16)
    A_log = mx.log(
        mx.random.uniform(low=1.0, high=16.0, shape=(HEADS,)).astype(mx.float32)
    )
    dt_bias = mx.random.normal((PROJECTION_DIM,)).astype(mx.float32)
    b_logits = mx.random.normal((1, WIDTH, HEADS), dtype=mx.bfloat16)
    values = (
        projected_qkv,
        initial_state,
        conv_weight,
        a_logits,
        A_log,
        dt_bias,
        b_logits,
    )
    mx.eval(*values)
    return values


def _make_arms(inputs):
    (
        projected_qkv,
        _initial_state,
        conv_weight,
        a_logits,
        A_log,
        dt_bias,
        b_logits,
    ) = inputs
    conv = KimiK3ShortConv(CHANNELS, CONV_KERNEL)
    conv.conv.weight = conv_weight
    conv.eval()
    scale = float(HEAD_DIM) ** -0.5
    eps = 1e-6 / HEAD_DIM

    def stock(state: mx.array) -> tuple[mx.array, ...]:
        qkv, new_state, history = conv(
            projected_qkv,
            state,
            None,
            None,
            return_state_history=True,
        )
        q = qkv[..., :PROJECTION_DIM].reshape(1, WIDTH, HEADS, HEAD_DIM)
        raw_k = qkv[..., PROJECTION_DIM : 2 * PROJECTION_DIM].reshape(
            1, WIDTH, HEADS, HEAD_DIM
        )
        v = qkv[..., 2 * PROJECTION_DIM :].reshape(1, WIDTH, HEADS, HEAD_DIM)
        q = (scale**2) * mx.fast.rms_norm(q, None, eps)
        k = scale * mx.fast.rms_norm(raw_k, None, eps)
        gk = compute_g_safe(
            A_log.reshape(HEADS, 1),
            a_logits,
            dt_bias.reshape(HEADS, HEAD_DIM),
            LOWER_BOUND,
        )
        beta = mx.sigmoid(b_logits)
        return q, k, raw_k, v, gk, new_state, history, beta

    def candidate(state: mx.array) -> tuple[mx.array, ...]:
        values = fused_k3_w3_prework_history(
            projected_qkv,
            state,
            conv_weight,
            a_logits,
            A_log,
            dt_bias,
            num_heads=HEADS,
            head_dim=HEAD_DIM,
            conv_kernel=CONV_KERNEL,
            lower_bound=LOWER_BOUND,
        )
        beta = mx.sigmoid(b_logits)
        return (*values, beta)

    return stock, candidate


def _build_chain(
    arm: Callable[[mx.array], tuple[mx.array, ...]],
    initial_state: mx.array,
    layers: int,
) -> tuple[mx.array, ...]:
    state = initial_state
    roots = []
    for _ in range(layers):
        outputs = arm(state)
        roots.extend(outputs)
        state = outputs[5]
    return tuple(roots)


def _measure_chain(
    arm: Callable[[mx.array], tuple[mx.array, ...]],
    initial_state: mx.array,
    layers: int,
) -> float:
    roots = _build_chain(arm, initial_state, layers)
    mx.synchronize()
    started = time.perf_counter_ns()
    mx.eval(*roots)
    elapsed_ns = time.perf_counter_ns() - started
    return elapsed_ns / 1_000_000.0


def _warmup(
    stock,
    candidate,
    initial_state: mx.array,
    layers: int,
    warmups: int,
) -> None:
    for index in range(warmups):
        arms = (stock, candidate) if index % 2 == 0 else (candidate, stock)
        for arm in arms:
            _measure_chain(arm, initial_state, layers)


def _paired_timing(
    stock,
    candidate,
    initial_state: mx.array,
    layers: int,
    trials: int,
) -> tuple[list[float], list[float], list[str]]:
    stock_ms = []
    candidate_ms = []
    orders = []
    for trial in range(trials):
        if trial % 2 == 0:
            order = (("stock", stock), ("candidate", candidate))
            orders.append("AB")
        else:
            order = (("candidate", candidate), ("stock", stock))
            orders.append("BA")
        measured = {}
        for label, arm in order:
            measured[label] = _measure_chain(arm, initial_state, layers)
        stock_ms.append(measured["stock"])
        candidate_ms.append(measured["candidate"])
    return stock_ms, candidate_ms, orders


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--source-parent", required=True)
    parser.add_argument("--candidate-commit", required=True)
    parser.add_argument("--seed", type=int, default=2026081603)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--trials", type=int, default=21)
    parser.add_argument("--layers", type=int, default=KDA_LAYERS)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not mx.metal.is_available():
        raise RuntimeError("the frozen benchmark requires Metal")
    if args.trials < 3 or args.trials % 2 == 0:
        raise ValueError("trials must be an odd integer of at least three")
    if args.warmups < 1 or args.layers != KDA_LAYERS:
        raise ValueError("use at least one warmup and exactly 69 KDA layers")
    if args.attempt_id in _existing_attempt_ids(args.ledger):
        raise ValueError(f"attempt id {args.attempt_id!r} is already in the ledger")

    root = Path(__file__).resolve().parents[1]
    source_paths = {
        "kernel": root / "mlx_lm/models/kimi_k3_w3_prework.py",
        "integration": root / "mlx_lm/models/kimi_k3.py",
        "benchmark": Path(__file__).resolve(),
        "tests": root / "tests/test_kimi_k3_w3_prework.py",
        "protocol": root / "benchmarks/kimi_k3_w3_prework_history.md",
    }
    provenance = {
        "source_parent": args.source_parent,
        "candidate_commit": args.candidate_commit,
        "sha256": {label: _sha256_file(path) for label, path in source_paths.items()},
        "python": platform.python_version(),
        "mlx": importlib.metadata.version("mlx"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "device": mx.device_info(),
    }
    started = {
        "schema": "kimi-k3-w3-prework-history-ledger-v1",
        "event": "attempt_started",
        "attempt_id": args.attempt_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": provenance,
        "parameters": {
            "seed": args.seed,
            "warmups": args.warmups,
            "trials": args.trials,
            "layers_per_chain": args.layers,
            "alternating_order": "AB/BA",
            "graph_construction_timed": False,
            "synchronization": "one mx.eval per 69-layer dependent chain",
        },
    }
    _append_event(args.ledger, started)

    inputs = _make_inputs(args.seed)
    stock, candidate = _make_arms(inputs)
    initial_state = inputs[1]

    stock_exact = _build_chain(stock, initial_state, args.layers)
    candidate_exact = _build_chain(candidate, initial_state, args.layers)
    mx.eval(*stock_exact, *candidate_exact)
    mismatches = [
        index
        for index, (expected, actual) in enumerate(
            zip(stock_exact, candidate_exact, strict=True)
        )
        if not _exact(actual, expected)
    ]
    if mismatches:
        failed = {
            "schema": "kimi-k3-w3-prework-history-ledger-v1",
            "event": "attempt_failed_exactness",
            "attempt_id": args.attempt_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "mismatch_root_indices": mismatches,
        }
        _append_event(args.ledger, failed)
        raise RuntimeError(f"chain exactness failed at roots {mismatches}")

    _warmup(
        stock,
        candidate,
        initial_state,
        args.layers,
        args.warmups,
    )
    stock_ms, candidate_ms, orders = _paired_timing(
        stock,
        candidate,
        initial_state,
        args.layers,
        args.trials,
    )
    paired_delta_ms = [
        stock_value - candidate_value
        for stock_value, candidate_value in zip(stock_ms, candidate_ms, strict=True)
    ]
    estimator_delta_ms = _median(paired_delta_ms)
    request_saving_ms = FULL_W3_CALLS * estimator_delta_ms
    projected_request_ms = BASELINE_REQUEST_MS - request_saving_ms
    projected_tps = (
        BASELINE_TOKENS / (projected_request_ms / 1000.0)
        if projected_request_ms > 0
        else math.inf
    )
    baseline_tps = BASELINE_TOKENS / (BASELINE_REQUEST_MS / 1000.0)
    relative_gain_percent = 100.0 * (projected_tps / baseline_tps - 1.0)
    clears_gate = estimator_delta_ms > 0.0 and relative_gain_percent > 0.5

    completed = {
        "schema": "kimi-k3-w3-prework-history-ledger-v1",
        "event": "attempt_completed",
        "attempt_id": args.attempt_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": provenance,
        "geometry": {
            "batch": 1,
            "width": WIDTH,
            "heads": HEADS,
            "head_dim": HEAD_DIM,
            "projection_dim": PROJECTION_DIM,
            "qkv_channels": CHANNELS,
            "conv_kernel": CONV_KERNEL,
            "kda_layers_per_full_call": args.layers,
            "full_w3_calls_per_request": FULL_W3_CALLS,
        },
        "sealed_input_sha256": {
            label: _sha256_array(value)
            for label, value in zip(
                (
                    "projected_qkv",
                    "initial_state",
                    "conv_weight",
                    "a_logits",
                    "A_log",
                    "dt_bias",
                    "b_logits",
                ),
                inputs,
                strict=True,
            )
        },
        "exactness": {
            "chain_roots_compared": len(stock_exact),
            "mismatch_count": 0,
            "root_order_per_layer": [
                "q",
                "k",
                "raw_k",
                "v",
                "gk",
                "next_conv_state",
                "conv_history",
                "stock_beta",
            ],
        },
        "timing": {
            "warmup_chains_per_arm": args.warmups,
            "paired_trials": args.trials,
            "orders": orders,
            "stock_ms_per_69_layer_chain": stock_ms,
            "candidate_ms_per_69_layer_chain": candidate_ms,
            "paired_delta_ms_stock_minus_candidate": paired_delta_ms,
            "stock_arm_median_ms": _median(stock_ms),
            "candidate_arm_median_ms": _median(candidate_ms),
            "estimator": "median paired delta",
            "estimator_delta_ms_per_full_w3_call": estimator_delta_ms,
            "candidate_wins": sum(value > 0.0 for value in paired_delta_ms),
        },
        "economics": {
            "baseline_request_ms_for_128_tokens": BASELINE_REQUEST_MS,
            "baseline_tps_recomputed": baseline_tps,
            "request_saving_ms_mechanical": request_saving_ms,
            "projected_request_ms": projected_request_ms,
            "projected_tps": projected_tps,
            "relative_gain_percent": relative_gain_percent,
            "predeclared_gate_percent": 0.5,
            "clears_offline_gate": clears_gate,
        },
        "decision": (
            "offline_gate_cleared_no_canary_authorized"
            if clears_gate
            else "reject_below_0_5_percent_no_canary"
        ),
        "evidence_boundary": (
            "Synthetic post-projection Metal seam only; stock projections, "
            "gated-delta, MoE, collectives, scheduling, and full target are not timed."
        ),
    }
    _append_event(args.ledger, completed)
    print(json.dumps(completed, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
