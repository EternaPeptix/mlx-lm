"""Capacity and acceptance break-even model for Kimi K3 ReplaySSM + DSpark."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ReplaySSMCapacity:
    full_state_history_bytes: int
    raw_history_bytes: int
    conv_history_bytes: int
    old_total_bytes: int
    replay_total_bytes: int
    reduction_ratio: float


def capacity_model(
    *,
    layers: int = 69,
    global_heads: int = 48,
    head_dim: int = 128,
    tensor_parallel: int = 2,
    verify_width: int = 3,
    conv_kernel: int = 4,
    activation_bytes: int = 2,
    state_bytes: int = 4,
    gate_bytes: int = 4,
    beta_bytes: int = 2,
) -> ReplaySSMCapacity:
    if global_heads % tensor_parallel:
        raise ValueError("KDA heads must divide tensor parallel size")
    if verify_width < 2:
        raise ValueError("verification width must be at least two")
    local_heads = global_heads // tensor_parallel

    full_state = layers * verify_width * local_heads * head_dim * head_dim * state_bytes
    raw_per_token_head = (
        head_dim * activation_bytes  # v
        + head_dim * activation_bytes  # raw pre-normalization k
        + head_dim * gate_bytes  # exact multiplicative gk
        + beta_bytes
    )
    raw = layers * verify_width * local_heads * raw_per_token_head
    conv = (
        layers
        * verify_width
        * (conv_kernel - 1)
        * (3 * local_heads * head_dim)
        * activation_bytes
    )
    old_total = full_state + conv
    replay_total = raw + conv
    return ReplaySSMCapacity(
        full_state_history_bytes=full_state,
        raw_history_bytes=raw,
        conv_history_bytes=conv,
        old_total_bytes=old_total,
        replay_total_bytes=replay_total,
        reduction_ratio=old_total / replay_total,
    )


def acceptance_break_even(
    *,
    target_step_ms: float,
    draft_step_ms: float,
    replay_step_ms: float,
    desired_tokens_per_second: float = 17.0,
) -> float:
    total_seconds = (target_step_ms + draft_step_ms + replay_step_ms) / 1000.0
    return desired_tokens_per_second * total_seconds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-width", type=int, default=3)
    parser.add_argument("--target-step-ms", type=float)
    parser.add_argument("--draft-step-ms", type=float, default=0.0)
    parser.add_argument("--replay-step-ms", type=float, default=0.0)
    parser.add_argument("--expected-accepted", type=float)
    args = parser.parse_args()

    result = {"capacity": asdict(capacity_model(verify_width=args.verify_width))}
    if args.target_step_ms is not None:
        threshold = acceptance_break_even(
            target_step_ms=args.target_step_ms,
            draft_step_ms=args.draft_step_ms,
            replay_step_ms=args.replay_step_ms,
        )
        result["throughput_model"] = {
            "accepted_tokens_per_step_required_for_17_tps": threshold,
            "expected_tokens_per_second": (
                args.expected_accepted
                / (
                    (args.target_step_ms + args.draft_step_ms + args.replay_step_ms)
                    / 1000.0
                )
                if args.expected_accepted is not None
                else None
            ),
        }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
