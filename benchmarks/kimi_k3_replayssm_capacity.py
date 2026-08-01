"""Capacity and acceptance break-even model for Kimi K3 ReplaySSM + DSpark."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Sequence


@dataclass(frozen=True)
class ReplaySSMCapacity:
    full_state_history_bytes: int
    raw_history_bytes: int
    conv_history_bytes: int
    old_total_bytes: int
    replay_total_bytes: int
    reduction_ratio: float


@dataclass(frozen=True)
class SpeculativeTier:
    """One pre-built DSpark verification tier in a cost-aware policy."""

    verify_width: int
    target_step_ms: float
    draft_step_ms: float
    replay_step_ms: float

    @property
    def gamma(self) -> int:
        return self.verify_width - 1

    @property
    def total_step_ms(self) -> float:
        return self.target_step_ms + self.draft_step_ms + self.replay_step_ms


@dataclass(frozen=True)
class TierEstimate:
    """Expected output rate for either ordinary decode or a DSpark tier."""

    name: str
    verify_width: int
    expected_accepted_draft_tokens: float
    expected_emitted_tokens: float
    total_step_ms: float
    expected_tokens_per_second: float


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


def emitted_tokens_break_even(
    *,
    target_step_ms: float,
    draft_step_ms: float,
    replay_step_ms: float,
    desired_tokens_per_second: float = 17.0,
) -> float:
    _validate_nonnegative_finite("target_step_ms", target_step_ms)
    _validate_nonnegative_finite("draft_step_ms", draft_step_ms)
    _validate_nonnegative_finite("replay_step_ms", replay_step_ms)
    _validate_nonnegative_finite("desired_tokens_per_second", desired_tokens_per_second)
    total_step_ms = target_step_ms + draft_step_ms + replay_step_ms
    if total_step_ms == 0.0:
        raise ValueError("total step time must be positive")
    total_seconds = total_step_ms / 1000.0
    return desired_tokens_per_second * total_seconds


def accepted_draft_tokens_break_even(
    *,
    target_step_ms: float,
    draft_step_ms: float,
    replay_step_ms: float,
    desired_tokens_per_second: float = 17.0,
) -> float:
    """Mean accepted proposals required in addition to the bonus token."""

    return max(
        0.0,
        emitted_tokens_break_even(
            target_step_ms=target_step_ms,
            draft_step_ms=draft_step_ms,
            replay_step_ms=replay_step_ms,
            desired_tokens_per_second=desired_tokens_per_second,
        )
        - 1.0,
    )


def acceptance_break_even(
    *,
    target_step_ms: float,
    draft_step_ms: float,
    replay_step_ms: float,
    desired_tokens_per_second: float = 17.0,
) -> float:
    """Compatibility alias returning required *emitted* tokens per round."""

    return emitted_tokens_break_even(
        target_step_ms=target_step_ms,
        draft_step_ms=draft_step_ms,
        replay_step_ms=replay_step_ms,
        desired_tokens_per_second=desired_tokens_per_second,
    )


def _validate_nonnegative_finite(name: str, value: float) -> None:
    if value < 0.0 or value == float("inf") or value != value:
        raise ValueError(f"{name} must be finite and non-negative")


def validate_tier(tier: SpeculativeTier) -> None:
    if tier.verify_width < 2:
        raise ValueError("speculative verify width must be at least two")
    _validate_nonnegative_finite("target_step_ms", tier.target_step_ms)
    _validate_nonnegative_finite("draft_step_ms", tier.draft_step_ms)
    _validate_nonnegative_finite("replay_step_ms", tier.replay_step_ms)
    if tier.total_step_ms == 0.0:
        raise ValueError("speculative tier total step time must be positive")


def validate_acceptance_survival(probabilities: Sequence[float]) -> None:
    """Validate P(accepted_prefix >= i) for i=1..gamma.

    Prefix acceptance is a survival curve, not an independent per-position
    probability.  It must therefore be bounded and monotonically decreasing.
    """

    previous = 1.0
    for probability in probabilities:
        if probability < 0.0 or probability > 1.0 or probability != probability:
            raise ValueError("acceptance survival values must be in [0, 1]")
        if probability > previous:
            raise ValueError("acceptance survival values must be non-increasing")
        previous = probability


def estimate_speculative_tier(
    tier: SpeculativeTier,
    acceptance_survival: Sequence[float],
) -> TierEstimate:
    """Estimate a tier from observed accepted-prefix survival probabilities."""

    validate_tier(tier)
    validate_acceptance_survival(acceptance_survival)
    if len(acceptance_survival) < tier.gamma:
        raise ValueError(
            "acceptance survival curve is shorter than the speculative gamma"
        )
    expected_accepted = sum(acceptance_survival[: tier.gamma])
    expected_emitted = 1.0 + expected_accepted
    return TierEstimate(
        name=f"width{tier.verify_width}",
        verify_width=tier.verify_width,
        expected_accepted_draft_tokens=expected_accepted,
        expected_emitted_tokens=expected_emitted,
        total_step_ms=tier.total_step_ms,
        expected_tokens_per_second=expected_emitted / (tier.total_step_ms / 1000.0),
    )


def select_cost_aware_tier(
    *,
    ordinary_step_ms: float,
    tiers: Sequence[SpeculativeTier],
    acceptance_survival: Sequence[float],
) -> tuple[TierEstimate, tuple[TierEstimate, ...]]:
    """Choose ordinary decode or the fastest pre-built DSpark tier.

    The selector consumes measured accepted-prefix survival rather than an
    independent-token approximation.  This makes it valid for correlated
    proposal failures and lets the same policy generalize across context size.
    """

    _validate_nonnegative_finite("ordinary_step_ms", ordinary_step_ms)
    if ordinary_step_ms == 0.0:
        raise ValueError("ordinary step time must be positive")
    if not tiers:
        raise ValueError("at least one speculative tier is required")
    validate_acceptance_survival(acceptance_survival)
    estimates = [
        TierEstimate(
            name="ordinary",
            verify_width=1,
            expected_accepted_draft_tokens=0.0,
            expected_emitted_tokens=1.0,
            total_step_ms=ordinary_step_ms,
            expected_tokens_per_second=1000.0 / ordinary_step_ms,
        )
    ]
    estimates.extend(
        estimate_speculative_tier(tier, acceptance_survival) for tier in tiers
    )
    return max(
        estimates, key=lambda estimate: estimate.expected_tokens_per_second
    ), tuple(estimates)


def _parse_acceptance_survival(raw: str) -> tuple[float, ...]:
    values = tuple(float(item) for item in raw.split(",") if item)
    if not values:
        raise argparse.ArgumentTypeError(
            "acceptance survival must contain comma-separated probabilities"
        )
    try:
        validate_acceptance_survival(values)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return values


def _parse_tier(raw: str) -> SpeculativeTier:
    try:
        width_raw, target_raw, draft_raw, replay_raw = raw.split(":")
        tier = SpeculativeTier(
            verify_width=int(width_raw),
            target_step_ms=float(target_raw),
            draft_step_ms=float(draft_raw),
            replay_step_ms=float(replay_raw),
        )
        validate_tier(tier)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "tier must be WIDTH:TARGET_MS:DRAFT_MS:REPLAY_MS"
        ) from error
    return tier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-width", type=int, default=3)
    parser.add_argument("--target-step-ms", type=float)
    parser.add_argument("--draft-step-ms", type=float, default=0.0)
    parser.add_argument("--replay-step-ms", type=float, default=0.0)
    parser.add_argument(
        "--expected-emitted",
        type=float,
        help="mean output tokens emitted by the complete speculative round",
    )
    parser.add_argument(
        "--expected-accepted",
        type=float,
        help="legacy alias for --expected-emitted",
    )
    parser.add_argument("--ordinary-step-ms", type=float)
    parser.add_argument(
        "--acceptance-survival",
        type=_parse_acceptance_survival,
        help="comma-separated P(accepted prefix >= i), i=1..gamma",
    )
    parser.add_argument(
        "--tier",
        action="append",
        type=_parse_tier,
        default=[],
        help="repeat WIDTH:TARGET_MS:DRAFT_MS:REPLAY_MS candidate tiers",
    )
    args = parser.parse_args()

    if args.expected_emitted is not None and args.expected_accepted is not None:
        parser.error("use only one of --expected-emitted or --expected-accepted")
    expected_emitted = (
        args.expected_emitted
        if args.expected_emitted is not None
        else args.expected_accepted
    )
    if expected_emitted is not None:
        try:
            _validate_nonnegative_finite("expected emitted tokens", expected_emitted)
        except ValueError as error:
            parser.error(str(error))

    result = {"capacity": asdict(capacity_model(verify_width=args.verify_width))}
    if args.target_step_ms is not None:
        emitted_threshold = emitted_tokens_break_even(
            target_step_ms=args.target_step_ms,
            draft_step_ms=args.draft_step_ms,
            replay_step_ms=args.replay_step_ms,
        )
        result["throughput_model"] = {
            "emitted_tokens_per_step_required_for_17_tps": emitted_threshold,
            "accepted_draft_tokens_per_step_required_for_17_tps": max(
                0.0, emitted_threshold - 1.0
            ),
            "expected_tokens_per_second": (
                expected_emitted
                / (
                    (args.target_step_ms + args.draft_step_ms + args.replay_step_ms)
                    / 1000.0
                )
                if expected_emitted is not None
                else None
            ),
        }
    adaptive_args = (
        args.ordinary_step_ms,
        args.acceptance_survival,
        args.tier,
    )
    if any(value is not None and value != [] for value in adaptive_args):
        if (
            args.ordinary_step_ms is None
            or args.acceptance_survival is None
            or not args.tier
        ):
            parser.error(
                "--ordinary-step-ms, --acceptance-survival, and at least one "
                "--tier are required together"
            )
        selected, estimates = select_cost_aware_tier(
            ordinary_step_ms=args.ordinary_step_ms,
            tiers=args.tier,
            acceptance_survival=args.acceptance_survival,
        )
        result["adaptive_tier_model"] = {
            "selected": selected.name,
            "estimates": [asdict(estimate) for estimate in estimates],
        }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
