# Kimi K3 EXO/MLX experiment

This branch is one part of a coordinated public experiment:

- [EXO](https://github.com/EternaPeptix/exo/tree/experiment/kimi-k3-exo-mlx-stack)
- [MLX-LM](https://github.com/EternaPeptix/mlx-lm/tree/experiment/kimi-k3-exo-mlx-stack)
- [MLX](https://github.com/EternaPeptix/mlx/tree/experiment/kimi-k3-exo-mlx-stack)

It contains the Kimi K3 support and TP2 changes used to run
`kernelpool/Kimi-K3-2bit-UVMAX` across two 512 GB M3 Ultra systems, including:

- external pipeline-partition support and bounded cache advancement;
- an opt-in exact-weight fused expert path;
- model-parallel output-head coverage used by EXO's vocabulary-parallel path;
- an opt-in segmented compiled decode schedule; and
- focused Metal and distributed tests for those paths.

## Feature flags

`MLX_LM_KIMI_K3_FUSED_EXPERTS=1` enables the exact-weight fused expert
prototype. `MLX_LM_KIMI_K3_COMPILED_DECODE=1` enables segmented compiled
decode. `MLX_LM_KIMI_K3_PACKED_MOE_FRONT=1` enables an exact, decode-only
packed QMV for four same-input MoE-front projections. All three default to off
and fail closed outside their supported Kimi K3 decode shapes.

When compiled decode is enabled,
`MLX_LM_KIMI_K3_COMPILED_DECODE_SEGMENTS` can select `all`, `none`, individual
segment indices, or inclusive ranges such as `0-11,24`. Production Kimi K3
uses segment IDs `0..24`. The selector is strict and immutable for the model
lifetime; invalid or out-of-range values fail during model construction. This
is diagnostic tooling for locating the output divergence, not a claim that a
particular compiled subset is safe.

## Current result

The exact-path reference produced a median `12.1544` decode tok/s in the
canonical short TP2 benchmark. Segmented compiled decode produced `12.6206`
tok/s, a `3.8%` improvement, with flat measured peak memory.

The compiled candidate did **not** preserve the deterministic completion
digest (`c84d…` reference versus `8f60…` candidate), so it is not a
production recommendation. It is retained here as reproducible experimental
work while segment-level bisection and speculative-cache work continue.

The packed MoE-front path is numerically exact in the focused Metal tests. A
real-weight single-layer measurement projected roughly `12.10` to `12.93`
tok/s if the isolated saving scales across all 92 MoE layers, at an additional
approximately `7.44 GB` per rank. That projection is not an end-to-end
throughput claim; a full TP2 completion-hash and memory A/B remains required.

The EXO repository contains the strict target-verification and divergence
diagnostic tools. A sampled width-2 run kept the same top-1 continuation but
diverged numerically in the first recurrent KDA layer, so the strict
verification gate remains failed.

The bisection selector passed 13 focused single-rank Metal tests (with one
expected TP2 skip) and a local two-rank exact logits/cache comparison for full
and mixed schedules. After integrating MoE-front packing, the combined branch
also passed all 14 focused packed/fused expert tests and all 13 active compiled
decode tests in the same Apple Metal runtime.
