# Kimi K3 EXO/MLX experiment

This branch is one part of a coordinated public experiment:

- [EXO](https://github.com/EternaPeptix/exo/tree/experiment/kimi-k3-distributed-optimizations)
- [MLX-LM](https://github.com/EternaPeptix/mlx-lm/tree/experiment/kimi-k3-distributed-optimizations)
- [MLX](https://github.com/EternaPeptix/mlx/tree/experiment/kimi-k3-distributed-optimizations)

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

`MLX_LM_KIMI_K3_ASYNC_DECODE_BOUNDARIES` enables eager, decode-only
asynchronous evaluation boundaries. It accepts `none`, `laguna8`, `block8`,
`all`, individual layer indices, or inclusive ranges. The companion
`MLX_LM_KIMI_K3_ASYNC_DECODE_STATE` selects `hidden` or `residual` state
submission. The path defaults to off and fails closed unless generation is
single-batch, single-token, eager Metal execution with a populated cache and a
full non-pipeline model. Compiled decode and asynchronous boundaries cannot be
enabled together.

When compiled decode is enabled,
`MLX_LM_KIMI_K3_COMPILED_DECODE_SEGMENTS` can select `all`, `none`, individual
segment indices, or inclusive ranges such as `0-11,24`. Production Kimi K3
uses segment IDs `0..24`. The selector is strict and immutable for the model
lifetime; invalid or out-of-range values fail during model construction. This
is diagnostic tooling for locating the output divergence, not a claim that a
particular compiled subset is safe.

## Current result

On a matched three-repetition canonical TP2 screening run, the feature-off
reference produced a median `12.0465` decode tok/s. The `laguna8` hidden-state
asynchronous schedule produced `12.9399` tok/s, a `7.4%` improvement, while
retaining the canonical completion digest and essentially unchanged prefill
throughput. Peak memory remained approximately `414 GB` per rank.

An earlier exact-path reference produced a median `12.1544` decode tok/s.
Segmented compiled decode produced `12.6206` tok/s, a `3.8%` improvement, with
flat measured peak memory.

The compiled candidate did **not** preserve the deterministic completion
digest (`c84d…` reference versus `8f60…` candidate), so it is not a
production recommendation. It is retained here as reproducible experimental
work while segment-level bisection and speculative-cache work continue.

The packed MoE-front path is numerically exact in the focused Metal tests, but
the full TP2 A/B produced `11.9632` tok/s and added approximately `7.45 GB` per
rank. It therefore remains disabled and is not part of the recommended
configuration. The hidden packed copy is invalidated before sharding and
rebuilt whenever any authoritative projection array changes.

The branch also exposes fail-closed prompt-lookup speculative verification
without an external draft model:

```python
stream_generate(
    model,
    tokenizer,
    prompt,
    prompt_lookup_num_tokens=7,
    prompt_lookup_max_ngram_size=4,
    speculative_round_callback=callback,
)
```

For transactional Kimi K3, the valid proposal range is 1–7 tokens because the
verification forward also includes the current input token. The CLI
equivalents are `--prompt-lookup-num-tokens`,
`--prompt-lookup-max-ngram-size`, and `--speculative-round-stats`. Kimi K3
cache updates are transactional across recurrent and attention layers, and
pipeline-parallel or batched KV-cache configurations fail closed. Prompt
lookup initializes from the explicit prompt token IDs and appends committed
outputs. Cached callers may also pass `prompt_lookup_history` to seed the
lookup index from the full logical token history without re-prefilling it; the
history must end with the explicit prompt suffix. Cache-only prefixes remain
invisible unless the caller supplies that history.

The EXO repository contains the strict target-verification and divergence
diagnostic tools. A sampled width-2 run kept the same top-1 continuation but
diverged numerically in the first recurrent KDA layer, so the strict
verification gate remains failed.

The bisection selector passed 13 focused single-rank Metal tests (with one
expected TP2 skip) and a local two-rank exact logits/cache comparison for full
and mixed schedules. After integrating MoE-front packing, the combined branch
completed 62 focused Metal/unit tests across packed/fused experts, compiled
decode, speculative cache transactions, generation lifecycle, and prompt
lookup (61 passed, one expected TP2 skip), plus all 22 prompt-cache
regressions. Full-model strict-v3,
long-memory, acceptance-rate, and live-cluster speed gates remain outstanding.
