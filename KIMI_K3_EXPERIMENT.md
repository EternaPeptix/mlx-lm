# Kimi K3 EXO/MLX experiment

This branch is one part of a coordinated public experiment:

- [EXO](https://github.com/EternaPeptix/exo/tree/experiment/kimi-k3-uvmax-optimization-stack)
- [MLX-LM](https://github.com/EternaPeptix/mlx-lm/tree/experiment/kimi-k3-uvmax-optimization-stack)
- [MLX](https://github.com/EternaPeptix/mlx/tree/experiment/kimi-k3-uvmax-optimization-stack)

It contains the Kimi K3 support and TP2 changes used to run
`kernelpool/Kimi-K3-2bit-UVMAX` across two 512 GB M3 Ultra systems, including:

- external pipeline-partition support and bounded cache advancement;
- an opt-in exact-weight fused expert path;
- model-parallel output-head coverage used by EXO's vocabulary-parallel path;
- an opt-in segmented compiled decode schedule; and
- an opt-in authoritative packed MoE-front path that avoids persistent duplicate
  projection storage;
- an opt-in exact row-tiled KDA prefill kernel; and
- focused Metal and distributed tests for those paths.

## Feature flags

`MLX_LM_KIMI_K3_FUSED_EXPERTS=1` enables the exact-weight fused expert
prototype. `MLX_LM_KIMI_K3_COMPILED_DECODE=1` enables segmented compiled
decode. `MLX_LM_KIMI_K3_PACKED_MOE_FRONT=1` enables an exact, decode-only
packed QMV for four same-input MoE-front projections.
`MLX_LM_KIMI_K3_AUTHORITATIVE_PACKED_MOE_FRONT=1` makes the packed
representation authoritative so the unpacked projection copies are not kept
for the model lifetime. `MLX_LM_EXPERIMENTAL_KDA_ROW_PREFILL=1` enables the
exact row-tiled Metal KDA recurrence for supported prefill shapes of at least
128 tokens. All of these features default to off and fail closed outside their
supported shapes.

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

The latest exact TP2 candidate combines the `laguna8` hidden-state
asynchronous decode schedule, authoritative packed MoE front, row-4 KDA
prefill, and exact fused experts. On the canonical 575-token prompt and
128-token decode, five candidate repetitions produced a median `13.2985`
decode tok/s versus `12.9824` for the matched fused-expert-off control
(`+2.44%`). Every repetition retained completion digest
`c84d0f0464acc5f0226e5a9686e2bb8ed4b243064dfafb99d7aa7fc5cd5b0c71`.
A separate 1,067-token coding-prompt screen produced `13.2637` versus
`12.9288` tok/s (`+2.59%`) and retained digest
`9936f17d98ac76b2a3ad3ab768e78fae5379259da0b745881f06e7cf9c7a7959`.
Peak memory remained approximately `414 GB` per rank for the canonical case.

The authoritative representation removes approximately `7.44 GB` decimal
(`6.93 GiB`) of persistent duplicate projection storage per TP2 rank. The
fused-expert and row-4 paths remain opt-in narrow specializations even though
the supported two-rank UVMAX configuration now has a successful exact
full-model A/B.

The row-tiled KDA prefill path is bit-exact in the focused Metal tests and uses
no scratch allocation. On an M3 Max KDA-only microbenchmark it improved the
recurrence from `0.920` to `0.644 ms` at 128 tokens (`1.43x`), from `3.255` to
`1.912 ms` at 512 tokens (`1.70x`), from `15.997` to `7.515 ms` at 2K
(`2.13x`), and from `115.852` to `36.046 ms` at 8K (`3.21x`). These are
kernel-level measurements.

The matched full-model TP2 prefill A/B reached `147.4168` prompt tok/s versus
`146.3746` at 2K target context (`+0.71%`) and `154.9942` versus a
`153.2673` warmed control at 8K (`+1.13%`). All observations retained digest
`e929f1fd6e350d723ee540dee6e7641916c622c9deb56912b1e8fddb252b52c6`,
and peak memory was unchanged within measurement noise.

The sanitized per-repetition record is published with the coordinated
[EXO branch](https://github.com/EternaPeptix/exo/blob/experiment/kimi-k3-uvmax-optimization-stack/docs/kimi_k3_tp2_benchmark_20260730.json).

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
