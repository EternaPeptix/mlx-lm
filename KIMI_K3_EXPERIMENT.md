# Kimi K3 EXO/MLX experiment

This branch is one part of a coordinated public experiment:

- [EXO](https://github.com/EternaPeptix/exo/tree/experiment/kimi-k3-uvmax-optimization-stack-v6)
- [MLX-LM](https://github.com/EternaPeptix/mlx-lm/tree/experiment/kimi-k3-uvmax-optimization-stack-v6)
- [MLX](https://github.com/EternaPeptix/mlx/tree/experiment/kimi-k3-uvmax-optimization-stack-v6)

It contains the Kimi K3 support and TP2 changes used to run
`kernelpool/Kimi-K3-2bit-UVMAX` across two 512 GB M3 Ultra systems, including:

- external pipeline-partition support and bounded cache advancement;
- an opt-in exact-weight fused expert path;
- model-parallel output-head coverage used by EXO's vocabulary-parallel path;
- an opt-in segmented compiled decode schedule;
- an opt-in authoritative packed MoE-front path that avoids persistent duplicate
  projection storage;
- an opt-in exact row-tiled KDA prefill kernel;
- opt-in exact fused expert-down/reduction and router-selection kernels;
- an opt-in exact AttnRes-to-RMSNorm decode fusion;
- opt-in zero-copy packs for KDA's same-input skinny and wide projections;
- an opt-in exact routed-up/shared/residual output fusion for TP2 decode; and
- focused Metal and distributed tests for those paths.

The v6 checkpoint keeps the benchmark-accepted runtime code at
`95fc8ad485e8d2568eda4e468c4169f6a556919a`; later commits on this branch add
only sanitized benchmark documentation. The default-off speculative-KDA
rewrite and blockwise/tiled dual-source MLA prototypes remain on separate
research branches because none has passed the strict real-model TP2 promotion
gate. The blockwise scalar MLA prototype bounds score memory but is slower
than the accepted expanded path, so it is capacity evidence rather than a
speed candidate.

## Feature flags

`MLX_LM_KIMI_K3_FUSED_EXPERTS=1` enables the exact-weight fused expert
prototype. `MLX_LM_KIMI_K3_COMPILED_DECODE=1` enables segmented compiled
decode. `MLX_LM_KIMI_K3_PACKED_MOE_FRONT=1` enables an exact, decode-only
packed QMV for four same-input MoE-front projections.
`MLX_LM_KIMI_K3_AUTHORITATIVE_PACKED_MOE_FRONT=1` makes the packed
representation authoritative so the unpacked projection copies are not kept
for the model lifetime. `MLX_LM_EXPERIMENTAL_KDA_ROW_PREFILL=1` enables the
exact row-tiled Metal KDA recurrence for supported prefill shapes of at least
128 tokens. `MLX_LM_KIMI_K3_FUSED_DOWN_REDUCE=1`, together with
`MLX_LM_KIMI_K3_FUSED_EXPERTS=1`, fuses the selected experts' down
projections, BF16 router multiplication, and MLX-compatible top-16 reduction
without materializing the sixteen expert rows.
`MLX_LM_KIMI_K3_FUSED_ROUTER=1` fuses sigmoid, exact sequential denominator
accumulation, group masking, and stable top-16 selection.
`MLX_LM_KIMI_K3_FUSED_ATTNRES_RMS=1` fuses the two AttnRes-to-RMSNorm
boundaries in each decoder layer while retaining the stock BF16
materialization and reduction order.
`MLX_LM_KIMI_K3_PACKED_KDA_SKINNY=1` concatenates KDA's compatible
rank-local `f_a` and `b` projection rows into one authoritative quantized
backing and one decode QMV. `MLX_LM_KIMI_K3_PACKED_KDA_WIDE=1` independently
does the same for the rank-local `qkv` and full-rank gate projections.
`MLX_LM_KIMI_K3_FUSED_ROUTED_UP_ADD=1` fuses the routed affine-8 up
projection's FP32-to-BF16 boundary with the BF16 shared-branch and decoder
residual additions. It is restricted to the exact two-rank, single-token
decode contract. All of these features default to off and fail closed outside
their supported shapes.

`MLX_LM_KIMI_K3_ASYNC_DECODE_BOUNDARIES` enables eager, decode-only
asynchronous evaluation boundaries. It accepts `none`, `laguna8`, `block8`,
`all`, individual layer indices, or inclusive ranges. The companion
`MLX_LM_KIMI_K3_ASYNC_DECODE_STATE` selects `hidden` or `residual` state
submission. The path defaults to off and fails closed unless generation is
single-batch, single-token, eager Metal execution with a populated cache and a
full non-pipeline model. Compiled decode and asynchronous boundaries cannot be
enabled together.

`MLX_LM_KIMI_K3_ASYNC_DECODE_WIDTH3=1` is a separate strict opt-in for the
width-three speculative verifier. It requires `laguna8`, hidden-state mode,
projected KV, and ReplaySSM. Unlike the single-token path, MLX-LM does not
submit work while the distributed verifier graph is being built. Callers must
explicitly pass `defer_async_decode_boundaries=True` to either target-forward
method; the immutable result then exposes the ordered hidden roots as
`deferred_async_decode_states`. A distributed caller may submit those roots
only after every rank has agreed that graph construction succeeded. Generic
three-token forwards and prompt prefill retain the legacy path, and a requested
deferred build outside a fresh authenticated width-three transaction fails
closed.

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
prefill, exact fused experts, fused down/route reduction, AttnRes/RMSNorm
fusion, exact fused routing, zero-copy KDA skinny/wide projection packing, and
the exact routed-up/shared/residual output fusion.
On the canonical 575-token prompt and 128-token decode, five repetitions
produced a median `14.2375` decode tok/s. The routed-up/add fusion adds
`+0.91%` over the wide-KDA stack (`14.1097`) and saves `0.636 ms/token`; the
complete stack is `+4.81%` over fused down/reduction alone (`13.5835`).
Every repetition retained completion digest
`c84d0f0464acc5f0226e5a9686e2bb8ed4b243064dfafb99d7aa7fc5cd5b0c71`.
A separate 1,067-token coding-prompt screen produced a three-run median
`14.2155` tok/s, `+0.99%` over the wide-KDA stack (`14.0764`) and
`+5.22%` over fused down/reduction alone (`13.5102`), while retaining digest
`9936f17d98ac76b2a3ad3ab768e78fae5379259da0b745881f06e7cf9c7a7959`.
Peak memory remained approximately `414 GB` per rank for the canonical case.

The routed-up/add kernel won 29 of 31 paired M3 Max microbenchmark trials. Its
paired median saving was `0.008308 ms/layer`, or a mechanical
`0.742 ms/token` across the 92 sparse layers. The live TP2 saving landed at
`0.636 ms/token`, within the predicted range.

The first fused-router prototype used a SIMD reduction for the sixteen raw
scores. It reached `13.7189` median decode tok/s, but all five live
repetitions changed the canonical digest to `905f…`. The corrected kernel
reproduces MLX's slot-0-through-15 FP32 denominator fold exactly. On top of
the AttnRes stack it retained the canonical digest and improved the five-run
median from `13.6714` to `13.8160` tok/s (`+1.06%`). The rejected SIMD
variant is not part of this branch.

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
[EXO branch](https://github.com/EternaPeptix/exo/blob/experiment/kimi-k3-uvmax-optimization-stack-v6/docs/kimi_k3_tp2_benchmark_20260730.json).

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

The EXO repository contains strict target-verification and divergence
diagnostic tools. Width 1 remains exact. Widths 2 and above retained the
sampled top-1 continuation but failed the numerical/cache gate. The
real-checkpoint layer-0 localizer found KDA attention exact when both paths
received the same prepared input, narrowing the first unresolved divergence
to the surrounding decoder-layer preparation/wrapper path. The strict gate
therefore remains failed.

The bisection selector passed 13 focused single-rank Metal tests (with one
expected TP2 skip) and a local two-rank exact logits/cache comparison for full
and mixed schedules. After integrating MoE-front packing, the combined branch
completed 62 focused Metal/unit tests across packed/fused experts, compiled
decode, speculative cache transactions, generation lifecycle, and prompt
lookup (61 passed, one expected TP2 skip), plus all 22 prompt-cache
regressions. Full-model strict-v3 multi-token equivalence remains failed;
long-memory and live prompt-lookup acceptance/speed gates remain outstanding.
