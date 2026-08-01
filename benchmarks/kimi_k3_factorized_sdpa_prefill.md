# Kimi K3 factorized SDPA prefill runbook

## Status and safety boundary

This is Stage 3 model wiring for the Stage 2 MLX primitive implemented at
`152f01807c8327ac154b8ed56dd9279a6f9506e6`.  The MLX-LM base is the accepted
public Kimi K3 v6 checkpoint
`855cf5e4cdc3ebe8e4e6d5e08ed93210687cb9fb`.

The path is strictly default-off.  The only enabling value is:

```sh
MLX_LM_KIMI_K3_FACTORIZED_SDPA_PREFILL=1
```

All other values are off.  Enabling also requires a Metal MLX runtime that
exports `mx.fast.factorized_scaled_dot_product_attention`.  Decode, training,
quantized caches, non-Metal execution, unavailable runtimes, unsupported
geometry/dtypes, and unsupported masks retain the accepted exact path.

Do not enable this flag for a production-size prompt until the runtime commit,
model-level numerical gates, and short-context allocator trace have all been
verified in the same environment.  The model intentionally does not catch an
error after a production-eligible primitive dispatch: silently entering the
primitive's score-materializing composite fallback would be unsafe at long
context.

## Model contract

For prefill only, the model passes the two Kimi K3 MLA score sources without
constructing the RoPE score tensor:

```python
mx.fast.factorized_scaled_dot_product_attention(
    q_nope,                         # [B, Hq, Q, 128]
    embed_q(kv_latent, False),      # [B, Hq, K, 128]
    unembed_out(kv_latent),         # [B, Hq, K, 128]
    q_pe,                           # [B, Hq, Q, 64]
    k_pe,                           # [B, 1, K, 64]
    scale0=(128 + 64) ** -0.5,
    scale1=(128 + 64) ** -0.5,
    mask=original_boolean_mask,
)
```

After TP2, `Hq=48`; an unsharded model has `Hq=96`.  The first fused runtime
specialization additionally requires FP16/BF16, equal input dtypes, D0/Dv=128,
D1=64, Q>8, primary and secondary KV head counts that independently divide
Hq, and a supported boolean mask (or no mask).  The model checks that exact
contract before dispatch so the runtime cannot select its composite fallback
for the intended production geometry.

`L==1` decode remains the accepted latent-cache path.  The cache is updated
before a prefill dispatch, so Q covers the new chunk while K/V and the RoPE K
source cover the old cache plus the new chunk.  Caches with `bits` remain on
the existing quantized implementation because the factorized API accepts five
plain arrays.

## Why the allocation matters at 128K and 1M

For one TP2 rank with 48 query heads and BF16 scores, the old independent RoPE
contribution is `[1, 48, Q, K]`:

```text
score bytes = 2 * 48 * Q * K
```

EXO's two long-context chunk policies keep Q*K constant:

| Context K | Q chunk | removed RoPE score | expanded K+V for one MLA layer | persistent 24-layer latent cache | boolean QxK mask |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 131,072 | 2,048 | 24 GiB | 3 GiB | 3.375 GiB | 256 MiB |
| 1,048,576 | 256 | 24 GiB | 24 GiB | 27 GiB | 256 MiB |

The expanded K+V column is the D128 working set for the active MLA layer.  The
latent-cache column is persistent across 24 MLA layers and includes D512
latent plus D64 RoPE keys.  Stage 3 removes the 24 GiB score allocation but
does not remove expanded K/V or an explicitly supplied boolean mask.  A later
D512 capacity specialization is required before 1M can avoid the 24 GiB
expanded K/V working set.

## Current Stage 2 evidence

The Stage 2 Metal test covers FP16/BF16, 48/96 query heads, primary head counts
48/12/1, a one-head secondary RoPE source, partial Q/K tiles, non-contiguous
layouts, causal and boolean masks, and unequal signed scales.  Its allocator
regression measured a 393,216-byte delta for both K=512 and K=4096 at
`B=1,H=48,Q=32,Dv=128`.  That delta is exactly the BF16 output and has zero
measured K slope.  One materialized BF16 score at K=4096 would already be
12 MiB.

This is synthetic primitive evidence, not a model promotion result.  No
model-level logits, TP2 equivalence, prefill throughput, or production context
claim is recorded by this Stage 3 wiring.

## Offline verification and later promotion sequence

Use the MLX runtime containing Stage 2 and keep the initial checks local.  Do
not build, download, deploy, or touch a live EXO service as part of this wiring
change.

1. Run the focused dispatch tests.  They monkeypatch the primitive and prove
   default-off behavior, exact five-input ordering, head/GQA shapes, both
   scales, original-mask forwarding, populated-cache lengths, and fallbacks:

   ```sh
   rtk pytest -q tests/test_kimi_k3_factorized_sdpa.py
   ```

2. With Stage 2 installed locally, rerun its numerical and allocator tests and
   confirm that the model's production geometry is reported as fused rather
   than composite.

3. Enable the flag only for short-context model gates.  Compare default-off
   and enabled logits, top-1 tokens, canonical/coding continuations, TP2 rank
   equivalence, deterministic digests, peak allocation, and prefill latency.
   Verify Q=1 decode path and throughput remain unchanged.

4. Run 128K allocator/capacity validation only after the short gates pass.
   Defer 1M until the expanded D128 K/V allocation fits or a validated D512
   capacity specialization exists.  Never exercise the composite reference at
   either production size.

