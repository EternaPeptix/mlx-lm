# Experimental Kimi K3 exact fused router

This default-off decode specialization fuses the Kimi K3 sigmoid router,
exact normalization, group masking, stable top-16 selection, and selected
weight extraction into one Metal dispatch.

Enable it with:

```bash
export MLX_LM_KIMI_K3_FUSED_ROUTER=1
```

The adapter is inference-only and fails closed unless the released K3
single-token BF16/FP32 geometry is present.

## Why the reduction order matters

An earlier candidate used a SIMD sum for the sixteen raw selected scores.
Although its selected expert IDs matched the reference, a one-ULP FP32
denominator difference could cross a BF16 rounding midpoint. That candidate
changed the canonical full-model completion digest and was rejected.

The accepted kernel deliberately reproduces the stock MLX expression's
slot-0-through-15 sequential FP32 denominator fold. A 40,960-row focused fuzz
test matched every selected ID and BF16 weight exactly.

## Live TP2 result

On top of the exact fused-down and AttnRes/RMSNorm stack, five canonical
575-prompt-token/128-generation-token repetitions retained completion digest
`c84d0f0464acc5f0226e5a9686e2bb8ed4b243064dfafb99d7aa7fc5cd5b0c71`
and produced:

```text
13.8496, 13.7280, 13.8051, 13.8160, 13.8415 tok/s
```

The median was `13.8160` tok/s versus `13.6714` for the matched AttnRes stack
(`+1.06%`), saving `0.766 ms/token`.

Three 1,067-token coding-prompt repetitions retained digest
`9936f17d98ac76b2a3ad3ab768e78fae5379259da0b745881f06e7cf9c7a7959`
and produced a median `13.7809` tok/s versus `13.6354` for the matched
AttnRes stack (`+1.07%`).
