# Experimental Kimi K3 fused router

This branch adds a decode-only Metal kernel that combines FP32 sigmoid,
correction-bias top-16 selection over 896 experts, uncorrected router-weight
gathering, normalization, and BF16 output. It is disabled by default:

```bash
export MLX_LM_KIMI_K3_FUSED_ROUTER=1
```

Unsupported shapes, training, non-Metal execution, and prompt batches use the
stock MLX-LM graph.

## Isolated result

The focused suite passes exact random, batched, tied-value, infinity, NaN,
all-NaN, and complete sparse-MoE cases. In nine paired M3 Max trials of 3,000
calls per arm, stock selection had a `0.265045 ms/layer` median and the fused
kernel had a `0.232443 ms/layer` median. The median paired saving was
`0.051914 ms/layer`, approximately `4.78 ms` across 92 sparse layers.

## Live TP2 rejection

The full two-rank M3 Ultra screen did **not** pass the deterministic output
gate. On the canonical 575-prompt-token/128-generation-token case, five
repetitions reached a `13.718854` tok/s median but all changed the expected
completion digest
`c84d0f0464acc5f0226e5a9686e2bb8ed4b243064dfafb99d7aa7fc5cd5b0c71`
to
`905f41a15d7933fd33da36382e1766469fffbe17c9da03084c22e62376f77cfc`.

This kernel is therefore a correctness investigation, not a release
recommendation. The coordinated v2 branch stops at the exact fused-down
runtime and does not include or enable this router.
