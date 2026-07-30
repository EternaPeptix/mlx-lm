# Experimental exact Kimi K3 expert kernel

This branch contains an opt-in Metal decode path for Kimi K3's affine
2-bit, group-size-128 routed experts. It fuses gate and up projection with
SiTU, then uses a tuned exact down projection. It does not dequantize,
requantize, or repack weights. Unsupported platforms, shapes, dtypes,
quantization modes, prefill, training, or activation settings use the stock
MLX-LM path.

Enable it before importing MLX-LM:

```bash
export MLX_LM_KIMI_K3_FUSED_EXPERTS=1
```

It remains disabled by default because it is a narrow Kimi K3 specialization.
The full two-rank A/B below validates the supported UVMAX TP2 configuration;
unsupported configurations continue to fail closed to the stock path.

## Real-weight microbenchmark

A fully warmed, dependency-preserving 92-step chain rotated routes across all
experts on four representative TP2 rank-local layers:

| Layer | Stock ms/layer | Exact candidate ms/layer | Reduction | Exact |
| ---: | ---: | ---: | ---: | :---: |
| 1 | 0.184349 | 0.144863 | 21.42% | yes |
| 17 | 0.186681 | 0.144792 | 22.44% | yes |
| 47 | 0.184505 | 0.144285 | 21.80% | yes |
| 91 | 0.186702 | 0.144409 | 22.65% | yes |

The isolated estimate saves about 3.77 ms per generated token, projecting a
measured 12.105 tok/s full-model baseline to roughly 12.68 tok/s. This is not
a claim of a measured end-to-end speedup, and it is not sufficient by itself
to reach 17 tok/s.

## Full-model TP2 A/B

The end-to-end run used two 512 GB M3 Ultra Mac Studios, rank-local TP2,
four-rail JACCL, temperature zero, no prefix cache, and the exact
`laguna8` hidden-state asynchronous schedule. The authoritative packed
MoE-front and row-4 KDA prefill options were enabled in both arms; only
`MLX_LM_KIMI_K3_FUSED_EXPERTS` changed.

| Prompt / decode | Control tok/s | Fused tok/s | Gain | Repetitions | Exact |
| --- | ---: | ---: | ---: | ---: | :---: |
| 575 / 128 canonical | 12.9824 | 13.2985 | 2.44% | 3 / 5 | yes |
| 1,067 / 128 coding | 12.9288 | 13.2637 | 2.59% | 3 / 3 | yes |

All canonical candidate repetitions produced
`c84d0f0464acc5f0226e5a9686e2bb8ed4b243064dfafb99d7aa7fc5cd5b0c71`.
All coding-prompt repetitions produced
`9936f17d98ac76b2a3ad3ab768e78fae5379259da0b745881f06e7cf9c7a7959`.
Peak MLX memory remained approximately 414 GB per rank for the canonical
case.

The sanitized per-repetition record is published with the coordinated
[EXO branch](https://github.com/EternaPeptix/exo/blob/experiment/kimi-k3-uvmax-optimization-stack/docs/kimi_k3_tp2_benchmark_20260730.json).
The result establishes an exact, repeatable decode improvement at two prompt
sizes. It does not imply that expert fusion removes the separate
context-dependent attention cost at very large contexts.
