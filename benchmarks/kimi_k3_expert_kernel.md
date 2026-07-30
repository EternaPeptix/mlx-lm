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

It is disabled by default pending a full two-rank model A/B.

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
