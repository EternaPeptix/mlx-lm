# Kimi K3 row-tiled KDA decode screen

This is a default-off screen of the existing exact row-tiled KDA Metal kernel
at the rank-local K3 TP2 decode geometry.  It changes only the recurrent KDA
kernel for `T=1`; the optimization is independent of prompt length because
KDA carries a fixed-size recurrent state.

## Command

```bash
python benchmarks/kimi_k3_kda_prefill.py \
  --tokens 1 --heads 48 --rows 1 2 4 8 --warmup 20 --repeats 101
```

## Local screen

Hardware: Apple M3 Max, 64 GB.  Inputs were BF16, recurrent state was FP32,
and all candidate outputs and states were bit-identical to the accepted
one-row-per-SIMD kernel.

| Rows per SIMD | Median | Speedup |
| ---: | ---: | ---: |
| accepted kernel | 0.24471 ms | 1.000x |
| 1 | 0.23792 ms | 1.029x |
| 2 | 0.23471 ms | 1.043x |
| 4 | 0.24613 ms | 0.994x |
| 8 | 0.24417 ms | 1.002x |

The row-2 candidate saves about 0.010 ms per KDA layer in this isolated local
screen.  Even if all 69 KDA layers realize the same saving on an M3 Ultra, its
optimistic contribution is only about 0.69 ms/token.  It is therefore a small
additive decode candidate, not a route from 14.31 to 17 tok/s by itself.

The runtime gate is `MLX_LM_EXPERIMENTAL_KDA_ROW_DECODE=1`; it is disabled by
default and cannot enable the prefill specialization.  Promotion requires a
real M3 Ultra layer microbenchmark and then a matched TP2 short-context A/B.
