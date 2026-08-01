# Kimi K3 row-tiled KDA decode screen

This is a default-off screen of the existing exact row-tiled KDA Metal kernel
at the rank-local K3 TP2 decode geometry.  It changes only the recurrent KDA
kernel for `T=1`; the optimization is independent of prompt length because
KDA carries a fixed-size recurrent state.

## Single-call command

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

This first method synchronizes every sub-millisecond kernel, so its apparent
10 microsecond/layer saving is contaminated by roughly 0.2 ms of host and sync
overhead.  It is retained only as a record of the initial screen.

## Amortized dependent-chain command

```bash
python benchmarks/kimi_k3_kda_decode.py \
  --heads 48 --rows 1 2 4 8 --chain-length 256 \
  --warmup 3 --repeats 21
```

This builds each 256-kernel dependent graph before timing, evaluates it with
one final synchronization, randomizes variant order, and divides elapsed time
by 256.  On the same 40-core M3 Max it measured:

| Variant | Median per kernel | Speedup |
| --- | ---: | ---: |
| accepted | 20.367 us | 1.000x |
| row 1 | 20.411 us | 0.998x |
| row 2 | 18.697 us | 1.089x |
| row 4 | 17.827 us | 1.142x |
| row 8 | 20.086 us | 1.014x |

Every 256-step final output and recurrent state was bit-identical.  Row four
won locally, but the 80-core M3 Ultra has different occupancy: row two retains
768 threadgroups per rank versus 384 for row four.  The runtime therefore
keeps a default row-two screen and accepts an explicit
`MLX_LM_EXPERIMENTAL_KDA_ROW_DECODE_ROWS={1,2,4,8}` override for the Ultra
sweep.  No tile should be promoted before that sweep.

The realistic row-two saving is about 1.7--3.3 microseconds per KDA layer, or
0.115--0.230 ms over 69 layers.  Applied perfectly to the 69.893 ms/token TP2
baseline, that moves 14.3076 only to roughly 14.33--14.35 tok/s.  It is a small
additive candidate, not a route to 17 tok/s.

The runtime gate is `MLX_LM_EXPERIMENTAL_KDA_ROW_DECODE=1`; it is disabled by
default and cannot enable the prefill specialization.  Promotion requires the
amortized tile sweep on an M3 Ultra and then a matched TP2 short-context A/B.
