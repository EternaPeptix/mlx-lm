# Experimental Kimi K3 fused down projection and route reduction

This branch adds an opt-in, decode-only Metal path after Kimi K3's fused
gate/up/SiTU kernel. The new dispatch computes the sixteen selected-expert
down projections, preserves each projection's FP32 accumulation and BF16
output cast, applies the BF16 router weights, and reproduces MLX's fixed
top-16 reduction tree. It emits the final `[1, 1, 3584]` routed branch instead
of materializing `[1, 1, 16, 3584]`.

Both exact-expert flags are required:

```bash
export MLX_LM_KIMI_K3_FUSED_EXPERTS=1
export MLX_LM_KIMI_K3_FUSED_DOWN_REDUCE=1
```

The new flag is disabled by default. Unsupported platforms, training, batch
or prompt shapes, route counts, dtypes, quantization layouts, and non-K3
dimensions return to the unchanged MLX-LM graph.

## Exactness

The focused local Metal regression subset passes 23 tests. The final combined
Kimi K3 run passes 59 tests with one expected skip. Coverage includes all
supported output and SIMD-group tiles, dynamic router weights, cached dynamic
projection weights, the complete fused gate/up/down/reduction path,
fail-closed cases, and the existing packed-front, compiled-decode,
speculative-cache, and expert kernels.

Top-16 reduction needs special handling. MLX's small BF16 strided-reduction
kernel forms partials for `(0, 8)`, `(1, 9)`, through `(7, 15)`, then folds
partials 1 through 7 into partial 0. A naive serial slot-0-through-15 loop is
not bit exact.

## Synthetic exact-geometry microbenchmark

The checked-in benchmark uses K3's TP2 rank-local `1536 -> 3584`, affine
2-bit/group-128 down projection with sixteen selected expert banks. It
compares the fusion with the existing tuned gather-QMV plus MLX router
multiply/reduction:

```bash
python benchmarks/kimi_k3_fused_down_reduce.py \
  --warmup 50 --iterations 500 --trials 21 \
  --tiles 4 --simdgroups 16
```

On the local 64 GB M3 Max, three fresh processes each ran 21 alternating
trials of 500 iterations:

| Process | Current tuned ms/layer | Fused R4/S16 ms/layer | Paired speedup |
| ---: | ---: | ---: | ---: |
| 1 | 0.433655 | 0.341543 | 1.2697x |
| 2 | 0.296906 | 0.292719 | 1.0143x |
| 3 | 0.320327 | 0.318584 | 1.0055x |

The fused path won in all three processes, but the large run-to-run range
means the M3 Max timing is directional rather than a stable throughput claim.
The median paired saving is `0.004188 ms/layer`, or approximately `0.385 ms`
across 92 sparse layers. The ratio of the two process-median timings is
`1.0055x`; the median of the paired speedups is `1.0143x`.

The memory result was stable in all three processes. Measured peak allocation
above the resident synthetic inputs fell from `0.116 MiB` to `0.007 MiB`; the
eliminated BF16 expert-row tensor itself is `0.109 MiB/layer`.

This path subsequently passed a real-weight, two-rank M3 Ultra TP2 gate. Five
canonical 575-prompt-token/128-generation-token repetitions retained
completion digest
`c84d0f0464acc5f0226e5a9686e2bb8ed4b243064dfafb99d7aa7fc5cd5b0c71`
and improved median decode from `13.2985` to `13.5835` tok/s (`+2.14%`).
Three 1,067-token coding-prompt repetitions retained digest
`9936f17d98ac76b2a3ad3ab768e78fae5379259da0b745881f06e7cf9c7a7959`
and improved median decode from `13.2637` to `13.5102` tok/s (`+1.86%`).
The feature remains opt-in because its supported geometry is deliberately
narrow.
