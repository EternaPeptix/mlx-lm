# Kimi K3 fused expert verification-width screen

## Result

Promote only verification width 2.  Keep width 1 unchanged and fail closed to
the stock sorted gather-QMM graph at every other sequence width.

The candidate extends the existing exact Kimi K3 TP2 fused expert path so that
the Metal grid treats the token and selected-expert slot as independent axes.
Both the fused gate/up/SiTU projection and the fused down/router reduction are
bit exact against the stock graph at width 2.

## Method

- Host: Apple M3 Max, 64 GB.
- Geometry: Kimi K3 TP2 rank-local dimensions, affine 2-bit/group-size 128,
  hidden 3584, intermediate 1536, top-k 16.
- Expert pool: 128 experts with each token selecting a shifted 16-expert set.
- Cache pressure: two independent full weight banks chained per trial.
- Timing: 12 fresh Python processes, five warmups and 31 alternating-order
  stock/fused pairs per process.
- Dependency: every bank consumes the prior bank's output.
- Correctness: every process first requires a bit-exact full-output comparison.

Reproduction command:

```shell
PYTHONPATH=. python benchmarks/kimi_k3_fused_expert_verify.py \
  --experts 128 --banks 2 --widths 2 --warmup 5 --trials 31
```

## Width-2 measurements

| Metric | Result |
|---|---:|
| Median stock time across processes | 0.746428 ms/layer |
| Median fused time across processes | 0.692219 ms/layer |
| Ratio of those medians | 1.0783x |
| Median of per-process paired savings | 0.057730 ms/layer |
| Processes with positive paired median | 11/12 |
| Processes with lower fused median | 10/12 |
| Positive individual trial pairs | 260/372 |
| Estimated saving over 92 layers | 5.311 ms/target call |

System activity caused substantial process-to-process variance, including two
raw-median reversals.  The paired statistic remained positive in 11 of 12
processes.  This is enough to retain a default-off width-2 maintenance
candidate, but not enough to enable it by default or extrapolate its speedup to
M3 Ultra without a live TP2 A/B.

## Rejected widths

An earlier 128-expert screen exercised widths 3, 4, 7, and 8 with the same
bit-exact kernel mapping.  Those widths were noisy and at least one fresh
process regressed at width 3 or 8.  Width 8 was approximately neutral overall.
They are deliberately excluded from dispatch instead of being hidden behind a
single permissive `L <= 8` condition.

## Production gate

The existing fused-expert environment flags remain default off.  Dispatch also
requires the exact Kimi K3 TP2 shapes, BF16 activations, affine 2-bit weights,
group size 128, batch size 1, top-k 16, and sequence width 1 or 2.  Unsupported
inputs return to the stock MLX-LM path.

Before promotion, repeat the benchmark on both M3 Ultra ranks and run an EXO
width-2 target-verifier A/B with fixed prompt, checkpoint, cache state, and
output/cache digests.
