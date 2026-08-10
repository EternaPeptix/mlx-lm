# Kimi K3 exact width-4 expert screen on M3 Max

## Decision

Keep this candidate default-off and do not deploy it.  The dedicated width-4
front kernel is exact and repeatably faster, but retaining MLX's native
affine2 down projection yields only a small full-chain improvement.  It does
not recover the width-4 verification regression or provide a plausible bridge
from the current 19.8096 tok/s width-4 result to 25 tok/s.

## Candidate

The candidate is intentionally restricted to the production TP2 expert
geometry: batch 1, verification width 4, top-k 8, 896 experts, BF16,
2-bit/group-128 affine weights, and rank-local 3584 -> 1536 -> 3584 expert
shapes.  Four SIMD groups own the four verification tokens in the gate/up/SiTU
Metal launch.  The selected tile computes eight outputs per SIMD group.  The
down projection remains MLX's native `gather_qmm(mode="affine2")` path.

Activation requires the independent selector
`MLX_LM_KIMI_K3_FUSED_EXPERT_WIDTH4_EXACT=1`.  Missing and malformed values
fail closed, and adjacent widths bypass the path.

## Method

- Device: local M3 Max Metal; no cluster host or live service was used.
- Native MLX core: `work/mlx-k3-affine6-q3-triplet-v1/python`.
- Four distinct, contiguous 896-expert banks per process.
- Five fresh processes, each with 5 warmups, 3 iterations per sample, and 21
  alternating-order paired trials.
- A 384 MiB cache scrub preceded every measured arm.
- Control: native affine2 gate, up, and down projections.
- Candidate: width-4 token-packed gate/up/SiTU, then native affine2 down.
- Exactness: front, down, and dependent full-chain outputs compared bitwise.

## Results

All five processes produced the same exact output SHA-256:
`a5bf4db4fc4660c732eca2ab3b405e45bff1081c1ba904ce7567e0acc1a71894`.

| Process | Control ms/layer | Candidate ms/layer | Independent saving | Paired saving | Paired wins |
|---:|---:|---:|---:|---:|---:|
| 33172 | 0.653920 | 0.630479 | 0.023441 ms | 0.017135 ms | 12/21 |
| 33331 | 0.673816 | 0.651556 | 0.022260 ms | 0.033413 ms | 15/21 |
| 33433 | 0.602865 | 0.597750 | 0.005115 ms | 0.015760 ms | 12/21 |
| 33545 | 0.783510 | 0.779396 | 0.004115 ms | 0.017431 ms | 13/21 |
| 33591 | 0.690333 | 0.662222 | 0.028111 ms | 0.024382 ms | 13/21 |

Across fresh processes, the median control and candidate measurements are
0.673816 and 0.651556 ms/layer (1.034x).  The median paired saving is
0.017431 ms/layer; every process favors the candidate, with 65/105 total
paired wins.  Projected over 69 expert layers, that is about 1.203 ms per
verification round, only 3.9% of the measured 31.2 ms/round gap to 25 tok/s.
Holding all other costs constant would move the 19.8096 tok/s width-4 result
to only about 19.97 tok/s.

The front component itself has a 0.034563 ms/layer median paired saving across
processes.  Roughly half is lost once the intermediate expert tensor is
materialized and consumed by native down projection.  A larger gain therefore
requires eliminating that boundary or attacking another round component, not
further tuning this launch tile.

## Reproduction

Run from this worktree with the accepted local affine2 MLX core first on
`PYTHONPATH` after this MLX-LM checkout:

```sh
rtk env PYTHONPATH="$PWD:/Users/jeweled/Documents/Codex/2026-07-24/we/work/mlx-k3-affine6-q3-triplet-v1/python" \
  /Users/jeweled/Documents/Codex/2026-07-24/we/work/exo-k3-authoritative-pack/.venv/bin/python \
  benchmarks/kimi_k3_width4_fused_expert.py \
  --banks 4 --warmup 5 --iterations 3 --trials 21 \
  --front-results 8 --down-results 4 --native-down \
  --cache-scrub-mib 384 --json-output /tmp/k3-width4.json
```

Raw fresh-process results are the adjacent
`kimi_k3_width4_fused_expert_m3max_p{1..5}.json` files.
