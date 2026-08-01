# Kimi K3 width-eight packed MoE front

## Status

Default-off cross-repository candidate.  The MLX-LM dispatcher admits only
ordinary width 1 or an explicitly enabled `B=1, L=8` target-verification call.
Widths 2--7, widths above 8, and batched inputs remain on the stock graph.

The candidate reuses the already-authoritative packed affine8/group-64 rows
for Kimi K3's four 7,168-input projections:

1. shared gate, 3,072 outputs;
2. shared up, 3,072 outputs;
3. router gate, 896 outputs; and
4. routed latent down, 3,584 outputs.

No weight is dequantized or requantized.  One 10,624-output QMM is split back
into the four original views.  Focused tests compare every view and the full
sparse-MoE result against the four-launch graph at width eight and require
bit equality.

## Gate

Existing packing remains controlled by either
`MLX_LM_KIMI_K3_PACKED_MOE_FRONT=1` or
`MLX_LM_KIMI_K3_AUTHORITATIVE_PACKED_MOE_FRONT=1`.  Multi-token use requires
the additional independent flag:

```text
MLX_LM_KIMI_K3_PACKED_MOE_FRONT_WIDTH8=1
```

This prevents an already-enabled decode optimization from silently changing
the speculative target graph.

## Directional performance evidence

The paired M3 Max MLX-core screen in
`benchmarks/python/k3_w8_packed_front_rowpair_paired_m3max.json` on the
width-eight row-pair branch measured 12/12 exact wins.  Four stock launches
plus benchmark-only concatenation took 0.793096 ms/layer; one packed row-pair
launch took 0.640388 ms/layer.  The production stock graph does not concatenate
its four results, so the conservative comparison excludes that benchmark-only
cost.

Combining the packed launch with the independently measured routed-up and
shared-down row-pair calls projects 16.798 ms saved over 92 sparse layers,
versus 14.266 ms for six separate row-pair calls.  Thus this MLX-LM change adds
about 2.53 ms/target call beyond the current row-pair candidate.  Applied to
the measured 255.13 ms W8 target and an assumed 8 ms draft, the 17 tok/s
break-even would move from 4.231 emitted tokens with separate row pairing to
about 4.188.

These are synthetic M3 Max kernel measurements, not a two-Mac model result.
Promotion requires the matching MLX 7,168-to-10,624 row-pair selector, a real
checkpoint test on both M3 Ultra ranks, output/cache digests, and fixed-prompt
EXO TP2 p50/p95 A/B evidence.
