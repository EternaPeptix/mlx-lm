# Kimi K3 authoritative wide KDA projection packing

This default-off candidate stacks on the authoritative skinny KDA pack and
combines the two remaining large same-input decode projections:

```text
qkv_proj: [7168] -> [18432]  (three TP2 rank-local segments)
g_proj:   [7168] -> [6144]   (TP2 rank-local full-rank gate)
```

Enable it independently with:

```sh
MLX_LM_KIMI_K3_PACKED_KDA_WIDE=1
```

The released checkpoint uses affine 6-bit/group-64 weights. Concatenating
their packed rows is lossless because MLX packs quantized values along the
input axis. The QKV output remains first, so its Q/K/V segment order is
unchanged.

## Authoritative storage and fallback

The pack evaluates one concatenated allocation, then repoints `qkv_proj` and
`g_proj` to zero-copy row views of it. The original parameter names and module
calls therefore remain available for multi-token prefill without a persistent
duplicate.

At TP2 geometry the source and packed representations are both exactly
`143,130,624` bytes per KDA layer. The local Metal storage test observed:

- `0` bytes of steady-state allocator growth after installation;
- `143,130,624` bytes of transient installation growth; and
- about `4.4–5.5 ms` to install one layer.

Across 69 KDA layers these projections already occupy about `9.20 GiB` per
rank. Packing does not increase that resident total. Lazy layer-by-layer
installation needs about `136.5 MiB` of transient headroom.

The candidate:

- is inference-only, default-off, and restricted to one total token;
- requires the full-rank gate, a 3:1 QKV/gate row ratio, and matching affine
  6-bit/group-64 layouts;
- keeps T>1 calls on the original projection modules;
- rebuilds if a source array is replaced; and
- detaches source views before tensor-parallel resharding or explicit
  invalidation.

## Exactness

Focused tests cover packed-byte and output equality, source mutation,
invalidation, unsupported layouts, unchanged prefill, parameter-name
stability, deployed-geometry storage, and a compiled KDA decode with both
skinny and wide packs enabled. All wide outputs were bit-for-bit equal to the
skinny-only reference.

## Incremental timing after skinny packing

The checked-in benchmark compares the same compiled projection/consumer
region with skinny packing enabled in both arms:

```sh
PYTHONPATH=. python benchmarks/kimi_k3_packed_kda_wide.py \
  --warmup 50 --iterations 50 --trials 31
```

On the local 64 GB M3 Max, the short-block run produced:

| Configuration | Median ms/KDA layer | Median paired speedup |
| --- | ---: | ---: |
| Skinny only | 0.68091 | 1.0x |
| Skinny + wide | 0.66374 | 1.0231x |

The `0.01717 ms/layer` median delta projects to about `1.18 ms/token` across
69 KDA layers. A separate 300-iteration × 9-trial run projected `1.43
ms/token`.

This small delta is sensitive to thermal and scheduling drift: a
1,000-iteration × 11-trial run reversed by `0.54 ms/token`, even though eager
timings still favored packing. Earlier real rank-local measurements of QKV
plus gate alone saved `0.0226 ms/layer`, or `1.56 ms/token` mechanically.
Taken together, a realistic pre-live expectation is roughly **0–1.5
ms/token**, centered near **1.2 ms/token**, not a guaranteed throughput gain.
A canonical two-rank completion-hash and throughput A/B remains the retention
gate.
