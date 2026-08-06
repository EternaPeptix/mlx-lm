# Experimental Kimi K3 persistent projected K/V cache

## Decision

This default-off candidate is awaiting a real-weight M3 Ultra gate. It has not
been deployed. It preserves Kimi K3's accepted expanded-Q3 arithmetic by
caching the exact BF16 projected K/V rows and projecting only each new
speculative suffix.

```bash
export MLX_LM_KIMI_K3_PROJECTED_KV_CACHE=1
export MLX_LM_KIMI_K3_PROJECTED_KV_CACHE_MAX_TOKENS=32768
```

The selector is restricted to the released TP2 (48 heads/rank from 96 global
heads) Q3 geometry, affine-6/group-64 K and V projections, BF16, Metal, and a
populated ordinary `KVCache` inside an explicitly open width-three speculative
transaction. A normal three-token prefill tail cannot activate it. Other
shapes use the accepted path.
Ordinary Q1 invalidates the expanded cache so a later Q3 verifier rebuilds it
from the authoritative latent state.

## Exactness mechanism

Naively projecting one to three new latent rows selects MLX's QMV/QVM kernels.
Their reduction order differs from the full-context QMM and changes BF16 bits.
The candidate pads every suffix to 32 rows, forcing the same QMM schedule as
the incumbent full projection, then keeps only the real rows.

The standalone production-geometry screen covered both projection
orientations, prefixes 574 and 7,704, and suffixes one, two, and three. All 12
M32-padded cases matched the full projection bit-for-bit. The naive cases all
failed and differed only in the suffix rows.

The model integration additionally covers:

- partial speculative acceptance followed by overwrite of stale tail rows;
- cancellation and projected-allocation rollback;
- a 256-row capacity crossing after partial acceptance;
- ordinary unmarked L3 prefill and unsupported-geometry fallback;
- width-two speculative fallback and nested-transaction rejection;
- prompt-state restore, which deliberately drops nonserialized projections;
- projected bytes in cache capacity accounting;
- eager and compiled ordinary-Q1 invalidation; and
- batch conversion to `BatchKVCache`, where the optimization is intentionally
  inactive.

On-disk prompt caches serialize this subclass as the portable base `KVCache`.
Loading therefore preserves the authoritative latent state but intentionally
does not reactivate projected caching inside that loaded cache object. A fresh
model/request cache is required to use the optimization.

## Local M3 Max projection screen

These are component timings, not model tok/s. Thirty-one paired canonical
trials at prefix 574 plus Q3 using the actual TP2 48-head/rank shape measured:

| Path | Median per MLA layer |
| --- | ---: |
| Full K+V projection | 0.953375 ms |
| Persistent-cache M32 append | 0.306208 ms |
| Saving | 0.647167 ms (3.113x) |

The local projection-only saving is 15.532008 ms across 24 MLA layers per
verification round. Both cached K and V were bit-exact.

At prefix 7,704 plus Q3, 21 paired trials measured 10.015875 ms/layer for the
full projection and 0.340917 ms/layer for the cached append. The 29.38x
component ratio demonstrates why this is primarily a long-context decode
optimization; it must not be extrapolated directly to end-to-end throughput.

## Memory trade-off

Across 24 MLA layers on each TP2 rank, expanded BF16 K+V costs 576 KiB per
context token in addition to the existing 27 KiB latent-plus-RoPE cache.

| Context | Expanded K+V extra | Total MLA cache/rank |
| ---: | ---: | ---: |
| 574 logical rows | 0.3153 GiB | 0.3301 GiB |
| 574 with 768-row capacity | 0.4219 GiB | 0.4417 GiB |
| 8K | 4.5000 GiB | 4.7109 GiB |
| 32K | 18.0000 GiB | 18.8438 GiB |
| 128K | 72.0000 GiB | 75.3750 GiB |
| 1M | 576.0000 GiB | 603.0000 GiB |

One-million-token projected caching is not a safe target once weights,
activations, allocator headroom, and growth transients are included. The strict
runtime cap is therefore at most 128K, with 32K the default experiment cap and
an additional live free-memory gate required before raising it. Above the
configured cap the candidate releases its expanded arrays and uses the accepted
latent/full-projection path.

## End-to-end ceiling

The accepted canonical run uses 50 verification rounds for 128 emitted tokens
at about 149.45 ms/round. Nineteen tok/s needs roughly 14.71 ms/round saved.
Prior M3 Ultra attribution measured the complete full K+V projection at about
13.90 ms/round, so this cache alone cannot guarantee 19 tok/s even if suffix
projection were free. Its theoretical canonical ceiling is about 18.9 tok/s;
the actual result will be lower. It remains the first unclosed exact candidate
with a plausible high-single-digit gain and far larger long-context upside.

## Verification

Local Metal regression result:

```text
314 passed, 3 skipped, 241 subtests passed
```

The next gate is a pinned real-weight screen on both M3 Ultras. It must prove:

1. exact K/V and target output at 574+3;
2. all acceptance boundaries and cancellation;
3. M32 suffix projection materially below the full projection;
4. canonical completion digest, posterior, and zero-fallback parity; and
5. a causal TP2 A-B-A before promotion.
