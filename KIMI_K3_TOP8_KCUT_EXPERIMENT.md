# Kimi K3 native-top16 to top8 K-cut experiment

## Decision

This branch is a **local candidate for a fresh-namespace fleet A/B only**.  It
must not replace the native-top16 service.  Top8 changes the model and is
deliberately lossy: it retains and renormalizes only the eight highest routed
experts instead of the checkpoint's native sixteen.

The original strict local `>=1.2x at every shape` gate is false because the
final 31-pair Q3 screen reached a `1.1446x` median paired speedup.  The active
goal weights prefill three times decode, however, and both prefill shapes were
stable, exact relative to stock K8, and near `2x`.  That is sufficiently
outsized to justify one isolated fleet A/B with explicit quality gates.  It is
not evidence for promotion.

No remote host or live service was touched by this branch.

## Safety contract

The selector is:

```text
MLX_LM_KIMI_K3_EXPERT_TOP_K
```

- Absent: use the checkpoint configuration verbatim.  This is the exact
  default-off behavior and preserves small/future non-released test models.
- `16`: explicitly select released Kimi K3's native behavior.
- `8`: select the lossy K-cut experiment.
- Any other explicit value: fail closed.
- Explicit `16` or `8` also fails closed unless the checkpoint's native
  `num_experts_per_token` is exactly 16.

The value is resolved when each sparse MoE module is constructed.  A launcher
must therefore set it before model loading; changing it in a loaded process is
unsupported.

## Fast-path audit

| Path | Native top16 | Experimental top8 | Action |
|---|---|---|---|
| Fused router | Exact | Exact | Narrow cached Metal specialization for 8 and 16; K8 also supports 1-4096 row prefill |
| Gate/up + SiTU | Route-dynamic | Route-dynamic | Unchanged |
| Tuned down QMV | Route-dynamic | Route-dynamic | Unchanged |
| Prefill sort/down/combine | Exact top16 | Exact top8 | Reduction width templated for 8/16 with stock BF16 association |
| Fused down/reduce | Top16-only | Falls back | Deliberately not adapted; the projected-KV A2 service has this selector off |
| TP2 expert sharding | Independent of route count | Independent of route count | Unchanged |
| Projected Q3 cache transaction | Independent of route count | Independent of route count | Unchanged |

## Local exactness and performance

Device: Apple M3 Max (`applegpu_g15s`, 64 GiB).  The benchmark used the exact
released rank-local tensor contract: 896 experts, 3584 hidden, 1536 TP2
intermediate, affine 2-bit/group-128 projections, and BF16 activations/router
weights.  It used deterministic synthetic packed values because this host does
not contain the full UVMAX checkpoint.  Thus it is representative of selected
route geometry and traffic, not checkpoint output quality or end-to-end EXO
throughput.

| Shape | Native top16 median | K8 median | Ratio of medians | Median paired speedup | Stock K8 exact | Original per-shape gate |
|---|---:|---:|---:|---:|---|---|
| Q3 | 2.583958 ms | 2.202791 ms | 1.1730x | 1.1446x | yes | fail |
| 512-token prefill | 122.656583 ms | 62.633083 ms | 1.9583x | 1.9597x | yes | pass |
| 4096-token prefill | 979.898834 ms | 483.813583 ms | 2.0254x | 1.9965x | yes | pass |

The Q3 breakdown confirms the limitation rather than hiding it:

- route stage: `1.0573x` median paired;
- route-dynamic expert stage after routing: `1.1612x` median paired.

The full Kimi K3 suite passed: `273 passed, 3 skipped, 257 subtests passed`.
The focused implementation checks cover default-off parity, explicit top16
parity, fail-closed selector parsing, stock-K8 equality, Q1/Q3/prefill shapes,
TP2 expert geometry, route-dynamic K8 kernels, the intentional top16-only
down/reduce fallback, and projected-cache transaction coexistence.

Machine-readable samples and the benchmark protocol are in
`benchmarks/results/kimi-k3-top8-kcut-m3max-20260806.json`.

## Required fleet A/B/A

Use a fresh namespace and fresh `EXO_HOME`; do not mutate or reuse live A2.
Keep EXO, MLX core, model shards, TP2 placement, four RDMA rails, projected-KV
cache, DSpark policy, prompts, and every unrelated selector identical.

1. **A1 native control:** selector `16`, canonical 574-token prompt / 128-token
   decode, then the actual 6754-token prefill / two-token tail.
2. **B1 K8:** selector `8`, same two workloads and repetitions.
3. **A2 restoration:** selector `16`, same workloads, then restore the exact
   incumbent service regardless of outcome.

Use at least three repetitions per arm.  Record prompt tok/s, effective decode
tok/s, `U = 3 * prompt_tok_s + effective_decode_tok_s`, accepted/drafted DSpark
tokens, emitted tokens per speculative round, errors/fallbacks, completion
digest stability within each arm, and peak memory.

The native completion digest is not a K8 correctness target.  Reject the
candidate on nondeterminism within K8, structural/tool-call failure, severe
DSpark acceptance collapse, or less than `+10%` weighted utility.  Before any
optional fast/lossy service is considered, run repository-editing, JSON/tool
schema, long-context retrieval, and coding-quality checks.  K8 must never be
silently promoted as the exact/native service.
