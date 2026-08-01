# Kimi K3 DSpark and exact ReplaySSM prototype

Date: 2026-08-01

Status: local, default-off implementation slice. This branch did not change an
EXO or Mac inference process and has not run the production checkpoint.

## Scope

This branch adds the target- and draft-side primitives needed to evaluate the
public `RadixArk/Kimi-K3-DSpark` checkpoint:

- post-layer target hidden-state taps at layers `7, 23, 51, 67, 83`;
- a strict contract for the released 2,249,289,601-parameter BF16 drafter;
- the five-layer Qwen3 GQA backbone, vanilla rank-256 Markov head, and
  confidence head with exact checkpoint key names;
- a pinned local-file loader with key, shape, dtype, element-count, byte-size,
  and SHA256 attestation;
- target embedding and vocabulary-head binding by reference, so the drafter
  does not own duplicate copies;
- an isolated greedy proposer that leaves target verification and acceptance
  to its caller;
- replicated-drafter placement as the only accepted TP2 placement; and
- an opt-in exact ReplaySSM rollback path for Kimi K3 KDA state.

The checkpoint's `block_size=7` is model-native gamma 7: the drafter runs a
seven-position block (anchor plus six mask tokens), produces seven proposals,
and the target verifies an eight-token window (anchor plus seven proposals).
Verification width three is retained only as an explicit screening override;
it still runs the full seven-position bidirectional draft backbone and projects
only its first two proposal positions. It is not presented as checkpoint-native
behavior.

The proposer, ReplaySSM, and optional stacked context-KV projection are all
local and default off:

```text
MLX_LM_KIMI_K3_DSPARK_PROPOSER=1
MLX_LM_KIMI_K3_REPLAYSSM_SPECULATIVE=1
MLX_LM_KIMI_K3_DSPARK_STACKED_CONTEXT_KV=1
```

Any value other than `0` or `1` fails closed. The stacked projection implements
the SGLang optimization of concatenating all five layers' target K/V weights,
performing one projection, then taking per-layer views. The per-layer path is
the default correctness reference. The existing target transaction guards
continue to require batch one, populated unpadded caches, Metal, no pipeline
parallelism, and verification width 2 through 8.

The loader API is:

```python
drafter = load_kimi_k3_dspark(
    checkpoint_dir,
    target_model,
    verify_weights_sha256=True,
)
proposer = KimiK3DSparkProposer(drafter)  # native gamma 7 / verify width 8
```

The loader reads only `config.json` and `model.safetensors`; it does not import
or execute remote checkpoint Python. The two borrowed target modules remain
excluded from the drafter's MLX parameter tree.

## Exact ReplaySSM contract

The wide verifier stores the raw KDA tuple `(v, raw_k, gk, beta)` instead of a
full FP32 recurrent-state checkpoint after every candidate token. On commit,
only the accepted prefix is folded from the immutable pre-round state through
the same `gated_delta_kernel` used by the verifier.

`gk` is deliberately stored after `compute_g`/`compute_g_safe`. It is the exact
multiplicative decay consumed by the MLX kernel; reconstructing it with a
log/exp round trip would not be bit-identical. `raw_k` is normalized again with
the same division-form RMS normalization used by the wide target. A replay
exception or shape/dtype mismatch rolls back the complete mixed MLA/KDA
transaction.

On a tiny Metal KDA layer, verification width three produced bit-identical
target outputs, short-convolution states, and FP32 recurrent states for
accepted-prefix lengths 1, 2, and 3 relative to the existing full-history path.

## Capacity model

For the production TP2 KDA geometry (69 KDA layers, 48 global heads, head
dimension 128, convolution kernel 4) under the explicit verification-width-3
screening scenario, the logical transaction-history payload is:

| History | Bytes per TP rank | MiB per TP rank |
| --- | ---: | ---: |
| Existing FP32 SSM plus BF16 convolution history | 337,029,120 | 321.416 |
| ReplaySSM raw inputs plus BF16 convolution history | 16,543,440 | 15.777 |
| Difference | 320,485,680 | 305.639 |

That is a 20.37x reduction in logical transaction-history bytes. This model is
not a process peak-memory measurement: MLX views, lazy graph dependencies,
allocator residency, the target activation graph, and the future DSpark
weights are outside its scope.

Reproduce it with:

```bash
python3 benchmarks/kimi_k3_replayssm_capacity.py \
  --target-step-ms 130 --draft-step-ms 10 --replay-step-ms 1 \
  --expected-emitted 2.6
```

The latency flags above are an illustrative break-even scenario, not measured
DSpark-on-MLX timings. They imply 2.397 emitted tokens per 141 ms round—or
1.397 accepted draft tokens plus the mandatory target bonus token—to clear
17 token/s, and 18.44 token/s at 2.6 emitted tokens per round. A live TP2 A/B
must supply target, draft, replay, and accepted-prefix measurements before
making a throughput claim.

The utility can also compare ordinary decode with pre-built width-three and
width-eight tiers using an observed accepted-prefix survival curve. These are
the cumulative probabilities `P(accepted_prefix >= i)`, not independent token
acceptance probabilities:

```bash
python3 benchmarks/kimi_k3_replayssm_capacity.py \
  --ordinary-step-ms 69.893 \
  --acceptance-survival 0.95,0.75,0.35,0.15,0.05,0.02,0.01 \
  --tier 3:132.64:8:0 --tier 8:255.13:8:0
```

This mirrors the cost-aware adaptive-speculation principle without putting a
policy in the serving path prematurely. The real controller must use an EMA
over measured round costs and survival counts, apply hysteresis, and switch
only among pre-validated ordinary, width-three, and width-eight states.

## Projected-context append storage

`KimiK3DSparkContextCache` uses an occupied-prefix capacity buffer instead of
concatenating the complete projected target context on every append. Capacity
doubles while small, then grows by at most 65,536 tokens at a time. Only the
occupied prefix is exposed to attention, so unused capacity cannot change the
attention inputs or offsets. A caller that knows the complete request budget
can allocate once:

```python
context_cache = proposer.make_context_cache(
    capacity_hint=prompt_tokens + max_generated_tokens,
)
```

For one million single-token appends, the default growth schedule produces 24
allocations, a final capacity of 1,048,576, and 7,929,600 copied history-token
positions per layer. Repeated exact-size concatenation would copy
499,999,500,000 positions, 63,054.8 times as many. The capacity hint reduces
intermediate history copies to zero when it covers the request.

The production five-layer BF16 DSpark context stores 20,480 bytes per logical
token across K and V. The bounded-growth schedule therefore keeps unused
capacity below 1.25 GiB across all five layers; at one million tokens its
48,576-token slack is about 0.93 GiB. These are deterministic allocation and
copy-volume calculations, not a claim about end-to-end latency or MLX allocator
peak residency.

## Verification

The cache-focused DSpark model suite passed 15 tests and 3 subtests. It covers
content and offsets before, at, and after capacity growth; capacity-hint
allocation; one-million-token copy/slack bounds; and bit-identical attention
output for single-append versus split-append context.

The complete Kimi K3 test glob on this branch passed 174 tests, skipped 2, and
passed 145 subtests. One existing fused SwitchGLU exact-equality test failed in
both this branch and the untouched `ebf0747` base under the same MLX runtime.

Earlier focused DSpark contract/model, target-cache, and ReplaySSM capacity
tests reported:

```text
40 passed, 20 subtests passed
```

They cover the complete 62-key, 2,249,289,601-element production inventory;
shape/dtype/key failures; no-copy target-module binding; the reference and
stacked context-KV paths; strict feature gates; native gamma 7 versus explicit
width-3 screening; and a synthetic end-to-end Markov proposal. The broader
Kimi K3 plus speculative-generation suite passed 154 tests, with one skip and
119 subtests. The known fused SwitchGLU equality test was excluded; its isolated
failure reproduces on the untouched v6 base and is unrelated to this branch.

## Pinned primary sources

- SGLang commit: `ae848116662ece923501131ce773a4602223237e`
- SGLang KDA ReplaySSM decode source SHA256:
  `bb51f807c932bc19b45eedc2a6dd3d1f6533389b3d788e0b91901b4208d6f331`
- SGLang ReplaySSM fold source SHA256:
  `fb79ad9e12ec1e485cc65e12d6394debee69fb21a14c50e1bb84a809f6a46e28`
- SGLang DSpark model source SHA256:
  `93098e69c05de76ffb72e63ba2ba670a9e9addb65819f76a211f2cbdf5bf884a`
- SGLang DSpark config source SHA256:
  `a4aa2d41bd4144024afadbc720ed712f24ec34b6b95732bd5e43abe3f0d8ea7c`
- `RadixArk/Kimi-K3-DSpark` revision:
  `eb03982e58d4fb79bcfc099e902158f562e2e27b`
- Raw `config.json`: 1,288 bytes, SHA256
  `6aed20890d95cd69cf2ec006d1f30506fbd4f3091d44ca8e8b93e9fc7d50928f`
- Raw `model.safetensors`: 4,498,585,858 bytes, SHA256
  `29df0e8eafb81909f785df55cb352b90d6a1500c609b1d60526c1a62b4d42495`
- Pinned snapshot: 6 files, 4,498,617,103 bytes.

The source model reports an average accepted length around 2.7 on its chat
evaluation, while its public checkpoint card reports higher full-block
acceptance on some workloads. Those CUDA results make width three useful as a
screening experiment; they do not change the released gamma-7 contract or
establish MLX acceptance or speed.

## Remaining end-to-end work

1. Replicate the draft on both target TP ranks and make proposal/acceptance
   decisions deterministic on both ranks.
2. Join target hidden taps, incremental context projection, the native
   width-eight target transaction, DSpark proposals, prefix verification,
   ReplaySSM commit, and target bonus emission in the generation loop behind a
   default-off feature gate.
3. Keep all TP ranks in lockstep for proposal IDs, accepted-prefix length, and
   target-cache resolution; fail closed on divergence.
4. Prove 256+ greedy tokens against baseline, then measure acceptance, target
   step latency, draft latency, ReplaySSM latency, peak memory, and effective
   token/s on the real TP2 checkpoint. Compare native width eight with the
   explicitly labeled width-three screening override.
