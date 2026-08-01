# Kimi K3 DSpark and exact ReplaySSM prototype

Date: 2026-08-01

Status: local prototype only. No model weights were downloaded and no EXO or
Mac inference process was changed.

## Scope

This branch adds the smallest target-side primitives needed to evaluate the
public `RadixArk/Kimi-K3-DSpark` checkpoint at an initial verification width of
three:

- post-layer target hidden-state taps at layers `7, 23, 51, 67, 83`;
- a strict contract for the released 2,249,289,601-parameter BF16 drafter;
- replicated-drafter placement as the only accepted TP2 placement; and
- an opt-in exact ReplaySSM rollback path for Kimi K3 KDA state.

It does not yet implement the DSpark model, weight loader, proposer, sampler,
or EXO rank coordination. The existing full-state-history rollback remains the
default. ReplaySSM is enabled only with:

```text
MLX_LM_KIMI_K3_REPLAYSSM_SPECULATIVE=1
```

Any value other than `0` or `1` fails closed. The existing transaction guards
continue to require batch one, populated unpadded caches, Metal, no pipeline
parallelism, and verification width 2 through 8.

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
dimension 128, convolution kernel 4, verification width 3), the logical
transaction-history payload is:

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
  --expected-accepted 2.6
```

The latency flags above are an illustrative break-even scenario, not measured
DSpark-on-MLX timings. They imply 2.397 accepted tokens per 141 ms round to
clear 17 token/s, and 18.44 token/s at 2.6 accepted tokens per round. A live
TP2 A/B must supply target, draft, replay, and acceptance measurements before
making a throughput claim.

## Verification

Focused tests:

```text
30 passed, 17 subtests passed
```

The broader Kimi K3 plus speculative-generation suite passed 145 tests (with
one skip) and had one unrelated fused SwitchGLU equality failure. The same
isolated failure reproduces on the untouched v6 base, so it is not introduced
by this branch.

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
- `RadixArk/Kimi-K3-DSpark` config SHA256:
  `410dd228c75ff91b57af8a1581d44d2ea096d5604f0d37fbd400470e90d961d3`

The source model reports an average accepted length around 2.7 on its chat
evaluation, while its public checkpoint card reports higher full-block
acceptance on some workloads. Those CUDA results motivate width three; they do
not establish MLX acceptance or speed.

## Remaining end-to-end work

1. Port the five-layer GQA DSpark architecture and its exact tensor-name loader.
2. Bind target embedding and vocabulary head, because the checkpoint owns
   neither.
3. Replicate the draft on both target TP ranks and make proposal/acceptance
   decisions deterministic on both ranks.
4. Join the hidden taps, width-three target transaction, and DSpark proposer to
   the generation loop behind one default-off feature gate.
5. Prove 256+ greedy tokens against baseline, then measure acceptance, target
   step latency, draft latency, ReplaySSM latency, peak memory, and effective
   token/s on the real TP2 checkpoint.
