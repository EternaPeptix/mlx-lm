# Kimi K3 width-three KDA prework/history fusion

## Status and scope

This is a separate, default-off, offline candidate based on MLX-LM control
`591e11093b55b3b03f7cbc5018cd3b7d47abba4f`.  It starts from the stock BF16
outputs of `qkv_proj` and `f_b_proj(f_a_proj(x))`.  It does **not** alter or
compose with packed projections, the DSpark Q/K-preparation experiment,
asynchronous scheduling, collectives, caches, weights, or the live service.

The candidate replaces the production TP2 rank-local width-three sequence

```text
exact four-tap short-conv + all rollback history rows
-> Q and K RMS normalization/scaling
-> bounded post-exp gk
```

with one Metal dispatch.  Raw K and V, the next convolution state, and all
three rollback checkpoints remain materialized.  `beta = sigmoid(b_logits)`
stays on the literal stock path.

The selector is strict and default-off:

```text
MLX_LM_KIMI_K3_W3_PREWORK_HISTORY=1
```

It admits only inference on Metal GPU with B=1, T=3, 48 rank-local heads,
D=128, a four-tap convolution, BF16 activations/state/weights, FP32 `A_log`
and `dt_bias`, the exact stock D=128 scale, `gate_lower_bound=-5`, the exact
`KimiK3ShortConv`/inner `Conv1d` types, populated speculative state, no mask or
lengths, and the existing ReplaySSM transaction.  Everything rejected by this
pre-projection gate uses stock.  If projected output contracts somehow drift
after admission, the explicit experiment fails closed instead of silently
falling back after reordering work.

## Predeclared exactness gate

Timing is forbidden until all of the following compare raw BF16 or FP32 bit
patterns against the stock control, including normal, small-magnitude,
saturating, and all-zero finite seeds:

1. normalized Q and K;
2. raw K and V;
3. post-exp FP32 `gk` and stock `beta`;
4. next BF16 convolution state;
5. every BF16 convolution history row at positions 0, 1, and 2;
6. final attention target output and final FP32 SSM state;
7. ReplaySSM cancellation/commit results for accepted prefixes 0, 1, 2, and 3;
8. an integration assertion that the production selector actually dispatched;
9. default-off, training, dtype, width, batch, mask, lengths, scale, exact
   convolution type, post-admission fail-closed, and non-transaction fallback
   coverage.

Any mismatch rejects the candidate without timing.  Arithmetic may not be
changed to approximate the stock path.  Beta remains stock even though it
costs a separate expression.

## Frozen timing protocol

The only decision-bearing timing attempt uses the production W3 Metal seam:

- B=1, T=3, H=48, D=128, KS=4;
- 69 dependent convolution-state layers per chain;
- sealed, evaluated random inputs with SHA-256 identities in the ledger;
- five warmup chains per arm;
- 21 paired trials in alternating AB/BA order;
- graph construction outside the timed interval;
- one `mx.eval` synchronization per complete 69-layer chain;
- raw stock samples, candidate samples, paired deltas, order, and win count;
- the **median paired delta** is the predeclared estimator.  Arm medians are
  descriptive and may not replace it.

The benchmark appends an `attempt_started` event before work and a completion
or exactness-failure event afterward.  Attempt IDs cannot be reused.  No run
may be discarded or selected because it looks better.

The request projection is fixed before timing:

```text
request saving = median paired chain delta * 45 full W3 calls
baseline       = 5684.585742 ms / 128 output tokens
promotion gate = strictly greater than 0.5% projected request throughput
```

If the projection is at or below 0.5%, the artifact is rejected and receives
zero frontier credit.  If it clears the offline gate, it remains only a
candidate: this task authorizes no host access, live service, or canary.

## Evidence boundary

This is a post-projection synthetic seam, not a checkpoint or full target
trajectory.  It excludes QKV/a/b projections, gated-delta recurrence, output
gating/projection, MoE, collectives, scheduler overlap, and cache-transaction
coordination.  Even a positive result is an optimistic mechanical projection;
it cannot be added blindly to packed-front or asynchronous ceilings.

After the exact candidate commit is frozen, run exactly:

```bash
PYTHONPATH=. python benchmarks/kimi_k3_w3_prework_history.py \
  --ledger benchmarks/results/kimi-k3-w3-prework-history-ledger.jsonl \
  --attempt-id attempt-001 \
  --source-parent 591e11093b55b3b03f7cbc5018cd3b7d47abba4f \
  --candidate-commit CANDIDATE_COMMIT \
  --warmups 5 --trials 21 --layers 69
```
