# Kimi K3 exact direct-wide short convolution

## Status

Local candidate only. Do not promote without the real-checkpoint TP2 gates
listed below.

The ordinary non-transactional KDA call had an arithmetic-path asymmetry:

- `T=1` decode used `k3_short_conv_step`, a recurrent Metal kernel;
- speculative `T=2..8` used `k3_short_conv_history`, which has the same ordered
  recurrence and also writes rollback checkpoints;
- ordinary non-transactional `T=2..8` fell through to the generic depthwise
  `Conv1d` implementation.

The real-checkpoint history-path localizer had already shown that the
projection, history short convolution, gated-delta recurrence, gates, output,
and both final cache states were bit-exact against repeated `T=1` calls. The
generic `Conv1d` fallback was the remaining unmeasured KDA path used by the
full target verifier.

This experiment adds a no-history recurrent writer with the same operation and
state-update order as the two proven recurrent kernels. It is default-off and
guarded by `MLX_LM_KIMI_K3_EXACT_WIDE_SHORT_CONV=1`. A malformed flag raises;
when requested for width `2..8`, unsupported device, dtype, padding, training,
or missing-state contracts raise instead of silently returning to `Conv1d`.
Widths above eight, including prefill, are unchanged.

## Local numerical evidence

On the local Apple GPU, BF16 inputs at the real TP2 rank-local convolution
width (`C=18,432`, kernel size four, `B=1`, `T=2`) produced:

| Path versus two `T=1` calls | Exact output | Different values | Max abs error | Exact final state |
| --- | ---: | ---: | ---: | ---: |
| Stock generic `Conv1d` | no | 10,915 / 36,864 | 0.00006103515625 | yes |
| Exact recurrent candidate | yes | 0 | 0 | yes |

Focused Metal tests also establish bit equality for:

- candidate direct output/state versus the existing history kernel at widths
  two and eight;
- direct non-history gated-delta output/state versus its history kernel and
  repeated `T=1` calls;
- a complete synthetic `KimiK3DeltaAttention` width-two call, including
  projections, Q/K normalization, gated-delta recurrence, gate, output
  projection, convolution cache, and FP32 recurrent cache.
- a complete synthetic decoder-layer width-two call with materialized model
  parameters and post-prefill-style cache state, including input RMSNorm,
  AttnRes block state, sparse MoE, final layer output, and both KDA caches.

This isolates the local numerical mismatch to the stock generic short
convolution; gated-delta's non-history kernel is not a second divergence in the
covered contract.

## Local timing screen

Seven 300-iteration trials at `C=18,432`, after warmup:

| Path | Median time per width-two call |
| --- | ---: |
| Stock generic `Conv1d` | 0.35447 ms |
| Exact recurrent candidate | 0.33773 ms |
| Existing history writer | 0.34776 ms |
| Two Python-issued `T=1` calls | 0.51959 ms |

These are local primitive timings, not end-to-end Kimi K3 throughput claims.
The candidate removes the numerical divergence without showing a primitive
regression in this screen.

## Required real-checkpoint experiment

Run the decoder-layer localizer twice on the accepted TP2 mesh, first with
`K3_DECODER_LAYER_LOCALIZER_EXACT_WIDE_SHORT_CONV=0`, then with it set to `1`
and this MLX-LM tree installed on both ranks. Promotion requires:

1. exact staged reproduction of both real endpoints;
2. the candidate's controlled direct KDA path to become exact, beginning at
   `conv_output` and including output plus both cache states;
3. the independently measured history-producing transaction path to remain
   exact;
4. the complete layer-zero staged path to reproduce the full-model layer-zero
   endpoint exactly;
5. the authoritative target-verification v5 width-two gate to pass before any
   effective-throughput benchmark is accepted.

No real-checkpoint result is claimed by this local experiment.
