# Experimental exact Kimi K3 MoE-front packing

This branch contains an opt-in, decode-only path that concatenates the
already-quantized output rows of four projections consuming the same sparse-MoE
input:

1. shared-expert gate;
2. shared-expert up;
3. router scores; and
4. routed-expert latent down.

One `quantized_matmul` replaces four independent calls. The implementation
does not dequantize or requantize weights, is disabled by default, and uses the
stock path for prefill, training, batches, unsupported quantization layouts,
and models without K3's shared/latent expert geometry.

Enable the experiment before importing MLX-LM:

```bash
export MLX_LM_KIMI_K3_PACKED_MOE_FRONT=1
```

## Correctness evidence

- Five focused Metal tests pass, including bit-exact packed rows, bit-exact
  complete sparse-MoE output, unchanged multi-token routing, and fail-closed
  unsupported quantization.
- The related exact expert-kernel suite still passes (14/14 combined tests).
- A read-only real-weight benchmark on TP2 rank 0, layer 1, measured all four
  outputs as bit-identical at the one-token decode shape.
- Multi-token calls deliberately remain on the original projections because
  wider QMM shapes can select a different BF16 accumulation tiling.

An end-to-end TP2 completion-hash A/B remains required before enabling this by
default.

## Real-weight timing

The real rank-local benchmark used 11 alternating trials with 500 iterations
per trial on an M3 Ultra:

| Shape | Four QMM median | Packed QMM median | Speedup | Saved |
| --- | ---: | ---: | ---: | ---: |
| `1 × 1 × 7168` | 0.3103 ms | 0.2528 ms | 1.228× | 0.0576 ms |

Kimi K3 has 92 sparse-MoE layers. Applying the isolated saving linearly gives
an optimistic `5.30 ms/token` estimate. Against the clean `82.61 ms/token`
TP2 baseline, that projects from about `12.10` to `12.93 tok/s`. This is a
projection, not an end-to-end throughput claim.

## Memory tradeoff

The prototype retains the original projections for unchanged prefill and
materializes one packed copy lazily per sparse layer. Real rank-local storage
is `80,912,384` bytes per layer, or approximately `7.44 GB` per TP2 rank.

A production version can avoid that duplication with a prepacked checkpoint
layout whose stock multi-token projections use row views into the same packed
storage. That lower-memory load contract is intentionally outside this
fail-closed experiment.
