# Experimental Kimi K3 AttnRes + RMSNorm fusion

This branch adds a default-off, inference-only Metal fusion for the two
AttnRes→RMSNorm boundaries in each Kimi K3 decoder layer. The stock path first
materializes a BF16 residual mixture and then launches MLX's RMSNorm kernel.
The fused path keeps the same BF16 materialization boundary in threadgroup
memory and performs MLX's looped RMS reduction in the same dispatch.

Enable it with:

```bash
export MLX_LM_KIMI_K3_FUSED_ATTNRES_RMS=1
```

The adapter fails closed unless every released-checkpoint invariant holds:
Metal GPU inference, one BF16 token, hidden width 7168, FP32 AttnRes reduction
inputs, a BF16 norm weight, and one through eight stored residual blocks.
Training, prefill, batches, other dtypes, other dimensions, and other residual
counts use the unchanged MLX-LM path. No distributed collective or JACCL
ordering changes.

## Exactness

The fused kernel reproduces both native reductions:

- AttnRes keeps the existing 512-thread accumulation and softmax order.
- The mixed value is explicitly rounded to BF16 before its square contributes
  to RMSNorm.
- RMSNorm keeps MLX's four-adjacent-values-per-thread, 1024-thread looped
  reduction, `metal::precise::rsqrt`, output cast, and weight multiply order.

Focused Metal tests are bit-identical for stored-residual counts 1, 2, 4, and
8. The feature/compiled-decode/fused-expert regression subset passes 37 tests
with one expected skip and 53 subtests.

## Synthetic exact-geometry microbenchmark

The checked-in benchmark uses BF16 `[1, 1, 7168]` tensors and the released
`attn_res_block_size=12` schedule:

```bash
PYTHONPATH=. python benchmarks/kimi_k3_attnres_rms.py \
  --warmup 20 --iterations 300 --trials 15
```

Three fresh processes on the local 64 GB M3 Max produced:

| Process | Stock projection | Fused projection | Saving | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 36.645 ms/token | 34.919 ms/token | 1.726 ms | 1.0494x |
| 2 | 36.749 ms/token | 34.678 ms/token | 2.070 ms | 1.0597x |
| 3 | 36.712 ms/token | 34.567 ms/token | 2.145 ms | 1.0620x |

The median isolated projection saves **2.070 ms/token**. The released
93-layer schedule has 185 fuseable boundaries per token per rank: 24 at each
stored-residual count 1 through 7 and 17 at count 8. This removes 185 Metal
dispatches and approximately 5.059 MiB of intermediate global-memory traffic
per token per rank.

Peak and active allocation above resident inputs are unchanged at 0.013672
MiB because MLX already donates the stock intermediate buffer to RMSNorm.
This is a latency/traffic optimization, not a memory-capacity optimization.

If the isolated saving transferred perfectly to the current 13.2985 tok/s
TP2 result, it would imply roughly 13.675 tok/s. That is only an upper-bound
projection: the real async/JACCL schedule can hide some dispatch cost, so a
two-rank completion-hash and throughput A/B is still required before enabling
the flag in deployment.
