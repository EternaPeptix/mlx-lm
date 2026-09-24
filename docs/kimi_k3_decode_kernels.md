# Kimi K3 decode kernels for Apple silicon

Metal kernels and adapters used to serve Kimi K3 decode on two M3 Ultra Mac
Studios (512 GB each) with two-way tensor parallelism over Thunderbolt RDMA
(JACCL). They target the rank-local TP2 layout of a mixed 2/4/6/8-bit affine
checkpoint with top-8 routed experts.

Every kernel here is exact: on the tested shapes it reproduces the output of
the MLX path it replaces bit for bit. The end-to-end check is a byte-identical
128-token greedy completion on a fixed 4,294-token prompt.

## Measured result

Single-stream decode, 4,294-token prompt, 128 generated tokens, greedy:

| Configuration | Decode time | tok/s |
|---|---:|---:|
| Before (fused experts, router, QK-RMS, wide KDA packing) | 8.82 s | 14.50 |
| Plus the flag set below | 8.44 s | 15.17–15.28 |

The two configurations produce the same completion bytes.

Flag set used for the second row:

```
MLX_LM_KIMI_K3_FUSED_EXPERTS=1
MLX_LM_KIMI_K3_FUSED_EXPERT_WIDTH3=1
MLX_LM_KIMI_K3_FUSED_ROUTER=1
MLX_LM_KIMI_K3_FUSED_QK_RMS_DECODE=1
MLX_LM_KIMI_K3_PACKED_KDA_WIDE=1
MLX_LM_KIMI_K3_PACKED_KDA_SKINNY=1
MLX_LM_KIMI_K3_FUSED_ATTNRES_RMS=1
MLX_LM_KIMI_K3_DERIVE_AFFINE2_BIAS=1
MLX_LM_KIMI_K3_ELIDE_AFFINE2_BIAS=1
MLX_LM_KIMI_K3_EXPERT_TOP_K=8
MLX_MAX_OPS_PER_BUFFER=400
MLX_MAX_MB_PER_BUFFER=4000
```

`MLX_LM_KIMI_K3_EXPERT_TOP_K=8` is lossy: it routes each token to 8 of the
checkpoint's native 16 experts. Everything else is exact.

## Modules

All modules live in `mlx_lm/models/`. Each exposes a `maybe_*` or `supports_*`
entry point that returns `None` (or `False`) when its preconditions are not met,
so the caller falls back to the stock MLX path.

| Module | What it does | Entry points |
|---|---|---|
| `kimi_k3_fused_switch_glu.py` | Routed-expert gate and up gather-QMVs plus SiTU activation in one dispatch (affine 2-bit, group 128). | `supports_fused_switch_situ`, `fused_switch_situ_decode` |
| `kimi_k3_tuned_gather_qmv.py` | Tuned per-expert down-projection gather-QMV for one token. | `supports_tuned_gather_qmv`, `tuned_gather_qmv` |
| `kimi_k3_fused_expert.py` | Adapter that routes decode and width-2/3/4 verification through the kernels above. | `maybe_fused_k3_switch_glu`, `maybe_fused_k3_switch_glu_reduce` |
| `kimi_k3_width4_fused_expert.py` | Width-four verification variants of the expert kernels. | `width4_switch_glu`, `width4_switch_glu_reduce`, … |
| `kimi_k3_fused_down_reduce.py` | Down-QMV fused with router weighting and expert reduction. | `fused_down_reduce_decode` |
| `kimi_k3_derived_bias.py` | Validates that affine-2 biases equal `-2 * scale` so kernels can derive them instead of reading them. | `derive_affine2_bias_enabled`, `validate_k3_biases_for_load`, … |
| `kimi_k3_fused_router.py` | Exact top-k expert selection with score correction and renormalization. | `maybe_fused_k3_router` |
| `kimi_k3_fused_qk_rms.py` | One-token KDA Q/K RMS normalization in one kernel. | `maybe_fused_kda_qk_rms` |
| `kimi_k3_attnres_rms.py` | Attention-residual mix fused with the following RMSNorm. | `maybe_fused_attnres_rms` |
| `kimi_k3_packed_kda_projections.py` | Packs the KDA projections that share an input (the wide q/k/v/gate set and the skinny f/b/g set) into single quantized matmuls. | `maybe_authoritative_packed_k3_kda_wide`, `maybe_authoritative_packed_k3_kda_skinny` |
| `kimi_k3_gated_delta.py` | Gated delta-rule recurrence with a row-tiled prefill kernel (four value rows per SIMD-group). Published under a new name so upstream `gated_delta.py` is unchanged. | `gated_delta_update`, `experimental_kda_row_prefill_kernel` |
| `kimi_k3_splitk_routed.py` | Experimental split-K versions of the routed gate/up and down kernels (see below). Not wired in. | `splitk_switch_situ_decode`, `splitk_gather_qmv_down` |

The call sites are in a patched `kimi_k3.py` that is not part of this branch.
Upstream `kimi_k3.py` can adopt them one at a time: for example,
`KimiK3SparseMoE` calls `maybe_fused_k3_router` and `maybe_fused_k3_switch_glu`
before its stock routing, and `KimiK3DeltaAttention._decode_core` calls
`maybe_authoritative_packed_k3_kda_wide` before `qkv_proj`.

## Requirements

- An Apple silicon GPU. The tested geometry is K3 TP2: hidden size 7168, MoE
  latent width 3584, rank-local expert intermediate width 1536, 48 KDA heads
  per rank.
- `MLX_LM_KIMI_K3_ELIDE_AFFINE2_BIAS=1` needs an MLX build whose Metal
  `gather_qmm` supports `mode="affine2"`. With stock MLX 0.32.2, set it to `0`.
  Decode speed was the same within noise; completion bytes differ from the
  custom build from about the fourth token because of quantization-path
  differences.

## Split-K routed kernels

`kimi_k3_splitk_routed.py` assigns one SIMD-group to each 512-wide input block
rather than walking the whole row with one SIMD-group. Each lane's block partials
are staged in threadgroup memory and summed in the original block order before
the SIMD reduction, so results match the row-walking kernels bit for bit.

In isolation (top-8, 92 layers, M3 Ultra) the gate/up plus down chain drops
from 9.85 ms to 9.00 ms per token. In the full model the gain did not show
(8.46 s versus 8.45 s per 128 tokens), so it is published for reference only.

## What limits decode on this hardware

Measurements from the same machines, for anyone continuing this work:

- GPU streaming reads top out at about 735 GB/s, about 90% of the M3 Ultra's
  819 GB/s interface peak. CPU and GPU together reach about 710 GB/s.
- Each rank reads about 31 GB of weights per token, which puts a floor of about
  42 ms per token. Decode currently takes about 66 ms.
- The large quantized matmuls already run within about 5% of the read ceiling
  for a dependent 143 MB dispatch (212 µs pure read versus 224 µs matmul).
- The remaining time is spread across roughly 15 dependent kernels per layer.
  Tested and ruled out: prefetching into the system-level cache from a second
  queue or a concurrent encoder (Metal serializes both), and a persistent
  single-dispatch kernel with atomic grid barriers (slower and not coherent
  without `coherent(device)` memory).

## Second optimization pass

A later pass shaved load and synchronization overhead inside the same kernels.
All changes keep the exact arithmetic of the first version: every expression
keeps its operand types and its association order, so the outputs stay
bit-identical by construction.

- Packed-weight reads load one `uint` per thread instead of four `uint8_t`
  reads; the same four bytes are extracted with shifts (`fused_switch_glu`,
  `tuned_gather_qmv`, `fused_down_reduce`, `width4_fused_expert`).
- Sixteen-element activation reads use four `ushort4` loads (same files), and
  the KDA kernels use `float4`/`ushort4` for state and per-step q/k/g
  (`gated_delta`, `fused_qk_rms`, `attnres_rms`). Packed loads stay 8-byte
  aligned because every activation and weight pointer here is.
- Each thread's top-k candidates are sorted once up front, so the router's
  selection loop needs one barrier per round instead of three (`fused_router`,
  16 rounds).
- The per-row epilogues run on parallel lanes: lane `r` writes row `r`
  instead of lane 0 looping `RESULTS` rows serially (`fused_switch_glu`,
  `width4_fused_expert`, `fused_down_reduce`, `tuned_gather_qmv`).
- `attnres_rms` packs its `mixed` buffer as `uint2`, drops two barriers
  (redundant zero-fill and the shared `rms_inv` broadcast; every SIMD group
  folds the 32 partials itself), and vectorizes the elementwise phases.
  The reduction that feeds it keeps its strided order.
- `gated_delta` caches each step's q/k/g in registers so the two inner passes
  share one load set.
- Softmax and reduction folds that one thread used to run serially now run
  identically on every thread, so their shared arrays and broadcast barriers
  are gone (`attnres_rms`, `attnres_mix` in `kimi_k3.py`, and the router's
  denominator; the router now has 18 barriers, `attnres_rms` three).
- `attnres_mix`'s output store uses the same `uint2` packing, and
  `gated_delta`'s optional state-history store uses `float4`.

Re-run the byte-identical 128-token greedy check and the per-token timing on
the M3 pair before adopting; this pass was verified for construction, not on
hardware.

## Third optimization pass

- The router's shared score tables are gone: the presort evaluates
  `sigmoid(gate) + bias` in registers, every thread records each round's
  winner score in a private `my_scores[TOP_K]` (the deterministic sigmoid of
  the same winner), and the denominator is folded locally.  The kernel now
  has exactly one barrier per extraction round — 16 total (`fused_router`).
- Sixteen-element activation reads now use two `uint4` loads instead of four
  `ushort4` loads; the same sixteen bytes are unpacked with shifts
  (`fused_switch_glu`, `tuned_gather_qmv`, `fused_down_reduce`,
  `width4_fused_expert`).  This assumes 16-byte-aligned activation pointers,
  which every caller still supplies.
- `attnres_rms` no longer stages its `mixed` values through a 14 KiB
  threadgroup buffer: the fill, reduction, and output loops share the same
  element-to-lane map, so each thread keeps its rounded BF16 values in a
  small register array.  That removes one more barrier — two remain
  (`mix_sums`, `rms_sums`).

- The QMV inner loops now issue a whole block's loads before the dot
  products: every row's 32-bit weight word plus its scale and bias read in
  one phase, the L2-resident activation reads next, and the FMA chain last
  (`fused_switch_glu`, `tuned_gather_qmv`, `fused_down_reduce`, all three
  `width4` sources).  Load scheduling only; the arithmetic is untouched.
- The per-row `simd_sum` epilogues now fold a `float4` (with a `float2`
  remainder) per reduction.  A vector `simd_sum` reduces each component
  through the same butterfly tree as a scalar call, so the per-row
  associations are unchanged while the shuffle count drops ~4x
  (same files as above).  The same treatment covers the per-accumulator
  `simd_sum` loops in `attnres_rms` and `kimi_k3.py`'s `attnres_mix`.
- `ensure_row_contiguous=True` is now set on `attnres_rms`, `fused_qk_rms`,
  and `attnres_mix` for consistency with the other kernels — a no-op for
  the contiguous inputs they always receive.
- The QMV k-loops double-buffer their per-block weight/scale/bias words in
  registers: block k+1's DRAM loads are issued before block k's FMAs run, so
  the DRAM stream never drains between blocks (all six QMV loops).  Loads
  are pure; the arithmetic is unchanged.
- Router fix: the replayed block-level `shuffle_down` butterfly completes
  on lane 0 only, so the winner is now `simd_broadcast_first` before every
  lane uses it (the shared `global_winner` had provided that broadcast
  before the barrier removal).  The same butterfly now starts at
  `SIMDGROUPS / 2` — only that many lanes hold candidates and the comparison
  is a strict total order, so the winner is unchanged.  Kernel name bumped
  to `_v6`.

Same caveat: verified for construction on a Linux box; re-run the
byte-identical check and timing on the M3 pair.
