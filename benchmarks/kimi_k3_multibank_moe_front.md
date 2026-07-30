# Experimental no-copy Kimi K3 multi-bank QMV

This branch contains an opt-in, decode-only Metal prototype for the four
affine-8/group-64 projections that consume the same Kimi K3 sparse-MoE input:

1. shared-expert gate;
2. shared-expert up;
3. router scores; and
4. routed-expert latent down.

Unlike `kimi_k3_packed_moe_front.py`, the prototype does not concatenate or
retain a copy of the quantized banks. One Metal dispatch binds the four
authoritative weight/scale/bias arrays, writes one output allocation, and
returns row views for the original projections.

Enable it before importing MLX-LM:

```bash
export MLX_LM_KIMI_K3_MULTIBANK_MOE_FRONT=1
```

The path is disabled by default and fails closed to stock MLX-LM for training,
multi-token inputs, non-BF16 arrays, non-affine quantization, layouts other
than 8-bit/group-64, or shapes that would not select MLX's native QMV-fast
kernel.

## Exactness contract

The custom Metal source reproduces the native affine-8 QMV-fast arithmetic for
each output row independently:

- eight BF16 input values are loaded per SIMD lane;
- the K loop advances by 256 values;
- each quantized dot product updates its own FP32 accumulator in the same
  order;
- `simd_sum` performs the same lane reduction; and
- each projection is cast back to BF16 before any output bias is added.

No accumulation crosses a projection boundary. Focused Metal tests cover
heterogeneous bank widths, output biases, complete sparse-MoE output,
compiled graphs with dynamic bank inputs, unsupported-shape fallback,
parameter identity, and default-off behavior. The new tests and the existing
packed-front suite pass together (`14 passed`). The complete focused Kimi K3
suite passed with `58 passed`, `1 skipped`, and `38 subtests passed`.

## M3 Ultra microbenchmark

The included benchmark uses the TP2 rank-local K3 geometry without loading
checkpoint data:

- input: `1 × 1 × 7168` BF16;
- output banks: `3072 + 3072 + 896 + 3584`;
- affine 8-bit weights with group size 64; and
- 40 warmups, 11 alternating trials, and 500 iterations per trial.

Run:

```bash
python benchmarks/kimi_k3_multibank_moe_front.py \
  --warmup 40 \
  --iterations 500 \
  --trials 11
```

All stock, packed, and no-copy outputs were bit identical.

| Path | Median per sparse layer | Relative to stock |
| --- | ---: | ---: |
| Four native QMV calls | 0.533232 ms | 1.000× |
| Packed native QMV | 0.522083 ms | 1.021× |
| No-copy multi-bank Metal | 0.552222 ms | 0.966× |

The no-copy prototype was 0.018990 ms slower per sparse layer in this run.
Applying that delta mechanically across 92 sparse layers would regress a token
by approximately 1.75 ms. It must therefore remain opt-in.

## Memory evidence

| State | Active allocator bytes |
| --- | ---: |
| Authoritative source banks resident | 80,984,144 |
| After no-copy output evaluation | 81,008,916 |
| After constructing/evaluating packed copy | 161,978,644 |

The authoritative quantized banks contain 80,912,384 bytes. The no-copy path
retains zero duplicate weight bytes; its measured active-memory increase was
24,772 bytes. The packed path retains another 80,912,384 bytes per sparse
layer, approximately 7.44 GB across K3's 92 sparse layers on each TP2 rank.

## Integration blocker

`mx.fast.metal_kernel` exposes each bank as three independent buffer bindings.
The generated kernel must select among 12 bank buffers at each output tile.
That buffer pressure and per-tile bank branch erased the launch savings on
this machine.

Recovering the packed path's speed without duplicate weights likely requires
one of:

- an MLX core primitive with a native heterogeneous-bank argument table;
- Metal argument-buffer/bindless support for the bank descriptors; or
- a checkpoint-native packed layout whose normal projections are views into
  one authoritative allocation.

The last option is the lowest-risk performance route because it reuses MLX's
already-fast, bit-exact native QMV while eliminating the retained second copy.
