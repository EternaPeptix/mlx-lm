# Kimi K3 KDA prefill prototype

This branch contains a default-off, exact Metal specialization for Kimi K3's
post-projection KDA recurrence. It is deliberately narrower than the generic
gated-delta kernel and is not enabled in normal MLX-LM runs.

## Source audit

The implementation was informed by three pinned upstream designs:

- MoonshotAI FlashKDA commit
  `1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b` splits chunk-16 KDA into a
  token/chunk-parallel K1 and a head-parallel K2. Its speed depends on
  tensor-core matrix operations, BF16-resident state, FP32 accumulation,
  aggressive scratch reuse, and the fixed 16-token inverse.
- llama.cpp draft PR
  [#26001](https://github.com/ggml-org/llama.cpp/pull/26001), head
  `98da4c0beb0bf47c48370252eaccedb8272ae58d`, is a three-stage chunk-16
  CUDA prefill implementation. Despite its title, its dispatch explicitly
  excludes KDA (`!kda`), so it does not handle Kimi K3's channel-wise gate.
- llama.cpp PR
  [#22587](https://github.com/ggml-org/llama.cpp/pull/22587), commit
  `b324c33d6ec09c6569aafd792ba0236375b97878`, identifies a useful exact
  alternative: one warp owns four recurrent-state value rows and reuses each
  q/k/g/beta load across them.

An isolated FlashKDA MLX/Metal port validated the chunk equations, but its
scalar K2 was slower than MLX-LM's current recurrent Metal kernel. At
`B=1,T=512,H=96,Dk=Dv=128`, it measured 6.95 ms versus 2.91 ms for the
current kernel and needed 66.84 MiB of bounded scratch. Larger Dv tiles did
not close the gap. Integrating that path would therefore regress prefill.

## Prototype

`experimental_kda_row_prefill_kernel()` maps four value rows to each
SIMD-group. Compared with the current one-row mapping, q, k, vector gate, and
beta are loaded once per token and reused four times. The recurrence order,
FP32 public state, reductions, and BF16 output are unchanged.

The production call site remains unchanged unless this environment variable
is truthy:

```text
MLX_LM_EXPERIMENTAL_KDA_ROW_PREFILL=1
```

The experimental dispatch fails closed unless all of these conditions hold:

- Metal GPU inference;
- vector gate (`g.ndim == 4`);
- no mask and no requested state history;
- at least 128 tokens;
- `Dk=Dv=128`, aligned q/k and value head counts;
- FP32 recurrent state.

Decode (`T=1`), training, speculative state-history capture, masked batches,
and unsupported shapes stay on the existing path.

## Correctness

The row-tiled kernel is bit-exact with the current recurrent Metal kernel for
the tested stable model-like inputs:

- row tiles 1, 2, 4, and 8;
- `T=128`, `Hk=2`, `Hv=4`, `Dk=Dv=128`;
- BF16 q/k/v and beta, FP32 vector gate and state;
- both direct kernel invocation and the environment-gated
  `gated_delta_update()` integration.

Output maximum absolute error: `0.0`.

Final-state maximum absolute error: `0.0`.

Unlike chunk reassociation, this path is not approximate.

## KDA-only microbenchmark

Environment:

- Apple M3 Max, 64 GiB;
- MLX `0.32.0.dev20260730+cc3f3e60`;
- `B=1,H=96,Dk=Dv=128`;
- BF16 inputs and FP32 recurrent state;
- four warmups and 15 synchronized samples through 2K, then three warmups
  and 11 samples at 8K.

These numbers measure only the KDA recurrence, not projections, MoE,
distributed transport, tokenization, or sampling.

| Prompt | Current recurrent | Row-4 | KDA speedup |
|---:|---:|---:|---:|
| 128 | 0.920 ms | 0.644 ms | 1.43x |
| 512 | 3.255 ms | 1.912 ms | 1.70x |
| 2,048 | 15.997 ms | 7.515 ms | 2.13x |
| 8,192 | 115.852 ms | 36.046 ms | 3.21x |

Row-8 lost to row-4 at every measured context, consistent with register
pressure reducing occupancy. The gain increasing through 8K argues against a
short-prompt-only tuning result.

Run the benchmark with:

```bash
python benchmarks/kimi_k3_kda_prefill.py \
  --tokens 128 512 2048 8192 \
  --heads 96 --rows 2 4 8 --warmup 4 --repeats 15
```

## Workspace scaling

All figures below exclude the required output tensor and model inputs.

| Kernel design, H=96 | Extra scratch | 128K | 1M |
|---|---:|---:|---:|
| Current recurrent Metal | none | 0 | 0 |
| Exact row-4 prototype | none | 0 | 0 |
| Bounded FlashKDA, 512-token window | 66.84 MiB | 66.84 MiB | 66.84 MiB |
| Unwindowed FlashKDA model | prompt-linear | 16.71 GiB | 133.69 GiB |
| llama.cpp PR #26001 GDN buffers | prompt-linear | 12.80 GiB | 102.38 GiB |

The FP32 recurrent state itself is constant at 6 MiB for `H=96,D=128`.
The BF16 output is inherently 24 KiB per token: 3 GiB at 128K and 24 GiB at
1M if the entire layer output remains materialized. The row-4 kernel adds
nothing on top of those required tensors.

## Integration surface

The next isolated A/B surface is a Kimi K3 prefill run from this branch with
the environment flag enabled on both Mac ranks. Record per-layer KDA timing,
whole-model prompt tokens/s, peak memory, and logits against the flag-off
control at 2K, 8K, and at least one larger prompt. The end-to-end gain will
depend on the fraction of prefill time spent in KDA; this microbenchmark alone
does not establish model-level speedup.

Do not replace the recurrent decode kernel with this path. If the live A/B is
positive, the next code step is a Kimi-specific named configuration switch
and a performance regression test, not making the experimental environment
flag the global default.
