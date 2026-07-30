# Kimi K3 exact skinny KDA projection packing

This default-off candidate applies SGLang's same-input skinny-projection idea
to the released Kimi K3 checkpoint without changing quantization or prefill.

Enable it with:

```sh
MLX_LM_KIMI_K3_PACKED_KDA_SKINNY=1
```

## Applicable projection topology

K3's `use_full_rank_gate=true` configuration does **not** instantiate
`g_a_proj` or `g_b_proj`.  The applicable local pair is therefore:

```text
f_a_proj: [7168] -> [128]  (replicated)
b_proj:   [7168] -> [48]   (TP2 rank-local)
```

Both use affine 6-bit/group-64 quantization after rank-local loading.  The
candidate concatenates their already-quantized rows as `[f_a | b]`, performs
one QMV, and splits views before `f_b_proj` and KDA consume the results.  For
an older low-rank-gate KDA configuration it instead packs
`[f_a | g_a | b]`.

The pinned SGLang K3 implementation makes the same full-rank choice in
`KimiK3DeltaAttention._merge_bfa_weights()`:

<https://github.com/sgl-project/sglang/blob/578edb240a6d6f6f2fa4c31497276955d7f73432/python/sglang/srt/models/kimi_k3.py>

## Expected cost and ceiling

At the deployed TP2 shape the parent allocation is 1,025,024 bytes per KDA
layer (about 0.98 MiB), or 67.45 MiB across 69 KDA layers per rank.  This is
not additional steady-state weight memory: the original projection parameters
are repointed to zero-copy row views, preserving their names and prefill
shapes while releasing the superseded allocations.  Installation temporarily
duplicates only the layer currently being packed.

The path removes one QMV launch per KDA layer, or 69 launches per generated
token.  A prior real-rank isolated prototype measured the `f_a+b` region at
0.2054 ms/layer separate versus 0.2012 ms/layer packed.  Taken mechanically,
that is about 0.29 ms/token.

A bounded local M3 Max benchmark at the deployed TP2 geometry
(`7168 -> [128, 48]`, followed by the real `f_b: 128 -> 6144` shape) found:

- eager, compiled, and compiled-consumer outputs bit-for-bit equal;
- two compiled-consumer runs at 1,000×9 and 2,000×11 iterations produced
  median paired speedups of 1.0119× and 1.0067×;
- their median-time deltas project to 0.91 and 0.78 ms/token across 69 KDA
  layers;
- a 1,025,024-byte source allocation became a 1,025,024-byte authoritative
  backing with 11,264 bytes of steady allocator overhead and a 1,048,576-byte
  transient installation peak.

The timing samples remain variable and this ceiling is far below the remaining
gap to 17 tok/s, so the canonical full-model live A/B was kept strict.

## Live TP2 result

The zero-copy skinny pack passed the real-weight, two-rank M3 Ultra TP2 gate
on top of the exact fused-down, AttnRes/RMSNorm, and corrected-router stack.
Five canonical 575-prompt-token/128-generation-token repetitions retained
completion digest
`c84d0f0464acc5f0226e5a9686e2bb8ed4b243064dfafb99d7aa7fc5cd5b0c71`
and produced:

```text
14.0846, 13.9339, 14.0258, 14.0268, 14.0272 tok/s
```

The median was `14.0268` tok/s versus `13.8160` for the matched exact-router
stack (`+1.53%`). This saves `1.088 ms/token`.

Three 1,067-token coding-prompt repetitions retained digest
`9936f17d98ac76b2a3ad3ab768e78fae5379259da0b745881f06e7cf9c7a7959`
and produced `13.9964`, `13.9971`, and `14.0106` tok/s. The median was
`13.9971` tok/s versus `13.7809` for the matched exact-router stack
(`+1.57%`). Peak memory remained approximately `414 GB` per rank for the
canonical case.

This accepted path removes one skinny-projection QMV launch in each of the 69
KDA layers. It remains opt-in because its supported quantization and decode
geometry are deliberately narrow.

## Exactness and generalization gates

- default off and fail closed unless every source is affine 6-bit/group-64;
- install only for one-total-token inference while the module is in eval mode;
- require bitwise equality for every packed output at T=1;
- keep T>1 prefill on the original module calls and require bitwise equality
  after authoritative row-view installation;
- invalidate/detach the pack before tensor-parallel resharding;
- require the canonical full-model completion digest before retaining it.

The decode graph is independent of prefix length, so any accepted launch
saving applies at large context as well.  This candidate deliberately does
not claim a prefill improvement.
