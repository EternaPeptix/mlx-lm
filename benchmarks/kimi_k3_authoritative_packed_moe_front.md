# Authoritative packed Kimi K3 MoE front

This experiment keeps the launch reduction of the exact packed Kimi K3
decode path without retaining a second copy of its quantized banks. It is
disabled by default and can be enabled before importing MLX-LM with:

```bash
export MLX_LM_KIMI_K3_AUTHORITATIVE_PACKED_MOE_FRONT=1
```

For each sparse layer, the implementation:

1. validates the four affine-8/group-64 projections;
2. concatenates their packed weights, scales, quantization biases, and
   optional output biases;
3. evaluates those concatenations once;
4. replaces each original module parameter with a row view into the packed
   allocation; and
5. keeps only the full packed parents in a hidden decode object.

The normal module parameter paths, logical shapes, prefill calls, checkpoint
names, and safetensors values remain unchanged. The hidden parents are not
part of `Module.parameters()` or serialized state.

## Correctness and lifecycle

The complete focused Kimi K3 suite passed:

```text
69 passed, 1 skipped, 38 subtests passed
```

The coverage includes:

- bit-exact rows and complete sparse-MoE output;
- multi-token stock prefill after installing the row views;
- default-off and unsupported-layout fallback;
- parameter traversal with unchanged names;
- `Module.update` and post-pack `load_weights` rebuilding the packed parent;
- safetensors save/load with no hidden packed keys;
- byte-identical detachment before post-pack sharding; and
- an allocator assertion that superseded source arrays are not retained.

The normal distributed load order already shards before the first inference.
`KimiK3ForCausalLM.shard()` invalidates packed state before changing the
relevant projections. If sharding is unusually requested after packing, the
invalidation path gives the source views independent byte-identical
allocations before dropping the packed parent.

## TP2 rank-local memory

The benchmark uses the TP2 rank-local K3 front geometry:

- input width: `7168`;
- output rows: `3072 + 3072 + 896 + 3584`;
- affine 8-bit weights, group size `64`; and
- `92` sparse layers.

Each layer's packed arrays contain `80,912,384` bytes.

| Layout | Source resident | Steady after packing | Extra steady bytes | Peak during packing |
| --- | ---: | ---: | ---: | ---: |
| Duplicating packed | 80,936,968 | 161,873,928 | 80,936,960 | 161,873,928 |
| Authoritative packed | 80,936,968 | 80,936,968 | 0 | 161,873,928 |

The small difference between logical bytes and active allocator bytes is
allocator metadata/granularity. Authoritative packing removes approximately
`7.44 GB` decimal (`6.93 GiB`) of persistent duplicate storage per TP2 rank
across 92 sparse layers. Both layouts transiently need one additional layer
allocation while that layer is being packed; the authoritative layout
releases the superseded banks immediately afterward.

Three process-isolated installation samples took `2.73–5.09 ms` per layer
with a `3.30 ms` median, or roughly `0.30 s` if applied serially to all 92
layers. This should ultimately be moved into a post-load/post-shard
preparation hook so the cost is paid before the first request. The current
prototype performs it lazily and synchronously on the first decode.

## Stable synthetic decode timing

Run:

```bash
python benchmarks/kimi_k3_authoritative_packed_moe_front.py \
  --warmup 40 \
  --iterations 500 \
  --trials 11
```

Three independent processes each ran 11 rotating trials of 500 iterations.
The median of their per-process medians was:

| Path | Median per sparse layer | Relative to stock |
| --- | ---: | ---: |
| Four native QMV calls | 0.538352 ms | 1.000× |
| Duplicating packed native QMV | 0.527620 ms | 1.020× |
| Authoritative packed native QMV | 0.526141 ms | 1.023× |

The authoritative and duplicating packed paths invoke the same native MLX
`quantized_matmul`; their `0.28%` difference is measurement noise rather than
an expected algorithmic difference. Both were bit exact against the four-call
stock result in every process.

Applying the authoritative median mechanically across 92 sparse layers saves
about `1.12 ms/token`. Against the prior clean TP2 baseline of
`82.61 ms/token` (`12.10 tok/s`), that projects to approximately
`81.49 ms/token` (`12.27 tok/s`), a `1.38%` improvement. This remains an
isolated-kernel projection; an end-to-end, real-checkpoint TP2 A/B is required
before treating it as a throughput result.

## Loader integration

A production loader should:

1. load and sanitize the ordinary checkpoint parameter names;
2. apply tensor/pipeline sharding;
3. release the loader's checkpoint-weight mapping and evaluate the rank-local
   parameters;
4. install authoritative packed fronts one layer at a time; and
5. discard temporary allocator cache before serving.

Fresh model loading needs no checkpoint conversion. Saving after packing still
emits the ordinary four projection tensors. A later full `load_weights`
replaces those views, and the next decode detects their identities and
rebuilds correctly. A serving stack that hot-loads an already-packed model
should explicitly discard/invalidate its old packed fronts during the load so
the stale parents do not remain resident until the first subsequent decode.
