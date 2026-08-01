# Kimi K3 TP2 routed-up output-row sharding prototype

This branch adds a strict, default-off ownership change guarded by:

```sh
MLX_LM_KIMI_K3_TP2_ROUTED_UP_COLUMN=1
```

The current TP plan shards the switch/shared expert intermediates but leaves
`routed_expert_up_proj` replicated.  Under the prototype, each of exactly two
ranks owns 3,584 of its 7,168 output rows.  Each rank computes its local
affine8/group64 projection with FP32 accumulation, rounds to BF16, performs
the stock ordered BF16 `routed + shared` and `residual + MoE` operations on
matching slices, then all-gathers a 7,168-element (14,336-byte) BF16 hidden
vector for the next replicated stage.

The checkpoint is named `2bit-UVMAX`, but its authoritative per-module config
assigns **affine8/group64** to `routed_expert_up_proj`.  Affine2 is rejected
rather than silently benchmarking a tensor that is not in this deployment.

The synthetic test constructs the full production geometry, independently
executes both output-row halves, applies a rank-ordered synthetic all-gather,
and requires bit-for-bit equality with the full projection and ordered adds.
Multi-token inputs use the native QMM path with the same ownership and
boundaries; the accepted one-token Metal writer is reused for decode.

The focused routed-up suites pass 12/12 tests.  The full Kimi K3 pattern runs
199 tests: 196 pass, two are distributed/environment skips, and the sole
failure is the pre-existing `fused_switch_glu` `(results=2, simds=1)`
exactness case.  The same isolated test fails unchanged on the untouched
`aa2e11e` parent.

Run the single-device critical-compute screen with:

```sh
python benchmarks/kimi_k3_tp2_routed_up_column.py
```

The script measures full versus half routed-up/add dispatches and reports the
maximum all-gather latency that still breaks even.  Its provisional 53.230 us
value comes from the prior 14,336-byte JACCL **all-sum** p50 and is not proof
of all-gather latency.  Promotion requires a dependency-serialized 14,336-byte
BF16 JACCL all-gather p50/p95 on the two M3 Ultra ranks, followed by the full
TP2 output-digest and throughput gate.  The rank-local checkpoint converter
also still marks this tensor replicated; the runtime safely slices it after
load, while a future converter change must update its ownership manifest and
avoid slicing the already-local tensor twice.

## Offline result

Hardware: Apple M3 Max, 64 GB.  Protocol: eight distinct projection banks
(enough to exceed the cache), 128 dependent warmups, then 15 alternating
trials of 512 dependent projections.  The output half was bit-exact in every
screen and all 15 timing pairs favored the half projection.

| Metric | Full 7,168 rows | Local 3,584 rows |
| --- | ---: | ---: |
| Median routed-up/add time | 83.663 us | 45.568 us |
| Critical compute saved |  | 38.096 us/layer |
| Zero-communication ceiling across 92 layers |  | 3.505 ms/token |
| Break-even all-gather p50 |  | **<38.096 us** |

The existing one-rail JACCL 14,336-byte all-sum p50 is 53.230 us.  Using that
only as a provisional latency proxy makes this proposal **1.392 ms/token
slower**, projecting the 69.893 ms/token baseline from 14.308 to about
14.028 tok/s.  Even an impossible zero-cost gather would project only about
15.063 tok/s.  Consequently this exact gather-after-every-layer design is not
a route to 17 tok/s on its own and should remain off unless the actual
all-gather measurement is unexpectedly below 38.096 us.  A viable successor
would have to retain sharded hidden state across more than one layer, fuse the
new exchange into an existing collective, or overlap it with independent
work; those are different ownership graphs and require separate correctness
proofs.
