# Kimi K3 decode-time derived affine bias

## Candidate

The K3 UVMAX routed expert banks use affine 2-bit weights with group size 128.
Most stored BF16 metadata obeys `bias == -2 * scale` bit for bit.  The opt-in
candidate sets:

```text
MLX_LM_KIMI_K3_DERIVE_AFFINE2_BIAS=1
```

It removes the bias metadata read from the strictly validated fused gate/up
kernel.  It now does the same for each independently validated down bank in
the tuned-down and fused-down/reduce kernels.  A non-exact down bank keeps its
stored bias and incumbent kernel.  The flag is exact `0`/`1` and defaults to
`0`.  Bias tensors remain loaded, so prefill, stock, and unsupported fallbacks
are unchanged.

Before enabling a kernel arm, model sanitization performs a complete bitwise
check of its scale and bias arrays and requires finite-normal BF16 scales.
Gate/up violations remain fatal.  Missing down metadata or an incompatible
projection remains fatal, but a complete non-exact down bank is marked
ineligible rather than failing the model load.  An alternate loader that
bypasses sanitization is guarded again at runtime and caches either the
derived or stored-bias decision on that projection.

## Complete TP2 down-bank audit

`kimi_k3_selective_down_scan.py` read every BF16 scale/bias pair directly from
the safetensors payloads without loading a model or changing either service.
Each rank contains 92 down banks of shape `[896, 3584, 12]`, or 38,535,168
metadata pairs per layer.  The scan read 14,180,941,824 bytes per rank.

| Rank | Exact banks | Non-exact layers (mismatched pairs) |
|---|---:|---|
| TP2 rank 0 | 84/92 | 24 (22), 25 (21), 36 (39), 37 (1), 48 (1), 64 (1), 81 (1), 83 (1) |
| TP2 rank 1 | 86/92 | 24 (23), 25 (10), 36 (15), 38 (1), 67 (1), 89 (1) |

No zero, subnormal, infinity, or NaN scale was found.  The two ranks have 81
layers in their exact-bank intersection (88.0%); the remaining 11 layers are
safe because each rank selects its down kernel independently.

## Exactness tests

The tests cover:

- strict default-off environment parsing;
- exact BF16 derivation semantics for signed zero, signed minimum subnormals,
  infinities, and normal values;
- fail-closed model-load validation for those values when the fast Metal
  arithmetic cannot preserve its uniform contract;
- finite-normal BF16 limits in the gate/up and tuned-down kernels;
- the accepted real-shape TP2 gate/up plus fused down/reduce chain;
- per-module invalid-down fallback while exact gate/up remains fused;
- valid-down selection and real-shape bit exactness; and
- strict Boolean kernel template arguments.

The two directly affected modules passed all 20 tests.  The complete
`test_kimi_k3*.py` family passed 136 tests with one expected skip against MLX
`2cfb83040011c273377a25df8ed16def80c6646c`.  The candidate starts exactly at
retained MLX-LM commit `2606b9c089177270cecc60fb130dc6e39e046d95`.

## Paired M3 Max screen

`kimi_k3_derived_bias.py` uses real TP2 dimensions (`3584 -> 1536 -> 3584`),
top-16 routing, four distinct 66 MB expert banks, 16 operations per sample,
and 63 balanced paired trials on an Apple M3 Max.  The selective-chain control
derives gate/up bias in both arms; only the candidate derives down bias.

| Path | Stored-down median | Derived-down median | Median ratio | Paired geometric mean | Paired wins |
|---|---:|---:|---:|---:|---:|
| Tuned down | 0.183396 ms | 0.180787 ms | 1.0144x | 1.0254x | 41/63 |
| Fused down/reduce | 0.186214 ms | 0.183823 ms | 1.0130x | 1.0160x | 36/63 |
| Selective full chain | 0.340406 ms/layer | 0.336531 ms/layer | 1.0115x | 1.0114x | 37/63 |

Every tested output was bit exact.  The selective full-chain gain is small but
positive, and applies to 81/92 layers on both ranks.  This passes an
experimental retain gate, not a production promotion gate: the projected
cluster-level decode gain is roughly 1%, and a whole-instance M3 Ultra TP2 A/B
is still required before deployment.

## Laguna row-tile audit

Laguna commit `d4cb1ae8d63cd3e59169bc7685d85ca7970241e6` doubles its
fused norm-plus-QKV threadgroup from two to four SIMD groups, taking the tile
from eight to sixteen rows.  That win amortizes one replicated RMSNorm
prologue across more dense INT8/group-32 QKV rows.

The lesson is orthogonal to bias derivation, but the K3 kernels already expose
and have separately swept this launch dimension.  K3's accepted routed
gate/up launch already uses four SIMD groups and the M3 Ultra real-weight
sweep selected two rows per SIMD group (eight rows per threadgroup), while the
fused down/reduce launch uses its independently selected four-row/sixteen-SIMD
geometry.  K3 has no shared RMSNorm prologue in these expert kernels and uses
affine 2-bit/group-128 routed weights, so Laguna's literal sixteen-row launch
is not transferable evidence.  It remains a valid separate M3 Ultra A/B, but
must not be bundled with this candidate because doing so would confound the
derived-bias result.

## Live A/B contract

Both ranks must use the same MLX-LM commit and the same flag value before
process start.  Keep the accepted TP2 feature set fixed and restart the whole
instance between arms:

```text
control:   MLX_LM_KIMI_K3_DERIVE_AFFINE2_BIAS=0
candidate: MLX_LM_KIMI_K3_DERIVE_AFFINE2_BIAS=1
```

Run the canonical 575-prompt-token/128-decode-token benchmark for at least
three repetitions and require the exact completion SHA to match the accepted
control.  The harness must pass this environment variable identically to both
ranks; changing it in a running process is unsupported because selectors and
compiled graphs are cached.
