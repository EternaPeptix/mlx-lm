# Kimi K3 decode-time derived affine bias

## Candidate

The K3 UVMAX routed expert banks use affine 2-bit weights with group size 128.
Their stored BF16 metadata obeys `bias == -2 * scale` bit for bit.  The opt-in
candidate sets:

```text
MLX_LM_KIMI_K3_DERIVE_AFFINE2_BIAS=1
```

It removes the bias metadata read from the fused gate/up, tuned down, and
fused down/reduce Metal kernels.  The flag is exact `0`/`1` and defaults to
`0`.  Bias tensors remain loaded so prefill, stock, and unsupported fallbacks
are unchanged.

Before enabling the kernel arm, model sanitization performs a complete bitwise
check of every routed gate/up/down scale and bias array.  It also requires
finite-normal BF16 scales.  Zero, signed zero, subnormal, infinity, NaN, a
missing metadata tensor, or one mismatched bias bit fails the requested load.
An alternate loader that bypasses sanitization is guarded again at runtime and
falls back to the incumbent path.

## Exactness tests

The tests cover:

- strict default-off environment parsing;
- exact BF16 derivation semantics for signed zero, signed minimum subnormals,
  infinities, and normal values;
- fail-closed model-load validation for those values when the fast Metal
  arithmetic cannot preserve its uniform contract;
- finite-normal BF16 limits in the gate/up and tuned-down kernels;
- the accepted real-shape TP2 gate/up plus fused down/reduce chain;
- invalid-metadata runtime fallback; and
- strict Boolean kernel template arguments.

All 32 focused tests passed against MLX
`2cfb83040011c273377a25df8ed16def80c6646c` and MLX-LM
`bf378e33831e745715a88418a44ce20ab1075b9b`.
The complete `test_kimi_k3*.py` family also passed: 213 tests with two
expected skips.

The accepted-stack integration is based on MLX-LM
`95fc8ad485e8d2568eda4e468c4169f6a556919a`.  Against the same exact MLX
runtime, that branch's complete `test_kimi_k3*.py` family passed 132 tests
with one expected skip, and the five focused derived-bias/expert modules
passed all 28 tests.  The two integration conflicts retained the accepted
one-token gate/up geometry and its existing expert tests; no width-two path
was introduced.

## Paired M3 Max screen

`kimi_k3_derived_bias.py` uses the real TP2 dimensions (`3584 -> 1536 ->
3584`), top-16 routing, four distinct 66 MB expert banks, eight operations per
sample, 31 balanced paired trials, and exact raw tensors on both arms.

| Repeat | Full-chain incumbent | Full-chain derived | Median ratio | Paired geometric mean | Paired wins |
|---|---:|---:|---:|---:|---:|
| 1 | 0.360733 ms/layer | 0.349395 ms/layer | 1.0325x | 1.0263x | 18/31 |
| 2 | 0.359820 ms/layer | 0.342983 ms/layer | 1.0491x | 1.0440x | 21/31 |

Every fused-front, tuned-down, fused-down/reduce, and complete-chain output was
bit exact in both runs.  The complete accepted chain remained positive in the
independent repeat, so the candidate passes the local retain/reject gate.  The
result is directional; the required next step is a whole-instance TP2 A/B on
the M3 Ultra pair.

Artifacts:

- `kimi_k3_derived_bias_paired_m3max.json`
- `kimi_k3_derived_bias_paired_m3max_r2.json`

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
