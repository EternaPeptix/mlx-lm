# Kimi K3 width-three full authoritative MoE-front composition

Date: 2026-08-16

Status: **default-off current-lineage composition candidate; no cluster,
model-weight, service, quality, memory, or end-to-end performance credit.**

## Source contract

- MLX-LM base: `591e11093b55b3b03f7cbc5018cd3b7d47abba4f`
- Current-lineage port commit: `b3e3d7c64eaf16746c97d2c7cf898158d73cf0d8`
- Native prerequisite: MLX current Q4 receipt parent `b5e6d4896fcc83e1c7933b552a2d3d6f820a21c4`
- Native packed-width implementation: `14a522c819d9ee9a8aa5d3c8298f4d94693aaca3`
- Native freeze receipt: `c75c7470` (`K3_CURRENT_Q3_PACKED_AFFINE8.md`)
- Runtime gates:
  - `MLX_LM_KIMI_K3_AUTHORITATIVE_PACKED_MOE_FRONT=1`
  - `MLX_LM_KIMI_K3_AUTHORITATIVE_PACKED_MOE_FRONT_WIDTH3=1`
  - `MLX_METAL_K3_AFFINE8_Q3_TRIPLET=1`

The width-three selector is stricter than the existing width-one and
width-eight packed paths. It admits only input `(1, 3, 7168)` and four
released TP2 BF16-input affine8/group-64/BF16 projections with output widths
`(3072, 3072, 896, 3584)` (total `10624`). Packed weights must be U32,
affine metadata BF16, and projections must have no post-matmul output bias.
Every other shape or layout stays on the prior path. Current 591 projected-KV
verification widths `(3, 4)`, `EXPERT_TOP_K`, width-four receipt behavior, and
existing W1/W8 front paths are retained.

## Why overlap is preserved

The full pack computes shared gate/up, router scores, and routed latent-down in
one QMM. Routed expert work and shared-expert SiTU/down work still branch from
that result. The model therefore keeps the existing routed/shared MOK overlap
for the exact width-three candidate while preserving the current width-four
projected-KV and MOK behavior. Width one, width eight, and other optimized
fronts retain their prior combined-reduction semantics.

## Verification boundary

The port is default-off and source-only. Focused tests cover strict selector
parsing, exact packed output geometry, bit-exact comparison against four
source projections, source-view invalidation, altered dimensions/layout/bias
fail-closed behavior, and the width-three MOK overlap split. Existing Kimi
tests must remain green. No benchmark or quality result is promoted by this
composition.

The sealed verification runtime is CPython 3.13.13 with interpreter SHA-256
`d11876ee519fe3d8866fbbe3ff9e0d1987f21a1da9c634cddb002ed0655d0803`, using the
local bundle at
`k3_unified_q3v2_affine8_q4_receipt_py313_runtime_20260813/bundle/python`:

- MLX core extension: `deeb6e2322508fd2dd000273b804c7401d0d86e2c8f8120d3fb6b8cf061989af`
- `libmlx.dylib`: `2b6c20fe6a1e3fe4855b0ac2e12b7c224eb7c2bc0faf58d66fe25fdd08844db6`
- default MLX metallib: `f2ca75128f3c3ef05c10b846288bf2a2c9fd84009e0b3a5d78d639818848dbc0`

With `PYTHONPATH` set to that bundle followed by this worktree, the focused
W3/MOK suite passed `34` tests and `16` subtests; the width-four fused-expert
suite passed `20` tests and `16` subtests; and the complete
`tests/test_kimi_k3*.py` suite passed `315` tests, skipped `2`, and reported
`347` subtests. Python compilation and `git diff --check` also passed.

Frozen source hashes at the port commit are:

- `mlx_lm/models/kimi_k3.py`: `39c59837a4a5d900d40ab8b6bb84201e2feefc9aabb3687a4e623854251d9383`
- `mlx_lm/models/kimi_k3_packed_moe_front.py`: `9dd1d75ca7022cc837165f9eae670df4df36263ef7011ccd989c9f015db9a159`
- `tests/test_kimi_k3_packed_moe_front.py`: `82f210261d45fd4a1a0b4b353b05dcf3a4b822f2bb43055dc01f180b5cf60357`
- `tests/test_kimi_k3_mok_overlap.py`: `faac10685341e274b0390b9890d71434c1cf6cbc84116a204a993bfc63cfe079`

For accounting only, an earlier source-derived screen estimated `1.4321088 ms`
per full-W3 call. Applying that estimate to 45 full-W3 calls would save
`64.444896 ms` from a `5684.585742 ms` request, or mechanically project
approximately `22.77523 tok/s` (`+1.147%`). This is an upper-bound projection,
not current-lineage evidence or credit.

Before any canary, perform a production-model lifecycle preflight: fresh
checkpoint load, shard, then first exact BF16 `(1, 3, 7168)` call; prove no
pre-load packed parent survives, post-shard installation is exact, and FP16,
FP32, width-one, width-four, width-eight, neighboring-width, and altered-layout
controls remain on stock paths. Any future evidence requires A1/B/A2 runs with
identical completion/quality, schedule, fallback/error counts, and peak-memory
parity.
