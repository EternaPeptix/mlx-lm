# Kimi K3 offline W3 source composition

Date: 2026-08-16

Status: **frozen default-off source/tests composition; no isolated timing,
cluster, service, model-weight, quality, memory, or performance evidence.**

## Lineage

The worktree and branch are isolated at:

- worktree: `work/mlx-lm-k3-offline-w3-composition-v1`
- branch: `codex/k3-offline-w3-composition-v1`
- exact control parent: `591e11093b55b3b03f7cbc5018cd3b7d47abba4f`
- source/test freeze: `c245c2075a4e2c4266dabb1c1635dfc03ff254c1`
- source/test tree: `27ce1c0292b68ce9db466fd32a86848b770a36eb`

The three frozen single-parent source candidates were semantically composed in
this order without importing their later receipt commits:

1. authoritative packed W3 source
   `bdfede4ccc21e3f69c90ba2d7839b0522b5e3cb3`, replayed as
   `1e8ae043b8b0a42f9e4fae2000b153ce9a19fc98`;
2. W3 KDA prework/history source
   `a257a4b7e9aba8a7d1b4e5b83749b8a6e9414e56`, replayed as
   `969365c8551fdaa78840df33b06c7aab851e0486`;
3. deferred W3 async source
   `27f6ec2c4b9b5e87f94baf1a856db8eb04d3cdf1`, replayed as
   `5642c02de728bd559504b8f7d1d5ef12bdb35319`.

All three named source worktrees had empty `git status --porcelain` before and
after composition. No source worktree was edited.

## Default-off contracts and seam

The independent selectors remain strict and default-off:

- packed MoE front:
  `MLX_LM_KIMI_K3_AUTHORITATIVE_PACKED_MOE_FRONT=1` plus
  `MLX_LM_KIMI_K3_AUTHORITATIVE_PACKED_MOE_FRONT_WIDTH3=1`;
- KDA prework/history: `MLX_LM_KIMI_K3_W3_PREWORK_HISTORY=1`, additionally
  requiring the existing ReplaySSM speculative selector;
- deferred scheduling: `MLX_LM_KIMI_K3_ASYNC_DECODE_WIDTH3=1`, with the
  existing `laguna8`, hidden-root, projected-KV, and ReplaySSM requirements.

The KDA candidate begins after its QKV and `a` projection calls. The packed
candidate replaces the later sparse-MoE front in the same decoder layer. The
deferred candidate captures the completed layer hidden state after attention,
residual/AttnRes, routed/shared MoE work, and MOK branching. Neither packed
projection storage nor deferred root ownership is absorbed into the KDA
kernel. Unsupported geometry returns to the pre-existing path before candidate
dispatch; post-KDA-selector contract drift continues to fail closed.

The composition does not alter the existing width-one or width-eight packed
fronts, width-four receipt route, projected-KV widths `(3, 4)`, top-k selector,
MOK routed/shared overlap, compiled decode, or default stock route.

## Explicit composition coverage

`tests/test_kimi_k3_w3_offline_composition.py` uses the production selectors
unchanged. Because the real packed and KDA kernels admit only released TP2
geometry, its small full-target integration replaces each admitted candidate
call with the exact stock arithmetic at that same seam. The frozen focused
source tests separately execute the real packed and KDA Metal kernels.

The full-target composition suite proves:

- selectors off, packed-only, KDA-only, and packed-plus-KDA all produce exact
  target logits, greedy tokens, auxiliary hidden taps, and closed caches;
- joint packed-plus-KDA speculative history matches for every KDA layer,
  including convolution checkpoints and ReplaySSM `(v, raw_k, gk, beta)`;
- resolution with consumed prefixes 1, 2, and 3 is exact for KDA, latent MLA,
  and projected-KV cache state;
- cancellation restores the complete pre-transaction cache;
- the deferred root equals the post-layer auxiliary tap at boundary 1, proving
  capture occurs after the composed target layer, and graph construction does
  not call `mx.async_eval`;
- unsupported width two executes stock with zero packed or KDA dispatches;
- model parameter arrays are unchanged after all joint target calls.

The packed source suite additionally covers exact W1/W3/W8 output, installed
source-view invalidation, altered layout/dtype/bias fallback, and no parameter
registration mutation. Existing tests retain exact W4 receipt, projected-KV
`(3, 4)`, top-k/width-four, MOK overlap, deferred greedy-token, cache/history,
and rollback coverage.

## Sealed local verification

The offline verification runtime was the existing sealed CPython 3.13.13
bundle. Its identities matched the prior frozen receipts:

- interpreter:
  `d11876ee519fe3d8866fbbe3ff9e0d1987f21a1da9c634cddb002ed0655d0803`;
- MLX core:
  `deeb6e2322508fd2dd000273b804c7401d0d86e2c8f8120d3fb6b8cf061989af`;
- `libmlx.dylib`:
  `2b6c20fe6a1e3fe4855b0ac2e12b7c224eb7c2bc0faf58d66fe25fdd08844db6`;
- default metallib:
  `f2ca75128f3c3ef05c10b846288bf2a2c9fd84009e0b3a5d78d639818848dbc0`.

Results:

- new composition file: **4 passed, 6 subtests passed**;
- focused packed/KDA/deferred/cache/W4 set: **159 passed, 2 skipped,
  189 subtests passed**;
- complete `tests/test_kimi_k3*.py` family: **333 passed, 2 skipped,
  391 subtests passed**;
- Ruff, Python compilation, and `git diff --check`: passed.

The two expected skips are pre-existing guarded cases. The only messages were
the runtime's three SWIG deprecation warnings. A first sandboxed collection
attempt could not access Metal; the identical sealed command passed with local
GPU access. No host, network, service, model checkpoint, or benchmark was
used. No timing command was run.

## Frozen file hashes

- `mlx_lm/models/kimi_k3.py`:
  `c111751d37a030ec16852b88cb20fb3e2aea537ca458521367fbc5e68b6b110c`
- `mlx_lm/models/kimi_k3_packed_moe_front.py`:
  `9dd1d75ca7022cc837165f9eae670df4df36263ef7011ccd989c9f015db9a159`
- `mlx_lm/models/kimi_k3_w3_prework.py`:
  `f8fcba947dc0e52c335522bd0d152b57679818a64db9edc6e6ba8f58834cb254`
- `tests/test_kimi_k3_packed_moe_front.py`:
  `82f210261d45fd4a1a0b4b353b05dcf3a4b822f2bb43055dc01f180b5cf60357`
- `tests/test_kimi_k3_mok_overlap.py`:
  `faac10685341e274b0390b9890d71434c1cf6cbc84116a204a993bfc63cfe079`
- `tests/test_kimi_k3_w3_prework.py`:
  `33123a6d755aaf82e35b68b9b8b5d8ea9c258e133b2aea9da4e59514f890d9d6`
- `tests/test_kimi_k3_compiled_decode.py`:
  `7ba00f0d27dbbef5d1957dee46647647dc92e4cf7a21bb7261b2be6cae7b6c07`
- `tests/test_kimi_k3_w3_offline_composition.py`:
  `962cfd1ce577faf478ead73438654ab3ac348939bfe650e0bc6c8c9a395e41ac`

## Blockers and evidence boundary

- **P0 offline source/test blockers: none.** The requested source-only
  composition and sealed local suites are complete.
- **P1 integration blocker:** no current EXO bilateral post-agreement
  materializer is composed here. Deferred roots must not be submitted before
  both TP ranks agree; submit/final-evaluate/cancel failure must poison shared
  target/JACCL state before another collective or request.
- **P1 production blocker:** fresh real-checkpoint load/shard/first-call
  lifecycle and exact production geometry must prove the packed views, KDA
  dispatch, MOK split, projected caches, top-k receipt, and unsupported
  fallbacks together.
- **P1 evidence blocker:** any performance claim still requires separately
  authorized matched A/B/A with exact source/runtime, completion and quality
  parity, topology, peak memory, fallback/error counts, and rollback state.

This receipt freezes source and tests only. It deliberately contains no
isolated projected timing and grants no canary or promotion authorization.
