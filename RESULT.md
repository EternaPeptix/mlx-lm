# Kimi K3 width-four fused-reduce adapter integration

Date: 2026-08-13

## Result

The production model adapter now selects the exact screened width-four
front8/down-results4/SIMDs8 fused down/router-reduce path. The path remains
default-off and requires all four opt-ins/contracts before dispatch:

- `MLX_LM_KIMI_K3_FUSED_EXPERTS=1`;
- `MLX_LM_KIMI_K3_FUSED_DOWN_REDUCE=1`;
- `MLX_LM_KIMI_K3_FUSED_EXPERT_WIDTH4_EXACT=1`; and
- `MLX_LM_KIMI_K3_DERIVE_AFFINE2_BIAS=1`, plus full validated derived-bias
  metadata for gate, up, and down projections.

Width four is handled before the generic width-one-through-three support
checks. A new public projection-level support predicate validates the exact
top-8 routes, BF16 router weights, 896-expert down bank, tile, and Metal
kernel before the fused front runs. Unsupported geometry returns `None` and
falls back to the stock model graph. The generic width-one-through-three
branch is unchanged.

## Source

- Integration parent:
  `784abd8da8a7db257e39ae2e260bcb960e18eed6`.
- Local branch:
  `experiment/k3-width4-fused-reduce-integration-v1`.
- Source worktree:
  `work/mlx-lm-k3-width4-fused-reduce-integration-v1`.
- The sealed commit is intentionally reported with `git rev-parse HEAD` rather
  than embedded here, which would make the commit self-referential. No push or
  cluster-host contact was performed.

Final SHA-256 values:

- `mlx_lm/models/kimi_k3_fused_expert.py`:
  `b736af039484b621c52a560b20ad86d4a421eea45372b45b263553c4da179cf7`;
- `mlx_lm/models/kimi_k3_width4_fused_expert.py`:
  `5e22a89e1c9b731eedfd342d6b18a3e3324574477e7991df8040658769e0a832`;
- `tests/test_kimi_k3_width4_fused_expert.py`:
  `c75700ad2ee6547b640137ebde82de0252fa4e5b96e113d31bc806a466f4cb57`.

## Verification

Mock-only selector and adapter routing tests, with no Metal kernel dispatch:

```sh
rtk env PYTHONPATH=/Users/jeweled/Documents/Codex/2026-07-24/we/work/mlx-lm-k3-width4-fused-reduce-integration-v1:/Users/jeweled/Documents/Codex/2026-07-24/we/work/mlx-k3-affine6-q4-quad-v1/build/lib.macosx-11.0-arm64-cpython-313:/Users/jeweled/Documents/Codex/2026-07-24/we/work/mlx-k3-affine6-q4-quad-v1/python /Users/jeweled/Documents/Codex/2026-07-24/we/work/exo-k3-authoritative-pack/.venv/bin/python -m unittest tests.test_kimi_k3_width4_fused_expert.Width4SelectorTest tests.test_kimi_k3_width4_fused_expert.Width4AdapterRoutingTest
```

Result: **6/6 passed**. This covers default-off behavior, exact selector
parsing, model-adapter selection, neighboring-geometry fallback, and
derived-bias fail-closed behavior.

Combined exact Q4 Python 3.13 Metal regression:

```sh
rtk env MLX_METAL_K3_AFFINE6_Q4_QUAD=1 PYTHONPATH=/Users/jeweled/Documents/Codex/2026-07-24/we/work/mlx-lm-k3-width4-fused-reduce-integration-v1:/Users/jeweled/Documents/Codex/2026-07-24/we/work/mlx-k3-affine6-q4-quad-v1/build/lib.macosx-11.0-arm64-cpython-313:/Users/jeweled/Documents/Codex/2026-07-24/we/work/mlx-k3-affine6-q4-quad-v1/python /Users/jeweled/Documents/Codex/2026-07-24/we/work/exo-k3-authoritative-pack/.venv/bin/python -m unittest tests.test_kimi_k3_fused_switch_glu tests.test_kimi_k3_fused_down_reduce tests.test_kimi_k3_fused_expert tests.test_kimi_k3_width4_fused_expert
```

Result: **38/38 passed**. The width-four adapter test invoked the compiled
adapter entry point and matched the stock reduced BF16 output bit-for-bit.
The existing width-one-through-three fused-expert suite passed unchanged.
The sandboxed launch could not see Metal, so the successful run used approved
host GPU access.

Static lint:

```sh
rtk /Users/jeweled/Documents/Codex/2026-07-24/we/work/exo-k3-authoritative-pack/.venv/bin/python -m ruff check mlx_lm/models/kimi_k3_fused_expert.py mlx_lm/models/kimi_k3_width4_fused_expert.py tests/test_kimi_k3_width4_fused_expert.py
```

Result: **all checks passed**. `git diff --check` also passed.

## Deployment boundary

This worktree is integration evidence only. It was tested with the local split
Q4 Python 3.13 runtime, which is not a production package. No checkpoint,
two-rank service, cluster host, throughput, completion parity, or A/B/A result
was exercised here. Promotion still requires a production-built runtime and
the planned real-weight two-Mac bracket; the isolated component ceiling must
not be reported as service tokens per second.
