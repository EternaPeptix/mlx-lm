# Kimi K3 width-four dispatch receipt

Date: 2026-08-13

## Result

This revision supersedes the non-transactional receipt in `03b2690`. It adds
opt-in, process-local evidence instrumentation to the exact Kimi K3 width-four
fused-expert adapter. It does not alter the selected kernel, synchronize Metal,
print per-layer or per-token data, or claim a performance gain.

The receipt is independently default-off. Set exactly:

```sh
MLX_LM_KIMI_K3_WIDTH4_DISPATCH_RECEIPT=1
```

Any other nonzero spelling is rejected. With the selector absent or `0`, the
existing fused-expert behavior is retained and no counters are updated.

## Receipt contract

The callable surface is:

- `snapshot_k3_width4_dispatch_receipt()` returns the current aggregate
  snapshot;
- `reset_k3_width4_dispatch_receipt()` clears all process-local counters; and
- `k3_width4_dispatch_receipt_enabled()` exposes the strict cached selector.

Snapshots use schema version 3 and contain:

- the current receipt generation;
- aggregate `attempted`, `supported`, `dispatched`, `fallback`, and `error`
  counters;
- the same counters separated into `switch_glu` and `switch_glu_reduce` paths;
- fallback reason classes (`geometry`, `selector`, or `metadata`) and exception
  type classes for terminal errors;
- the raw selector tuples observed at each instrumented attempt, including the
  receipt, fused-expert, down-reduce, width-four, and derived-bias selectors;
- aggregate terminal records with path, outcome, support state, reason class,
  selectors, and count; and
- aggregate stale-completion diagnostics by outcome and path; and
- the current selector state at snapshot time.

The implementation captures an in-flight attempt without publishing partial
counters. It commits exactly one terminal record under one lock when that
attempt dispatches, falls back, or raises. Snapshots copy only committed records
under that lock and derive every aggregate from the copy.

Reset is an epoch boundary: it increments the generation and clears the current
terminal records and stale diagnostics under the same lock. Each attempt
captures its generation at begin. If a pre-reset attempt completes afterward,
its outcome is excluded from the new generation's totals and increments only
the stale-completion diagnostic. This prevents pre-bracket work from
contaminating a protected post-reset measurement.

`attempted` is the number of terminal records, so the invariant is
`attempted == dispatched + fallback + error`. In-flight work is intentionally
invisible. `supported` means the width-four geometry and derived-bias contracts
passed; a compiled-call exception is both supported and terminally errored.
`dispatched` means the compiled candidate call returned its MLX graph value;
the receipt deliberately does not force evaluation. A `fallback` means an
instrumented width-four support or metadata guard returned to the stock graph
before candidate dispatch.

This distinction matters: the receipt proves Python adapter selection and
candidate graph construction. Normal downstream evaluation must still consume
that graph to prove device execution; no receipt-specific `mx.eval` or Metal
synchronization is introduced.

## Source identity

- Parent integration commit:
  `80e466255323c9e52d305acaddf270450b703d1d`.
- Superseded receipt commit:
  `03b2690`.
- Transactional receipt commit:
  use `git rev-parse HEAD` for this sealed source/docs commit, avoiding a
  self-referential commit identifier in its own contents.
- Branch:
  `experiment/k3-width4-dispatch-receipt-v1`.
- Worktree:
  `work/mlx-lm-k3-width4-dispatch-receipt-v1`.

No cluster hosts, network, model files, or service were used.

Final source/test SHA-256 values:

- `mlx_lm/models/kimi_k3_fused_expert.py`:
  `3e5e45337d521bf4e8f5e4b2578ab75cffe40d4fd7af39387acfb4c4e87bc95d`;
- `mlx_lm/models/kimi_k3_width4_fused_expert.py`:
  `5e22a89e1c9b731eedfd342d6b18a3e3324574477e7991df8040658769e0a832`;
- `tests/test_kimi_k3_width4_fused_expert.py`:
  `35a97c379d604acc5479768b2c5ba2cee148aa4b91ba35185b44e77eaa279c23`.

## Verification

Focused selector and mock adapter-routing tests:

```sh
rtk /Users/jeweled/Documents/Codex/2026-07-24/we/work/exo-k3-authoritative-pack/.venv/bin/python -m unittest tests.test_kimi_k3_width4_fused_expert.Width4SelectorTest tests.test_kimi_k3_width4_fused_expert.Width4AdapterRoutingTest -v
```

Result: **11/11 passed**. The receipt-specific coverage proves strict
default-off parsing, exact candidate dispatch accounting, raw selector capture,
classified unsupported-geometry fallback, atomic snapshot/reset behavior while
a worker is held inside candidate dispatch, generation-bound exclusion of that
pre-reset completion, and terminal error accounting for a compiled-call
exception. Existing width-four selector and adapter routing tests also remain
green. These focused tests mock the compiled candidate and do not dispatch a
Metal kernel.

Full fused-expert regression with the sealed unified Python 3.13 MLX runtime:

```sh
rtk env PYTHONPATH=/Users/jeweled/Documents/Codex/2026-07-24/we/work/k3_unified_q3v2_affine8_q4_aux_py313_runtime_20260813/bundle/python /Users/jeweled/Documents/Codex/2026-07-24/we/work/exo-k3-authoritative-pack/.venv/bin/python -m unittest tests.test_kimi_k3_fused_switch_glu tests.test_kimi_k3_fused_down_reduce tests.test_kimi_k3_fused_expert tests.test_kimi_k3_width4_fused_expert -v
```

Result: **43/43 passed**. This includes receipt-enabled evaluation of the real
width-four compiled reduce adapter against the stock graph with bit-exact BF16
output and terminal totals `1 attempted / 1 supported / 1 dispatched / 0
fallback / 0 error`. The width-one-through-three regressions also passed,
preserving the Config-B path.

Static verification:

```sh
rtk /Users/jeweled/Documents/Codex/2026-07-24/we/work/exo-k3-authoritative-pack/.venv/bin/python -m ruff check mlx_lm/models/kimi_k3_fused_expert.py tests/test_kimi_k3_width4_fused_expert.py
rtk git diff --check
rtk sha256sum -c SHA256SUMS
```

Result: Ruff, formatting, diff checking, and the sealed checksum manifest
passed.

## Evidence boundary

This is evidence instrumentation, not a throughput optimization. It has not
been exercised with real Kimi K3 weights, a two-rank JACCL service, completion
parity, or an A/B/A benchmark. A service run may use the snapshot only as one
dispatch receipt alongside its ordinary output consumption, rank receipts,
topology attestation, and performance measurements.
