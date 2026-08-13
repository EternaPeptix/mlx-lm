# Kimi K3 width-four dispatch receipt

Date: 2026-08-13

## Result

Commit `03b2690` adds opt-in, process-local evidence instrumentation to the
exact Kimi K3 width-four fused-expert adapter. It does not alter the selected
kernel, synchronize Metal, print per-layer or per-token data, or claim a
performance gain.

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

Snapshots use schema version 1 and contain:

- aggregate `attempted`, `supported`, `dispatched`, and `fallback` counters;
- the same counters separated into `switch_glu` and `switch_glu_reduce` paths;
- fallback reason classes (`geometry`, `selector`, or `metadata`);
- the raw selector tuples observed at each instrumented attempt, including the
  receipt, fused-expert, down-reduce, width-four, and derived-bias selectors;
  and
- the current selector state at snapshot time.

An `attempted` receipt begins only after the shared adapter guards have selected
the exact width-four branch: fused-expert selectors, inference state,
activation, affine-2 projection contract, and expert-count agreement have
already passed. `supported` means the width-four geometry and derived-bias
contracts passed. `dispatched` means the compiled candidate call returned its
MLX graph value; the receipt deliberately does not force evaluation. A
`fallback` means an instrumented width-four support or metadata guard returned
to the stock graph before candidate dispatch.

This distinction matters: the receipt proves Python adapter selection and
candidate graph construction. Normal downstream evaluation must still consume
that graph to prove device execution; no receipt-specific `mx.eval` or Metal
synchronization is introduced.

## Source identity

- Parent integration commit:
  `80e466255323c9e52d305acaddf270450b703d1d`.
- Receipt source commit:
  `03b2690`.
- Branch:
  `experiment/k3-width4-dispatch-receipt-v1`.
- Worktree:
  `work/mlx-lm-k3-width4-dispatch-receipt-v1`.

This documentation is committed after `03b2690`; naming the receipt source
commit avoids a self-referential documentation hash. No cluster hosts,
network, model files, or service were used.

Final source/test SHA-256 values:

- `mlx_lm/models/kimi_k3_fused_expert.py`:
  `ea8173b439279e87436d6a69440dcf7c7887be20ec22d94ebed9e2613dc595c0`;
- `mlx_lm/models/kimi_k3_width4_fused_expert.py`:
  `5e22a89e1c9b731eedfd342d6b18a3e3324574477e7991df8040658769e0a832`;
- `tests/test_kimi_k3_width4_fused_expert.py`:
  `52596a7519fe7d35c49a2939087c8cdcb407de26940b7ed2d95c6f452555938d`.

## Verification

Focused selector and mock adapter-routing tests:

```sh
rtk /Users/jeweled/Documents/Codex/2026-07-24/we/work/exo-k3-authoritative-pack/.venv/bin/python -m unittest tests.test_kimi_k3_width4_fused_expert.Width4SelectorTest tests.test_kimi_k3_width4_fused_expert.Width4AdapterRoutingTest -v
```

Result: **9/9 passed**. The receipt-specific coverage proves strict
default-off parsing, exact candidate dispatch accounting, raw selector capture,
and classified unsupported-geometry fallback. Existing width-four selector and
adapter routing tests also remain green. These focused tests mock the compiled
candidate and do not dispatch a Metal kernel.

Static verification:

```sh
rtk /Users/jeweled/Documents/Codex/2026-07-24/we/work/exo-k3-authoritative-pack/.venv/bin/python -m ruff check mlx_lm/models/kimi_k3_fused_expert.py tests/test_kimi_k3_width4_fused_expert.py
rtk git diff --check
rtk sha256sum -c SHA256SUMS
```

Result: Ruff, diff checking, and the sealed checksum manifest passed. The parent
integration commit had separately passed its 38-test Metal regression; that
suite was not rerun or represented as new receipt evidence here.

## Evidence boundary

This is evidence instrumentation, not a throughput optimization. It has not
been exercised with real Kimi K3 weights, a two-rank JACCL service, completion
parity, or an A/B/A benchmark. A service run may use the snapshot only as one
dispatch receipt alongside its ordinary output consumption, rank receipts,
topology attestation, and performance measurements.
