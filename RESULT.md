# Kimi K3 width-four dispatch receipt

Date: 2026-08-13

## Result

This revision supersedes the non-transactional receipt in `03b2690`. It adds
opt-in, process-local evidence instrumentation to the exact Kimi K3 width-four
fused-expert adapter. The instrumentation does not synchronize Metal or print
per-layer or per-token data.

It also repairs the full-checkpoint integration exposed by the first TP2
service canary. Gate/up derived-bias validation remains strict, while the down
projection now follows the model's existing selective contract: exact banks
use the all-derived width-four reducer and non-exact banks use the same fused
reducer with their authoritative stored down bias. Both are clean
`switch_glu_reduce` dispatches; neither is a fallback. No Metal kernel changed,
and the receipt schema and its prohibition on `switch_glu` fallback remain
unchanged.

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
instrumented width-four support or metadata guard returned `None` to its caller
before this candidate dispatched. The caller may then select another optimized
path or eventually the stock graph.

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
  `0b8eec17606b8a4fbd8c6656acbe085dd293436a33004c9282d8e549b8f2c66d`;
- `mlx_lm/models/kimi_k3_width4_fused_expert.py`:
  `5e22a89e1c9b731eedfd342d6b18a3e3324574477e7991df8040658769e0a832`;
- `tests/test_kimi_k3_width4_fused_expert.py`:
  `179342d5111ae0b6c60d97b9b6d9862b74a919d44af73ea00dbfcf735b5137fe`.

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

After the selective-down repair, the complete `test_kimi_k3*.py` family passes
with the sealed native runtime: **307 passed, 2 expected skips, 336 subtests**.
The width-four file alone passes **19 tests and 16 subtests**, including a real
Metal, bit-exact stored-down reducer comparison and a clean schema-v3 receipt.

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

The generation is captured at each instrumented adapter attempt, not at request
entry. A protected promotion bracket must therefore reset only at a fresh,
quiescent request boundary, then require `stale_completions.total == 0` in the
final snapshot. Any stale completion invalidates that bracket even though it is
correctly excluded from the current generation's throughput totals.
