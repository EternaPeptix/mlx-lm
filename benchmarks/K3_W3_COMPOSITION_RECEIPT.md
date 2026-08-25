# Kimi K3 request-local W3 composition receipt

Date: 2026-08-16

## Result

This child of the frozen offline W3 composition adds diagnostic evidence only.
It does not change a Metal kernel, synchronize a graph, measure latency, or
claim throughput credit.  The default-off receipt proves which Python-side
packed-front and KDA-prework branches were selected during one bound request.
Native Q3 dispatch is deliberately outside its claim boundary and must be
joined by EXO with the separate native counter.

The source commit is
`ce24ae6c9b12152ebf64262528b929950bd5efab`.  Its lineage is:

- frozen combined parent:
  `72164e3521b8ee68d605963cfd3266a7341279e6`;
- mature packed-receipt semantics replayed as `8743242`, `cae7e25`,
  `216cced`, and `abd00f2` (upstream donor tip `aa924740`);
- combined receipt source and tests: `ce24ae6`.

Branch: `codex/k3-offline-w3-composition-receipt-v1`.

Worktree: any clean checkout of the source commit above.

## Request contract

Set `MLX_LM_KIMI_K3_W3_COMPOSITION_RECEIPT=1`, then call:

- `begin_k3_w3_composition_receipt(request_token, model)`;
- `finish_k3_w3_composition_receipt(sequence, request_token, model)`; or
- `abort_k3_w3_composition_receipt(sequence, request_token)`.

The schema is `kimi-k3-w3-composition-receipt/v1`.  Begin clears abandoned
context state, validates the nonnegative int64 request token, snapshots every
selector, and proves exactly 92 released sparse layers and 69 released KDA
layers.  Finish checks the binding, re-validates the unchanged selector
snapshot, repeats both topology scans, proves complete counter partitions and
zero pending KDA admissions, clears the context, and returns scalar fields
only.  Abort clears the context and publishes no partial positive telemetry.

The selector snapshot binds:

- the authoritative packed parent and W3 pair;
- KDA prework and ReplaySSM;
- projected-KV enablement and its canonically spelled numeric cap;
- exact async boundary/state strings and the W3 async bit; and
- separate native Q3 route and dispatch-receipt bits.

The packed pair must be jointly zero or jointly one.  When it is one, both
native diagnostic selector bits must also be one.  This establishes route
identity but does not claim that native dispatch occurred.

The retained mature packed-front accounting partitions every helper call into
gate-disabled, non-contract, W1/W3 packed hit, unsupported, or dispatch
fallback outcomes.  It separately records output tensors, W1/W3 lazy
installation, invalidation, stale reset, and installed-parent counts before
and after the request.  Startup therefore permits either W1 or W3 to be the
first installer; later API requests can prove 92-to-92 with zero installs.

KDA accounting is reached only on the production `T > 1` path.  Every call is
partitioned into gate-disabled, non-contract, or admitted.  An admitted call
becomes successful only after the fused helper returns.  A returned `None` is
an explicit fallback and still triggers the existing fail-closed runtime
error; an exception leaves an unsettled admission, so only abort is valid.

No request-round count is hard-coded.  The receipt reports observed calls, and
EXO must validate them against the separately observed schedule.  The only
fixed cardinalities are released-model topology (92 sparse and 69 KDA layers).
This avoids fitting the receipt to the accepted control's 45 full W3 rounds.

## Verification

The sealed CPython 3.13.13 MLX runtime was used offline.  No cluster host,
network, service, model checkpoint, or timing command was touched.

Focused packed/KDA/combined receipt and composition regressions:

```sh
rtk env PYTHONPATH=/path/to/mlx-python:/path/to/mlx-lm /path/to/python -m unittest tests.test_kimi_k3_packed_moe_front tests.test_kimi_k3_w3_prework tests.test_kimi_k3_w3_composition_receipt tests.test_kimi_k3_w3_offline_composition tests.test_kimi_k3_compiled_decode tests.test_kimi_k3_mok_overlap
```

Result: **98 passed, 1 expected skip**.

Complete Kimi test family:

```sh
rtk env PYTHONPATH=/path/to/mlx-python:/path/to/mlx-lm /path/to/python -m unittest discover -s tests -p 'test_kimi_k3*.py'
```

Result: **360 passed, 2 expected skips**.

Static checks:

```sh
rtk python3 -m py_compile mlx_lm/models/kimi_k3.py mlx_lm/models/kimi_k3_packed_moe_front.py tests/test_kimi_k3_w3_prework.py tests/test_kimi_k3_w3_composition_receipt.py
rtk /path/to/python -m ruff check mlx_lm/models/kimi_k3.py mlx_lm/models/kimi_k3_packed_moe_front.py tests/test_kimi_k3_packed_moe_front.py tests/test_kimi_k3_w3_prework.py tests/test_kimi_k3_w3_composition_receipt.py
rtk /path/to/python -m ruff format --check mlx_lm/models/kimi_k3_packed_moe_front.py tests/test_kimi_k3_w3_composition_receipt.py
rtk git diff --check
```

All listed static checks passed.  The frozen composition parent already has
formatter-only drift in `kimi_k3.py` and `test_kimi_k3_w3_prework.py` under
this Ruff version; the new receipt implementation and its dedicated test are
formatted, and all changed files pass lint and whitespace checks.

## Frozen source hashes

- `mlx_lm/models/kimi_k3.py`:
  `7aff86896e70baf2b986808341b5f292f8c8ecd55d17a9693813cbe25b356c97`;
- `mlx_lm/models/kimi_k3_packed_moe_front.py`:
  `8c0ae8a767b8fd965369597d3be12c14190a72a33dddc2f849d5d9756b5e7f5a`;
- `tests/test_kimi_k3_w3_prework.py`:
  `68d577d5be80813274899b1c86f64d6ab7c03585ac2fc37d4febd75a5c6462bf`;
- `tests/test_kimi_k3_w3_composition_receipt.py`:
  `3765cee831d4e3ae473434a866cfca546cfbe150095b6343b091eca99146863d`.

## Promotion boundary

This branch remains zero-credit and offline-only.  It must not be used to
promote a speedup.  A future diagnostic run must join this raw MLX receipt with
rank agreement, request/setup/source identity, the native dispatch delta, the
EXO schedule/acceptance/cache receipt, and completion parity.  A separate
telemetry-off protected A/B/A is still required for performance evidence.
