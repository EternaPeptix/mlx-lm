# Kimi K3 authoritative W1+W3 packed-front request receipt

Status: default-off, offline diagnostic source. No service or cluster result is
claimed by this revision.

This child of `cbbceedce4ac1d072da72ea902e856ae2f6e6cc2` adds a strict
request-scoped causal receipt under
`MLX_LM_KIMI_K3_AUTHORITATIVE_PACKED_MOE_FRONT_RECEIPT=1`. The selector must be
enabled in both control and candidate arms. Its context-local begin, finish,
and abort lifecycle publishes only a fixed schema string, bounded integers,
and booleans. It never publishes prompt text, tokens, paths, tensor values,
module identities, exception strings, or source-layout details.

The authoritative implementation already accelerates exact width one as well
as the new exact width three. The receipt therefore labels the candidate
honestly as the combined W1+W3 route. It partitions every helper call into gate
disabled, noncontract, packed hit, unsupported source, or packed-dispatch
fallback; eligible calls, hits, returned tensors, installs, unsupported calls,
and dispatch fallbacks are also partitioned independently for width one and
width three. Aggregate identities are checked at finish. Ordinary/no-draft W1
calls are never hidden under generic noncontract accounting. Invalidation and
stale-reset counters remain independent side events and do not double-count a
helper outcome.

Begin and finish each require and traverse the exact
`language_model.model.layers` path read-only. The caller supplies an expected
sparse-layer count (92 in the EXO contract); traversal fails unless exactly
that many layers expose the production four-bank layout. It counts only
installed `AuthoritativePackedK3MoEFront` parents that still match those source
banks. This makes startup `0 -> 92` and later `92 -> 92` claims observations
rather than inference from calls seen during a request, and prevents a wrong
model wrapper from masquerading as a clean zero. The Python layer cannot
observe whether Metal internally selected its specialized kernel or a generic
implementation, so the receipt deliberately does not invent a native-fallback
counter. Native core, `libmlx`, metallib, JACCL, and selector-map identity
remain obligations of the signed arm receipt.

Instrumentation is structurally symmetric: matched A/B arms both enable the
receipt selector and pay the same context lifecycle, helper-call accounting,
model scans, bilateral downstream agreement, and structured logging. Only the
two authoritative MLX-LM gates change between control and candidate. The
native affine8-Q3-triplet selector stays enabled in both arms.

Because this instrumentation changes the MLX-LM source revision, it spends the
prior `cbbceed` timing projection as promotion evidence. That projection may be
historical context only; the receipt-bearing source requires fresh-bank
revalidation before any speed or overhead claim is made.
