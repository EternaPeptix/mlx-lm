# Kimi K3 full packed-front request receipt

Status: default-off, offline diagnostic source. No service or cluster result is
claimed by this revision.

This child of `cbbceedce4ac1d072da72ea902e856ae2f6e6cc2` adds a strict
request-scoped causal receipt under
`MLX_LM_KIMI_K3_AUTHORITATIVE_PACKED_MOE_FRONT_RECEIPT=1`. The selector must be
enabled in both control and candidate arms. Its context-local begin, finish,
and abort lifecycle publishes only a fixed schema string, bounded integers,
and booleans. It never publishes prompt text, tokens, paths, tensor values,
module identities, exception strings, or source-layout details.

The receipt counts every authoritative helper call into exactly one terminal
class: gate disabled, noncontract, exact width-three packed hit, unsupported
source, or packed-dispatch fallback. Terminal ordinary/no-draft calls are
normal noncontract calls. Exact hits additionally count their four returned
projection tensors. Lazy installs are counted only when an otherwise eligible
exact `(1, 3, 7168)` call creates the authoritative parent. Invalidation and
stale-reset counters are independent side events and do not double-count a
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
