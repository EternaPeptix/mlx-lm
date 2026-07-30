# Experimental Kimi K3 post-KDA RMSNorm/sigmoid fusion

This default-off, decode-only Metal path fuses the recurrent KDA output
RMSNorm, sigmoid gate, and final multiplication:

```bash
export MLX_LM_KIMI_K3_FUSED_POST_KDA_RMS_SIGMOID_GATE=1
```

It fails closed unless inference is on Metal with the released single-token
`[batch, 1, 96, 128]` geometry, supported dtype and norm weight, and a
positive finite epsilon. Unsupported shapes use the unchanged MLX-LM graph.

The implementation preserves the pinned deployment MLX runtime's exact
RMSNorm and sigmoid rounding. The full focused suite, including AttnRes,
router fuzzing, skinny/wide KDA packing, fused experts/down, and this path,
passed 50 tests and 166 subtests exactly.

## Timing

On the local M3 Max, three fresh compiled projections saved `0.0534`,
`0.0738`, and `0.0807 ms/token` across the 69 recurrent layers. This was
small enough that a full-model two-rank gate was required.

On two 512 GB M3 Ultra systems, five canonical
575-prompt-token/128-generation-token repetitions retained completion digest
`c84d0f0464acc5f0226e5a9686e2bb8ed4b243064dfafb99d7aa7fc5cd5b0c71`
and produced:

```text
14.1443, 14.1271, 14.1047, 14.1109, 14.0286 tok/s
```

The median was `14.1109` tok/s versus `14.1097` for the matched wide-KDA
stack: `+0.009%`, or only `0.006 ms/token`.

Three 1,067-token coding-prompt repetitions retained digest
`9936f17d98ac76b2a3ad3ab768e78fae5379259da0b745881f06e7cf9c7a7959`
and produced a median `14.0794` tok/s versus `14.0764` for the matched
control (`+0.021%`).

Both changes are below run-to-run noise. The experiment is exact and remains
public for further work, but it is excluded from the accepted solid stack
because it did not demonstrate a measurable end-to-end gain.
