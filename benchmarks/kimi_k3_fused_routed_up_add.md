# Kimi K3 routed-up/shared/residual decode fusion

This default-off prototype removes the two full-width elementwise dispatches
at the end of every Kimi K3 sparse MoE layer.  The stock TP2 path does:

1. affine-8/group-64 routed-up QMV, `3584 -> 7168`;
2. BF16 `routed + shared`; and
3. BF16 `residual + MoE output`.

The candidate reproduces the native MLX affine-QMV loop and its FP32-to-BF16
output boundary, then performs both BF16 additions in the output writer.  It
is enabled only by:

```sh
MLX_LM_KIMI_K3_FUSED_ROUTED_UP_ADD=1
```

Unsupported device, training, prefill/multi-token shape, dtype, model
geometry, quantization, or output bias returns to the stock MLX-LM path.

## Exactness and regression coverage

The focused suite checks the default-off and fail-closed contracts, verifies
three randomized full TP2-shape inputs bit for bit, and verifies that the
decoder does not add a consumed residual twice:

```text
python -m unittest -v tests.test_kimi_k3_fused_routed_up_add
Ran 5 tests: OK
```

The complete Kimi K3 test pattern produced 117 passes and one distributed
skip.  Its one failure,
`test_kimi_k3_fused_switch_glu.FusedSwitchGLUTest.test_matches_native_quantized_path`,
is unrelated and reproduces unchanged on the accepted `f6262cc` parent.

## Shape-realistic Metal result

Hardware: Apple M3 Max, 64 GB.

Geometry: one BF16 token, routed latent width 3,584, hidden width 7,168,
affine 8-bit/group-64 weights.  Protocol: 150 warmups, followed by 31
alternating-order trials of 800 synchronized iterations.

| Metric | Stock | Fused |
| --- | ---: | ---: |
| Independent median | 0.359729 ms/layer | 0.351662 ms/layer |
| Median paired delta |  | 0.008308 ms/layer saved |
| Median paired speedup |  | 1.0235x |
| Pairs won |  | 29 / 31 |
| Exact output |  | yes |

The independent-median delta is `0.008067 ms/layer`, mechanically projecting
to `0.742 ms/token` over K3's 92 sparse layers. The live asynchronous TP2
schedule may hide some of the removed dispatch latency, so the pre-deployment
expectation was `0.3-0.75 ms/token`, not the full isolated number as a
guarantee.

## Exact full-model TP2 result

The candidate was deployed over the accepted wide-KDA stack on two 512 GB M3
Ultra systems with four-rail JACCL and direct-mesh decode collectives.

| Prompt | Wide-KDA control | Routed-up/add | Gain |
| --- | ---: | ---: | ---: |
| Canonical, 575 prompt / 128 decode, 5 runs | 14.1097 tok/s | **14.2375 tok/s** | **+0.91%** |
| Code-shaped, 1,067 prompt / 128 decode, 3 runs | 14.0764 tok/s | **14.2155 tok/s** | **+0.99%** |

The canonical latency fell from `70.873` to `70.237 ms/token`, a measured
`0.636 ms/token` saving. Every canonical completion retained SHA-256
`c84d0f0464acc5f0226e5a9686e2bb8ed4b243064dfafb99d7aa7fc5cd5b0c71`;
every code-shaped completion retained SHA-256
`9936f17d98ac76b2a3ad3ab768e78fae5379259da0b745881f06e7cf9c7a7959`.
Peak memory remained approximately `414 GB` per rank.

## Rejected adjacent fusion

Fusing shared SiTU directly into the shared-expert down QMV was exact, but
output-parallel QMV recomputed tanh/sigmoid for each output tile and ran about
three times slower than materializing SiTU once.  It is intentionally not part
of this commit.

## Scope

This is decode-only and independent of prefix length in absolute work saved.
As attention cost rises with context, the relative throughput gain decreases.
It changes neither weights nor KV layout and therefore does not materially
increase context capacity; it only avoids two transient 7,168-element BF16
intermediates per active layer.
