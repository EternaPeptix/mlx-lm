"""Paired projection-only screen for K3's persistent expanded K/V cache."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import mlx.core as mx

from mlx_lm.models import kimi_k3


def _attention(num_heads):
    args = kimi_k3.TextArgs(
        hidden_size=8,
        num_attention_heads=96,
        num_key_value_heads=96,
        q_lora_rank=None,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        mla_use_nope=True,
        mla_use_output_gate=False,
    )
    attention = kimi_k3.KimiK3MLAAttention(args)
    attention.set_dtype(mx.bfloat16)
    attention.embed_q = attention.embed_q.to_quantized(64, 6)
    attention.unembed_out = attention.unembed_out.to_quantized(64, 6)
    if num_heads != 96:
        attention.embed_q.apply(lambda value: value[:num_heads])
        attention.unembed_out.apply(lambda value: value[:num_heads])
        attention.num_heads = num_heads
    attention.eval()
    mx.eval(attention.parameters())
    return attention


def _time(callable_):
    started = time.perf_counter_ns()
    callable_()
    return (time.perf_counter_ns() - started) / 1_000_000.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix-tokens", type=int, default=574)
    parser.add_argument("--suffix-tokens", type=int, default=3, choices=(1, 2, 3, 4))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--trials", type=int, default=21)
    parser.add_argument("--num-heads", type=int, default=48, choices=(48, 96))
    args = parser.parse_args()
    if args.prefix_tokens < 32:
        raise SystemExit("prefix must be at least 32 tokens")
    if not mx.metal.is_available() or mx.default_device() != mx.gpu:
        raise SystemExit("benchmark requires an Apple Metal GPU")

    mx.random.seed(20260805)
    attention = _attention(args.num_heads)
    total = args.prefix_tokens + args.suffix_tokens
    inputs = [
        mx.random.normal((1, 1, total, 512)).astype(mx.bfloat16) for _ in range(3)
    ]
    mx.eval(inputs)

    capacity = ((total + 255) // 256) * 256
    projected_keys = mx.zeros((1, args.num_heads, capacity, 128), dtype=mx.bfloat16)
    projected_values = mx.zeros((1, args.num_heads, capacity, 128), dtype=mx.bfloat16)
    prefix = inputs[0][..., : args.prefix_tokens, :]
    projected_keys[..., : args.prefix_tokens, :] = attention.embed_q(
        prefix, transpose=False
    )
    projected_values[..., : args.prefix_tokens, :] = attention.unembed_out(prefix)
    mx.eval(projected_keys, projected_values)

    def full(index):
        latent = inputs[index % len(inputs)]
        keys = attention.embed_q(latent, transpose=False)
        values = attention.unembed_out(latent)
        mx.eval(keys, values)
        return keys, values

    def incremental(index):
        latent = inputs[index % len(inputs)]
        suffix = latent[..., args.prefix_tokens :, :]
        padded = mx.pad(
            suffix,
            (
                (0, 0),
                (0, 0),
                (0, 32 - args.suffix_tokens),
                (0, 0),
            ),
        )
        keys = attention.embed_q(padded, transpose=False)[..., : args.suffix_tokens, :]
        values = attention.unembed_out(padded)[..., : args.suffix_tokens, :]
        projected_keys[..., args.prefix_tokens : total, :] = keys
        projected_values[..., args.prefix_tokens : total, :] = values
        mx.eval(projected_keys, projected_values)
        return (
            projected_keys[..., :total, :],
            projected_values[..., :total, :],
        )

    for index in range(args.warmup):
        full(index)
        incremental(index)

    full_ms = []
    incremental_ms = []
    for trial in range(args.trials):
        if trial % 2:
            incremental_ms.append(_time(lambda trial=trial: incremental(trial)))
            full_ms.append(_time(lambda trial=trial: full(trial)))
        else:
            full_ms.append(_time(lambda trial=trial: full(trial)))
            incremental_ms.append(_time(lambda trial=trial: incremental(trial)))

    expected_keys, expected_values = full(0)
    actual_keys, actual_values = incremental(0)
    mx.eval(expected_keys, expected_values, actual_keys, actual_values)
    keys_exact = bool(mx.array_equal(expected_keys, actual_keys).item())
    values_exact = bool(mx.array_equal(expected_values, actual_values).item())

    full_median = statistics.median(full_ms)
    incremental_median = statistics.median(incremental_ms)
    saving = full_median - incremental_median
    result = {
        "prefix_tokens": args.prefix_tokens,
        "suffix_tokens": args.suffix_tokens,
        "trials": args.trials,
        "num_heads": args.num_heads,
        "keys_bit_exact": keys_exact,
        "values_bit_exact": values_exact,
        "full_projection_median_ms_per_layer": full_median,
        "incremental_projection_median_ms_per_layer": incremental_median,
        "saving_median_ms_per_layer": saving,
        "projection_speedup": full_median / incremental_median,
        "projected_saving_ms_per_24_mla_layers": saving * 24,
        "full_ms": full_ms,
        "incremental_ms": incremental_ms,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not keys_exact or not values_exact:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
