# Copyright © 2026 Apple Inc.

import unittest

import mlx.core as mx

from mlx_lm.models.base import scaled_dot_product_attention
from mlx_lm.models.deepseek_v32 import (
    _fused_mla_prefill_attention,
    _indexer_topk,
)


class TestDeepseekV32Attention(unittest.TestCase):
    def test_streaming_indexer_topk_matches_full_scores(self):
        mx.random.seed(11)
        batch, heads, queries, keys, dims = 1, 4, 5, 19, 8
        topk = 5
        q = mx.random.normal((batch, heads, queries, dims))
        k = mx.random.normal((batch, 1, keys, dims))
        weights = mx.random.normal((batch, heads, queries, 1))
        mask = mx.tril(
            mx.ones((queries, keys), dtype=mx.bool_),
            k=keys - queries,
        )

        reference = _indexer_topk(
            q,
            k,
            weights,
            mask,
            topk=topk,
            key_chunk_size=0,
        )
        candidate = _indexer_topk(
            q,
            k,
            weights,
            mask,
            topk=topk,
            key_chunk_size=7,
        )
        mx.eval(reference, candidate)

        self.assertTrue(
            mx.array_equal(
                mx.sort(reference, axis=-1),
                mx.sort(candidate, axis=-1),
            )
        )

    def test_fused_mla_prefill_attention(self):
        mx.random.seed(7)
        batch, heads, tokens = 1, 4, 17
        content_dim, rope_dim, value_dim = 8, 4, 12
        scale = (content_dim + rope_dim) ** -0.5

        q_nope = mx.random.normal((batch, heads, tokens, content_dim))
        q_pe = mx.random.normal((batch, heads, tokens, rope_dim))
        k_nope = mx.random.normal((batch, heads, tokens, content_dim))
        k_pe = mx.random.normal((batch, 1, tokens, rope_dim))
        values = mx.random.normal((batch, heads, tokens, value_dim))
        mask = mx.tril(mx.ones((tokens, tokens), dtype=mx.bool_))

        pe_scores = (q_pe * scale) @ k_pe.swapaxes(-1, -2)
        pe_scores = mx.where(
            mask,
            pe_scores,
            mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype),
        )
        reference = scaled_dot_product_attention(
            q_nope,
            k_nope,
            values,
            cache=[None, None],
            scale=scale,
            mask=pe_scores,
        )
        candidate = _fused_mla_prefill_attention(
            q_nope,
            q_pe,
            k_nope,
            k_pe,
            values,
            scale=scale,
            mask=mask,
        )
        mx.eval(reference, candidate)

        self.assertTrue(mx.allclose(reference, candidate, rtol=1e-5, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
