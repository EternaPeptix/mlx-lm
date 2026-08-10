from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
import warnings
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten

from mlx_lm.models.kimi_k3 import VocabParallelHead
from mlx_lm.models.kimi_k3_dspark import (
    DSPARK_PROPOSER_ENV,
    DSPARK_STACKED_CONTEXT_KV_ENV,
    RADIXARK_KIMI_K3_DSPARK_CONFIG_BYTES,
    RADIXARK_KIMI_K3_DSPARK_CONFIG_SHA256,
    RADIXARK_KIMI_K3_DSPARK_PARAMETERS,
    RADIXARK_KIMI_K3_DSPARK_SNAPSHOT_BYTES,
    RADIXARK_KIMI_K3_DSPARK_SNAPSHOT_FILES,
    RADIXARK_KIMI_K3_DSPARK_WEIGHT_TENSORS,
    RADIXARK_KIMI_K3_DSPARK_WEIGHTS_BYTES,
    RADIXARK_KIMI_K3_DSPARK_WEIGHTS_SHA256,
    KimiK3DSparkArgs,
    KimiK3DSparkAttention,
    KimiK3DSparkCheckpointFile,
    KimiK3DSparkContextCache,
    KimiK3DSparkModel,
    KimiK3DSparkProposer,
    attest_kimi_k3_dspark_file,
    attest_kimi_k3_dspark_weights,
    kimi_k3_dspark_expected_weight_shapes,
    kimi_k3_dspark_parameter_count,
    kimi_k3_dspark_proposer_enabled,
    kimi_k3_dspark_stacked_context_kv_enabled,
)


def _tiny_args(*, block_size: int = 2) -> KimiK3DSparkArgs:
    return KimiK3DSparkArgs(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        vocab_size=16,
        rms_norm_eps=1e-5,
        rope_theta=10_000.0,
        max_position_embeddings=128,
        block_size=block_size,
        mask_token_id=15,
        target_layer_ids=(0, 1),
        markov_rank=4,
        weight_dtype=mx.float32,
    )


def _quantized_tiny_args() -> KimiK3DSparkArgs:
    return replace(
        _tiny_args(),
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=8,
        num_key_value_heads=2,
    )


def _flat_parameters(model: nn.Module) -> dict[str, mx.array]:
    return dict(tree_flatten(model.parameters()))


class KimiK3DSparkInventoryTest(unittest.TestCase):
    def test_production_inventory_is_exact(self):
        args = KimiK3DSparkArgs()
        shapes = kimi_k3_dspark_expected_weight_shapes(args)

        self.assertEqual(len(shapes), RADIXARK_KIMI_K3_DSPARK_WEIGHT_TENSORS)
        self.assertEqual(
            kimi_k3_dspark_parameter_count(args),
            RADIXARK_KIMI_K3_DSPARK_PARAMETERS,
        )
        self.assertEqual(shapes["fc.weight"], (7168, 35840))
        self.assertEqual(shapes["layers.4.self_attn.q_proj.weight"], (4096, 7168))
        self.assertEqual(shapes["layers.4.self_attn.k_proj.weight"], (1024, 7168))
        self.assertEqual(
            shapes["markov_head.markov_w1.weight"],
            (163840, 256),
        )
        self.assertEqual(shapes["confidence_head.proj.weight"], (1, 7424))

    def test_pinned_raw_file_and_snapshot_attestation_constants(self):
        self.assertEqual(RADIXARK_KIMI_K3_DSPARK_CONFIG_BYTES, 1_288)
        self.assertEqual(
            RADIXARK_KIMI_K3_DSPARK_CONFIG_SHA256,
            "6aed20890d95cd69cf2ec006d1f30506fbd4f3091d44ca8e8b93e9fc7d50928f",
        )
        self.assertEqual(RADIXARK_KIMI_K3_DSPARK_WEIGHTS_BYTES, 4_498_585_858)
        self.assertEqual(
            RADIXARK_KIMI_K3_DSPARK_WEIGHTS_SHA256,
            "29df0e8eafb81909f785df55cb352b90d6a1500c609b1d60526c1a62b4d42495",
        )
        self.assertEqual(RADIXARK_KIMI_K3_DSPARK_SNAPSHOT_FILES, 6)
        self.assertEqual(RADIXARK_KIMI_K3_DSPARK_SNAPSHOT_BYTES, 4_498_617_103)

    def test_tiny_model_keys_shapes_dtype_and_count_attest(self):
        args = _tiny_args()
        model = KimiK3DSparkModel(args)
        weights = _flat_parameters(model)
        expected = kimi_k3_dspark_expected_weight_shapes(args)

        self.assertEqual(set(weights), set(expected))
        self.assertEqual(
            {name: tuple(value.shape) for name, value in weights.items()},
            expected,
        )
        self.assertEqual(
            attest_kimi_k3_dspark_weights(weights, args),
            kimi_k3_dspark_parameter_count(args),
        )
        loaded = KimiK3DSparkModel(args)
        loaded.load_owned_weights(weights)
        loaded_weights = _flat_parameters(loaded)
        mx.eval(*weights.values(), *loaded_weights.values())
        self.assertTrue(
            all(
                bool(mx.array_equal(weights[name], loaded_weights[name]).item())
                for name in weights
            )
        )

        missing = dict(weights)
        missing.pop("norm.weight")
        with self.assertRaisesRegex(ValueError, "missing"):
            attest_kimi_k3_dspark_weights(missing, args)

        unexpected = dict(weights)
        unexpected["not_in_checkpoint"] = mx.zeros((1,))
        with self.assertRaisesRegex(ValueError, "unexpected"):
            attest_kimi_k3_dspark_weights(unexpected, args)

        wrong_shape = dict(weights)
        wrong_shape["norm.weight"] = mx.zeros((7,), dtype=mx.float32)
        with self.assertRaisesRegex(ValueError, "shape"):
            attest_kimi_k3_dspark_weights(wrong_shape, args)

        wrong_dtype = dict(weights)
        wrong_dtype["norm.weight"] = mx.zeros((8,), dtype=mx.float16)
        with self.assertRaisesRegex(ValueError, "dtype"):
            attest_kimi_k3_dspark_weights(wrong_dtype, args)

    def test_generic_file_attestation_checks_size_hash_and_symlink(self):
        payload = b"exact-dspark-test"
        manifest = KimiK3DSparkCheckpointFile(
            filename="payload.bin",
            size=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / manifest.filename
            path.write_bytes(payload)
            attest_kimi_k3_dspark_file(path, manifest)

            path.write_bytes(payload + b"!")
            with self.assertRaisesRegex(ValueError, "bytes"):
                attest_kimi_k3_dspark_file(path, manifest)


class KimiK3DSparkContextCacheTest(unittest.TestCase):
    @staticmethod
    def _context_chunk(start: int, length: int) -> tuple[mx.array, mx.array]:
        values = mx.arange(start, start + 6 * length, dtype=mx.float32)
        keys = values.reshape(1, length, 2, 3).transpose(0, 2, 1, 3)
        return keys, keys + 1_000

    def test_append_preserves_content_across_capacity_boundaries(self):
        cache = KimiK3DSparkContextCache(step=4, max_growth=8)
        expected_keys = []
        expected_values = []
        offset = 0

        for length, expected_capacity in ((3, 4), (1, 4), (1, 8), (3, 8), (5, 16)):
            keys, values = self._context_chunk(6 * offset, length)
            cache.append(keys, values)
            expected_keys.append(keys)
            expected_values.append(values)
            offset += length

            visible_keys = cache.keys
            visible_values = cache.values
            self.assertIsNotNone(visible_keys)
            self.assertIsNotNone(visible_values)
            assert visible_keys is not None and visible_values is not None
            mx.eval(visible_keys, visible_values)
            self.assertEqual(cache.length, offset)
            self.assertEqual(cache.capacity, expected_capacity)
            self.assertEqual(tuple(visible_keys.shape), (1, 2, offset, 3))
            self.assertTrue(
                bool(
                    mx.array_equal(
                        visible_keys,
                        mx.concatenate(expected_keys, axis=2),
                    ).item()
                )
            )
            self.assertTrue(
                bool(
                    mx.array_equal(
                        visible_values,
                        mx.concatenate(expected_values, axis=2),
                    ).item()
                )
            )

        self.assertEqual(cache.allocation_count, 3)
        self.assertEqual(cache.copied_tokens, 12)

    def test_capacity_hint_avoids_intermediate_history_copies(self):
        cache = KimiK3DSparkContextCache(
            capacity_hint=10,
            step=4,
            max_growth=8,
        )
        first = self._context_chunk(0, 2)
        second = self._context_chunk(12, 10)

        cache.append(*first)
        self.assertEqual(cache.capacity, 12)
        cache.append(*second)
        mx.eval(cache.keys, cache.values)

        self.assertEqual(cache.length, 12)
        self.assertEqual(cache.capacity, 12)
        self.assertEqual(cache.allocation_count, 1)
        self.assertEqual(cache.copied_tokens, 0)
        self.assertTrue(
            bool(
                mx.array_equal(
                    cache.keys,
                    mx.concatenate([first[0], second[0]], axis=2),
                ).item()
            )
        )

    def test_proposer_forwards_capacity_hint_to_every_draft_layer(self):
        model = KimiK3DSparkModel(_tiny_args())
        proposer = KimiK3DSparkProposer(model)
        context = proposer.make_context_cache(capacity_hint=512)
        taps = (
            mx.ones((1, 3, 8), dtype=mx.float32),
            mx.full((1, 3, 8), 0.5, dtype=mx.float32),
        )

        proposer.append_target_context(
            taps,
            0,
            context,
            use_stacked_context_kv=False,
        )
        mx.eval(
            *[cache.keys for cache in context],
            *[cache.values for cache in context],
        )

        self.assertEqual({cache.length for cache in context}, {3})
        self.assertEqual({cache.capacity for cache in context}, {512})
        self.assertEqual({cache.allocation_count for cache in context}, {1})
        self.assertEqual({cache.copied_tokens for cache in context}, {0})

    def test_split_and_single_append_have_identical_attention_semantics(self):
        args = _tiny_args()
        attention = KimiK3DSparkAttention(args)
        target_hidden = mx.arange(40, dtype=mx.float32).reshape(1, 5, 8) / 17
        noise_hidden = mx.arange(16, dtype=mx.float32).reshape(1, 2, 8) / 11
        single = KimiK3DSparkContextCache(step=4, max_growth=8)
        split = KimiK3DSparkContextCache(step=4, max_growth=8)

        single.append(*attention.project_context(target_hidden, offset=0))
        split.append(*attention.project_context(target_hidden[:, :2], offset=0))
        split.append(*attention.project_context(target_hidden[:, 2:], offset=2))
        single_output = attention(noise_hidden, block_offset=5, cache=single)
        split_output = attention(noise_hidden, block_offset=5, cache=split)
        mx.eval(single.keys, single.values, split.keys, split.values)
        mx.eval(single_output, split_output)

        self.assertEqual(single.length, split.length)
        self.assertTrue(bool(mx.array_equal(single.keys, split.keys).item()))
        self.assertTrue(bool(mx.array_equal(single.values, split.values).item()))
        self.assertTrue(
            bool(mx.allclose(single_output, split_output, rtol=0, atol=0).item())
        )

    def test_one_million_token_growth_has_bounded_slack_and_copy_cost(self):
        cache = KimiK3DSparkContextCache()
        total_tokens = 1_000_000
        capacity = 0
        copied_tokens = 0
        allocations = 0

        while capacity < total_tokens:
            required = capacity + 1
            if capacity:
                copied_tokens += capacity
            capacity = cache._planned_capacity(capacity, required)
            allocations += 1

        naive_copied_tokens = total_tokens * (total_tokens - 1) // 2
        self.assertLess(capacity - total_tokens, cache.max_growth)
        self.assertLess(copied_tokens, 8 * total_tokens)
        self.assertGreater(naive_copied_tokens // copied_tokens, 60_000)
        self.assertLess(allocations, 32)

    def test_capacity_contract_rejects_invalid_values(self):
        for kwargs in (
            {"capacity_hint": -1},
            {"step": 0},
            {"max_growth": 128},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    KimiK3DSparkContextCache(**kwargs)


class KimiK3DSparkModelTest(unittest.TestCase):
    def test_target_embedding_and_head_are_borrowed_not_owned(self):
        model = KimiK3DSparkModel(_tiny_args())
        before = _flat_parameters(model)
        embedding = nn.Embedding(16, 8)
        vocab_head = nn.Linear(8, 16, bias=False)

        self.assertIs(model.bind_target_modules(embedding, vocab_head), model)
        self.assertIs(model.target_embedding, embedding)
        self.assertIs(model.target_vocab_head, vocab_head)
        after = _flat_parameters(model)
        self.assertEqual(set(after), set(before))
        self.assertEqual(
            {name: id(value) for name, value in after.items()},
            {name: id(value) for name, value in before.items()},
        )
        self.assertFalse(any(name.startswith("_target") for name in after))

    def test_quantized_target_embedding_uses_logical_shape(self):
        args = _quantized_tiny_args()
        model = KimiK3DSparkModel(args)
        embedding = nn.QuantizedEmbedding(
            args.vocab_size,
            args.hidden_size,
            group_size=32,
            bits=2,
        )
        vocab_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

        self.assertEqual(tuple(embedding.weight.shape), (args.vocab_size, 2))
        self.assertEqual(
            (embedding.num_embeddings, embedding.dims),
            (args.vocab_size, args.hidden_size),
        )
        self.assertIs(model.bind_target_modules(embedding, vocab_head), model)

    def test_quantized_target_embedding_binds_with_vocab_parallel_head(self):
        args = _quantized_tiny_args()
        embedding = nn.QuantizedEmbedding(
            args.vocab_size,
            args.hidden_size,
            group_size=32,
            bits=2,
        )
        vocab_head = VocabParallelHead(
            nn.Linear(args.hidden_size, args.vocab_size, bias=False),
            mx.distributed.init(),
        )
        model = KimiK3DSparkModel(args)
        target = SimpleNamespace(
            language_model=SimpleNamespace(
                args=SimpleNamespace(tie_word_embeddings=False),
                model=SimpleNamespace(embed_tokens=embedding),
                lm_head=vocab_head,
            )
        )

        self.assertFalse(hasattr(vocab_head, "weight"))
        self.assertIs(model.bind_target(target), model)
        self.assertIs(model.target_embedding, embedding)
        self.assertIs(model.target_vocab_head, vocab_head)

    def test_target_embedding_shape_validation_rejects_dense_and_quantized(self):
        args = _quantized_tiny_args()
        vocab_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        invalid_embeddings = (
            nn.Embedding(args.vocab_size + 1, args.hidden_size),
            nn.Embedding(args.vocab_size, args.hidden_size + 1),
            nn.QuantizedEmbedding(
                args.vocab_size + 1,
                args.hidden_size,
                group_size=32,
                bits=2,
            ),
            nn.QuantizedEmbedding(
                args.vocab_size,
                args.hidden_size + 32,
                group_size=32,
                bits=2,
            ),
        )

        for embedding in invalid_embeddings:
            with self.subTest(embedding=type(embedding).__name__):
                with self.assertRaisesRegex(ValueError, "embedding shape"):
                    KimiK3DSparkModel(args).bind_target_modules(
                        embedding,
                        vocab_head,
                    )

    def test_target_head_shape_validation_is_unchanged(self):
        args = _quantized_tiny_args()
        embedding = nn.QuantizedEmbedding(
            args.vocab_size,
            args.hidden_size,
            group_size=32,
            bits=2,
        )
        wrong_head = nn.Linear(
            args.hidden_size,
            args.vocab_size + 1,
            bias=False,
        )

        with self.assertRaisesRegex(ValueError, "vocabulary head"):
            KimiK3DSparkModel(args).bind_target_modules(embedding, wrong_head)

    def test_reference_and_opt_in_stacked_context_projection_match(self):
        model = KimiK3DSparkModel(_tiny_args())
        reference = model.make_context_cache()
        stacked = model.make_context_cache()
        taps = (
            mx.arange(32, dtype=mx.float32).reshape(1, 4, 8) / 32,
            mx.arange(32, dtype=mx.float32).reshape(1, 4, 8) / 17,
        )

        model.append_target_context(
            taps,
            0,
            reference,
            use_stacked_context_kv=False,
        )
        model.append_target_context(
            taps,
            0,
            stacked,
            use_stacked_context_kv=True,
        )
        mx.eval(
            *[cache.keys for cache in reference],
            *[cache.values for cache in reference],
            *[cache.keys for cache in stacked],
            *[cache.values for cache in stacked],
        )

        for expected, actual in zip(reference, stacked, strict=True):
            self.assertTrue(
                bool(
                    mx.allclose(
                        expected.keys,
                        actual.keys,
                        rtol=1e-6,
                        atol=1e-6,
                    ).item()
                )
            )
            self.assertTrue(
                bool(
                    mx.allclose(
                        expected.values,
                        actual.values,
                        rtol=1e-6,
                        atol=1e-6,
                    ).item()
                )
            )

    def test_synthetic_end_to_end_native_proposal(self):
        args = _tiny_args()
        model = KimiK3DSparkModel(args)
        embedding = nn.Embedding(args.vocab_size, args.hidden_size)
        vocab_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        vocab_head.weight = mx.zeros_like(vocab_head.weight)
        model.bind_target_modules(embedding, vocab_head)

        w1 = [[0.0] * args.markov_rank for _ in range(args.vocab_size)]
        w1[1][0] = 1.0
        w1[2][1] = 1.0
        w2 = [[0.0] * args.markov_rank for _ in range(args.vocab_size)]
        w2[2][0] = 10.0
        w2[3][1] = 10.0
        model.markov_head.markov_w1.weight = mx.array(w1, dtype=mx.float32)
        model.markov_head.markov_w2.weight = mx.array(w2, dtype=mx.float32)

        proposer = KimiK3DSparkProposer(model)
        context = proposer.make_context_cache()
        taps = (
            mx.ones((1, 3, args.hidden_size), dtype=mx.float32),
            mx.full((1, 3, args.hidden_size), 0.5, dtype=mx.float32),
        )
        proposer.append_target_context(
            taps,
            0,
            context,
            use_stacked_context_kv=False,
        )
        with patch.dict(os.environ, {DSPARK_PROPOSER_ENV: "1"}):
            proposal = proposer.propose(1, context)
        mx.eval(
            proposal.tokens,
            proposal.base_logits,
            proposal.corrected_logits,
            proposal.confidence_logits,
        )

        self.assertEqual(proposal.tokens.tolist(), [[2, 3]])
        self.assertEqual(tuple(proposal.base_logits.shape), (1, 2, 16))
        self.assertEqual(tuple(proposal.corrected_logits.shape), (1, 2, 16))
        self.assertEqual(tuple(proposal.confidence_logits.shape), (1, 2))
        self.assertEqual(proposal.verify_width, 3)
        self.assertEqual(proposal.draft_block_width, 2)
        self.assertEqual(proposal.mode, "model_native")
        self.assertEqual({cache.length for cache in context}, {3})

    def test_gamma_seven_is_default_and_shorter_widths_are_screening_only(self):
        model = KimiK3DSparkModel(_tiny_args(block_size=7))
        native = KimiK3DSparkProposer(model)
        self.assertEqual(native.verify_width, 8)
        self.assertEqual(native.proposal_count, 7)
        self.assertEqual(native.mode, "model_native")

        for width in (3, 4):
            with (
                self.subTest(width=width, override=False),
                self.assertRaisesRegex(ValueError, "screening override"),
            ):
                KimiK3DSparkProposer(model, verify_width=width)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                screening = KimiK3DSparkProposer(
                    model,
                    verify_width=width,
                    screening_override=True,
                )
            self.assertEqual(screening.proposal_count, width - 1)
            self.assertEqual(screening.mode, f"width{width}_screening_override")
            self.assertEqual(len(caught), 1)
            self.assertIn("gamma=7", str(caught[0].message))

    def test_width_four_screening_proposes_three_tokens(self):
        args = _tiny_args(block_size=7)
        model = KimiK3DSparkModel(args)
        embedding = nn.Embedding(args.vocab_size, args.hidden_size)
        vocab_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        model.bind_target_modules(embedding, vocab_head)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            proposer = KimiK3DSparkProposer(
                model,
                verify_width=4,
                screening_override=True,
            )
        context = proposer.make_context_cache()
        taps = (
            mx.ones((1, 3, args.hidden_size), dtype=mx.float32),
            mx.full((1, 3, args.hidden_size), 0.5, dtype=mx.float32),
        )
        proposer.append_target_context(taps, 0, context, use_stacked_context_kv=False)
        with patch.dict(os.environ, {DSPARK_PROPOSER_ENV: "1"}):
            proposal = proposer.propose(1, context)
        mx.eval(
            proposal.tokens,
            proposal.base_logits,
            proposal.corrected_logits,
            proposal.confidence_logits,
        )

        self.assertEqual(tuple(proposal.tokens.shape), (1, 3))
        self.assertEqual(tuple(proposal.base_logits.shape), (1, 3, 16))
        self.assertEqual(tuple(proposal.corrected_logits.shape), (1, 3, 16))
        self.assertEqual(tuple(proposal.confidence_logits.shape), (1, 3))
        self.assertEqual(proposal.verify_width, 4)
        self.assertEqual(proposal.draft_block_width, 7)
        self.assertEqual(proposal.mode, "width4_screening_override")

    def test_feature_gates_are_strict_and_default_off(self):
        with patch.dict(os.environ):
            os.environ.pop(DSPARK_PROPOSER_ENV, None)
            os.environ.pop(DSPARK_STACKED_CONTEXT_KV_ENV, None)
            self.assertFalse(kimi_k3_dspark_proposer_enabled())
            self.assertFalse(kimi_k3_dspark_stacked_context_kv_enabled())
            proposer = KimiK3DSparkProposer(KimiK3DSparkModel(_tiny_args()))
            with self.assertRaisesRegex(RuntimeError, "disabled"):
                proposer.propose(0, [])

        with patch.dict(
            os.environ,
            {
                DSPARK_PROPOSER_ENV: "yes",
                DSPARK_STACKED_CONTEXT_KV_ENV: "true",
            },
        ):
            with self.assertRaisesRegex(ValueError, DSPARK_PROPOSER_ENV):
                kimi_k3_dspark_proposer_enabled()
            with self.assertRaisesRegex(ValueError, DSPARK_STACKED_CONTEXT_KV_ENV):
                kimi_k3_dspark_stacked_context_kv_enabled()


if __name__ == "__main__":
    unittest.main()
