# Copyright © 2026 Apple Inc.

import importlib
import unittest

import mlx.core as mx
import mlx.nn as nn


def _kimi_k3_config():
    return {
        "model_type": "kimi_k3",
        "vocab_size": 1024,
        "num_hidden_layers": 4,
        "text_config": {
            "model_type": "kimi_linear",
            "vocab_size": 1024,
            "hidden_size": 64,
            "num_hidden_layers": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "intermediate_size": 96,
            "rms_norm_eps": 1e-5,
            "hidden_act": "situ",
            "activation_situ_beta": 4.0,
            "activation_situ_linear_beta": 25.0,
            "linear_attn_config": {
                "kda_layers": [1, 2, 3],
                "full_attn_layers": [4],
                "num_heads": 2,
                "head_dim": 32,
                "short_conv_kernel_size": 4,
                "gate_lower_bound": -5.0,
                "use_full_rank_gate": True,
            },
            "num_experts": 8,
            "moe_intermediate_size": 32,
            "q_lora_rank": 24,
            "kv_lora_rank": 16,
            "qk_nope_head_dim": 16,
            "qk_rope_head_dim": 8,
            "v_head_dim": 16,
            "mla_use_nope": True,
            "mla_use_output_gate": True,
            "num_experts_per_token": 2,
            "num_shared_experts": 1,
            "first_k_dense_replace": 1,
            "routed_expert_hidden_size": 32,
            "latent_moe_use_norm": True,
            "attn_res_block_size": 2,
        },
    }


class TestModelParallel(unittest.TestCase):
    def test_shard(self):
        test_configs = [
            {
                "model_type": "deepseek_v3",
                "vocab_size": 1024,
                "hidden_size": 128,
                "intermediate_size": 256,
                "moe_intermediate_size": 256,
                "num_hidden_layers": 4,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "n_routed_experts": 4,
                "n_group": 2,
                "topk_group": 1,
                "num_experts_per_tok": 2,
                "n_shared_experts": 1,
                "kv_lora_rank": 4,
                "q_lora_rank": 4,
                "qk_rope_head_dim": 32,
                "v_head_dim": 16,
                "qk_nope_head_dim": 32,
                "rope_scaling": {
                    "beta_fast": 32,
                    "beta_slow": 1,
                    "factor": 40,
                    "mscale": 1.0,
                    "mscale_all_dim": 1.0,
                    "original_max_position_embeddings": 4096,
                    "type": "yarn",
                },
            },
            {
                "model_type": "llama",
                "hidden_size": 64,
                "num_hidden_layers": 4,
                "intermediate_size": 256,
                "num_attention_heads": 8,
                "num_key_value_heads": 4,
                "rms_norm_eps": 1e-5,
                "vocab_size": 128,
                "sliding_window": 4,
                "layer_types": [
                    "full_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "full_attention",
                ],
                "tie_word_embeddings": False,
                "rope_theta": 10000.0,
            },
            {
                "model_type": "glm4_moe_lite",
                "vocab_size": 1000,
                "hidden_size": 64,
                "intermediate_size": 128,
                "moe_intermediate_size": 32,
                "num_hidden_layers": 4,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "n_shared_experts": 1,
                "n_routed_experts": 4,
                "routed_scaling_factor": 1.0,
                "kv_lora_rank": 8,
                "q_lora_rank": 8,
                "qk_rope_head_dim": 8,
                "qk_nope_head_dim": 16,
                "v_head_dim": 8,
                "topk_method": "noaux_tc",
                "scoring_func": "sigmoid",
                "norm_topk_prob": True,
                "n_group": 1,
                "topk_group": 1,
                "num_experts_per_tok": 2,
                "moe_layer_freq": 1,
                "first_k_dense_replace": 1,
                "max_position_embeddings": 256,
                "rms_norm_eps": 1e-5,
                "rope_theta": 1000,
                "rope_scaling": None,
                "attention_bias": False,
                "partial_rotary_factor": 1.0,
                "tie_word_embeddings": False,
                "num_nextn_predict_layers": 1,
            },
        ]
        mx.random.seed(0)
        for config in test_configs:
            model_type = config["model_type"]
            with self.subTest(f"Testing {model_type}", model_type=model_type):
                arch = importlib.import_module(f"mlx_lm.models.{model_type}")
                args = arch.ModelArgs.from_dict(config)
                model = arch.Model(args)
                vocab_size = args.vocab_size
                x = mx.random.randint(0, vocab_size, shape=(32, 4))
                expected = model(x)
                model.shard()
                out = model(x)
                self.assertTrue(mx.allclose(expected, out, rtol=1e-3, atol=1e-3))

    def test_kimi_k3_vocab_parallel_head(self):
        group = mx.distributed.init()
        if group.size() == 1:
            self.skipTest("requires mlx.launch with at least two ranks")

        from mlx_lm.models import kimi_k3

        for quantized in (False, True):
            with self.subTest(quantized=quantized):
                mx.random.seed(0)
                args = kimi_k3.ModelArgs.from_dict(_kimi_k3_config())
                model = kimi_k3.Model(args)
                if quantized:
                    nn.quantize(
                        model.language_model.lm_head,
                        group_size=64,
                        bits=4,
                    )
                x = mx.random.randint(
                    0,
                    args.text_config.vocab_size,
                    shape=(2, 4),
                )
                expected = model(x)
                mx.eval(expected)
                expected_tokens = mx.argmax(expected[:, -1, :], axis=-1)

                model.shard_vocab_head(group)
                actual = model(x)
                actual_tokens = model.vocab_parallel_greedy(x)
                mx.eval(actual, expected_tokens, actual_tokens)

                self.assertEqual(actual.shape, expected.shape)
                self.assertTrue(mx.allclose(expected, actual, rtol=1e-3, atol=1e-3))
                self.assertTrue(mx.array_equal(expected_tokens, actual_tokens))

                wrapped = model.language_model.lm_head
                model.shard_vocab_head(group)
                self.assertIs(model.language_model.lm_head, wrapped)

                if not quantized:
                    wrapped.local_head.weight = mx.zeros_like(wrapped.local_head.weight)
                    tied = model.vocab_parallel_greedy(x)
                    mx.eval(tied)
                    self.assertEqual(tied.tolist(), [0, 0])

    def test_kimi_k3_speculative_cache_tp2_width_two_and_eight(self):
        group = mx.distributed.init()
        if group.size() != 2:
            self.skipTest("requires mlx.launch with exactly two ranks")

        from mlx_lm.models import kimi_k3
        from mlx_lm.models.cache import ArraysCache

        mx.random.seed(7)
        args = kimi_k3.ModelArgs.from_dict(_kimi_k3_config())
        model = kimi_k3.Model(args)
        model.set_dtype(mx.bfloat16)
        model.eval()
        model.shard(group)

        prefix = mx.array([[1, 2]], dtype=mx.uint32)

        def populated_cache():
            prompt_cache = model.make_cache()
            logits = model(prefix, cache=prompt_cache)
            mx.eval(logits, [entry.state for entry in prompt_cache])
            return prompt_cache

        def assert_cache_close(left, right):
            self.assertEqual(len(left), len(right))
            for left_entry, right_entry in zip(left, right, strict=True):
                self.assertIs(type(left_entry), type(right_entry))
                if isinstance(left_entry, ArraysCache):
                    for left_state, right_state in zip(
                        left_entry.cache,
                        right_entry.cache,
                        strict=True,
                    ):
                        self.assertTrue(
                            mx.allclose(
                                left_state,
                                right_state,
                                rtol=1e-3,
                                atol=1e-3,
                            )
                        )
                else:
                    self.assertEqual(left_entry.offset, right_entry.offset)
                    for left_state, right_state in zip(
                        left_entry.state,
                        right_entry.state,
                        strict=True,
                    ):
                        self.assertTrue(
                            mx.allclose(
                                left_state,
                                right_state,
                                rtol=1e-3,
                                atol=1e-3,
                            )
                        )

        for width, consumed in ((2, 1), (8, 8)):
            with self.subTest(width=width, consumed=consumed):
                wide_cache = populated_cache()
                sequential_cache = populated_cache()
                tokens = mx.arange(width, dtype=mx.uint32)[None] + mx.array(
                    [[3]], dtype=mx.uint32
                )

                transaction = model.begin_speculative_cache(wide_cache, width)
                wide_logits = model(tokens, cache=wide_cache)
                mx.eval(wide_logits)
                model.resolve_speculative_cache(transaction, consumed)

                sequential_logits = []
                for position in range(consumed):
                    logits = model(
                        tokens[:, position : position + 1],
                        cache=sequential_cache,
                    )
                    mx.eval(logits, [entry.state for entry in sequential_cache])
                    sequential_logits.append(logits)
                sequential_logits = mx.concatenate(sequential_logits, axis=1)

                logits_max_abs = mx.max(
                    mx.abs(wide_logits[:, :consumed] - sequential_logits)
                ).item()
                self.assertTrue(
                    mx.allclose(
                        wide_logits[:, :consumed],
                        sequential_logits,
                        rtol=2e-2,
                        atol=2e-2,
                    ),
                    f"speculative logits max abs diff: {logits_max_abs}",
                )
                assert_cache_close(wide_cache, sequential_cache)

                next_token = mx.array([[17]], dtype=mx.uint32)
                wide_next = model(next_token, cache=wide_cache)
                sequential_next = model(next_token, cache=sequential_cache)
                mx.eval(
                    wide_next,
                    sequential_next,
                    [entry.state for entry in wide_cache],
                    [entry.state for entry in sequential_cache],
                )
                self.assertTrue(
                    mx.allclose(
                        wide_next,
                        sequential_next,
                        rtol=1e-3,
                        atol=1e-3,
                    )
                )
                assert_cache_close(wide_cache, sequential_cache)


if __name__ == "__main__":
    unittest.main()
