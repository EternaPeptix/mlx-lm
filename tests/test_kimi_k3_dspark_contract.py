from __future__ import annotations

import copy
import unittest

from mlx_lm.models.kimi_k3 import _validate_aux_hidden_state_layer_ids
from mlx_lm.models.kimi_k3_dspark import (
    KIMI_K3_DSPARK_INITIAL_VERIFY_WIDTH,
    KIMI_K3_DSPARK_INTERMEDIATE_VERIFY_WIDTH,
    KIMI_K3_DSPARK_SCREENING_VERIFY_WIDTH,
    RADIXARK_KIMI_K3_DSPARK_PARAMETERS,
    RADIXARK_KIMI_K3_DSPARK_TARGET_LAYERS,
    KimiK3DSparkContract,
)


def _official_config():
    return {
        "architectures": ["DSparkDraftModel"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "block_size": 7,
        "confidence_head_with_markov": True,
        "dtype": "bfloat16",
        "enable_confidence_head": True,
        "head_dim": 64,
        "hidden_act": "silu",
        "hidden_size": 7168,
        "intermediate_size": 14336,
        "layer_types": ["full_attention"] * 5,
        "markov_head_type": "vanilla",
        "markov_rank": 256,
        "max_position_embeddings": 1_048_576,
        "model_type": "qwen3",
        "num_attention_heads": 64,
        "num_hidden_layers": 5,
        "num_key_value_heads": 16,
        "num_target_layers": 93,
        "rms_norm_eps": 1e-5,
        "rope_parameters": {
            "rope_theta": 10_000.0,
            "rope_type": "default",
        },
        "tie_word_embeddings": False,
        "vocab_size": 163840,
        "dflash_config": {
            "mask_token_id": 163824,
            "target_layer_ids": [7, 23, 51, 67, 83],
        },
    }


class KimiK3DSparkContractTest(unittest.TestCase):
    def test_official_contract_uses_replicated_native_gamma_seven(self):
        contract = KimiK3DSparkContract.from_config(_official_config())

        self.assertEqual(contract.verify_width, KIMI_K3_DSPARK_INITIAL_VERIFY_WIDTH)
        self.assertEqual(contract.num_draft_tokens, 7)
        self.assertEqual(contract.maximum_verify_width, 8)
        self.assertEqual(
            contract.target_layer_ids, RADIXARK_KIMI_K3_DSPARK_TARGET_LAYERS
        )
        self.assertEqual(contract.parameter_count, RADIXARK_KIMI_K3_DSPARK_PARAMETERS)
        self.assertEqual(contract.placement, "replicated")
        self.assertFalse(contract.owns_embedding)
        self.assertFalse(contract.owns_lm_head)
        self.assertFalse(contract.screening_override)

    def test_screened_widths_require_explicit_override(self):
        for width in (
            KIMI_K3_DSPARK_SCREENING_VERIFY_WIDTH,
            KIMI_K3_DSPARK_INTERMEDIATE_VERIFY_WIDTH,
        ):
            with (
                self.subTest(width=width, override=False),
                self.assertRaisesRegex(ValueError, "screening override"),
            ):
                KimiK3DSparkContract.from_config(
                    _official_config(),
                    verify_width=width,
                )

            contract = KimiK3DSparkContract.from_config(
                _official_config(),
                verify_width=width,
                screening_override=True,
            )
            self.assertEqual(contract.verify_width, width)
            self.assertEqual(contract.num_draft_tokens, width - 1)
            self.assertTrue(contract.screening_override)

    def test_checkpoint_shape_mismatch_fails_closed(self):
        for field, invalid in (
            ("hidden_size", 4096),
            ("num_hidden_layers", 4),
            ("markov_rank", 0),
            ("block_size", 6),
            ("rms_norm_eps", 1e-6),
        ):
            with self.subTest(field=field):
                config = copy.deepcopy(_official_config())
                config[field] = invalid
                with self.assertRaisesRegex(ValueError, field):
                    KimiK3DSparkContract.from_config(config)

        config = copy.deepcopy(_official_config())
        config["hidden_size"] = 7168.0
        with self.assertRaisesRegex(ValueError, "hidden_size"):
            KimiK3DSparkContract.from_config(config)

    def test_attention_and_rope_mismatch_fail_closed(self):
        for field, invalid in (
            ("layer_types", ["sliding_attention"] * 5),
            ("rope_parameters", {"rope_theta": 1_000_000.0, "rope_type": "default"}),
        ):
            with self.subTest(field=field):
                config = copy.deepcopy(_official_config())
                config[field] = invalid
                with self.assertRaisesRegex(ValueError, "layer_types|RoPE"):
                    KimiK3DSparkContract.from_config(config)

    def test_hidden_tap_mismatch_fails_closed(self):
        config = copy.deepcopy(_official_config())
        config["dflash_config"]["target_layer_ids"][-1] = 82
        with self.assertRaisesRegex(ValueError, "hidden taps"):
            KimiK3DSparkContract.from_config(config)

    def test_width_and_placement_are_bounded(self):
        for width in (1, 9):
            with (
                self.subTest(width=width),
                self.assertRaisesRegex(ValueError, "verify width"),
            ):
                KimiK3DSparkContract.from_config(
                    _official_config(),
                    verify_width=width,
                )
        for width in (2, 5, 6, 7):
            with (
                self.subTest(width=width),
                self.assertRaisesRegex(ValueError, "screening override"),
            ):
                KimiK3DSparkContract.from_config(
                    _official_config(),
                    verify_width=width,
                    screening_override=True,
                )
        with self.assertRaisesRegex(ValueError, "replicated"):
            KimiK3DSparkContract.from_config(
                _official_config(),
                placement="sharded",
            )

    def test_target_hidden_tap_ids_are_unique_ordered_and_in_range(self):
        self.assertEqual(
            _validate_aux_hidden_state_layer_ids(
                RADIXARK_KIMI_K3_DSPARK_TARGET_LAYERS,
                93,
            ),
            RADIXARK_KIMI_K3_DSPARK_TARGET_LAYERS,
        )
        for layer_ids in ((7, 7), (23, 7), (-1, 7), (7, 93)):
            with (
                self.subTest(layer_ids=layer_ids),
                self.assertRaises((TypeError, ValueError)),
            ):
                _validate_aux_hidden_state_layer_ids(layer_ids, 93)


if __name__ == "__main__":
    unittest.main()
