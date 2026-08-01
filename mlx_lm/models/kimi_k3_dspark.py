"""Fail-closed deployment contract for the RadixArk Kimi K3 DSpark draft."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

RADIXARK_KIMI_K3_DSPARK_MODEL = "RadixArk/Kimi-K3-DSpark"
RADIXARK_KIMI_K3_DSPARK_REVISION = "eb03982e58d4fb79bcfc099e902158f562e2e27b"
RADIXARK_KIMI_K3_DSPARK_CONFIG_SHA256 = (
    "410dd228c75ff91b57af8a1581d44d2ea096d5604f0d37fbd400470e90d961d3"
)
RADIXARK_KIMI_K3_DSPARK_PARAMETERS = 2_249_289_601
RADIXARK_KIMI_K3_DSPARK_TARGET_LAYERS = (7, 23, 51, 67, 83)
RADIXARK_KIMI_K3_DSPARK_BLOCK_SIZE = 7
KIMI_K3_DSPARK_INITIAL_VERIFY_WIDTH = 3


@dataclass(frozen=True)
class KimiK3DSparkContract:
    """Validated checkpoint and placement contract for replicated TP drafting."""

    target_layer_ids: tuple[int, ...]
    verify_width: int
    checkpoint_block_size: int
    parameter_count: int
    placement: str
    owns_embedding: bool
    owns_lm_head: bool

    @property
    def num_draft_tokens(self) -> int:
        return self.verify_width - 1

    @property
    def maximum_verify_width(self) -> int:
        return self.checkpoint_block_size + 1

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        verify_width: int = KIMI_K3_DSPARK_INITIAL_VERIFY_WIDTH,
        placement: str = "replicated",
    ) -> "KimiK3DSparkContract":
        """Validate the public 2.249B checkpoint before allocating weights."""

        architectures = config.get("architectures")
        if architectures != ["DSparkDraftModel"]:
            raise ValueError("Kimi K3 DSpark architecture contract does not match")

        expected_scalars = {
            "model_type": "qwen3",
            "block_size": RADIXARK_KIMI_K3_DSPARK_BLOCK_SIZE,
            "hidden_size": 7168,
            "intermediate_size": 14336,
            "num_hidden_layers": 5,
            "num_attention_heads": 64,
            "num_key_value_heads": 16,
            "head_dim": 64,
            "num_target_layers": 93,
            "vocab_size": 163840,
            "markov_rank": 256,
            "markov_head_type": "vanilla",
            "dtype": "bfloat16",
            "hidden_act": "silu",
            "rms_norm_eps": 1e-5,
            "max_position_embeddings": 1_048_576,
            "attention_bias": False,
            "attention_dropout": 0.0,
            "enable_confidence_head": True,
            "confidence_head_with_markov": True,
            "tie_word_embeddings": False,
        }
        for name, expected in expected_scalars.items():
            actual = config.get(name)
            if type(actual) is not type(expected) or actual != expected:
                raise ValueError(
                    f"Kimi K3 DSpark {name} must be {expected!r}, got {actual!r}"
                )

        if config.get("layer_types") != ["full_attention"] * 5:
            raise ValueError("Kimi K3 DSpark layer_types contract does not match")
        if config.get("rope_parameters") != {
            "rope_theta": 10_000.0,
            "rope_type": "default",
        }:
            raise ValueError("Kimi K3 DSpark RoPE contract does not match")

        dflash = config.get("dflash_config")
        if not isinstance(dflash, Mapping):
            raise ValueError("Kimi K3 DSpark dflash_config is missing")
        target_layer_ids = tuple(dflash.get("target_layer_ids", ()))
        if target_layer_ids != RADIXARK_KIMI_K3_DSPARK_TARGET_LAYERS:
            raise ValueError("Kimi K3 DSpark target hidden taps do not match")
        mask_token_id = dflash.get("mask_token_id")
        if type(mask_token_id) is not int or mask_token_id != 163824:
            raise ValueError("Kimi K3 DSpark mask token does not match")

        block_size = int(config["block_size"])
        if type(verify_width) is not int or not 2 <= verify_width <= block_size + 1:
            raise ValueError(
                f"Kimi K3 DSpark verify width must be in [2, {block_size + 1}]"
            )
        if placement != "replicated":
            raise ValueError(
                "Kimi K3 DSpark must be replicated on every target TP rank"
            )

        return cls(
            target_layer_ids=target_layer_ids,
            verify_width=verify_width,
            checkpoint_block_size=block_size,
            parameter_count=RADIXARK_KIMI_K3_DSPARK_PARAMETERS,
            placement=placement,
            owns_embedding=False,
            owns_lm_head=False,
        )
