import json
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx

from mlx_lm.models.switch_layers import (
    SORTED_QMM_PACKED_CONFIG_KEY,
    _pack_mxfp4_scales_for_sorted_qmm,
    _pack_mxfp4_weight_for_sorted_qmm,
)
from mlx_lm.pack_sorted_qmm import pack_checkpoint


class TestPackSortedQMM(unittest.TestCase):
    def test_pack_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            destination = Path(directory) / "packed"
            source.mkdir()
            (source / "config.json").write_text(
                json.dumps({"model_type": "test"})
            )
            (source / "tokenizer.json").write_text("{}")
            weight = mx.arange(256 * 8, dtype=mx.uint32).reshape(
                1, 256, 8
            )
            scales = mx.arange(256 * 2, dtype=mx.uint8).reshape(
                1, 256, 2
            )
            untouched = mx.arange(8, dtype=mx.float32)
            mx.save_safetensors(
                str(source / "model.safetensors"),
                {
                    "model.switch_mlp.gate_proj.weight": weight,
                    "model.switch_mlp.gate_proj.scales": scales,
                    "model.norm.weight": untouched,
                },
            )

            metadata = pack_checkpoint(source, destination)
            packed = mx.load(str(destination / "model.safetensors"))
            config = json.loads((destination / "config.json").read_text())

            self.assertEqual(metadata["packed_tensor_count"], 2)
            self.assertEqual(
                config[SORTED_QMM_PACKED_CONFIG_KEY]["packed_tensor_count"], 2
            )
            self.assertTrue((destination / "tokenizer.json").is_file())
            self.assertTrue(
                mx.array_equal(
                    packed["model.switch_mlp.gate_proj.weight"],
                    _pack_mxfp4_weight_for_sorted_qmm(weight),
                )
            )
            self.assertTrue(
                mx.array_equal(
                    packed["model.switch_mlp.gate_proj.scales"],
                    _pack_mxfp4_scales_for_sorted_qmm(scales),
                )
            )
            self.assertTrue(
                mx.array_equal(packed["model.norm.weight"], untouched)
            )


if __name__ == "__main__":
    unittest.main()
