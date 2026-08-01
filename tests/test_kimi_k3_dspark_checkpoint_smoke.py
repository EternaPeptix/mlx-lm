from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


_BENCHMARK = (
    Path(__file__).parents[1] / "benchmarks" / "kimi_k3_dspark_checkpoint_smoke.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "kimi_k3_dspark_checkpoint_smoke", _BENCHMARK
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def _safetensors_bytes(
    shapes: dict[str, tuple[int, ...]],
    *,
    dtype: str = "BF16",
) -> bytes:
    header = {}
    offset = 0
    for name, shape in shapes.items():
        elements = 1
        for dimension in shape:
            elements *= dimension
        size = elements * 2
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + size],
        }
        offset += size
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * ((-len(encoded)) % 8)
    return len(encoded).to_bytes(8, "little") + encoded + bytes(offset)


def _validate_tiny_config(config):
    if config != {"kind": "tiny-dspark"}:
        raise ValueError("tiny config mismatch")


def _tiny_checkpoint(directory: Path):
    shapes = {"alpha": (2, 3), "beta": (1,)}
    config_payload = b'{"kind":"tiny-dspark"}\n'
    model_payload = _safetensors_bytes(shapes)
    (directory / "config.json").write_bytes(config_payload)
    (directory / "model.safetensors").write_bytes(model_payload)
    contract = _MODULE.CheckpointContract(
        model_id="test/tiny-dspark",
        revision=_MODULE.PINNED_REVISION,
        config_bytes=len(config_payload),
        config_sha256=hashlib.sha256(config_payload).hexdigest(),
        model_bytes=len(model_payload),
        model_sha256=hashlib.sha256(model_payload).hexdigest(),
        expected_shapes=shapes,
        expected_dtype="BF16",
        expected_elements=7,
        config_validator=_validate_tiny_config,
    )
    return contract


class KimiK3DSparkMetadataSmokeTest(unittest.TestCase):
    def test_pure_python_contract_matches_local_runtime_source(self):
        source_path = (
            Path(__file__).parents[1]
            / "mlx_lm"
            / "models"
            / "kimi_k3_dspark.py"
        )
        source_tree = ast.parse(source_path.read_text())
        source_constants = {}
        for node in source_tree.body:
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            ):
                try:
                    source_constants[node.targets[0].id] = ast.literal_eval(
                        node.value
                    )
                except (ValueError, TypeError):
                    pass
        constant_pairs = {
            "RADIXARK_KIMI_K3_DSPARK_MODEL": _MODULE.MODEL_ID,
            "RADIXARK_KIMI_K3_DSPARK_REVISION": _MODULE.PINNED_REVISION,
            "RADIXARK_KIMI_K3_DSPARK_CONFIG_SHA256": _MODULE.CONFIG_SHA256,
            "RADIXARK_KIMI_K3_DSPARK_WEIGHTS_SHA256": _MODULE.MODEL_SHA256,
            "RADIXARK_KIMI_K3_DSPARK_CONFIG_BYTES": _MODULE.CONFIG_BYTES,
            "RADIXARK_KIMI_K3_DSPARK_WEIGHTS_BYTES": _MODULE.MODEL_BYTES,
            "RADIXARK_KIMI_K3_DSPARK_WEIGHT_TENSORS": _MODULE.WEIGHT_TENSORS,
            "RADIXARK_KIMI_K3_DSPARK_PARAMETERS": _MODULE.PARAMETERS,
            "RADIXARK_KIMI_K3_DSPARK_TARGET_LAYERS": _MODULE.TARGET_LAYERS,
            "RADIXARK_KIMI_K3_DSPARK_BLOCK_SIZE": _MODULE.BLOCK_SIZE,
        }
        for name, expected in constant_pairs.items():
            self.assertEqual(source_constants[name], expected, name)

        shape_function = next(
            node
            for node in source_tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "kimi_k3_dspark_expected_weight_shapes"
        )
        extracted = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__",
                    names=[ast.alias(name="annotations")],
                    level=0,
                ),
                shape_function,
            ],
            type_ignores=[],
        )
        ast.fix_missing_locations(extracted)
        namespace = {}
        exec(compile(extracted, str(source_path), "exec"), namespace)
        args = SimpleNamespace(
            hidden_size=7168,
            intermediate_size=14336,
            num_hidden_layers=5,
            num_attention_heads=64,
            num_key_value_heads=16,
            head_dim=64,
            vocab_size=163840,
            target_layer_ids=_MODULE.TARGET_LAYERS,
            markov_rank=256,
        )
        source_shapes = namespace["kimi_k3_dspark_expected_weight_shapes"](args)
        self.assertEqual(source_shapes, _MODULE.production_weight_shapes())

    def test_production_inventory_is_exact_without_importing_mlx(self):
        shapes = _MODULE.production_weight_shapes()

        self.assertEqual(len(shapes), _MODULE.WEIGHT_TENSORS)
        self.assertEqual(
            sum(map(_MODULE._shape_size, shapes.values())), _MODULE.PARAMETERS
        )
        self.assertEqual(shapes["fc.weight"], (7168, 35840))
        self.assertEqual(
            shapes["layers.4.self_attn.k_proj.weight"], (1024, 7168)
        )

    def test_default_mode_reads_header_and_never_enters_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            contract = _tiny_checkpoint(directory)
            with patch.object(
                _MODULE,
                "_load_runtime",
                side_effect=AssertionError("metadata mode imported MLX"),
            ):
                result = _MODULE.run_checkpoint_smoke(
                    directory,
                    contract=contract,
                )

        self.assertTrue(result["ok"])
        self.assertEqual(result["mode"], "metadata")
        self.assertFalse(result["metadata_only_allocates_tensors"])
        metadata = result["checkpoint"]["safetensors"]
        self.assertEqual(metadata["tensor_count"], 2)
        self.assertEqual(metadata["element_count"], 7)
        self.assertEqual(metadata["tensors"]["alpha"]["shape"], [2, 3])
        self.assertFalse(result["checkpoint"]["model_file"]["sha256_verified"])

    def test_shape_dtype_and_payload_layout_mismatches_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            contract = _tiny_checkpoint(directory)
            path = directory / "model.safetensors"

            bad_shape = dict(contract.expected_shapes)
            bad_shape["alpha"] = (3, 2)
            with self.assertRaisesRegex(_MODULE.SmokeError, "shape"):
                _MODULE.inspect_safetensors_metadata(
                    path,
                    bad_shape,
                    expected_dtype="BF16",
                    expected_elements=7,
                )

            path.write_bytes(
                _safetensors_bytes(
                    dict(contract.expected_shapes),
                    dtype="F16",
                )
            )
            with self.assertRaisesRegex(_MODULE.SmokeError, "dtype"):
                _MODULE.inspect_safetensors_metadata(
                    path,
                    contract.expected_shapes,
                    expected_dtype="BF16",
                    expected_elements=7,
                )

            payload = bytearray(_safetensors_bytes(dict(contract.expected_shapes)))
            payload.extend(b"\x00\x00")
            path.write_bytes(payload)
            with self.assertRaisesRegex(_MODULE.SmokeError, "coverage"):
                _MODULE.inspect_safetensors_metadata(
                    path,
                    contract.expected_shapes,
                    expected_dtype="BF16",
                    expected_elements=7,
                )

    def test_model_sha_is_optional_but_catches_same_size_payload_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            contract = _tiny_checkpoint(directory)
            model_path = directory / "model.safetensors"
            payload = bytearray(model_path.read_bytes())
            payload[-1] ^= 0xFF
            model_path.write_bytes(payload)

            report = _MODULE.inspect_checkpoint(
                directory,
                revision=contract.revision,
                verify_model_sha256=False,
                contract=contract,
            )
            self.assertFalse(report["model_file"]["sha256_verified"])
            with self.assertRaisesRegex(_MODULE.SmokeError, "SHA256"):
                _MODULE.inspect_checkpoint(
                    directory,
                    revision=contract.revision,
                    verify_model_sha256=True,
                    contract=contract,
                )

    def test_snapshot_revision_and_synthetic_load_contracts_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "snapshots" / ("0" * 40)
            directory.mkdir(parents=True)
            contract = _tiny_checkpoint(directory)
            with self.assertRaisesRegex(_MODULE.SmokeError, "snapshot path revision"):
                _MODULE.inspect_checkpoint(
                    directory,
                    revision=contract.revision,
                    verify_model_sha256=False,
                    contract=contract,
                )
        with self.assertRaisesRegex(_MODULE.SmokeError, "requires explicit --load"):
            _MODULE.run_checkpoint_smoke(
                "/does/not/matter",
                synthetic_forward=True,
            )

    def test_cli_writes_json_to_stdout_and_an_atomic_output_path(self):
        success = {
            "schema_version": 1,
            "ok": True,
            "mode": "metadata",
        }
        stdout = StringIO()
        with (
            patch.object(_MODULE, "run_checkpoint_smoke", return_value=success),
            redirect_stdout(stdout),
        ):
            self.assertEqual(_MODULE.main(["/local/checkpoint"]), 0)
        self.assertEqual(json.loads(stdout.getvalue()), success)

        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "reports" / "smoke.json"
            _MODULE._write_json(success, str(destination), pretty=True)
            self.assertEqual(json.loads(destination.read_text()), success)
            self.assertEqual(list(destination.parent.glob(".*.tmp-*")), [])

    def test_cli_failure_is_json_and_nonzero(self):
        stdout = StringIO()
        with (
            patch.object(
                _MODULE,
                "run_checkpoint_smoke",
                side_effect=_MODULE.SmokeError("fail closed"),
            ),
            redirect_stdout(stdout),
        ):
            self.assertEqual(_MODULE.main(["/local/checkpoint"]), 1)
        payload = json.loads(stdout.getvalue())
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error_type"], "SmokeError")
        self.assertEqual(payload["error"], "fail closed")


class _FakeMX:
    bfloat16 = "bf16"

    def __init__(self):
        self.active = 11
        self.baseline = self.active
        self.peak = 22
        self.clear_calls = 0
        self.eval_calls = 0

    def clear_cache(self):
        self.clear_calls += 1
        if self.clear_calls >= 2:
            self.active = self.baseline

    def synchronize(self):
        pass

    def get_active_memory(self):
        return self.active

    def get_peak_memory(self):
        return self.peak

    def reset_peak_memory(self):
        self.peak = self.active

    def eval(self, *_):
        self.eval_calls += 1


class _FakeDrafter:
    def __init__(self):
        self.args = SimpleNamespace()
        self._parameters = {
            name: SimpleNamespace(shape=shape, dtype="bf16")
            for name, shape in _MODULE.production_weight_shapes().items()
        }

    def parameters(self):
        return self._parameters


class KimiK3DSparkLoadSmokeTest(unittest.TestCase):
    def test_explicit_load_uses_exact_loader_evaluates_and_releases(self):
        fake_mx = _FakeMX()
        drafter = _FakeDrafter()
        loader_calls = []

        def load(checkpoint_dir, target, *, verify_weights_sha256):
            loader_calls.append((checkpoint_dir, target, verify_weights_sha256))
            fake_mx.active = 4_500_000_000
            fake_mx.peak = 4_600_000_000
            return drafter

        dspark = SimpleNamespace(
            RADIXARK_KIMI_K3_DSPARK_MODEL=_MODULE.MODEL_ID,
            RADIXARK_KIMI_K3_DSPARK_REVISION=_MODULE.PINNED_REVISION,
            RADIXARK_KIMI_K3_DSPARK_CONFIG_SHA256=_MODULE.CONFIG_SHA256,
            RADIXARK_KIMI_K3_DSPARK_WEIGHTS_SHA256=_MODULE.MODEL_SHA256,
            RADIXARK_KIMI_K3_DSPARK_CONFIG_BYTES=_MODULE.CONFIG_BYTES,
            RADIXARK_KIMI_K3_DSPARK_WEIGHTS_BYTES=_MODULE.MODEL_BYTES,
            RADIXARK_KIMI_K3_DSPARK_WEIGHT_TENSORS=_MODULE.WEIGHT_TENSORS,
            RADIXARK_KIMI_K3_DSPARK_PARAMETERS=_MODULE.PARAMETERS,
            DSPARK_PROPOSER_ENV="TEST_DSPARK_PROPOSER",
            KimiK3DSparkArgs=lambda: object(),
            kimi_k3_dspark_expected_weight_shapes=(
                lambda _: _MODULE.production_weight_shapes()
            ),
            load_kimi_k3_dspark=load,
            attest_kimi_k3_dspark_weights=lambda weights, args: _MODULE.PARAMETERS,
        )
        tree_flatten = lambda parameters: list(parameters.items())

        with (
            patch.object(
                _MODULE,
                "_load_runtime",
                return_value=(fake_mx, tree_flatten, dspark),
            ),
            patch.dict(os.environ, {}, clear=False),
        ):
            os.environ.pop(dspark.DSPARK_PROPOSER_ENV, None)
            report, synthetic = _MODULE.run_load_smoke(
                Path("/local/checkpoint"),
                verify_model_sha256=True,
                synthetic_forward=False,
                synthetic_gamma=7,
            )
            self.assertNotIn(dspark.DSPARK_PROPOSER_ENV, os.environ)

        self.assertIsNone(synthetic)
        self.assertEqual(len(loader_calls), 1)
        self.assertTrue(loader_calls[0][2])
        target_stub = loader_calls[0][1]
        self.assertEqual(
            target_stub.language_model.model.embed_tokens.weight.shape,
            (163840, 7168),
        )
        self.assertNotIn("mlx", type(target_stub).__module__)
        self.assertEqual(report["tensor_count"], _MODULE.WEIGHT_TENSORS)
        self.assertEqual(report["element_count"], _MODULE.PARAMETERS)
        self.assertTrue(report["release"]["python_model_references_dropped"])
        self.assertTrue(report["release"]["active_memory_returned_to_baseline"])
        self.assertGreaterEqual(fake_mx.clear_calls, 2)
        self.assertGreaterEqual(fake_mx.eval_calls, 2)


if __name__ == "__main__":
    unittest.main()
