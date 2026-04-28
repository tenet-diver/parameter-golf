import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from fastest.scripts.artifact_budget_analyzer import estimate_artifact_budget


class Fast4ArtifactBudgetAnalyzerTest(unittest.TestCase):
    def _baseline_config(self) -> dict:
        return {
            "vocab_size": 1024,
            "num_layers": 9,
            "model_dim": 512,
            "num_heads": 8,
            "num_kv_heads": 4,
            "mlp_mult": 2,
            "tie_embeddings": True,
        }

    def test_baseline_estimate_breaks_down_major_components(self) -> None:
        estimate = estimate_artifact_budget(self._baseline_config())

        components = {component.name: component for component in estimate.components}
        self.assertFalse(estimate.over_budget)
        self.assertGreater(estimate.headroom_bytes, 0)
        self.assertIn("embeddings", components)
        self.assertIn("attention_blocks", components)
        self.assertIn("mlp_blocks", components)
        self.assertIn("control_tensors", components)
        self.assertGreater(components["attention_blocks"].estimated_bytes, 0)
        self.assertGreater(components["mlp_blocks"].estimated_bytes, 0)

    def test_estimate_exposes_heads_and_quantization_overhead_for_packing_decisions(self) -> None:
        estimate = estimate_artifact_budget(
            {
                **self._baseline_config(),
                "tie_embeddings": False,
                "quantization_bits": 6,
                "quantization_scheme": "int6-per-row-zlib-projection",
            }
        )

        components = {component.name: component for component in estimate.components}

        self.assertEqual(estimate.quantization_scheme, "int6-per-row-zlib-projection")
        self.assertIn("output_head", components)
        self.assertIn("quantization_overhead", components)
        self.assertGreater(components["output_head"].parameter_count, 0)
        self.assertGreater(components["quantization_overhead"].raw_quantized_bytes, 0)
        self.assertIn("quantization_overhead", "\n".join(estimate.to_log_lines()))

    def test_large_candidate_flags_16mb_violation_before_training(self) -> None:
        config = {
            **self._baseline_config(),
            "vocab_size": 8192,
            "num_layers": 15,
            "model_dim": 768,
            "num_kv_heads": 8,
            "mlp_mult": 4,
            "tie_embeddings": False,
        }

        estimate = estimate_artifact_budget(config)

        self.assertTrue(estimate.over_budget)
        self.assertLess(estimate.headroom_bytes, 0)
        self.assertIn("output_head", {component.name for component in estimate.components})

    def test_code_bytes_are_included_when_script_path_is_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            code_path = Path(tmp_dir) / "train.py"
            code_path.write_text("print('candidate')\n")

            estimate = estimate_artifact_budget(self._baseline_config(), code_path=code_path)

            self.assertGreater(estimate.code_estimated_bytes, 0)
            self.assertEqual(
                estimate.total_estimated_bytes,
                estimate.model_estimated_bytes + estimate.code_estimated_bytes,
            )

    def test_invalid_attention_shape_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            estimate_artifact_budget({**self._baseline_config(), "model_dim": 513})

    def test_invalid_numeric_config_values_fail_closed(self) -> None:
        invalid_configs = [
            {**self._baseline_config(), "num_layers": "many"},
            {**self._baseline_config(), "quantization_bits": True},
        ]

        for config in invalid_configs:
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    estimate_artifact_budget(config)

    def test_cli_flags_over_budget_candidate_before_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "candidate.json"
            config_path.write_text(
                json.dumps(
                    {
                        **self._baseline_config(),
                        "vocab_size": 8192,
                        "num_layers": 15,
                        "model_dim": 768,
                        "num_kv_heads": 8,
                        "mlp_mult": 4,
                        "tie_embeddings": False,
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "fastest/scripts/artifact_budget_analyzer.py",
                    "--config",
                    str(config_path),
                    "--format",
                    "json",
                ],
                check=False,
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
            )

        self.assertEqual(result.returncode, 2, result.stderr)
        payload = json.loads(result.stdout)
        self.assertTrue(payload["overBudget"])
        self.assertLess(payload["headroomBytes"], 0)
        self.assertEqual(payload["limitBytes"], 16_000_000)
        self.assertIn("attention_blocks", {component["name"] for component in payload["components"]})


if __name__ == "__main__":
    unittest.main()
