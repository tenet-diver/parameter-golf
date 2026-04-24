import json
import tempfile
import unittest
from pathlib import Path

from measurement.control_pipeline import publishEvidenceBundle, submitControlRun
from measurement.regression_gate import validateCandidateRegressionGate


SPEC_CONTENT = {
    "schema_version": 1,
    "spec_id": "parameter-golf-control-v1",
    "dataset_id": "fineweb_val_v1",
    "tokenizer_id": "fineweb_1024_bpe_v1",
    "seed_set": [42, 1337, 2025],
    "metric": {
        "name": "val_bpb",
        "direction": "lower_is_better",
        "max_drift": 0.02,
    },
    "command_template": "python3 train_gpt.py --seed {seed}",
    "required_env": {"python": "3"},
}


class Fast7RegressionGateTest(unittest.TestCase):
    def _trusted_control_bundle(self, root: Path) -> tuple[Path, Path]:
        spec_path = root / "control_measurement.yaml"
        spec_path.write_text(json.dumps(SPEC_CONTENT, indent=2), encoding="utf-8")
        submission = submitControlRun(
            spec_ref=spec_path,
            campaign_tick_id="tick-fast7-control",
            evidence_store=root,
        )
        bundle_path = publishEvidenceBundle(
            submission=submission,
            seed_metrics={42: 1.21, 1337: 1.20, 2025: 1.22},
            log_files=[],
        )
        return spec_path, bundle_path

    def _valid_promising_claim(self) -> dict:
        return {
            "candidateId": "candidate-fast7-001",
            "metric": {"name": "val_bpb", "direction": "lower_is_better", "value": 1.18},
            "evaluation": {
                "dataset_id": "fineweb_val_v1",
                "tokenizer_id": "fineweb_1024_bpe_v1",
                "metric_name": "val_bpb",
            },
            "artifact": {
                "claimed_bytes": 1200,
                "claimed_limit_bytes": 16_000_000,
                "claimed_compression": {"codec": "zlib", "ratio": 0.78},
                "claimed_quantization_scheme": "int8-per-row-zlib-projection",
            },
            "artifact_manifest": {
                "bytes": 1200,
                "limit_bytes": 16_000_000,
                "compression": {"codec": "zlib", "ratio": 0.78},
                "quantization_scheme": "int8-per-row-zlib-projection",
            },
            "reproduction": {
                "status": "verified",
                "metric_value": 1.181,
                "tolerance": 0.005,
            },
        }

    def test_rejects_eval_drift_against_trusted_control_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            spec_path, bundle_path = self._trusted_control_bundle(Path(tmp_dir))
            claim = self._valid_promising_claim()
            claim["evaluation"]["dataset_id"] = "fineweb_train_sample"

            result = validateCandidateRegressionGate(claim, bundle_path=bundle_path, spec_ref=spec_path)

            self.assertEqual("reject", result["decision"])
            self.assertIn("eval-drift:dataset-id", result["violations"])

    def test_rejects_artifact_compression_mismatch_against_claimed_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            spec_path, bundle_path = self._trusted_control_bundle(Path(tmp_dir))
            claim = self._valid_promising_claim()
            claim["artifact_manifest"]["compression"]["codec"] = "brotli"
            claim["artifact_manifest"]["bytes"] = 1300

            result = validateCandidateRegressionGate(claim, bundle_path=bundle_path, spec_ref=spec_path)

            self.assertEqual("reject", result["decision"])
            self.assertIn("artifact-mismatch:bytes", result["violations"])
            self.assertIn("artifact-mismatch:compression-codec", result["violations"])

    def test_rejects_promising_result_when_reproduction_is_not_verified(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            spec_path, bundle_path = self._trusted_control_bundle(Path(tmp_dir))
            claim = self._valid_promising_claim()
            claim["reproduction"] = {"status": "failed", "reason": "rerun metric regressed"}

            result = validateCandidateRegressionGate(claim, bundle_path=bundle_path, spec_ref=spec_path)

            self.assertEqual("reject", result["decision"])
            self.assertTrue(result["promising"])
            self.assertIn("reproduction-failed:not-verified", result["violations"])

    def test_accepts_reproduced_candidate_with_matching_artifact_and_eval_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            spec_path, bundle_path = self._trusted_control_bundle(Path(tmp_dir))

            result = validateCandidateRegressionGate(
                self._valid_promising_claim(),
                bundle_path=bundle_path,
                spec_ref=spec_path,
            )

            self.assertEqual("accept", result["decision"])
            self.assertEqual([], result["violations"])


if __name__ == "__main__":
    unittest.main()
