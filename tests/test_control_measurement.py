import json
import tempfile
import unittest
from pathlib import Path

from measurement.control_pipeline import (
    CONTROL_EVIDENCE_SCHEMA_VERSION,
    evaluateControlTrust,
    publishEvidenceBundle,
    submitControlRun,
)


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


class ControlMeasurementTests(unittest.TestCase):
    def test_untrusted_when_checksum_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            spec_path = root / "control_measurement.yaml"
            spec_path.write_text(json.dumps(SPEC_CONTENT, indent=2), encoding="utf-8")

            submit = submitControlRun(
                spec_ref=spec_path,
                campaign_tick_id="tick-001",
                evidence_store=root,
            )
            bundle_path = publishEvidenceBundle(
                submission=submit,
                seed_metrics={42: 1.21, 1337: 1.20, 2025: 1.22},
                log_files=[],
            )

            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            bundle["checksums"].pop(str(spec_path), None)
            bundle_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")

            trust = evaluateControlTrust(bundle_path=bundle_path, spec_ref=spec_path)
            self.assertEqual(CONTROL_EVIDENCE_SCHEMA_VERSION, trust["schema_version"])
            self.assertEqual("untrusted", trust["state"])
            self.assertTrue(any("checksum" in reason for reason in trust["reasons"]))

    def test_resumable_finalization_for_reused_submission(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            spec_path = root / "control_measurement.yaml"
            spec_path.write_text(json.dumps(SPEC_CONTENT, indent=2), encoding="utf-8")

            initial = submitControlRun(
                spec_ref=spec_path,
                campaign_tick_id="tick-002",
                evidence_store=root,
            )
            bundle_path = publishEvidenceBundle(
                submission=initial,
                seed_metrics={},
                log_files=[],
            )
            initial_trust = evaluateControlTrust(bundle_path=bundle_path, spec_ref=spec_path)
            self.assertEqual("untrusted", initial_trust["state"])

            resumed = submitControlRun(
                spec_ref=spec_path,
                campaign_tick_id="tick-002",
                evidence_store=root,
            )
            self.assertTrue(resumed.reused_existing_bundle)
            finalized_bundle = publishEvidenceBundle(
                submission=resumed,
                seed_metrics={42: 1.21, 1337: 1.20, 2025: 1.22},
                log_files=[],
            )
            finalized_trust = evaluateControlTrust(
                bundle_path=finalized_bundle,
                spec_ref=spec_path,
            )
            self.assertEqual("trusted", finalized_trust["state"])


if __name__ == "__main__":
    unittest.main()
