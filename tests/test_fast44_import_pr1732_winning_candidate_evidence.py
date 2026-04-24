import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RECORD_DIR = REPO_ROOT / "records/track_10min_16mb/2026-04-19_SP8192_QuantumFusionPlus_Hadamard_AWQ"
SUBMISSION_PATH = RECORD_DIR / "submission.json"
README_PATH = RECORD_DIR / "README.md"
EVIDENCE_PATH = REPO_ROOT / "fastest/source/measurement_evidence.json"


class Fast44ImportPr1732WinningCandidateEvidenceTest(unittest.TestCase):
    def test_fast44_import_has_attributed_candidate_and_external_repro_status(self) -> None:
        self.assertTrue(RECORD_DIR.exists())
        self.assertTrue(SUBMISSION_PATH.exists())
        self.assertTrue(README_PATH.exists())

        submission = json.loads(SUBMISSION_PATH.read_text())
        readme = README_PATH.read_text()

        self.assertEqual(submission.get("val_bpb"), 1.0785)
        self.assertEqual(submission.get("improvements_over_sota", {}).get("pr_1493_bpp"), 1.0810)
        self.assertEqual(submission.get("source_attribution", {}).get("github_id"), "Victory963")
        self.assertEqual(submission.get("source_attribution", {}).get("source_pr"), "openai/pr-1732")
        self.assertEqual(submission.get("reproduction", {}).get("status"), "needs_external_8xh100_verification")

        self.assertIn("Victory963", readme)
        self.assertIn("PR #1732", readme)
        self.assertIn("1.0785", readme)
        self.assertIn("1.0810", readme)
        self.assertIn("needs external 8xH100 verification", readme)

        evidence = json.loads(EVIDENCE_PATH.read_text())
        records = evidence.get("experimentRecords", [])
        fast44_records = [
            record
            for record in records
            if isinstance(record, dict)
            and record.get("taskId") == "FAST-44"
            and record.get("experimentId") == "exp-fast44-pr1732-import-001"
            and record.get("status") == "accepted"
        ]
        self.assertEqual(len(fast44_records), 1)
        record = fast44_records[0]
        self.assertEqual(record.get("objectiveMetricName"), "val_bpb")
        self.assertEqual(record.get("objectiveValue"), 1.0785)
        self.assertEqual(record.get("comparison", {}).get("currentMergedSotaValBpb"), 1.0810)
        self.assertEqual(
            record.get("reproductionStatus", {}).get("status"),
            "needs_external_8xh100_verification",
        )


if __name__ == "__main__":
    unittest.main()
