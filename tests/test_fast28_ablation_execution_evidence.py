import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = REPO_ROOT / "fastest/source/measurement_evidence.json"


class Fast28AblationExecutionEvidenceTest(unittest.TestCase):
    def test_fast28_ablation_run_is_recorded_in_authoritative_evidence(self) -> None:
        evidence = json.loads(EVIDENCE_PATH.read_text())
        records = evidence.get("experimentRecords", [])
        self.assertIsInstance(records, list)

        fast28_records = [
            record
            for record in records
            if isinstance(record, dict)
            and record.get("taskId") == "FAST-28"
            and record.get("lane") == "ablation"
        ]
        self.assertEqual(len(fast28_records), 1)
        self.assertEqual(fast28_records[0].get("experimentId"), "exp-fast28-ablation-001")
        self.assertEqual(fast28_records[0].get("status"), "accepted")

        summary = evidence.get("summary", {})
        self.assertEqual(
            summary.get("mostRecentEvidenceId"),
            "evidence-exp-fast28-ablation-001",
        )


if __name__ == "__main__":
    unittest.main()
