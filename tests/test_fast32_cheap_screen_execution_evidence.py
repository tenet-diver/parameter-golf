import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = REPO_ROOT / "fastest/source/measurement_evidence.json"


class Fast32CheapScreenExecutionEvidenceTest(unittest.TestCase):
    def test_fast32_cheap_screen_record_has_execution_provenance_and_decision_rationale(self) -> None:
        evidence = json.loads(EVIDENCE_PATH.read_text())
        records = evidence.get("experimentRecords", [])
        self.assertIsInstance(records, list)

        fast32_records = [
            record
            for record in records
            if isinstance(record, dict)
            and record.get("taskId") == "FAST-32"
            and record.get("lane") == "cheap-screen"
            and record.get("experimentId") == "exp-fast32-cheap-screen-001"
        ]
        self.assertEqual(len(fast32_records), 1)
        record = fast32_records[0]

        self.assertEqual(record.get("objectiveMetricName"), "benchmark-score")
        self.assertEqual(record.get("objectiveValue"), 1.0743)

        commands = record.get("executionCommands")
        self.assertIsInstance(commands, list)
        self.assertTrue(commands)
        self.assertTrue(all(isinstance(command, str) and command for command in commands))

        artifact_paths = record.get("artifactPaths")
        self.assertIsInstance(artifact_paths, list)
        self.assertTrue(artifact_paths)
        self.assertTrue(all(isinstance(path, str) and path for path in artifact_paths))

        promotion_rationale = record.get("promotionRationale")
        self.assertIsInstance(promotion_rationale, dict)
        self.assertEqual(promotion_rationale.get("decision"), "hold")
        self.assertIn("reproducibility-manifest", promotion_rationale.get("missingEvidence", []))
        self.assertTrue(promotion_rationale.get("reason"))


if __name__ == "__main__":
    unittest.main()
