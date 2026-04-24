import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = REPO_ROOT / "fastest/source/measurement_evidence.json"
PACKET_PATH = REPO_ROOT / "planning/fast34_combination_execution_packet.json"


class Fast34CombinationRuntimeEvidenceTest(unittest.TestCase):
    def test_fast34_combination_record_and_execution_packet_exist(self) -> None:
        evidence = json.loads(EVIDENCE_PATH.read_text())
        records = evidence.get("experimentRecords", [])
        self.assertIsInstance(records, list)

        fast34_records = [
            record
            for record in records
            if isinstance(record, dict)
            and record.get("taskId") == "FAST-34"
            and record.get("lane") == "combination"
            and record.get("experimentId") == "exp-fast34-combination-001"
            and record.get("status") == "accepted"
        ]
        self.assertEqual(len(fast34_records), 1)
        record = fast34_records[0]

        self.assertEqual(record.get("objectiveMetricName"), "benchmark-score")
        self.assertIsInstance(record.get("objectiveValue"), (int, float))
        self.assertTrue(record.get("evidenceId"))
        self.assertTrue(record.get("traceId"))

        packet = json.loads(PACKET_PATH.read_text())
        self.assertEqual(packet.get("taskId"), "FAST-34")
        self.assertEqual(packet.get("candidateId"), "exp-fast34-combination-001")
        self.assertEqual(packet.get("status"), "success")
        self.assertEqual(packet.get("reasonCode"), "combination-success")
        self.assertEqual(packet.get("observedMetric", {}).get("name"), "benchmark-score")
        self.assertEqual(packet.get("observedMetric", {}).get("value"), record.get("objectiveValue"))


if __name__ == "__main__":
    unittest.main()
