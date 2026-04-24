import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = REPO_ROOT / "fastest/source/measurement_evidence.json"
STATUS_PATH = REPO_ROOT / "fastest/generated/campaign_status.json"
STATE_PATH = REPO_ROOT / "fastest/generated/campaign_state.json"
OUTCOME_PATH = REPO_ROOT / "fastest/generated/fast41_non_record_outcome.json"
PACKET_PATH = REPO_ROOT / "planning/fast41_non_record_exploration_execution_packet.json"


class Fast41NonRecordExplorationExecutionPacketTest(unittest.TestCase):
    def test_fast41_packet_evidence_and_projection_refresh_contract(self) -> None:
        outcome = json.loads(OUTCOME_PATH.read_text())
        packet = json.loads(PACKET_PATH.read_text())
        evidence = json.loads(EVIDENCE_PATH.read_text())
        status = json.loads(STATUS_PATH.read_text())
        state = json.loads(STATE_PATH.read_text())

        self.assertEqual(outcome.get("taskId"), "FAST-41")
        self.assertEqual(outcome.get("status"), "success")
        self.assertEqual(outcome.get("reasonCode"), "non-record-exploration-success")
        self.assertEqual(outcome.get("observedMetric"), {"name": "benchmark-score", "value": 1.0126})

        self.assertEqual(packet.get("candidateId"), "exp-fast41-non-record-001")
        self.assertEqual(packet.get("promotionRationale", {}).get("decision"), "reject")
        self.assertIn(
            "python fastest/scripts/run_non_record_exploration_candidate.py",
            packet.get("commands", [""])[0],
        )

        fast41_records = [
            record
            for record in evidence.get("experimentRecords", [])
            if isinstance(record, dict)
            and record.get("taskId") == "FAST-41"
            and record.get("experimentId") == "exp-fast41-non-record-001"
        ]
        self.assertEqual(len(fast41_records), 1)
        record = fast41_records[0]
        self.assertEqual(
            record.get("idempotencyKey"),
            "FAST-41:non-record-exploration:exp-fast41-non-record-001",
        )
        self.assertEqual(record.get("promotionRationale", {}).get("decision"), "reject")

        self.assertEqual(status.get("lastProgressAt"), outcome.get("completedAt"))
        self.assertEqual(status.get("updatedAt"), outcome.get("completedAt"))
        self.assertEqual(state.get("lastProgressAt"), outcome.get("completedAt"))
        self.assertEqual(state.get("campaign", {}).get("updatedAt"), outcome.get("completedAt"))


if __name__ == "__main__":
    unittest.main()
