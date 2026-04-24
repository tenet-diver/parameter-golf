import json
import unittest
from pathlib import Path

from fastest.scripts.render_campaign_evidence import apply_measurement_evidence


REPO_ROOT = Path(__file__).resolve().parents[1]
STATUS_PATH = REPO_ROOT / "fastest/generated/campaign_status.json"
STATE_PATH = REPO_ROOT / "fastest/generated/campaign_state.json"
SOURCE_PATH = REPO_ROOT / "fastest/source/measurement_evidence.json"
PACKET_PATH = REPO_ROOT / "planning/fast37_combination_execution_packet.json"


class Fast37CombinationExecutionPacketTest(unittest.TestCase):
    def test_fast37_combination_record_and_execution_packet_contract(self) -> None:
        source = json.loads(SOURCE_PATH.read_text())
        records = source.get("experimentRecords", [])
        self.assertIsInstance(records, list)

        fast37_records = [
            record
            for record in records
            if isinstance(record, dict)
            and record.get("taskId") == "FAST-37"
            and record.get("lane") == "combination"
            and record.get("experimentId") == "exp-fast37-combination-001"
            and record.get("status") == "accepted"
        ]
        self.assertEqual(len(fast37_records), 1)
        record = fast37_records[0]

        self.assertEqual(record.get("objectiveMetricName"), "benchmark-score")
        self.assertEqual(record.get("objectiveValue"), 1.0999)
        self.assertEqual(
            record.get("idempotencyKey"),
            "FAST-37:combination:exp-fast37-combination-001",
        )
        self.assertTrue(record.get("evidenceId"))
        self.assertTrue(record.get("traceId"))

        packet = json.loads(PACKET_PATH.read_text())
        self.assertEqual(packet.get("taskId"), "FAST-37")
        self.assertEqual(packet.get("candidateId"), "exp-fast37-combination-001")
        self.assertEqual(packet.get("status"), "success")
        self.assertEqual(packet.get("reasonCode"), "combination-success")
        self.assertEqual(packet.get("observedMetric", {}).get("name"), "benchmark-score")
        self.assertEqual(packet.get("observedMetric", {}).get("value"), record.get("objectiveValue"))
        self.assertIsInstance(packet.get("commands"), list)
        self.assertTrue(packet.get("commands"))
        self.assertIsInstance(packet.get("artifactPaths"), list)
        self.assertIn("fastest/generated/fast37_run_outcome.json", packet.get("artifactPaths"))
        self.assertIn("fastest/source/measurement_evidence.json", packet.get("artifactPaths"))
        self.assertIn("fastest/generated/campaign_status.json", packet.get("artifactPaths"))
        self.assertIn("fastest/generated/campaign_state.json", packet.get("artifactPaths"))
        self.assertEqual(packet.get("promotionRationale", {}).get("decision"), "promote")
        self.assertIsInstance(packet.get("promotionRationale", {}).get("reason"), str)
        self.assertTrue(packet.get("promotionRationale", {}).get("reason"))

    def test_generated_views_match_authoritative_measurement_projection(self) -> None:
        source = json.loads(SOURCE_PATH.read_text())
        status = json.loads(STATUS_PATH.read_text())
        state = json.loads(STATE_PATH.read_text())

        expected_status, expected_state = apply_measurement_evidence(
            source,
            json.loads(json.dumps(status)),
            json.loads(json.dumps(state)),
        )

        self.assertEqual(status, expected_status)
        self.assertEqual(state, expected_state)


if __name__ == "__main__":
    unittest.main()
