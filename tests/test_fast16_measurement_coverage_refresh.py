import json
import unittest
from pathlib import Path

from fastest.scripts.render_campaign_evidence import apply_measurement_evidence


REPO_ROOT = Path(__file__).resolve().parents[1]
STATUS_PATH = REPO_ROOT / "fastest/generated/campaign_status.json"
STATE_PATH = REPO_ROOT / "fastest/generated/campaign_state.json"
SOURCE_PATH = REPO_ROOT / "fastest/source/measurement_evidence.json"


class Fast16MeasurementCoverageRefreshTest(unittest.TestCase):
    def test_authoritative_evidence_contains_reproducible_fast16_measurement_record(self) -> None:
        source = json.loads(SOURCE_PATH.read_text())
        records = source["experimentRecords"]
        fast16_records = [
            record
            for record in records
            if record.get("taskId") == "FAST-16"
            and record.get("lane") == "measurement"
            and record.get("status") == "accepted"
        ]
        self.assertTrue(
            fast16_records,
            "Expected at least one accepted FAST-16 measurement record in authoritative evidence.",
        )

        record = fast16_records[-1]
        self.assertTrue(record.get("experimentId"))
        self.assertTrue(record.get("evidenceId"))
        self.assertTrue(record.get("idempotencyKey"))
        self.assertTrue(record.get("traceId"))

        run_config = record.get("runConfig")
        self.assertIsInstance(run_config, dict)
        self.assertIn("seed", run_config)

        budget_caps = record.get("budgetCaps")
        self.assertIsInstance(budget_caps, dict)
        self.assertIn("maxRuntimeSeconds", budget_caps)

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
