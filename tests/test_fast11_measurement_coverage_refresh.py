import json
import unittest
from pathlib import Path

from fastest.scripts.render_campaign_evidence import apply_measurement_evidence


REPO_ROOT = Path(__file__).resolve().parents[1]
STATUS_PATH = REPO_ROOT / "fastest/generated/campaign_status.json"
STATE_PATH = REPO_ROOT / "fastest/generated/campaign_state.json"
SOURCE_PATH = REPO_ROOT / "fastest/source/measurement_evidence.json"


class Fast11MeasurementCoverageRefreshTest(unittest.TestCase):
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
