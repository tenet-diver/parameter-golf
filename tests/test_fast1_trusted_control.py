import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
STATUS_PATH = REPO_ROOT / "fastest/generated/campaign_status.json"
STATE_PATH = REPO_ROOT / "fastest/generated/campaign_state.json"


class Fast1TrustedControlContractTest(unittest.TestCase):
    def test_trusted_control_baseline_established(self) -> None:
        status = json.loads(STATUS_PATH.read_text())
        state = json.loads(STATE_PATH.read_text())

        progress = status["progress"]
        completed = status["recentCompletedExperiments"]
        evidence_cache = state["evidenceSummaryCache"]

        self.assertEqual(status["trustedControlState"], "established")
        self.assertGreaterEqual(progress["totalExperiments"], 1)
        self.assertGreaterEqual(progress["acceptedExperiments"], 1)
        self.assertIsNotNone(progress["mostRecentEvidenceId"])
        self.assertTrue(completed)

        self.assertEqual(evidence_cache["trustedControlState"], "established")
        self.assertGreaterEqual(evidence_cache["totalExperiments"], 1)
        self.assertGreaterEqual(evidence_cache["acceptedExperiments"], 1)
        self.assertTrue(evidence_cache["recentCompletedExperiments"])


if __name__ == "__main__":
    unittest.main()
