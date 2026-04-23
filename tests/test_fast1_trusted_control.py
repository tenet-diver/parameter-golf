import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
STATUS_PATH = REPO_ROOT / "fastest/generated/campaign_status.json"
STATE_PATH = REPO_ROOT / "fastest/generated/campaign_state.json"
AUTHORITATIVE_EVIDENCE_PATH = REPO_ROOT / "fastest/source/measurement_evidence.json"


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

    def test_generated_campaign_views_are_derived_from_authoritative_evidence(self) -> None:
        status = json.loads(STATUS_PATH.read_text())
        state = json.loads(STATE_PATH.read_text())
        evidence = json.loads(AUTHORITATIVE_EVIDENCE_PATH.read_text())

        summary = evidence["summary"]
        recent_completed = evidence["recentCompletedExperiments"]

        self.assertTrue(summary["artifactIds"])
        self.assertTrue(recent_completed)

        self.assertEqual(status["trustedControlState"], summary["trustedControlState"])
        self.assertEqual(
            status["progress"]["totalExperiments"],
            summary["totalExperiments"],
        )
        self.assertEqual(
            status["progress"]["acceptedExperiments"],
            summary["acceptedExperiments"],
        )
        self.assertEqual(
            status["progress"]["mostRecentEvidenceId"],
            summary["mostRecentEvidenceId"],
        )
        self.assertEqual(status["recentCompletedExperiments"], recent_completed)

        self.assertEqual(
            state["evidenceSummaryCache"]["trustedControlState"],
            summary["trustedControlState"],
        )
        self.assertEqual(
            state["evidenceSummaryCache"]["artifactIds"],
            summary["artifactIds"],
        )
        self.assertEqual(
            state["evidenceSummaryCache"]["totalExperiments"],
            summary["totalExperiments"],
        )
        self.assertEqual(
            state["evidenceSummaryCache"]["acceptedExperiments"],
            summary["acceptedExperiments"],
        )
        self.assertEqual(
            state["evidenceSummaryCache"]["recentCompletedExperiments"],
            recent_completed,
        )


if __name__ == "__main__":
    unittest.main()
