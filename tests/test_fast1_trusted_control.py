import json
import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
STATUS_PATH = REPO_ROOT / "fastest/generated/campaign_status.json"
STATE_PATH = REPO_ROOT / "fastest/generated/campaign_state.json"
AUTHORITATIVE_EVIDENCE_PATH = REPO_ROOT / "fastest/source/measurement_evidence.json"
CONTROLLER_TICK_PATH = REPO_ROOT / "fastest/scripts/controller_tick.py"


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

    def test_controller_tick_regenerates_views_from_authoritative_evidence(self) -> None:
        original_source = AUTHORITATIVE_EVIDENCE_PATH.read_text()
        original_status = STATUS_PATH.read_text()
        original_state = STATE_PATH.read_text()

        try:
            evidence = json.loads(original_source)
            evidence["updatedAt"] = "2026-04-23T22:00:00.000Z"
            evidence["summary"]["artifactIds"] = ["evidence-fast1-control-regen-001"]
            evidence["summary"]["totalExperiments"] = 2
            evidence["summary"]["acceptedExperiments"] = 2
            evidence["summary"]["mostRecentEvidenceId"] = "evidence-fast1-control-regen-001"
            evidence["recentCompletedExperiments"] = [
                "exp-fast1-control-20260423-001",
                "exp-fast1-control-regen-001",
            ]
            evidence["experimentRecords"].append(
                {
                    "experimentId": "exp-fast1-control-regen-001",
                    "evidenceId": "evidence-fast1-control-regen-001",
                    "lane": "measurement",
                    "status": "accepted",
                    "completedAt": "2026-04-23T22:00:00.000Z",
                    "objectiveMetricName": "benchmark-score",
                    "objectiveValue": 1.01,
                }
            )
            AUTHORITATIVE_EVIDENCE_PATH.write_text(f"{json.dumps(evidence, indent=2)}\n")

            tick = subprocess.run(
                [sys.executable, str(CONTROLLER_TICK_PATH)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(tick.returncode, 0, msg=tick.stderr)

            status = json.loads(STATUS_PATH.read_text())
            state = json.loads(STATE_PATH.read_text())
            summary = evidence["summary"]

            self.assertEqual(status["progress"]["totalExperiments"], summary["totalExperiments"])
            self.assertEqual(
                status["progress"]["acceptedExperiments"],
                summary["acceptedExperiments"],
            )
            self.assertEqual(
                status["progress"]["mostRecentEvidenceId"],
                summary["mostRecentEvidenceId"],
            )
            self.assertEqual(
                status["recentCompletedExperiments"],
                evidence["recentCompletedExperiments"],
            )
            self.assertEqual(
                state["evidenceSummaryCache"]["artifactIds"],
                summary["artifactIds"],
            )
            self.assertEqual(
                state["evidenceSummaryCache"]["acceptedExperiments"],
                summary["acceptedExperiments"],
            )
            self.assertEqual(
                state["evidenceSummaryCache"]["totalExperiments"],
                summary["totalExperiments"],
            )
            self.assertEqual(
                state["evidenceSummaryCache"]["recentCompletedExperiments"],
                evidence["recentCompletedExperiments"],
            )
        finally:
            AUTHORITATIVE_EVIDENCE_PATH.write_text(original_source)
            STATUS_PATH.write_text(original_status)
            STATE_PATH.write_text(original_state)


if __name__ == "__main__":
    unittest.main()
