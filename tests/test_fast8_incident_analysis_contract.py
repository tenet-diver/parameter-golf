import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
INCIDENT_ANALYSIS_PATH = REPO_ROOT / "incidents/incident-analysis.json"


class Fast8IncidentAnalysisContractTest(unittest.TestCase):
    def test_incident_analysis_artifact_satisfies_fast8_contract(self) -> None:
        payload = json.loads(INCIDENT_ANALYSIS_PATH.read_text())

        self.assertEqual(payload["taskId"], "FAST-8")
        self.assertEqual(payload["sourcePlanning"]["taskId"], "FAST-7")

        self.assertEqual(
            payload["trigger"]["fingerprint"],
            "backlog-under-threshold-restore-10",
        )
        self.assertEqual(
            payload["trigger"]["artifactId"],
            "planning-backlog-trigger-backlog-under-threshold-restore-10-2026-04-23t22-20-34-841z",
        )

        timeline_events = payload["timeline"]["events"]
        self.assertGreaterEqual(len(timeline_events), 2)
        self.assertEqual(
            timeline_events[0]["timestamp"],
            "2026-04-23T22:20:15.000Z",
        )
        self.assertEqual(timeline_events[0]["eventType"], "alert-fired")

        planning_event = next(
            event
            for event in timeline_events
            if event.get("eventType") == "planning-artifact-created"
        )
        self.assertEqual(
            planning_event["timestamp"],
            "2026-04-23T22:20:34.841Z",
        )

        risks = payload["rankedRecurrenceRisks"]
        self.assertGreaterEqual(len(risks), 3)
        for index, risk in enumerate(risks, start=1):
            self.assertEqual(risk["rank"], index)
            self.assertTrue(risk.get("impact"))
            self.assertTrue(risk.get("confidence"))
            self.assertTrue(risk.get("validationPlan"))

        mitigations = payload["mitigationRequirements"]
        self.assertGreaterEqual(len(mitigations), 3)
        for mitigation in mitigations:
            self.assertTrue(mitigation.get("behavior"))
            self.assertTrue(mitigation.get("constraints"))
            self.assertTrue(mitigation.get("outcomes"))

        high_risk_mitigations = [
            mitigation
            for mitigation in mitigations
            if mitigation.get("riskLevel") == "high"
        ]
        self.assertTrue(high_risk_mitigations)
        for mitigation in high_risk_mitigations:
            self.assertTrue(mitigation.get("rollbackExpectations"))
            self.assertTrue(mitigation.get("degradationExpectations"))


if __name__ == "__main__":
    unittest.main()
