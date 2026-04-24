import json
import unittest

from fastest.scripts.render_campaign_evidence import apply_measurement_evidence


class Fast31QaDecisionContractTest(unittest.TestCase):
    def _base_source(self) -> dict:
        return {
            "updatedAt": "2026-04-24T17:30:00Z",
            "summary": {
                "artifactIds": [
                    "evidence-fast31-measurement-001",
                    "reproducibility-manifest",
                ],
                "trustedControlState": "established",
                "totalExperiments": 1,
                "acceptedExperiments": 1,
                "mostRecentEvidenceId": "evidence-fast31-measurement-001",
            },
            "recentCompletedExperiments": ["exp-fast31-measurement-001"],
            "promotionPolicy": {
                "policyVersion": "pg-constraint-clarity-v1",
                "requiredEvidence": [
                    "trusted-baseline-evidence",
                    "benchmark-measurement-evidence",
                    "reproducibility-manifest",
                ],
                "legalitySignals": {
                    "status": "legal",
                    "violations": [],
                    "conflicts": [],
                },
                "minimumAcceptedExperiments": 1,
            },
            "executionAttempts": [],
        }

    def _base_status(self) -> dict:
        return {
            "progress": {
                "totalExperiments": 0,
                "acceptedExperiments": 0,
                "mostRecentEvidenceId": None,
            },
            "blockers": [],
            "adapterStatus": {"details": {}},
        }

    def _base_state(self) -> dict:
        return {
            "campaign": {"metadata": {"artifactBudgetPolicy": "required-before-promotion"}},
            "evidenceSummaryCache": {},
        }

    def test_missing_manifest_or_waiver_for_latest_attempt_blocks_release_ready_adjudication(self) -> None:
        source = self._base_source()
        source["executionAttempts"] = [
            {
                "taskId": "FAST-31",
                "lane": "measurement",
                "attemptedAt": "2026-04-24T17:31:00Z",
                "candidateId": "exp-fast31-measurement-001",
                "commands": ["python fastest/scripts/run_cheap_screen_candidate.py --task-store-dir /tmp/tasks"],
                "artifactPaths": ["fastest/source/measurement_evidence.json"],
                "observedMetricOutput": {"status": "observed", "name": "benchmark-score", "value": 1.1021},
                "promotionDecisionRationale": {
                    "decision": "promote",
                    "advanceOutcome": "advance",
                    "reason": "candidate improves metric",
                    "missingEvidence": [],
                },
                "blockedReasonCode": None,
                "nextRecoveryTask": None,
                "reproducibilityEvidence": {
                    "type": "manifest",
                    "status": "missing",
                    "artifactPath": None,
                },
            }
        ]

        rendered_status, _ = apply_measurement_evidence(source, self._base_status(), self._base_state())
        adjudication = rendered_status["promotionAdjudication"]

        self.assertIn(adjudication["decision"], {"hold", "reject"})
        self.assertIn("reproducibility-manifest", adjudication["missingEvidence"])
        self.assertFalse(adjudication["promotableStateChange"])

    def test_waiver_with_required_approval_metadata_unblocks_attempt_level_adjudication(self) -> None:
        source = self._base_source()
        source["executionAttempts"] = [
            {
                "taskId": "FAST-31",
                "lane": "measurement",
                "attemptedAt": "2026-04-24T17:31:00Z",
                "candidateId": "exp-fast31-measurement-001",
                "commands": ["python fastest/scripts/run_cheap_screen_candidate.py --task-store-dir /tmp/tasks"],
                "artifactPaths": ["fastest/source/measurement_evidence.json"],
                "observedMetricOutput": {"status": "observed", "name": "benchmark-score", "value": 1.1021},
                "promotionDecisionRationale": {
                    "decision": "promote",
                    "advanceOutcome": "advance",
                    "reason": "candidate improves metric",
                    "missingEvidence": [],
                },
                "blockedReasonCode": None,
                "nextRecoveryTask": None,
                "reproducibilityEvidence": {
                    "type": "waiver",
                    "waiverId": "waiver-fast31-001",
                    "approver": "campaign-owner",
                    "approvedAt": "2026-04-24T17:32:00Z",
                    "expiresAt": "2026-05-01T00:00:00Z",
                    "reason": "task store had no measurement candidate at attempt time",
                },
            }
        ]

        rendered_status, rendered_state = apply_measurement_evidence(
            source,
            self._base_status(),
            self._base_state(),
        )
        adjudication = rendered_status["promotionAdjudication"]

        self.assertEqual(adjudication["decision"], "promote")
        self.assertEqual(adjudication["missingEvidence"], [])
        self.assertTrue(adjudication["promotableStateChange"])
        self.assertEqual(rendered_state["evidenceSummaryCache"]["latestPromotionAdjudication"], adjudication)


if __name__ == "__main__":
    unittest.main()
