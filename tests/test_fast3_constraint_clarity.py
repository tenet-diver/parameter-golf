import copy
import json
import unittest

from fastest.scripts.render_campaign_evidence import apply_measurement_evidence, adjudicate_promotion


POLICY_VERSION = "pg-constraint-clarity-v1"


class Fast3ConstraintClarityContractTest(unittest.TestCase):
    def _base_candidate(self) -> dict:
        return {
            "candidateId": "exp-fast3-candidate-001",
            "policyVersion": POLICY_VERSION,
            "artifactBudgetPolicy": "required-before-promotion",
            "requiredEvidence": [
                "trusted-baseline-evidence",
                "benchmark-measurement-evidence",
                "reproducibility-manifest",
            ],
            "providedEvidence": [
                "trusted-baseline-evidence",
                "benchmark-measurement-evidence",
                "reproducibility-manifest",
            ],
            "legalitySignals": {
                "status": "legal",
                "violations": [],
                "conflicts": [],
            },
            "promotionGatesSatisfied": True,
        }

    def test_promotes_when_evidence_complete_and_gates_satisfied(self) -> None:
        result = adjudicate_promotion(self._base_candidate())

        self.assertEqual(result["decision"], "promote")
        self.assertEqual(result["violatedRules"], [])
        self.assertEqual(result["missingEvidence"], [])

    def test_holds_when_required_evidence_is_missing(self) -> None:
        candidate = self._base_candidate()
        candidate["providedEvidence"] = ["trusted-baseline-evidence"]

        result = adjudicate_promotion(candidate)

        self.assertIn(result["decision"], {"hold", "reject"})
        self.assertEqual(
            result["missingEvidence"],
            ["benchmark-measurement-evidence", "reproducibility-manifest"],
        )
        self.assertFalse(result["promotableStateChange"])

    def test_rejects_on_legality_violation_with_policy_tied_reason_code(self) -> None:
        candidate = self._base_candidate()
        candidate["legalitySignals"] = {
            "status": "illegal",
            "violations": ["rule-illegal-dataset"],
            "conflicts": [],
        }

        result = adjudicate_promotion(candidate)

        self.assertEqual(result["decision"], "reject")
        self.assertIn(
            f"{POLICY_VERSION}:rule-illegal-dataset",
            result["violatedRules"],
        )

    def test_ambiguity_fails_closed_with_explicit_reason_codes(self) -> None:
        candidate = self._base_candidate()
        candidate["legalitySignals"] = {
            "status": "legal",
            "violations": [],
            "conflicts": ["conflicting-judge-signals"],
        }

        result = adjudicate_promotion(candidate)

        self.assertIn(result["decision"], {"hold", "reject"})
        self.assertIn(
            f"{POLICY_VERSION}:ambiguity:conflicting-judge-signals",
            result["ambiguityReasons"],
        )

    def test_output_is_byte_stable_for_same_input_and_policy_version(self) -> None:
        candidate = self._base_candidate()

        first = adjudicate_promotion(candidate)
        second = adjudicate_promotion(copy.deepcopy(candidate))

        first_bytes = json.dumps(first, sort_keys=True, separators=(",", ":")).encode("utf-8")
        second_bytes = json.dumps(second, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self.assertEqual(first_bytes, second_bytes)

    def test_apply_measurement_evidence_uses_authoritative_policy_contract(self) -> None:
        source = {
            "updatedAt": "2026-04-23T20:21:00.000Z",
            "summary": {
                "artifactIds": [
                    "evidence-fast3-policy-contract-001",
                ],
                "trustedControlState": "established",
                "totalExperiments": 2,
                "acceptedExperiments": 1,
                "mostRecentEvidenceId": "evidence-fast3-policy-contract-001",
            },
            "recentCompletedExperiments": [
                "exp-fast3-policy-contract-001",
            ],
            "promotionPolicy": {
                "policyVersion": "pg-constraint-clarity-v2",
                "requiredEvidence": [
                    "trusted-baseline-evidence",
                    "benchmark-measurement-evidence",
                    "reproducibility-manifest",
                    "legality-attestation",
                ],
                "legalitySignals": {
                    "status": "ambiguous",
                    "violations": [],
                    "conflicts": ["judge-disagreement"],
                },
                "minimumAcceptedExperiments": 2,
            },
        }
        status = {
            "progress": {
                "totalExperiments": 0,
                "acceptedExperiments": 0,
                "mostRecentEvidenceId": None,
            },
            "blockers": [],
            "adapterStatus": {"details": {}},
        }
        state = {
            "campaign": {"metadata": {"artifactBudgetPolicy": "required-before-promotion"}},
            "evidenceSummaryCache": {},
        }

        rendered_status, rendered_state = apply_measurement_evidence(source, status, state)
        adjudication = rendered_status["promotionAdjudication"]

        self.assertEqual(adjudication["policyVersion"], "pg-constraint-clarity-v2")
        self.assertEqual(adjudication["decision"], "reject")
        self.assertEqual(
            adjudication["missingEvidence"],
            ["legality-attestation", "reproducibility-manifest"],
        )
        self.assertIn(
            "pg-constraint-clarity-v2:ambiguity:judge-disagreement",
            adjudication["ambiguityReasons"],
        )
        self.assertIn(
            "pg-constraint-clarity-v2:promotion-gates-not-satisfied",
            adjudication["violatedRules"],
        )
        self.assertEqual(
            rendered_state["evidenceSummaryCache"]["latestPromotionAdjudication"],
            adjudication,
        )

    def test_apply_measurement_evidence_promotes_manifest_candidate_over_latest_evidence(self) -> None:
        source = {
            "updatedAt": "2026-04-24T16:10:00Z",
            "summary": {
                "artifactIds": [
                    "evidence-best-combination-001",
                    "evidence-latest-non-record-001",
                    "reproducibility-manifest",
                ],
                "trustedControlState": "established",
                "totalExperiments": 2,
                "acceptedExperiments": 2,
                "mostRecentEvidenceId": "evidence-latest-non-record-001",
            },
            "recentCompletedExperiments": [
                "best-combination-001",
                "latest-non-record-001",
            ],
            "promotionPolicy": {
                "policyVersion": POLICY_VERSION,
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
            "reproducibilityManifest": {
                "candidateId": "best-combination-001",
                "artifactPath": "fastest/source/reproducibility_manifest_best_combination_001.json",
                "status": "provided",
            },
            "experimentRecords": [
                {
                    "experimentId": "best-combination-001",
                    "evidenceId": "evidence-best-combination-001",
                    "lane": "combination",
                },
                {
                    "experimentId": "latest-non-record-001",
                    "evidenceId": "evidence-latest-non-record-001",
                    "lane": "non-record-exploration",
                },
            ],
        }
        status = {
            "progress": {
                "totalExperiments": 0,
                "acceptedExperiments": 0,
                "mostRecentEvidenceId": None,
            },
            "blockers": [],
            "adapterStatus": {"details": {}},
        }
        state = {
            "campaign": {"metadata": {"artifactBudgetPolicy": "required-before-promotion"}},
            "evidenceSummaryCache": {},
        }

        rendered_status, _ = apply_measurement_evidence(source, status, state)
        adjudication = rendered_status["promotionAdjudication"]

        self.assertEqual(adjudication["candidateId"], "evidence-best-combination-001")
        self.assertEqual(adjudication["decision"], "promote")

    def test_apply_measurement_evidence_blocks_cpu_subset_from_benchmark_progress(self) -> None:
        source = {
            "updatedAt": "2026-04-25T22:30:00Z",
            "summary": {
                "artifactIds": [
                    "evidence-exp-lr-half-256iter-d256-cpu",
                ],
                "trustedControlState": "established",
                "totalExperiments": 1,
                "acceptedExperiments": 1,
                "mostRecentEvidenceId": "evidence-exp-lr-half-256iter-d256-cpu",
            },
            "recentCompletedExperiments": [
                "exp-lr-half-256iter-d256-cpu",
            ],
            "promotionPolicy": {
                "policyVersion": POLICY_VERSION,
                "requiredEvidence": [
                    "trusted-baseline-evidence",
                    "benchmark-measurement-evidence",
                ],
                "legalitySignals": {
                    "status": "legal",
                    "violations": [],
                    "conflicts": [],
                },
                "minimumAcceptedExperiments": 1,
            },
            "experimentRecords": [
                {
                    "experimentId": "exp-lr-half-256iter-d256-cpu",
                    "evidenceId": "evidence-exp-lr-half-256iter-d256-cpu",
                    "lane": "cpu-subset",
                    "verificationClass": "cpu-subset",
                    "benchmarkProgressEligible": False,
                }
            ],
        }
        status = {
            "progress": {
                "totalExperiments": 0,
                "acceptedExperiments": 0,
                "mostRecentEvidenceId": None,
            },
            "blockers": [],
            "adapterStatus": {"details": {}},
        }
        state = {
            "campaign": {"metadata": {"artifactBudgetPolicy": "required-before-promotion"}},
            "evidenceSummaryCache": {},
        }

        rendered_status, _ = apply_measurement_evidence(source, status, state)
        adjudication = rendered_status["promotionAdjudication"]

        self.assertEqual(adjudication["candidateId"], "evidence-exp-lr-half-256iter-d256-cpu")
        self.assertIn("benchmark-measurement-evidence", adjudication["missingEvidence"])
        self.assertNotEqual(adjudication["decision"], "promote")


if __name__ == "__main__":
    unittest.main()
