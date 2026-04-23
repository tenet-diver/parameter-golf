import copy
import json
import unittest

from fastest.scripts.render_campaign_evidence import adjudicate_promotion


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


if __name__ == "__main__":
    unittest.main()
