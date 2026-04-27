import unittest

from fastest.scripts.submission_legality import classify_submission_legality


class Fast868SubmissionLegalityTest(unittest.TestCase):
    def test_classifies_leaderboard_legal_when_all_competition_rules_are_met(self) -> None:
        classification = classify_submission_legality(
            {
                "experimentId": "exp-fast868-legal",
                "artifactBytes": 15_993_232,
                "objectiveMetricName": "FineWeb validation bits-per-byte",
                "runtimeSeconds": 587,
                "hardware": "8xH100 SXM",
                "selfContainedArtifact": True,
                "externalDownloadsDuringEvaluation": False,
                "networkAccessDuringEvaluation": False,
                "usesValidationDataDuringTraining": False,
                "credibleTenMinutePath": True,
            }
        )

        self.assertEqual(classification["status"], "leaderboard-legal")
        self.assertEqual(classification["uncertaintyReasons"], [])
        self.assertIn("artifact-under-16mb", classification["passedRules"])
        self.assertIn("fineweb-bpb-scoring", classification["passedRules"])

    def test_classifies_known_non_leaderboard_runs_as_non_record_only(self) -> None:
        classification = classify_submission_legality(
            {
                "experimentId": "exp-fast868-non-record",
                "track": "non-record",
                "artifactBytes": 15_670_651,
                "objectiveMetricName": "FineWeb validation bits-per-byte",
                "runtimeSeconds": 14_400,
                "hardware": "8xH100 SXM",
                "selfContainedArtifact": True,
                "externalDownloadsDuringEvaluation": False,
                "networkAccessDuringEvaluation": False,
                "usesValidationDataDuringTraining": False,
                "credibleTenMinutePath": False,
            }
        )

        self.assertEqual(classification["status"], "non-record-only")
        self.assertIn("declared-non-record-track", classification["nonRecordReasons"])
        self.assertIn("runtime-exceeds-10-minute-leaderboard-cap", classification["nonRecordReasons"])
        self.assertEqual(classification["uncertaintyReasons"], [])

    def test_keeps_ambiguous_imports_uncertain_with_reason_codes(self) -> None:
        classification = classify_submission_legality(
            {
                "ideaId": "historical-pr-import",
                "artifactBytes": None,
                "objectiveMetricName": "val_bpb",
                "summary": "Imported PR mentions legal score-first TTT but lacks reproduction evidence.",
                "selfContainedArtifact": None,
                "usesValidationDataDuringTraining": None,
            }
        )

        self.assertEqual(classification["status"], "uncertain")
        self.assertIn("missing-artifact-size", classification["uncertaintyReasons"])
        self.assertIn("missing-runtime-or-credible-path", classification["uncertaintyReasons"])
        self.assertIn("missing-self-contained-evidence", classification["uncertaintyReasons"])
        self.assertIn("missing-validation-data-use-evidence", classification["uncertaintyReasons"])

    def test_known_rule_violations_take_priority_over_missing_evidence(self) -> None:
        cases = [
            ({"artifactBytes": 17_200_000}, "artifact-exceeds-16mb"),
            ({"objectiveMetricName": "local cpu-subset benchmark-score"}, "not-fineweb-validation-bpb"),
            ({"track": "non-record"}, "declared-non-record-track"),
            ({"runtimeSeconds": 601, "hardware": "8xH100"}, "runtime-exceeds-10-minute-leaderboard-cap"),
            ({"externalDownloadsDuringEvaluation": True}, "external-access-during-evaluation"),
            ({"usesValidationDataDuringTraining": True}, "validation-data-used-during-training"),
        ]

        for record, expected_reason in cases:
            with self.subTest(expected_reason=expected_reason):
                classification = classify_submission_legality(record)

                self.assertEqual(classification["status"], "non-record-only")
                self.assertIn(expected_reason, classification["nonRecordReasons"])
                self.assertTrue(classification["uncertaintyReasons"])

    def test_record_claim_requires_sota_margin_statistical_evidence_and_submission_files(self) -> None:
        classification = classify_submission_legality(
            {
                "experimentId": "exp-fast868-record-claim",
                "artifactBytes": 15_993_232,
                "objectiveMetricName": "FineWeb validation bits-per-byte",
                "runtimeSeconds": 587,
                "hardware": "8xH100 SXM",
                "selfContainedArtifact": True,
                "externalDownloadsDuringEvaluation": False,
                "networkAccessDuringEvaluation": False,
                "usesValidationDataDuringTraining": False,
                "credibleTenMinutePath": True,
                "track": "record",
                "sotaImprovementNats": 0.003,
                "statisticalPValue": 0.02,
                "changedTokenizerOrDataset": True,
                "tokenizerDatasetProof": False,
                "requiredSubmissionFiles": {
                    "readme": True,
                    "submissionJson": True,
                    "trainLog": False,
                    "trainScript": True,
                },
            }
        )

        self.assertEqual(classification["status"], "non-record-only")
        self.assertIn("record-improvement-below-0.005-nats", classification["nonRecordReasons"])
        self.assertIn("missing-statistically-significant-run-logs", classification["nonRecordReasons"])
        self.assertIn("missing-tokenizer-dataset-correctness-proof", classification["nonRecordReasons"])
        self.assertIn("missing-required-submission-files", classification["nonRecordReasons"])

    def test_record_claim_stays_uncertain_when_record_evidence_is_not_imported(self) -> None:
        classification = classify_submission_legality(
            {
                "experimentId": "exp-fast868-record-ambiguous",
                "artifactBytes": 15_993_232,
                "objectiveMetricName": "FineWeb validation bits-per-byte",
                "runtimeSeconds": 587,
                "hardware": "8xH100 SXM",
                "selfContainedArtifact": True,
                "externalDownloadsDuringEvaluation": False,
                "networkAccessDuringEvaluation": False,
                "usesValidationDataDuringTraining": False,
                "credibleTenMinutePath": True,
                "track": "record",
            }
        )

        self.assertEqual(classification["status"], "uncertain")
        self.assertIn("missing-record-improvement-evidence", classification["uncertaintyReasons"])
        self.assertIn("missing-statistical-significance-evidence", classification["uncertaintyReasons"])
        self.assertIn("missing-required-submission-file-evidence", classification["uncertaintyReasons"])


if __name__ == "__main__":
    unittest.main()
