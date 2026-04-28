import json
import unittest
from pathlib import Path

from fastest.scripts.submission_legality import classify_submission_legality


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = REPO_ROOT / "fastest/source/control-evidence/fast866_trusted_control_baseline.json"


class Fast866TrustedControlBaselineTest(unittest.TestCase):
    def test_report_persists_reproduced_score_artifact_command_and_classification(self) -> None:
        self.assertTrue(REPORT_PATH.exists(), "FAST-866 control baseline report is missing")

        report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

        self.assertEqual(report.get("taskId"), "FAST-866")
        self.assertEqual(report.get("competitionPreset"), "parameter-golf")
        self.assertEqual(report.get("targetRepo"), "/workspaces/parameter-golf")

        measurement = report.get("measurement")
        self.assertIsInstance(measurement, dict)
        self.assertEqual(measurement.get("scoreMetric"), "FineWeb validation bits-per-byte")
        self.assertIsInstance(measurement.get("valBpb"), float)
        self.assertGreater(measurement["valBpb"], 0.0)

        artifact = report.get("artifact")
        self.assertIsInstance(artifact, dict)
        self.assertIsInstance(artifact.get("bytes"), int)
        self.assertGreater(artifact["bytes"], 0)
        self.assertLessEqual(artifact["bytes"], 16_000_000)
        artifact_path = REPO_ROOT / artifact["path"]
        self.assertTrue(artifact_path.exists(), "quantized control artifact is missing")
        self.assertEqual(artifact["quantizedArtifactBytes"], artifact_path.stat().st_size)

        reproduction = report.get("reproduction")
        self.assertIsInstance(reproduction, dict)
        self.assertIsInstance(reproduction.get("runtimeCommand"), str)
        self.assertIn("python3", reproduction["runtimeCommand"])
        self.assertIn("train_gpt.py", reproduction["runtimeCommand"])
        self.assertNotIn("PYTHONPATH=/tmp", reproduction["runtimeCommand"])
        self.assertIsInstance(reproduction.get("configProvenance"), dict)
        self.assertTrue(reproduction["configProvenance"].get("sourceFiles"))
        self.assertNotEqual(reproduction["configProvenance"].get("env", {}).get("PYTHONPATH"), "/tmp")
        self.assertTrue(reproduction.get("logPath"))
        self.assertTrue((REPO_ROOT / reproduction["logPath"]).exists(), "control stdout log is missing")

        classification = report.get("classification")
        self.assertIsInstance(classification, dict)
        self.assertIn(classification.get("status"), {"leaderboard-legal", "non-record-only", "uncertain"})
        self.assertEqual(
            classification,
            classify_submission_legality(
                {
                    "artifactBytes": artifact["bytes"],
                    "objectiveMetricName": measurement["scoreMetric"],
                    "runtimeSeconds": reproduction.get("runtimeSeconds"),
                    "hardware": reproduction.get("hardware"),
                    "selfContainedArtifact": artifact.get("selfContained"),
                    "externalDownloadsDuringEvaluation": reproduction.get("externalDownloadsDuringEvaluation"),
                    "networkAccessDuringEvaluation": reproduction.get("networkAccessDuringEvaluation"),
                    "usesValidationDataDuringTraining": reproduction.get("usesValidationDataDuringTraining"),
                    "credibleTenMinutePath": reproduction.get("credibleTenMinutePath"),
                    "track": reproduction.get("track"),
                }
            ),
        )


if __name__ == "__main__":
    unittest.main()
