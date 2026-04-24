import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = REPO_ROOT / "fastest/source/measurement_evidence.json"


class Fast31MeasurementCoverageRefreshTest(unittest.TestCase):
    def test_fast31_execution_attempt_packet_has_required_measurement_fields(self) -> None:
        source = json.loads(SOURCE_PATH.read_text())

        attempts = source.get("executionAttempts")
        self.assertIsInstance(attempts, list)

        fast31_attempts = [
            attempt
            for attempt in attempts
            if isinstance(attempt, dict)
            and attempt.get("taskId") == "FAST-31"
            and attempt.get("lane") == "measurement"
        ]
        self.assertTrue(fast31_attempts)

        attempt = fast31_attempts[-1]
        self.assertIn("candidateId", attempt)

        commands = attempt.get("commands")
        self.assertIsInstance(commands, list)
        self.assertTrue(commands)
        self.assertTrue(all(isinstance(command, str) and command.strip() for command in commands))
        self.assertFalse(any(command.strip().lower() in {"todo", "tbd", "none"} for command in commands))

        artifact_paths = attempt.get("artifactPaths")
        self.assertIsInstance(artifact_paths, list)
        self.assertTrue(artifact_paths)
        self.assertTrue(all(isinstance(path, str) and path.strip() for path in artifact_paths))

        observed_metric_output = attempt.get("observedMetricOutput")
        self.assertIsInstance(observed_metric_output, dict)
        self.assertIn(observed_metric_output.get("status"), {"observed", "partial", "blocked"})

        promotion_rationale = attempt.get("promotionDecisionRationale")
        self.assertIsInstance(promotion_rationale, dict)
        self.assertIn(promotion_rationale.get("decision"), {"promote", "hold", "reject"})
        self.assertIn(promotion_rationale.get("advanceOutcome"), {"advance", "retry", "reject"})

        self.assertIn("blockedReasonCode", attempt)
        self.assertIn("nextRecoveryTask", attempt)

        reproducibility = attempt.get("reproducibilityEvidence")
        self.assertIsInstance(reproducibility, dict)
        self.assertIn(reproducibility.get("type"), {"manifest", "waiver"})


if __name__ == "__main__":
    unittest.main()
