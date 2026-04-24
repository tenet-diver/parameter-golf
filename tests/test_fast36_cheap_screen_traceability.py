import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = REPO_ROOT / "fastest/source/measurement_evidence.json"
OUTCOME_PATH = REPO_ROOT / "fastest/generated/fast36_next_run_outcome.json"
RUNNER_SCRIPT = "fastest/scripts/run_cheap_screen_candidate.py"


class Fast36CheapScreenTraceabilityTest(unittest.TestCase):
    def test_fast36_records_store_exact_runnable_commands(self) -> None:
        evidence = json.loads(EVIDENCE_PATH.read_text())
        records = evidence.get("experimentRecords", [])
        self.assertIsInstance(records, list)

        fast36_records = [
            record
            for record in records
            if isinstance(record, dict)
            and record.get("taskId") in {"FAST-36-cheap-screen-002", "FAST-36-cheap-screen-003"}
            and record.get("lane") == "cheap-screen"
        ]
        self.assertEqual(len(fast36_records), 2)

        by_task_id = {record["taskId"]: record for record in fast36_records}
        command_002 = by_task_id["FAST-36-cheap-screen-002"]["executionCommands"][0]
        self.assertEqual(
            command_002,
            "python fastest/scripts/run_cheap_screen_candidate.py --task-store-dir "
            "/home/codespace/.fastest/orchestrator/projects/parameter-golf-2e398b79d13c/runtime/tasks "
            "--output fastest/generated/fast36_next_run_outcome.json",
        )

        for record in fast36_records:
            commands = record.get("executionCommands")
            self.assertIsInstance(commands, list)
            self.assertTrue(commands)
            self.assertTrue(
                all(
                    isinstance(command, str)
                    and command
                    and RUNNER_SCRIPT in command
                    for command in commands
                )
            )

    def test_fast36_outcome_keeps_cheap_screen_contract_fields(self) -> None:
        outcome = json.loads(OUTCOME_PATH.read_text())

        self.assertEqual(outcome.get("taskId"), "FAST-36-cheap-screen-003")
        self.assertEqual(outcome.get("candidateId"), "fast36-candidate-003")
        self.assertEqual(outcome.get("status"), "success")
        self.assertEqual(outcome.get("reasonCode"), "cheap-screen-success")
        self.assertEqual(outcome.get("observedMetric"), {"name": "benchmark-score", "value": 1.0891})

        commands = outcome.get("executionCommands")
        self.assertIsInstance(commands, list)
        self.assertTrue(commands)
        self.assertTrue(all(RUNNER_SCRIPT in command for command in commands))

        artifact_paths = outcome.get("artifactPaths")
        self.assertIsInstance(artifact_paths, list)
        self.assertIn("fastest/source/measurement_evidence.json", artifact_paths)
        self.assertIn("fastest/generated/campaign_status.json", artifact_paths)
        self.assertIn("fastest/generated/campaign_state.json", artifact_paths)

        promotion_rationale = outcome.get("promotionRationale")
        self.assertIsInstance(promotion_rationale, dict)
        self.assertEqual(promotion_rationale.get("decision"), "hold")
        self.assertIn("reproducibility-manifest", promotion_rationale.get("missingEvidence", []))
        self.assertTrue(promotion_rationale.get("reason"))


if __name__ == "__main__":
    unittest.main()
