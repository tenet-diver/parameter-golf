import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
GATE_PATH = REPO_ROOT / "planning/backlog-validation-gate.json"
AUTHORITATIVE_TASK_STORE_DIR = (
    "/home/codespace/.fastest/orchestrator/projects/parameter-golf-fastest-run-2e398b79d13c/runtime/tasks"
)


class Fast10BacklogValidationGateContractTest(unittest.TestCase):
    def test_gate_artifact_satisfies_fast10_contract(self) -> None:
        gate = json.loads(GATE_PATH.read_text())

        self.assertEqual(gate["taskId"], "FAST-10")
        self.assertEqual(
            gate["sourcePlanning"],
            {
                "taskId": "FAST-7",
                "title": "Plan backlog replenishment after major backlog depletion alert",
            },
        )

        checks = gate["validationChecklist"]
        check_ids = [check["checkId"] for check in checks]
        self.assertEqual(
            check_ids,
            [
                "open-task-check",
                "recent-run-fingerprint-collision-check",
                "artifact-lineage-check",
            ],
        )

        open_task_check = checks[0]
        self.assertIn(
            f'TASK_STORE_DIR="{AUTHORITATIVE_TASK_STORE_DIR}"',
            open_task_check["command"],
        )
        self.assertIn(
            'sqlite_override=os.environ.get("TASK_STORE_SQLITE_FILE", "").strip()',
            open_task_check["command"],
        )
        self.assertIn(
            "os.path.join(task_store_dir, sqlite_override)",
            open_task_check["command"],
        )
        self.assertIn(
            "sqlite_override if os.path.isabs(sqlite_override)",
            open_task_check["command"],
        )
        self.assertIn(
            "os.access(sqlite_file, os.R_OK)",
            open_task_check["command"],
        )
        self.assertIn(
            "except sqlite3.OperationalError as exc",
            open_task_check["command"],
        )

        for check in checks:
            record_fields = check["outputRecordFields"]
            self.assertEqual(
                record_fields,
                ["checkId", "command", "observedResult", "timestamp"],
            )
            self.assertTrue(check["observedResult"])
            self.assertRegex(
                check["timestamp"],
                r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$",
            )

        self.assertIn(
            "Open FAST tasks checked (0): none",
            checks[0]["observedResult"],
        )
        self.assertIn(
            "Recent planning runs checked (0): none",
            checks[1]["observedResult"],
        )
        self.assertIn(
            "triggerFingerprint",
            checks[2]["observedResult"],
        )
        self.assertIn(
            "triggerArtifactId",
            checks[2]["observedResult"],
        )
        self.assertIn(
            "dedupeEvidence",
            checks[2]["observedResult"],
        )

        self.assertEqual(
            gate["rejectionCriteria"]["missingLineageFields"],
            ["triggerFingerprint", "triggerArtifactId", "dedupeEvidence"],
        )
        self.assertIn(
            "missing any required lineage field",
            gate["rejectionCriteria"]["action"],
        )

        self.assertIn("verbatim replay", gate["qaHandoff"]["validationRule"])
        self.assertIn("no additional interpretation", gate["qaHandoff"]["validationRule"])

        self.assertIn("all checklist commands replay", gate["gateBehavior"]["passPath"])
        self.assertIn("rejected", gate["gateBehavior"]["failPath"])


if __name__ == "__main__":
    unittest.main()
