import json
import sqlite3
import subprocess
import tempfile
import unittest
from pathlib import Path


class Fast14NextCombinationCandidateTest(unittest.TestCase):
    def _seed_files(self, tmp_root: Path) -> tuple[Path, Path, Path]:
        evidence_path = tmp_root / "measurement_evidence.json"
        status_path = tmp_root / "campaign_status.json"
        state_path = tmp_root / "campaign_state.json"

        evidence_path.write_text(
            json.dumps(
                {
                    "schemaVersion": 1,
                    "kind": "parameter-golf-measurement-evidence",
                    "updatedAt": "2026-04-23T20:21:00.000Z",
                    "summary": {
                        "artifactIds": ["evidence-fast1-control-20260423-001"],
                        "trustedControlState": "established",
                        "totalExperiments": 1,
                        "acceptedExperiments": 1,
                        "mostRecentEvidenceId": "evidence-fast1-control-20260423-001",
                    },
                    "recentCompletedExperiments": ["exp-fast1-control-20260423-001"],
                    "experimentRecords": [
                        {
                            "experimentId": "exp-fast1-control-20260423-001",
                            "evidenceId": "evidence-fast1-control-20260423-001",
                            "lane": "measurement",
                            "status": "accepted",
                            "completedAt": "2026-04-23T20:21:00.000Z",
                            "objectiveMetricName": "benchmark-score",
                            "objectiveValue": 1.0,
                        }
                    ],
                },
                indent=2,
            )
            + "\n"
        )
        status_path.write_text(
            json.dumps(
                {
                    "progress": {
                        "totalExperiments": 1,
                        "acceptedExperiments": 1,
                        "mostRecentEvidenceId": "evidence-fast1-control-20260423-001",
                    },
                    "recentCompletedExperiments": ["exp-fast1-control-20260423-001"],
                    "blockers": [],
                    "adapterStatus": {"details": {}},
                },
                indent=2,
            )
            + "\n"
        )
        state_path.write_text(
            json.dumps(
                {
                    "campaign": {"metadata": {"artifactBudgetPolicy": "required-before-promotion"}},
                    "evidenceSummaryCache": {},
                },
                indent=2,
            )
            + "\n"
        )
        return evidence_path, status_path, state_path

    def _make_task_store(self, sqlite_path: Path, pipeline_json: dict) -> None:
        conn = sqlite3.connect(sqlite_path)
        try:
            conn.execute(
                """
                CREATE TABLE task_store_tasks (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    status_order INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    pipeline_json TEXT
                )
                """
            )
            conn.execute(
                """
                INSERT INTO task_store_tasks (id, status, status_order, created_at, pipeline_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    "FAST-14-CANDIDATE",
                    "ready",
                    0,
                    "2026-04-24T00:00:00Z",
                    json.dumps(pipeline_json),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def test_cli_executes_next_runnable_combination_task_from_task_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            evidence_path, status_path, state_path = self._seed_files(tmp_root)
            output_path = tmp_root / "outcome.json"
            task_store_dir = tmp_root / "task-store"
            task_store_dir.mkdir(parents=True, exist_ok=True)
            sqlite_path = task_store_dir / "task-store.sqlite"
            self._make_task_store(
                sqlite_path,
                {
                    "lane": "combination",
                    "candidateId": "exp-fast14-combination-001",
                    "traceId": "trace-fast14-001",
                    "runConfig": {
                        "seed": 11,
                        "motifIds": ["motif-cheap-screen-a", "motif-ablation-b"],
                    },
                    "budgetCaps": {"maxRuntimeSeconds": 30, "maxCombinationExperiments": 1},
                },
            )

            repo_root = Path(__file__).resolve().parents[1]
            command = [
                "python",
                "fastest/scripts/run_combination_candidate.py",
                "--task-store-dir",
                str(task_store_dir),
                "--output",
                str(output_path),
                "--evidence-path",
                str(evidence_path),
                "--status-path",
                str(status_path),
                "--state-path",
                str(state_path),
            ]

            completed = subprocess.run(
                command,
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, msg=completed.stderr)

            outcome = json.loads(output_path.read_text())
            self.assertEqual(outcome["taskId"], "FAST-14-CANDIDATE")
            self.assertEqual(outcome["status"], "success")
            self.assertEqual(outcome["reasonCode"], "combination-success")

            evidence = json.loads(evidence_path.read_text())
            self.assertEqual(evidence["summary"]["totalExperiments"], 2)
            self.assertEqual(evidence["summary"]["acceptedExperiments"], 2)
            self.assertEqual(evidence["experimentRecords"][-1]["lane"], "combination")
            self.assertEqual(evidence["experimentRecords"][-1]["taskId"], "FAST-14-CANDIDATE")


if __name__ == "__main__":
    unittest.main()
