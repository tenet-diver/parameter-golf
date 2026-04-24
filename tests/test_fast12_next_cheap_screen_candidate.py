import json
import os
import sqlite3
import subprocess
import tempfile
import unittest
from pathlib import Path

from fastest.scripts.run_cheap_screen_candidate import (
    run_next_cheap_screen_candidate,
    select_next_candidate,
)


class Fast12NextCheapScreenCandidateTest(unittest.TestCase):
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
                    "FAST-12-CANDIDATE",
                    "ready",
                    0,
                    "2026-04-24T00:00:00Z",
                    json.dumps(pipeline_json),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def test_select_next_candidate_prefers_lowest_status_order_then_created_at(self) -> None:
        backlog = [
            {
                "taskId": "FAST-20",
                "status": "queued",
                "statusOrder": 5,
                "createdAt": "2026-04-24T00:00:10Z",
                "lane": "cheap-screen",
                "candidateId": "exp-b",
            },
            {
                "taskId": "FAST-19",
                "status": "queued",
                "statusOrder": 5,
                "createdAt": "2026-04-24T00:00:05Z",
                "lane": "cheap-screen",
                "candidateId": "exp-a",
            },
            {
                "taskId": "FAST-18",
                "status": "in_progress",
                "statusOrder": 1,
                "createdAt": "2026-04-24T00:00:01Z",
                "lane": "cheap-screen",
                "candidateId": "exp-in-progress",
            },
            {
                "taskId": "FAST-17",
                "status": "queued",
                "statusOrder": 0,
                "createdAt": "2026-04-24T00:00:01Z",
                "lane": "ablation",
                "candidateId": "exp-ablation",
            },
        ]

        selected = select_next_candidate(backlog, lane="cheap-screen")
        self.assertIsNotNone(selected)
        self.assertEqual(selected["taskId"], "FAST-19")
        self.assertEqual(selected["candidateId"], "exp-a")

    def test_run_next_candidate_returns_noop_when_no_cheap_screen_task_available(self) -> None:
        backlog = [
            {
                "taskId": "FAST-22",
                "status": "queued",
                "statusOrder": 0,
                "createdAt": "2026-04-24T00:00:00Z",
                "lane": "ablation",
                "candidateId": "exp-ablation",
            }
        ]
        runner_calls = 0

        def runner(_: dict) -> dict:
            nonlocal runner_calls
            runner_calls += 1
            return {"status": "success"}

        outcome = run_next_cheap_screen_candidate(backlog, runner=runner)
        self.assertEqual(outcome["status"], "success")
        self.assertEqual(outcome["reasonCode"], "no-task-available")
        self.assertEqual(runner_calls, 0)

    def test_run_next_candidate_rejects_invalid_lane(self) -> None:
        outcome = run_next_cheap_screen_candidate([], runner=lambda _: {}, lane="ablation")
        self.assertEqual(outcome["status"], "error")
        self.assertEqual(outcome["reasonCode"], "invalid-lane")

    def test_run_next_candidate_rejects_missing_task_id(self) -> None:
        backlog = [{"lane": "cheap-screen", "status": "ready", "candidateId": "exp-1"}]
        outcome = run_next_cheap_screen_candidate(backlog, runner=lambda _: {})
        self.assertEqual(outcome["status"], "error")
        self.assertEqual(outcome["reasonCode"], "task-io-invalid")
        self.assertIn("taskId", outcome["message"])

    def test_run_next_candidate_rejects_missing_candidate_id(self) -> None:
        backlog = [{"lane": "cheap-screen", "status": "ready", "taskId": "FAST-100"}]
        outcome = run_next_cheap_screen_candidate(backlog, runner=lambda _: {})
        self.assertEqual(outcome["status"], "error")
        self.assertEqual(outcome["reasonCode"], "task-io-invalid")
        self.assertIn("candidateId", outcome["message"])

    def test_run_next_candidate_delegates_execution_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            evidence_path, status_path, state_path = self._seed_files(tmp_root)
            backlog = [
                {
                    "taskId": "FAST-12-CANDIDATE",
                    "lane": "cheap-screen",
                    "status": "ready",
                    "statusOrder": 0,
                    "createdAt": "2026-04-24T00:00:00Z",
                    "candidateId": "exp-fast12-candidate-001",
                    "traceId": "trace-fast12-001",
                    "runConfig": {"seed": 42},
                    "budgetCaps": {"maxRuntimeSeconds": 30},
                }
            ]
            runner_calls = 0

            def runner(task: dict) -> dict:
                nonlocal runner_calls
                runner_calls += 1
                self.assertEqual(task["taskId"], "FAST-12-CANDIDATE")
                return {
                    "status": "success",
                    "objectiveMetricName": "benchmark-score",
                    "objectiveValue": 1.101,
                    "completedAt": "2026-04-24T00:00:01Z",
                    "failureCode": None,
                    "failureMessage": None,
                    "artifacts": {"evidenceId": "evidence-fast12-candidate-001"},
                }

            outcome = run_next_cheap_screen_candidate(
                backlog,
                runner=runner,
                evidence_path=evidence_path,
                status_path=status_path,
                state_path=state_path,
            )
            self.assertEqual(outcome["status"], "success")
            self.assertEqual(outcome["reasonCode"], "cheap-screen-success")
            self.assertEqual(runner_calls, 1)

            evidence = json.loads(evidence_path.read_text())
            self.assertEqual(evidence["summary"]["totalExperiments"], 2)
            self.assertEqual(evidence["experimentRecords"][-1]["taskId"], "FAST-12-CANDIDATE")

    def test_cli_executes_next_runnable_cheap_screen_task_from_task_store(self) -> None:
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
                    "lane": "cheap-screen",
                    "candidateId": "exp-fast12-cli-001",
                    "traceId": "trace-fast12-cli-001",
                    "runConfig": {"seed": 7},
                    "budgetCaps": {"maxRuntimeSeconds": 30},
                },
            )

            repo_root = Path(__file__).resolve().parents[1]
            command = [
                "python",
                "fastest/scripts/run_cheap_screen_candidate.py",
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

            self.assertEqual(completed.returncode, 0, completed.stderr)
            outcome = json.loads(output_path.read_text())
            self.assertEqual(outcome["status"], "success")
            self.assertEqual(outcome["reasonCode"], "cheap-screen-success")
            self.assertEqual(outcome["taskId"], "FAST-12-CANDIDATE")

            evidence = json.loads(evidence_path.read_text())
            self.assertEqual(evidence["summary"]["totalExperiments"], 2)
            self.assertEqual(evidence["experimentRecords"][-1]["taskId"], "FAST-12-CANDIDATE")

    def test_cli_ignores_out_of_root_sqlite_override_and_uses_authoritative_default(self) -> None:
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
                    "lane": "cheap-screen",
                    "candidateId": "exp-fast12-cli-override-001",
                    "traceId": "trace-fast12-cli-override-001",
                    "runConfig": {"seed": 11},
                    "budgetCaps": {"maxRuntimeSeconds": 30},
                },
            )

            repo_root = Path(__file__).resolve().parents[1]
            command = [
                "python",
                "fastest/scripts/run_cheap_screen_candidate.py",
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
            env = dict(os.environ)
            env["TASK_STORE_SQLITE_FILE"] = str(tmp_root / "outside.sqlite")

            completed = subprocess.run(
                command,
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            outcome = json.loads(output_path.read_text())
            self.assertEqual(outcome["status"], "success")
            self.assertEqual(outcome["reasonCode"], "cheap-screen-success")
            self.assertEqual(outcome["taskId"], "FAST-12-CANDIDATE")

            evidence = json.loads(evidence_path.read_text())
            self.assertEqual(evidence["summary"]["totalExperiments"], 2)
            self.assertEqual(evidence["experimentRecords"][-1]["taskId"], "FAST-12-CANDIDATE")


if __name__ == "__main__":
    unittest.main()
