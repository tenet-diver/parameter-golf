import json
import subprocess
import tempfile
import unittest
from pathlib import Path


class Fast35NonRecordExplorationExecutionPacketTest(unittest.TestCase):
    def _seed_files(self, tmp_root: Path) -> tuple[Path, Path, Path, Path, Path]:
        evidence_path = tmp_root / "measurement_evidence.json"
        status_path = tmp_root / "campaign_status.json"
        state_path = tmp_root / "campaign_state.json"
        task_path = tmp_root / "fast35_non_record_task.json"
        output_path = tmp_root / "fast35_non_record_outcome.json"

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

        task_path.write_text(
            json.dumps(
                {
                    "taskId": "FAST-35",
                    "candidateId": "exp-fast35-non-record-001",
                    "traceId": "trace-fast35-non-record-001",
                    "lane": "non-record-exploration",
                    "runConfig": {
                        "seed": 35,
                        "hypothesis": "activation-sandwich-low-rank-residual",
                    },
                    "budgetCaps": {
                        "maxRuntimeSeconds": 30,
                        "maxNonRecordExplorationExperiments": 1,
                    },
                },
                indent=2,
            )
            + "\n"
        )

        return evidence_path, status_path, state_path, task_path, output_path

    def test_cli_task_mode_writes_metric_and_non_promotion_rationale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            evidence_path, status_path, state_path, task_path, output_path = self._seed_files(tmp_root)

            repo_root = Path(__file__).resolve().parents[1]
            command = [
                "python",
                "fastest/scripts/run_non_record_exploration_candidate.py",
                "--task",
                str(task_path),
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
            self.assertEqual(outcome["status"], "success")
            self.assertEqual(outcome["reasonCode"], "non-record-exploration-success")
            self.assertEqual(outcome["observedMetric"], {"name": "benchmark-score", "value": 1.0793})

            artifact_paths = outcome.get("artifactPaths")
            self.assertIsInstance(artifact_paths, list)
            self.assertIn(str(evidence_path), artifact_paths)
            self.assertIn(str(status_path), artifact_paths)
            self.assertIn(str(state_path), artifact_paths)
            self.assertIn(str(output_path), artifact_paths)

            promotion_rationale = outcome.get("promotionRationale")
            self.assertIsInstance(promotion_rationale, dict)
            self.assertEqual(promotion_rationale.get("decision"), "reject")
            self.assertIn("non-record exploration", promotion_rationale.get("reason", ""))


if __name__ == "__main__":
    unittest.main()
