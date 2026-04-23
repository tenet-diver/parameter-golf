import json
import tempfile
import unittest
from pathlib import Path

from fastest.scripts.run_cheap_screen_candidate import execute_cheap_screen_candidate


class Fast4CheapScreenPipelineContractTest(unittest.TestCase):
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
                    "promotionPolicy": {
                        "policyVersion": "pg-constraint-clarity-v1",
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

    def _task(self) -> dict:
        return {
            "taskId": "FAST-4",
            "lane": "cheap-screen",
            "candidateId": "exp-fast4-cheap-screen-001",
            "runConfig": {"seed": 42},
            "budgetCaps": {"maxRuntimeSeconds": 30},
            "traceId": "trace-fast4-001",
        }

    def _successful_runner_result(self) -> dict:
        return {
            "status": "success",
            "objectiveMetricName": "benchmark-score",
            "objectiveValue": 1.1,
            "completedAt": "2026-04-23T21:00:00.000Z",
            "failureCode": None,
            "failureMessage": None,
            "artifacts": {"evidenceId": "evidence-fast4-cheap-screen-001"},
        }

    def test_success_appends_one_record_and_regenerates_views(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            evidence_path, status_path, state_path = self._seed_files(tmp_root)
            task = self._task()

            def runner(_: dict) -> dict:
                return self._successful_runner_result()

            result = execute_cheap_screen_candidate(
                task,
                runner=runner,
                evidence_path=evidence_path,
                status_path=status_path,
                state_path=state_path,
            )

            self.assertEqual(result["status"], "success")
            self.assertEqual(result["reasonCode"], "cheap-screen-success")

            evidence = json.loads(evidence_path.read_text())
            self.assertEqual(len(evidence["experimentRecords"]), 2)
            appended = evidence["experimentRecords"][-1]
            self.assertEqual(appended["lane"], "cheap-screen")
            self.assertEqual(appended["experimentId"], "exp-fast4-cheap-screen-001")
            self.assertEqual(appended["status"], "accepted")

            self.assertEqual(evidence["summary"]["totalExperiments"], 2)
            self.assertEqual(evidence["summary"]["acceptedExperiments"], 2)
            self.assertEqual(
                evidence["summary"]["mostRecentEvidenceId"],
                "evidence-fast4-cheap-screen-001",
            )
            self.assertEqual(
                evidence["recentCompletedExperiments"][-1],
                "exp-fast4-cheap-screen-001",
            )

            status = json.loads(status_path.read_text())
            state = json.loads(state_path.read_text())
            self.assertEqual(status["progress"]["totalExperiments"], 2)
            self.assertEqual(state["evidenceSummaryCache"]["totalExperiments"], 2)

    def test_duplicate_replay_is_idempotent_and_does_not_append(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            evidence_path, status_path, state_path = self._seed_files(tmp_root)
            task = self._task()

            execute_cheap_screen_candidate(
                task,
                runner=lambda _: self._successful_runner_result(),
                evidence_path=evidence_path,
                status_path=status_path,
                state_path=state_path,
            )
            second = execute_cheap_screen_candidate(
                task,
                runner=lambda _: self._successful_runner_result(),
                evidence_path=evidence_path,
                status_path=status_path,
                state_path=state_path,
            )

            evidence = json.loads(evidence_path.read_text())
            self.assertEqual(second["reasonCode"], "duplicate-task-replay")
            self.assertEqual(len(evidence["experimentRecords"]), 2)

    def test_corrupt_authoritative_evidence_fails_closed_without_mutating_views(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            evidence_path, status_path, state_path = self._seed_files(tmp_root)
            task = self._task()
            status_before = status_path.read_text()
            state_before = state_path.read_text()

            evidence_path.write_text("{bad-json")

            result = execute_cheap_screen_candidate(
                task,
                runner=lambda _: self._successful_runner_result(),
                evidence_path=evidence_path,
                status_path=status_path,
                state_path=state_path,
            )

            self.assertEqual(result["status"], "error")
            self.assertEqual(result["reasonCode"], "evidence-io-corrupt")
            self.assertEqual(status_path.read_text(), status_before)
            self.assertEqual(state_path.read_text(), state_before)

    def test_non_numeric_metric_is_schema_validation_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            evidence_path, status_path, state_path = self._seed_files(tmp_root)

            result = execute_cheap_screen_candidate(
                self._task(),
                runner=lambda _: {
                    **self._successful_runner_result(),
                    "objectiveValue": "1.1",
                },
                evidence_path=evidence_path,
                status_path=status_path,
                state_path=state_path,
            )

            self.assertEqual(result["status"], "error")
            self.assertEqual(result["reasonCode"], "schema-validation")

    def test_success_without_evidence_id_is_integrity_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            evidence_path, status_path, state_path = self._seed_files(tmp_root)

            result = execute_cheap_screen_candidate(
                self._task(),
                runner=lambda _: {
                    **self._successful_runner_result(),
                    "artifacts": {},
                },
                evidence_path=evidence_path,
                status_path=status_path,
                state_path=state_path,
            )

            self.assertEqual(result["status"], "error")
            self.assertEqual(result["reasonCode"], "integrity-error")

    def test_regeneration_failure_reports_partial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            evidence_path, status_path, state_path = self._seed_files(tmp_root)
            status_path.unlink()

            result = execute_cheap_screen_candidate(
                self._task(),
                runner=lambda _: self._successful_runner_result(),
                evidence_path=evidence_path,
                status_path=status_path,
                state_path=state_path,
            )

            self.assertEqual(result["status"], "partial")
            self.assertEqual(result["reasonCode"], "regeneration-write-failed")

    def test_no_success_result_emits_recoverable_no_results_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            evidence_path, status_path, state_path = self._seed_files(tmp_root)

            result = execute_cheap_screen_candidate(
                self._task(),
                runner=lambda _: {
                    **self._successful_runner_result(),
                    "status": "failed",
                    "failureCode": "benchmark-timeout",
                    "failureMessage": "runtime cap reached",
                },
                evidence_path=evidence_path,
                status_path=status_path,
                state_path=state_path,
            )

            evidence = json.loads(evidence_path.read_text())
            self.assertEqual(result["status"], "error")
            self.assertEqual(result["reasonCode"], "no-results-produced")
            self.assertEqual(evidence["summary"]["totalExperiments"], 2)
            self.assertEqual(evidence["summary"]["acceptedExperiments"], 1)
            self.assertEqual(evidence["experimentRecords"][-1]["status"], "failed-terminal")


if __name__ == "__main__":
    unittest.main()
