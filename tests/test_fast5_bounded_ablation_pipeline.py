import json
import tempfile
import unittest
from pathlib import Path

from fastest.scripts.run_bounded_ablation_candidate import execute_bounded_ablation_candidate


class Fast5BoundedAblationPipelineContractTest(unittest.TestCase):
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

    def _task(self, candidate_id: str = "exp-fast5-ablation-001") -> dict:
        return {
            "taskId": "FAST-5",
            "lane": "ablation",
            "candidateId": candidate_id,
            "runConfig": {"seed": 314},
            "budgetCaps": {"maxRuntimeSeconds": 30, "maxAblationExperiments": 1},
            "traceId": "trace-fast5-001",
        }

    def _successful_runner_result(self, evidence_id: str) -> dict:
        return {
            "status": "success",
            "objectiveMetricName": "benchmark-score",
            "objectiveValue": 1.02,
            "completedAt": "2026-04-23T21:00:00.000Z",
            "failureCode": None,
            "failureMessage": None,
            "artifacts": {"evidenceId": evidence_id},
        }

    def test_success_appends_single_ablation_record_and_updates_views(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            evidence_path, status_path, state_path = self._seed_files(tmp_root)

            result = execute_bounded_ablation_candidate(
                self._task(),
                runner=lambda _: self._successful_runner_result("evidence-fast5-ablation-001"),
                evidence_path=evidence_path,
                status_path=status_path,
                state_path=state_path,
            )

            self.assertEqual(result["status"], "success")
            self.assertEqual(result["reasonCode"], "ablation-success")

            evidence = json.loads(evidence_path.read_text())
            self.assertEqual(evidence["summary"]["totalExperiments"], 2)
            self.assertEqual(evidence["summary"]["acceptedExperiments"], 2)
            self.assertEqual(evidence["experimentRecords"][-1]["lane"], "ablation")
            self.assertEqual(evidence["experimentRecords"][-1]["experimentId"], "exp-fast5-ablation-001")

            status = json.loads(status_path.read_text())
            state = json.loads(state_path.read_text())
            self.assertEqual(status["progress"]["totalExperiments"], 2)
            self.assertEqual(state["evidenceSummaryCache"]["totalExperiments"], 2)

    def test_second_distinct_ablation_is_rejected_when_budget_is_exhausted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            evidence_path, status_path, state_path = self._seed_files(tmp_root)

            first = execute_bounded_ablation_candidate(
                self._task("exp-fast5-ablation-001"),
                runner=lambda _: self._successful_runner_result("evidence-fast5-ablation-001"),
                evidence_path=evidence_path,
                status_path=status_path,
                state_path=state_path,
            )
            second = execute_bounded_ablation_candidate(
                self._task("exp-fast5-ablation-002"),
                runner=lambda _: self._successful_runner_result("evidence-fast5-ablation-002"),
                evidence_path=evidence_path,
                status_path=status_path,
                state_path=state_path,
            )

            self.assertEqual(first["status"], "success")
            self.assertEqual(second["status"], "error")
            self.assertEqual(second["reasonCode"], "ablation-budget-exhausted")

            evidence = json.loads(evidence_path.read_text())
            ablation_records = [
                record for record in evidence["experimentRecords"] if record.get("lane") == "ablation"
            ]
            self.assertEqual(len(ablation_records), 1)

    def test_failed_first_ablation_still_exhausts_budget_when_cap_is_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            evidence_path, status_path, state_path = self._seed_files(tmp_root)

            first = execute_bounded_ablation_candidate(
                self._task("exp-fast5-ablation-001"),
                runner=lambda _: {
                    "status": "failed-terminal",
                    "objectiveMetricName": "benchmark-score",
                    "objectiveValue": 0.98,
                    "completedAt": "2026-04-23T21:00:00.000Z",
                    "failureCode": "benchmark-regression",
                    "failureMessage": "did not meet baseline",
                    "artifacts": {},
                },
                evidence_path=evidence_path,
                status_path=status_path,
                state_path=state_path,
            )
            second = execute_bounded_ablation_candidate(
                self._task("exp-fast5-ablation-002"),
                runner=lambda _: self._successful_runner_result("evidence-fast5-ablation-002"),
                evidence_path=evidence_path,
                status_path=status_path,
                state_path=state_path,
            )

            self.assertEqual(first["status"], "error")
            self.assertEqual(first["reasonCode"], "no-results-produced")
            self.assertEqual(second["status"], "error")
            self.assertEqual(second["reasonCode"], "ablation-budget-exhausted")

            evidence = json.loads(evidence_path.read_text())
            ablation_records = [
                record for record in evidence["experimentRecords"] if record.get("lane") == "ablation"
            ]
            self.assertEqual(len(ablation_records), 1)
            self.assertEqual(ablation_records[0]["status"], "failed-terminal")


if __name__ == "__main__":
    unittest.main()
