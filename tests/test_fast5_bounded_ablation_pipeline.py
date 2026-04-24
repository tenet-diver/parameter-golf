import json
import tempfile
import unittest
from pathlib import Path

from fastest.scripts.run_bounded_ablation_candidate import execute_bounded_ablation_candidate
from fastest.scripts.run_fast5_visible_motif_ablation_screen import (
    execute_visible_motif_ablation_screen,
)


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


class Fast5VisibleMotifAblationScreenTest(unittest.TestCase):
    def _matrix(self) -> dict:
        return {
            "schemaVersion": 1,
            "taskId": "FAST-5",
            "artifactLimitBytes": 16_000_000,
            "trustedControl": {
                "baselineId": "control-fast1-cpu-smoke",
                "configHash": "control-config-sha",
                "commit": "abc1234",
                "dataset": "fineweb_val_v1",
                "scoringCommand": "python measurement/control_pipeline.py score",
                "artifactMeasurementCommand": "python fastest/scripts/artifact_budget_analyzer.py",
                "seedPolicy": {"mode": "fixed", "seeds": [42]},
                "scoreBpb": 4.0,
                "artifactBytes": 1_000_000,
            },
            "ablations": [
                {
                    "runId": "fast5-sp8192",
                    "motif": "SP8192 tokenizer",
                    "variantConfigRef": "records/sp8192/train_gpt.py",
                    "seed": 42,
                    "scoreBpb": 3.95,
                    "artifactBytes": 17_000_000,
                    "legalityClass": "legal",
                    "runtimeSeconds": 420,
                    "status": "success",
                    "nextAction": "combine-later",
                },
                {
                    "runId": "fast5-3-layer-recurrence",
                    "motif": "3-layer recurrence",
                    "variantConfigRef": "records/recur/train_gpt.py",
                    "seed": 42,
                    "scoreBpb": 3.98,
                    "artifactBytes": 1_030_000,
                    "legalityClass": "legal",
                    "runtimeSeconds": 390,
                    "status": "success",
                    "nextAction": "deepen",
                },
                {
                    "runId": "fast5-parallel-residuals",
                    "motif": "parallel residuals",
                    "variantConfigRef": "records/parres/train_gpt.py",
                    "seed": 42,
                    "scoreBpb": 4.01,
                    "artifactBytes": 1_010_000,
                    "legalityClass": "legal",
                    "runtimeSeconds": 380,
                    "status": "success",
                    "nextAction": "drop",
                },
                {
                    "runId": "fast5-score-first-ttt",
                    "motif": "legal score-first TTT",
                    "variantConfigRef": "records/ttt/train_gpt.py",
                    "seed": 42,
                    "scoreBpb": 3.99,
                    "artifactBytes": 1_050_000,
                    "legalityClass": "legal",
                    "legalityNotes": "Score-first evaluation only; no validation-informed training or inference adaptation.",
                    "runtimeSeconds": 450,
                    "status": "success",
                    "nextAction": "deepen",
                },
                {
                    "runId": "fast5-qk-gain",
                    "motif": "QK-gain tuning",
                    "variantConfigRef": "records/qk/train_gpt.py",
                    "seed": 42,
                    "scoreBpb": 3.999,
                    "artifactBytes": 1_000_500,
                    "legalityClass": "legal",
                    "runtimeSeconds": 360,
                    "status": "success",
                    "nextAction": "rerun-near-threshold",
                },
                {
                    "runId": "fast5-hessian-aware-clipping",
                    "motif": "Hessian-aware clipping",
                    "variantConfigRef": "records/hessian/train_gpt.py",
                    "seed": 42,
                    "scoreBpb": 4.02,
                    "artifactBytes": 1_000_200,
                    "legalityClass": "legal",
                    "runtimeSeconds": 365,
                    "status": "success",
                    "nextAction": "drop",
                },
            ],
        }

    def test_screen_writes_deltas_legality_and_ranked_handoff_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            matrix_path = tmp_root / "matrix.json"
            output_path = tmp_root / "results.json"
            summary_path = tmp_root / "summary.md"
            matrix_path.write_text(json.dumps(self._matrix(), indent=2) + "\n")

            outcome = execute_visible_motif_ablation_screen(
                matrix_path=matrix_path,
                output_path=output_path,
                summary_path=summary_path,
            )

            self.assertEqual(outcome["status"], "success")
            self.assertEqual(outcome["reasonCode"], "fast5-screen-complete")

            results = json.loads(output_path.read_text())
            rows = results["rows"]
            self.assertEqual(results["taskId"], "FAST-5")
            self.assertEqual(results["baseline"]["baselineId"], "control-fast1-cpu-smoke")
            self.assertEqual({row["motif"] for row in rows}, {
                "SP8192 tokenizer",
                "3-layer recurrence",
                "parallel residuals",
                "legal score-first TTT",
                "QK-gain tuning",
                "Hessian-aware clipping",
            })
            self.assertTrue(all(row["baseline_id"] == "control-fast1-cpu-smoke" for row in rows))
            self.assertTrue(all("score_delta_bpb" in row for row in rows))
            self.assertTrue(all("artifact_delta_bytes" in row for row in rows))

            sp8192 = next(row for row in rows if row["motif"] == "SP8192 tokenizer")
            self.assertEqual(sp8192["score_delta_bpb"], -0.05)
            self.assertEqual(sp8192["artifact_delta_bytes"], 16_000_000)
            self.assertEqual(sp8192["legality_class"], "artifact-limit-fail")
            self.assertEqual(sp8192["next_action"], "shrink-or-drop")

            ttt = next(row for row in rows if row["motif"] == "legal score-first TTT")
            self.assertIn("Score-first evaluation only", ttt["legality_notes"])

            ranked = [row["motif"] for row in results["rankedRows"]]
            self.assertEqual(ranked[0], "SP8192 tokenizer")
            self.assertEqual(ranked[-1], "Hessian-aware clipping")
            self.assertIn("SP8192 tokenizer", summary_path.read_text())
            self.assertIn("shrink-or-drop", summary_path.read_text())


if __name__ == "__main__":
    unittest.main()
