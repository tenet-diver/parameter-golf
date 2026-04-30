import json
import tempfile
import unittest
from pathlib import Path

from fastest.scripts.maintain_ranked_idea_queue import build_ranked_idea_queue, write_ranked_idea_queue


REPO_ROOT = Path(__file__).resolve().parents[1]
QUEUE_PATH = REPO_ROOT / "planning/ranked-idea-queue.json"


class Fast8RankedIdeaQueueTest(unittest.TestCase):
    def test_queue_ranks_candidates_by_four_factor_composite(self) -> None:
        evidence = {
            "experimentRecords": [
                {
                    "experimentId": "local-winning-combo",
                    "lane": "combination",
                    "status": "accepted",
                    "objectiveMetricName": "benchmark-score",
                    "objectiveValue": 1.12,
                    "budgetCaps": {"maxRuntimeSeconds": 600},
                },
                {
                    "experimentId": "cheap-novel-probe",
                    "lane": "cheap-screen",
                    "status": "accepted",
                    "objectiveMetricName": "benchmark-score",
                    "objectiveValue": 1.04,
                    "budgetCaps": {"maxRuntimeSeconds": 30},
                },
                {
                    "experimentId": "imported-pr",
                    "lane": "submission-record",
                    "status": "accepted",
                    "objectiveMetricName": "val_bpb",
                    "objectiveValue": 1.0785,
                    "artifactBytes": 15978527,
                    "campaignStatus": "blocked-not-winning",
                    "reproductionStatus": {"status": "needs_external_8xh100_verification"},
                    "comparison": {"currentMergedSotaValBpb": 1.081, "deltaBpb": -0.0025},
                    "verificationGate": {"status": "blocked-not-winning"},
                    "budgetCaps": {"maxRuntimeSeconds": 600},
                },
            ],
            "summary": {"bestKnownBenchmarkScore": 1.0999},
        }
        ablation = {
            "baseline": {"scoreBpb": 1.11473509},
            "rankedRows": [
                {
                    "run_id": "local-ablation",
                    "motif": "3-layer recurrence",
                    "score_bpb": 1.081,
                    "score_delta_bpb": -0.033735,
                    "artifact_bytes": 15993232,
                    "runtime_seconds": 588,
                    "next_action": "deepen",
                    "status": "success",
                }
            ],
        }
        h100_results = [
            {
                "id": "baseline",
                "status": "completed",
                "family": "autoregressive",
                "hypothesis": "baseline_control",
                "finalValBpb": 4.0,
                "elapsedSeconds": 30,
                "int8SubmissionBytes": 500_000,
                "runDir": "/tmp/baseline",
                "rank": 2,
            },
            {
                "id": "local-batch-winner",
                "status": "completed",
                "family": "long_context_ar",
                "hypothesis": "long_context_qk_norm_improves_byte_compression",
                "hypothesisTags": ["long_context", "qk_norm"],
                "finalValBpb": 3.75,
                "elapsedSeconds": 42,
                "int8SubmissionBytes": 700_000,
                "runDir": "/tmp/local-batch-winner",
                "rank": 1,
            },
        ]

        queue = build_ranked_idea_queue(evidence=evidence, ablation_results=ablation, h100_results=h100_results)
        rows = queue["rankedIdeas"]

        self.assertEqual(queue["competitionPreset"], "parameter-golf")
        self.assertEqual(queue["rankingFormula"]["componentWeights"]["expectedInfoGain"], 0.35)
        self.assertEqual(rows[0]["ideaId"], "local-h100-batch-local-batch-winner")
        self.assertEqual(rows[0]["scores"]["expectedUpside"], 1.0)
        self.assertGreater(rows[0]["scores"]["composite"], rows[1]["scores"]["composite"])

        for row in rows:
            self.assertIn(
                row["sourceKind"],
                {"local-experiment", "imported-evidence", "local-ablation", "local-h100-batch"},
            )
            self.assertEqual(
                sorted(row["scores"]),
                ["composite", "cost", "expectedInfoGain", "expectedUpside", "mergeability"],
            )
            self.assertIsInstance(row["nextPlanningAction"], str)
            self.assertTrue(row["evidenceRefs"])

        imported = next(row for row in rows if row["ideaId"] == "imported-evidence-imported-pr")
        self.assertEqual(imported["scores"]["expectedUpside"], 1.0)
        self.assertLess(imported["scores"]["mergeability"], 0.4)
        self.assertIn("needs_external_8xh100_verification", imported["mergeabilityNotes"])
        self.assertEqual(imported["submissionClassification"]["status"], "uncertain")
        self.assertIn(
            "missing-self-contained-evidence",
            imported["submissionClassification"]["uncertaintyReasons"],
        )

        local_batch = next(row for row in rows if row["ideaId"] == "local-h100-batch-local-batch-winner")
        self.assertEqual(local_batch["sourceKind"], "local-h100-batch")
        self.assertEqual(local_batch["sourceLane"], "h100-candidate-batch")
        self.assertEqual(local_batch["scores"]["expectedUpside"], 1.0)
        self.assertEqual(local_batch["scores"]["cost"], 0.9545)
        self.assertIn("local-batch-winner", local_batch["evidenceRefs"][0])
        self.assertIn("long_context_qk_norm_improves_byte_compression", local_batch["nextPlanningAction"])

    def test_queue_artifact_is_durable_for_next_planning_cycle(self) -> None:
        self.assertTrue(QUEUE_PATH.exists())
        queue = json.loads(QUEUE_PATH.read_text())

        self.assertEqual(queue["taskId"], "FAST-873")
        self.assertEqual(queue["durability"]["artifactPath"], "planning/ranked-idea-queue.json")
        self.assertEqual(queue["durability"]["nextPlanningCycleInput"], True)
        self.assertIn("records/h100_candidate_batch/20260426T191455Z/results.json", queue["sourceInputs"])
        self.assertGreaterEqual(len(queue["rankedIdeas"]), 4)
        self.assertEqual(
            [row["rank"] for row in queue["rankedIdeas"]],
            list(range(1, len(queue["rankedIdeas"]) + 1)),
        )
        for row in queue["rankedIdeas"]:
            classification = row.get("submissionClassification", {})
            if classification.get("nonRecordReasons"):
                self.assertEqual(classification.get("status"), "non-record-only", row["ideaId"])

    def test_writer_persists_sorted_queue_with_stable_newline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "queue.json"
            queue = write_ranked_idea_queue(output_path=output)

            self.assertTrue(output.exists())
            self.assertTrue(output.read_text().endswith("\n"))
            persisted = json.loads(output.read_text())
            self.assertEqual(
                [row["ideaId"] for row in persisted["rankedIdeas"]],
                [row["ideaId"] for row in queue["rankedIdeas"]],
            )


if __name__ == "__main__":
    unittest.main()
