import json
import tempfile
import unittest
from pathlib import Path

from fastest.scripts.run_cpu_subset_experiment import (
    append_cpu_subset_evidence,
    build_cpu_subset_env,
    parse_final_val_bpb,
)


class CpuSubsetExperimentRunnerTest(unittest.TestCase):
    def test_parses_final_roundtrip_metric_from_train_output(self) -> None:
        val_loss, val_bpb = parse_final_val_bpb(
            "step:1/1 val_loss:4.1 val_bpb:4.2\n"
            "final_int8_zlib_roundtrip_exact val_loss:3.98765432 val_bpb:4.01234567\n"
        )

        self.assertEqual(val_loss, 3.98765432)
        self.assertEqual(val_bpb, 4.01234567)

    def test_builds_cpu_bounded_default_environment(self) -> None:
        env = build_cpu_subset_env(
            {
                "candidateId": "tiny",
                "seed": 7,
                "env": {
                    "ITERATIONS": 2,
                    "MODEL_DIM": "64",
                },
            },
            "cpu_subset_tiny",
        )

        self.assertEqual(env["RUN_ID"], "cpu_subset_tiny")
        self.assertEqual(env["SEED"], "7")
        self.assertEqual(env["ITERATIONS"], "2")
        self.assertEqual(env["MODEL_DIM"], "64")
        self.assertEqual(env["VAL_TOKEN_LIMIT"], "8192")
        self.assertEqual(env["MAX_WALLCLOCK_SECONDS"], "600")

    def test_appends_cpu_subset_evidence_without_benchmark_verified_class(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            evidence_path = root / "measurement_evidence.json"
            evidence_path.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "kind": "parameter-golf-measurement-evidence",
                        "updatedAt": "2026-04-25T00:00:00Z",
                        "summary": {
                            "artifactIds": [],
                            "trustedControlState": "established",
                            "totalExperiments": 0,
                            "acceptedExperiments": 0,
                            "mostRecentEvidenceId": None,
                        },
                        "recentCompletedExperiments": [],
                        "experimentRecords": [],
                    },
                )
            )
            run_dir = root / "fastest/generated/cpu_subset_runs/cpu_subset_test"
            run_dir.mkdir(parents=True)

            record = append_cpu_subset_evidence(
                candidate={"taskId": "FAST-CPU", "candidateId": "tiny", "seed": 7},
                run_id="cpu_subset_tiny",
                completed_at="2026-04-25T01:00:00Z",
                val_loss=4.1,
                val_bpb=4.2,
                command=["python", "train_gpt.py"],
                run_dir=run_dir,
                train_env=build_cpu_subset_env({"seed": 7}, "cpu_subset_tiny"),
                evidence_path=evidence_path,
            )

            self.assertEqual(record["objectiveMetricName"], "val_bpb")
            self.assertEqual(record["resultClass"], "cpu-subset")
            self.assertEqual(record["verificationClass"], "cpu-subset")
            self.assertNotEqual(record["verificationClass"], "benchmark-verified")
            self.assertEqual(record["promotionRationale"]["decision"], "hold")

            evidence = json.loads(evidence_path.read_text())
            self.assertEqual(evidence["summary"]["totalExperiments"], 1)
            self.assertEqual(evidence["summary"]["acceptedExperiments"], 1)
            self.assertEqual(evidence["recentCompletedExperiments"], ["exp-tiny"])

    def test_appends_model_factory_lineage_ranking_and_follow_up_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            evidence_path = root / "measurement_evidence.json"
            evidence_path.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "kind": "parameter-golf-measurement-evidence",
                        "updatedAt": "2026-04-25T00:00:00Z",
                        "summary": {
                            "artifactIds": ["evidence-exp-parent-cpu"],
                            "trustedControlState": "established",
                            "totalExperiments": 1,
                            "acceptedExperiments": 1,
                            "mostRecentEvidenceId": "evidence-exp-parent-cpu",
                        },
                        "recentCompletedExperiments": ["exp-parent-cpu"],
                        "experimentRecords": [
                            {
                                "experimentId": "exp-parent-cpu",
                                "evidenceId": "evidence-exp-parent-cpu",
                                "lane": "cpu-subset",
                                "status": "accepted",
                                "objectiveMetricName": "val_bpb",
                                "objectiveValue": 4.4,
                                "runConfig": {
                                    "seed": "1337",
                                    "iterations": 4,
                                    "trainSeqLen": 128,
                                    "trainBatchTokens": 2048,
                                    "valTokenLimit": 8192,
                                    "modelDim": 96,
                                    "numLayers": 1,
                                },
                            }
                        ],
                    },
                )
            )
            run_dir = root / "fastest/generated/cpu_subset_runs/cpu_subset_child"
            run_dir.mkdir(parents=True)

            record = append_cpu_subset_evidence(
                candidate={
                    "taskId": "FAST-49",
                    "candidateId": "child-cpu",
                    "parentExperimentId": "exp-parent-cpu",
                    "seed": 7,
                    "env": {"MODEL_DIM": "128", "VAL_TOKEN_LIMIT": "16384"},
                },
                run_id="cpu_subset_child",
                completed_at="2026-04-25T01:00:00Z",
                val_loss=4.0,
                val_bpb=4.1,
                command=["python", "train_gpt.py"],
                run_dir=run_dir,
                train_env=build_cpu_subset_env(
                    {"seed": 7, "env": {"MODEL_DIM": "128", "VAL_TOKEN_LIMIT": "16384"}},
                    "cpu_subset_child",
                ),
                evidence_path=evidence_path,
            )

            self.assertEqual(record["lineage"]["parentExperimentId"], "exp-parent-cpu")
            changed_factors = record["lineage"]["changedFactors"]
            self.assertIn(
                {"factor": "MODEL_DIM", "parentValue": "96", "candidateValue": "128"},
                changed_factors,
            )
            self.assertIn(
                {"factor": "VAL_TOKEN_LIMIT", "parentValue": "8192", "candidateValue": "16384"},
                changed_factors,
            )
            self.assertEqual(record["runtimeValidationCaps"]["maxRuntimeSeconds"], 600)
            self.assertEqual(record["runtimeValidationCaps"]["valTokenLimit"], 16384)
            self.assertEqual(record["runtimeValidationCaps"]["validationClass"], "cpu-subset")
            self.assertEqual(record["rankingTable"]["candidateRank"], 1)
            self.assertEqual(record["rankingTable"]["totalCandidates"], 2)
            self.assertEqual(record["modelFactoryDecision"]["promotionDecision"], "propose-follow-up")
            self.assertEqual(record["modelFactoryDecision"]["retirementDecision"], "retain")
            self.assertEqual(record["followUpTaskProposal"]["lane"], "cheap-screen")
            self.assertEqual(record["followUpTaskProposal"]["sourceLane"], "cpu-subset")
            self.assertTrue(record["followUpTaskProposal"]["taskId"].startswith("FAST-49-"))
            self.assertEqual(record["resultClass"], "cpu-subset")
            self.assertEqual(record["verificationClass"], "cpu-subset")


if __name__ == "__main__":
    unittest.main()
