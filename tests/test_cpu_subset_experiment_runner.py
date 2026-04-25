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


if __name__ == "__main__":
    unittest.main()
