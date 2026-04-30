import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from fastest.scripts.package_competition_submission import package_submission


class PackageCompetitionSubmissionTest(unittest.TestCase):
    def test_packages_best_completed_batch_row_as_record_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            batch_dir = root / "batch"
            run_a = batch_dir / "runs" / "01_a"
            run_b = batch_dir / "runs" / "02_b"
            run_a.mkdir(parents=True)
            run_b.mkdir(parents=True)
            (run_a / "stdout.txt").write_text("a log\n", encoding="utf-8")
            (run_a / "stderr.txt").write_text("", encoding="utf-8")
            (run_a / "result.json").write_text('{"id":"candidate_a"}\n', encoding="utf-8")
            (run_b / "stdout.txt").write_text("b log\n", encoding="utf-8")
            (run_b / "stderr.txt").write_text("", encoding="utf-8")
            (run_b / "result.json").write_text('{"id":"candidate_b"}\n', encoding="utf-8")
            (batch_dir / "results.json").write_text(
                json.dumps(
                    [
                        {
                            "id": "candidate_a",
                            "status": "completed",
                            "description": "A candidate",
                            "runDir": str(run_a),
                            "finalValBpb": 1.2,
                            "finalValLoss": 2.4,
                            "int8SubmissionBytes": 12_000_000,
                            "elapsedSeconds": 580,
                            "env": {"CANDIDATE_IMPL": "autoregressive_gpt", "SEED": "42"},
                            "hardwareLabel": "8xH100",
                            "nprocPerNode": 8,
                        },
                        {
                            "id": "candidate_b",
                            "status": "completed",
                            "description": "B candidate",
                            "runDir": str(run_b),
                            "finalValBpb": 1.1,
                            "finalValLoss": 2.2,
                            "int8SubmissionBytes": 11_000_000,
                            "elapsedSeconds": 570,
                            "env": {"CANDIDATE_IMPL": "autoregressive_gpt", "SEED": "314"},
                            "hardwareLabel": "8xH100",
                            "nprocPerNode": 8,
                        },
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            output_dir = root / "record"
            packaged = package_submission(
                Namespace(
                    batch_dir=[batch_dir],
                    candidate_id=None,
                    records_root=root / "records",
                    output_dir=output_dir,
                    slug=None,
                    author="Tester",
                    github_id="tester",
                    name=None,
                    date="2026-04-30",
                    track="10min_16mb",
                    nproc_per_node=None,
                    force=False,
                )
            )

            self.assertEqual(packaged, output_dir)
            submission = json.loads((output_dir / "submission.json").read_text(encoding="utf-8"))
            self.assertEqual(submission["candidate_id"], "candidate_b")
            self.assertEqual(submission["val_bpb"], 1.1)
            self.assertTrue((output_dir / "README.md").exists())
            self.assertTrue((output_dir / "train_gpt.py").exists())
            self.assertTrue((output_dir / "candidates" / "registry.py").exists())
            self.assertTrue((output_dir / "fastest" / "scripts" / "artifact_budget_analyzer.py").exists())
            self.assertTrue((output_dir / "train_seed314.log").exists())
            self.assertIn("--nproc_per_node", (output_dir / "run_submission.sh").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
