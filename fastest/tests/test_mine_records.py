import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastest.scripts import mine_records


class MineRecordsKnowledgeTests(unittest.TestCase):
    def test_load_records_and_summary_include_provenance_and_recent_movement(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            records_root = root / "records" / "track_10min_16mb"
            run_a = records_root / "2026-04-05_RunA"
            run_b = records_root / "2026-04-08_RunB"
            run_c = records_root / "2026-04-09_RunC"
            run_a.mkdir(parents=True)
            run_b.mkdir(parents=True)
            run_c.mkdir(parents=True)

            (run_a / "submission.json").write_text(
                json.dumps(
                    {
                        "track": "10min_16mb",
                        "name": "RunA",
                        "date": "2026-04-05",
                        "val_bpb": 1.1000,
                        "blurb": "SP8192 recurrence parallel residuals legal score-first TTT qk gain 5.0",
                    }
                )
            )
            (run_b / "submission.json").write_text(
                json.dumps(
                    {
                        "track": "10min_16mb",
                        "name": "RunB",
                        "date": "2026-04-08",
                        "val_bpb": 1.0900,
                        "blurb": "SP8192 recurrence parallel residuals legal score-first TTT qk gain 5.25",
                    }
                )
            )
            (run_c / "submission.json").write_text(
                json.dumps(
                    {
                        "track": "10min_16mb",
                        "name": "RunC",
                        "date": "2026-04-09",
                        "val_bpb": 1.0828,
                        "blurb": "SP8192 recurrence parallel residuals legal score-first TTT qk gain 5.25",
                    }
                )
            )

            with patch.object(mine_records, "ROOT", root), patch.object(
                mine_records, "RECORDS_ROOT", root / "records"
            ):
                loaded = mine_records.load_records()
                summary = mine_records.summarize(loaded)

        self.assertEqual(3, len(loaded))
        self.assertIn("source", loaded[0].__dict__)
        self.assertEqual("submission_json", loaded[0].source["kind"])
        self.assertIn("recent_movement", summary)
        self.assertGreaterEqual(len(summary["recent_movement"]), 2)
        self.assertEqual("2026-04-09", summary["best_public_record"]["date"])

    def test_legality_rule_checker_classifies_records_and_ideas(self) -> None:
        legal = mine_records.Record(
            path="records/track_10min_16mb/2026-04-09_Legal/submission.json",
            track="10min_16mb",
            name="Legal",
            date="2026-04-09",
            val_bpb=1.08,
            bytes_total=15_900_000,
            train_time_seconds=599,
            hardware="8xH100 80GB SXM",
            summary="legal score-first ttt",
            tags=["legal_ttt", "score_first_ttt"],
            source={"kind": "submission_json", "path": "x", "track_dir": "y"},
        )
        non_record = mine_records.Record(
            path="records/track_non_record_16mb/2026-04-09_NonRecord/submission.json",
            track="non-record",
            name="NonRecord",
            date="2026-04-09",
            val_bpb=1.05,
            bytes_total=15_800_000,
            train_time_seconds=None,
            hardware=None,
            summary="unlimited compute",
            tags=["ternary"],
            source={"kind": "submission_json", "path": "x", "track_dir": "y"},
        )
        uncertain = mine_records.Record(
            path="records/track_10min_16mb/2026-04-09_Uncertain/submission.json",
            track="10min_16mb",
            name="Uncertain",
            date="2026-04-09",
            val_bpb=1.12,
            bytes_total=None,
            train_time_seconds=None,
            hardware="8xH100 80GB SXM",
            summary="missing artifact evidence",
            tags=["qk_gain"],
            source={"kind": "submission_json", "path": "x", "track_dir": "y"},
        )

        legal_result = mine_records.classify_record_legality(legal)
        non_record_result = mine_records.classify_record_legality(non_record)
        uncertain_result = mine_records.classify_record_legality(uncertain)
        idea_status = mine_records.classify_ideas([legal, non_record, uncertain])

        self.assertEqual("leaderboard-legal", legal_result["status"])
        self.assertEqual("non-record-only", non_record_result["status"])
        self.assertEqual("uncertain", uncertain_result["status"])
        self.assertIn("missing_artifact_bytes", uncertain_result["uncertainty_reasons"])
        self.assertEqual("leaderboard-legal", idea_status["score_first_ttt"]["status"])
        self.assertEqual("non-record-only", idea_status["ternary"]["status"])
        self.assertEqual("uncertain", idea_status["qk_gain"]["status"])

    def test_legality_runtime_and_hardware_evidence_rules(self) -> None:
        missing_runtime = mine_records.Record(
            path="records/track_10min_16mb/2026-04-09_MissingRuntime/submission.json",
            track="10min_16mb",
            name="MissingRuntime",
            date="2026-04-09",
            val_bpb=1.10,
            bytes_total=15_500_000,
            train_time_seconds=None,
            hardware="8xH100 80GB SXM",
            summary="runtime not recorded",
            tags=[],
            source={"kind": "submission_json", "path": "x", "track_dir": "y"},
        )
        missing_hardware = mine_records.Record(
            path="records/track_10min_16mb/2026-04-09_MissingHardware/submission.json",
            track="10min_16mb",
            name="MissingHardware",
            date="2026-04-09",
            val_bpb=1.11,
            bytes_total=15_500_000,
            train_time_seconds=580,
            hardware=None,
            summary="hardware not recorded",
            tags=[],
            source={"kind": "submission_json", "path": "x", "track_dir": "y"},
        )
        over_runtime = mine_records.Record(
            path="records/track_10min_16mb/2026-04-09_OverRuntime/submission.json",
            track="10min_16mb",
            name="OverRuntime",
            date="2026-04-09",
            val_bpb=1.11,
            bytes_total=15_500_000,
            train_time_seconds=601,
            hardware="8xH100 80GB SXM",
            summary="runtime over limit",
            tags=[],
            source={"kind": "submission_json", "path": "x", "track_dir": "y"},
        )
        wrong_hardware = mine_records.Record(
            path="records/track_10min_16mb/2026-04-09_WrongHardware/submission.json",
            track="10min_16mb",
            name="WrongHardware",
            date="2026-04-09",
            val_bpb=1.11,
            bytes_total=15_500_000,
            train_time_seconds=560,
            hardware="4xA100 80GB",
            summary="different gpu fleet",
            tags=[],
            source={"kind": "submission_json", "path": "x", "track_dir": "y"},
        )

        missing_runtime_result = mine_records.classify_record_legality(missing_runtime)
        missing_hardware_result = mine_records.classify_record_legality(missing_hardware)
        over_runtime_result = mine_records.classify_record_legality(over_runtime)
        wrong_hardware_result = mine_records.classify_record_legality(wrong_hardware)

        self.assertEqual("uncertain", missing_runtime_result["status"])
        self.assertIn("missing_train_time_seconds", missing_runtime_result["uncertainty_reasons"])
        self.assertEqual("uncertain", missing_hardware_result["status"])
        self.assertIn("missing_hardware", missing_hardware_result["uncertainty_reasons"])
        self.assertEqual("uncertain", over_runtime_result["status"])
        self.assertIn("runtime_over_10min", over_runtime_result["uncertainty_reasons"])
        self.assertEqual("uncertain", wrong_hardware_result["status"])
        self.assertIn("hardware_not_8xh100", wrong_hardware_result["uncertainty_reasons"])

    def test_legality_detects_eval_drift_against_trusted_control(self) -> None:
        drifting_record = mine_records.Record(
            path="records/track_10min_16mb/2026-04-09_Drift/submission.json",
            track="10min_16mb",
            name="Drift",
            date="2026-04-09",
            val_bpb=1.09,
            bytes_total=15_500_000,
            train_time_seconds=590,
            hardware="8xH100 80GB SXM",
            summary="control drift check",
            tags=[],
            source={"kind": "submission_json", "path": "x", "track_dir": "y"},
            payload={
                "validation": {
                    "trusted_controls": [
                        {
                            "name": "control-a",
                            "expected_val_bpb": 1.1000,
                            "observed_val_bpb": 1.1060,
                            "max_drift_bpb": 0.0030,
                        }
                    ]
                }
            },
        )

        result = mine_records.classify_record_legality(drifting_record)

        self.assertEqual("uncertain", result["status"])
        self.assertIn("eval_drift_vs_trusted_control", result["uncertainty_reasons"])

    def test_legality_detects_artifact_or_compression_mismatch(self) -> None:
        mismatched_record = mine_records.Record(
            path="records/track_10min_16mb/2026-04-09_Mismatch/submission.json",
            track="10min_16mb",
            name="Mismatch",
            date="2026-04-09",
            val_bpb=1.09,
            bytes_total=15_995_000,
            train_time_seconds=590,
            hardware="8xH100 80GB SXM",
            summary="artifact mismatch check",
            tags=[],
            source={"kind": "submission_json", "path": "x", "track_dir": "y"},
            payload={
                "artifact_bytes_max": 15_995_000,
                "seed_results": {
                    "0": {"artifact_bytes": 15_990_000},
                    "1": {"artifact_bytes": 16_050_000},
                },
                "compression": "zstd-22",
                "validation": {
                    "artifact_probe": {
                        "compression": "brotli-11",
                    }
                },
                "compliance": {"artifact_under_16mb": True},
            },
        )

        result = mine_records.classify_record_legality(mismatched_record)

        self.assertEqual("uncertain", result["status"])
        self.assertIn("artifact_or_compression_mismatch", result["uncertainty_reasons"])

    def test_legality_fails_promising_result_without_reproducibility(self) -> None:
        unreproducible_record = mine_records.Record(
            path="records/track_10min_16mb/2026-04-09_Unreproducible/submission.json",
            track="10min_16mb",
            name="Unreproducible",
            date="2026-04-09",
            val_bpb=1.085,
            bytes_total=15_500_000,
            train_time_seconds=590,
            hardware="8xH100 80GB SXM",
            summary="promising result with repro failure",
            tags=[],
            source={"kind": "submission_json", "path": "x", "track_dir": "y"},
            payload={
                "validation": {
                    "reproducibility": {
                        "attempts": 3,
                        "successful_runs": 1,
                    }
                }
            },
        )

        result = mine_records.classify_record_legality(unreproducible_record)

        self.assertEqual("uncertain", result["status"])
        self.assertIn("promising_result_not_reproducible", result["uncertainty_reasons"])

    def test_idea_classification_prefers_uncertain_over_non_record_only(self) -> None:
        non_record = mine_records.Record(
            path="records/track_non_record_16mb/2026-04-09_NonRecord/submission.json",
            track="non-record",
            name="NonRecord",
            date="2026-04-09",
            val_bpb=1.05,
            bytes_total=15_800_000,
            train_time_seconds=None,
            hardware=None,
            summary="unlimited compute",
            tags=["shared_idea"],
            source={"kind": "submission_json", "path": "x", "track_dir": "y"},
        )
        uncertain = mine_records.Record(
            path="records/track_10min_16mb/2026-04-09_Uncertain/submission.json",
            track="10min_16mb",
            name="Uncertain",
            date="2026-04-09",
            val_bpb=1.12,
            bytes_total=None,
            train_time_seconds=None,
            hardware="8xH100 80GB SXM",
            summary="missing artifact evidence",
            tags=["shared_idea"],
            source={"kind": "submission_json", "path": "x", "track_dir": "y"},
        )

        idea_status = mine_records.classify_ideas([non_record, uncertain])

        self.assertEqual("uncertain", idea_status["shared_idea"]["status"])
        self.assertIn("missing_artifact_bytes", idea_status["shared_idea"]["uncertainty_reasons"])

    def test_main_persists_ranked_queue_with_local_learning_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            records_root = root / "records" / "track_10min_16mb"
            run_a = records_root / "2026-04-08_RunA"
            run_b = records_root / "2026-04-09_RunB"
            run_a.mkdir(parents=True)
            run_b.mkdir(parents=True)
            (run_a / "submission.json").write_text(
                json.dumps(
                    {
                        "track": "10min_16mb",
                        "name": "RunA",
                        "date": "2026-04-08",
                        "val_bpb": 1.10,
                        "bytes_total": 15_500_000,
                        "train_time_seconds": 590,
                        "hardware": "8xH100 80GB SXM",
                        "blurb": "SP8192 recurrence parallel residuals legal score-first TTT qk gain 5.0",
                    }
                )
            )
            (run_b / "submission.json").write_text(
                json.dumps(
                    {
                        "track": "10min_16mb",
                        "name": "RunB",
                        "date": "2026-04-09",
                        "val_bpb": 1.09,
                        "bytes_total": 15_400_000,
                        "train_time_seconds": 585,
                        "hardware": "8xH100 80GB SXM",
                        "blurb": "SP8192 recurrence parallel residuals legal score-first TTT qk gain 5.25",
                    }
                )
            )

            generated_root = root / "fastest" / "generated"
            log_path = root / "fastest" / "logs" / "experiment-log.jsonl"
            log_path.parent.mkdir(parents=True)
            log_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "title": "Cheap qk gain local sweep",
                                "status": "passed",
                                "ideas": ["qk_gain"],
                                "classification": "leaderboard-legal",
                            }
                        ),
                        json.dumps(
                            {
                                "title": "Parallel residual local retry",
                                "status": "failed",
                                "ideas": ["parallel_residuals"],
                                "classification": "unknown",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.object(mine_records, "ROOT", root), patch.object(
                mine_records, "RECORDS_ROOT", root / "records"
            ), patch.object(mine_records, "GENERATED_ROOT", generated_root), patch.object(
                mine_records, "LOCAL_EXPERIMENT_LOG_PATH", log_path
            ):
                mine_records.main()

            payload = json.loads((generated_root / "candidate_backlog.json").read_text(encoding="utf-8"))

        self.assertIn("generated_at", payload)
        self.assertIn("inputs", payload)
        self.assertEqual(2, payload["inputs"]["imported_record_count"])
        self.assertEqual(2, payload["inputs"]["local_learning"]["entry_count"])
        self.assertIn("tasks", payload)
        self.assertGreater(len(payload["tasks"]), 0)
        first = payload["tasks"][0]
        self.assertIn("ranking", first)
        self.assertIn("expected_info_gain", first["ranking"])
        self.assertIn("expected_upside", first["ranking"])
        self.assertIn("expected_cost", first["ranking"])
        self.assertIn("mergeability", first["ranking"])
        self.assertIn("composite_score", first["ranking"])


if __name__ == "__main__":
    unittest.main()
