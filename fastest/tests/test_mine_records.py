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


if __name__ == "__main__":
    unittest.main()
