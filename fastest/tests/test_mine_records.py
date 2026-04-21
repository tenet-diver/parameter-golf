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


if __name__ == "__main__":
    unittest.main()
