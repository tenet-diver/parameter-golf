import json
import tempfile
import unittest
from pathlib import Path

from fastest.scripts.ingest_leaderboard_knowledge import (
    build_leaderboard_knowledge,
    parse_readme_leaderboard,
    write_leaderboard_knowledge,
)


README_FIXTURE = """
| Run | Score | Author | Summary | Date | Info |
|-----|------:|--------|---------|------|------|
| SP8192 + Legal TTT | 1.0810 | bigbag | depth recurrence + parallel residuals | 2026-04-09 | [info](records/x) |

#### Unlimited Compute Leaderboard & Non-record Submissions

| Run | Score | Author | Summary | Date | Info |
|-----|------:|--------|---------|------|------|
| Binary | 1.1239 | Ciprian | 1-bit quantization | 2026-03-24 | [info](records/y) |
"""


class Fast2LeaderboardKnowledgeTest(unittest.TestCase):
    def test_readme_parser_preserves_track_and_score(self) -> None:
        rows = parse_readme_leaderboard(README_FIXTURE)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["track"], "10min_16mb")
        self.assertEqual(rows[1]["track"], "non_record_16mb")
        self.assertEqual(rows[0]["score"], 1.081)

    def test_builds_durable_knowledge_from_readme_and_submissions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            readme_path = root / "README.md"
            records_dir = root / "records"
            submission_dir = records_dir / "track_10min_16mb" / "winner"
            submission_dir.mkdir(parents=True)
            readme_path.write_text(README_FIXTURE)
            (submission_dir / "submission.json").write_text(
                json.dumps(
                    {
                        "name": "SP8192 + GPTQ",
                        "author": "tester",
                        "date": "2026-04-10",
                        "track": "10min_16mb",
                        "val_bpb": 1.079,
                        "bytes_total": 15_900_000,
                        "train_time_seconds": 580,
                        "technique_summary": "SP8192 GPTQ SDClip depth recurrence",
                    }
                )
            )

            knowledge = build_leaderboard_knowledge(
                readme_path=readme_path,
                records_dir=records_dir,
            )

            self.assertEqual(knowledge["kind"], "parameter-golf-leaderboard-knowledge")
            self.assertEqual(knowledge["summary"]["bestScore"], 1.079)
            self.assertEqual(knowledge["sources"]["submissionRecordCount"], 1)
            self.assertGreaterEqual(len(knowledge["motifs"]), 2)

    def test_writer_creates_machine_readable_source_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "leaderboard_knowledge.json"

            knowledge = write_leaderboard_knowledge(output_path=output_path)

            written = json.loads(output_path.read_text())
            self.assertEqual(written["summary"], knowledge["summary"])
            self.assertGreater(written["summary"]["totalRecords"], 0)


if __name__ == "__main__":
    unittest.main()
