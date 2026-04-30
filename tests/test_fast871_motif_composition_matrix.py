import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = REPO_ROOT / "records/fast871_motif_composition_matrix.json"
SUMMARY_PATH = REPO_ROOT / "records/fast871_motif_composition_matrix_summary.md"
ARCH_HANDOFF_PATH = REPO_ROOT / "docs/reports/architecture-handoff.md"
IMPLEMENTATION_PLAN_PATH = REPO_ROOT / "docs/reports/implementation-plan.md"


class Fast871MotifCompositionMatrixTest(unittest.TestCase):
    def test_fast871_matrix_ranks_combinations_and_interactions(self) -> None:
        matrix = json.loads(MATRIX_PATH.read_text())

        self.assertEqual(matrix["schemaVersion"], 1)
        self.assertEqual(matrix["taskId"], "FAST-871")
        self.assertEqual(matrix["artifactLimitBytes"], 16_000_000)
        self.assertEqual(matrix["competitionPreset"], "parameter-golf")

        source = matrix["sourceAblationPacket"]
        self.assertEqual(source["taskId"], "FAST-870")
        self.assertTrue(Path(REPO_ROOT / source["path"]).exists())

        combinations = matrix["combinations"]
        self.assertGreaterEqual(len(combinations), 6)
        self.assertEqual(
            [row["rank"] for row in combinations],
            list(range(1, len(combinations) + 1)),
        )

        required_fields = {
            "rank",
            "combinationId",
            "motifIds",
            "arity",
            "evidenceRecordPaths",
            "validationBpb",
            "artifactBytes",
            "artifactHeadroomBytes",
            "interactionClass",
            "compositionVerdict",
            "rankRationale",
            "followUpTask",
        }
        interaction_classes = {row["interactionClass"] for row in combinations}
        self.assertTrue({"positive", "neutral", "negative"}.issubset(interaction_classes))

        for row in combinations:
            self.assertTrue(required_fields.issubset(row), row["combinationId"])
            self.assertEqual(row["arity"], len(row["motifIds"]))
            self.assertIn(row["arity"], {2, 3})
            self.assertLessEqual(row["artifactBytes"], matrix["artifactLimitBytes"])
            self.assertEqual(
                row["artifactHeadroomBytes"],
                matrix["artifactLimitBytes"] - row["artifactBytes"],
            )
            self.assertTrue(row["evidenceRecordPaths"])
            for record_path in row["evidenceRecordPaths"]:
                self.assertTrue(Path(REPO_ROOT / record_path).exists(), record_path)
            self.assertIn(
                row["compositionVerdict"],
                {"promote", "hold", "reject"},
            )
            if row["followUpTask"] is not None:
                self.assertIn(row["compositionVerdict"], {"promote", "hold"})
                self.assertIn("FAST-", row["followUpTask"]["taskTitle"])
                self.assertTrue(row["followUpTask"]["evidenceBasis"])
            else:
                self.assertEqual(row["compositionVerdict"], "reject")

        top = combinations[0]
        self.assertEqual(top["interactionClass"], "positive")
        self.assertIn("recurrence", top["combinationId"])
        self.assertIn("parallel-residuals", top["motifIds"])
        self.assertIn("legal-score-first-ttt", top["motifIds"])

    def test_fast871_summary_and_handoffs_capture_design_boundary(self) -> None:
        summary = SUMMARY_PATH.read_text()
        self.assertIn("FAST-871 Motif Composition Matrix", summary)
        self.assertIn("positive", summary.lower())
        self.assertIn("neutral", summary.lower())
        self.assertIn("negative", summary.lower())
        self.assertIn("follow-up", summary.lower())

        architecture_handoff = ARCH_HANDOFF_PATH.read_text()
        self.assertIn("FAST-871", architecture_handoff)
        self.assertIn("/workspaces/parameter-golf", architecture_handoff)
        self.assertIn("No Fastest runtime, API, or fast-ui interface changes", architecture_handoff)
        self.assertIn("composition matrix", architecture_handoff.lower())

        implementation_plan = IMPLEMENTATION_PLAN_PATH.read_text()
        self.assertIn("FAST-871", implementation_plan)
        self.assertIn("Tests first", implementation_plan)
        self.assertIn("Validators", implementation_plan)


if __name__ == "__main__":
    unittest.main()
