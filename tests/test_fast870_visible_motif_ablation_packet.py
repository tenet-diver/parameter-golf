import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = REPO_ROOT / "records/fast870_visible_motif_ablation_screen_results.json"
SUMMARY_PATH = REPO_ROOT / "records/fast870_visible_motif_ablation_screen_summary.md"
ARCH_HANDOFF_PATH = REPO_ROOT / "docs/reports/architecture-handoff.md"
IMPLEMENTATION_PLAN_PATH = REPO_ROOT / "docs/reports/implementation-plan.md"


class Fast870VisibleMotifAblationPacketTest(unittest.TestCase):
    def test_fast870_results_compare_every_visible_motif_to_one_trusted_control(self) -> None:
        results = json.loads(RESULTS_PATH.read_text())

        self.assertEqual(results["schemaVersion"], 1)
        self.assertEqual(results["taskId"], "FAST-870")
        self.assertEqual(results["artifactLimitBytes"], 16_000_000)

        baseline = results["trustedControl"]
        self.assertEqual(
            baseline["controlConfigId"],
            "control-2026-04-09-sp8192-3layer-parres-qk525-legalttt",
        )
        self.assertEqual(baseline["validationBpb"], 1.0810)
        self.assertLessEqual(baseline["artifactBytes"], results["artifactLimitBytes"])

        rows = results["ablations"]
        self.assertEqual(
            {row["motifId"] for row in rows},
            {
                "sp8192-tokenizer",
                "3-layer-recurrence",
                "parallel-residuals",
                "legal-score-first-ttt",
                "qk-gain-tuning",
                "hessian-aware-clipping",
            },
        )

        required_fields = {
            "motifId",
            "controlConfigId",
            "ablationConfigId",
            "commandOrEntrypoint",
            "seedPolicy",
            "artifactPath",
            "artifactBytes",
            "validationBpb",
            "scoreDeltaBpb",
            "artifactDeltaBytes",
            "legalityClass",
            "runDurationEstimate",
            "nextAction",
        }
        for row in rows:
            self.assertTrue(required_fields.issubset(row), row["motifId"])
            self.assertEqual(row["controlConfigId"], baseline["controlConfigId"])
            self.assertAlmostEqual(
                row["scoreDeltaBpb"],
                round(row["validationBpb"] - baseline["validationBpb"], 6),
                places=6,
            )
            self.assertEqual(row["artifactDeltaBytes"], row["artifactBytes"] - baseline["artifactBytes"])
            self.assertIn(row["legalityClass"], {"legal", "illegal-size", "legality-review"})
            self.assertTrue(row["nextAction"])

        self.assertTrue(any(row.get("isolationClass") == "partial-confounded" for row in rows))

    def test_fast870_handoff_and_summary_name_boundaries_validation_and_next_actions(self) -> None:
        summary = SUMMARY_PATH.read_text()
        self.assertIn("FAST-870 Visible Motif Ablation Screen", summary)
        self.assertIn("score delta", summary.lower())
        self.assertIn("artifact delta", summary.lower())
        self.assertIn("legal-score-first-ttt", summary)

        architecture_handoff = ARCH_HANDOFF_PATH.read_text()
        self.assertIn("FAST-870", architecture_handoff)
        self.assertIn("/workspaces/parameter-golf", architecture_handoff)
        self.assertIn("No Fastest runtime, API, or fast-ui interface changes", architecture_handoff)

        implementation_plan = IMPLEMENTATION_PLAN_PATH.read_text()
        self.assertIn("FAST-870", implementation_plan)
        self.assertIn("Tests first", implementation_plan)
        self.assertIn("Validators", implementation_plan)


if __name__ == "__main__":
    unittest.main()
