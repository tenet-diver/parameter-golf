import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
FEATURE_SPEC_PATH = REPO_ROOT / "planning/ablation-feature-spec.json"


class Fast28AblationFeatureSpecContractTest(unittest.TestCase):
    def test_feature_spec_satisfies_fast28_design_first_contract(self) -> None:
        spec = json.loads(FEATURE_SPEC_PATH.read_text())

        self.assertEqual(spec["taskId"], "FAST-28")
        self.assertEqual(spec["title"], "Run the next ablation experiment for parameter-golf")
        self.assertEqual(spec["taskType"], "spike")
        self.assertEqual(spec["workflow"]["name"], "feature-spec")
        self.assertEqual(spec["workflow"]["variant"], "design-first")
        self.assertEqual(
            spec["workflow"]["template"],
            "system-default system-feature-spec-design-first",
        )

        behavior_contract = spec["behaviorContract"]
        self.assertEqual(behavior_contract["lane"], "ablation")
        self.assertEqual(behavior_contract["candidateSeed"], "ablation-next-candidate")
        self.assertEqual(behavior_contract["maxAblationExperiments"], 1)
        self.assertFalse(behavior_contract["claimsBenchmarkReadiness"])

        architecture_handoff = spec["architectureHandoff"]
        self.assertTrue(architecture_handoff["boundaries"])
        self.assertTrue(architecture_handoff["interfaces"])
        self.assertTrue(architecture_handoff["risks"])
        self.assertTrue(architecture_handoff["implementationReadyDecisions"])

        self.assertTrue(
            any("task store" in entry.lower() for entry in architecture_handoff["boundaries"])
        )
        self.assertTrue(
            any("evidence" in entry.lower() for entry in architecture_handoff["interfaces"])
        )
        self.assertTrue(any("budget" in entry.lower() for entry in architecture_handoff["risks"]))
        self.assertTrue(
            any(
                "maxablationexperiments" in entry.lower()
                for entry in architecture_handoff["implementationReadyDecisions"]
            )
        )

        migration = spec["migrationAndOperations"]
        self.assertEqual(migration["dataMigrationRequired"], False)
        self.assertTrue(migration["operationalConstraints"])


if __name__ == "__main__":
    unittest.main()
