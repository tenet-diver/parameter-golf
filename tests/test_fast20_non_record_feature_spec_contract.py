import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
FEATURE_SPEC_PATH = REPO_ROOT / "planning/non-record-exploration-feature-spec.json"


class Fast20NonRecordFeatureSpecContractTest(unittest.TestCase):
    def test_feature_spec_satisfies_fast20_design_first_contract(self) -> None:
        spec = json.loads(FEATURE_SPEC_PATH.read_text())

        self.assertEqual(spec["taskId"], "FAST-20")
        self.assertEqual(spec["taskType"], "spike")
        self.assertEqual(spec["workflow"]["name"], "feature-spec")
        self.assertEqual(spec["workflow"]["variant"], "design-first")

        behavior_contract = spec["behaviorContract"]
        self.assertEqual(behavior_contract["lane"], "non-record-exploration")
        self.assertEqual(behavior_contract["directionTag"], "non-record-new-direction")
        self.assertEqual(behavior_contract["maxExperimentsPerDirection"], 1)
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
        self.assertTrue(
            any("readiness" in entry.lower() for entry in architecture_handoff["risks"])
        )
        self.assertTrue(
            any("maxExperimentsPerDirection".lower() in entry.lower() for entry in architecture_handoff["implementationReadyDecisions"])
        )

        migration = spec["migrationAndOperations"]
        self.assertEqual(migration["dataMigrationRequired"], False)
        self.assertTrue(migration["operationalConstraints"])


if __name__ == "__main__":
    unittest.main()
