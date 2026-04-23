import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = REPO_ROOT / "planning/guarded-replenishment-policy.json"
TRIGGER_ARTIFACT_PATH = (
    REPO_ROOT
    / "planning/backlog-triggers/planning-backlog-trigger-backlog-under-threshold-restore-10-2026-04-23t22-20-34-841z.json"
)


class Fast9GuardedReplenishmentPolicyContractTest(unittest.TestCase):
    def test_policy_artifact_satisfies_fast9_contract(self) -> None:
        policy = json.loads(POLICY_PATH.read_text())

        self.assertEqual(policy["taskId"], "FAST-9")
        self.assertEqual(
            policy["sourcePlanning"]["taskId"],
            "FAST-7",
        )
        self.assertEqual(
            policy["trigger"]["fingerprint"],
            "backlog-under-threshold-restore-10",
        )
        self.assertEqual(
            policy["trigger"]["artifactPath"],
            "planning/backlog-triggers/planning-backlog-trigger-backlog-under-threshold-restore-10-2026-04-23t22-20-34-841z.json",
        )

        self.assertTrue(policy["triggerPolicy"]["thresholds"])
        self.assertEqual(policy["triggerPolicy"]["cooldownMinutes"], 240)
        self.assertEqual(policy["triggerPolicy"]["maxTicketsPerCycle"], 3)

        objective = policy["queueCoverageObjective"]
        self.assertEqual(objective["targetPlannedTasks"], 10)
        self.assertIn("approved slices", objective["approach"])
        self.assertIn("measuredBy", objective)

        non_goals = policy["explicitNonGoals"]
        self.assertGreaterEqual(len(non_goals), 2)
        self.assertTrue(
            any("filler" in non_goal.lower() for non_goal in non_goals),
        )
        self.assertTrue(
            any("bypass" in non_goal.lower() for non_goal in non_goals),
        )

        approval = policy["approvalAndGovernance"]
        self.assertGreaterEqual(len(approval["checkpoints"]), 2)
        self.assertIn("roles", approval)
        self.assertTrue(approval["roles"].get("policyOwner"))
        self.assertTrue(approval["roles"].get("executionOwner"))
        self.assertTrue(approval["revisitTriggersForIrreversibleDecisions"])

        dedupe = policy["dedupeConflictHandling"]
        self.assertEqual(
            dedupe["onOpenTaskConflict"]["action"],
            "skip-issuance-and-escalate",
        )
        self.assertEqual(
            dedupe["onRecentRunCollision"]["action"],
            "pause-and-request-review",
        )
        self.assertTrue(dedupe["auditRequirements"])

    def test_referenced_trigger_artifact_exists_and_matches_fingerprint(self) -> None:
        trigger_artifact = json.loads(TRIGGER_ARTIFACT_PATH.read_text())

        self.assertEqual(
            trigger_artifact["triggerKind"],
            "backlog-depletion-product-owner-planning",
        )
        self.assertEqual(
            trigger_artifact["triggerFingerprint"],
            "backlog-under-threshold-restore-10",
        )
        self.assertEqual(trigger_artifact["dedupeEvidence"]["openFastTasksChecked"], 0)
        self.assertEqual(
            trigger_artifact["dedupeEvidence"]["recentPlanningRunsChecked"],
            0,
        )


if __name__ == "__main__":
    unittest.main()
