import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKET_PATH = REPO_ROOT / "planning/fast46_public_pr_triage_execution_packet.json"
NOTE_PATH = REPO_ROOT / "fastest/source/fast-46-public-pr-triage-evidence.md"


class Fast46PublicPrTriagePacketTest(unittest.TestCase):
    def test_fast46_packet_contains_required_pr_triage_and_decision(self) -> None:
        packet = json.loads(PACKET_PATH.read_text())

        self.assertEqual(packet.get("taskId"), "FAST-46")
        self.assertEqual(packet.get("status"), "completed")
        self.assertEqual(packet.get("lane"), "legal-evidence:public-pr-audit")

        inspected = packet.get("inspectedPrs")
        self.assertIsInstance(inspected, list)
        numbers = {entry.get("prNumber") for entry in inspected if isinstance(entry, dict)}
        self.assertTrue({1698, 1722, 1795, 1801, 1797, 1807}.issubset(numbers))

        ranked = packet.get("rankedCandidates")
        self.assertIsInstance(ranked, list)
        self.assertGreaterEqual(len(ranked), 6)
        self.assertEqual(ranked[0].get("rank"), 1)
        self.assertEqual(ranked[0].get("prNumber"), 1797)
        self.assertEqual(ranked[0].get("finalDisposition"), "advance")

        by_pr = {entry["prNumber"]: entry for entry in ranked if isinstance(entry, dict) and "prNumber" in entry}
        self.assertEqual(by_pr[1722].get("legalityStatus"), "illegal")
        self.assertIn("SLOT", by_pr[1722].get("blockingReasons", [""])[0])
        self.assertEqual(by_pr[1698].get("finalDisposition"), "blocked")
        self.assertIn("16,000,000", " ".join(by_pr[1698].get("blockingReasons", [])))
        self.assertEqual(by_pr[1795].get("finalDisposition"), "blocked")
        self.assertIn("organizer", " ".join(by_pr[1795].get("blockingReasons", [])).lower())

        decision = packet.get("promotionDecision", {})
        self.assertEqual(decision.get("candidatePr"), 1797)
        self.assertEqual(decision.get("advanceOutcome"), "advance")
        self.assertTrue(decision.get("rationale"))

    def test_fast46_campaign_evidence_note_contains_repro_commands_and_outcome(self) -> None:
        note = NOTE_PATH.read_text()

        self.assertIn("# FAST-46 Public PR Triage Evidence Note", note)
        self.assertIn("openai/parameter-golf/pull/1698", note)
        self.assertIn("openai/parameter-golf/pull/1722", note)
        self.assertIn("openai/parameter-golf/pull/1795", note)
        self.assertIn("openai/parameter-golf/pull/1797", note)
        self.assertIn("openai/parameter-golf/pull/1801", note)
        self.assertIn("openai/parameter-golf/pull/1807", note)
        self.assertIn("Decision: advance PR #1797", note)
        self.assertIn("Artifact paths", note)


if __name__ == "__main__":
    unittest.main()
