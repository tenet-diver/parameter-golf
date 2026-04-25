import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKET_PATH = REPO_ROOT / "planning/fast47_pr1797_source_audit_execution_packet.json"
NOTE_PATH = REPO_ROOT / "fastest/source/fast-47-pr1797-source-audit-evidence.md"
EVIDENCE_PATH = REPO_ROOT / "fastest/source/measurement_evidence.json"


class Fast47Pr1797SourceAuditPacketTest(unittest.TestCase):
    def test_fast47_packet_captures_pr1797_source_level_runnable_legality_evidence(self) -> None:
        packet = json.loads(PACKET_PATH.read_text())

        self.assertEqual(packet.get("taskId"), "FAST-47")
        self.assertEqual(packet.get("candidatePr"), 1797)
        self.assertEqual(packet.get("status"), "completed")

        evidence_records = packet.get("evidenceRecords")
        self.assertIsInstance(evidence_records, list)
        self.assertGreaterEqual(len(evidence_records), 6)

        evidence_ids = {record.get("id") for record in evidence_records if isinstance(record, dict)}
        self.assertIn("E-1797-FILES-TREE", evidence_ids)
        self.assertIn("E-1797-SOURCE-DIFF", evidence_ids)
        self.assertIn("E-1797-SUBMISSION-METADATA", evidence_ids)
        self.assertIn("E-1797-TRAIN-LOGS", evidence_ids)
        self.assertIn("E-1797-DISCUSSION", evidence_ids)
        self.assertIn("E-1797-LEGALITY-GATE", evidence_ids)

        decision = packet.get("promotionDecision")
        self.assertIsInstance(decision, dict)
        self.assertIn(decision.get("advanceOutcome"), {"advance-to-external-reproduction", "blocked", "hold"})
        self.assertIsInstance(decision.get("rationale"), str)
        self.assertTrue(decision.get("rationale"))
        refs = decision.get("decisionEvidenceRefs")
        self.assertIsInstance(refs, list)
        self.assertTrue(refs)
        for evidence_id in refs:
            self.assertIn(evidence_id, evidence_ids)

    def test_fast47_sources_are_durable_and_measurement_record_is_traceable(self) -> None:
        note_text = NOTE_PATH.read_text()
        self.assertIn("PR #1797", note_text)
        self.assertIn("source-level", note_text)

        evidence = json.loads(EVIDENCE_PATH.read_text())
        records = evidence.get("experimentRecords", [])
        fast47_records = [
            record
            for record in records
            if isinstance(record, dict)
            and record.get("taskId") == "FAST-47"
            and record.get("experimentId") == "exp-fast47-pr1797-source-audit-001"
        ]
        self.assertEqual(len(fast47_records), 1)

        record = fast47_records[0]
        self.assertEqual(record.get("lane"), "legal-evidence:source-level-audit")
        self.assertEqual(record.get("runConfig", {}).get("candidatePr"), 1797)
        self.assertIn(
            record.get("campaignStatus"),
            {"advance-to-external-reproduction", "blocked", "hold"},
        )
        self.assertIn(
            "planning/fast47_pr1797_source_audit_execution_packet.json",
            record.get("artifactPaths", []),
        )


if __name__ == "__main__":
    unittest.main()
