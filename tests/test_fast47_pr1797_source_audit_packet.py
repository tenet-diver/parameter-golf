import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKET_PATH = REPO_ROOT / "planning/fast47_pr1797_source_audit_execution_packet.json"
NOTE_PATH = REPO_ROOT / "fastest/source/fast-47-pr1797-source-audit-evidence.md"
EVIDENCE_PATH = REPO_ROOT / "fastest/source/measurement_evidence.json"
IMPORTED_CANDIDATE_DIR = REPO_ROOT / "records/track_10min_16mb/2026-04-24_PR1787Base_Smear_LQERAsym_PhasedTTT_1.06157"
REQUIRED_IMPORTED_FILES = [
    "ATTRIBUTION.md",
    "README.md",
    "submission.json",
    "prepare_caseops_data.py.source_excerpt.md",
    "train_gpt.py.source_excerpt.md",
    "train_seed42.log.excerpt.txt",
    "train_seed314.log.excerpt.txt",
    "train_seed1234.log.excerpt.txt",
]


class Fast47Pr1797SourceAuditPacketTest(unittest.TestCase):
    def test_fast47_packet_captures_pr1797_source_level_runnable_legality_evidence(self) -> None:
        packet = json.loads(PACKET_PATH.read_text())

        self.assertEqual(packet.get("taskId"), "FAST-47")
        self.assertEqual(packet.get("candidatePr"), 1797)
        self.assertEqual(packet.get("status"), "completed")

        commands = packet.get("commands")
        self.assertIsInstance(commands, list)
        self.assertTrue(
            any("train_gpt.py" in command for command in commands),
            "FAST-47 packet must capture direct train_gpt.py source/diff retrieval.",
        )

        evidence_records = packet.get("evidenceRecords")
        self.assertIsInstance(evidence_records, list)
        self.assertGreaterEqual(len(evidence_records), 7)

        evidence_ids = {record.get("id") for record in evidence_records if isinstance(record, dict)}
        self.assertIn("E-1797-FILES-TREE", evidence_ids)
        self.assertIn("E-1797-SOURCE-DIFF", evidence_ids)
        self.assertIn("E-1797-SUBMISSION-METADATA", evidence_ids)
        self.assertIn("E-1797-TRAIN-LOGS", evidence_ids)
        self.assertIn("E-1797-DISCUSSION", evidence_ids)
        self.assertIn("E-1797-LEGALITY-GATE", evidence_ids)
        self.assertIn("E-1797-TRAIN-EVAL-IMPL-SOURCE", evidence_ids)

        by_id = {
            record["id"]: record
            for record in evidence_records
            if isinstance(record, dict) and isinstance(record.get("id"), str)
        }
        train_eval_impl = by_id["E-1797-TRAIN-EVAL-IMPL-SOURCE"]
        self.assertIn("train_gpt.py", train_eval_impl.get("retrievalCommand", ""))
        self.assertIn("train/eval", train_eval_impl.get("excerpt", ""))

        assessment = packet.get("runnableLegalityAssessment")
        self.assertIsInstance(assessment, dict)
        self.assertTrue(assessment.get("sourceLevelProofComplete"))
        proof_matrix = assessment.get("proofMatrix")
        self.assertIsInstance(proof_matrix, list)
        self.assertEqual(len(proof_matrix), 5)

        by_path = {
            entry.get("path"): entry for entry in proof_matrix if isinstance(entry, dict)
        }
        self.assertSetEqual(
            set(by_path.keys()),
            {"train", "eval", "submission", "logs", "discussion"},
        )
        for key in ("train", "eval"):
            refs = by_path[key].get("evidenceRefs")
            self.assertIsInstance(refs, list)
            self.assertIn("E-1797-TRAIN-EVAL-IMPL-SOURCE", refs)

        for entry in proof_matrix:
            self.assertIsInstance(entry, dict)
            self.assertEqual(entry.get("status"), "complete")
            refs = entry.get("evidenceRefs")
            self.assertIsInstance(refs, list)
            self.assertTrue(refs)
            for evidence_id in refs:
                self.assertIn(evidence_id, evidence_ids)
            self.assertNotIn("blockerReasonCode", entry)

        decision = packet.get("promotionDecision")
        self.assertIsInstance(decision, dict)
        self.assertEqual(decision.get("advanceOutcome"), "advance-to-external-reproduction")
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
        self.assertEqual(record.get("campaignStatus"), "advance-to-external-reproduction")
        self.assertEqual(
            record.get("verificationGate", {}).get("status"),
            "advance-to-external-reproduction",
        )

        artifact_paths = record.get("artifactPaths", [])
        self.assertIn(
            "planning/fast47_pr1797_source_audit_execution_packet.json",
            artifact_paths,
        )
        self.assertIn(
            "records/track_10min_16mb/2026-04-24_PR1787Base_Smear_LQERAsym_PhasedTTT_1.06157/train_gpt.py.source_excerpt.md",
            artifact_paths,
        )

        self.assertTrue(IMPORTED_CANDIDATE_DIR.is_dir())
        for filename in REQUIRED_IMPORTED_FILES:
            imported_file = IMPORTED_CANDIDATE_DIR / filename
            self.assertTrue(
                imported_file.is_file(),
                f"Expected imported PR1797 artifact is missing: {imported_file}",
            )


if __name__ == "__main__":
    unittest.main()
