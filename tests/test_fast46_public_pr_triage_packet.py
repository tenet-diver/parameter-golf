import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKET_PATH = REPO_ROOT / "planning/fast46_public_pr_triage_execution_packet.json"
NOTE_PATH = REPO_ROOT / "fastest/source/fast-46-public-pr-triage-evidence.md"


class Fast46PublicPrTriagePacketTest(unittest.TestCase):
    def _load_packet(self) -> dict:
        return json.loads(PACKET_PATH.read_text())

    def _load_note(self) -> str:
        return NOTE_PATH.read_text()

    def _extract_note_commands(self, note: str) -> list[str]:
        lines = note.splitlines()
        in_section = False
        commands: list[str] = []
        for line in lines:
            if line.strip() == "## Repro commands (audited trail)":
                in_section = True
                continue
            if in_section and line.startswith("## "):
                break
            if in_section and line.strip().startswith("- `") and line.strip().endswith("`"):
                commands.append(line.strip()[3:-1])
        return commands

    def test_fast46_packet_has_traceable_evidence_for_each_ranked_candidate(self) -> None:
        packet = self._load_packet()

        self.assertEqual(packet.get("taskId"), "FAST-46")
        self.assertEqual(packet.get("status"), "completed")
        self.assertEqual(packet.get("lane"), "legal-evidence:public-pr-audit")

        ranked = packet.get("rankedCandidates")
        self.assertIsInstance(ranked, list)
        self.assertGreaterEqual(len(ranked), 6)

        evidence_records = packet.get("evidenceRecords")
        self.assertIsInstance(evidence_records, list)
        self.assertGreaterEqual(len(evidence_records), 12)

        evidence_ids = set()
        by_id = {}
        for record in evidence_records:
            self.assertIsInstance(record, dict)
            record_id = record.get("id")
            self.assertIsInstance(record_id, str)
            self.assertNotIn(record_id, evidence_ids)
            evidence_ids.add(record_id)
            by_id[record_id] = record

            self.assertIsInstance(record.get("prNumber"), int)
            self.assertIsInstance(record.get("capturedAtUtc"), str)
            self.assertIsInstance(record.get("sourceType"), str)
            self.assertIn(
                record.get("sourceType"),
                {
                    "github-open-ref",
                    "github-index-snapshot",
                    "github-open-ref-error",
                    "github-pr-file-snapshot",
                    "github-pr-diff-snippet",
                },
            )
            self.assertIsInstance(record.get("sourceUrl"), str)
            self.assertIn("github.com/openai/parameter-golf", record.get("sourceUrl"))
            self.assertIsInstance(record.get("retrievalCommand"), str)
            self.assertTrue(record.get("retrievalCommand").startswith("web."))
            self.assertIsInstance(record.get("excerpt"), str)
            self.assertTrue(record.get("excerpt"))

        required = {1698, 1722, 1795, 1797, 1801, 1807}
        ranked_by_pr = {
            entry["prNumber"]: entry
            for entry in ranked
            if isinstance(entry, dict) and isinstance(entry.get("prNumber"), int)
        }
        self.assertTrue(required.issubset(set(ranked_by_pr)))

        for pr in required:
            refs = ranked_by_pr[pr].get("evidenceRefs")
            self.assertIsInstance(refs, dict)
            for key in [
                "prSnapshotRef",
                "claimRef",
                "legalityRef",
                "runnableRef",
                "codeCompletenessRef",
            ]:
                self.assertIsInstance(refs.get(key), str)
                self.assertIn(refs[key], evidence_ids)
                self.assertEqual(by_id[refs[key]]["prNumber"], pr)

        self.assertEqual(ranked_by_pr[1722].get("legalityStatus"), "illegal")
        self.assertEqual(ranked_by_pr[1722].get("finalDisposition"), "illegal")

        direct_source_types = {"github-pr-file-snapshot", "github-pr-diff-snippet"}
        advanced = [
            entry
            for entry in ranked
            if isinstance(entry, dict) and entry.get("finalDisposition") == "advance"
        ]
        for entry in advanced:
            refs = entry["evidenceRefs"]
            for key in ["legalityRef", "runnableRef", "codeCompletenessRef"]:
                record = by_id[refs[key]]
                self.assertIn(
                    record["sourceType"],
                    direct_source_types,
                    (
                        f"advanced PR #{entry['prNumber']} must have direct PR-source evidence "
                        f"for {key}; got {record['sourceType']}"
                    ),
                )
                self.assertNotIn("index", record["sourceType"])
                self.assertNotIn("error", record["sourceType"])

        promotion = packet.get("promotionDecision")
        self.assertIsInstance(promotion, dict)
        if promotion.get("advanceOutcome") == "advance":
            decision_refs = promotion.get("decisionEvidenceRefs")
            self.assertIsInstance(decision_refs, list)
            self.assertGreater(len(decision_refs), 0)
            for evidence_id in decision_refs:
                self.assertIn(evidence_id, by_id)
                record = by_id[evidence_id]
                self.assertIn(
                    record["sourceType"],
                    direct_source_types,
                    "promotion advanceOutcome requires direct PR-source legality/runnable evidence",
                )
                self.assertNotIn("index", record["sourceType"])
                self.assertNotIn("error", record["sourceType"])

    def test_fast46_packet_proves_open_low_bpb_universe_coverage(self) -> None:
        packet = self._load_packet()
        coverage = packet.get("universeCoverage")
        self.assertIsInstance(coverage, dict)

        self.assertEqual(coverage.get("thresholdValBpb"), 1.0785)
        self.assertEqual(coverage.get("stateFilter"), "open")

        universe = coverage.get("openLowBpbUniverse")
        self.assertIsInstance(universe, list)
        self.assertGreaterEqual(len(universe), 6)

        universe_prs = set()
        for entry in universe:
            self.assertIsInstance(entry, dict)
            pr = entry.get("prNumber")
            self.assertIsInstance(pr, int)
            universe_prs.add(pr)
            self.assertIsInstance(entry.get("claimedValBpb"), float)
            self.assertLess(entry["claimedValBpb"], 1.0785)
            self.assertIsInstance(entry.get("sourceRef"), str)

        required = {1698, 1722, 1795, 1797, 1801, 1807}
        self.assertTrue(required.issubset(universe_prs))

        triaged = set(coverage.get("triagedPrNumbers", []))
        self.assertTrue(universe_prs.issubset(triaged))
        self.assertEqual(coverage.get("coverageStatus"), "complete-for-captured-universe")

    def test_fast46_repro_commands_are_reconciled_between_artifacts(self) -> None:
        packet = self._load_packet()
        note = self._load_note()

        packet_commands = packet.get("commands")
        self.assertIsInstance(packet_commands, list)
        self.assertIn("web.open https://github.com/openai/parameter-golf/pull/1722", packet_commands)

        note_commands = self._extract_note_commands(note)
        self.assertEqual(packet_commands, note_commands)

        self.assertIn("Evidence record IDs", note)


if __name__ == "__main__":
    unittest.main()
