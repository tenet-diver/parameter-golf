import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
FRONTIER_PATH = REPO_ROOT / "planning/research-ingestion-frontier.json"
PACKET_PATH = REPO_ROOT / "planning/fast50_research_ingestion_execution_packet.json"


class Fast50ResearchIngestionExecutionPacketTest(unittest.TestCase):
    def test_execution_packet_maps_frontier_candidates_to_lane_attributed_work(self) -> None:
        frontier = json.loads(FRONTIER_PATH.read_text())
        packet = json.loads(PACKET_PATH.read_text())

        self.assertEqual(packet.get("taskId"), "FAST-50")
        self.assertEqual(packet.get("kind"), "research-ingestion-execution-packet")
        self.assertEqual(packet.get("status"), "completed")

        commands = packet.get("commands")
        self.assertIsInstance(commands, list)
        self.assertGreater(len(commands), 0)
        self.assertTrue(
            any("build_research_ingestion_frontier.py" in command for command in commands),
            "Execution packet must record the deterministic builder command.",
        )

        artifact_paths = packet.get("artifactPaths")
        self.assertIsInstance(artifact_paths, list)
        self.assertIn("planning/research-ingestion-frontier.json", artifact_paths)
        self.assertIn("planning/fast50_research_ingestion_execution_packet.json", artifact_paths)

        retained_ids = {
            idea.get("candidateId")
            for idea in frontier.get("retainedIdeas", [])
            if isinstance(idea, dict)
        }
        self.assertGreater(len(retained_ids), 0)

        candidate_packets = packet.get("candidatePackets")
        self.assertIsInstance(candidate_packets, list)
        self.assertEqual(len(candidate_packets), len(retained_ids))

        for entry in candidate_packets:
            self.assertIsInstance(entry, dict)
            candidate_id = entry.get("candidateId")
            self.assertIsInstance(candidate_id, str)
            self.assertIn(candidate_id, retained_ids)
            self.assertTrue(candidate_id.startswith("fast50-frontier-"))
            self.assertIsInstance(entry.get("lane"), str)
            self.assertTrue(entry.get("lane"))
            self.assertIsInstance(entry.get("firstCheapScreen"), dict)
            self.assertIsInstance(entry.get("stopCondition"), dict)
            self.assertIn(entry.get("promotionDecision"), {"advance", "retry", "reject", "blocked"})

        top_candidate_id = packet.get("topCandidateId")
        if top_candidate_id is not None:
            self.assertIn(top_candidate_id, retained_ids)
        else:
            self.assertIsInstance(packet.get("blockedReasonCode"), str)
            self.assertTrue(packet["blockedReasonCode"])


if __name__ == "__main__":
    unittest.main()
