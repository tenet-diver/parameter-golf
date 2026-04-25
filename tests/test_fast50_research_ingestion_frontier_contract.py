import json
import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
FRONTIER_PATH = REPO_ROOT / "planning/research-ingestion-frontier.json"


class Fast50ResearchIngestionFrontierContractTest(unittest.TestCase):
    def test_frontier_retains_only_executable_scored_ideas_with_required_fields(self) -> None:
        frontier = json.loads(FRONTIER_PATH.read_text())

        self.assertEqual(frontier.get("taskId"), "FAST-50")
        self.assertEqual(frontier.get("kind"), "research-ingestion-frontier")

        sources = frontier.get("sources")
        self.assertIsInstance(sources, list)
        source_by_id = {source["sourceId"]: source for source in sources if isinstance(source, dict) and source.get("sourceId")}

        self.assertNotIn(
            "paper-slimpajama-distill-2024",
            source_by_id,
            "Placeholder/non-credible paper source must not appear in FAST-50 frontier provenance.",
        )
        for source in sources:
            if not isinstance(source, dict):
                continue
            source_type = source.get("sourceType")
            if source_type not in {"small-model-paper", "compression-training-paper"}:
                continue
            source_url = source.get("sourceUrl", "")
            is_arxiv_abs = bool(re.match(r"^https://arxiv\.org/abs/(?:\d{4}\.\d{5}|\d{7})(?:v\d+)?$", source_url))
            is_doi = source_url.startswith("https://doi.org/")
            self.assertTrue(
                is_arxiv_abs or is_doi,
                f"Paper source must use a credible arXiv abs or DOI URL: {source.get('sourceId')}",
            )

        retained = frontier.get("retainedIdeas")
        self.assertIsInstance(retained, list)
        self.assertGreater(len(retained), 0)

        for idea in retained:
            self.assertIsInstance(idea, dict)
            self.assertEqual(idea.get("executionStatus"), "retained")
            self.assertIsInstance(idea.get("ideaId"), str)
            self.assertIsInstance(idea.get("candidateId"), str)
            self.assertTrue(idea["candidateId"].startswith("fast50-frontier-"))
            self.assertIsInstance(idea.get("claim"), str)
            self.assertTrue(idea["claim"])
            self.assertIsInstance(idea.get("parameterGolfBottleneck"), str)
            self.assertTrue(idea["parameterGolfBottleneck"])
            self.assertIn(
                idea.get("legalityClass"),
                {"legal", "legal-review-needed", "blocked-illegal"},
            )
            self.assertIn(
                idea.get("computeClass"),
                {"cheap", "moderate", "expensive"},
            )

            provenance = idea.get("provenance")
            self.assertIsInstance(provenance, list)
            self.assertGreater(len(provenance), 0)
            for source_ref in provenance:
                self.assertIsInstance(source_ref, dict)
                self.assertIsInstance(source_ref.get("sourceId"), str)
                self.assertTrue(source_ref.get("sourceId"))
                source = source_by_id.get(source_ref["sourceId"])
                if source and source.get("sourceType") == "parameter-golf-leaderboard":
                    locator = source_ref.get("locator", "")
                    self.assertIsInstance(locator, str)
                    self.assertRegex(
                        locator,
                        r"^(entry|record|commit|path):",
                        "Leaderboard provenance locator must be concrete (entry/record/commit/path).",
                    )

            self.assertIsInstance(idea.get("firstCheapScreen"), dict)
            self.assertTrue(idea["firstCheapScreen"].get("screenId"))
            self.assertIsInstance(idea.get("stopCondition"), dict)
            self.assertTrue(idea["stopCondition"].get("condition"))

            follow_ups = idea.get("followUpTaskProposals")
            self.assertIsInstance(follow_ups, list)
            self.assertGreater(len(follow_ups), 0)
            for proposal in follow_ups:
                self.assertIsInstance(proposal, dict)
                self.assertIsInstance(proposal.get("lane"), str)
                self.assertTrue(proposal["lane"])

            score = idea.get("score")
            self.assertIsInstance(score, dict)
            self.assertIsInstance(score.get("composite"), (int, float))

        rejected = frontier.get("rejectedIdeas")
        self.assertIsInstance(rejected, list)
        self.assertTrue(
            any(item.get("reasonCode") == "non-executable-claim" for item in rejected if isinstance(item, dict))
        )

        conflicting = [idea for idea in retained if isinstance(idea, dict) and idea.get("claimConflict")]
        self.assertGreater(len(conflicting), 0)
        for idea in conflicting:
            self.assertEqual(idea.get("legalityClass"), "legal-review-needed")


if __name__ == "__main__":
    unittest.main()
