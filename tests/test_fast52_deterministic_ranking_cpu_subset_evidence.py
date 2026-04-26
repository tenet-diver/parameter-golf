import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = REPO_ROOT / "fastest/source/measurement_evidence.json"


class Fast52DeterministicRankingCpuSubsetEvidenceTest(unittest.TestCase):
    def test_fast52_records_cpu_subset_measurement_and_ranking_contract(self) -> None:
        evidence = json.loads(EVIDENCE_PATH.read_text())
        records = evidence.get("experimentRecords", [])
        self.assertIsInstance(records, list)

        fast52_records = [
            record
            for record in records
            if isinstance(record, dict)
            and record.get("taskId") == "FAST-52"
            and record.get("lane") == "cpu-subset"
            and record.get("lineage", {}).get("parentFrontierId")
            == "deepseek-v4-deterministic-ranking"
        ]
        self.assertTrue(
            fast52_records,
            "Expected at least one FAST-52 cpu-subset measurement record with deepseek-v4-deterministic-ranking lineage.",
        )

        record = fast52_records[-1]
        self.assertEqual(record.get("resultClass"), "cpu-subset")
        self.assertEqual(record.get("verificationClass"), "cpu-subset")

        lineage = record.get("lineage")
        self.assertIsInstance(lineage, dict)
        changed_factors = lineage.get("changedFactors")
        self.assertIsInstance(changed_factors, list)
        self.assertTrue(changed_factors)

        observed = record.get("observedMetricOutput")
        self.assertIsInstance(observed, dict)
        self.assertEqual(observed.get("name"), "val_bpb")
        self.assertIsInstance(observed.get("value"), float)

        runtime_caps = record.get("runtimeValidationCaps")
        self.assertIsInstance(runtime_caps, dict)
        self.assertEqual(runtime_caps.get("validationClass"), "cpu-subset")
        self.assertIsInstance(runtime_caps.get("maxRuntimeSeconds"), int)
        self.assertLessEqual(runtime_caps.get("maxRuntimeSeconds"), 600)

        ranking_table = record.get("rankingTable")
        self.assertIsInstance(ranking_table, dict)
        self.assertEqual(ranking_table.get("metricName"), "val_bpb")
        self.assertIsInstance(ranking_table.get("candidateRank"), int)

        decision = record.get("modelFactoryDecision")
        self.assertIsInstance(decision, dict)
        self.assertIn(decision.get("promotionDecision"), {"hold", "propose-follow-up"})
        self.assertIn(decision.get("retirementDecision"), {"retain", "retire", "defer"})


if __name__ == "__main__":
    unittest.main()
