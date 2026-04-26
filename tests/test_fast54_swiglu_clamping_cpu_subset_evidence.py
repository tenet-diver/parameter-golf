import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = REPO_ROOT / "fastest/source/measurement_evidence.json"


class Fast54SwiGluClampingCpuSubsetEvidenceTest(unittest.TestCase):
    def test_fast54_records_paired_cpu_subset_control_and_treatment_with_contract_metadata(self) -> None:
        evidence = json.loads(EVIDENCE_PATH.read_text())
        records = evidence.get("experimentRecords", [])
        self.assertIsInstance(records, list)

        fast54_records = [
            record
            for record in records
            if isinstance(record, dict)
            and record.get("taskId") == "FAST-54"
            and record.get("lane") == "cpu-subset"
            and record.get("experimentId")
            in {"exp-fast54-swiglu-control-cpu", "exp-fast54-swiglu-clamped-cpu"}
        ]
        self.assertEqual(len(fast54_records), 2)

        by_experiment = {record["experimentId"]: record for record in fast54_records}
        control = by_experiment["exp-fast54-swiglu-control-cpu"]
        treatment = by_experiment["exp-fast54-swiglu-clamped-cpu"]

        self.assertEqual(treatment.get("lineage", {}).get("parentExperimentId"), control.get("experimentId"))

        for record in fast54_records:
            self.assertEqual(record.get("resultClass"), "cpu-subset")
            self.assertEqual(record.get("verificationClass"), "cpu-subset")
            self.assertEqual(record.get("lineage", {}).get("parentFrontierId"), "deepseek-v4-swiglu-clamping")

            observed = record.get("observedMetricOutput")
            self.assertIsInstance(observed, dict)
            self.assertEqual(observed.get("name"), "val_bpb")
            self.assertIsInstance(observed.get("value"), float)
            self.assertIsInstance(observed.get("valLoss"), float)

            runtime_caps = record.get("runtimeValidationCaps")
            self.assertIsInstance(runtime_caps, dict)
            self.assertEqual(runtime_caps.get("validationClass"), "cpu-subset")
            self.assertEqual(runtime_caps.get("maxRuntimeSeconds"), 600)

            ranking_table = record.get("rankingTable")
            self.assertIsInstance(ranking_table, dict)
            self.assertEqual(ranking_table.get("metricName"), "val_bpb")
            self.assertEqual(ranking_table.get("direction"), "lower_is_better")
            self.assertIsInstance(ranking_table.get("candidateRank"), int)

            decision = record.get("modelFactoryDecision")
            self.assertIsInstance(decision, dict)
            self.assertIn(decision.get("promotionDecision"), {"hold", "propose-follow-up"})
            self.assertIn(decision.get("retirementDecision"), {"retain", "retire"})


if __name__ == "__main__":
    unittest.main()
