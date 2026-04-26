import json
import unittest
from pathlib import Path

from fastest.scripts import run_cpu_subset_experiment as cpu_subset_runner

REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = REPO_ROOT / "fastest/source/measurement_evidence.json"


class Fast52DeterministicRankingCpuSubsetEvidenceTest(unittest.TestCase):
    def _load_fast52_record(self) -> dict:
        evidence = json.loads(EVIDENCE_PATH.read_text())
        records = evidence.get("experimentRecords", [])
        self.assertIsInstance(records, list)
        fast52_records = [
            record
            for record in records
            if isinstance(record, dict)
            and record.get("taskId") == "FAST-52"
            and record.get("experimentId") == "exp-fast52-deterministic-ranking-cpu"
        ]
        self.assertTrue(fast52_records, "Expected FAST-52 measurement record to exist.")
        return fast52_records[-1]

    def test_fast52_records_cpu_subset_measurement_and_ranking_contract(self) -> None:
        record = self._load_fast52_record()
        self.assertEqual(record.get("resultClass"), "cpu-subset")
        self.assertEqual(record.get("verificationClass"), "cpu-subset")

        lineage = record.get("lineage")
        self.assertIsInstance(lineage, dict)
        self.assertEqual(lineage.get("parentFrontierId"), "deepseek-v4-deterministic-ranking")
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

    def test_fast52_trust_resolution_is_resolved_and_runner_verified(self) -> None:
        record = self._load_fast52_record()

        lineage_resolution = cpu_subset_runner._resolve_parent_lineage(
            parent_experiment_id=record.get("lineage", {}).get("parentExperimentId"),
            registry_path=cpu_subset_runner.DEFAULT_TRUSTED_PARENT_REGISTRY_PATH,
        )
        self.assertEqual(lineage_resolution.get("status"), "resolved")
        self.assertEqual(record.get("lineageResolution"), lineage_resolution)

        attestation_resolution = cpu_subset_runner._resolve_runner_attestation(
            attestation_ref=record.get("attestationRef"),
            run_id="cpu_subset_fast52-deterministic-ranking-cpu",
            registry_path=cpu_subset_runner.DEFAULT_TRUSTED_RUNNER_ATTESTATION_REGISTRY_PATH,
        )
        self.assertEqual(attestation_resolution.get("status"), "resolved")
        self.assertEqual(record.get("attestationResolution"), attestation_resolution)
        self.assertEqual(record.get("verificationStatus"), "passed")

    def test_fast52_has_runner_owned_deterministic_and_seed_variance_artifacts(self) -> None:
        run_dir = (
            REPO_ROOT / "fastest/generated/cpu_subset_runs/cpu_subset_fast52-deterministic-ranking-cpu"
        )
        deterministic_path = run_dir / "deterministic_rerun.json"
        seed_variance_path = run_dir / "seed_variance.json"
        self.assertTrue(deterministic_path.exists())
        self.assertTrue(seed_variance_path.exists())

        deterministic = json.loads(deterministic_path.read_text())
        seed_variance = json.loads(seed_variance_path.read_text())

        self.assertEqual(deterministic.get("source"), "cpu-subset-runner")
        self.assertEqual(seed_variance.get("source"), "cpu-subset-runner")
        self.assertEqual(deterministic.get("status"), "passed")
        self.assertEqual(seed_variance.get("status"), "passed")
        self.assertIsInstance(deterministic.get("evidenceRef"), str)
        self.assertIsInstance(seed_variance.get("evidenceRef"), str)

    def test_fast52_references_committed_artifact_receipts_with_hashes(self) -> None:
        record = self._load_fast52_record()
        manifest_path_raw = record.get("artifactReceiptManifestPath")
        self.assertIsInstance(
            manifest_path_raw,
            str,
            "FAST-52 record must reference a committed artifact receipt manifest.",
        )
        manifest_path = REPO_ROOT / manifest_path_raw
        self.assertTrue(manifest_path.exists(), "Artifact receipt manifest path must exist in repository.")

        manifest = json.loads(manifest_path.read_text())
        self.assertIsInstance(manifest, dict)
        self.assertEqual(manifest.get("attestationRef"), record.get("attestationRef"))
        receipts = manifest.get("artifactReceipts")
        self.assertIsInstance(receipts, list)
        self.assertEqual(len(receipts), len(record.get("artifactPaths", [])))
        for receipt in receipts:
            self.assertIsInstance(receipt, dict)
            self.assertIsInstance(receipt.get("artifactPath"), str)
            self.assertIsInstance(receipt.get("sha256"), str)
            self.assertEqual(len(receipt.get("sha256")), 64)


if __name__ == "__main__":
    unittest.main()
