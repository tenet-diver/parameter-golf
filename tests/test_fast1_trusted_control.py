import json
import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
STATUS_PATH = REPO_ROOT / "fastest/generated/campaign_status.json"
STATE_PATH = REPO_ROOT / "fastest/generated/campaign_state.json"
AUTHORITATIVE_EVIDENCE_PATH = REPO_ROOT / "fastest/source/measurement_evidence.json"
CONTROLLER_TICK_PATH = REPO_ROOT / "fastest/scripts/controller_tick.py"


class Fast1TrustedControlContractTest(unittest.TestCase):
    def test_authoritative_evidence_contains_reproducible_fast1_trusted_control_record(self) -> None:
        evidence = json.loads(AUTHORITATIVE_EVIDENCE_PATH.read_text())
        records = evidence["experimentRecords"]

        trusted_control_records = [
            record
            for record in records
            if record.get("taskId") == "FAST-1"
            and record.get("lane") == "measurement"
            and record.get("status") == "accepted"
        ]
        self.assertTrue(
            trusted_control_records,
            "Expected at least one accepted FAST-1 measurement record in authoritative evidence.",
        )

        record = trusted_control_records[-1]
        self.assertTrue(record.get("experimentId"))
        self.assertTrue(record.get("evidenceId"))
        self.assertTrue(record.get("idempotencyKey"))
        self.assertTrue(record.get("traceId"))

        run_config = record.get("runConfig")
        self.assertIsInstance(run_config, dict)
        self.assertTrue("seed" in run_config or "seedSet" in run_config)

        budget_caps = record.get("budgetCaps")
        self.assertIsInstance(budget_caps, dict)
        self.assertIn("maxRuntimeSeconds", budget_caps)

        if record["evidenceId"] == "evidence-fast1-control-20260424-cpu-smoke-001":
            self.assertEqual(record["submissionClass"], "non-record-only")
            self.assertGreater(record["objectiveValue"], 0)
            self.assertLessEqual(record["artifactBytes"], record["budgetCaps"]["artifactLimitBytes"])
            self.assertTrue(record["executionCommands"])
            self.assertTrue(record["configProvenance"]["specHash"])
            self.assertEqual(record["evidenceBundle"]["trustState"], "trusted")
            bundle_path = REPO_ROOT / record["evidenceBundle"]["localPath"]
            self.assertTrue(bundle_path.is_file())
            self.assertTrue(bundle_path.is_relative_to(REPO_ROOT))

            bundle = json.loads(bundle_path.read_text())
            for log_path_text in bundle["artifacts"]["logs"]:
                log_path = REPO_ROOT / log_path_text
                self.assertTrue(log_path.is_file())
                self.assertTrue(log_path.is_relative_to(REPO_ROOT))
                digest = hashlib.sha256(log_path.read_bytes()).hexdigest()
                self.assertEqual(bundle["checksums"][log_path_text], digest)

    def test_trusted_control_baseline_established(self) -> None:
        status = json.loads(STATUS_PATH.read_text())
        state = json.loads(STATE_PATH.read_text())

        progress = status["progress"]
        completed = status["recentCompletedExperiments"]
        evidence_cache = state["evidenceSummaryCache"]

        self.assertEqual(status["trustedControlState"], "established")
        self.assertGreaterEqual(progress["totalExperiments"], 1)
        self.assertGreaterEqual(progress["acceptedExperiments"], 1)
        self.assertIsNotNone(progress["mostRecentEvidenceId"])
        self.assertTrue(completed)

        self.assertEqual(evidence_cache["trustedControlState"], "established")
        self.assertGreaterEqual(evidence_cache["totalExperiments"], 1)
        self.assertGreaterEqual(evidence_cache["acceptedExperiments"], 1)
        self.assertTrue(evidence_cache["recentCompletedExperiments"])

    def test_generated_campaign_views_are_derived_from_authoritative_evidence(self) -> None:
        status = json.loads(STATUS_PATH.read_text())
        state = json.loads(STATE_PATH.read_text())
        evidence = json.loads(AUTHORITATIVE_EVIDENCE_PATH.read_text())

        summary = evidence["summary"]
        recent_completed = evidence["recentCompletedExperiments"]

        self.assertTrue(summary["artifactIds"])
        self.assertTrue(recent_completed)

        self.assertEqual(status["trustedControlState"], summary["trustedControlState"])
        self.assertEqual(
            status["progress"]["totalExperiments"],
            summary["totalExperiments"],
        )
        self.assertEqual(
            status["progress"]["acceptedExperiments"],
            summary["acceptedExperiments"],
        )
        self.assertEqual(
            status["progress"]["mostRecentEvidenceId"],
            summary["mostRecentEvidenceId"],
        )
        self.assertEqual(status["recentCompletedExperiments"], recent_completed)

        self.assertEqual(
            state["evidenceSummaryCache"]["trustedControlState"],
            summary["trustedControlState"],
        )
        self.assertEqual(
            state["evidenceSummaryCache"]["artifactIds"],
            summary["artifactIds"],
        )
        self.assertEqual(
            state["evidenceSummaryCache"]["totalExperiments"],
            summary["totalExperiments"],
        )
        self.assertEqual(
            state["evidenceSummaryCache"]["acceptedExperiments"],
            summary["acceptedExperiments"],
        )
        self.assertEqual(
            state["evidenceSummaryCache"]["recentCompletedExperiments"],
            recent_completed,
        )

    def test_controller_tick_regenerates_views_from_authoritative_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source_path = Path(tmp_dir) / "measurement_evidence.json"
            status_path = Path(tmp_dir) / "campaign_status.json"
            state_path = Path(tmp_dir) / "campaign_state.json"
            source_path.write_text(AUTHORITATIVE_EVIDENCE_PATH.read_text())
            status_path.write_text(STATUS_PATH.read_text())
            state_path.write_text(STATE_PATH.read_text())

            evidence = json.loads(source_path.read_text())
            evidence["updatedAt"] = "2026-04-23T22:00:00.000Z"
            evidence["summary"]["artifactIds"] = ["evidence-fast1-control-regen-001"]
            evidence["summary"]["totalExperiments"] = 2
            evidence["summary"]["acceptedExperiments"] = 2
            evidence["summary"]["mostRecentEvidenceId"] = "evidence-fast1-control-regen-001"
            evidence["recentCompletedExperiments"] = [
                "exp-fast1-control-20260423-001",
                "exp-fast1-control-regen-001",
            ]
            evidence["experimentRecords"].append(
                {
                    "experimentId": "exp-fast1-control-regen-001",
                    "evidenceId": "evidence-fast1-control-regen-001",
                    "lane": "measurement",
                    "status": "accepted",
                    "completedAt": "2026-04-23T22:00:00.000Z",
                    "objectiveMetricName": "benchmark-score",
                    "objectiveValue": 1.01,
                }
            )
            source_path.write_text(f"{json.dumps(evidence, indent=2)}\n")

            tick = subprocess.run(
                [
                    sys.executable,
                    str(CONTROLLER_TICK_PATH),
                    "--source-path",
                    str(source_path),
                    "--status-path",
                    str(status_path),
                    "--state-path",
                    str(state_path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(tick.returncode, 0, msg=tick.stderr)

            status = json.loads(status_path.read_text())
            state = json.loads(state_path.read_text())
            summary = evidence["summary"]

            self.assertEqual(status["progress"]["totalExperiments"], summary["totalExperiments"])
            self.assertEqual(
                status["progress"]["acceptedExperiments"],
                summary["acceptedExperiments"],
            )
            self.assertEqual(
                status["progress"]["mostRecentEvidenceId"],
                summary["mostRecentEvidenceId"],
            )
            self.assertEqual(
                status["recentCompletedExperiments"],
                evidence["recentCompletedExperiments"],
            )
            self.assertEqual(
                state["evidenceSummaryCache"]["artifactIds"],
                summary["artifactIds"],
            )
            self.assertEqual(
                state["evidenceSummaryCache"]["acceptedExperiments"],
                summary["acceptedExperiments"],
            )
            self.assertEqual(
                state["evidenceSummaryCache"]["totalExperiments"],
                summary["totalExperiments"],
            )
            self.assertEqual(
                state["evidenceSummaryCache"]["recentCompletedExperiments"],
                evidence["recentCompletedExperiments"],
            )

    def test_controller_tick_check_fails_when_generated_views_are_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source_path = Path(tmp_dir) / "measurement_evidence.json"
            status_path = Path(tmp_dir) / "campaign_status.json"
            state_path = Path(tmp_dir) / "campaign_state.json"
            source_path.write_text(AUTHORITATIVE_EVIDENCE_PATH.read_text())
            status_path.write_text(STATUS_PATH.read_text())
            state_path.write_text(STATE_PATH.read_text())

            status = json.loads(status_path.read_text())
            state = json.loads(state_path.read_text())

            status["trustedControlState"] = "verified"
            status["recentCompletedExperiments"] = ["exp-stale-view"]
            state["evidenceSummaryCache"]["trustedControlState"] = "verified"
            state["evidenceSummaryCache"]["recentCompletedExperiments"] = ["exp-stale-view"]

            status_path.write_text(f"{json.dumps(status, indent=2)}\n")
            state_path.write_text(f"{json.dumps(state, indent=2)}\n")

            tick = subprocess.run(
                [
                    sys.executable,
                    str(CONTROLLER_TICK_PATH),
                    "--check",
                    "--source-path",
                    str(source_path),
                    "--status-path",
                    str(status_path),
                    "--state-path",
                    str(state_path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(tick.returncode, 0)


if __name__ == "__main__":
    unittest.main()
