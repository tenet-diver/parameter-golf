import ast
import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RECORD_DIR = REPO_ROOT / "records/track_10min_16mb/2026-04-19_SP8192_QuantumFusionPlus_Hadamard_AWQ"
SCRIPT_PATH = RECORD_DIR / "train_gpt_sp8192_fusion.py"
SUBMISSION_PATH = RECORD_DIR / "submission.json"
EVIDENCE_PATH = REPO_ROOT / "fastest/source/measurement_evidence.json"


class Fast45BlockUnverifiedPr1732WinningClaimTest(unittest.TestCase):
    def _has_runnable_entrypoints(self, script_text: str) -> bool:
        tree = ast.parse(script_text)
        function_names = {node.name.lower() for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
        required_entrypoint_prefixes = ("train", "eval", "ttt", "serialize", "quant")
        has_named_entrypoints = any(
            name.startswith(required_entrypoint_prefixes) and name != "main" for name in function_names
        )
        has_minimal_runtime_ops = "torch.save(" in script_text and "torch.load(" in script_text
        return has_named_entrypoints and has_minimal_runtime_ops

    def test_pr1732_import_is_blocked_not_winning_when_script_lacks_runnable_path(self) -> None:
        script_text = SCRIPT_PATH.read_text()
        submission = json.loads(SUBMISSION_PATH.read_text())
        evidence = json.loads(EVIDENCE_PATH.read_text())

        has_runnable_entrypoints = self._has_runnable_entrypoints(script_text)
        self.assertFalse(
            has_runnable_entrypoints,
            "This guard test assumes the current imported script still lacks runnable train/eval/TTT/serialization entrypoints.",
        )

        records = evidence.get("experimentRecords", [])
        fast44_records = [
            record
            for record in records
            if isinstance(record, dict)
            and record.get("taskId") == "FAST-44"
            and record.get("experimentId") == "exp-fast44-pr1732-import-001"
        ]
        self.assertEqual(len(fast44_records), 1)
        record = fast44_records[0]

        self.assertEqual(record.get("campaignStatus"), "blocked-not-winning")
        self.assertIn("blocked", record.get("verificationGate", {}).get("status", ""))
        self.assertEqual(
            submission.get("local_campaign_evidence", {}).get("campaign_status"),
            "blocked-not-winning",
        )
        self.assertNotIn("winning candidate evidence", record.get("classificationReason", "").lower())
        self.assertNotEqual(record.get("reproductionStatus", {}).get("status"), "verified")


if __name__ == "__main__":
    unittest.main()
