import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from fastest.scripts import log_experiment


class LogExperimentSmokeTests(unittest.TestCase):
    def test_main_creates_log_entry_at_env_overridden_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "custom" / "experiment-log.jsonl"
            argv = [
                "log_experiment.py",
                "--title",
                "File backend create smoke",
                "--category",
                "smoke",
                "--status",
                "passed",
                "--summary",
                "create one entry",
                "--metric",
                "val_bpb=1.2345",
            ]

            stdout = io.StringIO()
            with patch("sys.argv", argv), patch.dict(
                "os.environ", {"FASTEST_EXPERIMENT_LOG_PATH": str(log_path)}, clear=False
            ), redirect_stdout(stdout):
                log_experiment.main()

            self.assertTrue(log_path.exists())
            lines = log_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(1, len(lines))
            entry = json.loads(lines[0])
            self.assertEqual("File backend create smoke", entry["title"])
            self.assertEqual({"val_bpb": "1.2345"}, entry["metrics"])
            self.assertIn(str(log_path), stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
