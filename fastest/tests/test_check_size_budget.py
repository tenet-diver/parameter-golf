import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from fastest.scripts import check_size_budget


class CheckSizeBudgetAnalyzerTests(unittest.TestCase):
    def test_component_breakdown_and_budget_violation_for_estimated_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            code_path = Path(tmpdir) / "train_gpt.py"
            code_path.write_text("print('x')\n")

            argv = [
                "check_size_budget.py",
                str(code_path),
                "--target-total-bytes",
                "5000",
                "--vocab-size",
                "128",
                "--d-model",
                "64",
                "--n-layers",
                "2",
                "--n-heads",
                "4",
                "--quant-bits",
                "8",
            ]

            stdout = io.StringIO()
            with patch("sys.argv", argv), redirect_stdout(stdout), self.assertRaises(SystemExit) as exc:
                check_size_budget.main()

        self.assertEqual(2, exc.exception.code)
        output = stdout.getvalue()
        self.assertIn("model_bytes_estimate=", output)
        self.assertIn("component_embeddings_bytes=", output)
        self.assertIn("component_blocks_bytes=", output)
        self.assertIn("component_heads_bytes=", output)
        self.assertIn("quantization_bits=8", output)


if __name__ == "__main__":
    unittest.main()
