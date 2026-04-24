import importlib
import os
import unittest
from pathlib import Path
from unittest import mock


AUTHORITATIVE_TASK_STORE_DIR = Path(
    "/home/codespace/.fastest/orchestrator/projects/parameter-golf-2e398b79d13c/runtime/tasks"
)


class Fast34CombinationAuthoritativeTaskStoreDefaultTest(unittest.TestCase):
    def test_default_task_store_dir_falls_back_to_authoritative_root_when_env_missing(self) -> None:
        import fastest.scripts.run_combination_candidate as combination_module

        original_task_store_dir = os.environ.get("TASK_STORE_DIR")
        try:
            with mock.patch.dict(os.environ, {}, clear=True):
                reloaded = importlib.reload(combination_module)
                self.assertEqual(reloaded.DEFAULT_TASK_STORE_DIR, AUTHORITATIVE_TASK_STORE_DIR)
        finally:
            if original_task_store_dir is None:
                os.environ.pop("TASK_STORE_DIR", None)
            else:
                os.environ["TASK_STORE_DIR"] = original_task_store_dir
            importlib.reload(combination_module)


if __name__ == "__main__":
    unittest.main()
