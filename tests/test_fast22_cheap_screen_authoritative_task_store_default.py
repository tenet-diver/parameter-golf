import importlib
import os
import unittest
from pathlib import Path
from unittest import mock


AUTHORITATIVE_TASK_STORE_DIR = Path(
    "/home/codespace/.fastest/orchestrator/projects/parameter-golf-fastest-run-2e398b79d13c/runtime/tasks"
)


class Fast22CheapScreenAuthoritativeTaskStoreDefaultTest(unittest.TestCase):
    def test_default_task_store_dir_ignores_task_store_env_override(self) -> None:
        import fastest.scripts.run_cheap_screen_candidate as cheap_screen_module

        original_task_store_dir = os.environ.get("TASK_STORE_DIR")
        try:
            with mock.patch.dict(os.environ, {"TASK_STORE_DIR": "/tmp/non-authoritative-task-store"}):
                reloaded = importlib.reload(cheap_screen_module)
                self.assertEqual(reloaded.DEFAULT_TASK_STORE_DIR, AUTHORITATIVE_TASK_STORE_DIR)
        finally:
            if original_task_store_dir is None:
                os.environ.pop("TASK_STORE_DIR", None)
            else:
                os.environ["TASK_STORE_DIR"] = original_task_store_dir
            importlib.reload(cheap_screen_module)


if __name__ == "__main__":
    unittest.main()
