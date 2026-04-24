import importlib
import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock


AUTHORITATIVE_TASK_STORE_DIR = Path(
    "/home/codespace/.fastest/orchestrator/projects/parameter-golf-2e398b79d13c/runtime/tasks"
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

    def test_task_store_load_uses_explicit_read_only_sqlite_uri(self) -> None:
        import fastest.scripts.run_cheap_screen_candidate as cheap_screen_module

        with tempfile.TemporaryDirectory() as tmp_dir:
            task_store_dir = Path(tmp_dir)
            sqlite_path = task_store_dir / "task-store.sqlite"

            conn = sqlite3.connect(sqlite_path)
            try:
                conn.execute(
                    """
                    CREATE TABLE task_store_tasks (
                        id TEXT PRIMARY KEY,
                        status TEXT NOT NULL,
                        status_order INTEGER NOT NULL,
                        created_at TEXT NOT NULL,
                        pipeline_json TEXT
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO task_store_tasks (id, status, status_order, created_at, pipeline_json)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        "FAST-22-CANDIDATE",
                        "ready",
                        0,
                        "2026-04-24T00:00:00Z",
                        json.dumps(
                            {
                                "lane": "cheap-screen",
                                "candidateId": "exp-fast22-001",
                                "traceId": "trace-fast22-001",
                            }
                        ),
                    ),
                )
                conn.commit()
            finally:
                conn.close()

            observed: dict[str, object] = {}
            real_connect = sqlite3.connect

            def track_connect(database: object, *args: object, **kwargs: object) -> sqlite3.Connection:
                observed["database"] = database
                observed["uri"] = kwargs.get("uri")
                return real_connect(database, *args, **kwargs)

            with mock.patch.object(cheap_screen_module.sqlite3, "connect", side_effect=track_connect):
                tasks = cheap_screen_module._load_tasks_from_task_store(task_store_dir)

            self.assertEqual(len(tasks), 1)
            self.assertEqual(observed.get("uri"), True)
            self.assertIsInstance(observed.get("database"), str)
            self.assertIn("mode=ro", observed["database"])


if __name__ == "__main__":
    unittest.main()
