import json
import tempfile
import unittest
from pathlib import Path

from fastest.scripts.run_gpu_experiment_queue import (
    inventory_by_lane_status,
    load_queue,
    materialize_candidate_file,
    select_experiments,
)


class GpuExperimentQueueTest(unittest.TestCase):
    def test_default_queue_is_valid_and_has_no_generation_cap(self) -> None:
        queue = load_queue(Path("planning/gpu_experiment_queue.json"))

        self.assertEqual(queue["kind"], "parameter-golf-gpu-experiment-queue")
        self.assertNotIn("inventoryCaps", queue)
        self.assertGreaterEqual(len(inventory_by_lane_status(queue)), 1)

    def test_selects_only_ready_and_queued_experiments_by_default(self) -> None:
        queue = {
            "experiments": [
                {"id": "ready", "lane": "h100", "status": "ready", "candidate": {"id": "ready"}},
                {"id": "queued", "lane": "h100", "status": "queued", "candidate": {"id": "queued"}},
                {"id": "blocked", "lane": "h100", "status": "blocked", "candidate": {"id": "blocked"}},
            ]
        }

        selected = select_experiments(
            queue,
            statuses={"ready", "queued"},
            lanes=set(),
            only=set(),
            max_experiments=None,
        )

        self.assertEqual([experiment["id"] for experiment in selected], ["ready", "queued"])

    def test_materializes_runner_candidate_file_with_queue_metadata(self) -> None:
        experiments = [
            {
                "id": "candidate-a",
                "lane": "h100-1x",
                "status": "ready",
                "priority": 1,
                "expectedSignal": "signal",
                "gpuPlan": {"firstPass": "1xH100"},
                "candidate": {
                    "id": "candidate-a",
                    "implementation": "autoregressive_gpt",
                    "env": {"CANDIDATE_IMPL": "autoregressive_gpt"},
                },
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            candidate_file = materialize_candidate_file(
                experiments,
                queue_path=Path("planning/gpu_experiment_queue.json"),
                export_root=Path(tmp),
            )

            payload = json.loads(candidate_file.read_text(encoding="utf-8"))

        self.assertEqual(payload[0]["id"], "candidate-a")
        self.assertEqual(payload[0]["queueMetadata"]["lane"], "h100-1x")
        self.assertEqual(payload[0]["queueMetadata"]["expectedSignal"], "signal")


if __name__ == "__main__":
    unittest.main()
