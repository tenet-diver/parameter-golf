import argparse
import json
import tempfile
import unittest
from pathlib import Path

from fastest.scripts.run_gpu_experiment_queue import (
    build_batch_command,
    inventory_by_lane_status,
    load_queue,
    materialize_candidate_file,
    select_experiments,
    write_export_runbook,
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
                {"id": "ready", "lane": "h100", "status": "ready", "priority": 2, "candidate": {"id": "ready"}},
                {"id": "queued", "lane": "h100", "status": "queued", "priority": 1, "candidate": {"id": "queued"}},
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

        self.assertEqual([experiment["id"] for experiment in selected], ["queued", "ready"])

    def test_shards_selected_experiments_for_parallel_gpu_pods(self) -> None:
        queue = {
            "experiments": [
                {"id": "p1", "lane": "h100", "status": "ready", "priority": 1, "candidate": {"id": "p1"}},
                {"id": "p2", "lane": "h100", "status": "ready", "priority": 2, "candidate": {"id": "p2"}},
                {"id": "p3", "lane": "h100", "status": "ready", "priority": 3, "candidate": {"id": "p3"}},
                {"id": "p4", "lane": "h100", "status": "ready", "priority": 4, "candidate": {"id": "p4"}},
            ]
        }

        selected = select_experiments(
            queue,
            statuses={"ready"},
            lanes=set(),
            only=set(),
            max_experiments=None,
            shard_count=2,
            shard_index=1,
        )

        self.assertEqual([experiment["id"] for experiment in selected], ["p2", "p4"])

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

    def test_export_runbook_contains_independently_runnable_batch_command(self) -> None:
        experiments = [
            {
                "id": "candidate-a",
                "lane": "h100-1x",
                "status": "ready",
                "priority": 1,
                "candidate": {"id": "candidate-a"},
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            candidate_file = materialize_candidate_file(
                experiments,
                queue_path=Path("planning/gpu_experiment_queue.json"),
                export_root=Path(tmp),
            )
            args = argparse.Namespace(
                queue=Path("planning/gpu_experiment_queue.json"),
                output_root=Path("records/h100_candidate_batch"),
                timeout_seconds=900,
                stop_on_failure=False,
                smoke=False,
                keep_raw_checkpoints=False,
                auto_tune_batch=True,
                tune_only=False,
                batch_tune_target_memory_fraction=0.9,
                batch_tune_max_tokens=2_097_152,
                batch_tune_timeout_seconds=180,
                shard_index=0,
                shard_count=2,
            )

            write_export_runbook(candidate_file=candidate_file, experiments=experiments, args=args)
            manifest = json.loads((candidate_file.parent / "run_manifest.json").read_text(encoding="utf-8"))
            command = build_batch_command(args, candidate_file)

        self.assertEqual(manifest["selectedExperimentIds"], ["candidate-a"])
        self.assertEqual(manifest["shard"], {"index": 0, "count": 2})
        self.assertEqual(manifest["batchCommand"], command)
        self.assertEqual(command[0:2], ["python", "fastest/scripts/run_h100_candidate_batch.py"])


if __name__ == "__main__":
    unittest.main()
