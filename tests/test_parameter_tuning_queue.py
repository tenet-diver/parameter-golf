import json
import tempfile
import unittest
from pathlib import Path

from fastest.scripts.build_parameter_tuning_queue import (
    build_tuning_experiments,
    load_json,
    update_gpu_queue,
)
from fastest.scripts.run_gpu_experiment_queue import load_queue


class ParameterTuningQueueTest(unittest.TestCase):
    def test_default_tuning_spec_generates_legal_ready_gpu_packets(self) -> None:
        spec = load_json(Path("planning/parameter_tuning_space.json"))

        experiments = build_tuning_experiments(spec)

        self.assertGreaterEqual(len(experiments), 10)
        self.assertEqual(len({experiment["id"] for experiment in experiments}), len(experiments))
        self.assertTrue(all(experiment["status"] == "ready" for experiment in experiments))
        self.assertTrue(
            all(
                "pretrained weights" in experiment["candidate"]["tuningMetadata"]["legality"]
                for experiment in experiments
            )
        )
        self.assertTrue(
            any(
                experiment["candidate"]["env"].get("LOCAL_SGD_SYNC_STEPS") == "8"
                for experiment in experiments
            )
        )

    def test_update_gpu_queue_replaces_generated_ids_without_capping_inventory(self) -> None:
        queue = {
            "schemaVersion": 1,
            "kind": "parameter-golf-gpu-experiment-queue",
            "experiments": [
                {"id": "manual", "candidate": {"id": "manual"}},
                {"id": "gpuq_tune_dense-control-optimizer_base", "candidate": {"id": "old"}},
            ],
        }
        generated = [
            {
                "id": "gpuq_tune_dense-control-optimizer_base",
                "candidate": {"id": "gpuq_tune_dense-control-optimizer_base"},
            },
            {
                "id": "gpuq_tune_dense-control-optimizer_qk_gain_init-5.75",
                "candidate": {"id": "gpuq_tune_dense-control-optimizer_qk_gain_init-5.75"},
            },
        ]

        updated = update_gpu_queue(queue, generated)

        self.assertEqual(
            [experiment["id"] for experiment in updated["experiments"]],
            [
                "manual",
                "gpuq_tune_dense-control-optimizer_base",
                "gpuq_tune_dense-control-optimizer_qk_gain_init-5.75",
            ],
        )

    def test_generated_queue_can_be_consumed_by_gpu_queue_loader(self) -> None:
        spec = load_json(Path("planning/parameter_tuning_space.json"))
        experiments = build_tuning_experiments(spec)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "queue.json"
            path.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "kind": "parameter-golf-generated-parameter-tuning-gpu-queue",
                        "experiments": experiments,
                    }
                )
            )

            queue = load_queue(path)

        self.assertEqual(len(queue["experiments"]), len(experiments))


if __name__ == "__main__":
    unittest.main()
