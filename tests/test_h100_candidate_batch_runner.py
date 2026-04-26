import tempfile
import unittest
from pathlib import Path

from fastest.scripts.run_h100_candidate_batch import (
    apply_smoke_overrides,
    build_analysis_payload,
    command_for_candidate,
    load_candidates,
    parse_metrics,
    prune_raw_checkpoint,
    rank_rows,
    summarize_groups,
    write_csv,
    write_hypothesis_graph,
)


class H100CandidateBatchRunnerTest(unittest.TestCase):
    def test_parses_final_metrics_and_artifact_sizes(self) -> None:
        metrics = parse_metrics(
            "\n".join(
                [
                    "model_params:123456",
                    "artifact_budget:state:within-budget total_bytes:1234 limit_bytes:16000000 headroom_bytes:15998766 quantization:int8",
                    "step:17/20000 train_loss:3.2100 train_time:4567ms step_avg:268.65ms",
                    "peak memory allocated: 9876 MiB reserved: 11111 MiB",
                    "Total submission size: 42000000 bytes",
                    "Total submission size int8+zlib: 15500000 bytes",
                    "final_int8_zlib_roundtrip_exact val_loss:1.23456789 val_bpb:1.11111111",
                ]
            )
        )

        self.assertEqual(metrics["modelParams"], 123456)
        self.assertEqual(metrics["completedSteps"], 17)
        self.assertEqual(metrics["finalValLoss"], 1.23456789)
        self.assertEqual(metrics["finalValBpb"], 1.11111111)
        self.assertEqual(metrics["rawSubmissionBytes"], 42000000)
        self.assertEqual(metrics["int8SubmissionBytes"], 15500000)
        self.assertEqual(metrics["artifactBudgetState"], "within-budget")
        self.assertEqual(metrics["peakAllocatedMiB"], 9876)

    def test_smoke_default_command_uses_plain_python(self) -> None:
        command = command_for_candidate({"id": "baseline"}, smoke=True)

        self.assertTrue(command[0].endswith("python") or "python" in Path(command[0]).name)
        self.assertTrue(command[-1].endswith("train_gpt.py"))

    def test_smoke_overrides_clear_incompatible_layer_orders(self) -> None:
        env = {
            "ENCODER_LAYER_ORDER": "0,1,2,3,4,5",
            "DECODER_LAYER_ORDER": "4,5,6,7,8",
            "NUM_LAYERS": "9",
        }

        apply_smoke_overrides(env)

        self.assertNotIn("ENCODER_LAYER_ORDER", env)
        self.assertNotIn("DECODER_LAYER_ORDER", env)
        self.assertEqual(env["NUM_LAYERS"], "2")
        self.assertEqual(env["MODEL_DIM"], "128")

    def test_ranks_scored_rows_before_unscored_rows(self) -> None:
        ranked = rank_rows(
            [
                {"id": "failed", "finalValBpb": None},
                {"id": "worse", "finalValBpb": 1.2},
                {"id": "better", "finalValBpb": 1.1},
            ]
        )

        self.assertEqual([row["id"] for row in ranked], ["better", "worse", "failed"])
        self.assertEqual([row["rank"] for row in ranked], [1, 2, None])

    def test_loads_candidate_file_and_writes_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_file = root / "candidates.json"
            candidate_file.write_text('[{"id": "x", "env": {"MODEL_DIM": 128}}]\n')

            candidates = load_candidates(candidate_file)
            self.assertEqual(candidates[0]["id"], "x")

            csv_path = root / "leaderboard.csv"
            write_csv(
                csv_path,
                [
                    {
                        "rank": 1,
                        "id": "x",
                        "status": "completed",
                        "family": "test",
                        "finalValBpb": 1.0,
                    }
                ],
            )
            self.assertIn("finalValBpb", csv_path.read_text())

    def test_prunes_raw_checkpoint_by_default_but_keeps_quantized_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "final_model.pt").write_bytes(b"raw")
            (run_dir / "final_model.int8.ptz").write_bytes(b"quantized")

            state = prune_raw_checkpoint(run_dir, keep_raw_checkpoint=False)

            self.assertFalse((run_dir / "final_model.pt").exists())
            self.assertTrue((run_dir / "final_model.int8.ptz").exists())
            self.assertFalse(state["rawCheckpointKept"])
            self.assertEqual(state["quantizedArtifactBytes"], len(b"quantized"))

    def test_summarizes_groups_and_hypothesis_graph(self) -> None:
        rows = rank_rows(
            [
                {
                    "id": "baseline",
                    "status": "completed",
                    "family": "autoregressive",
                    "quantization": "int8",
                    "hypothesis": "control",
                    "hypothesisTags": ["control"],
                    "finalValBpb": 1.2,
                },
                {
                    "id": "candidate",
                    "status": "completed",
                    "family": "autoregressive",
                    "quantization": "int8",
                    "hypothesis": "better_shape",
                    "hypothesisTags": ["shape"],
                    "finalValBpb": 1.1,
                },
            ]
        )

        groups = summarize_groups(rows, "family")
        self.assertEqual(groups[0]["groupKey"], "autoregressive")
        self.assertEqual(groups[0]["bestCandidate"], "candidate")
        self.assertEqual(groups[0]["completed"], 2)

        with tempfile.TemporaryDirectory() as tmp:
            graph_path = Path(tmp) / "hypothesis_graph.md"
            payload = build_analysis_payload(
                rows,
                hardware={"hostname": "test", "gpus": [{"name": "H100"}]},
                started_at="20260426T000000Z",
            )
            write_hypothesis_graph(graph_path, payload)
            graph = graph_path.read_text()
            self.assertIn("graph LR", graph)
            self.assertIn("better_shape", graph)


if __name__ == "__main__":
    unittest.main()
