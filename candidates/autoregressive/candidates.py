from __future__ import annotations

from candidates.registry import CandidateSpec


IMPLEMENTATION = "autoregressive_gpt"


def candidate_specs() -> list[CandidateSpec]:
    return [
        CandidateSpec(
            id="ar_baseline_int8",
            family="autoregressive",
            hypothesis="baseline_control",
            hypothesis_tags=("control", "autoregressive", "int8_artifact"),
            quantization="int8-zlib-artifact",
            description="Repo baseline: 9L 512d GQA transformer with tied embeddings.",
            implementation=IMPLEMENTATION,
            env={"CANDIDATE_IMPL": IMPLEMENTATION},
        ),
        CandidateSpec(
            id="ar_nonquant_reference",
            family="autoregressive",
            hypothesis="quantization_artifact_cost",
            hypothesis_tags=("control", "raw_checkpoint", "artifact_size"),
            quantization="raw-fp32-reference-plus-int8-eval",
            description="Baseline training kept to compare raw checkpoint size against quantized artifact size.",
            implementation=IMPLEMENTATION,
            env={"CANDIDATE_IMPL": IMPLEMENTATION, "RUN_NOTE": "nonquant-reference"},
        ),
        CandidateSpec(
            id="parallel_residual_qk5",
            family="autoregressive",
            hypothesis="parallel_residual_improves_optimization",
            hypothesis_tags=("parallel_residual", "qk_gain", "leaderboard_motif"),
            quantization="int8-zlib-artifact",
            description="Parallel residual lane probe with higher QK gain.",
            implementation=IMPLEMENTATION,
            env={
                "CANDIDATE_IMPL": IMPLEMENTATION,
                "PARALLEL_RESIDUAL": "1",
                "QK_GAIN_INIT": "5.0",
                "MATRIX_LR": "0.035",
            },
        ),
        CandidateSpec(
            id="depth_recurrence_loop45",
            family="recurrent_autoregressive_transformer",
            hypothesis="depth_recurrence_increases_compute_per_parameter",
            hypothesis_tags=("depth_recurrence", "parameter_sharing", "layer_reuse"),
            quantization="int8-zlib-artifact",
            description="Reuses real transformer blocks to test depth-recurrence style parameter sharing pressure.",
            implementation=IMPLEMENTATION,
            env={
                "CANDIDATE_IMPL": IMPLEMENTATION,
                "ENCODER_LAYER_ORDER": "0,1,2,3,4,5",
                "DECODER_LAYER_ORDER": "4,5,6,7,8",
                "QK_GAIN_INIT": "5.0",
            },
        ),
        CandidateSpec(
            id="swiglu_clamped",
            family="autoregressive_mlp",
            hypothesis="swiglu_clamps_improve_capacity_stability",
            hypothesis_tags=("swiglu", "activation_clamp", "mlp"),
            quantization="int8-zlib-artifact",
            description="SwiGLU MLP with conservative activation clamps.",
            implementation=IMPLEMENTATION,
            env={
                "CANDIDATE_IMPL": IMPLEMENTATION,
                "ACTIVATION_MODE": "swiglu",
                "SWIGLU_CLAMP_ENABLED": "1",
                "SWIGLU_LINEAR_CLAMP_MIN": "-10",
                "SWIGLU_LINEAR_CLAMP_MAX": "10",
                "SWIGLU_GATE_CLAMP_MAX": "10",
                "MLP_MULT": "2",
            },
        ),
        CandidateSpec(
            id="qk_rmsnorm_longctx",
            family="long_context_autoregressive",
            hypothesis="long_context_qk_norm_improves_byte_compression",
            hypothesis_tags=("long_context", "qk_norm", "stability"),
            quantization="int8-zlib-artifact",
            description="Longer context plus QK RMSNorm to test stability under 2048-token training.",
            implementation=IMPLEMENTATION,
            env={
                "CANDIDATE_IMPL": IMPLEMENTATION,
                "TRAIN_SEQ_LEN": "2048",
                "TRAIN_BATCH_TOKENS": "524288",
                "ATTN_NORM_MODE": "qk_rmsnorm",
                "ATTN_NORM_EPS": "1e-5",
                "QK_GAIN_INIT": "1.0",
            },
        ),
        CandidateSpec(
            id="local_global_sparse_attention",
            family="sparse_attention_autoregressive",
            hypothesis="local_global_sparse_attention_reduces_dependencies_with_global_summary_tokens",
            hypothesis_tags=("sparse_attention", "local_window", "global_tokens", "throughput"),
            quantization="int8-zlib-artifact",
            description=(
                "Local-window causal attention with a small number of global source tokens, "
                "kept behind an explicit dense fallback."
            ),
            implementation=IMPLEMENTATION,
            env={
                "CANDIDATE_IMPL": IMPLEMENTATION,
                "SPARSE_ATTN_MODE": "local_global",
                "SPARSE_ATTN_WINDOW": "256",
                "SPARSE_ATTN_GLOBAL_TOKENS": "4",
                "PARALLEL_RESIDUAL": "1",
                "QK_GAIN_INIT": "5.25",
            },
        ),
        CandidateSpec(
            id="wide_shallow",
            family="autoregressive_shape",
            hypothesis="width_beats_depth_at_fixed_artifact_budget",
            hypothesis_tags=("shape", "wide", "shallow"),
            quantization="int8-zlib-artifact",
            description="Fewer wider layers to test width versus depth under the same artifact budget.",
            implementation=IMPLEMENTATION,
            env={
                "CANDIDATE_IMPL": IMPLEMENTATION,
                "NUM_LAYERS": "7",
                "MODEL_DIM": "608",
                "NUM_HEADS": "8",
                "NUM_KV_HEADS": "4",
                "MLP_MULT": "2",
            },
        ),
        CandidateSpec(
            id="narrow_deep",
            family="autoregressive_shape",
            hypothesis="depth_beats_width_at_fixed_artifact_budget",
            hypothesis_tags=("shape", "narrow", "deep"),
            quantization="int8-zlib-artifact",
            description="More narrower layers to test depth and computation density.",
            implementation=IMPLEMENTATION,
            env={
                "CANDIDATE_IMPL": IMPLEMENTATION,
                "NUM_LAYERS": "13",
                "MODEL_DIM": "448",
                "NUM_HEADS": "7",
                "NUM_KV_HEADS": "1",
                "MLP_MULT": "2",
            },
        ),
    ]
