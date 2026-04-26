from __future__ import annotations

from candidates.registry import CandidateSpec


IMPLEMENTATION = "routed_moe_gpt"


def candidate_specs() -> list[CandidateSpec]:
    return [
        CandidateSpec(
            id="moe_top2_4expert",
            family="mixture_of_experts",
            hypothesis="top2_moe_increases_conditional_capacity_under_artifact_budget",
            hypothesis_tags=("moe", "top2_routing", "conditional_capacity"),
            quantization="int8-zlib-artifact",
            description="Real routed MoE with a learned token router, 4 experts, and top-2 expert weighting.",
            implementation=IMPLEMENTATION,
            env={
                "CANDIDATE_IMPL": IMPLEMENTATION,
                "NUM_LAYERS": "5",
                "MODEL_DIM": "384",
                "NUM_HEADS": "6",
                "NUM_KV_HEADS": "3",
                "MLP_MULT": "2",
                "MOE_NUM_EXPERTS": "4",
                "MOE_TOP_K": "2",
            },
        )
    ]
