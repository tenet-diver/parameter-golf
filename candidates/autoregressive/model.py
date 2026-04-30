from __future__ import annotations

from candidates.autoregressive.architecture import (
    GPT,
    CastedLinear,
    MLP,
    restore_low_dim_params_to_fp32,
)


def create_model(args):
    return GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        attn_norm_mode=args.attn_norm_mode,
        attn_norm_eps=args.attn_norm_eps,
        sparse_attn_mode=args.sparse_attn_mode,
        sparse_attn_window=args.sparse_attn_window,
        sparse_attn_global_tokens=args.sparse_attn_global_tokens,
        activation_mode=args.activation_mode,
        swiglu_clamp_enabled=args.swiglu_clamp_enabled,
        swiglu_linear_clamp_min=args.swiglu_linear_clamp_min,
        swiglu_linear_clamp_max=args.swiglu_linear_clamp_max,
        swiglu_gate_clamp_max=args.swiglu_gate_clamp_max,
        parallel_residual=args.parallel_residual,
        encoder_layer_order=args.encoder_layer_order,
        decoder_layer_order=args.decoder_layer_order,
        moe_num_experts=1,
        moe_top_k=1,
    )


__all__ = ["CastedLinear", "MLP", "create_model", "restore_low_dim_params_to_fp32"]
