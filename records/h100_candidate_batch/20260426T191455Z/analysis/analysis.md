# Candidate Analysis Views

- Hardware: `unknown`
- Host: `codespaces-bde58a`
- Metric: `final_int8_zlib_roundtrip_exact.val_bpb`

## By Family

| Group | Completed | Best candidate | Best val_bpb | Median val_bpb | Mean val_bpb |
|---|---:|---|---:|---:|---:|
| text_diffusion_proxy | 1/1 | `text_diffusion_proxy` | 3.75282305 | 3.75282305 | 3.75282305 |
| moe_proxy | 1/1 | `moe_proxy_wide_mlp` | 4.04616470 | 4.04616470 | 4.04616470 |
| long_context_ar | 1/1 | `qk_rmsnorm_longctx` | 4.06662657 | 4.06662657 | 4.06662657 |
| universal_transformer_proxy | 1/1 | `depth_recurrence_loop45` | 4.06764226 | 4.06764226 | 4.06764226 |
| autoregressive | 3/3 | `ar_baseline_int8` | 4.07025636 | 4.07025636 | 4.07171691 |
| autoregressive_shape | 2/2 | `narrow_deep` | 4.07025636 | 4.07025636 | 4.07025636 |
| jepa_proxy | 1/1 | `jepa_style_encoder_decoder_proxy` | 4.07036567 | 4.07036567 | 4.07036567 |
| state_space_proxy | 1/1 | `ssm_proxy_linear_attention_budget` | 4.07261945 | 4.07261945 | 4.07261945 |
| autoregressive_mlp | 1/1 | `swiglu_clamped` | 4.09115050 | 4.09115050 | 4.09115050 |

## By Quantization

| Group | Completed | Best candidate | Best val_bpb | Median val_bpb | Mean val_bpb |
|---|---:|---|---:|---:|---:|
| int8-zlib-artifact | 11/11 | `text_diffusion_proxy` | 3.75282305 | 4.07025636 | 4.04116357 |
| raw-fp32-reference-plus-int8-eval | 1/1 | `ar_nonquant_reference` | 4.07025636 | 4.07025636 | 4.07025636 |

## By Hypothesis

| Group | Completed | Best candidate | Best val_bpb | Median val_bpb | Mean val_bpb |
|---|---:|---|---:|---:|---:|
| diffusion_like_soft_targets_need_native_objective | 1/1 | `text_diffusion_proxy` | 3.75282305 | 3.75282305 | 3.75282305 |
| conditional_capacity_merits_native_moe | 1/1 | `moe_proxy_wide_mlp` | 4.04616470 | 4.04616470 | 4.04616470 |
| long_context_qk_norm_improves_byte_compression | 1/1 | `qk_rmsnorm_longctx` | 4.06662657 | 4.06662657 | 4.06662657 |
| depth_recurrence_increases_compute_per_parameter | 1/1 | `depth_recurrence_loop45` | 4.06764226 | 4.06764226 | 4.06764226 |
| baseline_control | 1/1 | `ar_baseline_int8` | 4.07025636 | 4.07025636 | 4.07025636 |
| depth_beats_width_at_fixed_artifact_budget | 1/1 | `narrow_deep` | 4.07025636 | 4.07025636 | 4.07025636 |
| quantization_artifact_cost | 1/1 | `ar_nonquant_reference` | 4.07025636 | 4.07025636 | 4.07025636 |
| width_beats_depth_at_fixed_artifact_budget | 1/1 | `wide_shallow` | 4.07025636 | 4.07025636 | 4.07025636 |
| latent_prediction_scaffold_merits_native_jepa | 1/1 | `jepa_style_encoder_decoder_proxy` | 4.07036567 | 4.07036567 | 4.07036567 |
| cheap_sequence_mixing_can_trade_attention_for_depth | 1/1 | `ssm_proxy_linear_attention_budget` | 4.07261945 | 4.07261945 | 4.07261945 |
| parallel_residual_improves_optimization | 1/1 | `parallel_residual_qk5` | 4.07463801 | 4.07463801 | 4.07463801 |
| swiglu_clamps_improve_capacity_stability | 1/1 | `swiglu_clamped` | 4.09115050 | 4.09115050 | 4.09115050 |

## By Hypothesis Tag

| Group | Completed | Best candidate | Best val_bpb | Median val_bpb | Mean val_bpb |
|---|---:|---|---:|---:|---:|
| softcap | 1/1 | `text_diffusion_proxy` | 3.75282305 | 3.75282305 | 3.75282305 |
| text_diffusion_proxy | 1/1 | `text_diffusion_proxy` | 3.75282305 | 3.75282305 | 3.75282305 |
| untied_head | 1/1 | `text_diffusion_proxy` | 3.75282305 | 3.75282305 | 3.75282305 |
| capacity | 1/1 | `moe_proxy_wide_mlp` | 4.04616470 | 4.04616470 | 4.04616470 |
| moe_proxy | 1/1 | `moe_proxy_wide_mlp` | 4.04616470 | 4.04616470 | 4.04616470 |
| wide_mlp | 1/1 | `moe_proxy_wide_mlp` | 4.04616470 | 4.04616470 | 4.04616470 |
| long_context | 2/2 | `qk_rmsnorm_longctx` | 4.06662657 | 4.06962301 | 4.06962301 |
| qk_norm | 1/1 | `qk_rmsnorm_longctx` | 4.06662657 | 4.06662657 | 4.06662657 |
| stability | 1/1 | `qk_rmsnorm_longctx` | 4.06662657 | 4.06662657 | 4.06662657 |
| depth_recurrence | 1/1 | `depth_recurrence_loop45` | 4.06764226 | 4.06764226 | 4.06764226 |
| parameter_sharing | 1/1 | `depth_recurrence_loop45` | 4.06764226 | 4.06764226 | 4.06764226 |
| universal_transformer_proxy | 1/1 | `depth_recurrence_loop45` | 4.06764226 | 4.06764226 | 4.06764226 |
| artifact_size | 1/1 | `ar_nonquant_reference` | 4.07025636 | 4.07025636 | 4.07025636 |
| autoregressive | 1/1 | `ar_baseline_int8` | 4.07025636 | 4.07025636 | 4.07025636 |
| control | 2/2 | `ar_baseline_int8` | 4.07025636 | 4.07025636 | 4.07025636 |
| deep | 1/1 | `narrow_deep` | 4.07025636 | 4.07025636 | 4.07025636 |
| int8_artifact | 1/1 | `ar_baseline_int8` | 4.07025636 | 4.07025636 | 4.07025636 |
| narrow | 1/1 | `narrow_deep` | 4.07025636 | 4.07025636 | 4.07025636 |
| raw_checkpoint | 1/1 | `ar_nonquant_reference` | 4.07025636 | 4.07025636 | 4.07025636 |
| shallow | 1/1 | `wide_shallow` | 4.07025636 | 4.07025636 | 4.07025636 |
| shape | 2/2 | `narrow_deep` | 4.07025636 | 4.07025636 | 4.07025636 |
| wide | 1/1 | `wide_shallow` | 4.07025636 | 4.07025636 | 4.07025636 |
| encoder_decoder | 1/1 | `jepa_style_encoder_decoder_proxy` | 4.07036567 | 4.07036567 | 4.07036567 |
| jepa_proxy | 1/1 | `jepa_style_encoder_decoder_proxy` | 4.07036567 | 4.07036567 | 4.07036567 |
| skip_reuse | 1/1 | `jepa_style_encoder_decoder_proxy` | 4.07036567 | 4.07036567 | 4.07036567 |
| small_kv | 1/1 | `ssm_proxy_linear_attention_budget` | 4.07261945 | 4.07261945 | 4.07261945 |
| state_space_proxy | 1/1 | `ssm_proxy_linear_attention_budget` | 4.07261945 | 4.07261945 | 4.07261945 |
| leaderboard_motif | 1/1 | `parallel_residual_qk5` | 4.07463801 | 4.07463801 | 4.07463801 |
| parallel_residual | 1/1 | `parallel_residual_qk5` | 4.07463801 | 4.07463801 | 4.07463801 |
| qk_gain | 1/1 | `parallel_residual_qk5` | 4.07463801 | 4.07463801 | 4.07463801 |
| activation_clamp | 1/1 | `swiglu_clamped` | 4.09115050 | 4.09115050 | 4.09115050 |
| mlp | 1/1 | `swiglu_clamped` | 4.09115050 | 4.09115050 | 4.09115050 |
| swiglu | 1/1 | `swiglu_clamped` | 4.09115050 | 4.09115050 | 4.09115050 |
