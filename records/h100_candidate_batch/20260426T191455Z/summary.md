# H100 Candidate Batch Summary

- Started: `20260426T191455Z`
- Output directory: `/workspaces/parameter-golf/records/h100_candidate_batch/20260426T191455Z`
- Completed candidates: `12/12`
- Metric: `final_int8_zlib_roundtrip_exact val_bpb` (lower is better)

Best run: `text_diffusion_proxy` with `val_bpb=3.75282305`.

## Ranking

| Rank | Candidate | Family | val_bpb | Steps | step_avg_ms | int8 bytes | Artifact budget |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | `text_diffusion_proxy` | text_diffusion_proxy | 3.75282305 | 2 | 967.02 | 612599 | within-budget |
| 2 | `moe_proxy_wide_mlp` | moe_proxy | 4.04616470 | 2 | 1698.04 | 770207 | within-budget |
| 3 | `qk_rmsnorm_longctx` | long_context_ar | 4.06662657 | 2 | 3702.94 | 544508 | within-budget |
| 4 | `depth_recurrence_loop45` | universal_transformer_proxy | 4.06764226 | 2 | 596.83 | 544440 | within-budget |
| 5 | `ar_baseline_int8` | autoregressive | 4.07025636 | 2 | 735.05 | 544599 | within-budget |
| 6 | `ar_nonquant_reference` | autoregressive | 4.07025636 | 2 | 1176.56 | 544599 | within-budget |
| 7 | `narrow_deep` | autoregressive_shape | 4.07025636 | 2 | 1046.00 | 544599 | within-budget |
| 8 | `wide_shallow` | autoregressive_shape | 4.07025636 | 2 | 747.00 | 544599 | within-budget |
| 9 | `jepa_style_encoder_decoder_proxy` | jepa_proxy | 4.07036567 | 2 | 806.96 | 544603 | within-budget |
| 10 | `ssm_proxy_linear_attention_budget` | state_space_proxy | 4.07261945 | 2 | 1196.00 | 544462 | within-budget |
| 11 | `parallel_residual_qk5` | autoregressive | 4.07463801 | 2 | 1663.67 | 555086 | within-budget |
| 12 | `swiglu_clamped` | autoregressive_mlp | 4.09115050 | 2 | 1600.32 | 662360 | within-budget |

## Best By Candidate Type

| Type | Completed | Best candidate | Best val_bpb | Median val_bpb |
|---|---:|---|---:|---:|
| text_diffusion_proxy | 1/1 | `text_diffusion_proxy` | 3.75282305 | 3.75282305 |
| moe_proxy | 1/1 | `moe_proxy_wide_mlp` | 4.04616470 | 4.04616470 |
| long_context_ar | 1/1 | `qk_rmsnorm_longctx` | 4.06662657 | 4.06662657 |
| universal_transformer_proxy | 1/1 | `depth_recurrence_loop45` | 4.06764226 | 4.06764226 |
| autoregressive | 3/3 | `ar_baseline_int8` | 4.07025636 | 4.07025636 |
| autoregressive_shape | 2/2 | `narrow_deep` | 4.07025636 | 4.07025636 |
| jepa_proxy | 1/1 | `jepa_style_encoder_decoder_proxy` | 4.07036567 | 4.07036567 |
| state_space_proxy | 1/1 | `ssm_proxy_linear_attention_budget` | 4.07261945 | 4.07261945 |
| autoregressive_mlp | 1/1 | `swiglu_clamped` | 4.09115050 | 4.09115050 |

## Best By Quantization

| Quantization | Completed | Best candidate | Best val_bpb | Median val_bpb |
|---|---:|---|---:|---:|
| int8-zlib-artifact | 11/11 | `text_diffusion_proxy` | 3.75282305 | 4.07025636 |
| raw-fp32-reference-plus-int8-eval | 1/1 | `ar_nonquant_reference` | 4.07025636 | 4.07025636 |

## Charts

- `charts/final_val_bpb.svg`
- `charts/speed_vs_bpb.svg`
- `analysis/analysis.md`
- `hypothesis_graph.md`

## Coverage Notes

- Built-in candidates use the current `train_gpt.py` API. Families labeled `*_proxy` are not native implementations of JEPA, text diffusion, SSM, or MoE.
- To run native implementations, pass `--candidate-file candidates.json`; each candidate can provide `command`, `env`, `family`, and `description`.
