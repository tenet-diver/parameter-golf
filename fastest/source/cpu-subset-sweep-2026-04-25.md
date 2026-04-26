# CPU subset sweep - 2026-04-25

Purpose: replace proxy progress with real local PyTorch CPU evidence while no GPUs are available.

Environment:
- PyTorch CPU runtime; CUDA unavailable.
- Runner: `fastest/scripts/run_cpu_subset_experiment.py`.
- Dataset/tokenizer: local FineWeb 10B SP1024 subset paths configured by the runner.
- Shared tiny screen defaults: 1 layer, model dim 96, 4 heads, 2 KV heads, sequence length 128, train batch tokens 2048, validation token limit 8192.
- Evidence class: `cpu-subset`; these records are not `benchmark-verified` and must not be promoted as leaderboard progress.

Results, lower `val_bpb` is better:

| Candidate | Iterations | LR scale | val_bpb |
| --- | ---: | ---: | ---: |
| baseline-tiny-cpu | 4 | 1.0 | 4.26207938 |
| lr-half-tiny-cpu | 4 | 0.5 | 3.88462382 |
| lr-quarter-tiny-cpu | 4 | 0.25 | 3.84994109 |
| lr-half-8iter-tiny-cpu | 8 | 0.5 | 3.66020801 |
| lr-quarter-8iter-tiny-cpu | 8 | 0.25 | 3.62737283 |
| lr-eighth-8iter-tiny-cpu | 8 | 0.125 | 3.71135630 |
| lr-half-16iter-tiny-cpu | 16 | 0.5 | 3.41242975 |
| lr-quarter-16iter-tiny-cpu | 16 | 0.25 | 3.43200090 |
| lr-half-32iter-tiny-cpu | 32 | 0.5 | 3.18292065 |
| lr-half-64iter-tiny-cpu | 64 | 0.5 | 2.95971435 |
| lr-half-128iter-tiny-cpu | 128 | 0.5 | 2.80297923 |
| lr-half-256iter-tiny-cpu | 256 | 0.5 | 2.67928586 |
| lr-half-256iter-val32k-tiny-cpu | 256 | 0.5 | 2.70817084 |
| lr-half-256iter-d128-cpu | 256 | 0.5 | 2.66077075 |
| lr-half-256iter-d192-cpu | 256 | 0.5 | 2.61832018 |

Current cheap-screen winner: `lr-half-256iter-d192-cpu`.
Validation-stability check: `lr-half-256iter-val32k-tiny-cpu` held the same direction with a 32k-token validation cap at `2.70817084`.

Interpretation:
- The default tiny-run rates were too aggressive for this reduced CPU screen.
- Half-rate training improves consistently as iteration count increases through 256 tiny iterations.
- The 256-iteration half-rate candidate remains strong under a larger 32k-token validation cap.
- Increasing model width from dim 96 to 128 and 192 improved the same half-rate 256-iteration screen, with dim 192 best so far at `2.61832018`.
- The next real CPU-only step is to test whether the half-rate direction survives a larger subset or a modest architecture increase while staying under the 10-minute CPU budget.

## FAST-54 SwiGLU clamping paired CPU-subset screen (2026-04-26)

Frontier record: `deepseek-v4-swiglu-clamping` (DeepSeek V4 report direction).
Task: `FAST-54`, lane `architecture-stability`, runtime cap `600s`, validation cap `32768` tokens.

Paired setup:
- Control: `swiglu`, clamping disabled.
- Treatment: `swiglu`, `deepseek_v4` clamping enabled (`linear [-10, 10]`, `gate <= 10`).
- Shared config: seed `1337`, iterations `256`, val batch size `8192`, result/verification class `cpu-subset`.

Measured output:

| Candidate | val_loss | val_bpb | max train_loss | train_loss spikes (>2x final train_loss) |
| --- | ---: | ---: | ---: | ---: |
| fast54-swiglu-control-cpu | 4.72845912 | 2.73757892 | 7.5313 | 0 |
| fast54-swiglu-clamped-cpu | 4.72823119 | 2.73744696 | 7.5313 | 0 |

Delta (`clamped - control`):
- `val_bpb`: `-0.00013196` (slight improvement).
- `val_loss`: `-0.00022793` (slight improvement).
- Loss-spike proxy: unchanged (`0` vs `0`).

Decision from measured CPU-subset evidence only:
- Ranking rows in `measurement_evidence.json`: control rank `6/24`, clamped rank `6/25`.
- Runner decision: `promotionDecision=hold`, `retirementDecision=retire` for both FAST-54 runs.
- This is not benchmark or leaderboard progress; it is a CPU-subset stability screen signal only.
