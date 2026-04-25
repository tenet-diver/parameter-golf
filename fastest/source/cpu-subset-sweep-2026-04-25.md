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
