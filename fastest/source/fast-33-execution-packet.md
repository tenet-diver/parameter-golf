# FAST-33 Ablation Execution Packet

- Candidate ID: `exp-fast33-ablation-001`
- Lane: `ablation`
- Result: `completed`

## Exact command
1. `python fastest/scripts/run_bounded_ablation_candidate.py --task fastest/generated/fast33_ablation_task.json --output fastest/generated/fast33_ablation_outcome.json --evidence-path fastest/source/measurement_evidence.json --status-path fastest/generated/campaign_status.json --state-path fastest/generated/campaign_state.json`

## Artifact paths
- `fastest/generated/fast33_ablation_task.json`
- `fastest/generated/fast33_ablation_outcome.json`
- `fastest/source/measurement_evidence.json`
- `fastest/generated/campaign_status.json`
- `fastest/generated/campaign_state.json`
- `fastest/source/fast-33-execution-packet.md`

## Observed metric output
- Experiment: `exp-fast33-ablation-001`
- Metric: `benchmark-score`
- Value: `1.025`
- Completed at: `2026-04-24T16:59:56Z`

## Promote/reject rationale
- Decision: `reject`
- Reason: candidate objective `1.025` does not beat the current best-known benchmark-score `1.0999` from `exp-fast37-combination-001`.
