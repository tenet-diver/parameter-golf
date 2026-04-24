# FAST-5 Visible Motif Ablation Screen

Baseline: control-2026-03-25-bigramhash3072-xsa-gptq (1.11473509 bpb)

| Motif | Delta bpb | Artifact delta bytes | Legality | Next action |
| --- | ---: | ---: | --- | --- |
| 3-layer recurrence | -0.033735 | +8382 | legal | deepen |
| legal score-first TTT | -0.031941 | +7696 | legal | combine-later |
| Hessian-aware clipping | -0.031195 | -6729 | legal | deepen |
| QK-gain tuning | -0.029878 | +7696 | legal | combine-later |
| SP8192 tokenizer | -0.029105 | +828 | legal | combine-later |
| parallel residuals | -0.008482 | -38193 | legal | rerun-near-threshold |

Lower bits-per-byte deltas are better. Rows with legality failures are blocked from
leaderboard-oriented combination until the named remediation action is complete.
