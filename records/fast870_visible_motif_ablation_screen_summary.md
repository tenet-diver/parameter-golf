# FAST-870 Visible Motif Ablation Screen

Trusted control: `control-2026-04-09-sp8192-3layer-parres-qk525-legalttt`
from `records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT/README.md`.
Control score is 1.0810 FineWeb validation bpb with max visible artifact
15,993,232 bytes.

| Motif id | Score delta bpb | Artifact delta bytes | Legality | Isolation | Next action |
| --- | ---: | ---: | --- | --- | --- |
| sp8192-tokenizer | +0.008700 | +6,301 | legal | partial-confounded | isolate-tokenizer-toggle-before-combination |
| 3-layer-recurrence | +0.001200 | -1,746 | legal | partial-confounded | keep-and-run-clean-recurrence-toggle |
| parallel-residuals | +0.001790 | -686 | legal | partial-confounded | keep-and-run-clean-parallel-residual-toggle |
| legal-score-first-ttt | +0.001700 | 0 | legal | clean-visible-contrast | keep-with-eval-budget-watch |
| qk-gain-tuning | +0.001790 | -686 | legal | partial-confounded | run-clean-qk5p0-to-qk5p25-sweep |
| hessian-aware-clipping | +0.002540 | -13,583 | legal | partial-confounded | port-hessian-clipping-onto-control-stack |

Positive score delta means the ablation or contrast scored worse than the
trusted control, so the control motif remains provisionally useful. The only
clean visible contrast is `legal-score-first-ttt`, using the control record's
own sliding bpb versus post-TTT bpb. The other rows are cheap screens from
leaderboard evidence and are intentionally marked partial where a visible
record cannot isolate the motif without changing adjacent components.

All rows remain under the 16MB artifact limit. The `sp8192-tokenizer` contrast
is legal but has only 467 bytes of artifact headroom, so it should not be
combined until a clean tokenizer toggle confirms both score and packaging
margin. Rows with partial isolation should feed a single clean toggle or narrow
sweep before any combination attempt.
