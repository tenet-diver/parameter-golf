# FAST-871 Motif Composition Matrix

FAST-871 ranks visible Parameter Golf motif combinations from FAST-870 evidence.
This is planning evidence, not benchmark-verified combination evidence.

| Rank | Combination | Interaction | Verdict | Follow-up |
| --- | --- | --- | --- | --- |
| 1 | recurrence + parallel residuals + legal score-first TTT | positive | promote | isolate stack on common control |
| 2 | recurrence + legal score-first TTT + QK gain | positive | promote | QK 5.0 vs 5.25 sweep |
| 3 | SP8192 tokenizer + recurrence + legal score-first TTT | neutral | hold | controlled tokenizer toggle |
| 4 | Hessian clipping + recurrence + parallel residuals | neutral | hold | port clipping onto control stack |
| 5 | parallel residuals + QK gain | negative | hold | clean parallel residual toggle |
| 6 | SP8192 tokenizer + Hessian clipping | negative | reject | none |

Positive rows can become rehearsal packets after clean toggles. Neutral rows
need cheap-screen evidence. Negative rows should not consume H100 time unless a
new accepted argument changes their evidence state.
