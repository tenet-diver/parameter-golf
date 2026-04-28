# FAST-870 Implementation Plan

## Tests first

1. Add `tests/test_fast870_visible_motif_ablation_packet.py` to require a
   FAST-870 result ledger, summary, architecture handoff, and implementation
   plan.
2. Assert one trusted control baseline and exactly six motif rows:
   `sp8192-tokenizer`, `3-layer-recurrence`, `parallel-residuals`,
   `legal-score-first-ttt`, `qk-gain-tuning`, and
   `hessian-aware-clipping`.
3. Assert every row records score delta, artifact delta, legality class, run
   estimate, command or entrypoint, seed policy, artifact path, and next action.

## Implementation

1. Use the 2026-04-09 SP8192 + 3-layer recurrence + parallel residuals +
   QK-Gain 5.25 + legal TTT record as the trusted control.
2. Populate a FAST-870 JSON ledger from checked-in leaderboard records, using
   max visible artifact bytes for size accountability.
3. Mark visible contrasts as `partial-confounded` where the leaderboard record
   changes more than the target motif. Do not promote those rows directly to a
   combination candidate.
4. Write a concise markdown summary for operator review.
5. Record this architecture handoff and rollback/validation constraints in
   `docs/reports/architecture-handoff.md`.

## Validators

- `python -m unittest tests.test_fast870_visible_motif_ablation_packet`
- `git diff --check`

## Handoff

The packet optimizes information gain rather than leaderboard maximization.
The next executable work is to run clean toggles for partial rows before any
combined candidate: tokenizer, recurrence depth, parallel residuals, QK gain,
and Hessian-aware clipping on the trusted control stack.
