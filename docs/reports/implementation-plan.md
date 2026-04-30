# FAST-871 Implementation Plan

## Tests first

1. Add `tests/test_fast871_motif_composition_matrix.py` to require a FAST-871
   composition matrix, summary, architecture handoff, and implementation plan.
2. Assert ranked combinations expose motif IDs, interaction class, verdict,
   artifact headroom, evidence paths, and follow-up tasks.
3. Assert positive, neutral, and negative interactions all appear so the next
   planning cycle can promote, hold, or reject work deliberately.

## Implementation

1. Use FAST-870 visible motif evidence as the source packet.
2. Rank combinations by submission relevance, artifact headroom, visible bpb,
   and whether the interaction is clean enough to justify more work.
3. Mark rows as `promote`, `hold`, or `reject`; include follow-up task packets
   only for promote/hold rows.
4. Keep all claims at planning-evidence strength until a real runner validates
   them.

## Validators

- `python -m unittest tests.test_fast871_motif_composition_matrix`
- `git diff --check`

## Handoff

The next executable work is to convert promoted rows into cheap-screen or
one-H100 rehearsal packets with explicit stop conditions.

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
