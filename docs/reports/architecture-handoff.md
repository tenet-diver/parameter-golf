# FAST-870 Architecture Handoff

Date: 2026-04-28
Task: FAST-870 Run cheap ablations on strongest visible Parameter Golf motifs

## Boundary

Implementation stays in `/workspaces/parameter-golf`. No Fastest runtime, API,
or fast-ui interface changes are required.
No Fastest runtime, API, or fast-ui interface changes are part of this task.
The work product is a repo-local experiment packet that converts visible
leaderboard evidence into a repeatable ablation screen before combination
attempts.

## Changed Experiment Surfaces

- `records/fast870_visible_motif_ablation_screen_results.json` records the
  trusted control, six visible motif contrasts, score deltas, artifact deltas,
  legality classes, isolation classes, run estimates, and next actions.
- `records/fast870_visible_motif_ablation_screen_summary.md` gives the operator
  handoff table for quick review.
- `docs/reports/implementation-plan.md` captures the tests-first execution
  packet and validators used for this spike.

## Control Decision

The trusted control is the strongest visible 10min_16mb record:
`records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT/README.md`.
It reports a 3-seed mean of 1.0810 bpb, max visible artifact of 15,993,232
bytes, and explicit score-first TTT legality evidence.

## Interface Contract

Every ablation row must expose:

- motif id
- control config id
- ablation config id
- command or script entrypoint
- seed policy
- artifact path
- artifact bytes
- validation bpb
- score delta bpb
- artifact delta bytes
- legality class
- run duration estimate
- next action

Rows may also carry `isolationClass` and `confounds` because several visible
leaderboard records do not isolate a single motif cleanly. A partial visible
contrast is useful for screening, but it is not sufficient evidence for a
combination candidate.

## Rollback Constraints

Rollback is file-local: remove the FAST-870 files under `records/` and
`docs/reports/`, plus the targeted FAST-870 contract test. No generated model
artifacts, task-store state, or platform routes are changed.

## Validation Evidence

Use `python -m unittest tests.test_fast870_visible_motif_ablation_packet` for
the packet contract and `git diff --check` for markdown/JSON whitespace hygiene.
No GPU training is required for this documentation and ledger spike because the
source of truth is visible checked-in leaderboard evidence.
