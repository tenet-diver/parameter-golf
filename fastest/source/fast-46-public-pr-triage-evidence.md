# FAST-46 Public PR Triage Evidence Note

Date: 2026-04-24
Snapshot time (UTC): 2026-04-24T19:16:03Z
Campaign lane: `target-repo-campaign`
Triage scope: open public PRs claiming `val_bpb < 1.0785`

## Repro commands (audited trail)
- `web.open https://github.com/openai/parameter-golf/pulls`
- `web.open https://github.com/openai/parameter-golf/pull/1698`
- `web.open https://github.com/openai/parameter-golf/pull/1722`
- `web.open https://github.com/openai/parameter-golf/pull/1795`
- `web.open https://github.com/openai/parameter-golf/pull/1797`
- `web.open https://github.com/openai/parameter-golf/pull/1801`
- `web.open https://github.com/openai/parameter-golf/pull/1807`

## Captured universe (`open`, `<1.0785`)
- Required audited set: `#1698`, `#1722`, `#1795`, `#1797`, `#1801`, `#1807`
- Additional open low-BPB entries from pulls index snapshot: `#1729` (`1.0678`), `#1727` (`1.07217`), `#1707` (`1.07399`), `#1728` (`1.07706`)
- Coverage status: complete for captured universe in this run (`universeCoverage.coverageStatus = complete-for-captured-universe`)

## Ranked disposition summary
- `#1797` (`hold`): strongest claim among inspected set, but only open-ref excerpts were captured; no PR file/diff source evidence for legality+runnable gates.
- `#1801` (`hold`): strong score, runnable evidence present, legality interpretation still review-sensitive.
- `#1729` (`hold`): index-level strong claim but no per-PR legality/runnable snapshot captured in this run.
- `#1807` (`hold`): runnable packet present; pre-quant framing and legal interpretation risk.
- `#1707` (`hold`), `#1728` (`hold`), `#1727` (`hold`): index-level claims only in this run.
- `#1698` (`blocked`): posted bytes conflict with decimal cap interpretation.
- `#1795` (`blocked`): legality ruling pending and title/body claim mismatch.
- `#1722` (`illegal`): SLOT v3 + Pre-Quant TTT title and no direct runnable packet available in this run.

## Decision
Decision: no candidate advanced.

Rationale: this run captured PR/index page excerpts only and did not capture PR file/diff source lines proving runnable `train/eval/submission` paths plus legality gates, so all candidates remain `hold`/`blocked`/`illegal`.

## Evidence record IDs
- Universe/index: `E-UNIVERSE-INDEX-2026-04-24`
- PR #1698: `E-1698-TITLE-STATE`, `E-1698-CLAIM-TABLE`, `E-1698-LEGALITY-SIZE-RISK`, `E-1698-RUNNABLE-COMMAND`
- PR #1722: `E-1722-PR-INDEX`, `E-1722-OPENREF-ERROR`
- PR #1795: `E-1795-TITLE-STATE`, `E-1795-CLAIM-DISCREPANCY`, `E-1795-LEGALITY-PENDING`, `E-1795-RUNNABLE-EVIDENCE`
- PR #1797: `E-1797-TITLE-STATE`, `E-1797-BUDGETS-CAP`, `E-1797-LEGALITY-RUNNABLE`, `E-1797-CODE-COMPLETE`
- PR #1801: `E-1801-TITLE-STATE`, `E-1801-RESULTS`, `E-1801-LEGALITY-THREAD`, `E-1801-RUNNABLE-TESTPLAN`
- PR #1807: `E-1807-TITLE-STATE`, `E-1807-RESULTS-BUDGET`, `E-1807-LEGALITY-COMPLIANCE`
- Additional universe entries: `E-1729-INDEX-ENTRY`, `E-1728-INDEX-ENTRY`, `E-1727-INDEX-ENTRY`, `E-1707-INDEX-ENTRY`

## Artifact paths
- `planning/fast46_public_pr_triage_execution_packet.json`
- `fastest/source/fast-46-public-pr-triage-evidence.md`
- `tests/test_fast46_public_pr_triage_packet.py`
