# FAST-47 PR1797 Source Audit Evidence

Date: 2026-04-25
Task: FAST-47
Candidate PR: #1797 (`openai/parameter-golf`)
Lane: `legal-evidence:source-level-audit`

## Scope
Audit PR #1797 as the top held low-BPB candidate and verify source-level runnable train/eval/submission legality evidence before any promotion.

## Commands and observed results
- `mcp__codex_apps__github._get_pr_info(repository_full_name=openai/parameter-golf, pr_number=1797)`
  - Observed: PR is open, head sha `04d35edaad74fc88b5ef08a814c94596b616ec1b`, 6 commits, 20 changed files, body claims 3-seed mean `val_bpb = 1.06157`.
- `web.open https://github.com/openai/parameter-golf/pull/1797/files`
  - Observed: files tree contains the 2026-04-24 record directory with `README.md`, `submission.json`, `train_gpt.py`, and seed logs (`train_seed1234.log`, `train_seed314.log`, `train_seed42.log`).
  - Observed: repeated UI retrieval failures: "Failed to load comments", "Failed to load files", and "There was an error while loading".
- `web.click turn1view0 id=89` then `web.open turn4view1 lineno=400`
  - Observed: commit-level source diff for `prepare_caseops_data.py` shows canonical byte-count exporter wiring (`surface_piece_original_byte_counts(...)`) and typed return path.

## Runnable and legality assessment
- Train path proof: partial (file presence confirmed, but direct `train_gpt.py` head content not fully captured in this run).
- Eval path proof: partial (PR body test-plan claims captured; source-level eval path lines not fully captured).
- Submission path proof: partial (`submission.json` file presence confirmed; direct payload content not captured).
- Legality gate proof: partial (diff/commit text shows byte-count correction and canonical exporter claim; full end-to-end source proof still incomplete).

## Decision
Decision: `blocked`

Rationale: source-level runnable legality proof is incomplete from captured artifacts. This run has durable metadata + file-tree + commit-diff evidence, but does not yet include direct PR head `train_gpt.py` body, `submission.json` body, log body lines, or fully loaded discussion evidence.

Promotion outcome: do not call winning; do not advance to external reproduction until the missing source-level proofs are captured.

## Artifacts
- `planning/fast47_pr1797_source_audit_execution_packet.json`
- `fastest/source/fast-47-pr1797-source-audit-evidence.md`
- `fastest/source/measurement_evidence.json`
