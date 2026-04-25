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
- `web.open https://github.com/openai/parameter-golf/pull/1797 lineno=205`
  - Observed: conversation captures 3-seed metric table and budget/cap claims (`<= 16,000,000` bytes and `<= 600s` train/eval).
- `web.open https://github.com/openai/parameter-golf/pull/1797/files lineno=295`
  - Observed: files tree lines `313-323` contain the PR-head `2026-04-24_PR1787Base_Smear_LQERAsym_PhasedTTT_1.06157` folder with `submission.json`, `train_gpt.py`, and `train_seed1234/314/42.log`.
  - Observed: repeated UI retrieval failures include "Failed to load comments", "Failed to load files", and "There was an error while loading".
- `web.open https://github.com/openai/parameter-golf/pull/1797/commits/04d35edaad74fc88b5ef08a814c94596b616ec1b lineno=418`
  - Observed: commit-level source diff for `prepare_caseops_data.py` includes canonical byte-count exporter wiring:
    - `byte_counts = surface_piece_original_byte_counts(...)`
    - `return np.asarray(list(byte_counts), dtype=np.uint16)`

## Runnable and legality assessment
- Train path proof: partial (file presence confirmed, but direct `train_gpt.py` head content not fully captured in this run).
- Eval path proof: partial (PR body test-plan claims captured; source-level eval path lines not fully captured).
- Submission path proof: partial (`submission.json` file presence confirmed; direct payload content not captured).
- Legality gate proof: partial (diff/commit text shows byte-count correction and canonical exporter claim; full end-to-end source proof still incomplete).
- Blocker matrix (from execution packet):
  - `train`: `missing_train_gpt_source_body`
  - `eval`: `missing_eval_source_body`
  - `submission`: `missing_submission_json_body`
  - `logs`: `missing_train_log_body_lines`
  - `discussion`: `github_files_discussion_load_error`

## Decision
Decision: `blocked`

Rationale: source-level runnable legality proof is incomplete from captured artifacts. This run has durable metadata + file-tree + commit-diff evidence, but does not yet include direct PR head `train_gpt.py` body, `submission.json` body, log body lines, or fully loaded discussion evidence.

Promotion outcome: do not call winning; do not advance to external reproduction until the missing source-level proofs are captured.

## Artifacts
- `planning/fast47_pr1797_source_audit_execution_packet.json`
- `fastest/source/fast-47-pr1797-source-audit-evidence.md`
- `fastest/source/measurement_evidence.json`
