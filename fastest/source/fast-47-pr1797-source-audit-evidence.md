# FAST-47 PR1797 Source Audit Evidence

Date: 2026-04-25
Task: FAST-47
Candidate PR: #1797 (`openai/parameter-golf`)
Lane: `legal-evidence:source-level-audit`

## Scope
Audit PR #1797 as the top held low-BPB candidate and verify source-level runnable train/eval/submission legality evidence before any promotion.

## Commands and observed results
- `mcp__codex_apps__github._get_pr_info(repository_full_name=openai/parameter-golf, pr_number=1797)`
  - Observed: open PR, head sha `04d35edaad74fc88b5ef08a814c94596b616ec1b`, 6 commits, 20 changed files, PR body claims 3-seed mean `val_bpb = 1.06157`.
- `mcp__codex_apps__github._list_pr_changed_filenames(repo_full_name=openai/parameter-golf, pr_number=1797)`
  - Observed: changed-file list contains PR-head submission folder with `README.md`, `submission.json`, `prepare_caseops_data.py`, `train_gpt.py`, and `train_seed1234/314/42.log`.
- `mcp__codex_apps__github._fetch_file(.../train_gpt.py, ref=04d35...)`
  - Observed: source-level train/eval implementation entrypoints are present (`train_model`, `eval_val_ttt_phased`, `train_and_eval`, `main`).
- `mcp__codex_apps__github._fetch_pr_file_patch(.../submission.json)`
  - Observed: submission source includes per-seed train/eval time, artifact bytes, and metric values (`val_bpb: 1.06157`, `artifact_bytes_max: 15953718`).
- `mcp__codex_apps__github._fetch_pr_file_patch(.../README.md)`
  - Observed: source-level runnable command paths for data prep (`prepare_caseops_data.py`) and 3-seed train/eval loop using `torchrun ... train_gpt.py`.
- `mcp__codex_apps__github._fetch_pr_file_patch(.../prepare_caseops_data.py)`
  - Observed: legality-path source wires canonical byte-count exporter `surface_piece_original_byte_counts(...)` and BOS-per-document shard behavior (`BOS_ID=1`) for eval parity.
- `mcp__codex_apps__github._fetch_pr_file_patch(.../train_seed314.log)`
  - Observed: run log body includes `stopping_early: wallclock_cap train_time: 599474ms`, quantized submission size `15951189` bytes, and `quantized_ttt_phased val_bpb:1.06082659 total_eval_time:494.8s`.
- `mcp__codex_apps__github._fetch_pr_comments(repo_full_name=openai/parameter-golf, pr_number=1797)`
  - Observed: durable discussion comment (`issuecomment-4309925822`) attributes commit `04d35ed` byte-count fix and states canonical sidecar parity verification.

## Runnable and legality assessment
- Train path proof: complete.
- Eval path proof: complete.
- Submission path proof: complete.
- Legality gate proof: complete.
- Discussion evidence: complete.

## Imported attributed candidate artifacts
- `records/track_10min_16mb/2026-04-24_PR1787Base_Smear_LQERAsym_PhasedTTT_1.06157/ATTRIBUTION.md`
- `records/track_10min_16mb/2026-04-24_PR1787Base_Smear_LQERAsym_PhasedTTT_1.06157/README.md`
- `records/track_10min_16mb/2026-04-24_PR1787Base_Smear_LQERAsym_PhasedTTT_1.06157/submission.json`
- `records/track_10min_16mb/2026-04-24_PR1787Base_Smear_LQERAsym_PhasedTTT_1.06157/prepare_caseops_data.py.source_excerpt.md`
- `records/track_10min_16mb/2026-04-24_PR1787Base_Smear_LQERAsym_PhasedTTT_1.06157/train_gpt.py.source_excerpt.md`
- `records/track_10min_16mb/2026-04-24_PR1787Base_Smear_LQERAsym_PhasedTTT_1.06157/train_seed42.log.excerpt.txt`
- `records/track_10min_16mb/2026-04-24_PR1787Base_Smear_LQERAsym_PhasedTTT_1.06157/train_seed314.log.excerpt.txt`
- `records/track_10min_16mb/2026-04-24_PR1787Base_Smear_LQERAsym_PhasedTTT_1.06157/train_seed1234.log.excerpt.txt`

## Decision
Decision: `advance-to-external-reproduction`

Rationale: source-file-level runnable legality proof is complete for PR #1797 from PR-head files, train/eval implementation source, seed log source, and discussion evidence. The remaining gate is external reproduction on required 8xH100 hardware; this audit still does not call the submission winning.

## Artifacts
- `planning/fast47_pr1797_source_audit_execution_packet.json`
- `fastest/source/fast-47-pr1797-source-audit-evidence.md`
- `fastest/source/measurement_evidence.json`
- `records/track_10min_16mb/2026-04-24_PR1787Base_Smear_LQERAsym_PhasedTTT_1.06157/*`
