#!/usr/bin/env bash
set -euo pipefail

# Runs only the best experiment behind PR 2069:
#   best4x_ttt_disabled_qk525
#
# Expected setup:
#   PARAMETER_GOLF_REF=codex/frontier-deadline-experiments
#   scripts/setup_runpod_parameter_golf_deadline.sh

PARAMETER_GOLF_DIR="${PARAMETER_GOLF_DIR:-/workspace/parameter-golf}"
PG_REF="${PG_REF:-codex/frontier-deadline-experiments}"

cd "$PARAMETER_GOLF_DIR"

if [ -f /workspace/pg_deadline_env.sh ]; then
  # shellcheck disable=SC1091
  source /workspace/pg_deadline_env.sh
fi

git fetch origin "$PG_REF" || true
if git rev-parse --verify "origin/$PG_REF" >/dev/null 2>&1; then
  git switch "$PG_REF"
  git pull --ff-only origin "$PG_REF" || true
fi

export PG_QUEUE="${PG_QUEUE:-planning/four_h100_best_experiment_queue.json}"
export PG_GPUS="${PG_GPUS:-8}"
export PG_ONLY="${PG_ONLY:-best4x_ttt_disabled_qk525}"
export PG_STATUS="${PG_STATUS:-ready}"
export PG_MAX_EXPERIMENTS="${PG_MAX_EXPERIMENTS:-1}"
export PG_OUTPUT_ROOT="${PG_OUTPUT_ROOT:-records/pr2069_best_8xh100}"
export PG_EXPORT_ROOT="${PG_EXPORT_ROOT:-records/pr2069_best_8xh100_exports}"
export PG_LOG_ROOT="${PG_LOG_ROOT:-records/pr2069_best_8xh100_logs}"
export PG_TIMEOUT_SECONDS="${PG_TIMEOUT_SECONDS:-900}"
export PG_STOP_ON_FAILURE="${PG_STOP_ON_FAILURE:-1}"
export PG_AUTO_TUNE="${PG_AUTO_TUNE:-0}"
export PG_PACKAGE_SUBMISSION="${PG_PACKAGE_SUBMISSION:-1}"
export PG_SUBMISSION_TRACK="${PG_SUBMISSION_TRACK:-10min_16mb}"
export PG_SUBMISSION_RECORDS_ROOT="${PG_SUBMISSION_RECORDS_ROOT:-records/track_10min_16mb}"
export PG_SUBMISSION_CANDIDATE_ID="${PG_SUBMISSION_CANDIDATE_ID:-best4x_ttt_disabled_qk525}"
export PG_SUBMISSION_AUTHOR="${PG_SUBMISSION_AUTHOR:-Rustam Eynaliyev}"
export PG_SUBMISSION_GITHUB_ID="${PG_SUBMISSION_GITHUB_ID:-tenet-diver}"
export PG_SUBMISSION_NAME="${PG_SUBMISSION_NAME:-8xH100 QK5.25 TTT-disabled PR2069 rerun}"
export PG_SUBMISSION_FORCE="${PG_SUBMISSION_FORCE:-1}"

run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
export PG_SUBMISSION_SLUG="${PG_SUBMISSION_SLUG:-$(date -u +%Y-%m-%d)_pr2069-best-8xh100_${run_stamp}}"
submission_dir="${PG_SUBMISSION_RECORDS_ROOT}/${PG_SUBMISSION_SLUG}"
push_branch="${PG_PUSH_BRANCH:-submission/pr2069-best-8xh100-${run_stamp}}"

echo "Running PR 2069 best candidate on ${PG_GPUS} GPUs: ${PG_ONLY}"
scripts/run_deadline_experiment_batch.sh

if [ ! -f "${submission_dir}/submission.json" ]; then
  echo "Expected packaged submission was not created: ${submission_dir}/submission.json" >&2
  exit 1
fi

if [ -z "$(git config user.name || true)" ]; then
  git config user.name "${GIT_AUTHOR_NAME:-Parameter Golf Runner}"
fi
if [ -z "$(git config user.email || true)" ]; then
  git config user.email "${GIT_AUTHOR_EMAIL:-runner@parameter-golf.local}"
fi

git switch -c "$push_branch"
git add "$submission_dir"
git commit -m "add pr2069 best 8xh100 submission package"
git push -u origin "$push_branch"

echo "Packaged submission: ${submission_dir}"
echo "Pushed branch: ${push_branch}"
