#!/usr/bin/env bash
set -euo pipefail

cd /workspace

export PARAMETER_GOLF_REPO_URL="${PARAMETER_GOLF_REPO_URL:-https://github.com/tenet-diver/parameter-golf.git}"
export PARAMETER_GOLF_REF="${PARAMETER_GOLF_REF:-codex/frontier-deadline-experiments}"
export PARAMETER_GOLF_DIR="${PARAMETER_GOLF_DIR:-/workspace/parameter-golf}"
export PG_TRAIN_SHARDS="${PG_TRAIN_SHARDS:-80}"

curl -fsSL \
  "https://raw.githubusercontent.com/tenet-diver/parameter-golf/${PARAMETER_GOLF_REF}/scripts/setup_runpod_parameter_golf_deadline.sh" \
  -o /workspace/setup_pg_deadline.sh

bash /workspace/setup_pg_deadline.sh

cd "$PARAMETER_GOLF_DIR"
source /workspace/pg_deadline_env.sh

git fetch origin "$PARAMETER_GOLF_REF"
git switch "$PARAMETER_GOLF_REF"
git pull --ff-only origin "$PARAMETER_GOLF_REF" || true

chmod +x ./run_pr2069_best_8xh100.sh

if tmux has-session -t pr2069best 2>/dev/null; then
  tmux attach -t pr2069best
else
  tmux new -s pr2069best './run_pr2069_best_8xh100.sh'
fi
