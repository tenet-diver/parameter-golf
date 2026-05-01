#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PG_QUEUE="${PG_QUEUE:-planning/four_h100_best_experiment_queue.json}"
export PG_GPUS="${PG_GPUS:-4}"
export PG_STATUS="${PG_STATUS:-ready}"
export PG_MAX_EXPERIMENTS="${PG_MAX_EXPERIMENTS:-4}"
export PG_OUTPUT_ROOT="${PG_OUTPUT_ROOT:-records/h100_4x_best_candidate_batch}"
export PG_EXPORT_ROOT="${PG_EXPORT_ROOT:-records/h100_4x_best_queue_exports}"
export PG_LOG_ROOT="${PG_LOG_ROOT:-records/h100_4x_best_logs}"
export PG_TIMEOUT_SECONDS="${PG_TIMEOUT_SECONDS:-900}"
export PG_AUTO_TUNE="${PG_AUTO_TUNE:-0}"

exec "$SCRIPT_DIR/run_deadline_experiment_batch.sh"
