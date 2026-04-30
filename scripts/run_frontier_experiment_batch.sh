#!/usr/bin/env bash
set -euo pipefail

if [ -z "${PARAMETER_GOLF_DIR:-}" ]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PARAMETER_GOLF_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

cd "$PARAMETER_GOLF_DIR"

PG_QUEUE="${PG_QUEUE:-planning/frontier_gpu_experiment_queue.json}"
PG_GPUS="${PG_GPUS:-1}"
PG_STATUS="${PG_STATUS:-ready}"
PG_ONLY="${PG_ONLY:-}"
PG_LANES="${PG_LANES:-${PG_LANE:-}}"
PG_SEEDS="${PG_SEEDS:-}"
PG_OUTPUT_ROOT="${PG_OUTPUT_ROOT:-records/frontier_h100_candidate_batch}"
PG_EXPORT_ROOT="${PG_EXPORT_ROOT:-records/frontier_gpu_experiment_queue_exports}"
PG_TIMEOUT_SECONDS="${PG_TIMEOUT_SECONDS:-900}"
PG_BATCH_TUNE_TARGET_MEMORY_FRACTION="${PG_BATCH_TUNE_TARGET_MEMORY_FRACTION:-0.90}"
PG_BATCH_TUNE_MAX_TOKENS="${PG_BATCH_TUNE_MAX_TOKENS:-2097152}"
PG_BATCH_TUNE_TIMEOUT_SECONDS="${PG_BATCH_TUNE_TIMEOUT_SECONDS:-180}"
PG_SHARD_COUNT="${PG_SHARD_COUNT:-1}"
PG_SHARD_INDEX="${PG_SHARD_INDEX:-0}"
PG_STOP_ON_FAILURE="${PG_STOP_ON_FAILURE:-0}"
PG_SMOKE="${PG_SMOKE:-0}"
PG_TUNE_ONLY="${PG_TUNE_ONLY:-0}"
PG_EXPORT_ONLY="${PG_EXPORT_ONLY:-0}"
PG_EXPORT_ALL_SHARDS="${PG_EXPORT_ALL_SHARDS:-0}"
PG_KEEP_RAW_CHECKPOINTS="${PG_KEEP_RAW_CHECKPOINTS:-0}"
PG_DRY_RUN="${PG_DRY_RUN:-0}"
PG_PRINT_INVENTORY="${PG_PRINT_INVENTORY:-0}"
PG_ALLOW_FEWER_GPUS="${PG_ALLOW_FEWER_GPUS:-0}"
PG_LOG_ROOT="${PG_LOG_ROOT:-records/frontier_deadline_logs}"
PG_PACKAGE_SUBMISSION="${PG_PACKAGE_SUBMISSION:-0}"
PG_SUBMISSION_AUTHOR="${PG_SUBMISSION_AUTHOR:-}"
PG_SUBMISSION_GITHUB_ID="${PG_SUBMISSION_GITHUB_ID:-}"
PG_SUBMISSION_NAME="${PG_SUBMISSION_NAME:-}"
PG_SUBMISSION_CANDIDATE_ID="${PG_SUBMISSION_CANDIDATE_ID:-}"
PG_SUBMISSION_OUTPUT_DIR="${PG_SUBMISSION_OUTPUT_DIR:-}"
PG_SUBMISSION_SLUG="${PG_SUBMISSION_SLUG:-}"
PG_SUBMISSION_FORCE="${PG_SUBMISSION_FORCE:-0}"

if [ -z "${PG_SUBMISSION_TRACK+x}" ]; then
  if [ "$PG_GPUS" -eq 8 ]; then
    PG_SUBMISSION_TRACK="10min_16mb"
  else
    PG_SUBMISSION_TRACK="non-record-unlimited-compute-16mb"
  fi
fi
if [ -z "${PG_SUBMISSION_RECORDS_ROOT+x}" ]; then
  if [ "$PG_GPUS" -eq 8 ]; then
    PG_SUBMISSION_RECORDS_ROOT="records/track_10min_16mb"
  else
    PG_SUBMISSION_RECORDS_ROOT="records/track_non_record_16mb"
  fi
fi

if ! [[ "$PG_GPUS" =~ ^[0-9]+$ ]] || [ "$PG_GPUS" -lt 1 ]; then
  echo "PG_GPUS must be a positive integer, got '$PG_GPUS'" >&2
  exit 2
fi

if [ -z "${PG_MAX_EXPERIMENTS+x}" ]; then
  if [ "$PG_GPUS" -eq 1 ]; then
    PG_MAX_EXPERIMENTS=8
  else
    PG_MAX_EXPERIMENTS=3
  fi
fi

if [ -z "${PG_AUTO_TUNE+x}" ]; then
  if [ "$PG_GPUS" -eq 1 ]; then
    PG_AUTO_TUNE=1
  else
    PG_AUTO_TUNE=0
  fi
fi

bool_true() {
  case "${1:-}" in
    1 | true | TRUE | yes | YES | y | Y) return 0 ;;
    *) return 1 ;;
  esac
}

split_csv() {
  local raw="$1"
  local item
  raw="${raw// /}"
  if [ -z "$raw" ]; then
    return 0
  fi
  IFS=',' read -ra items <<< "$raw"
  for item in "${items[@]}"; do
    if [ -n "$item" ]; then
      printf '%s\n' "$item"
    fi
  done
}

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L || true
  available_gpu_count="$(nvidia-smi -L 2>/dev/null | sed '/^$/d' | wc -l | tr -d ' ')"
  if [ "${available_gpu_count:-0}" -lt "$PG_GPUS" ] && ! bool_true "$PG_ALLOW_FEWER_GPUS"; then
    echo "Requested PG_GPUS=$PG_GPUS but nvidia-smi reports ${available_gpu_count:-0} GPU(s)." >&2
    echo "Set PG_ALLOW_FEWER_GPUS=1 only for dry-run or nonstandard debugging." >&2
    exit 2
  fi
fi

export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"

queue_args=(
  --queue "$PG_QUEUE"
  --export-root "$PG_EXPORT_ROOT"
  --output-root "$PG_OUTPUT_ROOT"
  --timeout-seconds "$PG_TIMEOUT_SECONDS"
  --nproc-per-node "$PG_GPUS"
  --shard-count "$PG_SHARD_COUNT"
  --shard-index "$PG_SHARD_INDEX"
)

while IFS= read -r status; do
  queue_args+=(--status "$status")
done < <(split_csv "$PG_STATUS")

inferred_lanes=()
if [ -n "$PG_ONLY" ]; then
  while IFS= read -r candidate_id; do
    queue_args+=(--only "$candidate_id")
  done < <(split_csv "$PG_ONLY")
elif [ -n "$PG_LANES" ]; then
  while IFS= read -r lane; do
    queue_args+=(--lane "$lane")
  done < <(split_csv "$PG_LANES")
else
  mapfile -t inferred_lanes < <(python3 - "$PG_QUEUE" "$PG_GPUS" <<'PY'
import json
import sys
from pathlib import Path

queue_path = Path(sys.argv[1])
gpus = sys.argv[2]
payload = json.loads(queue_path.read_text(encoding="utf-8"))
prefix = f"h100-{gpus}x-frontier-"
lanes = sorted(
    {
        str(experiment.get("lane") or "")
        for experiment in payload.get("experiments", [])
        if str(experiment.get("lane") or "").startswith(prefix)
    }
)
for lane in lanes:
    print(lane)
PY
)
  for lane in "${inferred_lanes[@]}"; do
    queue_args+=(--lane "$lane")
  done
fi

if [ -n "$PG_MAX_EXPERIMENTS" ]; then
  queue_args+=(--max-experiments "$PG_MAX_EXPERIMENTS")
fi
if [ -n "$PG_SEEDS" ]; then
  while IFS= read -r seed; do
    queue_args+=(--seed "$seed")
  done < <(split_csv "$PG_SEEDS")
fi

if bool_true "$PG_PRINT_INVENTORY"; then
  queue_args+=(--print-inventory)
fi
if bool_true "$PG_DRY_RUN"; then
  queue_args+=(--dry-run --print-inventory)
fi
if bool_true "$PG_EXPORT_ONLY"; then
  queue_args+=(--export-only)
fi
if bool_true "$PG_EXPORT_ALL_SHARDS"; then
  queue_args+=(--export-all-shards --export-only)
fi
if bool_true "$PG_STOP_ON_FAILURE"; then
  queue_args+=(--stop-on-failure)
fi
if bool_true "$PG_SMOKE"; then
  queue_args+=(--smoke)
fi
if bool_true "$PG_TUNE_ONLY"; then
  queue_args+=(--tune-only)
fi
if bool_true "$PG_KEEP_RAW_CHECKPOINTS"; then
  queue_args+=(--keep-raw-checkpoints)
fi
if bool_true "$PG_AUTO_TUNE"; then
  queue_args+=(
    --auto-tune-batch
    --batch-tune-target-memory-fraction "$PG_BATCH_TUNE_TARGET_MEMORY_FRACTION"
    --batch-tune-max-tokens "$PG_BATCH_TUNE_MAX_TOKENS"
    --batch-tune-timeout-seconds "$PG_BATCH_TUNE_TIMEOUT_SECONDS"
  )
fi

mkdir -p "$PG_LOG_ROOT" "$PG_OUTPUT_ROOT"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_log="$PG_LOG_ROOT/frontier_batch_${timestamp}_${PG_GPUS}x.log"

echo "Parameter Golf frontier batch"
echo "repo:     $PARAMETER_GOLF_DIR"
echo "queue:    $PG_QUEUE"
echo "gpus:     $PG_GPUS"
echo "statuses: $PG_STATUS"
echo "only:     ${PG_ONLY:-<none>}"
echo "lanes:    ${PG_LANES:-${inferred_lanes[*]:-<none>}}"
echo "seeds:    ${PG_SEEDS:-<default>}"
echo "max:      ${PG_MAX_EXPERIMENTS:-<none>}"
echo "log:      $run_log"
printf 'command: python3 fastest/scripts/run_gpu_experiment_queue.py'
printf ' %q' "${queue_args[@]}"
printf '\n'

set +e
python3 fastest/scripts/run_gpu_experiment_queue.py "${queue_args[@]}" 2>&1 | tee "$run_log"
rc=${PIPESTATUS[0]}
set -e

latest_output=""
if ! bool_true "$PG_DRY_RUN" && ! bool_true "$PG_EXPORT_ONLY" && ! bool_true "$PG_EXPORT_ALL_SHARDS" && [ -d "$PG_OUTPUT_ROOT" ]; then
  latest_output="$(find "$PG_OUTPUT_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2- || true)"
fi

if [ -n "$latest_output" ]; then
  echo
  echo "Latest artifact directory: $latest_output"
  [ -f "$latest_output/leaderboard.csv" ] && echo "Leaderboard: $latest_output/leaderboard.csv"
  [ -f "$latest_output/summary.md" ] && echo "Summary:     $latest_output/summary.md"
  [ -f "$latest_output/model_factory_evidence.json" ] && echo "Evidence:    $latest_output/model_factory_evidence.json"
  [ -f "$latest_output/charts/README.md" ] && echo "Charts:      $latest_output/charts/README.md"
  [ -f "$latest_output/charts/final_val_bpb.svg" ] && echo "Bpb chart:   $latest_output/charts/final_val_bpb.svg"
  [ -f "$latest_output/charts/speed_vs_bpb.svg" ] && echo "Speed chart: $latest_output/charts/speed_vs_bpb.svg"
  [ -d "$latest_output/runs" ] && echo "Run curves:  $latest_output/runs/*/training_curve.svg"

  if bool_true "$PG_PACKAGE_SUBMISSION"; then
    package_args=(
      --batch-dir "$latest_output"
      --records-root "$PG_SUBMISSION_RECORDS_ROOT"
      --track "$PG_SUBMISSION_TRACK"
      --nproc-per-node "$PG_GPUS"
    )
    [ -n "$PG_SUBMISSION_AUTHOR" ] && package_args+=(--author "$PG_SUBMISSION_AUTHOR")
    [ -n "$PG_SUBMISSION_GITHUB_ID" ] && package_args+=(--github-id "$PG_SUBMISSION_GITHUB_ID")
    [ -n "$PG_SUBMISSION_NAME" ] && package_args+=(--name "$PG_SUBMISSION_NAME")
    [ -n "$PG_SUBMISSION_CANDIDATE_ID" ] && package_args+=(--candidate-id "$PG_SUBMISSION_CANDIDATE_ID")
    [ -n "$PG_SUBMISSION_OUTPUT_DIR" ] && package_args+=(--output-dir "$PG_SUBMISSION_OUTPUT_DIR")
    [ -n "$PG_SUBMISSION_SLUG" ] && package_args+=(--slug "$PG_SUBMISSION_SLUG")
    bool_true "$PG_SUBMISSION_FORCE" && package_args+=(--force)
    echo
    python3 fastest/scripts/package_competition_submission.py "${package_args[@]}"
  fi
fi

exit "$rc"
