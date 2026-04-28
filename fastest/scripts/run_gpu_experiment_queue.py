from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastest.scripts import run_h100_candidate_batch


DEFAULT_QUEUE_PATH = REPO_ROOT / "planning/gpu_experiment_queue.json"
DEFAULT_EXPORT_ROOT = REPO_ROOT / "records/gpu_experiment_queue_exports"
RUNNABLE_STATUSES = {"ready", "queued"}


def utc_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_queue(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("GPU experiment queue must be a JSON object")
    experiments = payload.get("experiments")
    if not isinstance(experiments, list):
        raise ValueError("GPU experiment queue must contain an experiments list")
    for index, experiment in enumerate(experiments):
        if not isinstance(experiment, dict):
            raise ValueError(f"experiment #{index} is not an object")
        if not isinstance(experiment.get("id"), str) or not experiment["id"].strip():
            raise ValueError(f"experiment #{index} must have a non-empty id")
        candidate = experiment.get("candidate")
        if not isinstance(candidate, dict):
            raise ValueError(f"experiment {experiment['id']!r} must contain a candidate object")
        if candidate.get("id") != experiment["id"]:
            raise ValueError(f"experiment {experiment['id']!r} candidate.id must match experiment id")
    return payload


def inventory_by_lane_status(queue: dict[str, Any]) -> dict[str, dict[str, int]]:
    inventory: dict[str, dict[str, int]] = {}
    for experiment in queue["experiments"]:
        lane = str(experiment.get("lane") or "default")
        status = str(experiment.get("status") or "unknown")
        lane_inventory = inventory.setdefault(lane, {})
        lane_inventory[status] = lane_inventory.get(status, 0) + 1
    return inventory


def select_experiments(
    queue: dict[str, Any],
    *,
    statuses: set[str],
    lanes: set[str],
    only: set[str],
    max_experiments: int | None,
) -> list[dict[str, Any]]:
    selected = []
    for experiment in queue["experiments"]:
        status = str(experiment.get("status") or "")
        lane = str(experiment.get("lane") or "default")
        experiment_id = str(experiment["id"])
        if status not in statuses:
            continue
        if lanes and lane not in lanes:
            continue
        if only and experiment_id not in only:
            continue
        selected.append(experiment)
    if max_experiments is not None:
        selected = selected[:max_experiments]
    return selected


def materialize_candidate_file(
    experiments: list[dict[str, Any]],
    *,
    queue_path: Path,
    export_root: Path,
) -> Path:
    export_dir = export_root / utc_slug()
    export_dir.mkdir(parents=True, exist_ok=False)
    candidates = []
    for experiment in experiments:
        candidate = dict(experiment["candidate"])
        candidate["queueMetadata"] = {
            "queuePath": str(queue_path),
            "lane": experiment.get("lane"),
            "status": experiment.get("status"),
            "priority": experiment.get("priority"),
            "expectedSignal": experiment.get("expectedSignal"),
            "gpuPlan": experiment.get("gpuPlan"),
        }
        candidates.append(candidate)
    candidate_file = export_dir / "candidates.json"
    candidate_file.write_text(json.dumps(candidates, indent=2) + "\n", encoding="utf-8")
    (export_dir / "selected_experiments.json").write_text(
        json.dumps(experiments, indent=2) + "\n",
        encoding="utf-8",
    )
    return candidate_file


def run_queue(args: argparse.Namespace) -> int:
    queue = load_queue(args.queue)
    if args.print_inventory:
        print(json.dumps({"inventory": inventory_by_lane_status(queue)}, indent=2))
    statuses = set(args.status or RUNNABLE_STATUSES)
    lanes = set(args.lane or [])
    only = set(args.only or [])
    experiments = select_experiments(
        queue,
        statuses=statuses,
        lanes=lanes,
        only=only,
        max_experiments=args.max_experiments,
    )
    if not experiments:
        raise ValueError("no GPU experiments selected")
    print(f"selected {len(experiments)} experiment(s)")
    if args.dry_run:
        for experiment in experiments:
            print(f"{experiment['id']}: {experiment.get('lane')} {experiment.get('status')}")
        return 0
    candidate_file = materialize_candidate_file(
        experiments,
        queue_path=args.queue,
        export_root=args.export_root,
    )
    print(f"candidate_file={candidate_file}")
    batch_args = SimpleNamespace(
        candidate_file=candidate_file,
        output_root=args.output_root,
        timeout_seconds=args.timeout_seconds,
        only=[],
        max_candidates=None,
        stop_on_failure=args.stop_on_failure,
        dry_run=False,
        smoke=args.smoke,
        keep_raw_checkpoints=args.keep_raw_checkpoints,
        auto_tune_batch=args.auto_tune_batch,
        tune_only=args.tune_only,
        batch_tune_target_memory_fraction=args.batch_tune_target_memory_fraction,
        batch_tune_max_tokens=args.batch_tune_max_tokens,
        batch_tune_timeout_seconds=args.batch_tune_timeout_seconds,
    )
    return run_h100_candidate_batch.run_batch(batch_args)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run queued Parameter Golf GPU experiments.")
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE_PATH)
    parser.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT_ROOT)
    parser.add_argument("--output-root", type=Path, default=run_h100_candidate_batch.DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--status", action="append", choices=sorted(RUNNABLE_STATUSES | {"blocked", "done"}))
    parser.add_argument("--lane", action="append", default=[])
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--max-experiments", type=int, default=None)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--keep-raw-checkpoints", action="store_true")
    parser.add_argument("--auto-tune-batch", action="store_true")
    parser.add_argument("--tune-only", action="store_true")
    parser.add_argument("--print-inventory", action="store_true")
    parser.add_argument("--batch-tune-target-memory-fraction", type=float, default=0.90)
    parser.add_argument("--batch-tune-max-tokens", type=int, default=2_097_152)
    parser.add_argument("--batch-tune-timeout-seconds", type=int, default=180)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return run_queue(parse_args(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    raise SystemExit(main())
