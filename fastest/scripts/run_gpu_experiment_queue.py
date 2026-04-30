from __future__ import annotations

import argparse
import json
import shlex
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
    shard_count: int = 1,
    shard_index: int = 0,
) -> list[dict[str, Any]]:
    if shard_count < 1:
        raise ValueError("shard_count must be at least 1")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("shard_index must be between 0 and shard_count - 1")
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
    selected = sorted(
        selected,
        key=lambda experiment: (
            int(experiment.get("priority", 1_000_000))
            if isinstance(experiment.get("priority"), int)
            else 1_000_000,
            str(experiment["id"]),
        ),
    )
    if shard_count > 1:
        selected = [
            experiment
            for index, experiment in enumerate(selected)
            if index % shard_count == shard_index
        ]
    if max_experiments is not None:
        selected = selected[:max_experiments]
    return selected


def materialize_candidate_file(
    experiments: list[dict[str, Any]],
    *,
    queue_path: Path,
    export_root: Path,
    export_dir: Path | None = None,
) -> Path:
    export_dir = export_dir or export_root / utc_slug()
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


def build_batch_command(args: argparse.Namespace, candidate_file: Path) -> list[str]:
    command = [
        "python",
        "fastest/scripts/run_h100_candidate_batch.py",
        "--candidate-file",
        str(candidate_file),
        "--output-root",
        str(args.output_root),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--nproc-per-node",
        str(args.nproc_per_node),
    ]
    if args.stop_on_failure:
        command.append("--stop-on-failure")
    if args.smoke:
        command.append("--smoke")
    if args.keep_raw_checkpoints:
        command.append("--keep-raw-checkpoints")
    if args.auto_tune_batch:
        command.append("--auto-tune-batch")
    if args.tune_only:
        command.append("--tune-only")
    command.extend(["--batch-tune-target-memory-fraction", str(args.batch_tune_target_memory_fraction)])
    command.extend(["--batch-tune-max-tokens", str(args.batch_tune_max_tokens)])
    command.extend(["--batch-tune-timeout-seconds", str(args.batch_tune_timeout_seconds)])
    return command


def write_export_runbook(
    *,
    candidate_file: Path,
    experiments: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    export_dir = candidate_file.parent
    command = build_batch_command(args, candidate_file)
    shell_command = " ".join(shlex.quote(part) for part in command)
    run_command_path = export_dir / "run_command.sh"
    run_command_path.write_text(f"#!/usr/bin/env bash\nset -euo pipefail\n{shell_command}\n", encoding="utf-8")
    run_command_path.chmod(run_command_path.stat().st_mode | 0o111)
    (export_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema": "parameter-golf-gpu-queue-export/v1",
                "queuePath": str(args.queue),
                "candidateFile": str(candidate_file),
                "selectedExperimentIds": [str(experiment["id"]) for experiment in experiments],
                "shard": {"index": args.shard_index, "count": args.shard_count},
                "batchCommand": command,
                "shellCommand": shell_command,
                "expectedEvidenceArtifact": str(args.output_root / "<run-slug>" / "model_factory_evidence.json"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def materialize_all_shard_runbooks(args: argparse.Namespace, queue: dict[str, Any]) -> Path:
    if args.shard_count < 1:
        raise ValueError("shard_count must be at least 1")
    if args.shard_count == 1:
        raise ValueError("--export-all-shards requires --shard-count greater than 1")
    if not args.export_only:
        raise ValueError("--export-all-shards must be combined with --export-only")

    statuses = set(args.status or RUNNABLE_STATUSES)
    lanes = set(args.lane or [])
    only = set(args.only or [])
    export_dir = args.export_root / f"{utc_slug()}_all_shards"
    export_dir.mkdir(parents=True, exist_ok=False)
    shard_manifests: list[dict[str, Any]] = []

    for shard_index in range(args.shard_count):
        experiments = select_experiments(
            queue,
            statuses=statuses,
            lanes=lanes,
            only=only,
            max_experiments=args.max_experiments,
            shard_count=args.shard_count,
            shard_index=shard_index,
        )
        shard_dir = export_dir / f"shard_{shard_index:02d}_of_{args.shard_count:02d}"
        if experiments:
            shard_args = argparse.Namespace(**vars(args))
            shard_args.shard_index = shard_index
            candidate_file = materialize_candidate_file(
                experiments,
                queue_path=args.queue,
                export_root=args.export_root,
                export_dir=shard_dir,
            )
            write_export_runbook(candidate_file=candidate_file, experiments=experiments, args=shard_args)
            shard_manifests.append(
                {
                    "shardIndex": shard_index,
                    "experimentCount": len(experiments),
                    "selectedExperimentIds": [str(experiment["id"]) for experiment in experiments],
                    "candidateFile": str(candidate_file),
                    "runManifest": str(candidate_file.parent / "run_manifest.json"),
                    "runCommand": str(candidate_file.parent / "run_command.sh"),
                }
            )
        else:
            shard_dir.mkdir(parents=True, exist_ok=False)
            (shard_dir / "EMPTY_SHARD.txt").write_text("No experiments selected for this shard.\n", encoding="utf-8")
            shard_manifests.append(
                {
                    "shardIndex": shard_index,
                    "experimentCount": 0,
                    "selectedExperimentIds": [],
                    "emptyShardMarker": str(shard_dir / "EMPTY_SHARD.txt"),
                }
            )

    (export_dir / "all_shards_manifest.json").write_text(
        json.dumps(
            {
                "schema": "parameter-golf-gpu-queue-all-shards-export/v1",
                "queuePath": str(args.queue),
                "shardCount": args.shard_count,
                "statusFilter": sorted(statuses),
                "laneFilter": sorted(lanes),
                "onlyFilter": sorted(only),
                "maxExperimentsPerShard": args.max_experiments,
                "shards": shard_manifests,
                "operatorInstruction": "Copy or mount the whole export directory on GPU pods; each pod runs its shard_N/run_command.sh.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return export_dir


def run_queue(args: argparse.Namespace) -> int:
    queue = load_queue(args.queue)
    if args.print_inventory:
        print(json.dumps({"inventory": inventory_by_lane_status(queue)}, indent=2))
    if args.export_all_shards:
        export_dir = materialize_all_shard_runbooks(args, queue)
        print(f"all_shards_export={export_dir}")
        print(f"all_shards_manifest={export_dir / 'all_shards_manifest.json'}")
        return 0
    statuses = set(args.status or RUNNABLE_STATUSES)
    lanes = set(args.lane or [])
    only = set(args.only or [])
    experiments = select_experiments(
        queue,
        statuses=statuses,
        lanes=lanes,
        only=only,
        max_experiments=args.max_experiments,
        shard_count=args.shard_count,
        shard_index=args.shard_index,
    )
    if not experiments:
        raise ValueError("no GPU experiments selected")
    print(f"selected {len(experiments)} experiment(s)")
    if args.dry_run:
        for experiment in experiments:
            print(
                f"{experiment['id']}: {experiment.get('lane')} {experiment.get('status')} "
                f"priority={experiment.get('priority')} shard={args.shard_index}/{args.shard_count}"
            )
        return 0
    candidate_file = materialize_candidate_file(
        experiments,
        queue_path=args.queue,
        export_root=args.export_root,
    )
    write_export_runbook(candidate_file=candidate_file, experiments=experiments, args=args)
    print(f"candidate_file={candidate_file}")
    if args.export_only:
        print(f"run_manifest={candidate_file.parent / 'run_manifest.json'}")
        print(f"run_command={candidate_file.parent / 'run_command.sh'}")
        return 0
    batch_args = SimpleNamespace(
        candidate_file=candidate_file,
        output_root=args.output_root,
        timeout_seconds=args.timeout_seconds,
        nproc_per_node=args.nproc_per_node,
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
    parser.add_argument("--shard-count", type=int, default=1, help="Split selected experiments into N independent GPU-pod shards.")
    parser.add_argument("--shard-index", type=int, default=0, help="Run the zero-based shard index for this pod.")
    parser.add_argument(
        "--export-all-shards",
        action="store_true",
        help="With --export-only, materialize one runbook per shard from the same queue snapshot.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--nproc-per-node", type=int, default=1, help="torchrun processes per node for default candidate commands.")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--export-only", action="store_true", help="Materialize shard candidates and runbook without starting training.")
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
