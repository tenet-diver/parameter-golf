#!/usr/bin/env python3
"""Append a structured experiment result to the local learning ledger."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LOG_PATH = ROOT / "fastest" / "logs" / "experiment-log.jsonl"
LOG_PATH_ENV = "FASTEST_EXPERIMENT_LOG_PATH"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--status", required=True, choices=["planned", "running", "passed", "failed", "inconclusive", "rejected"])
    parser.add_argument("--summary", required=True)
    parser.add_argument("--idea", action="append", default=[])
    parser.add_argument("--metric", action="append", default=[], help="name=value")
    parser.add_argument("--classification", default="unknown", choices=["leaderboard-legal", "non-record-only", "unknown"])
    parser.add_argument("--next-step", action="append", default=[])
    return parser.parse_args()


def parse_metrics(raw_metrics: list[str]) -> dict[str, str]:
    metrics: dict[str, str] = {}
    for item in raw_metrics:
        if "=" not in item:
            raise SystemExit(f'Invalid --metric value "{item}": expected name=value')
        name, value = item.split("=", 1)
        metrics[name.strip()] = value.strip()
    return metrics


def resolve_log_path() -> Path:
    override = os.environ.get(LOG_PATH_ENV)
    if override:
        return Path(override)
    return LOG_PATH


def main() -> None:
    args = parse_args()
    log_path = resolve_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "title": args.title,
        "category": args.category,
        "status": args.status,
        "summary": args.summary,
        "ideas": args.idea,
        "metrics": parse_metrics(args.metric),
        "classification": args.classification,
        "next_steps": args.next_step,
    }
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
    print(f"appended {log_path}")


if __name__ == "__main__":
    main()
