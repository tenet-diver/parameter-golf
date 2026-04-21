#!/usr/bin/env python3
"""Summarize which public Parameter Golf motifs tend to compose well."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from .mine_records import load_local_experiment_learnings, load_records
except ImportError:  # pragma: no cover - support direct script execution
    from mine_records import load_local_experiment_learnings, load_records


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = ROOT / "fastest" / "generated" / "motif_composition_matrix.json"
TARGET_TRACK = "10min_16mb"
POSITIVE_DELTA_THRESHOLD = -0.001
NEGATIVE_DELTA_THRESHOLD = 0.001


def round_score(value: float) -> float:
    return round(value, 6)


def classify_interaction(delta: float) -> str:
    if delta <= POSITIVE_DELTA_THRESHOLD:
        return "positive"
    if delta >= NEGATIVE_DELTA_THRESHOLD:
        return "negative"
    return "neutral"


def build_single_summary(records: list[Any]) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {}
    buckets: dict[str, list[float]] = defaultdict(list)
    for record in records:
        for tag in record.tags:
            buckets[tag].append(record.val_bpb)
    for tag, scores in buckets.items():
        summary[tag] = {
            "count": len(scores),
            "best_val_bpb": round_score(min(scores)),
            "mean_val_bpb": round_score(sum(scores) / len(scores)),
        }
    return summary


def build_pair_rows(records: list[Any], single_summary: dict[str, dict[str, float | int]]) -> list[dict[str, object]]:
    pair_scores: dict[tuple[str, str], list[float]] = defaultdict(list)
    pair_paths: dict[tuple[str, str], list[str]] = defaultdict(list)
    for record in records:
        tags = sorted(set(record.tags))
        for index, left in enumerate(tags):
            for right in tags[index + 1 :]:
                pair = (left, right)
                pair_scores[pair].append(record.val_bpb)
                pair_paths[pair].append(record.path)

    rows: list[dict[str, object]] = []
    for (left, right), scores in pair_scores.items():
        left_mean = float(single_summary[left]["mean_val_bpb"])
        right_mean = float(single_summary[right]["mean_val_bpb"])
        pair_mean = sum(scores) / len(scores)
        expected_mean = (left_mean + right_mean) / 2.0
        synergy_delta = pair_mean - expected_mean
        rows.append(
            {
                "pair": [left, right],
                "count": len(scores),
                "best_val_bpb": round_score(min(scores)),
                "mean_val_bpb": round_score(pair_mean),
                "synergy_delta_vs_single_mean": round_score(synergy_delta),
                "interaction": classify_interaction(synergy_delta),
                "evidence_paths": pair_paths[(left, right)][:5],
            }
        )

    rows.sort(
        key=lambda item: (
            item["best_val_bpb"],
            item["synergy_delta_vs_single_mean"],
            -int(item["count"]),
            item["pair"],
        )
    )
    return rows


def summarize_interactions(pair_rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    positive = [row for row in pair_rows if row["interaction"] == "positive"]
    neutral = [row for row in pair_rows if row["interaction"] == "neutral"]
    negative = [row for row in pair_rows if row["interaction"] == "negative"]
    return {
        "positive": sorted(positive, key=lambda row: (row["synergy_delta_vs_single_mean"], row["pair"]))[:12],
        "neutral": sorted(neutral, key=lambda row: (-int(row["count"]), row["pair"]))[:12],
        "negative": sorted(negative, key=lambda row: (row["synergy_delta_vs_single_mean"], row["pair"]))[:12],
    }


def build_recommendations(pair_rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    combinations_to_try = [
        row
        for row in pair_rows
        if row["interaction"] == "positive" and int(row["count"]) >= 2
    ]
    combinations_to_avoid = [
        row
        for row in pair_rows
        if row["interaction"] == "negative" and int(row["count"]) >= 2
    ]
    neutral_controls = [row for row in pair_rows if row["interaction"] == "neutral" and int(row["count"]) >= 2]
    return {
        "combinations_to_try": combinations_to_try[:8],
        "combinations_to_avoid": combinations_to_avoid[:8],
        "neutral_controls": neutral_controls[:8],
    }


def summarize_mined_evidence() -> dict[str, object]:
    entries = load_local_experiment_learnings()
    status_counts: dict[str, int] = defaultdict(int)
    idea_counts: dict[str, int] = defaultdict(int)
    for entry in entries:
        status = str(entry.get("status") or "unknown")
        status_counts[status] += 1
        for raw_idea in entry.get("ideas", []):
            idea = str(raw_idea).strip().lower()
            if idea:
                idea_counts[idea] += 1
    top_ideas = [
        {"idea": idea, "mentions": count}
        for idea, count in sorted(idea_counts.items(), key=lambda item: (-item[1], item[0]))[:8]
    ]
    return {
        "entry_count": len(entries),
        "status_counts": dict(sorted(status_counts.items())),
        "top_ideas": top_ideas,
    }


def main() -> None:
    leaderboard = [
        record
        for record in load_records()
        if record.track == TARGET_TRACK and record.val_bpb is not None and record.tags
    ]
    leaderboard.sort(key=lambda record: (record.val_bpb, record.date))
    single_summary = build_single_summary(leaderboard)
    pair_rows = build_pair_rows(leaderboard, single_summary)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "track": TARGET_TRACK,
        "record_count": len(leaderboard),
        "best_public_val_bpb": round_score(min(record.val_bpb for record in leaderboard)),
        "motif_summary": [
            {
                "tag": tag,
                **summary,
            }
            for tag, summary in sorted(
                single_summary.items(),
                key=lambda item: (float(item[1]["best_val_bpb"]), -int(item[1]["count"]), item[0]),
            )
        ],
        "pair_summary": pair_rows,
        "interaction_matrix": summarize_interactions(pair_rows),
        "mined_evidence": summarize_mined_evidence(),
        "recommendations": build_recommendations(pair_rows),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {OUTPUT_PATH.relative_to(ROOT)} with {len(pair_rows)} motif pairs")


if __name__ == "__main__":
    main()
