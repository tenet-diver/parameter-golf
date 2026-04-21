#!/usr/bin/env python3
"""Mine Parameter Golf records into a local research backlog."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RECORDS_ROOT = ROOT / "records"
GENERATED_ROOT = ROOT / "fastest" / "generated"
LOCAL_EXPERIMENT_LOG_PATH = ROOT / "fastest" / "logs" / "experiment-log.jsonl"
ARTIFACT_LIMIT_BYTES = 16_000_000


KEYWORDS: dict[str, tuple[str, ...]] = {
    "sp8192": ("sp8192", "sentencepiece 8192", "8192-token sentencepiece", "8192bpe"),
    "sp4096": ("sp4096", "4096-vocab", "4096 vocab", "4096-token"),
    "recurrence": ("recurrence", "recur", "loop layers", "depth recurrence"),
    "parallel_residuals": ("parallel residual", "parallel residuals"),
    "legal_ttt": ("legal ttt", "legal score-first ttt"),
    "score_first_ttt": ("score-first ttt", "score first ttt"),
    "qk_gain": ("qk-gain", "qk gain"),
    "sdclip": ("sdclip", "sd-clip", "hessian-aware sdclip"),
    "gptq_embeddings": ("gptq embeddings", "gptq-quantize embeddings"),
    "muoneq_r": ("muoneq-r", "row-normalized muon", "muon eq-r"),
    "ema": (" ema", "ema "),
    "high_wd": ("wd=0.090", "wd 0.095", "high wd", "weight decay"),
    "xsa": ("xsa", "cross-layer", "efficient partial xsa"),
    "mlp3x": ("mlp3x", "3x mlp"),
    "mlp4x": ("mlp 4x", "mlp4x"),
    "qat": ("qat", "quantization-aware"),
    "int6": ("int6", "all-int6"),
    "sliding_window": ("sliding window", "stride=64", "sliding eval"),
    "bigramhash": ("bigramhash", "bigram hash"),
    "partial_rope": ("partial rope",),
    "smeargate": ("smeargate",),
    "ternary": ("ternary", "1 bit", "binary"),
}


@dataclass
class Record:
    path: str
    track: str
    name: str
    date: str
    val_bpb: float | None
    bytes_total: int | None
    train_time_seconds: int | None
    hardware: str | None
    summary: str
    tags: list[str]
    source: dict[str, str]
    payload: dict[str, Any] | None = None


def parse_date(raw: str, fallback: str) -> str:
    candidate = raw or fallback
    if "T" in candidate:
        candidate = candidate.split("T", 1)[0]
    return candidate


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def load_local_experiment_learnings(path: Path | None = None) -> list[dict[str, Any]]:
    target = path or LOCAL_EXPERIMENT_LOG_PATH
    if not target.exists():
        return []
    entries: list[dict[str, Any]] = []
    with target.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                entries.append(payload)
    return entries


def detect_tags(text: str) -> list[str]:
    normalized = f" {text.lower()} "
    tags = [tag for tag, needles in KEYWORDS.items() if any(needle in normalized for needle in needles)]
    return sorted(set(tags))


def load_records() -> list[Record]:
    records: list[Record] = []
    for path in sorted(RECORDS_ROOT.glob("**/submission.json")):
        payload = read_json(path)
        parent_track = path.parent.parent.name
        inferred_track = parent_track[6:] if parent_track.startswith("track_") else "unknown"
        track = str(payload.get("track") or inferred_track)
        name = str(payload.get("name") or payload.get("run_name") or path.parent.name)
        summary_parts = [
            name,
            str(payload.get("blurb") or ""),
            str(payload.get("technique_summary") or ""),
            str(payload.get("architecture") or ""),
        ]
        summary = " ".join(part for part in summary_parts if part).strip()
        date = parse_date(str(payload.get("date") or ""), path.parent.name[:10])
        val_bpb = payload.get("val_bpb")
        if isinstance(val_bpb, str):
            try:
                val_bpb = float(val_bpb)
            except ValueError:
                val_bpb = None
        bytes_total = payload.get("bytes_total") or payload.get("artifact_bytes_mean")
        if isinstance(bytes_total, str):
            try:
                bytes_total = int(bytes_total)
            except ValueError:
                bytes_total = None
        train_time_seconds = payload.get("train_time_seconds")
        if isinstance(train_time_seconds, str):
            try:
                train_time_seconds = int(float(train_time_seconds))
            except ValueError:
                train_time_seconds = None
        elif isinstance(train_time_seconds, float):
            train_time_seconds = int(train_time_seconds)
        elif not isinstance(train_time_seconds, int):
            train_time_seconds = None
        hardware = payload.get("hardware")
        if not isinstance(hardware, str):
            hardware = None
        records.append(
            Record(
                path=str(path.relative_to(ROOT)),
                track=track,
                name=name,
                date=date,
                val_bpb=float(val_bpb) if isinstance(val_bpb, (int, float)) else None,
                bytes_total=int(bytes_total) if isinstance(bytes_total, (int, float)) else None,
                train_time_seconds=train_time_seconds,
                hardware=hardware,
                summary=summary,
                tags=detect_tags(summary),
                source={
                    "kind": "submission_json",
                    "path": str(path.relative_to(ROOT)),
                    "track_dir": str(path.parent.relative_to(ROOT)),
                },
                payload=payload,
            )
        )
    return records


def iso_to_ordinal(value: str) -> int:
    return datetime.strptime(value, "%Y-%m-%d").toordinal()


def is_8xh100_hardware(hardware: str | None) -> bool | None:
    if hardware is None:
        return None
    normalized = re.sub(r"\s+", "", hardware.lower())
    return "8xh100" in normalized


def classify_record_legality(record: Record) -> dict[str, Any]:
    summary = record.summary.lower()
    non_record_reasons: list[str] = []
    uncertainty_reasons: list[str] = []
    is_track_non_record = "non_record" in record.track or "non-record" in record.track

    if is_track_non_record:
        non_record_reasons.append("track_marked_non_record")
    if "unlimited compute" in summary or "not intended to satisfy the 10-minute cutoff" in summary:
        non_record_reasons.append("declared_unlimited_compute")
    if record.bytes_total is not None and record.bytes_total > ARTIFACT_LIMIT_BYTES:
        non_record_reasons.append("artifact_over_16mb")

    if record.track == "10min_16mb":
        if record.val_bpb is None:
            uncertainty_reasons.append("missing_val_bpb")
        if record.bytes_total is None:
            uncertainty_reasons.append("missing_artifact_bytes")
        if record.train_time_seconds is None:
            uncertainty_reasons.append("missing_train_time_seconds")
        elif record.train_time_seconds > 600:
            uncertainty_reasons.append("runtime_over_10min")
        hardware_is_8xh100 = is_8xh100_hardware(record.hardware)
        if record.hardware is None:
            uncertainty_reasons.append("missing_hardware")
        elif hardware_is_8xh100 is False:
            uncertainty_reasons.append("hardware_not_8xh100")
        uncertainty_reasons.extend(validation_uncertainty_reasons(record))
    elif not is_track_non_record and record.track != "non-record":
        uncertainty_reasons.append("unknown_track")

    if non_record_reasons:
        status = "non-record-only"
    elif uncertainty_reasons:
        status = "uncertain"
    else:
        status = "leaderboard-legal"

    return {
        "status": status,
        "rules": {
            "track": record.track,
            "artifact_limit_bytes": ARTIFACT_LIMIT_BYTES,
            "artifact_within_limit": None if record.bytes_total is None else record.bytes_total <= ARTIFACT_LIMIT_BYTES,
            "has_val_bpb": record.val_bpb is not None,
            "has_train_time_seconds": record.train_time_seconds is not None,
            "train_within_10min": None if record.train_time_seconds is None else record.train_time_seconds <= 600,
            "has_hardware": record.hardware is not None,
            "hardware_is_8xh100": is_8xh100_hardware(record.hardware),
        },
        "non_record_reasons": sorted(set(non_record_reasons)),
        "uncertainty_reasons": sorted(set(uncertainty_reasons)),
    }


def coerce_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return None
    return None


def normalize_text(value: Any) -> str | None:
    return value.strip().lower() if isinstance(value, str) and value.strip() else None


def validation_uncertainty_reasons(record: Record) -> list[str]:
    payload = record.payload if isinstance(record.payload, dict) else {}
    validation = payload.get("validation")
    validation_data = validation if isinstance(validation, dict) else None

    reasons: list[str] = []

    if validation_data is not None:
        controls = validation_data.get("trusted_controls")
        if isinstance(controls, list):
            for control in controls:
                if not isinstance(control, dict):
                    continue
                expected = coerce_float(control.get("expected_val_bpb"))
                observed = coerce_float(control.get("observed_val_bpb"))
                max_drift = coerce_float(control.get("max_drift_bpb"))
                if expected is None or observed is None or max_drift is None:
                    continue
                if abs(observed - expected) > max_drift:
                    reasons.append("eval_drift_vs_trusted_control")
                    break

    seed_results = payload.get("seed_results")
    seed_artifact_bytes: list[int] = []
    if isinstance(seed_results, dict):
        for item in seed_results.values():
            if not isinstance(item, dict):
                continue
            artifact_bytes = coerce_int(item.get("artifact_bytes") or item.get("bytes_total"))
            if artifact_bytes is not None:
                seed_artifact_bytes.append(artifact_bytes)

    mismatch_found = False
    if seed_artifact_bytes:
        max_seed_artifact = max(seed_artifact_bytes)
        claimed_max = coerce_int(payload.get("artifact_bytes_max") or record.bytes_total)
        if claimed_max is not None and claimed_max < max_seed_artifact:
            mismatch_found = True
        compliance = payload.get("compliance")
        artifact_under_16mb = compliance.get("artifact_under_16mb") if isinstance(compliance, dict) else None
        if artifact_under_16mb is True and max_seed_artifact > ARTIFACT_LIMIT_BYTES:
            mismatch_found = True

    claimed_compression = normalize_text(payload.get("compression"))
    artifact_probe = validation_data.get("artifact_probe") if validation_data is not None else None
    observed_compression = normalize_text(artifact_probe.get("compression")) if isinstance(artifact_probe, dict) else None
    if claimed_compression and observed_compression and claimed_compression != observed_compression:
        mismatch_found = True

    if mismatch_found:
        reasons.append("artifact_or_compression_mismatch")

    if validation_data is not None:
        repro = validation_data.get("reproducibility")
        if isinstance(repro, dict):
            promising = record.val_bpb is not None and record.val_bpb <= 1.10
            attempts = coerce_int(repro.get("attempts"))
            successful_runs = coerce_int(repro.get("successful_runs"))
            status = normalize_text(repro.get("status"))
            failed_status = status in {"failed", "unreproducible", "not_reproducible"}
            insufficient_success = (
                attempts is not None and successful_runs is not None and attempts >= 2 and successful_runs < 2
            )
            if promising and (failed_status or insufficient_success):
                reasons.append("promising_result_not_reproducible")

    return reasons


def classify_ideas(records: list[Record]) -> dict[str, dict[str, Any]]:
    by_tag: dict[str, list[tuple[Record, dict[str, Any]]]] = defaultdict(list)
    for record in records:
        legality = classify_record_legality(record)
        for tag in record.tags:
            by_tag[tag].append((record, legality))

    result: dict[str, dict[str, Any]] = {}
    for tag, tagged_records in by_tag.items():
        statuses = {legality["status"] for _, legality in tagged_records}
        if "leaderboard-legal" in statuses:
            tag_status = "leaderboard-legal"
        elif "uncertain" in statuses:
            tag_status = "uncertain"
        else:
            tag_status = "non-record-only"
        uncertainty_reasons: set[str] = set()
        for _, legality in tagged_records:
            if legality["status"] == "uncertain":
                uncertainty_reasons.update(legality["uncertainty_reasons"])
        result[tag] = {
            "status": tag_status,
            "evidence_paths": [record.path for record, _ in tagged_records][:5],
            "uncertainty_reasons": sorted(uncertainty_reasons),
        }
    return result


def summarize_local_learning(entries: list[dict[str, Any]]) -> dict[str, Any]:
    per_idea: dict[str, dict[str, int]] = defaultdict(
        lambda: {"attempted": 0, "passed": 0, "failed": 0, "inconclusive": 0, "rejected": 0}
    )
    for entry in entries:
        raw_ideas = entry.get("ideas")
        ideas = [item for item in raw_ideas if isinstance(item, str)] if isinstance(raw_ideas, list) else []
        status = str(entry.get("status") or "").lower()
        for idea in ideas:
            stats = per_idea[idea]
            stats["attempted"] += 1
            if status in {"passed", "failed", "inconclusive", "rejected"}:
                stats[status] += 1

    return {
        "entry_count": len(entries),
        "ideas": dict(sorted(per_idea.items())),
    }


def rank_task(
    task: dict[str, Any],
    *,
    frequency: Counter[str],
    top_count: int,
    local_learning: dict[str, Any],
) -> dict[str, float]:
    ideas = [idea for idea in task.get("ideas", []) if isinstance(idea, str)]
    idea_classification = task.get("idea_classification", {})
    local_ideas = local_learning.get("ideas", {})

    uncertainty_count = 0
    legal_count = 0
    attempt_count = 0
    failure_count = 0
    prevalence = 0.0
    novelty_count = 0

    for idea in ideas:
        classification = idea_classification.get(idea, {}) if isinstance(idea_classification, dict) else {}
        status = classification.get("status")
        if status == "uncertain":
            uncertainty_count += 1
        elif status == "leaderboard-legal":
            legal_count += 1

        stats = local_ideas.get(idea, {}) if isinstance(local_ideas, dict) else {}
        attempted = int(stats.get("attempted", 0))
        failed = int(stats.get("failed", 0))
        attempt_count += attempted
        failure_count += failed
        if attempted == 0:
            novelty_count += 1

        prevalence += frequency.get(idea, 0) / max(top_count, 1)

    category = str(task.get("category") or "")
    category_cost = {
        "knowledge": 0.25,
        "measurement": 0.35,
        "validation": 0.45,
        "baseline": 0.55,
        "experiment": 0.60,
    }.get(category, 0.50)

    expected_info_gain = clamp01(
        0.30
        + 0.20 * (novelty_count / max(len(ideas), 1))
        + 0.20 * (uncertainty_count / max(len(ideas), 1))
        + 0.10 * (1.0 / (1 + attempt_count))
    )
    expected_upside = clamp01(
        0.25 + 0.20 * (prevalence / max(len(ideas), 1)) + 0.20 * (legal_count / max(len(ideas), 1))
    )
    expected_cost = clamp01(
        category_cost + 0.08 * max(len(ideas) - 1, 0) + 0.04 * failure_count - 0.06 * min(attempt_count, 3)
    )
    mergeability = clamp01(
        0.40
        + 0.30 * (legal_count / max(len(ideas), 1))
        - 0.20 * (uncertainty_count / max(len(ideas), 1))
        - 0.05 * max(len(ideas) - 2, 0)
    )
    composite_score = round(
        0.35 * expected_info_gain + 0.35 * expected_upside + 0.30 * mergeability - 0.25 * expected_cost,
        6,
    )
    return {
        "expected_info_gain": round(expected_info_gain, 4),
        "expected_upside": round(expected_upside, 4),
        "expected_cost": round(expected_cost, 4),
        "mergeability": round(mergeability, 4),
        "composite_score": composite_score,
    }


def leaderboard_tasks(records: list[Record], local_learning: dict[str, Any]) -> list[dict[str, Any]]:
    ranked = [record for record in records if record.track == "10min_16mb" and record.val_bpb is not None]
    ranked.sort(key=lambda record: (record.val_bpb, record.date))
    top = ranked[:8]
    idea_legality = classify_ideas(records)

    frequency = Counter(tag for record in top for tag in record.tags)
    tasks: list[dict[str, Any]] = []
    priority = 1

    tasks.append(
        {
            "priority": priority,
            "category": "baseline",
            "title": "Reproduce the current internal baseline and top public control stack",
            "rationale": "Autonomous work needs a trusted control before it can judge any new idea.",
            "evidence": [record.path for record in top[:3]],
            "ideas": ["baseline", "reproducibility"],
        }
    )
    priority += 1

    if {"sp8192", "recurrence", "parallel_residuals", "legal_ttt", "qk_gain"} <= set(frequency):
        tasks.append(
            {
                "priority": priority,
                "category": "experiment",
                "title": "Prepare a local-to-remote experiment plan around the current dominant SP8192 stack",
                "rationale": "Top entries as of 2026-04-09 converge on SP8192 + recurrence + parallel residuals + QK gain + legal score-first TTT.",
                "evidence": [record.path for record in top[:5]],
                "ideas": ["sp8192", "recurrence", "parallel_residuals", "qk_gain", "legal_ttt"],
            }
        )
        priority += 1

    tasks.extend(
        [
            {
                "priority": priority,
                "category": "validation",
                "title": "Build a legality and evaluation checklist for score-first TTT and related test-time adaptation",
                "rationale": "Test-time training appears in the strongest runs but carries rule risk if implemented loosely.",
                "evidence": [record.path for record in top if "ttt" in record.summary.lower()][:5],
                "ideas": ["legal_ttt", "score_first_ttt"],
            },
            {
                "priority": priority + 1,
                "category": "experiment",
                "title": "Ablate QK gain around the public winning range before composing more changes",
                "rationale": "Recent record movement includes QK gain changes from 5.0 to 5.25; that is cheap to isolate compared with new architectures.",
                "evidence": [record.path for record in top if "qk_gain" in record.tags][:5],
                "ideas": ["qk_gain"],
            },
            {
                "priority": priority + 2,
                "category": "experiment",
                "title": "Probe whether Hessian-aware SDClip composes cleanly with the latest public winning stack",
                "rationale": "SDClip variants improved recent runs but are not obviously present in the very latest top entry.",
                "evidence": [record.path for record in ranked if "sdclip" in record.tags][:5],
                "ideas": ["sdclip", "sp8192", "parallel_residuals"],
            },
            {
                "priority": priority + 3,
                "category": "measurement",
                "title": "Create a motif-composition matrix from prior records to avoid naive stack-everything experiments",
                "rationale": "The repo contains enough historical records to learn which ideas compose and which ideas likely interfere.",
                "evidence": [record.path for record in ranked[:12]],
                "ideas": ["analysis", "composition"],
            },
            {
                "priority": priority + 4,
                "category": "knowledge",
                "title": "Log negative or inconclusive local results immediately so the backlog stops rediscovering them",
                "rationale": "The non-record single-GPU exploration path shows that cheap experiments are useful only when they generate durable learning.",
                "evidence": [record.path for record in records if "1x5090" in record.summary.lower()][:3],
                "ideas": ["learning", "non_record"],
            },
        ]
    )
    for task in tasks:
        task["idea_classification"] = {
            idea: idea_legality.get(idea, {"status": "uncertain", "evidence_paths": [], "uncertainty_reasons": ["no_supporting_records"]})
            for idea in task["ideas"]
        }
        task["ranking"] = rank_task(
            task,
            frequency=frequency,
            top_count=len(top),
            local_learning=local_learning,
        )

    tasks.sort(key=lambda task: (-task["ranking"]["composite_score"], task["title"]))
    for index, task in enumerate(tasks, start=1):
        task["priority"] = index
    return tasks


def summarize(records: list[Record]) -> dict[str, Any]:
    leaderboard = [record for record in records if record.track == "10min_16mb" and record.val_bpb is not None]
    leaderboard.sort(key=lambda record: (record.val_bpb, record.date))

    tag_scores: dict[str, list[float]] = defaultdict(list)
    for record in leaderboard:
        for tag in record.tags:
            tag_scores[tag].append(record.val_bpb)  # type: ignore[arg-type]

    motif_summary = []
    for tag, scores in sorted(tag_scores.items(), key=lambda item: (min(item[1]), -len(item[1]))):
        motif_summary.append(
            {
                "tag": tag,
                "count": len(scores),
                "best_val_bpb": min(scores),
                "mean_val_bpb": round(sum(scores) / len(scores), 6),
            }
        )

    latest = max(records, key=lambda record: record.date)
    best = leaderboard[0] if leaderboard else None
    top_scores = [
        {
            "name": record.name,
            "date": record.date,
            "val_bpb": record.val_bpb,
            "path": record.path,
            "tags": record.tags,
            "source": record.source,
        }
        for record in leaderboard[:10]
    ]

    movement_rows: list[dict[str, Any]] = []
    best_so_far: float | None = None
    for record in sorted(leaderboard, key=lambda item: (item.date, item.val_bpb)):
        if best_so_far is None or record.val_bpb < best_so_far:
            delta = None if best_so_far is None else round(best_so_far - record.val_bpb, 6)
            movement_rows.append(
                {
                    "date": record.date,
                    "name": record.name,
                    "val_bpb": record.val_bpb,
                    "delta_vs_previous_best": delta,
                    "path": record.path,
                    "tags": record.tags,
                    "source": record.source,
                }
            )
            best_so_far = record.val_bpb

    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "record_count": len(records),
        "leaderboard_count": len(leaderboard),
        "best_public_record": None
        if best is None
        else {
            "name": best.name,
            "date": best.date,
            "val_bpb": best.val_bpb,
            "path": best.path,
            "tags": best.tags,
            "source": best.source,
        },
        "latest_record_date": latest.date,
        "top_scores": top_scores,
        "motif_summary": motif_summary,
        "recent_movement": movement_rows[-10:],
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def main() -> None:
    records = load_records()
    local_learning_entries = load_local_experiment_learnings()
    local_learning = summarize_local_learning(local_learning_entries)
    summary = summarize(records)
    tasks = leaderboard_tasks(records, local_learning)

    record_index = [
        {
            "path": record.path,
            "track": record.track,
            "name": record.name,
            "date": record.date,
            "val_bpb": record.val_bpb,
            "bytes_total": record.bytes_total,
            "train_time_seconds": record.train_time_seconds,
            "hardware": record.hardware,
            "tags": record.tags,
            "legality": classify_record_legality(record),
            "source": record.source,
        }
        for record in sorted(records, key=lambda record: (record.date, record.path), reverse=True)
    ]

    write_json(GENERATED_ROOT / "record_index.json", {"records": record_index})
    write_json(GENERATED_ROOT / "leaderboard_snapshot.json", summary)
    write_json(
        GENERATED_ROOT / "candidate_backlog.json",
        {
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "inputs": {
                "imported_record_count": len(records),
                "imported_record_source": str(RECORDS_ROOT.relative_to(ROOT)),
                "local_learning": {
                    "entry_count": local_learning["entry_count"],
                    "log_path": str(LOCAL_EXPERIMENT_LOG_PATH.relative_to(ROOT)),
                    "ideas": local_learning["ideas"],
                },
            },
            "tasks": tasks,
        },
    )

    print(f"wrote {GENERATED_ROOT / 'record_index.json'}")
    print(f"wrote {GENERATED_ROOT / 'leaderboard_snapshot.json'}")
    print(f"wrote {GENERATED_ROOT / 'candidate_backlog.json'}")
    if summary["best_public_record"] is not None:
        best = summary["best_public_record"]
        print(f"best_public_record: {best['date']} {best['name']} val_bpb={best['val_bpb']}")
    print(f"candidate_tasks: {len(tasks)}")


if __name__ == "__main__":
    main()
