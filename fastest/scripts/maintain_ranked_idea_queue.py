from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastest.scripts.submission_legality import classify_submission_legality


EVIDENCE_PATH = REPO_ROOT / "fastest" / "source" / "measurement_evidence.json"
ABLATION_RESULTS_PATH = REPO_ROOT / "records" / "fast5_visible_motif_ablation_screen_results.json"
H100_RESULTS_PATH = REPO_ROOT / "records" / "h100_candidate_batch" / "20260426T191455Z" / "results.json"
OUTPUT_PATH = REPO_ROOT / "planning" / "ranked-idea-queue.json"

ARTIFACT_LIMIT_BYTES = 16_000_000
TARGET_RUNTIME_SECONDS = 600
WEIGHTS = {
    "expectedInfoGain": 0.35,
    "expectedUpside": 0.35,
    "cost": 0.15,
    "mergeability": 0.15,
}
LOCAL_BENCHMARK_BEST = 1.0999


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected object JSON at {path}")
    return payload


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError(f"expected list JSON at {path}")
    return [item for item in payload if isinstance(item, dict)]


def _clamp(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 4)


def _runtime_seconds(record: dict[str, Any]) -> float:
    budget = record.get("budgetCaps") if isinstance(record.get("budgetCaps"), dict) else {}
    value = (
        record.get("runtime_seconds")
        or record.get("elapsedSeconds")
        or budget.get("maxRuntimeSeconds")
        or TARGET_RUNTIME_SECONDS
    )
    return float(value) if isinstance(value, (int, float)) else float(TARGET_RUNTIME_SECONDS)


def _cost_score(record: dict[str, Any]) -> float:
    runtime = _runtime_seconds(record)
    return _clamp(1.0 - (runtime / TARGET_RUNTIME_SECONDS) * 0.65)


def _artifact_score(record: dict[str, Any]) -> float:
    artifact_bytes = (
        record.get("artifactBytes")
        or record.get("artifact_bytes")
        or record.get("int8SubmissionBytes")
        or record.get("quantizedArtifactBytes")
        or record.get("artifactBudgetBytes")
    )
    if not isinstance(artifact_bytes, (int, float)):
        return 0.7
    if artifact_bytes > ARTIFACT_LIMIT_BYTES:
        return 0.0
    remaining_ratio = (ARTIFACT_LIMIT_BYTES - float(artifact_bytes)) / ARTIFACT_LIMIT_BYTES
    return _clamp(0.55 + remaining_ratio * 8.0)


def _mergeability(record: dict[str, Any], source_kind: str) -> tuple[float, str]:
    notes: list[str] = []
    score = 0.75
    if source_kind == "imported-evidence":
        score = 0.32
        notes.append("imported evidence requires runnable local reproduction")
    if source_kind == "local-ablation":
        score = 0.68
        notes.append("local ablation needs implementation extraction before merge")
    if source_kind == "local-experiment":
        score = 0.82
        notes.append("local experiment already has campaign evidence")
    if source_kind == "local-h100-batch":
        score = 0.74
        notes.append("local batch result needs full-duration reproduction before merge")

    reproduction = record.get("reproductionStatus")
    if isinstance(reproduction, dict) and reproduction.get("status"):
        notes.append(str(reproduction["status"]))
        score -= 0.12
    verification = record.get("verificationGate")
    if isinstance(verification, dict) and verification.get("status") == "blocked-not-winning":
        notes.append("verification gate blocked")
        score -= 0.08
    if _artifact_score(record) == 0.0:
        notes.append("artifact exceeds 16MB limit")
        score = min(score, 0.1)
    family = str(record.get("family") or "")
    if source_kind == "local-h100-batch" and "proxy" in family:
        notes.append("proxy family needs native implementation before promotion")
        score -= 0.12
    return _clamp(score), "; ".join(notes)


def _benchmark_upside(record: dict[str, Any]) -> float:
    value = record.get("objectiveValue")
    if not isinstance(value, (int, float)):
        return 0.0
    delta = float(value) - LOCAL_BENCHMARK_BEST
    return _clamp(0.5 + delta / 0.04)


def _val_bpb_upside(record: dict[str, Any]) -> float:
    comparison = record.get("comparison")
    if isinstance(comparison, dict) and isinstance(comparison.get("deltaBpb"), (int, float)):
        return _clamp(0.5 + abs(float(comparison["deltaBpb"])) / 0.005)
    value = record.get("objectiveValue")
    if isinstance(value, (int, float)):
        return _clamp((1.12 - float(value)) / 0.05)
    return 0.0


def _ablation_upside(record: dict[str, Any], baseline_score: float) -> float:
    delta = record.get("score_delta_bpb")
    if isinstance(delta, (int, float)):
        return _clamp(abs(float(delta)) / 0.035)
    score = record.get("score_bpb")
    if isinstance(score, (int, float)):
        return _clamp((baseline_score - float(score)) / 0.035)
    return 0.0


def _h100_batch_upside(record: dict[str, Any], baseline_bpb: float) -> float:
    value = record.get("finalValBpb")
    if not isinstance(value, (int, float)):
        return 0.0
    return _clamp((baseline_bpb - float(value)) / 0.25)


def _info_gain(record: dict[str, Any], source_kind: str) -> float:
    lane = str(record.get("lane") or source_kind)
    base = {
        "cheap-screen": 0.78,
        "ablation": 0.84,
        "combination": 0.7,
        "non-record-exploration": 0.9,
        "submission-record": 0.64,
        "local-ablation": 0.72,
        "local-h100-batch": 0.86,
    }.get(lane, 0.65)
    if source_kind == "imported-evidence":
        base += 0.08
    if record.get("next_action") == "deepen":
        base += 0.04
    return _clamp(base)


def _composite(scores: dict[str, float]) -> float:
    return _clamp(sum(scores[name] * weight for name, weight in WEIGHTS.items()))


def _local_experiment_ideas(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    records = evidence.get("experimentRecords")
    if not isinstance(records, list):
        return []
    ideas = []
    for record in records:
        if not isinstance(record, dict) or record.get("status") != "accepted":
            continue
        lane = record.get("lane")
        if lane not in {"cheap-screen", "ablation", "combination", "non-record-exploration"}:
            continue
        experiment_id = str(record.get("experimentId") or record.get("evidenceId"))
        scores = _score_record(record, "local-experiment", _benchmark_upside(record))
        ideas.append(
            {
                "ideaId": f"local-experiment-{experiment_id}",
                "title": f"Follow up local {lane} result {experiment_id}",
                "sourceKind": "local-experiment",
                "sourceLane": lane,
                "evidenceRefs": [str(record.get("evidenceId") or experiment_id)],
                "scores": scores,
                "submissionClassification": classify_submission_legality(record),
                "mergeabilityNotes": _mergeability(record, "local-experiment")[1],
                "nextPlanningAction": _next_action(record, "local-experiment"),
            }
        )
    return ideas


def _imported_evidence_ideas(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    records = evidence.get("experimentRecords")
    if not isinstance(records, list):
        return []
    ideas = []
    for record in records:
        if not isinstance(record, dict) or record.get("lane") != "submission-record":
            continue
        experiment_id = str(record.get("experimentId") or record.get("evidenceId"))
        scores = _score_record(record, "imported-evidence", _val_bpb_upside(record))
        ideas.append(
            {
                "ideaId": f"imported-evidence-{experiment_id}",
                "title": f"Reproduce imported candidate {experiment_id}",
                "sourceKind": "imported-evidence",
                "sourceLane": "submission-record",
                "evidenceRefs": [str(record.get("evidenceId") or experiment_id)],
                "scores": scores,
                "submissionClassification": classify_submission_legality(record),
                "mergeabilityNotes": _mergeability(record, "imported-evidence")[1],
                "nextPlanningAction": "extract runnable reproduction path before promotion",
            }
        )
    return ideas


def _ablation_ideas(ablation_results: dict[str, Any]) -> list[dict[str, Any]]:
    baseline = ablation_results.get("baseline")
    baseline_score = 1.11473509
    if isinstance(baseline, dict) and isinstance(baseline.get("scoreBpb"), (int, float)):
        baseline_score = float(baseline["scoreBpb"])
    rows = ablation_results.get("rankedRows") or ablation_results.get("rows") or []
    ideas = []
    if not isinstance(rows, list):
        return ideas
    for row in rows:
        if not isinstance(row, dict) or row.get("status") not in {None, "success"}:
            continue
        run_id = str(row.get("run_id") or row.get("motif"))
        scores = _score_record(row, "local-ablation", _ablation_upside(row, baseline_score))
        ideas.append(
            {
                "ideaId": f"local-ablation-{run_id}",
                "title": f"Develop ablation motif: {row.get('motif') or run_id}",
                "sourceKind": "local-ablation",
                "sourceLane": "ablation-screen",
                "evidenceRefs": [str(row.get("variant_config_ref") or run_id)],
                "scores": scores,
                "submissionClassification": classify_submission_legality(row),
                "mergeabilityNotes": _mergeability(row, "local-ablation")[1],
                "nextPlanningAction": str(row.get("next_action") or "plan bounded follow-up"),
            }
        )
    return ideas


def _relative_result_ref(record: dict[str, Any]) -> str:
    run_dir = record.get("runDir")
    if isinstance(run_dir, str) and run_dir:
        path = Path(run_dir) / "result.json"
        try:
            return str(path.resolve().relative_to(REPO_ROOT))
        except ValueError:
            return str(path)
    return f"records/h100_candidate_batch/20260426T191455Z/runs/{record.get('id')}/result.json"


def _h100_batch_baseline(results: list[dict[str, Any]]) -> float:
    for row in results:
        if row.get("hypothesis") == "baseline_control" and isinstance(row.get("finalValBpb"), (int, float)):
            return float(row["finalValBpb"])
    scored = [float(row["finalValBpb"]) for row in results if isinstance(row.get("finalValBpb"), (int, float))]
    return max(scored) if scored else 4.0


def _h100_batch_ideas(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline = _h100_batch_baseline(results)
    ideas = []
    for row in results:
        if row.get("status") != "completed" or row.get("hypothesis") == "baseline_control":
            continue
        if not isinstance(row.get("finalValBpb"), (int, float)):
            continue
        candidate_id = str(row.get("id"))
        scores = _score_record(row, "local-h100-batch", _h100_batch_upside(row, baseline))
        hypothesis = str(row.get("hypothesis") or candidate_id)
        ideas.append(
            {
                "ideaId": f"local-h100-batch-{candidate_id}",
                "title": f"Promote local batch learning: {candidate_id}",
                "sourceKind": "local-h100-batch",
                "sourceLane": "h100-candidate-batch",
                "evidenceRefs": [_relative_result_ref(row)],
                "scores": scores,
                "submissionClassification": classify_submission_legality(row),
                "mergeabilityNotes": _mergeability(row, "local-h100-batch")[1],
                "nextPlanningAction": f"reproduce and scale hypothesis {hypothesis}",
            }
        )
    return ideas


def _score_record(record: dict[str, Any], source_kind: str, expected_upside: float) -> dict[str, float]:
    mergeability, _ = _mergeability(record, source_kind)
    scores = {
        "expectedInfoGain": _info_gain(record, source_kind),
        "expectedUpside": _clamp(expected_upside),
        "cost": _cost_score(record),
        "mergeability": mergeability,
    }
    scores["composite"] = _composite(scores)
    return scores


def _next_action(record: dict[str, Any], source_kind: str) -> str:
    if source_kind == "local-experiment" and record.get("lane") == "combination":
        return "prepare mergeable reproduction packet for strongest combination"
    if source_kind == "local-experiment" and record.get("lane") == "cheap-screen":
        return "screen adjacent cheap variant before expensive promotion"
    if source_kind == "local-experiment" and record.get("lane") == "non-record-exploration":
        return "convert learning into legal bounded candidate if signal persists"
    return "plan bounded follow-up experiment"


def build_ranked_idea_queue(
    *,
    evidence: dict[str, Any] | None = None,
    ablation_results: dict[str, Any] | None = None,
    h100_results: list[dict[str, Any]] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    evidence = evidence if evidence is not None else _load_json(EVIDENCE_PATH)
    ablation_results = ablation_results if ablation_results is not None else _load_json(ABLATION_RESULTS_PATH)
    h100_results = h100_results if h100_results is not None else _load_json_list(H100_RESULTS_PATH)
    generated_at = generated_at or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    ideas = [
        *_local_experiment_ideas(evidence),
        *_imported_evidence_ideas(evidence),
        *_ablation_ideas(ablation_results),
        *_h100_batch_ideas(h100_results),
    ]
    ranked = sorted(
        ideas,
        key=lambda idea: (-idea["scores"]["composite"], -idea["scores"]["expectedInfoGain"], idea["ideaId"]),
    )
    for index, idea in enumerate(ranked, start=1):
        idea["rank"] = index

    return {
        "schemaVersion": 1,
        "kind": "ranked-competition-idea-queue",
        "taskId": "FAST-873",
        "title": "Maintain ranked idea queue scored by info gain, score upside, cost, and mergeability",
        "generatedAt": generated_at,
        "competitionPreset": "parameter-golf",
        "competitionAnchor": {
            "artifactLimitBytes": ARTIFACT_LIMIT_BYTES,
            "scoring": "FineWeb validation bits-per-byte plus local benchmark-score campaign signal",
            "leaderboardConstraint": "credible path toward 10-minute-on-8xH100 execution",
        },
        "sourceInputs": [
            "fastest/source/measurement_evidence.json",
            "records/fast5_visible_motif_ablation_screen_results.json",
            "records/h100_candidate_batch/20260426T191455Z/results.json",
        ],
        "architectureHandoff": {
            "implementationBoundary": "planning artifact builder only; runners continue consuming explicit task packets",
            "changedInterfaces": ["planning/ranked-idea-queue.json"],
            "rolloutConstraints": [
                "Queue is advisory input for the next planning cycle.",
                "Imported candidates remain blocked until local runnable reproduction evidence exists.",
            ],
        },
        "rankingFormula": {
            "componentWeights": WEIGHTS,
            "componentSemantics": {
                "expectedInfoGain": "likelihood a bounded run changes planning knowledge",
                "expectedUpside": "score improvement potential from imported or local evidence",
                "cost": "preference for cheap runtime and budget-constrained validation",
                "mergeability": "readiness to become legal, reproducible, reviewable competition code",
            },
        },
        "durability": {
            "artifactPath": "planning/ranked-idea-queue.json",
            "nextPlanningCycleInput": True,
            "updateCommand": "python fastest/scripts/maintain_ranked_idea_queue.py",
        },
        "rankedIdeas": ranked,
    }


def write_ranked_idea_queue(output_path: Path = OUTPUT_PATH) -> dict[str, Any]:
    queue = build_ranked_idea_queue()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(f"{json.dumps(queue, indent=2)}\n")
    return queue


def main() -> None:
    parser = argparse.ArgumentParser(description="Maintain the durable ranked parameter-golf idea queue.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    queue = write_ranked_idea_queue(args.output)
    print(f"ranked_idea_queue:ideas:{len(queue['rankedIdeas'])} output:{args.output}")


if __name__ == "__main__":
    main()
