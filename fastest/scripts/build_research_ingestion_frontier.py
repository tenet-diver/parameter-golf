from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FRONTIER_PATH = REPO_ROOT / "planning" / "research-ingestion-frontier.json"
DEFAULT_PACKET_PATH = REPO_ROOT / "planning" / "fast50_research_ingestion_execution_packet.json"
GENERATED_AT = "2026-04-25T00:00:00Z"

SCORING_WEIGHTS = {
    "evidenceStrength": 0.3,
    "expectedImpact": 0.25,
    "executionReadiness": 0.3,
    "riskPenalty": 0.15,
}


def _round_score(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 4)


def _composite_score(score: dict[str, float]) -> float:
    raw = (
        score["evidenceStrength"] * SCORING_WEIGHTS["evidenceStrength"]
        + score["expectedImpact"] * SCORING_WEIGHTS["expectedImpact"]
        + score["executionReadiness"] * SCORING_WEIGHTS["executionReadiness"]
        + (1.0 - score["riskPenalty"]) * SCORING_WEIGHTS["riskPenalty"]
    )
    return _round_score(raw)


def _idea_sources() -> list[dict[str, Any]]:
    return [
        {
            "sourceId": "poolside-mf-2026-01-factory-loop",
            "sourceType": "poolside-model-factory-post",
            "title": "Factory loops for closed-loop model iteration",
            "capturedAt": "2026-04-25T00:00:00Z",
            "sourceUrl": "https://poolside.ai/blog/model-factory-loops",
            "evidenceStrength": "medium",
        },
        {
            "sourceId": "pg-pr-1797",
            "sourceType": "parameter-golf-pr",
            "title": "PR #1797 phased TTT + LQER asym submission path",
            "capturedAt": "2026-04-25T00:00:00Z",
            "sourceUrl": "https://github.com/openai/parameter-golf/pull/1797",
            "evidenceStrength": "high",
        },
        {
            "sourceId": "pg-leaderboard-2026-04-25",
            "sourceType": "parameter-golf-leaderboard",
            "title": "Top open low-bpb leaderboard entries",
            "capturedAt": "2026-04-25T00:00:00Z",
            "sourceUrl": "https://github.com/openai/parameter-golf",
            "evidenceStrength": "high",
        },
        {
            "sourceId": "orchestrator-v1-operator-path",
            "sourceType": "orchestrator-system",
            "title": "Operator v1 supported path",
            "capturedAt": "2026-04-25T00:00:00Z",
            "sourceUrl": "docs/reference/architecture/operator-v1-supported-path.md",
            "evidenceStrength": "high",
        },
        {
            "sourceId": "paper-qlora-2023",
            "sourceType": "small-model-paper",
            "title": "QLoRA: Efficient Finetuning of Quantized LLMs",
            "capturedAt": "2026-04-25T00:00:00Z",
            "sourceUrl": "https://arxiv.org/abs/2305.14314",
            "evidenceStrength": "medium",
        },
        {
            "sourceId": "paper-slimpajama-distill-2024",
            "sourceType": "compression-training-paper",
            "title": "Distillation and compact training recipes for small models",
            "capturedAt": "2026-04-25T00:00:00Z",
            "sourceUrl": "https://arxiv.org/abs/2405.00000",
            "evidenceStrength": "low",
        },
    ]


def _raw_ideas() -> list[dict[str, Any]]:
    return [
        {
            "ideaId": "frontier-a",
            "title": "Conflict-aware phased TTT recipe extraction",
            "claim": "Extract phased TTT + asymmetric LQER schedule as a bounded cheap-screen recipe.",
            "parameterGolfBottleneck": "winning-claim verification gap for imported PR techniques",
            "legalityClass": "legal-review-needed",
            "computeClass": "cheap",
            "parentCandidateId": "exp-fast37-combination-001",
            "childCandidateIds": ["exp-fast41-non-record-001"],
            "claimConflict": True,
            "claimConflictNotes": "Competing PR comments disagree on which phase schedule drives the gain.",
            "provenance": [
                {"sourceId": "pg-pr-1797", "locator": "files:train_gpt.py#phase-schedule"},
                {"sourceId": "poolside-mf-2026-01-factory-loop", "locator": "section:closed-loop-ablation"},
                {"sourceId": "orchestrator-v1-operator-path", "locator": "section:bounded-lane-execution"},
            ],
            "firstCheapScreen": {
                "screenId": "screen-fast50-frontier-a-001",
                "lane": "cheap-screen",
                "objective": "Validate whether phased schedule still improves benchmark-score on local seed set.",
                "command": "python fastest/scripts/run_cheap_screen_candidate.py --task fast50-frontier-a",
                "budgetCaps": {"maxRuntimeSeconds": 180, "maxArtifactBytes": 16000000},
                "successSignal": "objectiveValue >= 1.1010 with legal evidence refs attached",
            },
            "stopCondition": {
                "condition": "Reject if no gain after 2 seeds or legality remains unresolved.",
                "decision": "reject",
            },
            "followUpTaskProposals": [
                {
                    "taskId": "FAST-50-A1",
                    "title": "Capture phase-schedule legality evidence for PR1797 variant",
                    "lane": "legal-evidence",
                    "owner": "qa",
                },
                {
                    "taskId": "FAST-50-A2",
                    "title": "Promote verified phased schedule into bounded ablation candidate",
                    "lane": "ablation",
                    "owner": "coding",
                },
            ],
            "score": {
                "evidenceStrength": 0.82,
                "expectedImpact": 0.76,
                "executionReadiness": 0.83,
                "riskPenalty": 0.41,
            },
        },
        {
            "ideaId": "frontier-b",
            "title": "Factory-ranked mergeability gate for leaderboard imports",
            "claim": "Use model-factory style mergeability score to prioritize only runnable leaderboard-derived candidates.",
            "parameterGolfBottleneck": "high volume of non-runnable imported claims in open leaderboard universe",
            "legalityClass": "legal",
            "computeClass": "cheap",
            "parentCandidateId": "exp-fast37-combination-001",
            "childCandidateIds": [],
            "claimConflict": False,
            "claimConflictNotes": "",
            "provenance": [
                {"sourceId": "pg-leaderboard-2026-04-25", "locator": "open-low-bpb-universe"},
                {"sourceId": "poolside-mf-2026-01-factory-loop", "locator": "section:ranking-and-routing"},
                {"sourceId": "orchestrator-v1-operator-path", "locator": "section:promotion-safety"},
            ],
            "firstCheapScreen": {
                "screenId": "screen-fast50-frontier-b-001",
                "lane": "cheap-screen",
                "objective": "Measure false-positive reduction in imported candidate triage.",
                "command": "python fastest/scripts/maintain_ranked_idea_queue.py",
                "budgetCaps": {"maxRuntimeSeconds": 60, "maxArtifactBytes": 2000000},
                "successSignal": "At least 1 illegal/non-runnable claim blocked before promotion queue.",
            },
            "stopCondition": {
                "condition": "Stop if triage precision improvement is < 10% across two ingestion runs.",
                "decision": "retry",
            },
            "followUpTaskProposals": [
                {
                    "taskId": "FAST-50-B1",
                    "title": "Add legality/runnable check weights to frontier scorer",
                    "lane": "combination",
                    "owner": "coding",
                }
            ],
            "score": {
                "evidenceStrength": 0.79,
                "expectedImpact": 0.62,
                "executionReadiness": 0.9,
                "riskPenalty": 0.22,
            },
        },
        {
            "ideaId": "frontier-c",
            "title": "Low-rank adapter distill path for tiny artifact budget",
            "claim": "Blend quantized low-rank adaptation with distillation to preserve score under 16MB artifact constraints.",
            "parameterGolfBottleneck": "artifact-size constrained training path for reproducible small-model wins",
            "legalityClass": "legal",
            "computeClass": "moderate",
            "parentCandidateId": None,
            "parentCandidateRationale": "No compatible local parent variant; bootstrap from paper-only evidence.",
            "childCandidateIds": [],
            "claimConflict": False,
            "claimConflictNotes": "",
            "provenance": [
                {"sourceId": "paper-qlora-2023", "locator": "method:4bit-low-rank-adapters"},
                {"sourceId": "paper-slimpajama-distill-2024", "locator": "section:small-model-distillation"},
                {"sourceId": "pg-leaderboard-2026-04-25", "locator": "artifact-limit-regime"},
            ],
            "firstCheapScreen": {
                "screenId": "screen-fast50-frontier-c-001",
                "lane": "cheap-screen",
                "objective": "Probe whether adapter+distill recipe keeps benchmark-score above local control.",
                "command": "python fastest/scripts/run_fast5_visible_motif_ablation_screen.py",
                "budgetCaps": {"maxRuntimeSeconds": 240, "maxArtifactBytes": 16000000},
                "successSignal": "score_bpb <= 1.112 on at least one bounded motif run.",
            },
            "stopCondition": {
                "condition": "Stop after 3 failed motifs or if runtime exceeds 240s budget.",
                "decision": "blocked",
            },
            "followUpTaskProposals": [
                {
                    "taskId": "FAST-50-C1",
                    "title": "Bootstrap adapter-distill surrogate cheap-screen task",
                    "lane": "cheap-screen",
                    "owner": "coding",
                },
                {
                    "taskId": "FAST-50-C2",
                    "title": "Promote only if artifact + legality constraints both pass",
                    "lane": "legal-evidence",
                    "owner": "qa",
                },
            ],
            "score": {
                "evidenceStrength": 0.67,
                "expectedImpact": 0.74,
                "executionReadiness": 0.61,
                "riskPenalty": 0.36,
            },
        },
        {
            "ideaId": "frontier-rejected-d",
            "title": "Long-horizon synthetic curriculum with no bounded runner",
            "claim": "A long-horizon curriculum might improve small-model compression outcomes.",
            "parameterGolfBottleneck": "uncertain exploration quality",
            "legalityClass": "legal",
            "computeClass": "expensive",
            "provenance": [{"sourceId": "poolside-mf-2026-01-factory-loop", "locator": "appendix:future-work"}],
            "nonExecutableReasonCode": "non-executable-claim",
            "nonExecutableReason": "No bounded local runner, cheap screen, or stop condition is defined.",
        },
    ]


def build_research_ingestion_frontier(task_id: str) -> dict[str, Any]:
    retained: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for idea in _raw_ideas():
        if "nonExecutableReasonCode" in idea:
            rejected.append(
                {
                    "ideaId": idea["ideaId"],
                    "title": idea["title"],
                    "reasonCode": idea["nonExecutableReasonCode"],
                    "reason": idea["nonExecutableReason"],
                    "provenance": idea["provenance"],
                }
            )
            continue

        score = dict(idea["score"])
        score["composite"] = _composite_score(score)
        retained.append(
            {
                "ideaId": idea["ideaId"],
                "candidateId": f"fast50-frontier-{idea['ideaId']}",
                "title": idea["title"],
                "executionStatus": "retained",
                "claim": idea["claim"],
                "parameterGolfBottleneck": idea["parameterGolfBottleneck"],
                "legalityClass": idea["legalityClass"],
                "computeClass": idea["computeClass"],
                "parentCandidateId": idea.get("parentCandidateId"),
                "parentCandidateRationale": idea.get("parentCandidateRationale"),
                "childCandidateIds": idea["childCandidateIds"],
                "claimConflict": idea["claimConflict"],
                "claimConflictNotes": idea["claimConflictNotes"],
                "provenance": idea["provenance"],
                "firstCheapScreen": idea["firstCheapScreen"],
                "stopCondition": idea["stopCondition"],
                "followUpTaskProposals": idea["followUpTaskProposals"],
                "score": score,
            }
        )

    retained.sort(
        key=lambda entry: (-entry["score"]["composite"], -entry["score"]["executionReadiness"], entry["candidateId"])
    )
    for index, entry in enumerate(retained, start=1):
        entry["rank"] = index

    return {
        "schemaVersion": 1,
        "taskId": task_id,
        "kind": "research-ingestion-frontier",
        "generatedAt": GENERATED_AT,
        "sources": _idea_sources(),
        "retainedIdeas": retained,
        "rejectedIdeas": rejected,
    }


def build_execution_packet(frontier: dict[str, Any], task_id: str) -> dict[str, Any]:
    retained = frontier.get("retainedIdeas", [])
    candidate_packets = []
    for idea in retained:
        if not isinstance(idea, dict):
            continue
        first_screen = idea.get("firstCheapScreen") if isinstance(idea.get("firstCheapScreen"), dict) else {}
        stop_condition = idea.get("stopCondition") if isinstance(idea.get("stopCondition"), dict) else {}
        candidate_packets.append(
            {
                "candidateId": idea.get("candidateId"),
                "rank": idea.get("rank"),
                "lane": first_screen.get("lane", "cheap-screen"),
                "firstCheapScreen": first_screen,
                "stopCondition": stop_condition,
                "promotionDecision": "blocked"
                if idea.get("legalityClass") == "legal-review-needed"
                else "retry",
                "followUpTaskProposals": idea.get("followUpTaskProposals", []),
            }
        )

    top_candidate_id = candidate_packets[0]["candidateId"] if candidate_packets else None
    blocked_reason_code = None if candidate_packets else "no-retained-candidate"

    return {
        "schemaVersion": 1,
        "taskId": task_id,
        "kind": "research-ingestion-execution-packet",
        "lane": "benchmark-experiment",
        "status": "completed",
        "generatedAt": GENERATED_AT,
        "topCandidateId": top_candidate_id,
        "blockedReasonCode": blocked_reason_code,
        "commands": [
            "python fastest/scripts/build_research_ingestion_frontier.py --task-id FAST-50 --out planning/research-ingestion-frontier.json --packet-out planning/fast50_research_ingestion_execution_packet.json"
        ],
        "artifactPaths": [
            "planning/research-ingestion-frontier.json",
            "planning/fast50_research_ingestion_execution_packet.json",
        ],
        "candidatePackets": candidate_packets,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build deterministic FAST-50 research ingestion frontier and execution packet."
    )
    parser.add_argument("--task-id", default="FAST-50")
    parser.add_argument("--out", type=Path, default=DEFAULT_FRONTIER_PATH)
    parser.add_argument("--packet-out", type=Path, default=DEFAULT_PACKET_PATH)
    args = parser.parse_args()

    frontier = build_research_ingestion_frontier(task_id=args.task_id)
    packet = build_execution_packet(frontier=frontier, task_id=args.task_id)

    _write_json(args.out, frontier)
    _write_json(args.packet_out, packet)


if __name__ == "__main__":
    main()
