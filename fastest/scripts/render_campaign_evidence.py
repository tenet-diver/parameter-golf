import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = REPO_ROOT / "fastest/source/measurement_evidence.json"
STATUS_PATH = REPO_ROOT / "fastest/generated/campaign_status.json"
STATE_PATH = REPO_ROOT / "fastest/generated/campaign_state.json"


BASELINE_MISSING_BLOCKER = "No verified benchmark evidence exists yet."
DEFAULT_POLICY_VERSION = "pg-constraint-clarity-v1"
DEFAULT_REQUIRED_EVIDENCE = [
    "trusted-baseline-evidence",
    "benchmark-measurement-evidence",
    "reproducibility-manifest",
]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(f"{json.dumps(payload, indent=2)}\n")


def _normalized_string_list(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    normalized = {value for value in values if isinstance(value, str) and value}
    return sorted(normalized)


def _normalized_non_negative_int(value: object, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int) and value >= 0:
        return value
    return default


def adjudicate_promotion(candidate: dict) -> dict:
    policy_version = candidate.get("policyVersion", DEFAULT_POLICY_VERSION)
    required_evidence = _normalized_string_list(candidate.get("requiredEvidence"))
    if not required_evidence:
        required_evidence = list(DEFAULT_REQUIRED_EVIDENCE)

    provided_evidence = set(_normalized_string_list(candidate.get("providedEvidence")))
    missing_evidence = [artifact for artifact in required_evidence if artifact not in provided_evidence]

    legality_signals = candidate.get("legalitySignals", {})
    legality_status = legality_signals.get("status")
    violations = _normalized_string_list(legality_signals.get("violations"))
    conflicts = _normalized_string_list(legality_signals.get("conflicts"))

    violated_rules = [f"{policy_version}:{rule}" for rule in violations]
    ambiguity_reasons = [f"{policy_version}:ambiguity:{reason}" for reason in conflicts]

    if legality_status == "ambiguous":
        ambiguity_reasons.append(f"{policy_version}:ambiguity:status-ambiguous")
    if legality_status == "illegal" and not violated_rules:
        violated_rules.append(f"{policy_version}:illegal-status-without-rule-code")
    if not candidate.get("promotionGatesSatisfied", False):
        violated_rules.append(f"{policy_version}:promotion-gates-not-satisfied")

    violated_rules = sorted(set(violated_rules))
    ambiguity_reasons = sorted(set(ambiguity_reasons))

    decision = "promote"
    if violated_rules:
        decision = "reject"
    elif missing_evidence or ambiguity_reasons:
        decision = "hold"

    return {
        "candidateId": candidate.get("candidateId"),
        "policyVersion": policy_version,
        "decision": decision,
        "violatedRules": violated_rules,
        "missingEvidence": missing_evidence,
        "ambiguityReasons": ambiguity_reasons,
        "promotableStateChange": decision == "promote",
    }


def apply_measurement_evidence(source: dict, status: dict, state: dict) -> tuple[dict, dict]:
    summary = source["summary"]
    recent_completed = source["recentCompletedExperiments"]
    promotion_policy = source.get("promotionPolicy", {})

    status["trustedControlState"] = summary["trustedControlState"]
    status["lastProgressAt"] = source.get("updatedAt", status.get("lastProgressAt"))
    status["progress"]["totalExperiments"] = summary["totalExperiments"]
    status["progress"]["acceptedExperiments"] = summary["acceptedExperiments"]
    status["progress"]["mostRecentEvidenceId"] = summary["mostRecentEvidenceId"]
    status["recentCompletedExperiments"] = recent_completed

    blockers = [
        blocker
        for blocker in status.get("blockers", [])
        if blocker != BASELINE_MISSING_BLOCKER
    ]
    if summary["trustedControlState"] == "missing":
        blockers.append(BASELINE_MISSING_BLOCKER)
    status["blockers"] = blockers

    adapter_details = status.setdefault("adapterStatus", {}).setdefault("details", {})
    adapter_details["trustedControlState"] = summary["trustedControlState"]
    adapter_details["readiness"] = (
        "baseline-established"
        if summary["trustedControlState"] == "established"
        else "baseline-missing"
    )
    if summary["trustedControlState"] == "established":
        status["adapterStatus"]["summary"] = (
            "Trusted control baseline established; keep Parameter Golf work centered "
            "on measurement and cheap-screen validation."
        )
    else:
        status["adapterStatus"]["summary"] = (
            "Trusted control baseline missing; keep Parameter Golf work centered on "
            "measurement and cheap-screen validation."
        )

    state["lastProgressAt"] = source.get("updatedAt", state.get("lastProgressAt"))
    evidence_cache = state["evidenceSummaryCache"]
    evidence_cache["artifactIds"] = summary["artifactIds"]
    evidence_cache["trustedControlState"] = summary["trustedControlState"]
    evidence_cache["recentCompletedExperiments"] = recent_completed
    evidence_cache["acceptedExperiments"] = summary["acceptedExperiments"]
    evidence_cache["totalExperiments"] = summary["totalExperiments"]

    provided_evidence = _normalized_string_list(summary.get("artifactIds"))
    if summary.get("trustedControlState") in {"verified", "established"}:
        provided_evidence.append("trusted-baseline-evidence")
    if summary.get("mostRecentEvidenceId"):
        provided_evidence.append("benchmark-measurement-evidence")

    policy_version = promotion_policy.get("policyVersion", DEFAULT_POLICY_VERSION)
    required_evidence = _normalized_string_list(promotion_policy.get("requiredEvidence"))
    if not required_evidence:
        required_evidence = list(DEFAULT_REQUIRED_EVIDENCE)

    legality_signals = promotion_policy.get("legalitySignals")
    if not isinstance(legality_signals, dict):
        legality_signals = {
            "status": "legal" if summary.get("trustedControlState") in {"verified", "established"} else "ambiguous",
            "violations": [],
            "conflicts": [],
        }

    minimum_accepted_experiments = _normalized_non_negative_int(
        promotion_policy.get("minimumAcceptedExperiments"),
        1,
    )

    promotion_candidate = {
        "candidateId": summary.get("mostRecentEvidenceId"),
        "policyVersion": policy_version,
        "artifactBudgetPolicy": state.get("campaign", {}).get("metadata", {}).get("artifactBudgetPolicy"),
        "requiredEvidence": required_evidence,
        "providedEvidence": provided_evidence,
        "legalitySignals": legality_signals,
        "promotionGatesSatisfied": summary.get("acceptedExperiments", 0) >= minimum_accepted_experiments,
    }
    adjudication = adjudicate_promotion(promotion_candidate)
    status["promotionAdjudication"] = adjudication
    evidence_cache["latestPromotionAdjudication"] = adjudication

    return status, state


def _check_projection_is_fresh(source: dict, status: dict, state: dict) -> bool:
    expected_status, expected_state = apply_measurement_evidence(
        source,
        json.loads(json.dumps(status)),
        json.loads(json.dumps(state)),
    )
    return expected_status == status and expected_state == state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render campaign views from authoritative evidence.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero when generated views are stale instead of rewriting them.",
    )
    args = parser.parse_args(argv)

    source = _load_json(SOURCE_PATH)
    status = _load_json(STATUS_PATH)
    state = _load_json(STATE_PATH)

    if args.check:
        if _check_projection_is_fresh(source, status, state):
            return 0
        print("Generated campaign views are stale; run controller tick to refresh projections.")
        return 1

    rendered_status, rendered_state = apply_measurement_evidence(source, status, state)
    _write_json(STATUS_PATH, rendered_status)
    _write_json(STATE_PATH, rendered_state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
