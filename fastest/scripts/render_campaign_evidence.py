import argparse
import json
from datetime import datetime, timezone
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
REQUIRED_FAST31_ATTEMPT_FIELDS = (
    "candidateId",
    "commands",
    "artifactPaths",
    "observedMetricOutput",
    "promotionDecisionRationale",
    "blockedReasonCode",
    "nextRecoveryTask",
    "reproducibilityEvidence",
)


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


def _parse_utc_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _is_populated_str_list(value: object) -> bool:
    if not isinstance(value, list) or not value:
        return False
    return all(isinstance(item, str) and item.strip() for item in value)


def _latest_fast31_measurement_attempt(source: dict) -> dict | None:
    attempts = source.get("executionAttempts")
    if not isinstance(attempts, list):
        return None

    selected: dict | None = None
    selected_time = datetime.min.replace(tzinfo=timezone.utc)
    for attempt in attempts:
        if not isinstance(attempt, dict):
            continue
        if attempt.get("taskId") != "FAST-31":
            continue
        if attempt.get("lane") != "measurement":
            continue
        stamp = (
            _parse_utc_timestamp(attempt.get("attemptedAt"))
            or _parse_utc_timestamp(attempt.get("completedAt"))
            or datetime.min.replace(tzinfo=timezone.utc)
        )
        if selected is None or stamp >= selected_time:
            selected = attempt
            selected_time = stamp
    return selected


def _attempt_missing_required_fields(attempt: dict) -> list[str]:
    missing_fields: list[str] = []
    for field in REQUIRED_FAST31_ATTEMPT_FIELDS:
        if field not in attempt:
            missing_fields.append(field)

    if "commands" in attempt and not _is_populated_str_list(attempt.get("commands")):
        missing_fields.append("commands")
    if "artifactPaths" in attempt and not _is_populated_str_list(attempt.get("artifactPaths")):
        missing_fields.append("artifactPaths")

    observed_metric = attempt.get("observedMetricOutput")
    if "observedMetricOutput" in attempt:
        if not isinstance(observed_metric, dict):
            missing_fields.append("observedMetricOutput")
        elif observed_metric.get("status") not in {"observed", "partial", "blocked"}:
            missing_fields.append("observedMetricOutput.status")

    rationale = attempt.get("promotionDecisionRationale")
    if "promotionDecisionRationale" in attempt:
        if not isinstance(rationale, dict):
            missing_fields.append("promotionDecisionRationale")
        else:
            if rationale.get("decision") not in {"promote", "hold", "reject"}:
                missing_fields.append("promotionDecisionRationale.decision")
            if rationale.get("advanceOutcome") not in {"advance", "retry", "reject"}:
                missing_fields.append("promotionDecisionRationale.advanceOutcome")

    reproducibility = attempt.get("reproducibilityEvidence")
    if "reproducibilityEvidence" in attempt:
        if not isinstance(reproducibility, dict):
            missing_fields.append("reproducibilityEvidence")
        elif reproducibility.get("type") not in {"manifest", "waiver"}:
            missing_fields.append("reproducibilityEvidence.type")

    return sorted(set(missing_fields))


def _reproducibility_requirement_satisfied(attempt: dict, now: datetime) -> bool:
    reproducibility = attempt.get("reproducibilityEvidence")
    if not isinstance(reproducibility, dict):
        return False

    evidence_type = reproducibility.get("type")
    if evidence_type == "manifest":
        if reproducibility.get("status") != "provided":
            return False
        if not isinstance(reproducibility.get("artifactPath"), str) or not reproducibility.get("artifactPath"):
            return False
        candidate_id = attempt.get("candidateId")
        manifest_candidate_id = reproducibility.get("candidateId")
        if (
            isinstance(candidate_id, str)
            and candidate_id
            and candidate_id != "none"
            and isinstance(manifest_candidate_id, str)
            and manifest_candidate_id
            and manifest_candidate_id != candidate_id
        ):
            return False
        return True

    if evidence_type == "waiver":
        for field in ("waiverId", "approver", "approvedAt", "expiresAt"):
            value = reproducibility.get(field)
            if not isinstance(value, str) or not value:
                return False
        expires_at = _parse_utc_timestamp(reproducibility.get("expiresAt"))
        if expires_at is None:
            return False
        return expires_at > now

    return False


def _apply_fast31_attempt_adjudication(
    source: dict,
    adjudication: dict,
    required_evidence: list[str],
) -> dict:
    attempt = _latest_fast31_measurement_attempt(source)
    if attempt is None:
        return adjudication

    now = _parse_utc_timestamp(source.get("updatedAt")) or datetime.now(timezone.utc)
    missing_fields = _attempt_missing_required_fields(attempt)
    attempt_candidate_id = attempt.get("candidateId")
    if (
        isinstance(attempt_candidate_id, str)
        and attempt_candidate_id == "none"
        and attempt.get("blockedReasonCode") == "no-task-available"
    ):
        return adjudication

    reproducibility_ok = _reproducibility_requirement_satisfied(attempt, now)

    rationale = attempt.get("promotionDecisionRationale")
    if not isinstance(rationale, dict):
        rationale = {}

    missing_evidence = _normalized_string_list(rationale.get("missingEvidence"))
    if "reproducibility-manifest" in required_evidence and not reproducibility_ok:
        if "reproducibility-manifest" not in missing_evidence:
            missing_evidence.append("reproducibility-manifest")
    missing_evidence = sorted(set(missing_evidence))

    violated_rules = list(adjudication.get("violatedRules", []))
    if missing_fields:
        violated_rules.append(
            f"{adjudication.get('policyVersion', DEFAULT_POLICY_VERSION)}:execution-attempt-packet-invalid"
        )
    if attempt.get("blockedReasonCode"):
        violated_rules.append(
            f"{adjudication.get('policyVersion', DEFAULT_POLICY_VERSION)}:execution-attempt-blocked"
        )
    violated_rules = sorted(set(violated_rules))

    decision = rationale.get("decision")
    if decision not in {"promote", "hold", "reject"}:
        decision = "hold"
    if violated_rules:
        decision = "reject"
    elif missing_evidence and decision == "promote":
        decision = "hold"

    result = dict(adjudication)
    result["candidateId"] = attempt_candidate_id if attempt_candidate_id is not None else result.get("candidateId")
    result["decision"] = decision
    result["violatedRules"] = violated_rules
    result["missingEvidence"] = missing_evidence
    result["promotableStateChange"] = decision == "promote" and not missing_evidence and not violated_rules
    return result


def _promotion_candidate_id(source: dict, summary: dict) -> str | None:
    manifest = source.get("reproducibilityManifest")
    if not isinstance(manifest, dict):
        return summary.get("mostRecentEvidenceId")

    manifest_candidate_id = manifest.get("candidateId")
    if not isinstance(manifest_candidate_id, str) or not manifest_candidate_id:
        return summary.get("mostRecentEvidenceId")

    records = source.get("experimentRecords")
    if not isinstance(records, list):
        return summary.get("mostRecentEvidenceId")

    for record in records:
        if not isinstance(record, dict):
            continue
        if record.get("experimentId") == manifest_candidate_id:
            evidence_id = record.get("evidenceId")
            if isinstance(evidence_id, str) and evidence_id:
                return evidence_id
            return manifest_candidate_id

    return summary.get("mostRecentEvidenceId")


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
        "candidateId": _promotion_candidate_id(source, summary),
        "policyVersion": policy_version,
        "artifactBudgetPolicy": state.get("campaign", {}).get("metadata", {}).get("artifactBudgetPolicy"),
        "requiredEvidence": required_evidence,
        "providedEvidence": provided_evidence,
        "legalitySignals": legality_signals,
        "promotionGatesSatisfied": summary.get("acceptedExperiments", 0) >= minimum_accepted_experiments,
    }
    adjudication = adjudicate_promotion(promotion_candidate)
    adjudication = _apply_fast31_attempt_adjudication(source, adjudication, required_evidence)
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
    parser.add_argument("--source-path", type=Path, default=SOURCE_PATH)
    parser.add_argument("--status-path", type=Path, default=STATUS_PATH)
    parser.add_argument("--state-path", type=Path, default=STATE_PATH)
    args = parser.parse_args(argv)

    source = _load_json(args.source_path)
    status = _load_json(args.status_path)
    state = _load_json(args.state_path)

    if args.check:
        if _check_projection_is_fresh(source, status, state):
            return 0
        print("Generated campaign views are stale; run controller tick to refresh projections.")
        return 1

    rendered_status, rendered_state = apply_measurement_evidence(source, status, state)
    _write_json(args.status_path, rendered_status)
    _write_json(args.state_path, rendered_state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
