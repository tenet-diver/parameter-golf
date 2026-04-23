import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = REPO_ROOT / "fastest/source/measurement_evidence.json"
STATUS_PATH = REPO_ROOT / "fastest/generated/campaign_status.json"
STATE_PATH = REPO_ROOT / "fastest/generated/campaign_state.json"


BASELINE_MISSING_BLOCKER = "No verified benchmark evidence exists yet."


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(f"{json.dumps(payload, indent=2)}\n")


def apply_measurement_evidence(source: dict, status: dict, state: dict) -> tuple[dict, dict]:
    summary = source["summary"]
    recent_completed = source["recentCompletedExperiments"]

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

    return status, state


def main() -> None:
    source = _load_json(SOURCE_PATH)
    status = _load_json(STATUS_PATH)
    state = _load_json(STATE_PATH)

    rendered_status, rendered_state = apply_measurement_evidence(source, status, state)
    _write_json(STATUS_PATH, rendered_status)
    _write_json(STATE_PATH, rendered_state)


if __name__ == "__main__":
    main()
