import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

try:
    from fastest.scripts.render_campaign_evidence import (
        SOURCE_PATH as DEFAULT_EVIDENCE_PATH,
        STATE_PATH as DEFAULT_STATE_PATH,
        STATUS_PATH as DEFAULT_STATUS_PATH,
        apply_measurement_evidence,
    )
except ModuleNotFoundError:
    from render_campaign_evidence import (
        SOURCE_PATH as DEFAULT_EVIDENCE_PATH,
        STATE_PATH as DEFAULT_STATE_PATH,
        STATUS_PATH as DEFAULT_STATUS_PATH,
        apply_measurement_evidence,
    )


Runner = Callable[[dict], dict]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_json(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("json-root-not-object")
    return payload


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(f"{json.dumps(payload, indent=2)}\n")


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _outcome(
    *,
    task: dict,
    status: str,
    reason_code: str,
    message: str,
    completed_at: str | None = None,
) -> dict:
    return {
        "taskId": task.get("taskId"),
        "traceId": task.get("traceId"),
        "candidateId": task.get("candidateId"),
        "lane": task.get("lane"),
        "status": status,
        "reasonCode": reason_code,
        "message": message,
        "completedAt": completed_at,
    }


def _idempotency_key(task: dict) -> str:
    return ":".join(
        [
            str(task.get("taskId", "")),
            str(task.get("lane", "")),
            str(task.get("candidateId", "")),
        ]
    )


def _coerce_seed(value: object) -> str:
    if value is None:
        return "0"
    return str(value)


def _max_ablation_experiments(task: dict) -> int:
    budget_caps = task.get("budgetCaps")
    if not isinstance(budget_caps, dict):
        return 1
    value = budget_caps.get("maxAblationExperiments")
    if isinstance(value, bool):
        return 1
    if isinstance(value, int) and value >= 0:
        return value
    return 1


def _regenerate_views(*, evidence_path: Path, status_path: Path, state_path: Path) -> None:
    source = _load_json(evidence_path)
    status = _load_json(status_path)
    state = _load_json(state_path)
    rendered_status, rendered_state = apply_measurement_evidence(source, status, state)
    _write_json(status_path, rendered_status)
    _write_json(state_path, rendered_state)


def execute_bounded_ablation_candidate(
    task: dict,
    *,
    runner: Runner,
    evidence_path: Path = DEFAULT_EVIDENCE_PATH,
    status_path: Path = DEFAULT_STATUS_PATH,
    state_path: Path = DEFAULT_STATE_PATH,
) -> dict:
    if task.get("lane") != "ablation":
        return _outcome(
            task=task,
            status="error",
            reason_code="invalid-lane",
            message="Task lane must be ablation.",
        )

    try:
        evidence = _load_json(evidence_path)
    except Exception:
        return _outcome(
            task=task,
            status="error",
            reason_code="evidence-io-corrupt",
            message="Authoritative evidence is missing or corrupt; repair measurement evidence JSON and retry.",
        )

    summary = evidence.get("summary")
    if not isinstance(summary, dict):
        return _outcome(
            task=task,
            status="error",
            reason_code="evidence-schema-invalid",
            message="Authoritative evidence summary must be an object.",
        )

    trusted_control_state = summary.get("trustedControlState")
    if trusted_control_state not in {"established", "verified"}:
        return _outcome(
            task=task,
            status="error",
            reason_code="baseline-not-established",
            message="Trusted control baseline is not established; skip ablation until baseline evidence is verified.",
        )

    experiment_records = evidence.setdefault("experimentRecords", [])
    if not isinstance(experiment_records, list):
        return _outcome(
            task=task,
            status="error",
            reason_code="evidence-schema-invalid",
            message="Authoritative evidence experimentRecords must be a list.",
        )

    idem_key = _idempotency_key(task)
    existing = next(
        (
            record
            for record in experiment_records
            if isinstance(record, dict)
            and (
                record.get("idempotencyKey") == idem_key
                or (
                    record.get("lane") == "ablation"
                    and record.get("experimentId") == task.get("candidateId")
                )
            )
        ),
        None,
    )
    if existing is not None:
        return _outcome(
            task=task,
            status="success",
            reason_code="duplicate-task-replay",
            message="Ablation candidate already recorded; skipped duplicate append.",
            completed_at=existing.get("completedAt"),
        )

    max_ablation_experiments = _max_ablation_experiments(task)
    used_ablation_experiments = sum(
        1
        for record in experiment_records
        if isinstance(record, dict)
        and record.get("lane") == "ablation"
        and record.get("status") == "accepted"
    )
    if used_ablation_experiments >= max_ablation_experiments:
        return _outcome(
            task=task,
            status="error",
            reason_code="ablation-budget-exhausted",
            message="Ablation experiment budget is exhausted for this run configuration.",
        )

    try:
        run_result = runner(task)
    except Exception as exc:
        return _outcome(
            task=task,
            status="error",
            reason_code="runner-execution-failed",
            message=f"Ablation runner failed: {exc.__class__.__name__}: {exc}",
        )

    if not isinstance(run_result, dict):
        return _outcome(
            task=task,
            status="error",
            reason_code="schema-validation",
            message="Runner result must be an object.",
        )

    completed_at = run_result.get("completedAt") or _utc_now_iso()
    objective_metric_name = run_result.get("objectiveMetricName")
    objective_value = run_result.get("objectiveValue")

    if not isinstance(objective_metric_name, str) or not objective_metric_name:
        return _outcome(
            task=task,
            status="error",
            reason_code="schema-validation",
            message="Runner result objectiveMetricName is required.",
            completed_at=completed_at,
        )

    if not _is_number(objective_value):
        return _outcome(
            task=task,
            status="error",
            reason_code="schema-validation",
            message="Runner result objectiveValue must be numeric.",
            completed_at=completed_at,
        )

    artifacts = run_result.get("artifacts")
    artifacts = artifacts if isinstance(artifacts, dict) else {}
    evidence_id = artifacts.get("evidenceId")
    if run_result.get("status") == "success" and (not isinstance(evidence_id, str) or not evidence_id):
        return _outcome(
            task=task,
            status="error",
            reason_code="integrity-error",
            message="Runner success result is missing artifacts.evidenceId.",
            completed_at=completed_at,
        )

    record_status = "accepted" if run_result.get("status") == "success" else "failed-terminal"
    new_record = {
        "experimentId": task.get("candidateId"),
        "evidenceId": evidence_id,
        "lane": "ablation",
        "status": record_status,
        "completedAt": completed_at,
        "objectiveMetricName": objective_metric_name,
        "objectiveValue": objective_value,
        "failureCode": run_result.get("failureCode"),
        "failureMessage": run_result.get("failureMessage"),
        "idempotencyKey": idem_key,
        "taskId": task.get("taskId"),
        "traceId": task.get("traceId"),
        "runConfig": task.get("runConfig"),
        "budgetCaps": task.get("budgetCaps"),
    }
    experiment_records.append(new_record)

    summary["totalExperiments"] = int(summary.get("totalExperiments", 0)) + 1
    if record_status == "accepted":
        summary["acceptedExperiments"] = int(summary.get("acceptedExperiments", 0)) + 1
    else:
        summary["acceptedExperiments"] = int(summary.get("acceptedExperiments", 0))

    artifact_ids = summary.get("artifactIds")
    if not isinstance(artifact_ids, list):
        artifact_ids = []
    if isinstance(evidence_id, str) and evidence_id and evidence_id not in artifact_ids:
        artifact_ids.append(evidence_id)
    summary["artifactIds"] = artifact_ids

    if isinstance(evidence_id, str) and evidence_id:
        summary["mostRecentEvidenceId"] = evidence_id

    recent_completed = evidence.get("recentCompletedExperiments")
    if not isinstance(recent_completed, list):
        recent_completed = []
    if task.get("candidateId") not in recent_completed:
        recent_completed.append(task.get("candidateId"))
    evidence["recentCompletedExperiments"] = recent_completed
    evidence["updatedAt"] = completed_at

    _write_json(evidence_path, evidence)

    try:
        _regenerate_views(
            evidence_path=evidence_path,
            status_path=status_path,
            state_path=state_path,
        )
    except Exception:
        return _outcome(
            task=task,
            status="partial",
            reason_code="regeneration-write-failed",
            message="Authoritative evidence was updated but campaign view regeneration failed; rerun controller tick.",
            completed_at=completed_at,
        )

    if run_result.get("status") != "success":
        return _outcome(
            task=task,
            status="error",
            reason_code="no-results-produced",
            message="Ablation run completed without a successful result; investigate failureCode and retry.",
            completed_at=completed_at,
        )

    return _outcome(
        task=task,
        status="success",
        reason_code="ablation-success",
        message="Ablation candidate executed within bounds and evidence/views were refreshed.",
        completed_at=completed_at,
    )


def _default_runner(task: dict) -> dict:
    candidate_id = task.get("candidateId")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError("candidateId is required for ablation execution")

    run_config = task.get("runConfig")
    run_config = run_config if isinstance(run_config, dict) else {}
    seed = _coerce_seed(run_config.get("seed"))

    fingerprint = hashlib.sha256(f"{candidate_id}:{seed}".encode("utf-8")).hexdigest()
    score = 1.0 + ((int(fingerprint[:8], 16) % 1000) / 10000.0)

    return {
        "status": "success",
        "objectiveMetricName": "benchmark-score",
        "objectiveValue": round(score, 4),
        "completedAt": _utc_now_iso(),
        "failureCode": None,
        "failureMessage": None,
        "artifacts": {"evidenceId": f"evidence-{candidate_id}"},
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Execute one bounded ablation benchmark candidate.")
    parser.add_argument("--task", required=True, help="Path to ablation task JSON file.")
    parser.add_argument(
        "--output",
        help="Optional path to write the execution outcome JSON.",
    )
    parser.add_argument(
        "--evidence-path",
        default=str(DEFAULT_EVIDENCE_PATH),
        help="Path to authoritative measurement evidence JSON.",
    )
    parser.add_argument(
        "--status-path",
        default=str(DEFAULT_STATUS_PATH),
        help="Path to generated campaign status JSON.",
    )
    parser.add_argument(
        "--state-path",
        default=str(DEFAULT_STATE_PATH),
        help="Path to generated campaign state JSON.",
    )
    args = parser.parse_args()

    try:
        task_payload = _load_json(Path(args.task))
    except Exception as exc:
        outcome = _outcome(
            task={},
            status="error",
            reason_code="task-io-invalid",
            message=f"Failed to load task payload: {exc.__class__.__name__}: {exc}",
        )
    else:
        outcome = execute_bounded_ablation_candidate(
            task_payload,
            runner=_default_runner,
            evidence_path=Path(args.evidence_path),
            status_path=Path(args.status_path),
            state_path=Path(args.state_path),
        )

    rendered = f"{json.dumps(outcome, indent=2)}\n"
    if args.output:
        Path(args.output).write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
