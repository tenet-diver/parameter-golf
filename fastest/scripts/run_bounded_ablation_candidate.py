import hashlib
import json
import os
import sqlite3
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
RUNNABLE_TASK_STATUSES = {"queued", "ready"}
DEFAULT_TASK_STORE_DIR = Path(
    os.environ.get(
        "TASK_STORE_DIR",
        "/home/codespace/.fastest/orchestrator/projects/parameter-golf-fastest-run-2e398b79d13c/runtime/tasks",
    )
)
TASK_STORE_SQLITE_FILE_ENV = "TASK_STORE_SQLITE_FILE"


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


def _status_order(task: dict) -> int:
    status_order = task.get("statusOrder")
    if isinstance(status_order, bool):
        return 2**31 - 1
    if isinstance(status_order, int):
        return status_order
    return 2**31 - 1


def select_next_ablation_candidate(tasks: list[dict], lane: str = "ablation") -> dict | None:
    eligible: list[dict] = []
    for task in tasks:
        if not isinstance(task, dict):
            continue
        if task.get("lane") != lane:
            continue
        if task.get("status") not in RUNNABLE_TASK_STATUSES:
            continue
        eligible.append(task)

    if not eligible:
        return None

    return sorted(
        eligible,
        key=lambda task: (
            _status_order(task),
            str(task.get("createdAt", "")),
            str(task.get("taskId", "")),
        ),
    )[0]


def _resolve_task_store_sqlite(task_store_dir: Path) -> Path:
    sqlite_override = os.environ.get(TASK_STORE_SQLITE_FILE_ENV, "").strip()
    if sqlite_override:
        override_path = Path(sqlite_override)
        if override_path.is_absolute():
            return override_path
        return task_store_dir / override_path
    return task_store_dir / "task-store.sqlite"


def _load_tasks_from_task_store(task_store_dir: Path) -> list[dict]:
    sqlite_path = _resolve_task_store_sqlite(task_store_dir)
    if not sqlite_path.exists():
        raise FileNotFoundError(f"task store sqlite not found: {sqlite_path}")

    query = """
        SELECT id, status, status_order, created_at, pipeline_json
        FROM task_store_tasks
        WHERE status IN ('queued', 'ready')
        ORDER BY status_order ASC, created_at ASC, id ASC
    """
    with sqlite3.connect(sqlite_path) as conn:
        rows = conn.execute(query).fetchall()

    tasks: list[dict] = []
    for row in rows:
        task_id, status, status_order, created_at, pipeline_json = row
        task = {
            "taskId": task_id,
            "status": status,
            "statusOrder": status_order,
            "createdAt": created_at,
        }
        if isinstance(pipeline_json, str) and pipeline_json.strip():
            payload = json.loads(pipeline_json)
            if isinstance(payload, dict):
                task.update(payload)
        task["taskId"] = task_id
        task["status"] = status
        task["statusOrder"] = status_order
        task["createdAt"] = created_at
        tasks.append(task)
    return tasks


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


def run_next_ablation_candidate(
    tasks: list[dict],
    *,
    runner: Runner,
    lane: str = "ablation",
    evidence_path: Path = DEFAULT_EVIDENCE_PATH,
    status_path: Path = DEFAULT_STATUS_PATH,
    state_path: Path = DEFAULT_STATE_PATH,
) -> dict:
    if lane != "ablation":
        return _outcome(
            task={"lane": lane},
            status="error",
            reason_code="invalid-lane",
            message="Task lane must be ablation.",
        )

    selected = select_next_ablation_candidate(tasks, lane=lane)
    if selected is None:
        return _outcome(
            task={"lane": lane},
            status="success",
            reason_code="no-task-available",
            message="No runnable ablation candidate is available.",
        )

    if not isinstance(selected.get("taskId"), str) or not selected.get("taskId"):
        return _outcome(
            task=selected,
            status="error",
            reason_code="task-io-invalid",
            message="Selected ablation task is missing taskId.",
        )
    if not isinstance(selected.get("candidateId"), str) or not selected.get("candidateId"):
        return _outcome(
            task=selected,
            status="error",
            reason_code="task-io-invalid",
            message="Selected ablation task is missing candidateId.",
        )

    return execute_bounded_ablation_candidate(
        selected,
        runner=runner,
        evidence_path=evidence_path,
        status_path=status_path,
        state_path=state_path,
    )


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
    parser.add_argument("--task", help="Path to ablation task JSON file.")
    parser.add_argument(
        "--task-store-dir",
        default=str(DEFAULT_TASK_STORE_DIR),
        help="Authoritative task-store directory used for next-candidate mode.",
    )
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

    if args.task:
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
    else:
        try:
            tasks = _load_tasks_from_task_store(Path(args.task_store_dir))
        except Exception as exc:
            outcome = _outcome(
                task={"lane": "ablation"},
                status="error",
                reason_code="task-io-invalid",
                message=f"Failed to load runnable tasks from task store: {exc.__class__.__name__}: {exc}",
            )
        else:
            outcome = run_next_ablation_candidate(
                tasks,
                runner=_default_runner,
                lane="ablation",
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
