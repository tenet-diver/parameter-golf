from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = REPO_ROOT / "fastest/generated/cpu_subset_runs"
CPU_SUBSET_MAX_WALLCLOCK_SECONDS = 600
CPU_SUBSET_CAP_POLICY_SOURCE = "runner-fixed-fast-51"
FINAL_BPB_RE = re.compile(
    r"final_int8_zlib_roundtrip_exact\s+val_loss:(?P<loss>[0-9.]+)\s+val_bpb:(?P<bpb>[0-9.]+)"
)


def _resolve_shared_repo_root() -> Path:
    for parent in REPO_ROOT.parents:
        if parent.name == "parameter-golf":
            return parent
    return REPO_ROOT


def _resolve_default_data_path() -> Path:
    local = REPO_ROOT / "data/datasets/fineweb10B_sp1024"
    if local.exists():
        return local
    shared = _resolve_shared_repo_root() / "data/datasets/fineweb10B_sp1024"
    if shared.exists():
        return shared
    return local


def _resolve_default_tokenizer_path() -> Path:
    local = REPO_ROOT / "data/tokenizers/fineweb_1024_bpe.model"
    if local.exists():
        return local
    shared = _resolve_shared_repo_root() / "data/tokenizers/fineweb_1024_bpe.model"
    if shared.exists():
        return shared
    return local


DEFAULT_CPU_ENV = {
    "DATA_PATH": str(_resolve_default_data_path()),
    "TOKENIZER_PATH": str(_resolve_default_tokenizer_path()),
    "VOCAB_SIZE": "1024",
    "NUM_LAYERS": "1",
    "MODEL_DIM": "96",
    "NUM_HEADS": "4",
    "NUM_KV_HEADS": "2",
    "MLP_MULT": "2",
    "TRAIN_SEQ_LEN": "128",
    "TRAIN_BATCH_TOKENS": "2048",
    "VAL_BATCH_SIZE": "4096",
    "VAL_TOKEN_LIMIT": "8192",
    "ITERATIONS": "4",
    "WARMUP_STEPS": "0",
    "WARMDOWN_ITERS": "0",
    "VAL_LOSS_EVERY": "2",
    "TRAIN_LOG_EVERY": "1",
    "ATTN_NORM_MODE": "baseline",
    "ATTN_NORM_EPS": "1e-6",
    "MAX_WALLCLOCK_SECONDS": str(CPU_SUBSET_MAX_WALLCLOCK_SECONDS),
    "ARTIFACT_BUDGET_STRICT": "1",
}

RUN_CONFIG_TO_FACTOR_KEY = {
    "iterations": "ITERATIONS",
    "trainSeqLen": "TRAIN_SEQ_LEN",
    "trainBatchTokens": "TRAIN_BATCH_TOKENS",
    "valTokenLimit": "VAL_TOKEN_LIMIT",
    "modelDim": "MODEL_DIM",
    "numLayers": "NUM_LAYERS",
}

CPU_SUBSET_MUON_ENV_KEYS = frozenset(
    {
        "MATRIX_LR",
        "MUON_BACKEND_STEPS",
        "MUON_MOMENTUM",
        "MUON_MOMENTUM_WARMUP_START",
        "MUON_MOMENTUM_WARMUP_STEPS",
        "CONTROL_TENSOR_NAME_PATTERNS",
    }
)

CPU_SUBSET_ENV_OVERRIDE_ALLOWLIST = (
    frozenset(DEFAULT_CPU_ENV.keys()) | frozenset({"SEED"}) | CPU_SUBSET_MUON_ENV_KEYS
)
CPU_SUBSET_SERIALIZED_FACTOR_ALLOWLIST = frozenset(
    {
        "ATTN_NORM_EPS",
        "ATTN_NORM_MODE",
        "ARTIFACT_BUDGET_STRICT",
        "CONTROL_TENSOR_NAME_PATTERNS",
        "ITERATIONS",
        "MAX_WALLCLOCK_SECONDS",
        "MATRIX_LR",
        "MLP_MULT",
        "MODEL_DIM",
        "MUON_BACKEND_STEPS",
        "MUON_MOMENTUM",
        "MUON_MOMENTUM_WARMUP_START",
        "MUON_MOMENTUM_WARMUP_STEPS",
        "NUM_HEADS",
        "NUM_KV_HEADS",
        "NUM_LAYERS",
        "SEED",
        "TRAIN_BATCH_TOKENS",
        "TRAIN_LOG_EVERY",
        "TRAIN_SEQ_LEN",
        "VAL_BATCH_SIZE",
        "VAL_LOSS_EVERY",
        "VAL_TOKEN_LIMIT",
        "VOCAB_SIZE",
        "WARMDOWN_ITERS",
        "WARMUP_STEPS",
    }
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} root must be an object")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{json.dumps(payload, indent=2)}\n")


def parse_final_val_bpb(output: str) -> tuple[float, float]:
    matches = list(FINAL_BPB_RE.finditer(output))
    if not matches:
        raise ValueError("train_gpt output did not contain final_int8_zlib_roundtrip_exact val_bpb")
    match = matches[-1]
    return float(match.group("loss")), float(match.group("bpb"))


def build_cpu_subset_env(candidate: dict[str, Any], run_id: str) -> dict[str, str]:
    env = dict(DEFAULT_CPU_ENV)
    overrides = candidate.get("env")
    if isinstance(overrides, dict):
        for key, value in overrides.items():
            if (
                isinstance(key, str)
                and key in CPU_SUBSET_ENV_OVERRIDE_ALLOWLIST
                and isinstance(value, (str, int, float))
            ):
                env[key] = str(value)
    # Runner-level policy owns the hard cap; candidate configs are non-authoritative.
    env["MAX_WALLCLOCK_SECONDS"] = str(CPU_SUBSET_MAX_WALLCLOCK_SECONDS)
    env["RUN_ID"] = run_id
    env["SEED"] = str(candidate.get("seed", env.get("SEED", "1337")))
    return env


def regenerate_views(evidence_path: Path, status_path: Path, state_path: Path) -> None:
    source = load_json(evidence_path)
    status = load_json(status_path)
    state = load_json(state_path)
    rendered_status, rendered_state = apply_measurement_evidence(source, status, state)
    write_json(status_path, rendered_status)
    write_json(state_path, rendered_state)


def artifact_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _int_from_env(env: dict[str, str], key: str, default: int) -> int:
    raw = env.get(key)
    if raw is None:
        return default
    try:
        return int(float(raw))
    except ValueError:
        return default


def _cpu_subset_records(records: list[Any]) -> list[dict[str, Any]]:
    subset: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        if record.get("lane") != "cpu-subset":
            continue
        if not isinstance(record.get("experimentId"), str):
            continue
        subset.append(record)
    return subset


def _resolve_parent_experiment_id(candidate: dict[str, Any], records: list[dict[str, Any]]) -> str | None:
    parent_experiment_id = candidate.get("parentExperimentId")
    if isinstance(parent_experiment_id, str) and parent_experiment_id:
        return parent_experiment_id

    parent_candidate_id = candidate.get("parentCandidateId")
    if isinstance(parent_candidate_id, str) and parent_candidate_id:
        return f"exp-{parent_candidate_id}"

    if records:
        latest = records[-1].get("experimentId")
        if isinstance(latest, str) and latest:
            return latest
    return None


def _factor_snapshot_from_env(train_env: dict[str, str]) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for key in sorted(train_env.keys()):
        if (
            key in CPU_SUBSET_SERIALIZED_FACTOR_ALLOWLIST
            and key.isupper()
            and isinstance(train_env[key], str)
        ):
            snapshot[key] = train_env[key]
    return snapshot


def _factor_snapshot_from_record(record: dict[str, Any]) -> dict[str, str]:
    model_factory = record.get("modelFactory")
    if isinstance(model_factory, dict):
        normalized = model_factory.get("normalizedFactors")
        if isinstance(normalized, dict):
            snapshot = {
                str(key): str(value)
                for key, value in normalized.items()
                if isinstance(key, str) and key in CPU_SUBSET_SERIALIZED_FACTOR_ALLOWLIST
            }
            if snapshot:
                return snapshot

    run_config = record.get("runConfig")
    if not isinstance(run_config, dict):
        return {}

    snapshot: dict[str, str] = {}
    for run_key, factor_key in RUN_CONFIG_TO_FACTOR_KEY.items():
        value = run_config.get(run_key)
        if value is None:
            continue
        snapshot[factor_key] = str(value)
    return snapshot


def _changed_factors(parent: dict[str, str], current: dict[str, str]) -> list[dict[str, str]]:
    changed: list[dict[str, str]] = []
    for key in sorted(set(parent.keys()) | set(current.keys())):
        parent_value = parent.get(key)
        current_value = current.get(key)
        if parent_value == current_value:
            continue
        changed.append(
            {
                "factor": key,
                "parentValue": parent_value if parent_value is not None else "unset",
                "candidateValue": current_value if current_value is not None else "unset",
            }
        )
    return changed


def _build_ranking_table(
    records: list[dict[str, Any]],
    current_experiment_id: str,
    current_objective_value: float,
) -> dict[str, Any]:
    ranking_rows: list[dict[str, Any]] = []
    for record in records:
        objective_value = record.get("objectiveValue")
        experiment_id = record.get("experimentId")
        if not _is_numeric(objective_value):
            continue
        if not isinstance(experiment_id, str) or not experiment_id:
            continue
        ranking_rows.append({"experimentId": experiment_id, "objectiveValue": float(objective_value)})
    ranking_rows.append({"experimentId": current_experiment_id, "objectiveValue": current_objective_value})

    ranked = sorted(ranking_rows, key=lambda row: (row["objectiveValue"], row["experimentId"]))
    best = ranked[0]["objectiveValue"]
    candidate_rank = len(ranked)
    for index, row in enumerate(ranked, start=1):
        row["rank"] = index
        row["deltaToBest"] = round(row["objectiveValue"] - best, 12)
        if row["experimentId"] == current_experiment_id:
            candidate_rank = index
    return {
        "metricName": "val_bpb",
        "direction": "lower_is_better",
        "candidateRank": candidate_rank,
        "totalCandidates": len(ranked),
        "rows": ranked,
    }


def sanitize_candidate_id(candidate_id: Any) -> str:
    raw = str(candidate_id or "")
    sanitized = re.sub(r"[^a-zA-Z0-9]+", "-", raw).strip("-").lower()
    return sanitized or "candidate"


def _resolve_run_directory(run_root: Path, run_id: str) -> Path:
    resolved_root = run_root.resolve()
    run_dir = (resolved_root / run_id).resolve()
    try:
        run_dir.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("Resolved run directory escapes run_root containment.") from exc
    return run_dir


def _run_with_hard_timeout(
    *,
    command: list[str],
    run_dir: Path,
    env: dict[str, str],
    timeout_seconds: int,
) -> dict[str, Any]:
    process = subprocess.Popen(
        command,
        cwd=run_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
        return {
            "timedOut": False,
            "stdout": stdout,
            "stderr": stderr,
            "returnCode": int(process.returncode or 0),
            "timeoutContainmentMode": "host-level",
            "processTerminationScope": "process-group",
            "terminationReason": "completed",
        }
    except subprocess.TimeoutExpired as timeout_exc:
        stdout = timeout_exc.stdout or ""
        stderr = timeout_exc.stderr or ""
        termination_scope = "process-group"
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        except Exception:
            termination_scope = "process"
            process.kill()
        process.wait(timeout=5)
        return {
            "timedOut": True,
            "stdout": stdout,
            "stderr": stderr,
            "returnCode": int(process.returncode or -9),
            "timeoutContainmentMode": "host-level",
            "processTerminationScope": termination_scope,
            "terminationReason": "hard-timeout",
        }


def _proposal_task_id(task_id: str, candidate_id: str) -> str:
    suffix = re.sub(r"[^a-zA-Z0-9-]+", "-", candidate_id).strip("-").lower() or "candidate"
    return f"{task_id}-{suffix}-follow-up"


def append_cpu_subset_evidence(
    *,
    candidate: dict[str, Any],
    run_id: str,
    completed_at: str,
    val_loss: float,
    val_bpb: float,
    command: list[str],
    run_dir: Path,
    train_env: dict[str, str],
    evidence_path: Path,
) -> dict[str, Any]:
    evidence = load_json(evidence_path)
    records = evidence.setdefault("experimentRecords", [])
    if not isinstance(records, list):
        raise ValueError("experimentRecords must be a list")

    candidate_id = str(candidate.get("candidateId") or run_id)
    task_id = str(candidate.get("taskId") or "manual-cpu-subset")
    experiment_id = f"exp-{candidate_id}"
    idempotency_key = f"{task_id}:cpu-subset:{experiment_id}"
    existing = next(
        (
            record
            for record in records
            if isinstance(record, dict) and record.get("idempotencyKey") == idempotency_key
        ),
        None,
    )
    if existing is not None:
        return existing

    cpu_records = _cpu_subset_records(records)
    evidence_id = f"evidence-{experiment_id}"
    parent_experiment_id = _resolve_parent_experiment_id(candidate, cpu_records)
    parent_record = next(
        (
            record
            for record in cpu_records
            if isinstance(record.get("experimentId"), str) and record["experimentId"] == parent_experiment_id
        ),
        None,
    )
    parent_frontier_id = candidate.get("parentFrontierId")
    if not isinstance(parent_frontier_id, str) or not parent_frontier_id:
        fallback_frontier_id = candidate.get("frontierId")
        if isinstance(fallback_frontier_id, str) and fallback_frontier_id:
            parent_frontier_id = fallback_frontier_id
        else:
            parent_frontier_id = None
    current_factors = _factor_snapshot_from_env(train_env)
    parent_factors = _factor_snapshot_from_record(parent_record) if parent_record else {}
    ranking_table = _build_ranking_table(cpu_records, experiment_id, val_bpb)
    effective_runtime_cap_sec = _int_from_env(train_env, "MAX_WALLCLOCK_SECONDS", 0)
    if effective_runtime_cap_sec <= 0:
        raise ValueError("MAX_WALLCLOCK_SECONDS must be present and greater than zero.")
    candidate_rank = int(ranking_table["candidateRank"])
    follow_up_lane = "cheap-screen" if candidate_rank == 1 else "cpu-subset"
    promotion_decision = "propose-follow-up" if candidate_rank == 1 else "hold"
    retirement_decision = "retain" if candidate_rank <= 3 else "retire"

    record = {
        "experimentId": experiment_id,
        "evidenceId": evidence_id,
        "lane": "cpu-subset",
        "status": "accepted",
        "completedAt": completed_at,
        "objectiveMetricName": "val_bpb",
        "objectiveValue": val_bpb,
        "failureCode": None,
        "failureMessage": None,
        "idempotencyKey": idempotency_key,
        "taskId": task_id,
        "traceId": f"trace-{candidate_id}",
        "runConfig": {
            "seed": train_env.get("SEED"),
            "iterations": _int_from_env(train_env, "ITERATIONS", 0),
            "trainSeqLen": _int_from_env(train_env, "TRAIN_SEQ_LEN", 0),
            "trainBatchTokens": _int_from_env(train_env, "TRAIN_BATCH_TOKENS", 0),
            "valTokenLimit": _int_from_env(train_env, "VAL_TOKEN_LIMIT", 0),
            "modelDim": _int_from_env(train_env, "MODEL_DIM", 0),
            "numLayers": _int_from_env(train_env, "NUM_LAYERS", 0),
        },
        "budgetCaps": {
            "maxRuntimeSeconds": effective_runtime_cap_sec,
            "hostClass": "cpu-smoke",
        },
        "runtimeValidationCaps": {
            "maxRuntimeSeconds": effective_runtime_cap_sec,
            "iterations": _int_from_env(train_env, "ITERATIONS", 0),
            "valTokenLimit": _int_from_env(train_env, "VAL_TOKEN_LIMIT", 0),
            "valBatchSize": _int_from_env(train_env, "VAL_BATCH_SIZE", 0),
            "validationClass": "cpu-subset",
            "capPolicySource": CPU_SUBSET_CAP_POLICY_SOURCE,
        },
        "effectiveRuntimeCapSec": effective_runtime_cap_sec,
        "capPolicySource": CPU_SUBSET_CAP_POLICY_SOURCE,
        "timeoutContainmentMode": "host-level",
        "processTerminationScope": "process-group",
        "capOverrideActor": "none",
        "enforcementProof": {
            "runnerHardCapSeconds": CPU_SUBSET_MAX_WALLCLOCK_SECONDS,
            "envMaxWallclockSeconds": effective_runtime_cap_sec,
            "source": CPU_SUBSET_CAP_POLICY_SOURCE,
        },
        "resultClass": "cpu-subset",
        "verificationClass": "cpu-subset",
        "classificationReason": (
            "Real local PyTorch CPU run on a reduced FineWeb token subset; useful for "
            "ranking cheap ideas, not leaderboard or benchmark-verified progress."
        ),
        "lineage": {
            "source": "model-factory",
            "parentExperimentId": parent_experiment_id,
            "parentFrontierId": parent_frontier_id,
            "changedFactors": _changed_factors(parent_factors, current_factors),
        },
        "modelFactory": {
            "owner": "cpu-subset-runner",
            "normalizedFactors": current_factors,
        },
        "rankingTable": ranking_table,
        "modelFactoryDecision": {
            "promotionDecision": promotion_decision,
            "retirementDecision": retirement_decision,
            "reason": (
                "Candidate ranking is derived from CPU-subset val_bpb only; retain top-ranked "
                "directions for lane-attributed follow-up while keeping benchmark progress gated."
            ),
        },
        "followUpTaskProposal": {
            "sourceLane": "cpu-subset",
            "lane": follow_up_lane,
            "taskId": _proposal_task_id(task_id, candidate_id),
            "candidateId": candidate_id,
            "parentExperimentId": parent_experiment_id,
            "title": f"Follow up CPU-subset candidate {candidate_id}",
            "summary": (
                f"Candidate rank {candidate_rank}/{ranking_table['totalCandidates']} on CPU-subset val_bpb; "
                f"proposal lane is {follow_up_lane}."
            ),
            "runtimeValidationCaps": {
                "maxRuntimeSeconds": effective_runtime_cap_sec,
                "valTokenLimit": _int_from_env(train_env, "VAL_TOKEN_LIMIT", 0),
            },
        },
        "benchmarkProgressEligible": False,
        "observedMetricOutput": {
            "name": "val_bpb",
            "value": val_bpb,
            "valLoss": val_loss,
            "direction": "lower_is_better",
        },
        "executionCommands": [" ".join(command)],
        "artifactPaths": [
            artifact_path(run_dir / "stdout.txt"),
            artifact_path(run_dir / "stderr.txt"),
            artifact_path(run_dir / "result.json"),
            artifact_path(run_dir / "final_model.int8.ptz"),
        ],
        "promotionRationale": {
            "decision": "hold",
            "reason": "CPU-subset evidence can rank ideas but cannot claim benchmark progress.",
            "missingEvidence": ["benchmark-verified-run"],
        },
    }
    records.append(record)

    summary = evidence.setdefault("summary", {})
    if isinstance(summary, dict):
        artifact_ids = summary.get("artifactIds")
        if not isinstance(artifact_ids, list):
            artifact_ids = []
        if evidence_id not in artifact_ids:
            artifact_ids.append(evidence_id)
        summary["artifactIds"] = artifact_ids
        summary["totalExperiments"] = int(summary.get("totalExperiments", 0)) + 1
        summary["acceptedExperiments"] = int(summary.get("acceptedExperiments", 0)) + 1
        summary["mostRecentEvidenceId"] = evidence_id

    recent = evidence.get("recentCompletedExperiments")
    if not isinstance(recent, list):
        recent = []
    if experiment_id not in recent:
        recent.append(experiment_id)
    evidence["recentCompletedExperiments"] = recent
    evidence["updatedAt"] = completed_at
    write_json(evidence_path, evidence)
    return record


def run_cpu_subset_experiment(
    *,
    candidate: dict[str, Any],
    evidence_path: Path,
    status_path: Path,
    state_path: Path,
    run_root: Path,
) -> dict[str, Any]:
    candidate_id = str(candidate.get("candidateId") or f"cpu-subset-{utc_now_iso()}")
    run_id = f"cpu_subset_{sanitize_candidate_id(candidate_id)}"
    run_dir = _resolve_run_directory(run_root, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    train_env = build_cpu_subset_env(candidate, run_id)
    effective_runtime_cap_sec = _int_from_env(train_env, "MAX_WALLCLOCK_SECONDS", 0)
    if effective_runtime_cap_sec <= 0:
        raise ValueError("Runner hard timeout must be configured as a positive integer.")
    command = [sys.executable, str(REPO_ROOT / "train_gpt.py")]
    env = os.environ.copy()
    env.update(train_env)
    completed = _run_with_hard_timeout(
        command=command,
        run_dir=run_dir,
        env=env,
        timeout_seconds=effective_runtime_cap_sec,
    )
    (run_dir / "stdout.txt").write_text(str(completed.get("stdout", "")))
    (run_dir / "stderr.txt").write_text(str(completed.get("stderr", "")))
    if bool(completed.get("timedOut")):
        result = {
            "status": "timeout_enforced",
            "reasonCode": "cpu-subset-timeout-enforced",
            "returnCode": int(completed.get("returnCode", -9)),
            "stdoutPath": artifact_path(run_dir / "stdout.txt"),
            "stderrPath": artifact_path(run_dir / "stderr.txt"),
            "effectiveRuntimeCapSec": effective_runtime_cap_sec,
            "capPolicySource": CPU_SUBSET_CAP_POLICY_SOURCE,
            "timeoutContainmentMode": str(completed.get("timeoutContainmentMode")),
            "processTerminationScope": str(completed.get("processTerminationScope")),
            "capOverrideActor": "none",
            "enforcementProof": {
                "runnerHardCapSeconds": CPU_SUBSET_MAX_WALLCLOCK_SECONDS,
                "effectiveRuntimeCapSec": effective_runtime_cap_sec,
                "terminationReason": str(completed.get("terminationReason")),
            },
            "terminationReason": str(completed.get("terminationReason")),
            "lifecycleEvent": "hard_timeout_enforced",
        }
        write_json(run_dir / "result.json", result)
        return result

    if int(completed.get("returnCode", 1)) != 0:
        result = {
            "status": "error",
            "reasonCode": "cpu-subset-run-failed",
            "returnCode": int(completed.get("returnCode", 1)),
            "stdoutPath": artifact_path(run_dir / "stdout.txt"),
            "stderrPath": artifact_path(run_dir / "stderr.txt"),
            "effectiveRuntimeCapSec": effective_runtime_cap_sec,
            "capPolicySource": CPU_SUBSET_CAP_POLICY_SOURCE,
        }
        write_json(run_dir / "result.json", result)
        return result

    val_loss, val_bpb = parse_final_val_bpb(str(completed.get("stdout", "")))
    completed_at = utc_now_iso()
    record = append_cpu_subset_evidence(
        candidate=candidate,
        run_id=run_id,
        completed_at=completed_at,
        val_loss=val_loss,
        val_bpb=val_bpb,
        command=command,
        run_dir=run_dir,
        train_env=train_env,
        evidence_path=evidence_path,
    )
    regenerate_views(evidence_path, status_path, state_path)
    result = {
        "status": "success",
        "reasonCode": "cpu-subset-success",
        "candidateId": record["experimentId"],
        "observedMetric": record["observedMetricOutput"],
        "resultClass": "cpu-subset",
        "artifactPaths": record["artifactPaths"],
    }
    write_json(run_dir / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one real CPU-subset Parameter Golf experiment.")
    parser.add_argument("--candidate", help="Path to candidate JSON. Defaults to a tiny baseline candidate.")
    parser.add_argument("--evidence-path", default=str(DEFAULT_EVIDENCE_PATH))
    parser.add_argument("--status-path", default=str(DEFAULT_STATUS_PATH))
    parser.add_argument("--state-path", default=str(DEFAULT_STATE_PATH))
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument("--output", help="Optional result JSON output path.")
    args = parser.parse_args()

    candidate = {
        "taskId": "manual-cpu-subset",
        "candidateId": "baseline-tiny-cpu",
        "seed": 1337,
    }
    if args.candidate:
        candidate = load_json(Path(args.candidate))

    result = run_cpu_subset_experiment(
        candidate=candidate,
        evidence_path=Path(args.evidence_path),
        status_path=Path(args.status_path),
        state_path=Path(args.state_path),
        run_root=Path(args.run_root),
    )
    if args.output:
        write_json(Path(args.output), result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
