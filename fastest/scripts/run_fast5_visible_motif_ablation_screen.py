import json
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MATRIX_PATH = REPO_ROOT / "planning/fast5_visible_motif_ablation_matrix.json"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "records/fast5_visible_motif_ablation_screen_results.json"
DEFAULT_SUMMARY_PATH = REPO_ROOT / "records/fast5_visible_motif_ablation_screen_summary.md"
REQUIRED_MOTIFS = {
    "SP8192 tokenizer",
    "3-layer recurrence",
    "parallel residuals",
    "legal score-first TTT",
    "QK-gain tuning",
    "Hessian-aware clipping",
}
REQUIRED_BASELINE_FIELDS = {
    "baselineId",
    "configHash",
    "commit",
    "dataset",
    "scoringCommand",
    "artifactMeasurementCommand",
    "seedPolicy",
    "scoreBpb",
    "artifactBytes",
}
REQUIRED_ABLATION_FIELDS = {
    "runId",
    "motif",
    "variantConfigRef",
    "seed",
    "scoreBpb",
    "artifactBytes",
    "legalityClass",
    "runtimeSeconds",
    "status",
    "nextAction",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_json(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("json-root-not-object")
    return payload


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{json.dumps(payload, indent=2)}\n")


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_trusted_control(control: dict) -> list[str]:
    missing = sorted(field for field in REQUIRED_BASELINE_FIELDS if field not in control)
    errors = [f"trustedControl missing {field}" for field in missing]
    for field in ("scoreBpb", "artifactBytes"):
        if field in control and not _is_number(control[field]):
            errors.append(f"trustedControl {field} must be numeric")
    return errors


def _validate_ablation(ablation: dict) -> list[str]:
    missing = sorted(field for field in REQUIRED_ABLATION_FIELDS if field not in ablation)
    errors = [f"{ablation.get('runId', '<unknown>')} missing {field}" for field in missing]
    for field in ("scoreBpb", "artifactBytes", "runtimeSeconds"):
        if field in ablation and not _is_number(ablation[field]):
            errors.append(f"{ablation.get('runId', '<unknown>')} {field} must be numeric")
    if ablation.get("dependency_coupled") is True:
        coupled_fields = ablation.get("coupledFields")
        if not isinstance(coupled_fields, list) or not coupled_fields:
            errors.append(f"{ablation.get('runId', '<unknown>')} dependency_coupled needs coupledFields")
    return errors


def _classify_legality(ablation: dict, artifact_limit_bytes: int) -> str:
    if ablation.get("status") != "success":
        return "invalid"
    artifact_bytes = ablation.get("artifactBytes")
    if _is_number(artifact_bytes) and artifact_bytes > artifact_limit_bytes:
        return "artifact-limit-fail"
    legality_class = ablation.get("legalityClass")
    if isinstance(legality_class, str) and legality_class:
        return legality_class
    return "unknown"


def _next_action(ablation: dict, legality_class: str, score_delta_bpb: float) -> str:
    if ablation.get("status") != "success":
        return "investigate-or-rerun"
    if legality_class == "artifact-limit-fail":
        return "shrink-or-drop"
    if legality_class not in {"legal", "non-record-only"}:
        return "legality-remediation-before-combination"
    configured_action = ablation.get("nextAction")
    if isinstance(configured_action, str) and configured_action:
        return configured_action
    if score_delta_bpb <= -0.01:
        return "deepen"
    if abs(score_delta_bpb) <= 0.005:
        return "rerun-near-threshold"
    return "drop"


def _result_row(ablation: dict, control: dict, artifact_limit_bytes: int) -> dict:
    baseline_score = float(control["scoreBpb"])
    baseline_artifact = int(control["artifactBytes"])
    score_bpb = float(ablation["scoreBpb"])
    artifact_bytes = int(ablation["artifactBytes"])
    score_delta = round(score_bpb - baseline_score, 6)
    legality_class = _classify_legality(ablation, artifact_limit_bytes)

    return {
        "run_id": ablation["runId"],
        "baseline_id": control["baselineId"],
        "motif": ablation["motif"],
        "variant_config_ref": ablation["variantConfigRef"],
        "seed": ablation["seed"],
        "score_bpb": score_bpb,
        "score_delta_bpb": score_delta,
        "artifact_bytes": artifact_bytes,
        "artifact_delta_bytes": artifact_bytes - baseline_artifact,
        "legality_class": legality_class,
        "legality_notes": ablation.get("legalityNotes", ""),
        "runtime_seconds": ablation["runtimeSeconds"],
        "status": ablation["status"],
        "failure_reason": ablation.get("failureReason"),
        "next_action": _next_action(ablation, legality_class, score_delta),
    }


def _render_summary(payload: dict) -> str:
    lines = [
        "# FAST-5 Visible Motif Ablation Screen",
        "",
        f"Baseline: {payload['baseline']['baselineId']} ({payload['baseline']['scoreBpb']} bpb)",
        "",
        "| Motif | Delta bpb | Artifact delta bytes | Legality | Next action |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for row in payload["rankedRows"]:
        lines.append(
            "| {motif} | {score_delta_bpb:+.6f} | {artifact_delta_bytes:+d} | "
            "{legality_class} | {next_action} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Lower bits-per-byte deltas are better. Rows with legality failures are blocked from",
            "leaderboard-oriented combination until the named remediation action is complete.",
            "",
        ]
    )
    return "\n".join(lines)


def execute_visible_motif_ablation_screen(
    *,
    matrix_path: Path = DEFAULT_MATRIX_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    summary_path: Path = DEFAULT_SUMMARY_PATH,
) -> dict:
    matrix = _load_json(matrix_path)
    control = matrix.get("trustedControl")
    ablations = matrix.get("ablations")
    if not isinstance(control, dict):
        raise ValueError("trustedControl must be an object")
    if not isinstance(ablations, list):
        raise ValueError("ablations must be a list")

    errors = _validate_trusted_control(control)
    for ablation in ablations:
        if not isinstance(ablation, dict):
            errors.append("ablation row must be an object")
            continue
        errors.extend(_validate_ablation(ablation))

    motifs = {row.get("motif") for row in ablations if isinstance(row, dict)}
    missing_motifs = sorted(REQUIRED_MOTIFS - motifs)
    if missing_motifs:
        errors.append(f"missing required motifs: {', '.join(missing_motifs)}")
    if errors:
        return {
            "taskId": matrix.get("taskId"),
            "status": "error",
            "reasonCode": "fast5-screen-config-invalid",
            "errors": errors,
        }

    artifact_limit = int(matrix.get("artifactLimitBytes", 16_000_000))
    rows = [_result_row(ablation, control, artifact_limit) for ablation in ablations]
    ranked_rows = sorted(
        rows,
        key=lambda row: (
            row["status"] != "success",
            row["score_delta_bpb"],
            row["artifact_delta_bytes"],
            row["motif"],
        ),
    )
    now = _utc_now_iso()
    payload = {
        "schemaVersion": 1,
        "taskId": matrix.get("taskId", "FAST-5"),
        "generatedAt": now,
        "artifactLimitBytes": artifact_limit,
        "baseline": control,
        "rows": rows,
        "rankedRows": ranked_rows,
    }
    _write_json(output_path, payload)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(_render_summary(payload))
    return {
        "taskId": matrix.get("taskId", "FAST-5"),
        "status": "success",
        "reasonCode": "fast5-screen-complete",
        "completedAt": now,
        "outputPath": str(output_path),
        "summaryPath": str(summary_path),
        "rowCount": len(rows),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the FAST-5 visible motif ablation screen.")
    parser.add_argument("--matrix", default=str(DEFAULT_MATRIX_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY_PATH))
    args = parser.parse_args()

    outcome = execute_visible_motif_ablation_screen(
        matrix_path=Path(args.matrix),
        output_path=Path(args.output),
        summary_path=Path(args.summary),
    )
    print(f"{json.dumps(outcome, indent=2)}")


if __name__ == "__main__":
    main()
