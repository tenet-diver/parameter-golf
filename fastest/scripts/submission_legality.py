from __future__ import annotations

from typing import Any


ARTIFACT_LIMIT_BYTES = 16_000_000
TARGET_RUNTIME_SECONDS = 600

LEADERBOARD_LEGAL = "leaderboard-legal"
NON_RECORD_ONLY = "non-record-only"
UNCERTAIN = "uncertain"


def _first_present(record: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in record:
            return record[key]
    return None


def _nested_first_present(record: dict[str, Any], container: str, *keys: str) -> Any:
    nested = record.get(container)
    if not isinstance(nested, dict):
        return None
    return _first_present(nested, *keys)


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _text(value: Any) -> str:
    return str(value).strip().lower() if value is not None else ""


def _metric_is_fineweb_bpb(record: dict[str, Any]) -> bool | None:
    metric = _text(_first_present(record, "objectiveMetricName", "metric", "scoreMetric", "scoring"))
    dataset = _text(_first_present(record, "dataset", "datasetId", "dataset_id", "validationDataset"))
    if not metric and not dataset:
        return None
    metric_is_bpb = "bpb" in metric or "bits-per-byte" in metric or "bits per byte" in metric
    metric_is_loss = "benchmark-score" in metric or "cpu-subset" in metric or "local" in metric
    if metric_is_loss:
        return False
    if "fineweb" in metric and metric_is_bpb:
        return True
    if metric in {"val_bpb", "validation_bpb"}:
        return True
    if "fineweb" in dataset and (metric_is_bpb or metric in {"val_bpb", "validation_bpb"}):
        return True
    return False if metric else None


def _runtime_is_leaderboard_credible(record: dict[str, Any]) -> bool | None:
    credible_path = _bool_or_none(_first_present(record, "credibleTenMinutePath", "leaderboardRuntimeCredible"))
    runtime = _number(
        _first_present(record, "runtimeSeconds", "runtime_seconds", "trainRuntimeSeconds", "wallTimeSeconds")
    )
    if runtime is None:
        runtime = _number(_nested_first_present(record, "budgetCaps", "maxRuntimeSeconds", "runtimeSeconds"))
    hardware = _text(_first_present(record, "hardware", "hardwareClass", "accelerator"))
    hardware_is_h100 = "8xh100" in hardware.replace(" ", "") or ("8x" in hardware and "h100" in hardware)

    if credible_path is True:
        return True
    if credible_path is False:
        return False
    if runtime is None:
        return None
    if runtime <= TARGET_RUNTIME_SECONDS and (hardware_is_h100 or not hardware):
        return True if hardware_is_h100 else None
    return False


def _declared_non_record(record: dict[str, Any]) -> bool:
    values = [
        _text(_first_present(record, "track", "submissionClass", "submission_class", "lane")),
        _text(_nested_first_present(record, "metadata", "track", "submissionClass")),
    ]
    return any("non-record" in value or "unlimited" in value or "cpu-subset" in value for value in values)


def classify_submission_legality(record: dict[str, Any]) -> dict[str, Any]:
    passed_rules: list[str] = []
    non_record_reasons: list[str] = []
    uncertainty_reasons: list[str] = []

    artifact_bytes = _number(
        _first_present(record, "artifactBytes", "artifact_bytes", "bytes_total", "bytesTotal")
    )
    if artifact_bytes is None:
        uncertainty_reasons.append("missing-artifact-size")
    elif artifact_bytes <= ARTIFACT_LIMIT_BYTES:
        passed_rules.append("artifact-under-16mb")
    else:
        non_record_reasons.append("artifact-exceeds-16mb")

    metric_is_fineweb = _metric_is_fineweb_bpb(record)
    if metric_is_fineweb is True:
        passed_rules.append("fineweb-bpb-scoring")
    elif metric_is_fineweb is False:
        non_record_reasons.append("not-fineweb-validation-bpb")
    else:
        uncertainty_reasons.append("missing-fineweb-bpb-scoring-evidence")

    runtime_is_credible = _runtime_is_leaderboard_credible(record)
    if runtime_is_credible is True:
        passed_rules.append("credible-10min-8xh100-path")
    elif runtime_is_credible is False:
        non_record_reasons.append("runtime-exceeds-10-minute-leaderboard-cap")
    else:
        uncertainty_reasons.append("missing-runtime-or-credible-path")

    self_contained = _bool_or_none(_first_present(record, "selfContainedArtifact", "self_contained_artifact"))
    if self_contained is True:
        passed_rules.append("self-contained-artifact")
    elif self_contained is False:
        non_record_reasons.append("artifact-not-self-contained")
    else:
        uncertainty_reasons.append("missing-self-contained-evidence")

    external_downloads = _bool_or_none(
        _first_present(record, "externalDownloadsDuringEvaluation", "external_downloads_during_evaluation")
    )
    network_access = _bool_or_none(
        _first_present(record, "networkAccessDuringEvaluation", "network_access_during_evaluation")
    )
    if external_downloads is True or network_access is True:
        non_record_reasons.append("external-access-during-evaluation")
    elif external_downloads is False and network_access is False:
        passed_rules.append("no-external-evaluation-access")
    else:
        uncertainty_reasons.append("missing-external-access-evidence")

    validation_use = _bool_or_none(
        _first_present(record, "usesValidationDataDuringTraining", "validationDataUsedDuringTraining")
    )
    if validation_use is False:
        passed_rules.append("no-validation-data-training-use")
    elif validation_use is True:
        non_record_reasons.append("validation-data-used-during-training")
    else:
        uncertainty_reasons.append("missing-validation-data-use-evidence")

    if _declared_non_record(record):
        non_record_reasons.append("declared-non-record-track")

    status = LEADERBOARD_LEGAL
    if non_record_reasons:
        status = NON_RECORD_ONLY
    elif uncertainty_reasons:
        status = UNCERTAIN

    return {
        "status": status,
        "policy": {
            "preset": "parameter-golf",
            "artifactLimitBytes": ARTIFACT_LIMIT_BYTES,
            "leaderboardRuntimeSeconds": TARGET_RUNTIME_SECONDS,
            "scoring": "FineWeb validation bits-per-byte",
        },
        "passedRules": sorted(set(passed_rules)),
        "nonRecordReasons": sorted(set(non_record_reasons)),
        "uncertaintyReasons": sorted(set(uncertainty_reasons)),
    }


def classify_submission_legality_batch(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    classified = []
    for record in records:
        if not isinstance(record, dict):
            continue
        identity = _first_present(record, "experimentId", "ideaId", "candidateId", "submissionId")
        classified.append(
            {
                "id": identity,
                "classification": classify_submission_legality(record),
            }
        )
    return classified
