from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .control_pipeline import evaluateControlTrust


def _load_json(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _nested(mapping: dict[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        parsed = float(value)
        if math.isfinite(parsed):
            return parsed
    return None


def _text(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _numbers_match(left: Any, right: Any, *, tolerance: float = 1e-12) -> bool:
    left_number = _number(left)
    right_number = _number(right)
    if left_number is None or right_number is None:
        return False
    return abs(left_number - right_number) <= tolerance


def _metric_improved(candidate_value: float, control_value: float, direction: str) -> bool:
    if direction == "higher_is_better":
        return candidate_value > control_value
    return candidate_value < control_value


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _append_eval_drift_violations(
    violations: list[str],
    candidate: dict[str, Any],
    control_bundle: dict[str, Any],
) -> None:
    evaluation = candidate.get("evaluation")
    if not isinstance(evaluation, dict):
        violations.append("eval-drift:evaluation-missing")
        return

    snapshot = control_bundle.get("spec_snapshot")
    if not isinstance(snapshot, dict):
        violations.append("eval-drift:control-fingerprint-missing")
        return

    control_reference = candidate.get("control_reference")
    if not isinstance(control_reference, dict):
        violations.append("eval-drift:control-reference-missing")
    else:
        if control_reference.get("bundle_id") != control_bundle.get("bundle_id"):
            violations.append("eval-drift:control-bundle-id")
        if control_reference.get("spec_hash") != control_bundle.get("spec_hash"):
            violations.append("eval-drift:control-spec-hash")

    if evaluation.get("dataset_id") != snapshot.get("dataset_id"):
        violations.append("eval-drift:dataset-id")
    if evaluation.get("tokenizer_id") != snapshot.get("tokenizer_id"):
        violations.append("eval-drift:tokenizer-id")
    if evaluation.get("metric_name") != _nested(snapshot, "metric", "name"):
        violations.append("eval-drift:metric-name")

    candidate_metric = candidate.get("metric")
    if not isinstance(candidate_metric, dict):
        violations.append("eval-drift:candidate-metric-missing")
        return
    if candidate_metric.get("name") != _nested(snapshot, "metric", "name"):
        violations.append("eval-drift:candidate-metric-name")
    if candidate_metric.get("direction") != _nested(snapshot, "metric", "direction"):
        violations.append("eval-drift:candidate-metric-direction")


def _append_artifact_mismatch_violations(
    violations: list[str],
    candidate: dict[str, Any],
) -> None:
    artifact = candidate.get("artifact")
    manifest = candidate.get("artifact_manifest")
    if not isinstance(artifact, dict):
        violations.append("artifact-mismatch:claim-missing")
        return
    if not isinstance(manifest, dict):
        violations.append("artifact-mismatch:manifest-missing")
        return

    if not _numbers_match(artifact.get("claimed_bytes"), manifest.get("bytes")):
        violations.append("artifact-mismatch:bytes")
    if not _numbers_match(artifact.get("claimed_limit_bytes"), manifest.get("limit_bytes")):
        violations.append("artifact-mismatch:limit-bytes")
    if artifact.get("claimed_sha256") != manifest.get("sha256"):
        violations.append("artifact-mismatch:sha256")

    claimed_bytes = _number(artifact.get("claimed_bytes"))
    claimed_limit = _number(artifact.get("claimed_limit_bytes"))
    if claimed_bytes is None or claimed_limit is None or claimed_bytes > claimed_limit:
        violations.append("artifact-mismatch:artifact-limit")

    claimed_compression = artifact.get("claimed_compression")
    actual_compression = manifest.get("compression")
    if not isinstance(claimed_compression, dict) or not isinstance(actual_compression, dict):
        violations.append("artifact-mismatch:compression-config")
    else:
        if claimed_compression.get("codec") != actual_compression.get("codec"):
            violations.append("artifact-mismatch:compression-codec")
        if not _numbers_match(claimed_compression.get("ratio"), actual_compression.get("ratio")):
            violations.append("artifact-mismatch:compression-ratio")

    if artifact.get("claimed_quantization_scheme") != manifest.get("quantization_scheme"):
        violations.append("artifact-mismatch:quantization-scheme")

    manifest_path = _text(manifest.get("path"))
    if manifest_path:
        artifact_path = Path(manifest_path)
        if not artifact_path.exists():
            violations.append("artifact-mismatch:file-missing")
            return
        manifest_bytes = _number(manifest.get("bytes"))
        if manifest_bytes is None or artifact_path.stat().st_size != int(manifest_bytes):
            violations.append("artifact-mismatch:file-bytes")
        manifest_sha256 = _text(manifest.get("sha256"))
        if manifest_sha256 is None or _sha256_file(artifact_path) != manifest_sha256:
            violations.append("artifact-mismatch:file-sha256")


def _append_reproduction_violations(
    violations: list[str],
    candidate: dict[str, Any],
    *,
    promising: bool,
) -> None:
    if not promising:
        return

    reproduction = candidate.get("reproduction")
    if not isinstance(reproduction, dict) or reproduction.get("status") != "verified":
        violations.append("reproduction-failed:not-verified")
        return

    claimed_value = _number(_nested(candidate, "metric", "value"))
    reproduced_value = _number(reproduction.get("metric_value"))
    tolerance = _number(reproduction.get("tolerance"))
    if tolerance is None:
        tolerance = 0.0

    if claimed_value is None or reproduced_value is None:
        violations.append("reproduction-failed:metric-missing")
        return
    if abs(claimed_value - reproduced_value) > tolerance:
        violations.append("reproduction-failed:metric-mismatch")


def validateCandidateRegressionGate(
    candidate: dict[str, Any],
    *,
    bundle_path: Path | str,
    spec_ref: Path | str,
) -> dict[str, Any]:
    control_trust = evaluateControlTrust(bundle_path=bundle_path, spec_ref=spec_ref)
    control_bundle = _load_json(bundle_path)
    violations: list[str] = []

    if control_trust.get("state") != "trusted":
        violations.append("control-untrusted")

    _append_eval_drift_violations(violations, candidate, control_bundle)
    _append_artifact_mismatch_violations(violations, candidate)

    candidate_value = _number(_nested(candidate, "metric", "value"))
    control_value = _number(control_trust.get("primary_metric"))
    direction = _text(_nested(candidate, "metric", "direction")) or "lower_is_better"
    promising = (
        candidate_value is not None
        and control_value is not None
        and _metric_improved(candidate_value, control_value, direction)
    )
    _append_reproduction_violations(violations, candidate, promising=promising)

    unique_violations = sorted(set(violations))
    return {
        "schema_version": 1,
        "candidateId": candidate.get("candidateId"),
        "decision": "reject" if unique_violations else "accept",
        "promising": promising,
        "controlTrustState": control_trust.get("state"),
        "controlPrimaryMetric": control_trust.get("primary_metric"),
        "candidatePrimaryMetric": candidate_value,
        "violations": unique_violations,
    }
