from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

CONTROL_EVIDENCE_SCHEMA_VERSION = 1
_INDEX_FILE_NAME = "control_measurement_index.json"


@dataclass(frozen=True)
class ControlRunSubmission:
    spec_ref: Path
    spec_hash: str
    campaign_tick_id: str
    evidence_store: Path
    bundle_id: str
    reused_existing_bundle: bool
    spec: dict[str, Any]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def _load_spec(path: Path) -> dict[str, Any]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    required_top = {
        "schema_version",
        "spec_id",
        "dataset_id",
        "tokenizer_id",
        "seed_set",
        "metric",
        "command_template",
        "required_env",
    }
    missing = sorted(required_top.difference(spec.keys()))
    if missing:
        raise ValueError(f"control spec missing keys: {', '.join(missing)}")
    if not isinstance(spec["seed_set"], list) or not spec["seed_set"]:
        raise ValueError("control spec seed_set must be a non-empty list")
    return spec


def _read_index(index_path: Path) -> dict[str, str]:
    if not index_path.exists():
        return {}
    return json.loads(index_path.read_text(encoding="utf-8"))


def _write_index(index_path: Path, data: Mapping[str, str]) -> None:
    index_path.write_text(json.dumps(data, sort_keys=True, indent=2), encoding="utf-8")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return sum(values) / len(values)


def _environment_manifest() -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
    }


def _validate_environment(spec: dict[str, Any], env: Mapping[str, str]) -> list[str]:
    reasons: list[str] = []
    required_env = spec.get("required_env", {})
    required_python_prefix = str(required_env.get("python", "")).strip()
    if required_python_prefix and not env.get("python_version", "").startswith(required_python_prefix):
        reasons.append(
            "environment incompatibility: "
            f"python_version={env.get('python_version')} does not match required prefix {required_python_prefix}"
        )
    required_system = str(required_env.get("system", "")).strip()
    if required_system and env.get("system", "").lower() != required_system.lower():
        reasons.append(
            "environment incompatibility: "
            f"system={env.get('system')} does not match required system {required_system}"
        )
    return reasons


def submitControlRun(spec_ref: Path | str, campaign_tick_id: str, evidence_store: Path | str) -> ControlRunSubmission:
    spec_path = Path(spec_ref)
    evidence_root = Path(evidence_store)
    evidence_root.mkdir(parents=True, exist_ok=True)
    bundles_dir = evidence_root / "bundles"
    bundles_dir.mkdir(parents=True, exist_ok=True)

    spec = _load_spec(spec_path)
    spec_hash = _sha256_bytes(_canonical_json(spec).encode("utf-8"))
    key = f"{campaign_tick_id}:{spec_hash}"

    index_path = evidence_root / _INDEX_FILE_NAME
    index = _read_index(index_path)

    reused = key in index
    if reused:
        bundle_id = index[key]
    else:
        bundle_id = _sha256_bytes(key.encode("utf-8"))[:20]
        index[key] = bundle_id
        _write_index(index_path, index)

    return ControlRunSubmission(
        spec_ref=spec_path,
        spec_hash=spec_hash,
        campaign_tick_id=campaign_tick_id,
        evidence_store=evidence_root,
        bundle_id=bundle_id,
        reused_existing_bundle=reused,
        spec=spec,
    )


def publishEvidenceBundle(
    submission: ControlRunSubmission,
    seed_metrics: Mapping[int, float],
    log_files: list[Path | str],
) -> Path:
    bundle_path = submission.evidence_store / "bundles" / f"{submission.bundle_id}.json"
    existing_bundle: dict[str, Any] | None = None
    if bundle_path.exists():
        existing_bundle = json.loads(bundle_path.read_text(encoding="utf-8"))

    env_manifest = _environment_manifest()
    env_issues = _validate_environment(submission.spec, env_manifest)

    ordered_seeds = [int(seed) for seed in submission.spec["seed_set"]]
    existing_by_seed: dict[int, float] = {}
    if existing_bundle:
        for row in existing_bundle.get("seed_metrics", []):
            seed = int(row.get("seed"))
            value = float(row.get("value", float("nan")))
            existing_by_seed[seed] = value

    metric_values: list[float] = []
    metric_rows: list[dict[str, Any]] = []
    for seed in ordered_seeds:
        value = float(seed_metrics.get(seed, existing_by_seed.get(seed, float("nan"))))
        metric_rows.append({"seed": seed, "value": value})
        metric_values.append(value)

    checksum_map: dict[str, str] = {}
    if existing_bundle:
        checksum_map.update(existing_bundle.get("checksums", {}))
    checksum_map[str(submission.spec_ref)] = _sha256_file(submission.spec_ref)

    log_paths: list[str] = list(existing_bundle.get("artifacts", {}).get("logs", [])) if existing_bundle else []
    for log in log_files:
        path = Path(log)
        if str(path) not in log_paths:
            log_paths.append(str(path))
        if path.exists():
            checksum_map[str(path)] = _sha256_file(path)

    bundle = {
        "schema_version": CONTROL_EVIDENCE_SCHEMA_VERSION,
        "bundle_id": submission.bundle_id,
        "campaign_tick_id": submission.campaign_tick_id,
        "created_at": existing_bundle.get("created_at", _utc_now_iso()) if existing_bundle else _utc_now_iso(),
        "updated_at": _utc_now_iso(),
        "spec_ref": str(submission.spec_ref),
        "spec_hash": submission.spec_hash,
        "spec_snapshot": {
            "spec_id": submission.spec["spec_id"],
            "dataset_id": submission.spec["dataset_id"],
            "tokenizer_id": submission.spec["tokenizer_id"],
            "seed_set": ordered_seeds,
            "metric": submission.spec["metric"],
            "command_template": submission.spec["command_template"],
        },
        "env_manifest": env_manifest,
        "env_issues": env_issues,
        "seed_metrics": metric_rows,
        "primary_metric": _safe_mean(metric_values),
        "checksums": checksum_map,
        "artifacts": {"logs": log_paths},
    }
    bundle_path.write_text(json.dumps(bundle, sort_keys=True, indent=2), encoding="utf-8")
    return bundle_path


def evaluateControlTrust(bundle_path: Path | str, spec_ref: Path | str) -> dict[str, Any]:
    bundle_file = Path(bundle_path)
    spec_path = Path(spec_ref)
    bundle = json.loads(bundle_file.read_text(encoding="utf-8"))
    spec = _load_spec(spec_path)

    reasons: list[str] = []

    if bundle.get("schema_version") != CONTROL_EVIDENCE_SCHEMA_VERSION:
        reasons.append(
            "schema mismatch: "
            f"bundle={bundle.get('schema_version')} expected={CONTROL_EVIDENCE_SCHEMA_VERSION}"
        )

    expected_spec_hash = _sha256_bytes(_canonical_json(spec).encode("utf-8"))
    if bundle.get("spec_hash") != expected_spec_hash:
        reasons.append("spec hash mismatch")

    snapshot = bundle.get("spec_snapshot", {})
    if snapshot.get("dataset_id") != spec.get("dataset_id"):
        reasons.append("dataset mismatch versus spec fingerprint")
    if snapshot.get("tokenizer_id") != spec.get("tokenizer_id"):
        reasons.append("tokenizer mismatch versus spec fingerprint")

    checksums = bundle.get("checksums", {})
    if not isinstance(checksums, dict):
        reasons.append("checksum map is invalid")
        checksums = {}

    required_paths = {str(spec_path)}
    for log_path in bundle.get("artifacts", {}).get("logs", []):
        required_paths.add(str(log_path))

    for required in sorted(required_paths):
        if required not in checksums:
            reasons.append(f"missing checksum for artifact {required}")

    for path_str, expected_hash in sorted(checksums.items()):
        path = Path(path_str)
        if not path.exists():
            reasons.append(f"corrupt or missing artifact {path_str}")
            continue
        actual_hash = _sha256_file(path)
        if actual_hash != expected_hash:
            reasons.append(f"checksum mismatch for artifact {path_str}")

    env_manifest = bundle.get("env_manifest", {})
    reasons.extend(_validate_environment(spec, env_manifest))

    rows = bundle.get("seed_metrics", [])
    spec_seeds = [int(seed) for seed in spec.get("seed_set", [])]
    observed_by_seed: dict[int, float] = {}
    for row in rows:
        seed = int(row.get("seed"))
        value = float(row.get("value", float("nan")))
        observed_by_seed[seed] = value

    missing_seeds = [seed for seed in spec_seeds if seed not in observed_by_seed]
    if missing_seeds:
        reasons.append(f"missing seed metrics for seeds {missing_seeds}")

    metric_values: list[float] = []
    for seed in spec_seeds:
        value = observed_by_seed.get(seed, float("nan"))
        if not math.isfinite(value):
            reasons.append(f"invalid measurement value for seed {seed}")
        metric_values.append(value)

    if metric_values and all(math.isfinite(v) for v in metric_values):
        drift = max(metric_values) - min(metric_values)
        max_drift = float(spec["metric"].get("max_drift", 0.0))
        if drift - max_drift > 1e-12:
            reasons.append(
                "metric drift beyond tolerance: "
                f"drift={drift:.6f} tolerance={max_drift:.6f}"
            )

    primary_metric = bundle.get("primary_metric")
    if not isinstance(primary_metric, (int, float)) or not math.isfinite(float(primary_metric)):
        reasons.append("invalid-measurement: aggregate metric is empty or NaN")

    state = "trusted" if not reasons else "untrusted"
    return {
        "schema_version": CONTROL_EVIDENCE_SCHEMA_VERSION,
        "bundle_id": bundle.get("bundle_id"),
        "state": state,
        "reasons": reasons,
        "primary_metric": primary_metric,
    }
