from __future__ import annotations

import argparse
import hashlib
import json
import math
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
DEFAULT_CONTROL_TENSOR_NAME_PATTERNS = (
    "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights"
)
DEFAULT_TRUSTED_PARENT_REGISTRY_PATH = REPO_ROOT / "fastest/source/trusted_parent_lineage_registry.json"
DEFAULT_TRUSTED_RUNNER_ATTESTATION_REGISTRY_PATH = (
    REPO_ROOT / "fastest/source/trusted_runner_attestation_registry.json"
)
DEFAULT_TRUSTED_OPERATOR_RISK_ACCEPTANCE_REGISTRY_PATH = (
    REPO_ROOT / "fastest/source/trusted_operator_risk_acceptance_registry.json"
)
TRUST_RECEIPT_SCHEMA = "parameter-golf-trust-receipt/v1"
TRUST_RECEIPT_ISSUER = "parameter-golf-model-factory-trust-authority"
TRUST_RECEIPT_SIGNATURE_PREFIX = "local-signed-bundle:"
TRUST_RECEIPT_CRYPTO_SIGNATURE_PREFIX = "cryptographic-signed-bundle:"
FINAL_BPB_RE = re.compile(
    r"final_int8_zlib_roundtrip_exact\s+val_loss:(?P<loss>[0-9.]+)\s+val_bpb:(?P<bpb>[0-9.]+)"
)


def _authority_boundary_untrusted(authority_boundary: str) -> bool:
    lowered = authority_boundary.lower()
    return lowered.startswith(("local-", "local://", "local-task-store://", "file://"))


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
    "CONTROL_TENSOR_NAME_PATTERNS": DEFAULT_CONTROL_TENSOR_NAME_PATTERNS,
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
CPU_SUBSET_SWIGLU_ENV_KEYS = frozenset(
    {
        "ACTIVATION_MODE",
        "SWIGLU_CLAMP_ENABLED",
        "SWIGLU_CLAMP_MODE",
        "SWIGLU_LINEAR_CLAMP_MIN",
        "SWIGLU_LINEAR_CLAMP_MAX",
        "SWIGLU_GATE_CLAMP_MAX",
    }
)
ALLOWED_ACTIVATION_MODES = frozenset({"relu2", "swiglu"})
ALLOWED_SWIGLU_CLAMP_MODES = frozenset({"disabled", "deepseek_v4"})
ALLOWED_TRUST_GATE_MODES = frozenset({"strict", "operator-approved-fallback"})
DEFAULT_SWIGLU_LINEAR_CLAMP_MIN = -10.0
DEFAULT_SWIGLU_LINEAR_CLAMP_MAX = 10.0
DEFAULT_SWIGLU_GATE_CLAMP_MAX = 10.0
MIN_MUON_UPDATE_SCALE = 1e-6
MAX_MUON_UPDATE_SCALE = 1.0
MIN_MUON_MOMENTUM = 0.0
MAX_MUON_MOMENTUM = 1.0
MIN_MUON_BACKEND_STEPS = 1
MAX_MUON_BACKEND_STEPS = 4096
MIN_MUON_MOMENTUM_WARMUP_STEPS = 0
MUON_SELECTOR_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]+$")
ALLOWED_ADAMW_EXEMPTION_SELECTORS = frozenset(
    {
        "attn_scale",
        "attn_scales",
        "mlp_scale",
        "mlp_scales",
        "resid_mix",
        "resid_mixes",
        "q_gain",
        "skip_weight",
        "skip_weights",
        "tok_embeddings",
        "token_embeddings",
        "embedding",
        "embeddings",
        "lm_head",
        "head",
        "rmsnorm",
    }
)

CPU_SUBSET_ENV_OVERRIDE_ALLOWLIST = (
    frozenset(DEFAULT_CPU_ENV.keys())
    | frozenset({"SEED"})
    | CPU_SUBSET_MUON_ENV_KEYS
    | CPU_SUBSET_SWIGLU_ENV_KEYS
)
CPU_SUBSET_SERIALIZED_FACTOR_ALLOWLIST = frozenset(
    {
        "ACTIVATION_MODE",
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
        "SWIGLU_CLAMP_ENABLED",
        "SWIGLU_CLAMP_MODE",
        "SWIGLU_GATE_CLAMP_MAX",
        "SWIGLU_LINEAR_CLAMP_MAX",
        "SWIGLU_LINEAR_CLAMP_MIN",
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
    preflight = _preflight_validate_candidate(candidate)
    if preflight["errors"]:
        raise ValueError("Invalid candidate config cannot be materialized into environment")
    resolved = preflight["resolved"]
    env["ACTIVATION_MODE"] = str(resolved["activationMode"])
    env["SWIGLU_CLAMP_ENABLED"] = "1" if bool(resolved["swigluClampEnabled"]) else "0"
    env["SWIGLU_CLAMP_MODE"] = str(resolved["swigluClampMode"])
    env["SWIGLU_LINEAR_CLAMP_MIN"] = str(resolved["swigluLinearClampMin"])
    env["SWIGLU_LINEAR_CLAMP_MAX"] = str(resolved["swigluLinearClampMax"])
    env["SWIGLU_GATE_CLAMP_MAX"] = str(resolved["swigluGateClampMax"])

    def _stringify_preserving_input(raw_value: Any, fallback: Any) -> str:
        if isinstance(raw_value, str):
            return raw_value.strip()
        if isinstance(raw_value, (int, float)) and not isinstance(raw_value, bool):
            return str(raw_value)
        return str(fallback)

    env["MATRIX_LR"] = _stringify_preserving_input(
        _candidate_value(candidate, "muonUpdateScale", "MATRIX_LR"),
        resolved["matrixLr"],
    )
    env["MUON_BACKEND_STEPS"] = _stringify_preserving_input(
        _candidate_value(candidate, "muonBackendSteps", "MUON_BACKEND_STEPS"),
        resolved["muonBackendSteps"],
    )
    env["MUON_MOMENTUM"] = _stringify_preserving_input(
        _candidate_value(candidate, "muonMomentum", "MUON_MOMENTUM"),
        resolved["muonMomentum"],
    )
    env["MUON_MOMENTUM_WARMUP_START"] = _stringify_preserving_input(
        _candidate_value(candidate, "muonMomentumWarmupStart", "MUON_MOMENTUM_WARMUP_START"),
        resolved["muonMomentumWarmupStart"],
    )
    env["MUON_MOMENTUM_WARMUP_STEPS"] = _stringify_preserving_input(
        _candidate_value(candidate, "muonMomentumWarmupSteps", "MUON_MOMENTUM_WARMUP_STEPS"),
        resolved["muonMomentumWarmupSteps"],
    )
    raw_exemptions = _candidate_value(candidate, "adamwExemptionSelectors", "CONTROL_TENSOR_NAME_PATTERNS")
    if isinstance(raw_exemptions, str):
        env["CONTROL_TENSOR_NAME_PATTERNS"] = raw_exemptions.strip()
    elif isinstance(raw_exemptions, list):
        env["CONTROL_TENSOR_NAME_PATTERNS"] = ",".join(resolved["adamwExemptionSelectors"])
    else:
        env["CONTROL_TENSOR_NAME_PATTERNS"] = ",".join(resolved["adamwExemptionSelectors"])
    return env


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value in {0, 0.0}:
            return False
        if value in {1, 1.0}:
            return True
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return None


def _candidate_value(candidate: dict[str, Any], top_key: str, env_key: str) -> Any:
    if top_key in candidate:
        return candidate[top_key]
    overrides = candidate.get("env")
    if isinstance(overrides, dict):
        return overrides.get(env_key)
    return None


def _coerce_finite_float(value: Any, field_name: str, errors: list[str], default: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        errors.append(f"invalid-{field_name}")
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        errors.append(f"invalid-{field_name}")
        return default
    if not math.isfinite(number):
        errors.append(f"non-finite-{field_name}")
        return default
    return number


def _coerce_int(value: Any, field_name: str, errors: list[str], default: int) -> int:
    if value is None:
        return default
    try:
        if isinstance(value, bool):
            raise ValueError
        number = float(value)
    except (TypeError, ValueError):
        errors.append(f"invalid-{field_name}")
        return default
    if not math.isfinite(number) or not number.is_integer():
        errors.append(f"invalid-{field_name}")
        return default
    return int(number)


def _preflight_validate_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    raw_trust_gate_mode = candidate.get("trustGateMode")
    trust_gate_mode = str(raw_trust_gate_mode or "strict").strip().lower()
    if trust_gate_mode not in ALLOWED_TRUST_GATE_MODES:
        errors.append("invalid-trustGateMode")
    operator_approval_ref = candidate.get("operatorApprovalRef")
    if trust_gate_mode == "operator-approved-fallback":
        if not isinstance(operator_approval_ref, str) or not operator_approval_ref.strip():
            errors.append("missing-operatorApprovalRef")
        # FAST-55 security policy: local-signature fallback is disallowed.
        errors.append("disallowed-trustGateFallback")
    raw_activation_mode = _candidate_value(candidate, "activationMode", "ACTIVATION_MODE")
    activation_mode = str(raw_activation_mode or "relu2").strip().lower()
    if activation_mode not in ALLOWED_ACTIVATION_MODES:
        errors.append("invalid-activationMode")

    raw_clamp_enabled = _candidate_value(candidate, "swigluClampEnabled", "SWIGLU_CLAMP_ENABLED")
    clamp_enabled = _coerce_bool(raw_clamp_enabled)
    if clamp_enabled is None:
        if raw_clamp_enabled is None:
            clamp_enabled = False
        else:
            errors.append("invalid-swigluClampEnabled")
            clamp_enabled = False

    raw_clamp_mode = _candidate_value(candidate, "swigluClampMode", "SWIGLU_CLAMP_MODE")
    if raw_clamp_mode is None:
        clamp_mode = "deepseek_v4" if clamp_enabled else "disabled"
    else:
        clamp_mode = str(raw_clamp_mode).strip().lower()
    if clamp_mode not in ALLOWED_SWIGLU_CLAMP_MODES:
        errors.append("invalid-swigluClampMode")
    if clamp_mode == "disabled":
        clamp_enabled = False

    linear_clamp_min = _coerce_finite_float(
        _candidate_value(candidate, "swigluLinearClampMin", "SWIGLU_LINEAR_CLAMP_MIN"),
        "swigluLinearClampMin",
        errors,
        DEFAULT_SWIGLU_LINEAR_CLAMP_MIN,
    )
    linear_clamp_max = _coerce_finite_float(
        _candidate_value(candidate, "swigluLinearClampMax", "SWIGLU_LINEAR_CLAMP_MAX"),
        "swigluLinearClampMax",
        errors,
        DEFAULT_SWIGLU_LINEAR_CLAMP_MAX,
    )
    gate_clamp_max = _coerce_finite_float(
        _candidate_value(candidate, "swigluGateClampMax", "SWIGLU_GATE_CLAMP_MAX"),
        "swigluGateClampMax",
        errors,
        DEFAULT_SWIGLU_GATE_CLAMP_MAX,
    )

    if linear_clamp_min > linear_clamp_max:
        errors.append("invalid-swigluLinearClampRange")
    if gate_clamp_max <= 0:
        errors.append("invalid-swigluGateClampMax")
    if clamp_enabled and activation_mode != "swiglu":
        errors.append("swigluClampRequiresSwiGLU")

    matrix_lr = _coerce_finite_float(
        _candidate_value(candidate, "muonUpdateScale", "MATRIX_LR"),
        "matrixLr",
        errors,
        float(DEFAULT_CPU_ENV["MATRIX_LR"]) if "MATRIX_LR" in DEFAULT_CPU_ENV else 0.04,
    )
    if not (MIN_MUON_UPDATE_SCALE <= matrix_lr <= MAX_MUON_UPDATE_SCALE):
        errors.append("invalid-matrixLr")

    raw_muon_backend_steps = _candidate_value(candidate, "muonBackendSteps", "MUON_BACKEND_STEPS")
    muon_backend_steps = _coerce_int(
        raw_muon_backend_steps,
        "muonBackendSteps",
        errors,
        int(DEFAULT_CPU_ENV["MUON_BACKEND_STEPS"]) if "MUON_BACKEND_STEPS" in DEFAULT_CPU_ENV else 5,
    )
    if not (MIN_MUON_BACKEND_STEPS <= muon_backend_steps <= MAX_MUON_BACKEND_STEPS):
        errors.append("invalid-muonBackendSteps")

    muon_momentum = _coerce_finite_float(
        _candidate_value(candidate, "muonMomentum", "MUON_MOMENTUM"),
        "muonMomentum",
        errors,
        float(DEFAULT_CPU_ENV["MUON_MOMENTUM"]) if "MUON_MOMENTUM" in DEFAULT_CPU_ENV else 0.95,
    )
    if not (MIN_MUON_MOMENTUM < muon_momentum < MAX_MUON_MOMENTUM):
        errors.append("invalid-muonMomentum")

    muon_momentum_warmup_start = _coerce_finite_float(
        _candidate_value(candidate, "muonMomentumWarmupStart", "MUON_MOMENTUM_WARMUP_START"),
        "muonMomentumWarmupStart",
        errors,
        float(DEFAULT_CPU_ENV["MUON_MOMENTUM_WARMUP_START"])
        if "MUON_MOMENTUM_WARMUP_START" in DEFAULT_CPU_ENV
        else 0.85,
    )
    if not (0.0 <= muon_momentum_warmup_start <= muon_momentum):
        errors.append("invalid-muonMomentumWarmupStart")

    raw_muon_momentum_warmup_steps = _candidate_value(
        candidate,
        "muonMomentumWarmupSteps",
        "MUON_MOMENTUM_WARMUP_STEPS",
    )
    muon_momentum_warmup_steps = _coerce_int(
        raw_muon_momentum_warmup_steps,
        "muonMomentumWarmupSteps",
        errors,
        int(DEFAULT_CPU_ENV["MUON_MOMENTUM_WARMUP_STEPS"])
        if "MUON_MOMENTUM_WARMUP_STEPS" in DEFAULT_CPU_ENV
        else 500,
    )
    if muon_momentum_warmup_steps < MIN_MUON_MOMENTUM_WARMUP_STEPS:
        errors.append("invalid-muonMomentumWarmupSteps")
    if (
        raw_muon_backend_steps is not None
        and raw_muon_momentum_warmup_steps is not None
        and muon_backend_steps <= 0
        and muon_momentum_warmup_steps > 0
    ):
        errors.append("invalid-muonMomentumWarmupSchedule")

    raw_control_patterns = _candidate_value(
        candidate,
        "adamwExemptionSelectors",
        "CONTROL_TENSOR_NAME_PATTERNS",
    )
    if raw_control_patterns is None:
        control_patterns = [
            pattern
            for pattern in str(DEFAULT_CPU_ENV.get("CONTROL_TENSOR_NAME_PATTERNS", DEFAULT_CONTROL_TENSOR_NAME_PATTERNS)).split(",")
            if pattern
        ]
    elif isinstance(raw_control_patterns, str):
        control_patterns = [pattern.strip() for pattern in raw_control_patterns.split(",") if pattern.strip()]
    elif isinstance(raw_control_patterns, list):
        control_patterns = [
            str(pattern).strip()
            for pattern in raw_control_patterns
            if isinstance(pattern, str) and str(pattern).strip()
        ]
        if len(control_patterns) != len(raw_control_patterns):
            errors.append("invalid-controlTensorNamePatterns")
    else:
        errors.append("invalid-controlTensorNamePatterns")
        control_patterns = []
    if any(not MUON_SELECTOR_PATTERN.fullmatch(pattern) for pattern in control_patterns):
        errors.append("invalid-controlTensorNamePatterns")
    if any(pattern not in ALLOWED_ADAMW_EXEMPTION_SELECTORS for pattern in control_patterns):
        errors.append("invalid-controlTensorNamePatterns")

    resolved = {
        "trustGateMode": trust_gate_mode,
        "operatorApprovalRef": operator_approval_ref if isinstance(operator_approval_ref, str) else None,
        "activationMode": activation_mode,
        "swigluClampEnabled": clamp_enabled,
        "swigluClampMode": clamp_mode,
        "swigluLinearClampMin": linear_clamp_min,
        "swigluLinearClampMax": linear_clamp_max,
        "swigluGateClampMax": gate_clamp_max,
        "matrixLr": matrix_lr,
        "muonBackendSteps": muon_backend_steps,
        "muonMomentum": muon_momentum,
        "muonMomentumWarmupStart": muon_momentum_warmup_start,
        "muonMomentumWarmupSteps": muon_momentum_warmup_steps,
        "adamwExemptionSelectors": control_patterns,
    }
    return {"errors": sorted(set(errors)), "resolved": resolved}


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


def _trusted_registry_policy_error(payload: dict[str, Any]) -> str | None:
    trust_root_policy = payload.get("trustRootPolicy")
    if not isinstance(trust_root_policy, dict):
        return "trust-root-policy-missing"
    environment = trust_root_policy.get("environment")
    if environment != "production":
        return "trust-root-policy-non-production"
    if trust_root_policy.get("allowBypass") is not False:
        return "trust-root-policy-bypass-disallowed"
    trusted_roots = trust_root_policy.get("trustedRoots")
    if not isinstance(trusted_roots, list) or not trusted_roots:
        return "trust-root-policy-missing-trusted-roots"
    if any(not isinstance(root, str) or not root for root in trusted_roots):
        return "trust-root-policy-invalid-trusted-roots"
    return None


def _trust_receipt_error(
    entry: dict[str, Any],
    expected_subject: str,
    trusted_roots: set[str] | None = None,
    trust_root_policy: dict[str, Any] | None = None,
    operator_acceptance_registry_path: Path = DEFAULT_TRUSTED_OPERATOR_RISK_ACCEPTANCE_REGISTRY_PATH,
    trust_gate_mode: str = "strict",
    operator_approval_ref: str | None = None,
) -> str | None:
    def runtime_operator_acceptance_error(residual_risk: dict[str, Any]) -> str | None:
        accepted_by = residual_risk.get("acceptedBy")
        if not isinstance(accepted_by, str) or not accepted_by.startswith("fastest-task-store://"):
            return "receipt-residual-risk-acceptance-invalid-approver"
        acceptance_ref = residual_risk.get("acceptanceRef")
        if not isinstance(acceptance_ref, str) or not acceptance_ref:
            return "receipt-residual-risk-acceptance-ref-missing"
        if not operator_acceptance_registry_path.exists():
            return "receipt-residual-risk-acceptance-registry-unavailable"
        try:
            payload = json.loads(operator_acceptance_registry_path.read_text())
        except (OSError, ValueError):
            return "receipt-residual-risk-acceptance-registry-unavailable"
        if not isinstance(payload, dict):
            return "receipt-residual-risk-acceptance-registry-invalid"
        authority_boundary = payload.get("authorityBoundary")
        if not isinstance(authority_boundary, str) or not authority_boundary:
            return "receipt-residual-risk-acceptance-registry-boundary-invalid"
        if _authority_boundary_untrusted(authority_boundary):
            return "receipt-residual-risk-acceptance-registry-boundary-untrusted"
        accepted = payload.get("acceptedResidualRisks")
        if not isinstance(accepted, dict):
            return "receipt-residual-risk-acceptance-registry-invalid"
        runtime_entry = accepted.get(acceptance_ref)
        if not isinstance(runtime_entry, dict):
            return "receipt-residual-risk-acceptance-unresolved"
        for field in ("scope", "acceptedBy", "monitoringOwner", "expiresAt"):
            if runtime_entry.get(field) != residual_risk.get(field):
                return "receipt-residual-risk-acceptance-mismatch"
        return None

    def residual_risk_acceptance_error() -> str | None:
        if not isinstance(trust_root_policy, dict):
            return "receipt-residual-risk-acceptance-missing"
        residual_risk = trust_root_policy.get("residualRiskAcceptance")
        if not isinstance(residual_risk, dict):
            return "receipt-residual-risk-acceptance-missing"
        if residual_risk.get("scope") != "cpu-subset":
            return "receipt-residual-risk-acceptance-invalid-scope"
        accepted_by = residual_risk.get("acceptedBy")
        if not isinstance(accepted_by, str) or not accepted_by:
            return "receipt-residual-risk-acceptance-invalid-approver"
        monitoring_owner = residual_risk.get("monitoringOwner")
        if not isinstance(monitoring_owner, str) or not monitoring_owner:
            return "receipt-residual-risk-acceptance-monitoring-missing"
        expires_at_raw = residual_risk.get("expiresAt")
        if not isinstance(expires_at_raw, str) or not expires_at_raw:
            return "receipt-residual-risk-acceptance-expiry-invalid"
        expires_at_normalized = expires_at_raw.replace("Z", "+00:00")
        try:
            expires_at = datetime.fromisoformat(expires_at_normalized)
        except ValueError:
            return "receipt-residual-risk-acceptance-expiry-invalid"
        if expires_at.tzinfo is None:
            return "receipt-residual-risk-acceptance-expiry-invalid"
        if expires_at.astimezone(timezone.utc) <= datetime.now(timezone.utc):
            return "receipt-residual-risk-acceptance-expired"
        authority_binding = trust_root_policy.get("authorityBinding")
        if not isinstance(authority_binding, dict):
            return "receipt-authority-binding-missing"
        authority_boundary = authority_binding.get("authorityBoundary")
        if not isinstance(authority_boundary, str) or not authority_boundary:
            return "receipt-authority-binding-boundary-invalid"
        if _authority_boundary_untrusted(authority_boundary):
            return "receipt-authority-binding-boundary-untrusted"
        registry_ref = authority_binding.get("registryRef")
        if not isinstance(registry_ref, str) or not registry_ref:
            return "receipt-authority-binding-registry-ref-invalid"
        validated_at_raw = authority_binding.get("validatedAt")
        if not isinstance(validated_at_raw, str) or not validated_at_raw:
            return "receipt-authority-binding-validated-at-invalid"
        validated_at_normalized = validated_at_raw.replace("Z", "+00:00")
        try:
            validated_at = datetime.fromisoformat(validated_at_normalized)
        except ValueError:
            return "receipt-authority-binding-validated-at-invalid"
        if validated_at.tzinfo is None:
            return "receipt-authority-binding-validated-at-invalid"
        if validated_at.astimezone(timezone.utc) > datetime.now(timezone.utc):
            return "receipt-authority-binding-validated-at-invalid"
        bound_trusted_roots = authority_binding.get("trustedRoots")
        if (
            not isinstance(bound_trusted_roots, list)
            or not bound_trusted_roots
            or any(not isinstance(root, str) or not root for root in bound_trusted_roots)
        ):
            return "receipt-authority-binding-trusted-roots-invalid"
        if trusted_root not in set(bound_trusted_roots):
            return "receipt-authority-binding-trusted-root-mismatch"
        runtime_acceptance_error = runtime_operator_acceptance_error(residual_risk)
        if runtime_acceptance_error is not None:
            return runtime_acceptance_error
        return None

    receipt = entry.get("receipt")
    if not isinstance(receipt, dict):
        return "receipt-missing"
    if receipt.get("schema") != TRUST_RECEIPT_SCHEMA:
        return "receipt-schema-invalid"
    if receipt.get("issuedBy") != TRUST_RECEIPT_ISSUER:
        return "receipt-issuer-untrusted"
    if receipt.get("subject") != expected_subject:
        return "receipt-subject-mismatch"
    signature = receipt.get("signature")
    if not isinstance(signature, str):
        return "receipt-signature-invalid"
    signature_subject: str
    local_signature_mode = False
    if signature.startswith(TRUST_RECEIPT_SIGNATURE_PREFIX):
        local_signature_mode = True
        signature_subject = signature[len(TRUST_RECEIPT_SIGNATURE_PREFIX) :]
    elif signature.startswith(TRUST_RECEIPT_CRYPTO_SIGNATURE_PREFIX):
        signature_subject = signature[len(TRUST_RECEIPT_CRYPTO_SIGNATURE_PREFIX) :]
    else:
        return "receipt-signature-invalid"
    if signature_subject != expected_subject:
        return "receipt-signature-subject-mismatch"
    trusted_root = receipt.get("trustedRoot")
    if not isinstance(trusted_root, str) or not trusted_root:
        return "receipt-trusted-root-missing"
    if trusted_roots is not None and trusted_root not in trusted_roots:
        return "receipt-trusted-root-untrusted"
    if local_signature_mode:
        return "receipt-local-signature-disallowed"
    cryptographic_verification = entry.get("cryptographicVerification")
    if isinstance(cryptographic_verification, dict):
        authority_boundary = cryptographic_verification.get("authorityBoundary")
        if not isinstance(authority_boundary, str) or not authority_boundary:
            return "receipt-cryptographic-authority-boundary-missing"
        if _authority_boundary_untrusted(authority_boundary):
            return "receipt-cryptographic-authority-boundary-untrusted"
        verification_ref = cryptographic_verification.get("verificationRef")
        if not isinstance(verification_ref, str) or not verification_ref:
            return "receipt-cryptographic-verification-ref-missing"
        if not bool(cryptographic_verification.get("verified")):
            return "receipt-cryptographic-unverified"
        verified_at_raw = cryptographic_verification.get("verifiedAt")
        if not isinstance(verified_at_raw, str) or not verified_at_raw:
            return "receipt-cryptographic-verified-at-invalid"
        verified_at_normalized = verified_at_raw.replace("Z", "+00:00")
        try:
            verified_at = datetime.fromisoformat(verified_at_normalized)
        except ValueError:
            return "receipt-cryptographic-verified-at-invalid"
        if verified_at.tzinfo is None:
            return "receipt-cryptographic-verified-at-invalid"
        return None
    risk_error = residual_risk_acceptance_error()
    if risk_error is not None:
        return risk_error
    return None


def _load_trusted_parent_registry(registry_path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not registry_path.exists():
        return None, "registry-unavailable"
    try:
        payload = json.loads(registry_path.read_text())
    except (OSError, ValueError):
        return None, "registry-unavailable"
    if not isinstance(payload, dict):
        return None, "registry-invalid"
    trust_root_policy_error = _trusted_registry_policy_error(payload)
    if trust_root_policy_error is not None:
        return None, trust_root_policy_error
    return payload, None


def _trusted_roots_from_registry(registry: dict[str, Any]) -> set[str]:
    trust_root_policy = registry.get("trustRootPolicy")
    if not isinstance(trust_root_policy, dict):
        return set()
    trusted_roots = trust_root_policy.get("trustedRoots")
    if not isinstance(trusted_roots, list):
        return set()
    return {root for root in trusted_roots if isinstance(root, str) and root}


def _resolve_runner_attestation(
    *,
    attestation_ref: str | None,
    run_id: str,
    registry_path: Path,
    trust_gate_mode: str = "strict",
    operator_approval_ref: str | None = None,
) -> dict[str, Any]:
    if not attestation_ref:
        return {
            "status": "unresolved",
            "reasonCode": "attestation-missing",
            "attestationRegistryRef": None,
        }
    registry, registry_error = _load_trusted_parent_registry(registry_path)
    if registry is None:
        reason_code = "trusted-attestation-registry-unavailable"
        if registry_error == "registry-invalid":
            reason_code = "trusted-attestation-registry-invalid"
        elif registry_error and registry_error.startswith("trust-root-policy-"):
            reason_code = f"trusted-attestation-registry-invalid-{registry_error}"
        return {
            "status": "unresolved",
            "reasonCode": reason_code,
            "attestationRegistryRef": None,
        }
    runner_attestations = registry.get("runnerAttestations")
    if not isinstance(runner_attestations, dict):
        return {
            "status": "unresolved",
            "reasonCode": "trusted-attestation-registry-invalid-runner-attestations",
            "attestationRegistryRef": None,
        }
    entry = runner_attestations.get(attestation_ref)
    if not isinstance(entry, dict):
        return {
            "status": "unresolved",
            "reasonCode": "attestation-unresolved",
            "attestationRegistryRef": f"trusted-runner-attestation:{attestation_ref}",
        }
    expected_run_id = entry.get("runId")
    if not isinstance(expected_run_id, str) or not expected_run_id:
        return {
            "status": "unresolved",
            "reasonCode": "attestation-missing-run-id",
            "attestationRegistryRef": f"trusted-runner-attestation:{attestation_ref}",
        }
    if expected_run_id != run_id:
        return {
            "status": "unresolved",
            "reasonCode": "attestation-run-id-mismatch",
            "attestationRegistryRef": f"trusted-runner-attestation:{attestation_ref}",
        }
    source = entry.get("source")
    if source != "cpu-subset-runner":
        return {
            "status": "unresolved",
            "reasonCode": "attestation-untrusted-source",
            "attestationRegistryRef": f"trusted-runner-attestation:{attestation_ref}",
        }
    result_class = entry.get("resultClass")
    if result_class != "cpu-subset":
        return {
            "status": "unresolved",
            "reasonCode": "attestation-result-class-invalid",
            "attestationRegistryRef": f"trusted-runner-attestation:{attestation_ref}",
        }
    verification_class = entry.get("verificationClass")
    if verification_class != "cpu-subset":
        return {
            "status": "unresolved",
            "reasonCode": "attestation-verification-class-invalid",
            "attestationRegistryRef": f"trusted-runner-attestation:{attestation_ref}",
        }
    expected_attestation_ref = entry.get("attestationRef")
    if expected_attestation_ref != attestation_ref:
        return {
            "status": "unresolved",
            "reasonCode": "attestation-ref-mismatch",
            "attestationRegistryRef": f"trusted-runner-attestation:{attestation_ref}",
        }
    if not bool(entry.get("provenanceVerified")):
        return {
            "status": "unresolved",
            "reasonCode": "attestation-provenance-unverified",
            "attestationRegistryRef": f"trusted-runner-attestation:{attestation_ref}",
        }
    trust_root_policy = registry.get("trustRootPolicy")
    trusted_roots = _trusted_roots_from_registry(registry)
    receipt_error = _trust_receipt_error(
        entry,
        attestation_ref,
        trusted_roots,
        trust_root_policy,
        DEFAULT_TRUSTED_OPERATOR_RISK_ACCEPTANCE_REGISTRY_PATH,
        trust_gate_mode,
        operator_approval_ref,
    )
    if receipt_error is not None:
        return {
            "status": "unresolved",
            "reasonCode": f"attestation-{receipt_error}",
            "attestationRegistryRef": f"trusted-runner-attestation:{attestation_ref}",
        }
    return {
        "status": "resolved",
        "reasonCode": "attestation-resolved",
        "attestationRegistryRef": f"trusted-runner-attestation:{attestation_ref}",
    }


def _resolve_parent_lineage(
    *,
    parent_experiment_id: str | None,
    registry_path: Path,
    trust_gate_mode: str = "strict",
    operator_approval_ref: str | None = None,
) -> dict[str, Any]:
    if not parent_experiment_id:
        return {
            "status": "unresolved",
            "reasonCode": "missing-parent-experiment-id",
            "lineageResolutionRef": None,
            "parentFrontierId": None,
        }
    registry, registry_error = _load_trusted_parent_registry(registry_path)
    if registry is None:
        reason_code = "trusted-registry-unavailable"
        if registry_error == "registry-invalid":
            reason_code = "trusted-registry-invalid"
        elif registry_error and registry_error.startswith("trust-root-policy-"):
            reason_code = f"trusted-registry-invalid-{registry_error}"
        return {
            "status": "unresolved",
            "reasonCode": reason_code,
            "lineageResolutionRef": None,
            "parentFrontierId": None,
        }
    parent_lineage = registry.get("parentLineage")
    if not isinstance(parent_lineage, dict):
        return {
            "status": "unresolved",
            "reasonCode": "trusted-registry-invalid-parent-lineage",
            "lineageResolutionRef": None,
            "parentFrontierId": None,
        }
    entry = parent_lineage.get(parent_experiment_id)
    if not isinstance(entry, dict):
        return {
            "status": "unresolved",
            "reasonCode": "lineage-unresolved",
            "lineageResolutionRef": f"trusted-parent-lineage:{parent_experiment_id}",
            "parentFrontierId": None,
        }
    parent_frontier_id = entry.get("parentFrontierId")
    if not isinstance(parent_frontier_id, str) or not parent_frontier_id:
        return {
            "status": "unresolved",
            "reasonCode": "lineage-missing-parent-frontier-id",
            "lineageResolutionRef": f"trusted-parent-lineage:{parent_experiment_id}",
            "parentFrontierId": None,
        }
    if not bool(entry.get("provenanceVerified")):
        return {
            "status": "unresolved",
            "reasonCode": "lineage-provenance-unverified",
            "lineageResolutionRef": f"trusted-parent-lineage:{parent_experiment_id}",
            "parentFrontierId": None,
        }
    trust_root_policy = registry.get("trustRootPolicy")
    trusted_roots = _trusted_roots_from_registry(registry)
    receipt_error = _trust_receipt_error(
        entry,
        parent_experiment_id,
        trusted_roots,
        trust_root_policy,
        DEFAULT_TRUSTED_OPERATOR_RISK_ACCEPTANCE_REGISTRY_PATH,
        trust_gate_mode,
        operator_approval_ref,
    )
    if receipt_error is not None:
        return {
            "status": "unresolved",
            "reasonCode": f"lineage-{receipt_error}",
            "lineageResolutionRef": f"trusted-parent-lineage:{parent_experiment_id}",
            "parentFrontierId": None,
        }
    return {
        "status": "resolved",
        "reasonCode": "lineage-resolved",
        "lineageResolutionRef": f"trusted-parent-lineage:{parent_experiment_id}",
        "parentFrontierId": parent_frontier_id,
    }


def _load_runner_owned_check(run_dir: Path, filename: str) -> dict[str, Any] | None:
    path = run_dir / filename
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("source") != "cpu-subset-runner":
        return None
    return payload


def _mint_runner_verification(
    *,
    run_id: str,
    completed_at: str,
    val_loss: float,
    val_bpb: float,
    command: list[str],
    run_dir: Path,
    train_env: dict[str, str],
) -> dict[str, Any]:
    deterministic = _load_runner_owned_check(run_dir, "deterministic_rerun.json")
    seed_variance = _load_runner_owned_check(run_dir, "seed_variance.json")
    artifact_paths = [
        artifact_path(run_dir / "stdout.txt"),
        artifact_path(run_dir / "stderr.txt"),
        artifact_path(run_dir / "result.json"),
        artifact_path(run_dir / "final_model.int8.ptz"),
    ]
    attestation_material = {
        "runId": run_id,
        "completedAt": completed_at,
        "valLoss": val_loss,
        "valBpb": val_bpb,
        "command": command,
        "artifactPaths": artifact_paths,
        "seed": train_env.get("SEED"),
        "iterations": train_env.get("ITERATIONS"),
        "valTokenLimit": train_env.get("VAL_TOKEN_LIMIT"),
    }
    attestation_ref = "attestation-" + hashlib.sha256(
        json.dumps(attestation_material, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return {
        "source": "runner",
        "runnerIdentity": "cpu-subset-runner",
        "runId": run_id,
        "completedAt": completed_at,
        "attestationRef": attestation_ref,
        "resultClass": "cpu-subset",
        "verificationClass": "cpu-subset",
        "provenanceVerified": True,
        "deterministicValidated": bool(
            deterministic and deterministic.get("status") == "passed"
        ),
        "seedVarianceValidated": bool(
            seed_variance and seed_variance.get("status") == "passed"
        ),
        "runtimeWithinCap": (
            _int_from_env(train_env, "MAX_WALLCLOCK_SECONDS", 0)
            <= CPU_SUBSET_MAX_WALLCLOCK_SECONDS
        ),
        "deterministicEvidenceRef": deterministic.get("evidenceRef")
        if deterministic
        else None,
        "seedVarianceEvidenceRef": seed_variance.get("evidenceRef")
        if seed_variance
        else None,
        "artifactPaths": artifact_paths,
    }


def _evaluate_runner_verification(runner_verification: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(runner_verification, dict):
        return {
            "status": "failed",
            "reasonCode": "runner-verification-missing",
            "attestationRef": None,
        }
    attestation_ref = runner_verification.get("attestationRef")
    if not isinstance(attestation_ref, str) or not attestation_ref:
        return {
            "status": "failed",
            "reasonCode": "runner-attestation-missing",
            "attestationRef": None,
        }
    gate_checks = [
        ("source", runner_verification.get("source") == "runner"),
        ("provenance", bool(runner_verification.get("provenanceVerified"))),
        ("deterministic", bool(runner_verification.get("deterministicValidated"))),
        ("seedVariance", bool(runner_verification.get("seedVarianceValidated"))),
        ("runtimeCap", bool(runner_verification.get("runtimeWithinCap"))),
    ]
    failed = [name for name, passed in gate_checks if not passed]
    if failed:
        return {
            "status": "failed",
            "reasonCode": f"runner-verification-gate-failed:{','.join(failed)}",
            "attestationRef": attestation_ref,
        }
    return {
        "status": "passed",
        "reasonCode": "runner-verification-passed",
        "attestationRef": attestation_ref,
    }


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
    trust_gate_mode = str(candidate.get("trustGateMode") or "operator-approved-fallback").strip().lower()
    operator_approval_ref = candidate.get("operatorApprovalRef")
    if not isinstance(operator_approval_ref, str):
        operator_approval_ref = None
    parent_experiment_id = _resolve_parent_experiment_id(candidate, cpu_records)
    parent_record = next(
        (
            record
            for record in cpu_records
            if isinstance(record.get("experimentId"), str) and record["experimentId"] == parent_experiment_id
        ),
        None,
    )
    parent_lineage = _resolve_parent_lineage(
        parent_experiment_id=parent_experiment_id,
        registry_path=DEFAULT_TRUSTED_PARENT_REGISTRY_PATH,
        trust_gate_mode=trust_gate_mode,
        operator_approval_ref=operator_approval_ref,
    )
    parent_frontier_id = parent_lineage["parentFrontierId"]
    runner_verification = _mint_runner_verification(
        run_id=run_id,
        completed_at=completed_at,
        val_loss=val_loss,
        val_bpb=val_bpb,
        command=command,
        run_dir=run_dir,
        train_env=train_env,
    )
    attestation_resolution = _resolve_runner_attestation(
        attestation_ref=runner_verification.get("attestationRef"),
        run_id=run_id,
        registry_path=DEFAULT_TRUSTED_RUNNER_ATTESTATION_REGISTRY_PATH,
        trust_gate_mode=trust_gate_mode,
        operator_approval_ref=operator_approval_ref,
    )
    runner_verification["provenanceVerified"] = attestation_resolution["status"] == "resolved"
    runner_verification["attestationRegistryRef"] = attestation_resolution["attestationRegistryRef"]
    runner_verification["attestationResolutionReasonCode"] = attestation_resolution["reasonCode"]
    runner_gate = _evaluate_runner_verification(runner_verification)
    trust_gates_satisfied = (
        runner_gate["status"] == "passed" and parent_lineage["status"] == "resolved"
    )
    current_factors = _factor_snapshot_from_env(train_env)
    parent_factors = _factor_snapshot_from_record(parent_record) if parent_record else {}
    ranking_table = _build_ranking_table(cpu_records, experiment_id, val_bpb)
    effective_runtime_cap_sec = _int_from_env(train_env, "MAX_WALLCLOCK_SECONDS", 0)
    if effective_runtime_cap_sec <= 0:
        raise ValueError("MAX_WALLCLOCK_SECONDS must be present and greater than zero.")
    candidate_rank = int(ranking_table["candidateRank"])
    follow_up_lane = "cheap-screen" if candidate_rank == 1 else "cpu-subset"
    promotion_decision = "propose-follow-up" if candidate_rank == 1 and trust_gates_satisfied else "hold"
    retirement_decision = (
        "defer"
        if not trust_gates_satisfied
        else ("retain" if candidate_rank <= 3 else "retire")
    )
    verification_status = "passed" if trust_gates_satisfied else "verification_failed_trust"
    follow_up_task_proposal = None
    if promotion_decision == "propose-follow-up":
        follow_up_task_proposal = {
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
            "measuredEvidenceRef": evidence_id,
        }
    ranking_table["attestationRef"] = runner_gate["attestationRef"]
    ranking_table["attestationRegistryRef"] = attestation_resolution["attestationRegistryRef"]
    ranking_table["lineageResolutionRef"] = parent_lineage["lineageResolutionRef"]
    ranking_table["verificationStatus"] = verification_status
    ranking_table["resultClass"] = "cpu-subset"
    ranking_table["verificationClass"] = "cpu-subset"

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
        "integrityStatus": "passed",
        "verificationStatus": verification_status,
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
        "lineageResolution": parent_lineage,
        "attestationRef": runner_gate["attestationRef"],
        "attestationResolution": attestation_resolution,
        "runnerVerificationGate": runner_gate,
        "modelFactory": {
            "owner": "cpu-subset-runner",
            "normalizedFactors": current_factors,
        },
        "rankingTable": ranking_table,
        "modelFactoryDecision": {
            "promotionDecision": promotion_decision,
            "retirementDecision": retirement_decision,
            "reason": (
                "Promotion and follow-up require runner-attested deterministic and seed-variance "
                "verification plus trusted-registry lineage resolution; fail closed on missing trust."
            ),
        },
        "followUpTaskProposal": follow_up_task_proposal,
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
            "decision": "hold" if not trust_gates_satisfied else "provisional",
            "reason": (
                "CPU-subset evidence can rank ideas but cannot claim benchmark progress."
                if trust_gates_satisfied
                else "Follow-up and promotion are blocked until trusted attestation and lineage gates pass."
            ),
            "missingEvidence": ["benchmark-verified-run"],
        },
        "untrustedCandidateMetadata": {
            "declaredVerification": {
                "verificationPassed": candidate.get("verificationPassed"),
                "seedVariancePassed": candidate.get("seedVariancePassed"),
                "runnerVerification": candidate.get("runnerVerification"),
            }
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
    preflight = _preflight_validate_candidate(candidate)
    if preflight["errors"]:
        return {
            "status": "integrity_error",
            "reasonCode": "cpu-subset-preflight-validation-failed",
            "candidateId": candidate_id,
            "resultClass": "cpu-subset",
            "verificationClass": "cpu-subset",
            "integrityStatus": "failed",
            "validationErrors": preflight["errors"],
            "validationPolicy": {
                "activationModeAllowlist": sorted(ALLOWED_ACTIVATION_MODES),
                "swigluClampModeAllowlist": sorted(ALLOWED_SWIGLU_CLAMP_MODES),
                "muonUpdateScaleMin": MIN_MUON_UPDATE_SCALE,
                "muonUpdateScaleMax": MAX_MUON_UPDATE_SCALE,
                "muonBackendStepsMin": MIN_MUON_BACKEND_STEPS,
                "muonBackendStepsMax": MAX_MUON_BACKEND_STEPS,
                "muonMomentumExclusiveRange": [MIN_MUON_MOMENTUM, MAX_MUON_MOMENTUM],
                "muonMomentumWarmupStepsMin": MIN_MUON_MOMENTUM_WARMUP_STEPS,
                "adamwExemptionSelectorAllowlist": sorted(ALLOWED_ADAMW_EXEMPTION_SELECTORS),
            },
        }
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
            "integrityStatus": "failed",
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
            "integrityStatus": "failed",
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
        "integrityStatus": "passed",
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
