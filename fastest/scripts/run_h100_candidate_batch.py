from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import platform
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from candidates.registry import default_candidate_dicts

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "records/h100_candidate_batch"
DEFAULT_DATA_PATH = REPO_ROOT / "data/datasets/fineweb10B_sp1024"
DEFAULT_TOKENIZER_PATH = REPO_ROOT / "data/tokenizers/fineweb_1024_bpe.model"

FINAL_METRIC_RE = re.compile(
    r"final_int8_zlib_roundtrip_exact\s+val_loss:(?P<loss>[0-9.]+)\s+val_bpb:(?P<bpb>[0-9.]+)"
)
MODEL_PARAMS_RE = re.compile(r"model_params:(?P<params>[0-9]+)")
TRAIN_STEP_RE = re.compile(
    r"step:(?P<step>[0-9]+)/(?P<iterations>[0-9]+)\s+train_loss:(?P<loss>[0-9.]+)\s+"
    r"train_time:(?P<ms>[0-9.]+)ms\s+step_avg:(?P<avg>[0-9.]+)ms"
)
VAL_STEP_RE = re.compile(
    r"step:(?P<step>[0-9]+)/(?P<iterations>[0-9]+)\s+val_loss:(?P<loss>[0-9.]+)\s+"
    r"val_bpb:(?P<bpb>[0-9.]+)\s+train_time:(?P<ms>[0-9.]+)ms\s+step_avg:(?P<avg>[0-9.]+)ms"
)
INT8_SIZE_RE = re.compile(r"Total submission size int8\+zlib:\s+(?P<size>[0-9]+)\s+bytes")
RAW_SIZE_RE = re.compile(r"Total submission size:\s+(?P<size>[0-9]+)\s+bytes")
PEAK_MEMORY_RE = re.compile(r"peak memory allocated:\s+(?P<allocated>[0-9]+)\s+MiB\s+reserved:\s+(?P<reserved>[0-9]+)\s+MiB")
ARTIFACT_BUDGET_RE = re.compile(
    r"artifact_budget:state:(?P<state>[^ ]+)\s+total_bytes:(?P<total>-?[0-9]+)\s+"
    r"limit_bytes:(?P<limit>[0-9]+)\s+headroom_bytes:(?P<headroom>-?[0-9]+)"
)
BATCH_TUNE_RE = re.compile(
    r"batch_tune_success\s+train_batch_tokens:(?P<tokens>[0-9]+)\s+"
    r"train_seq_len:(?P<seq>[0-9]+)\s+train_loss:(?P<loss>[0-9.]+)\s+"
    r"step_time:(?P<step_ms>[0-9.]+)ms\s+peak_allocated_mib:(?P<allocated>[0-9]+)\s+"
    r"peak_reserved_mib:(?P<reserved>[0-9]+)"
)


BASE_ENV = {
    "DATA_PATH": str(DEFAULT_DATA_PATH),
    "TOKENIZER_PATH": str(DEFAULT_TOKENIZER_PATH),
    "VOCAB_SIZE": "1024",
    "MAX_WALLCLOCK_SECONDS": "600",
    "VAL_LOSS_EVERY": "0",
    "TRAIN_LOG_EVERY": "200",
    "ARTIFACT_BUDGET_STRICT": "0",
    "SKIP_PRE_QUANT_FINAL_EVAL": "1",
}

STANDARD_IMPLEMENTATIONS = {"autoregressive_gpt", "routed_moe_gpt"}
LEGACY_DEFAULT_IMPLEMENTATION = "autoregressive_gpt"
UNSUPPORTED_ARCHITECTURE_CLAIMS = (
    "jepa",
    "text diffusion",
    "text_diffusion",
    "state space",
    "state-space",
    "ssm",
)

def utc_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_candidates(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return default_candidate_dicts()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("candidate file must contain a JSON list")
    candidates: list[dict[str, Any]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"candidate #{index} is not an object")
        if not isinstance(item.get("id"), str) or not item["id"]:
            raise ValueError(f"candidate #{index} must have a non-empty id")
        candidates.append(normalize_candidate_manifest(item, index))
    return candidates


def normalize_candidate_manifest(item: dict[str, Any], index: int) -> dict[str, Any]:
    candidate = dict(item)
    env = candidate.get("env", {})
    if env is None:
        env = {}
    if not isinstance(env, dict):
        raise ValueError(f"candidate #{index} env must be an object")
    env = {str(key): str(value) for key, value in env.items()}

    implementation = candidate.get("implementation")
    if implementation is not None and not isinstance(implementation, str):
        raise ValueError(f"candidate #{index} implementation must be a string")
    implementation = implementation.strip() if isinstance(implementation, str) else ""

    if not implementation:
        env_implementation = env.get("CANDIDATE_IMPL", "").strip()
        if env_implementation:
            implementation = env_implementation
        elif candidate.get("command") is None:
            implementation = LEGACY_DEFAULT_IMPLEMENTATION
            warnings.warn(
                f"legacy candidate manifest {candidate['id']!r} did not declare an implementation; "
                f"defaulting to {implementation!r}",
                RuntimeWarning,
                stacklevel=2,
            )

    validate_candidate_architecture_claim(candidate, implementation)

    if implementation:
        candidate["implementation"] = implementation
        if candidate.get("command") is None and implementation in STANDARD_IMPLEMENTATIONS:
            env_implementation = env.get("CANDIDATE_IMPL", "").strip()
            if env_implementation and env_implementation != implementation:
                raise ValueError(
                    f"candidate {candidate['id']!r} has implementation {implementation!r} "
                    f"but env CANDIDATE_IMPL={env_implementation!r}"
                )
            if not env_implementation:
                env["CANDIDATE_IMPL"] = implementation
                warnings.warn(
                    f"legacy candidate manifest {candidate['id']!r} did not route CANDIDATE_IMPL; "
                    f"defaulting to {implementation!r}",
                    RuntimeWarning,
                    stacklevel=2,
                )
    candidate["env"] = env
    return candidate


def validate_candidate_architecture_claim(candidate: dict[str, Any], implementation: str) -> None:
    claim_text = " ".join(
        str(candidate.get(key, ""))
        for key in ("id", "family", "hypothesis", "description", "implementation")
    ).lower()
    tags = candidate.get("hypothesisTags", [])
    if isinstance(tags, list):
        claim_text = f"{claim_text} {' '.join(str(tag).lower() for tag in tags)}"
    elif isinstance(tags, str):
        claim_text = f"{claim_text} {tags.lower()}"

    for claim in UNSUPPORTED_ARCHITECTURE_CLAIMS:
        if claim in claim_text:
            raise ValueError(
                f"candidate {candidate['id']!r} has unsupported architecture claim {claim!r}; "
                "native implementation is not registered"
            )
    if "moe" in claim_text or "mixture_of_experts" in claim_text:
        if implementation != "routed_moe_gpt":
            raise ValueError(
                f"candidate {candidate['id']!r} has unsupported architecture claim 'moe' "
                f"for implementation {implementation or 'unspecified'!r}"
            )


def safe_id(raw: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", raw).strip("-._")
    return cleaned or "candidate"


def command_for_candidate(
    candidate: dict[str, Any],
    *,
    smoke: bool = False,
    nproc_per_node: int = 1,
) -> list[str]:
    command = candidate.get("command")
    if command is None:
        if smoke:
            return [sys.executable, str(REPO_ROOT / "train_gpt.py")]
        if nproc_per_node < 1:
            raise ValueError(f"nproc_per_node must be at least 1, got {nproc_per_node}")
        return [
            "torchrun",
            "--standalone",
            f"--nproc_per_node={nproc_per_node}",
            str(REPO_ROOT / "train_gpt.py"),
        ]
    if isinstance(command, str):
        return shlex.split(command)
    if isinstance(command, list) and all(isinstance(part, str) for part in command):
        return list(command)
    raise ValueError(f"candidate {candidate.get('id')!r} has invalid command")


def candidate_uses_default_train_command(candidate: dict[str, Any]) -> bool:
    return candidate.get("command") is None


def build_env(candidate: dict[str, Any], run_id: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(BASE_ENV)
    overrides = candidate.get("env", {})
    if not isinstance(overrides, dict):
        raise ValueError(f"candidate {candidate.get('id')!r} env must be an object")
    for key, value in overrides.items():
        if not isinstance(key, str):
            raise ValueError(f"candidate {candidate.get('id')!r} env contains non-string key")
        env[key] = str(value)
    env["RUN_ID"] = run_id
    return env


def apply_smoke_overrides(env: dict[str, str]) -> None:
    # CPU smoke validates runner/training/artifact plumbing with a tiny shape.
    # Candidate-specific recurrence orders can reference layers that do not
    # exist in the tiny model, so clear them while preserving other knobs.
    env.pop("ENCODER_LAYER_ORDER", None)
    env.pop("DECODER_LAYER_ORDER", None)
    env.update(
        {
            "ITERATIONS": "2",
            "WARMUP_STEPS": "0",
            "WARMDOWN_ITERS": "0",
            "NUM_LAYERS": "2",
            "MODEL_DIM": "128",
            "NUM_HEADS": "4",
            "NUM_KV_HEADS": "2",
            "TRAIN_SEQ_LEN": "128",
            "TRAIN_BATCH_TOKENS": "8192",
            "VAL_BATCH_SIZE": "1024",
            "VAL_TOKEN_LIMIT": "4096",
            "MAX_WALLCLOCK_SECONDS": "0",
            "TRAIN_LOG_EVERY": "1",
            "ARTIFACT_BUDGET_STRICT": "0",
        }
    )


def int_env(env: dict[str, str], key: str, default: int) -> int:
    try:
        return int(float(env.get(key, str(default))))
    except (TypeError, ValueError):
        return default


def round_batch_tokens(tokens: int, train_seq_len: int, grad_accum_steps: int = 8, world_size: int = 1) -> int:
    unit = max(train_seq_len * grad_accum_steps * world_size, 1)
    return max(unit, (max(tokens, unit) // unit) * unit)


def first_gpu_memory_mib(hardware: dict[str, Any]) -> int | None:
    gpus = hardware.get("gpus")
    if not isinstance(gpus, list) or not gpus:
        return None
    raw = gpus[0].get("memoryTotalMiB") if isinstance(gpus[0], dict) else None
    try:
        return int(float(str(raw).strip()))
    except (TypeError, ValueError):
        return None


def collect_hardware_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "capturedAt": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cudaVisibleDevices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpus": [],
    }
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,memory.total,driver_version,power.limit",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        info["nvidiaSmiAvailable"] = False
        return info

    info["nvidiaSmiAvailable"] = result.returncode == 0
    info["nvidiaSmiStderr"] = result.stderr.strip()
    if result.returncode != 0:
        return info
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 6:
            continue
        info["gpus"].append(
            {
                "index": parts[0],
                "name": parts[1],
                "uuid": parts[2],
                "memoryTotalMiB": parts[3],
                "driverVersion": parts[4],
                "powerLimitW": parts[5],
            }
        )
    return info


def parse_batch_tune(stdout: str) -> dict[str, Any] | None:
    matches = list(BATCH_TUNE_RE.finditer(stdout))
    if not matches:
        return None
    match = matches[-1]
    tokens = int(match.group("tokens"))
    step_ms = float(match.group("step_ms"))
    return {
        "trainBatchTokens": tokens,
        "trainSeqLen": int(match.group("seq")),
        "trainLoss": float(match.group("loss")),
        "stepMs": step_ms,
        "tokensPerSecond": tokens / (step_ms / 1000.0) if step_ms > 0 else None,
        "peakAllocatedMiB": int(match.group("allocated")),
        "peakReservedMiB": int(match.group("reserved")),
    }


def stream_subprocess(
    *,
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    start = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []

    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
        timed_out = False
        termination = "completed"
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        termination = "hard-timeout"
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        except Exception:
            process.kill()
        process.wait(timeout=10)

    stdout_chunks.append(stdout)
    stderr_chunks.append(stderr)
    stdout_path.write_text("".join(stdout_chunks), encoding="utf-8")
    stderr_path.write_text("".join(stderr_chunks), encoding="utf-8")
    return {
        "returnCode": int(process.returncode or 0),
        "timedOut": timed_out,
        "terminationReason": termination,
        "elapsedSeconds": round(time.monotonic() - start, 3),
        "stdout": "".join(stdout_chunks),
        "stderr": "".join(stderr_chunks),
    }


def run_batch_tune_probe(
    *,
    candidate: dict[str, Any],
    run_dir: Path,
    env: dict[str, str],
    tokens: int,
    timeout_seconds: int,
    nproc_per_node: int,
) -> dict[str, Any]:
    probe_dir = run_dir / "batch_tune" / f"tokens_{tokens}"
    probe_dir.mkdir(parents=True, exist_ok=True)
    probe_env = dict(env)
    probe_env.update(
        {
            "BATCH_TUNE_ONLY": "1",
            "TRAIN_BATCH_TOKENS": str(tokens),
            "WARMUP_STEPS": "0",
            "VAL_LOSS_EVERY": "0",
            "TRAIN_LOG_EVERY": "0",
            "SKIP_PRE_QUANT_FINAL_EVAL": "1",
            "ARTIFACT_BUDGET_STRICT": "0",
        }
    )
    completed = stream_subprocess(
        command=command_for_candidate(candidate, smoke=False, nproc_per_node=nproc_per_node),
        cwd=probe_dir,
        env=probe_env,
        stdout_path=probe_dir / "stdout.txt",
        stderr_path=probe_dir / "stderr.txt",
        timeout_seconds=timeout_seconds,
    )
    tune_metrics = parse_batch_tune(completed["stdout"])
    success = completed["returnCode"] == 0 and tune_metrics is not None and not completed["timedOut"]
    result = {
        "tokens": tokens,
        "success": success,
        "returnCode": completed["returnCode"],
        "timedOut": completed["timedOut"],
        "terminationReason": completed["terminationReason"],
        "elapsedSeconds": completed["elapsedSeconds"],
        "probeDir": str(probe_dir),
    }
    if tune_metrics is not None:
        result.update(tune_metrics)
    (probe_dir / "probe_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def auto_tune_batch_size(
    *,
    candidate: dict[str, Any],
    run_dir: Path,
    env: dict[str, str],
    hardware: dict[str, Any],
    target_memory_fraction: float,
    max_tokens: int,
    timeout_seconds: int,
    nproc_per_node: int,
) -> dict[str, Any]:
    train_seq_len = int_env(env, "TRAIN_SEQ_LEN", 1024)
    initial_tokens = round_batch_tokens(int_env(env, "TRAIN_BATCH_TOKENS", 524_288), train_seq_len)
    max_tokens = round_batch_tokens(max_tokens, train_seq_len)
    gpu_memory_mib = first_gpu_memory_mib(hardware)
    target_reserved_mib = int(gpu_memory_mib * target_memory_fraction) if gpu_memory_mib is not None else None
    probes: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    first_bad_tokens: int | None = None

    def probe(tokens: int) -> dict[str, Any]:
        result = run_batch_tune_probe(
            candidate=candidate,
            run_dir=run_dir,
            env=env,
            tokens=tokens,
            timeout_seconds=timeout_seconds,
            nproc_per_node=nproc_per_node,
        )
        probes.append(result)
        return result

    tokens = min(initial_tokens, max_tokens)
    result = probe(tokens)
    while not result["success"] and tokens > train_seq_len * 8:
        first_bad_tokens = tokens
        tokens = round_batch_tokens(tokens // 2, train_seq_len)
        result = probe(tokens)
    if not result["success"]:
        summary = {
            "enabled": True,
            "status": "failed",
            "selectedTrainBatchTokens": initial_tokens,
            "targetMemoryFraction": target_memory_fraction,
            "targetReservedMiB": target_reserved_mib,
            "gpuMemoryTotalMiB": gpu_memory_mib,
            "probes": probes,
        }
        (run_dir / "batch_tune_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return summary

    best = result
    while best["tokens"] < max_tokens:
        if target_reserved_mib is not None and best.get("peakReservedMiB", 0) >= target_reserved_mib:
            break
        next_tokens = round_batch_tokens(best["tokens"] * 2, train_seq_len)
        if next_tokens <= best["tokens"]:
            break
        next_tokens = min(next_tokens, max_tokens)
        result = probe(next_tokens)
        if result["success"]:
            best = result
            continue
        first_bad_tokens = next_tokens
        break

    if first_bad_tokens is not None and best is not None:
        low = best["tokens"]
        high = first_bad_tokens
        unit = train_seq_len * 8
        while high - low > unit:
            mid = round_batch_tokens((low + high) // 2, train_seq_len)
            if mid <= low or mid >= high:
                break
            result = probe(mid)
            if result["success"]:
                best = result
                low = mid
            else:
                high = mid

    if best is None:
        selected = initial_tokens
        status = "failed"
    else:
        selected = int(best["tokens"])
        status = "completed"
        env["TRAIN_BATCH_TOKENS"] = str(selected)

    summary = {
        "enabled": True,
        "status": status,
        "selectedTrainBatchTokens": selected,
        "targetMemoryFraction": target_memory_fraction,
        "targetReservedMiB": target_reserved_mib,
        "gpuMemoryTotalMiB": gpu_memory_mib,
        "maxTrainBatchTokens": max_tokens,
        "bestProbe": best,
        "probes": probes,
    }
    (run_dir / "batch_tune_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def tokens_per_second(train_batch_tokens: Any, step_avg_ms: Any) -> float | None:
    tokens = numeric(train_batch_tokens)
    step_ms = numeric(step_avg_ms)
    if tokens is None or step_ms is None or step_ms <= 0:
        return None
    return tokens / (step_ms / 1000.0)


def prune_raw_checkpoint(run_dir: Path, keep_raw_checkpoint: bool) -> dict[str, Any]:
    raw_path = run_dir / "final_model.pt"
    quantized_path = run_dir / "final_model.int8.ptz"
    raw_existed = raw_path.exists()
    if raw_existed and not keep_raw_checkpoint:
        raw_path.unlink()
    return {
        "rawCheckpointKept": raw_path.exists(),
        "rawCheckpointPath": str(raw_path) if raw_path.exists() else None,
        "quantizedArtifactPath": str(quantized_path) if quantized_path.exists() else None,
        "quantizedArtifactBytes": quantized_path.stat().st_size if quantized_path.exists() else None,
    }


def parse_metrics(stdout: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "finalValLoss": None,
        "finalValBpb": None,
        "modelParams": None,
        "completedSteps": None,
        "targetIterations": None,
        "lastTrainLoss": None,
        "trainTimeMs": None,
        "stepAvgMs": None,
        "rawSubmissionBytes": None,
        "int8SubmissionBytes": None,
        "peakAllocatedMiB": None,
        "peakReservedMiB": None,
        "artifactBudgetState": None,
        "artifactBudgetBytes": None,
        "artifactBudgetLimitBytes": None,
        "artifactBudgetHeadroomBytes": None,
    }
    if match := FINAL_METRIC_RE.search(stdout):
        result["finalValLoss"] = float(match.group("loss"))
        result["finalValBpb"] = float(match.group("bpb"))
    if match := MODEL_PARAMS_RE.search(stdout):
        result["modelParams"] = int(match.group("params"))
    train_matches = list(TRAIN_STEP_RE.finditer(stdout))
    if train_matches:
        match = train_matches[-1]
        result["completedSteps"] = int(match.group("step"))
        result["targetIterations"] = int(match.group("iterations"))
        result["lastTrainLoss"] = float(match.group("loss"))
        result["trainTimeMs"] = float(match.group("ms"))
        result["stepAvgMs"] = float(match.group("avg"))
    val_matches = list(VAL_STEP_RE.finditer(stdout))
    if val_matches and result["completedSteps"] is None:
        match = val_matches[-1]
        result["completedSteps"] = int(match.group("step"))
        result["targetIterations"] = int(match.group("iterations"))
        result["trainTimeMs"] = float(match.group("ms"))
        result["stepAvgMs"] = float(match.group("avg"))
    raw_matches = list(RAW_SIZE_RE.finditer(stdout))
    if raw_matches:
        result["rawSubmissionBytes"] = int(raw_matches[-1].group("size"))
    int8_matches = list(INT8_SIZE_RE.finditer(stdout))
    if int8_matches:
        result["int8SubmissionBytes"] = int(int8_matches[-1].group("size"))
    if match := PEAK_MEMORY_RE.search(stdout):
        result["peakAllocatedMiB"] = int(match.group("allocated"))
        result["peakReservedMiB"] = int(match.group("reserved"))
    if match := ARTIFACT_BUDGET_RE.search(stdout):
        result["artifactBudgetState"] = match.group("state")
        result["artifactBudgetBytes"] = int(match.group("total"))
        result["artifactBudgetLimitBytes"] = int(match.group("limit"))
        result["artifactBudgetHeadroomBytes"] = int(match.group("headroom"))
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "rank",
        "id",
        "status",
        "family",
        "hardwareLabel",
        "quantization",
        "finalValBpb",
        "finalValLoss",
        "completedSteps",
        "stepAvgMs",
        "tokensPerSecond",
        "selectedTrainBatchTokens",
        "batchTuneStatus",
        "modelParams",
        "int8SubmissionBytes",
        "rawSubmissionBytes",
        "artifactBudgetState",
        "peakAllocatedMiB",
        "elapsedSeconds",
        "runDir",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def write_group_csv(path: Path, groups: list[dict[str, Any]]) -> None:
    columns = [
        "groupKey",
        "count",
        "completed",
        "bestCandidate",
        "bestValBpb",
        "medianValBpb",
        "meanValBpb",
        "bestStepAvgMs",
        "bestTokensPerSecond",
        "bestSelectedTrainBatchTokens",
        "bestInt8SubmissionBytes",
        "bestArtifactHeadroomBytes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for group in groups:
            writer.writerow({column: group.get(column) for column in columns})


def numeric(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        return float(value)
    return None


def median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def summarize_groups(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        group_key = str(row.get(key) or "unknown")
        grouped.setdefault(group_key, []).append(row)

    summaries: list[dict[str, Any]] = []
    for group_key, group_rows in grouped.items():
        scored = [row for row in group_rows if numeric(row.get("finalValBpb")) is not None]
        ranked = sorted(scored, key=lambda row: (float(row["finalValBpb"]), str(row["id"])))
        values = [float(row["finalValBpb"]) for row in ranked]
        best = ranked[0] if ranked else None
        summaries.append(
            {
                "groupKey": group_key,
                "count": len(group_rows),
                "completed": len(scored),
                "bestCandidate": best.get("id") if best else None,
                "bestValBpb": best.get("finalValBpb") if best else None,
                "medianValBpb": median(values),
                "meanValBpb": sum(values) / len(values) if values else None,
                "bestStepAvgMs": best.get("stepAvgMs") if best else None,
                "bestTokensPerSecond": best.get("tokensPerSecond") if best else None,
                "bestSelectedTrainBatchTokens": best.get("selectedTrainBatchTokens") if best else None,
                "bestInt8SubmissionBytes": best.get("int8SubmissionBytes") if best else None,
                "bestArtifactHeadroomBytes": best.get("artifactBudgetHeadroomBytes") if best else None,
                "rows": [row.get("id") for row in group_rows],
            }
        )
    return sorted(
        summaries,
        key=lambda group: (
            float("inf") if numeric(group.get("bestValBpb")) is None else float(group["bestValBpb"]),
            str(group["groupKey"]),
        ),
    )


def tag_list(candidate: dict[str, Any]) -> list[str]:
    raw = candidate.get("hypothesisTags", [])
    if isinstance(raw, str):
        return [part.strip() for part in raw.split(",") if part.strip()]
    if isinstance(raw, list):
        return [str(part).strip() for part in raw if str(part).strip()]
    return []


def build_analysis_payload(
    rows: list[dict[str, Any]],
    *,
    hardware: dict[str, Any],
    started_at: str,
) -> dict[str, Any]:
    ranked = rank_rows([dict(row) for row in rows])
    family_groups = summarize_groups(ranked, "family")
    hardware_groups = summarize_groups(ranked, "hardwareLabel")
    quantization_groups = summarize_groups(ranked, "quantization")
    hypothesis_groups = summarize_groups(ranked, "hypothesis")
    tag_rows: list[dict[str, Any]] = []
    for row in ranked:
        for tag in row.get("hypothesisTags", []):
            tagged = dict(row)
            tagged["hypothesisTag"] = tag
            tag_rows.append(tagged)
    tag_groups = summarize_groups(tag_rows, "hypothesisTag")
    return {
        "schema": "parameter-golf-h100-candidate-analysis/v1",
        "startedAt": started_at,
        "metric": {"name": "final_int8_zlib_roundtrip_exact.val_bpb", "direction": "lower_is_better"},
        "hardware": hardware,
        "views": {
            "leaderboard": ranked,
            "byFamily": family_groups,
            "byHardware": hardware_groups,
            "byQuantization": quantization_groups,
            "byHypothesis": hypothesis_groups,
            "byHypothesisTag": tag_groups,
        },
    }


def present_evidence_for_row(row: dict[str, Any]) -> list[str]:
    present: list[str] = []
    if numeric(row.get("finalValBpb")) is not None:
        present.append("benchmark-measurement-evidence")
    if numeric(row.get("int8SubmissionBytes")) is not None or numeric(row.get("artifactBudgetBytes")) is not None:
        present.append("artifact-budget")
    if row.get("batchTuneStatus") in {"completed", "skipped"} or numeric(row.get("selectedTrainBatchTokens")) is not None:
        present.append("runner-config")
    if row.get("hardware"):
        present.append("hardware-manifest")
    if row.get("quantizedArtifactPath"):
        present.append("quantized-artifact")
    return sorted(set(present))


def build_model_factory_evidence_payload(
    rows: list[dict[str, Any]],
    *,
    started_at: str,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for row in rank_rows([dict(item) for item in rows]):
        candidate_id = str(row.get("id") or "unknown")
        safe_candidate_id = safe_id(candidate_id)
        present_evidence = present_evidence_for_row(row)
        metric_readings = []
        if numeric(row.get("finalValBpb")) is not None:
            metric_readings.append(
                {
                    "metricName": "final_int8_zlib_roundtrip_exact.val_bpb",
                    "metricValue": float(row["finalValBpb"]),
                    "unit": "bpb",
                    "direction": "minimize",
                }
            )
        if numeric(row.get("tokensPerSecond")) is not None:
            metric_readings.append(
                {
                    "metricName": "train_tokens_per_second",
                    "metricValue": float(row["tokensPerSecond"]),
                    "unit": "tokens/second",
                    "direction": "maximize",
                }
            )
        if numeric(row.get("int8SubmissionBytes")) is not None:
            metric_readings.append(
                {
                    "metricName": "int8_zlib_submission_bytes",
                    "metricValue": int(row["int8SubmissionBytes"]),
                    "unit": "bytes",
                    "direction": "minimize",
                }
            )
        run_dir = str(row.get("runDir") or "")
        artifact_refs = [
            {"kind": "log", "uri": f"{run_dir}/stdout.txt"},
            {"kind": "log", "uri": f"{run_dir}/stderr.txt"},
            {"kind": "report", "uri": f"{run_dir}/result.json"},
        ]
        if row.get("quantizedArtifactPath"):
            artifact_refs.append({"kind": "checkpoint", "uri": str(row["quantizedArtifactPath"])})
        if output_dir is not None:
            artifact_refs.append({"kind": "report", "uri": str(output_dir / "summary.md")})
            artifact_refs.append({"kind": "report", "uri": str(output_dir / "analysis/analysis.json")})

        status = str(row.get("status") or "unknown")
        completed = status == "completed"
        artifact_budget_state = str(row.get("artifactBudgetState") or "")
        if artifact_budget_state == "within-budget":
            legality_status = "clear"
        elif artifact_budget_state:
            legality_status = "conflict"
        else:
            legality_status = "unknown"
        factors = sorted(
            set(
                [
                    str(row.get("family") or "unknown"),
                    *[str(tag) for tag in row.get("hypothesisTags", [])],
                ]
            )
        )
        records.append(
            {
                "evidenceId": f"evidence-h100-batch-{started_at}-{safe_candidate_id}",
                "campaignId": "parameter-golf",
                "candidateId": candidate_id,
                "sourceId": "run_h100_candidate_batch.py",
                "sourceKind": "model-factory",
                "observedAt": started_at,
                "producedBy": "fastest/scripts/run_h100_candidate_batch.py",
                "runId": row.get("env", {}).get("RUN_ID") if isinstance(row.get("env"), dict) else run_dir,
                "experimentId": candidate_id,
                "claim": {
                    "claimId": f"claim-{safe_candidate_id}",
                    "claimType": "idea",
                    "text": str(row.get("hypothesis") or row.get("description") or candidate_id),
                },
                "metricReadings": metric_readings,
                "factorSet": {
                    "factors": factors,
                    "incompatibleWith": [],
                },
                "trustSignals": {
                    "trustClass": "candidate" if completed else "unknown",
                    "reproducibilityClass": "candidate" if completed else "unknown",
                    "independentSource": False,
                    "freshness": "current",
                },
                "legalitySignals": {
                    "status": legality_status,
                    "conflicts": [] if legality_status == "clear" else ["artifact-budget-or-run-incomplete"],
                },
                "submissionSignals": {
                    "readiness": "not-ready",
                    "leaderboardClaim": "none",
                    "requiredEvidence": [
                        "benchmark-measurement-evidence",
                        "artifact-budget",
                        "hardware-manifest",
                        "reproducibility-manifest",
                    ],
                    "presentEvidence": present_evidence,
                },
                "artifactRefs": artifact_refs,
                "supersedesEvidenceIds": [],
            }
        )
    return {
        "schema": "parameter-golf-model-factory-evidence-export/v1",
        "generatedAt": started_at,
        "records": records,
    }


def write_analysis_outputs(output_dir: Path, rows: list[dict[str, Any]], hardware: dict[str, Any], started_at: str) -> None:
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    payload = build_analysis_payload(rows, hardware=hardware, started_at=started_at)
    (analysis_dir / "analysis.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    evidence_payload = build_model_factory_evidence_payload(rows, started_at=started_at, output_dir=output_dir)
    (analysis_dir / "model_factory_evidence.json").write_text(json.dumps(evidence_payload, indent=2) + "\n", encoding="utf-8")
    (output_dir / "model_factory_evidence.json").write_text(json.dumps(evidence_payload, indent=2) + "\n", encoding="utf-8")
    write_group_csv(analysis_dir / "by_family.csv", payload["views"]["byFamily"])
    write_group_csv(analysis_dir / "by_hardware.csv", payload["views"]["byHardware"])
    write_group_csv(analysis_dir / "by_quantization.csv", payload["views"]["byQuantization"])
    write_group_csv(analysis_dir / "by_hypothesis.csv", payload["views"]["byHypothesis"])
    write_group_csv(analysis_dir / "by_hypothesis_tag.csv", payload["views"]["byHypothesisTag"])
    write_analysis_markdown(analysis_dir / "analysis.md", payload)
    write_hypothesis_graph(output_dir / "hypothesis_graph.md", payload)
    write_hypothesis_graph_json(output_dir / "hypothesis_graph.json", payload)


def write_analysis_markdown(path: Path, payload: dict[str, Any]) -> None:
    hardware = payload["hardware"]
    gpu_names = ", ".join(gpu.get("name", "unknown") for gpu in hardware.get("gpus", [])) or "unknown"
    lines = [
        "# Candidate Analysis Views",
        "",
        f"- Hardware: `{gpu_names}`",
        f"- Host: `{hardware.get('hostname') or 'unknown'}`",
        f"- Metric: `{payload['metric']['name']}`",
        "",
    ]
    for title, key in (
        ("By Family", "byFamily"),
        ("By Hardware", "byHardware"),
        ("By Quantization", "byQuantization"),
        ("By Hypothesis", "byHypothesis"),
        ("By Hypothesis Tag", "byHypothesisTag"),
    ):
        lines.extend(
            [
                f"## {title}",
                "",
                "| Group | Completed | Best candidate | Best val_bpb | Median val_bpb | Mean val_bpb |",
                "|---|---:|---|---:|---:|---:|",
            ]
        )
        for group in payload["views"][key]:
            lines.append(
                "| {group} | {completed}/{count} | `{best}` | {best_bpb} | {median_bpb} | {mean_bpb} |".format(
                    group=group["groupKey"],
                    completed=group["completed"],
                    count=group["count"],
                    best=group.get("bestCandidate") or "",
                    best_bpb=format_float(group.get("bestValBpb")),
                    median_bpb=format_float(group.get("medianValBpb")),
                    mean_bpb=format_float(group.get("meanValBpb")),
                )
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def format_float(value: Any) -> str:
    number = numeric(value)
    return "" if number is None else f"{number:.8f}"


def write_hypothesis_graph_json(path: Path, payload: dict[str, Any]) -> None:
    nodes: list[dict[str, Any]] = [
        {"id": "objective", "kind": "objective", "label": "Lower validation BPB"},
        {"id": "constraint", "kind": "constraint", "label": "10 min, H100 budget, 16MB artifact"},
    ]
    edges: list[dict[str, str]] = [{"source": "constraint", "target": "objective", "relation": "bounds"}]
    seen_nodes = {node["id"] for node in nodes}
    for row in payload["views"]["leaderboard"]:
        family_id = f"family:{row.get('family') or 'unknown'}"
        hypothesis_id = f"hypothesis:{row.get('hypothesis') or row.get('id')}"
        candidate_id = f"candidate:{row.get('id')}"
        for node_id, kind, label in (
            (family_id, "family", str(row.get("family") or "unknown")),
            (hypothesis_id, "hypothesis", str(row.get("hypothesis") or row.get("id"))),
            (candidate_id, "candidate", str(row.get("id"))),
        ):
            if node_id not in seen_nodes:
                nodes.append({"id": node_id, "kind": kind, "label": label})
                seen_nodes.add(node_id)
        edges.extend(
            [
                {"source": family_id, "target": hypothesis_id, "relation": "frames"},
                {"source": hypothesis_id, "target": candidate_id, "relation": "tested_by"},
                {"source": candidate_id, "target": "objective", "relation": result_relation(row)},
            ]
        )
    graph = {"schema": "parameter-golf-hypothesis-graph/v1", "nodes": nodes, "edges": edges}
    path.write_text(json.dumps(graph, indent=2) + "\n", encoding="utf-8")


def write_hypothesis_graph(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Hypothesis Graph",
        "",
        "```mermaid",
        "graph LR",
        "  constraint[\"10 min / H100 budget / 16MB\"] --> objective[\"lower val_bpb\"]",
    ]
    for row in payload["views"]["leaderboard"]:
        family = mermaid_id(f"family_{row.get('family') or 'unknown'}")
        hypothesis = mermaid_id(f"hypothesis_{row.get('hypothesis') or row.get('id')}")
        candidate = mermaid_id(f"candidate_{row.get('id')}")
        lines.extend(
            [
                f"  {family}[\"{escape_mermaid(str(row.get('family') or 'unknown'))}\"] --> {hypothesis}[\"{escape_mermaid(str(row.get('hypothesis') or row.get('id')))}\"]",
                f"  {hypothesis} --> {candidate}[\"{escape_mermaid(str(row.get('id')))}\"]",
                f"  {candidate} -- \"{result_relation(row)}\" --> objective",
            ]
        )
    lines.extend(["```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def result_relation(row: dict[str, Any]) -> str:
    if row.get("status") != "completed" or numeric(row.get("finalValBpb")) is None:
        return "unscored"
    rank = row.get("rank")
    if rank == 1:
        return "best"
    if isinstance(rank, int) and rank <= 3:
        return "promising"
    return "measured"


def mermaid_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", value)


def escape_mermaid(value: str) -> str:
    return value.replace('"', "'")


def svg_bar_chart(path: Path, rows: list[dict[str, Any]]) -> None:
    scored = [row for row in rows if numeric(row.get("finalValBpb")) is not None]
    width = 1100
    row_h = 34
    height = max(180, 90 + row_h * len(scored))
    if not scored:
        path.write_text("<svg xmlns='http://www.w3.org/2000/svg' width='900' height='120'><text x='20' y='60'>No scored runs</text></svg>\n")
        return
    values = [float(row["finalValBpb"]) for row in scored]
    best = min(values)
    worst = max(values)
    span = max(worst - best, 1e-9)
    lines = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='white'/>",
        "<text x='24' y='36' font-family='Arial' font-size='22' font-weight='700'>H100 candidate final val_bpb</text>",
    ]
    for index, row in enumerate(scored):
        y = 70 + index * row_h
        value = float(row["finalValBpb"])
        bar_w = 180 + int((1.0 - ((value - best) / span if span else 0.0)) * 560)
        color = "#1f77b4" if index else "#2ca02c"
        label = f"{row.get('rank', '')}. {row['id']} ({row.get('family', '')})"
        lines.extend(
            [
                f"<text x='24' y='{y + 20}' font-family='Arial' font-size='14'>{escape_xml(label)}</text>",
                f"<rect x='420' y='{y + 4}' width='{bar_w}' height='22' fill='{color}' opacity='0.85'/>",
                f"<text x='{430 + bar_w}' y='{y + 20}' font-family='Arial' font-size='14'>{value:.6f}</text>",
            ]
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def svg_scatter_chart(path: Path, rows: list[dict[str, Any]]) -> None:
    points = [
        row
        for row in rows
        if numeric(row.get("finalValBpb")) is not None and numeric(row.get("stepAvgMs")) is not None
    ]
    width, height = 900, 560
    if not points:
        path.write_text("<svg xmlns='http://www.w3.org/2000/svg' width='900' height='120'><text x='20' y='60'>No speed/scored runs</text></svg>\n")
        return
    xs = [float(row["stepAvgMs"]) for row in points]
    ys = [float(row["finalValBpb"]) for row in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    x_span = max(max_x - min_x, 1e-9)
    y_span = max(max_y - min_y, 1e-9)

    def sx(value: float) -> float:
        return 80 + (value - min_x) / x_span * 720

    def sy(value: float) -> float:
        return 470 - (value - min_y) / y_span * 380

    lines = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='white'/>",
        "<text x='24' y='36' font-family='Arial' font-size='22' font-weight='700'>Speed versus final val_bpb</text>",
        "<line x1='80' y1='470' x2='800' y2='470' stroke='#333'/>",
        "<line x1='80' y1='90' x2='80' y2='470' stroke='#333'/>",
        "<text x='350' y='525' font-family='Arial' font-size='14'>step average (ms)</text>",
        "<text x='18' y='80' font-family='Arial' font-size='14'>val_bpb</text>",
    ]
    for row in points:
        x = sx(float(row["stepAvgMs"]))
        y = sy(float(row["finalValBpb"]))
        lines.extend(
            [
                f"<circle cx='{x:.1f}' cy='{y:.1f}' r='7' fill='#d62728' opacity='0.8'/>",
                f"<text x='{x + 10:.1f}' y='{y + 4:.1f}' font-family='Arial' font-size='12'>{escape_xml(str(row['id']))}</text>",
            ]
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def escape_xml(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scored = sorted(
        [row for row in rows if numeric(row.get("finalValBpb")) is not None],
        key=lambda row: (float(row["finalValBpb"]), str(row["id"])),
    )
    for rank, row in enumerate(scored, start=1):
        row["rank"] = rank
    unscored = [row for row in rows if numeric(row.get("finalValBpb")) is None]
    for row in unscored:
        row["rank"] = None
    return scored + unscored


def hardware_label(hardware: dict[str, Any]) -> str:
    gpus = hardware.get("gpus")
    if isinstance(gpus, list) and gpus:
        first_gpu = gpus[0] if isinstance(gpus[0], dict) else {}
        name = str(first_gpu.get("name") or "GPU")
        return f"{len(gpus)}x{name}"
    return "cpu"


def write_summary(path: Path, rows: list[dict[str, Any]], started_at: str, output_dir: Path) -> None:
    scored = [row for row in rows if numeric(row.get("finalValBpb")) is not None]
    family_groups = summarize_groups(rows, "family")
    quantization_groups = summarize_groups(rows, "quantization")
    lines = [
        "# H100 Candidate Batch Summary",
        "",
        f"- Started: `{started_at}`",
        f"- Output directory: `{output_dir}`",
        f"- Completed candidates: `{len(scored)}/{len(rows)}`",
        f"- Metric: `final_int8_zlib_roundtrip_exact val_bpb` (lower is better)",
        "",
    ]
    tuned_rows = [row for row in rows if row.get("batchTuneStatus")]
    if tuned_rows:
        lines.extend(
            [
                "## Batch Tuning",
                "",
                "| Candidate | Status | Selected train tokens | Probe peak reserved MiB | Probe tokens/sec |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for row in rows:
            tune = row.get("batchTune") if isinstance(row.get("batchTune"), dict) else {}
            best_probe = tune.get("bestProbe") if isinstance(tune.get("bestProbe"), dict) else {}
            lines.append(
                "| `{id}` | {status} | {tokens} | {reserved} | {tps} |".format(
                    id=row.get("id", ""),
                    status=row.get("batchTuneStatus") or "",
                    tokens=row.get("selectedTrainBatchTokens") or "",
                    reserved=best_probe.get("peakReservedMiB", ""),
                    tps=format_float(best_probe.get("tokensPerSecond")),
                )
            )
        lines.append("")
    if scored:
        best = scored[0]
        lines.extend(
            [
                f"Best run: `{best['id']}` with `val_bpb={best['finalValBpb']:.8f}`.",
                "",
                "## Ranking",
                "",
                "| Rank | Candidate | Family | val_bpb | Steps | train tokens | tok/s | step_avg_ms | int8 bytes | Artifact budget |",
                "|---:|---|---|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in scored:
            lines.append(
                "| {rank} | `{id}` | {family} | {bpb:.8f} | {steps} | {tokens} | {tps} | {avg} | {size} | {budget} |".format(
                    rank=row["rank"],
                    id=row["id"],
                    family=row.get("family", ""),
                    bpb=float(row["finalValBpb"]),
                    steps=row.get("completedSteps") or "",
                    tokens=row.get("selectedTrainBatchTokens") or "",
                    tps=f"{float(row['tokensPerSecond']):.0f}" if numeric(row.get("tokensPerSecond")) is not None else "",
                    avg=f"{float(row['stepAvgMs']):.2f}" if numeric(row.get("stepAvgMs")) is not None else "",
                    size=row.get("int8SubmissionBytes") or "",
                    budget=row.get("artifactBudgetState") or "",
                )
            )
        lines.extend(
            [
                "",
                "## Best By Candidate Type",
                "",
                "| Type | Completed | Best candidate | Best val_bpb | Median val_bpb |",
                "|---|---:|---|---:|---:|",
            ]
        )
        for group in family_groups:
            lines.append(
                "| {group} | {completed}/{count} | `{best}` | {best_bpb} | {median_bpb} |".format(
                    group=group["groupKey"],
                    completed=group["completed"],
                    count=group["count"],
                    best=group.get("bestCandidate") or "",
                    best_bpb=format_float(group.get("bestValBpb")),
                    median_bpb=format_float(group.get("medianValBpb")),
                )
            )
        lines.extend(
            [
                "",
                "## Best By Quantization",
                "",
                "| Quantization | Completed | Best candidate | Best val_bpb | Median val_bpb |",
                "|---|---:|---|---:|---:|",
            ]
        )
        for group in quantization_groups:
            lines.append(
                "| {group} | {completed}/{count} | `{best}` | {best_bpb} | {median_bpb} |".format(
                    group=group["groupKey"],
                    completed=group["completed"],
                    count=group["count"],
                    best=group.get("bestCandidate") or "",
                    best_bpb=format_float(group.get("bestValBpb")),
                    median_bpb=format_float(group.get("medianValBpb")),
                )
            )
    failed = [row for row in rows if row.get("status") != "completed"]
    if failed:
        lines.extend(["", "## Failed Or Unscored", ""])
        for row in failed:
            lines.append(
                f"- `{row['id']}`: status `{row.get('status')}`, return code `{row.get('returnCode')}`, "
                f"see `{row.get('runDir')}`"
            )
    lines.extend(
        [
            "",
            "## Charts",
            "",
            "- `charts/final_val_bpb.svg`",
            "- `charts/speed_vs_bpb.svg`",
            "- `analysis/analysis.md`",
            "- `hypothesis_graph.md`",
            "",
            "## Coverage Notes",
            "",
            "- Built-in candidates are implementation-backed by modules under `candidates/`; JEPA, text diffusion, and SSM are intentionally absent until native implementations are added.",
            "- To run native implementations, pass `--candidate-file candidates.json`; each candidate can provide `command`, `env`, `family`, and `description`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_batch(args: argparse.Namespace) -> int:
    candidates = load_candidates(args.candidate_file)
    selected_ids = set(args.only or [])
    if selected_ids:
        candidates = [candidate for candidate in candidates if candidate["id"] in selected_ids]
    if args.max_candidates is not None:
        candidates = candidates[: args.max_candidates]
    if not candidates:
        raise ValueError("no candidates selected")

    if args.dry_run:
        for candidate in candidates:
            print(f"{candidate['id']}: {candidate.get('family', '')} - {candidate.get('description', '')}")
        return 0

    started_at = utc_slug()
    output_dir = (args.output_root / started_at).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "runs").mkdir()
    (output_dir / "charts").mkdir()
    hardware = collect_hardware_info()
    (output_dir / "hardware.json").write_text(json.dumps(hardware, indent=2) + "\n", encoding="utf-8")
    (output_dir / "candidate_manifest.json").write_text(
        json.dumps(candidates, indent=2) + "\n",
        encoding="utf-8",
    )

    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        candidate_id = safe_id(candidate["id"])
        run_id = f"h100_batch_{started_at}_{index:02d}_{candidate_id}"
        run_dir = output_dir / "runs" / f"{index:02d}_{candidate_id}"
        run_dir.mkdir(parents=True)
        env = build_env(candidate, run_id)
        if args.smoke:
            apply_smoke_overrides(env)
        batch_tune: dict[str, Any] = {"enabled": False}
        if args.auto_tune_batch and not args.smoke:
            if not candidate_uses_default_train_command(candidate):
                batch_tune = {
                    "enabled": True,
                    "status": "skipped",
                    "reason": "custom-command-does-not-support-standard-BATCH_TUNE_ONLY-probe",
                    "selectedTrainBatchTokens": int_env(env, "TRAIN_BATCH_TOKENS", 524_288),
                }
            else:
                print(f"[{index}/{len(candidates)}] tuning batch for {candidate_id}")
                batch_tune = auto_tune_batch_size(
                    candidate=candidate,
                    run_dir=run_dir,
                    env=env,
                    hardware=hardware,
                    target_memory_fraction=args.batch_tune_target_memory_fraction,
                    max_tokens=args.batch_tune_max_tokens,
                    timeout_seconds=args.batch_tune_timeout_seconds,
                    nproc_per_node=args.nproc_per_node,
                )
        command = command_for_candidate(candidate, smoke=args.smoke, nproc_per_node=args.nproc_per_node)
        if args.tune_only:
            completed = {
                "returnCode": 0,
                "timedOut": False,
                "terminationReason": "tune-only",
                "elapsedSeconds": 0.0,
                "stdout": "",
                "stderr": "",
            }
            metrics = parse_metrics("")
            artifact_state = prune_raw_checkpoint(run_dir, args.keep_raw_checkpoints)
            status = "tuned" if batch_tune.get("status") == "completed" else str(batch_tune.get("status") or "tune_skipped")
        else:
            print(f"[{index}/{len(candidates)}] running {candidate_id}")
            completed = stream_subprocess(
                command=command,
                cwd=run_dir,
                env=env,
                stdout_path=run_dir / "stdout.txt",
                stderr_path=run_dir / "stderr.txt",
                timeout_seconds=args.timeout_seconds,
            )
            metrics = parse_metrics(completed["stdout"])
            artifact_state = prune_raw_checkpoint(run_dir, args.keep_raw_checkpoints)
            status = "completed" if completed["returnCode"] == 0 and metrics["finalValBpb"] is not None else "failed"
            if completed["timedOut"]:
                status = "timed_out"
        selected_train_batch_tokens = int_env(env, "TRAIN_BATCH_TOKENS", 524_288)
        derived_tokens_per_second = tokens_per_second(selected_train_batch_tokens, metrics.get("stepAvgMs"))
        row = {
            "id": candidate["id"],
            "family": candidate.get("family", ""),
            "hardwareLabel": hardware_label(hardware),
            "quantization": candidate.get("quantization", ""),
            "hypothesis": candidate.get("hypothesis", candidate.get("id")),
            "hypothesisTags": tag_list(candidate),
            "description": candidate.get("description", ""),
            "hardware": hardware,
            "status": status,
            "returnCode": completed["returnCode"],
            "timedOut": completed["timedOut"],
            "terminationReason": completed["terminationReason"],
            "elapsedSeconds": completed["elapsedSeconds"],
            "runDir": str(run_dir),
            "command": command,
            "nprocPerNode": args.nproc_per_node,
            "env": {
                key: env[key]
                for key in sorted(
                    set(BASE_ENV)
                    | set(candidate.get("env", {}))
                    | {
                        "RUN_ID",
                        "ITERATIONS",
                        "TRAIN_BATCH_TOKENS",
                        "TRAIN_SEQ_LEN",
                        "VAL_BATCH_SIZE",
                        "VAL_TOKEN_LIMIT",
                        "SEED",
                    }
                )
                if key in env
            },
            "selectedTrainBatchTokens": selected_train_batch_tokens,
            "tokensPerSecond": derived_tokens_per_second,
            "batchTuneStatus": batch_tune.get("status") if batch_tune.get("enabled") else None,
            "batchTune": batch_tune,
            **artifact_state,
            **metrics,
        }
        (run_dir / "result.json").write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
        rows.append(row)
        ranked = rank_rows(rows)
        write_csv(output_dir / "leaderboard.csv", ranked)
        (output_dir / "results.json").write_text(json.dumps(ranked, indent=2) + "\n", encoding="utf-8")
        svg_bar_chart(output_dir / "charts/final_val_bpb.svg", ranked)
        svg_scatter_chart(output_dir / "charts/speed_vs_bpb.svg", ranked)
        write_analysis_outputs(output_dir, ranked, hardware, started_at)
        write_summary(output_dir / "summary.md", ranked, started_at, output_dir)
        if status == "completed":
            print(f"[{index}/{len(candidates)}] {candidate_id} val_bpb={metrics['finalValBpb']:.8f}")
        elif args.tune_only:
            print(f"[{index}/{len(candidates)}] {candidate_id} tune_status={status} train_batch_tokens={selected_train_batch_tokens}")
        else:
            print(f"[{index}/{len(candidates)}] {candidate_id} status={status} return_code={completed['returnCode']}")
        if args.stop_on_failure and status != "completed":
            break

    ranked = rank_rows(rows)
    write_csv(output_dir / "leaderboard.csv", ranked)
    (output_dir / "results.json").write_text(json.dumps(ranked, indent=2) + "\n", encoding="utf-8")
    svg_bar_chart(output_dir / "charts/final_val_bpb.svg", ranked)
    svg_scatter_chart(output_dir / "charts/speed_vs_bpb.svg", ranked)
    write_analysis_outputs(output_dir, ranked, hardware, started_at)
    write_summary(output_dir / "summary.md", ranked, started_at, output_dir)
    print(f"wrote {output_dir}")
    return 1 if any(row.get("status") not in {"completed", "tuned"} for row in rows) else 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sequential Parameter Golf candidate experiments on H100 pods.")
    parser.add_argument("--candidate-file", type=Path, default=None, help="JSON list of candidate definitions.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Directory for batch outputs.")
    parser.add_argument("--timeout-seconds", type=int, default=900, help="Host timeout per candidate, including final eval.")
    parser.add_argument("--nproc-per-node", type=int, default=1, help="torchrun processes per node for default candidate commands.")
    parser.add_argument("--only", action="append", default=[], help="Candidate id to run; can be repeated.")
    parser.add_argument("--max-candidates", type=int, default=None, help="Run only the first N selected candidates.")
    parser.add_argument("--stop-on-failure", action="store_true", help="Stop the batch after the first failed candidate.")
    parser.add_argument("--dry-run", action="store_true", help="Print selected candidates without running them.")
    parser.add_argument("--smoke", action="store_true", help="Use tiny overrides for local wiring tests.")
    parser.add_argument("--keep-raw-checkpoints", action="store_true", help="Keep final_model.pt files in each run directory.")
    parser.add_argument("--auto-tune-batch", action="store_true", help="Run batch-size tuning probes before each scored H100 run.")
    parser.add_argument("--tune-only", action="store_true", help="Run setup/tuning and skip the scored training run.")
    parser.add_argument(
        "--batch-tune-target-memory-fraction",
        type=float,
        default=0.90,
        help="Target fraction of first GPU memory to reserve during automatic batch tuning.",
    )
    parser.add_argument(
        "--batch-tune-max-tokens",
        type=int,
        default=2_097_152,
        help="Maximum TRAIN_BATCH_TOKENS value considered during automatic batch tuning.",
    )
    parser.add_argument(
        "--batch-tune-timeout-seconds",
        type=int,
        default=180,
        help="Host timeout per batch tuning probe.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    return run_batch(args)


if __name__ == "__main__":
    raise SystemExit(main())
