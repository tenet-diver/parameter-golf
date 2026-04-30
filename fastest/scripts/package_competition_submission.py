from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BATCH_ROOT = REPO_ROOT / "records/h100_candidate_batch"
DEFAULT_RECORDS_ROOT = REPO_ROOT / "records/track_10min_16mb"


def safe_slug(raw: str) -> str:
    cleaned = []
    previous_dash = False
    for char in raw.strip().lower():
        if char.isalnum():
            cleaned.append(char)
            previous_dash = False
        elif not previous_dash:
            cleaned.append("-")
            previous_dash = True
    return "".join(cleaned).strip("-") or "submission"


def numeric(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        return float(value)
    return None


def latest_batch_dir(batch_root: Path = DEFAULT_BATCH_ROOT) -> Path:
    candidates = [
        path
        for path in batch_root.iterdir()
        if path.is_dir() and (path / "results.json").exists()
    ] if batch_root.exists() else []
    if not candidates:
        raise FileNotFoundError(f"no batch results found under {batch_root}")
    return max(candidates, key=lambda path: (path / "results.json").stat().st_mtime)


def normalize_batch_dirs(paths: list[Path]) -> list[Path]:
    if not paths:
        return [latest_batch_dir()]
    normalized = []
    for path in paths:
        if str(path) == "latest":
            normalized.append(latest_batch_dir())
        else:
            normalized.append(path)
    return normalized


def load_rows(batch_dirs: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for batch_dir in batch_dirs:
        results_path = batch_dir / "results.json"
        if not results_path.exists():
            raise FileNotFoundError(f"missing batch results: {results_path}")
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"{results_path} must contain a JSON list")
        for row in payload:
            if not isinstance(row, dict):
                raise ValueError(f"{results_path} contains a non-object row")
            annotated = dict(row)
            annotated["_batchDir"] = str(batch_dir)
            rows.append(annotated)
    return rows


def completed_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("status") == "completed" and numeric(row.get("finalValBpb")) is not None
    ]


def sort_seed_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (seed_label(row), str(row.get("runDir") or "")))


def select_rows(rows: list[dict[str, Any]], candidate_id: str | None) -> list[dict[str, Any]]:
    scored = completed_rows(rows)
    if not scored:
        raise ValueError("no completed scored runs found")
    if candidate_id:
        selected = [row for row in scored if str(row.get("id")) == candidate_id]
        if not selected:
            raise ValueError(f"no completed scored runs found for candidate {candidate_id!r}")
        return sort_seed_rows(selected)

    by_candidate: dict[str, list[dict[str, Any]]] = {}
    for row in scored:
        by_candidate.setdefault(str(row.get("id") or ""), []).append(row)
    best_candidate_id, best_rows = min(
        by_candidate.items(),
        key=lambda item: (
            statistics.mean(float(row["finalValBpb"]) for row in item[1]),
            min(float(row["finalValBpb"]) for row in item[1]),
            item[0],
        ),
    )
    if not best_candidate_id:
        raise ValueError("completed scored runs are missing candidate ids")
    return sort_seed_rows(best_rows)


def seed_label(row: dict[str, Any]) -> str:
    env = row.get("env")
    if isinstance(env, dict) and env.get("SEED") is not None:
        return str(env["SEED"])
    return "1337"


def row_artifact_bytes(row: dict[str, Any]) -> int | None:
    for key in ("int8SubmissionBytes", "artifactBudgetBytes", "quantizedArtifactBytes"):
        value = row.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
    return None


def row_train_seconds(row: dict[str, Any]) -> float | None:
    elapsed = numeric(row.get("elapsedSeconds"))
    if elapsed is not None:
        return elapsed
    train_ms = numeric(row.get("trainTimeMs"))
    return train_ms / 1000.0 if train_ms is not None else None


def hardware_label(row: dict[str, Any]) -> str:
    hardware = row.get("hardware")
    if isinstance(hardware, dict):
        gpus = hardware.get("gpus")
        if isinstance(gpus, list) and gpus:
            first = gpus[0] if isinstance(gpus[0], dict) else {}
            name = str(first.get("name") or "GPU")
            return f"{len(gpus)}x{name}"
    return str(row.get("hardwareLabel") or "unknown")


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def utc_submission_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def title_from_row(row: dict[str, Any]) -> str:
    description = str(row.get("description") or "").strip()
    if description:
        return description.rstrip(".")
    hypothesis = str(row.get("hypothesis") or "").replace("_", " ").strip()
    if hypothesis:
        return hypothesis
    return str(row.get("id") or "Parameter Golf submission")


def row_model_bytes(row: dict[str, Any]) -> int | None:
    for key in ("quantizedArtifactBytes", "modelBytes", "bytes_model_int8_zlib"):
        value = row.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
    return None


def build_submission_payload(
    rows: list[dict[str, Any]],
    *,
    author: str,
    github_id: str,
    name: str,
    track: str,
    date: str,
    code_bytes: int,
) -> dict[str, Any]:
    bpbs = [float(row["finalValBpb"]) for row in rows]
    losses = [float(row["finalValLoss"]) for row in rows if numeric(row.get("finalValLoss")) is not None]
    effective_artifact_bytes = [
        (row_model_bytes(row) + code_bytes)
        if row_model_bytes(row) is not None
        else row_artifact_bytes(row)
        for row in rows
    ]
    seed_results: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows, start=1):
        seed = seed_label(row)
        key = seed if seed not in seed_results else f"{seed}_run{index}"
        seed_results[key] = {
            "val_bpb": row.get("finalValBpb"),
            "val_loss": row.get("finalValLoss"),
            "artifact_bytes": effective_artifact_bytes[index - 1],
            "model_bytes": row_model_bytes(row),
            "code_bytes": code_bytes,
            "train_time_s": row_train_seconds(row),
            "run_dir": row.get("runDir"),
        }

    blurb = title_from_row(rows[0])
    payload: dict[str, Any] = {
        "author": author,
        "github_id": github_id,
        "name": name,
        "blurb": blurb,
        "date": date,
        "track": track,
        "candidate_id": rows[0].get("id"),
        "val_bpb": statistics.mean(bpbs),
        "seeds": list(seed_results),
        "seed_results": seed_results,
        "hardware": hardware_label(rows[0]),
        "technique_summary": title_from_row(rows[0]),
        "bytes_code": code_bytes,
        "compliance": {
            "train_under_600s": all((row_train_seconds(row) or float("inf")) <= 600 for row in rows),
            "artifact_under_16mb": all((value or 16_000_001) <= 16_000_000 for value in effective_artifact_bytes),
            "three_seeds": len(rows) >= 3,
            "generated_from_h100_batch": True,
        },
        "provenance": {
            "candidate_id": rows[0].get("id"),
            "batch_dirs": sorted({str(row.get("_batchDir")) for row in rows}),
            "source_git_commit": git_commit(),
            "generated_by": "fastest/scripts/package_competition_submission.py",
        },
    }
    if losses:
        payload["val_loss"] = statistics.mean(losses)
    if len(bpbs) > 1:
        payload["val_bpb_std"] = statistics.stdev(bpbs)
    max_model_bytes = max((row_model_bytes(row) or 0) for row in rows)
    if max_model_bytes:
        payload["bytes_model_int8_zlib"] = max_model_bytes
    max_bytes = max((value or 0) for value in effective_artifact_bytes)
    if max_bytes:
        payload["bytes_total"] = max_bytes
    max_train = max((row_train_seconds(row) or 0.0) for row in rows)
    if max_train:
        payload["train_time_seconds"] = max_train
    return payload


def shell_quote_env(env: dict[str, Any]) -> list[str]:
    excluded = {"RUN_ID", "DATA_PATH", "TOKENIZER_PATH", "SUBMISSION_CODE_PATHS"}
    lines = []
    for key in sorted(env):
        if key in excluded:
            continue
        value = str(env[key])
        lines.append(f"export {key}={json.dumps(value)}")
    return lines


def write_run_script(path: Path, row: dict[str, Any], nproc_per_node: int) -> None:
    env = dict(row.get("env") if isinstance(row.get("env"), dict) else {})
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"',
        'export DATA_PATH="${DATA_PATH:-$REPO_ROOT/data/datasets/fineweb10B_sp1024}"',
        'export TOKENIZER_PATH="${TOKENIZER_PATH:-$REPO_ROOT/data/tokenizers/fineweb_1024_bpe.model}"',
        f'export NPROC_PER_NODE="${{NPROC_PER_NODE:-{nproc_per_node}}}"',
        'export RUN_ID="${RUN_ID:-submission_rerun}"',
        *shell_quote_env(env),
        'torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" "$SCRIPT_DIR/train_gpt.py"',
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(path.stat().st_mode | 0o111)


def reproduction_command(row: dict[str, Any], nproc_per_node: int) -> list[str]:
    env = row.get("env") if isinstance(row.get("env"), dict) else {}
    exports = shell_quote_env(env)
    return [
        *exports,
        f'export NPROC_PER_NODE="${{NPROC_PER_NODE:-{nproc_per_node}}}"',
        'export DATA_PATH="${DATA_PATH:-../../data/datasets/fineweb10B_sp1024}"',
        'export TOKENIZER_PATH="${TOKENIZER_PATH:-../../data/tokenizers/fineweb_1024_bpe.model}"',
        'export RUN_ID="${RUN_ID:-submission_rerun}"',
        'torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" train_gpt.py',
    ]


def write_readme(path: Path, rows: list[dict[str, Any]], payload: dict[str, Any]) -> None:
    row = rows[0]
    warning = ""
    if len(rows) < 3:
        warning = (
            "\n> Note: this generated package contains fewer than 3 seeds. "
            "For a new SOTA record PR, rerun the selected candidate with additional seeds "
            "or submit as a non-record result.\n"
        )
    lines = [
        f"# Submission: {payload['name']}",
        "",
        f"Candidate: `{row.get('id')}`",
        f"Mean val_bpb: `{payload['val_bpb']:.8f}`",
        f"Artifact bytes: `{payload.get('bytes_total', '')}`",
        f"Hardware: `{payload.get('hardware', 'unknown')}`",
        warning,
        "## Results",
        "",
        "| Seed | val_bpb | val_loss | artifact bytes | train seconds |",
        "|---|---:|---:|---:|---:|",
    ]
    for seed, result in payload["seed_results"].items():
        lines.append(
            "| {seed} | {bpb} | {loss} | {artifact} | {seconds} |".format(
                seed=seed,
                bpb=format_number(result.get("val_bpb"), 8),
                loss=format_number(result.get("val_loss"), 8),
                artifact=result.get("artifact_bytes") or "",
                seconds=format_number(result.get("train_time_s"), 3),
            )
        )
    lines.extend(
        [
            "",
            "## Technique",
            "",
            str(payload.get("technique_summary") or row.get("description") or row.get("hypothesis") or row.get("id")),
            "",
            "Candidate environment is captured in `candidate_env.json`; source batch rows are copied into `run_result*.json`.",
            "",
            "## Reproduction",
            "",
            "From this record folder:",
            "",
            "```bash",
            *reproduction_command(row, int(row.get("nprocPerNode") or 8)),
            "```",
            "",
            "For a 3-seed verification run:",
            "",
            "```bash",
            "for SEED in 42 314 1234; do",
            "  SEED=$SEED RUN_ID=submission_seed${SEED} \\",
            "    torchrun --standalone --nproc_per_node=8 train_gpt.py > train_seed${SEED}_rerun.log 2>&1",
            "done",
            "```",
            "",
            "## Compliance Notes",
            "",
            f"- Training under 600s in packaged run(s): `{payload['compliance']['train_under_600s']}`",
            f"- Artifact under 16,000,000 bytes in packaged run(s): `{payload['compliance']['artifact_under_16mb']}`",
            f"- Three-seed evidence included: `{payload['compliance']['three_seeds']}`",
            "",
            "## Included Files",
            "",
            "- `README.md`",
            "- `submission.json`",
            "- `train_gpt.py`",
            "- `train.log`",
            "- `candidate_env.json`",
            "- `train*.log`",
            "- `run_result*.json`",
            "",
        ]
    )
    path.write_text("\n".join(line for line in lines if line is not None), encoding="utf-8")


def format_number(value: Any, digits: int) -> str:
    number = numeric(value)
    return "" if number is None else f"{number:.{digits}f}"


def _module_source(relative_path: str) -> str:
    source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
    lines = [
        line
        for line in source.splitlines()
        if line.strip() != "from __future__ import annotations"
    ]
    return "\n".join(lines) + "\n"


def bundled_module_prelude() -> str:
    modules = [
        ("fastest.scripts.artifact_budget_analyzer", "fastest/scripts/artifact_budget_analyzer.py"),
        ("candidates.registry", "candidates/registry.py"),
        ("candidates.autoregressive.architecture", "candidates/autoregressive/architecture.py"),
        ("candidates.autoregressive.model", "candidates/autoregressive/model.py"),
        ("candidates.moe.model", "candidates/moe/model.py"),
        ("candidates.autoregressive.candidates", "candidates/autoregressive/candidates.py"),
        ("candidates.moe.candidates", "candidates/moe/candidates.py"),
    ]
    lines = [
        "# --- Bundled local modules for Parameter Golf submission compliance ---",
        "import sys as _pg_sys",
        "import types as _pg_types",
        "",
        "def _pg_install_package(name):",
        "    module = _pg_sys.modules.get(name)",
        "    if module is None:",
        "        module = _pg_types.ModuleType(name)",
        "        module.__path__ = []",
        "        module.__package__ = name",
        "        _pg_sys.modules[name] = module",
        "    return module",
        "",
        "def _pg_install_module(name, source):",
        "    package_name, _, short_name = name.rpartition('.')",
        "    if package_name:",
        "        parts = package_name.split('.')",
        "        for index in range(1, len(parts) + 1):",
        "            _pg_install_package('.'.join(parts[:index]))",
        "    module = _pg_types.ModuleType(name)",
        "    module.__file__ = '<bundled ' + name + '>'",
        "    module.__package__ = package_name",
        "    _pg_sys.modules[name] = module",
        "    if package_name:",
        "        setattr(_pg_sys.modules[package_name], short_name, module)",
        "    exec(compile(source, module.__file__, 'exec'), module.__dict__)",
        "",
    ]
    for name, relative_path in modules:
        lines.append(f"_pg_install_module({name!r}, {_module_source(relative_path)!r})")
    lines.extend(
        [
            "del _pg_install_module, _pg_install_package, _pg_sys, _pg_types",
            "# --- End bundled local modules ---",
            "",
        ]
    )
    return "\n".join(lines)


def write_self_contained_train_script(path: Path) -> None:
    train_source = _module_source("train_gpt.py")
    content = (
        '"""Self-contained Parameter Golf training script generated from the experiment branch."""\n'
        "from __future__ import annotations\n\n"
        f"{bundled_module_prelude()}"
        f"{train_source}"
    )
    path.write_text(content, encoding="utf-8")


def copy_run_artifacts(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    for index, row in enumerate(rows, start=1):
        seed = seed_label(row)
        suffix = f"seed{seed}" if len(rows) == 1 else f"seed{seed}_run{index}"
        run_dir = Path(str(row.get("runDir") or ""))
        if (run_dir / "stdout.txt").exists():
            if index == 1:
                shutil.copy2(run_dir / "stdout.txt", output_dir / "train.log")
            shutil.copy2(run_dir / "stdout.txt", output_dir / f"train_{suffix}.log")
        if (run_dir / "stderr.txt").exists():
            shutil.copy2(run_dir / "stderr.txt", output_dir / f"stderr_{suffix}.txt")
        if (run_dir / "result.json").exists():
            shutil.copy2(run_dir / "result.json", output_dir / f"run_result_{suffix}.json")
    (output_dir / "candidate_env.json").write_text(
        json.dumps(rows[0].get("env", {}), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "packaged_batch_rows.json").write_text(
        json.dumps(rows, indent=2) + "\n",
        encoding="utf-8",
    )


def package_submission(args: argparse.Namespace) -> Path:
    batch_dirs = normalize_batch_dirs(args.batch_dir)
    rows = select_rows(load_rows(batch_dirs), args.candidate_id)
    name = args.name or title_from_row(rows[0])
    date = args.date or utc_submission_timestamp()
    author = args.author or "TODO"
    github_id = args.github_id or "TODO"
    preliminary_score = f"{float(rows[0]['finalValBpb']):.5f}".replace(".", "p")
    slug = args.slug or f"{date[:10]}_{safe_slug(str(rows[0].get('id') or name))}_{preliminary_score}"
    output_dir = args.output_dir or (args.records_root / slug)
    if output_dir.exists():
        if not args.force:
            raise FileExistsError(f"{output_dir} already exists; pass --force to replace it")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    write_self_contained_train_script(output_dir / "train_gpt.py")
    code_bytes = (output_dir / "train_gpt.py").stat().st_size
    payload = build_submission_payload(
        rows,
        author=author,
        github_id=github_id,
        name=name,
        track=args.track,
        date=date,
        code_bytes=code_bytes,
    )
    copy_run_artifacts(output_dir, rows)
    nproc = int(args.nproc_per_node or rows[0].get("nprocPerNode") or 8)
    (output_dir / "submission.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_readme(output_dir / "README.md", rows, payload)
    return output_dir


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package H100 batch results as a PR-ready Parameter Golf submission folder.")
    parser.add_argument("--batch-dir", action="append", type=Path, default=[], help="Batch directory with results.json; can be repeated. Defaults to latest.")
    parser.add_argument("--candidate-id", default=None, help="Candidate id to package. Defaults to best completed run.")
    parser.add_argument("--records-root", type=Path, default=DEFAULT_RECORDS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--slug", default=None)
    parser.add_argument("--author", default=None)
    parser.add_argument("--github-id", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--date", default=None)
    parser.add_argument("--track", default="10min_16mb")
    parser.add_argument("--nproc-per-node", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    output_dir = package_submission(parse_args(sys.argv[1:] if argv is None else argv))
    print(f"submission_dir={output_dir}")
    print(f"submission_json={output_dir / 'submission.json'}")
    print(f"readme={output_dir / 'README.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
