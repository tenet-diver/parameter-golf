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


def select_rows(rows: list[dict[str, Any]], candidate_id: str | None) -> list[dict[str, Any]]:
    scored = completed_rows(rows)
    if not scored:
        raise ValueError("no completed scored runs found")
    if candidate_id:
        selected = [row for row in scored if str(row.get("id")) == candidate_id]
        if not selected:
            raise ValueError(f"no completed scored runs found for candidate {candidate_id!r}")
        return sorted(selected, key=lambda row: (seed_label(row), str(row.get("runDir") or "")))
    return [min(scored, key=lambda row: (float(row["finalValBpb"]), str(row.get("id") or "")))]


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


def title_from_row(row: dict[str, Any]) -> str:
    description = str(row.get("description") or "").strip()
    if description:
        return description.rstrip(".")
    hypothesis = str(row.get("hypothesis") or "").replace("_", " ").strip()
    if hypothesis:
        return hypothesis
    return str(row.get("id") or "Parameter Golf submission")


def build_submission_payload(
    rows: list[dict[str, Any]],
    *,
    author: str,
    github_id: str,
    name: str,
    track: str,
    date: str,
) -> dict[str, Any]:
    bpbs = [float(row["finalValBpb"]) for row in rows]
    losses = [float(row["finalValLoss"]) for row in rows if numeric(row.get("finalValLoss")) is not None]
    seed_results: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows, start=1):
        seed = seed_label(row)
        key = seed if seed not in seed_results else f"{seed}_run{index}"
        seed_results[key] = {
            "val_bpb": row.get("finalValBpb"),
            "val_loss": row.get("finalValLoss"),
            "artifact_bytes": row_artifact_bytes(row),
            "train_time_s": row_train_seconds(row),
            "run_dir": row.get("runDir"),
        }

    payload: dict[str, Any] = {
        "author": author,
        "github_id": github_id,
        "name": name,
        "date": date,
        "track": track,
        "candidate_id": rows[0].get("id"),
        "val_bpb": statistics.mean(bpbs),
        "seeds": list(seed_results),
        "seed_results": seed_results,
        "hardware": hardware_label(rows[0]),
        "technique_summary": title_from_row(rows[0]),
        "compliance": {
            "train_under_600s": all((row_train_seconds(row) or float("inf")) <= 600 for row in rows),
            "artifact_under_16mb": all((row_artifact_bytes(row) or 16_000_001) <= 16_000_000 for row in rows),
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
    max_bytes = max((row_artifact_bytes(row) or 0) for row in rows)
    if max_bytes:
        payload["bytes_total"] = max_bytes
    max_train = max((row_train_seconds(row) or 0.0) for row in rows)
    if max_train:
        payload["train_time_seconds"] = max_train
    return payload


def shell_quote_env(env: dict[str, Any]) -> list[str]:
    excluded = {"RUN_ID", "DATA_PATH", "TOKENIZER_PATH"}
    lines = []
    for key in sorted(env):
        if key in excluded:
            continue
        value = str(env[key])
        lines.append(f"export {key}={json.dumps(value)}")
    return lines


def write_run_script(path: Path, row: dict[str, Any], nproc_per_node: int) -> None:
    env = row.get("env") if isinstance(row.get("env"), dict) else {}
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
            "Candidate environment is captured in `candidate_env.json`; the source batch row is copied into `run_result*.json`.",
            "",
            "## Reproduction",
            "",
            "From this record folder:",
            "",
            "```bash",
            "./run_submission.sh > train_rerun.log 2>&1",
            "```",
            "",
            "For a 3-seed verification run:",
            "",
            "```bash",
            "for SEED in 42 314 1234; do",
            "  SEED=$SEED RUN_ID=submission_seed${SEED} ./run_submission.sh > train_seed${SEED}_rerun.log 2>&1",
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
            "- `run_submission.sh`",
            "- `candidate_env.json`",
            "- `train*.log`",
            "- `run_result*.json`",
            "- `candidates/` and `fastest/scripts/artifact_budget_analyzer.py` local dependencies",
            "",
        ]
    )
    path.write_text("\n".join(line for line in lines if line is not None), encoding="utf-8")


def format_number(value: Any, digits: int) -> str:
    number = numeric(value)
    return "" if number is None else f"{number:.{digits}f}"


def copy_sources(output_dir: Path) -> None:
    shutil.copy2(REPO_ROOT / "train_gpt.py", output_dir / "train_gpt.py")
    if (REPO_ROOT / "requirements.txt").exists():
        shutil.copy2(REPO_ROOT / "requirements.txt", output_dir / "requirements.txt")
    shutil.copytree(
        REPO_ROOT / "candidates",
        output_dir / "candidates",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    helper_dir = output_dir / "fastest" / "scripts"
    helper_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "fastest" / "__init__.py").write_text("", encoding="utf-8")
    (helper_dir / "__init__.py").write_text("", encoding="utf-8")
    shutil.copy2(
        REPO_ROOT / "fastest" / "scripts" / "artifact_budget_analyzer.py",
        helper_dir / "artifact_budget_analyzer.py",
    )


def copy_run_artifacts(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    for index, row in enumerate(rows, start=1):
        seed = seed_label(row)
        suffix = f"seed{seed}" if len(rows) == 1 else f"seed{seed}_run{index}"
        run_dir = Path(str(row.get("runDir") or ""))
        if (run_dir / "stdout.txt").exists():
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
    date = args.date or datetime.now(timezone.utc).date().isoformat()
    author = args.author or "TODO"
    github_id = args.github_id or "TODO"
    payload = build_submission_payload(
        rows,
        author=author,
        github_id=github_id,
        name=name,
        track=args.track,
        date=date,
    )
    score = f"{payload['val_bpb']:.5f}".replace(".", "p")
    slug = args.slug or f"{date}_{safe_slug(str(rows[0].get('id') or name))}_{score}"
    output_dir = args.output_dir or (args.records_root / slug)
    if output_dir.exists():
        if not args.force:
            raise FileExistsError(f"{output_dir} already exists; pass --force to replace it")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    copy_sources(output_dir)
    copy_run_artifacts(output_dir, rows)
    nproc = int(args.nproc_per_node or rows[0].get("nprocPerNode") or 8)
    write_run_script(output_dir / "run_submission.sh", rows[0], nproc)
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
